import numpy as np
import polars as pl
import torch

from ._runtime import training_threads
from .base import BaseRecommender


def sample_training_pairs(fit_df, warm_mask, neg_per_pos=3, seed=42):
    if neg_per_pos < 0:
        raise ValueError("neg_per_pos must be nonnegative")
    rng = np.random.default_rng(seed)
    positives = fit_df.filter(pl.col("rating") >= 4)
    users = [positives["user_idx"].to_numpy().astype(np.int64)]
    items = [positives["item_idx"].to_numpy().astype(np.int64)]
    warm = np.flatnonzero(warm_mask)
    if not len(fit_df):
        return users[0], items[0], np.empty(0, dtype=np.float32)
    user_ids = fit_df["user_idx"].to_numpy()
    item_ids = fit_df["item_idx"].to_numpy()
    positive_counts = np.bincount(
        positives["user_idx"].to_numpy(), minlength=int(user_ids.max()) + 1
    )
    order = np.argsort(user_ids, kind="stable")
    sorted_users, sorted_items = user_ids[order], item_ids[order]
    boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_users)) + 1, len(order)]
    warm_positions = np.full(len(warm_mask), -1, dtype=np.int64)
    warm_positions[warm] = np.arange(len(warm))
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        user = int(sorted_users[start])
        size = int(positive_counts[user]) * neg_per_pos
        if not size:
            continue
        seen = np.unique(warm_positions[sorted_items[start:end]])
        seen = seen[seen >= 0]
        n_candidates = len(warm) - len(seen)
        if not n_candidates:
            continue
        # Draw ranks in the complement, without scanning the whole catalogue per user.
        ranks = rng.choice(n_candidates, size=size)
        positions = ranks + np.searchsorted(
            seen - np.arange(len(seen)), ranks, side="right"
        )
        users.append(np.full(size, user, dtype=np.int64))
        items.append(warm[positions])
    users, items = np.concatenate(users), np.concatenate(items)
    labels = np.r_[
        np.ones(len(positives)), np.zeros(len(users) - len(positives))
    ].astype(np.float32)
    return users, items, labels


class FactorizationMachine(torch.nn.Module):
    def __init__(self, n_values, d=8, seed=42):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.w = torch.nn.Embedding(n_values, 1)
            self.v = torch.nn.Embedding(n_values, d)
            torch.nn.init.normal_(self.w.weight, std=0.01)
            torch.nn.init.normal_(self.v.weight, std=0.01)
        self.w0 = torch.nn.Parameter(torch.zeros(1))

    def forward(self, index, x=None):
        if x is None:
            x = torch.ones_like(index, dtype=self.v.weight.dtype)
        vx = self.v(index) * x.unsqueeze(-1)
        interaction = 0.5 * (vx.sum(1).square() - vx.square().sum(1)).sum(1)
        return self.w0 + (self.w(index).squeeze(-1) * x).sum(1) + interaction


def _standardize(values, fit_mask):
    reference = values[fit_mask]
    mean = float(reference.mean()) if len(reference) else 0.0
    std = float(reference.std()) if len(reference) else 1.0
    return ((values - mean) / max(std, 1e-8)).astype(np.float32)


class FM(BaseRecommender):
    def __init__(
        self,
        d=8,
        l2=0.0,
        lr=0.02,
        epochs=5,
        batch_size=1024,
        neg_per_pos=3,
        seed=42,
        use_meta=True,
    ):
        self.d, self.l2, self.lr = d, l2, lr
        self.epochs, self.batch_size = epochs, batch_size
        self.neg_per_pos, self.seed = neg_per_pos, seed
        self.use_meta, self.item_meta = use_meta, None

    def _build(self, fit_df, n_users, n_items):
        self.n_users, self.n_items = n_users, n_items
        self.item_fields, self.user_fields = [], []
        self.item_values, self.user_values = [], []
        self.sizes = [n_users, n_items]
        if not self.use_meta:
            self.offsets = np.r_[0, np.cumsum(self.sizes)]
            return self.item_fields, self.user_fields, self.sizes
        if self.item_meta is None:
            raise ValueError("use_meta=True requires metadata in item_idx order")
        category = (
            self.item_meta["root_category"]
            .fill_null("unknown")
            .cast(pl.String)
            .to_numpy()
        )
        if category.shape != (n_items,):
            raise ValueError(
                "metadata must contain one row per protocol item, in item_idx order"
            )
        users = fit_df["user_idx"].to_numpy().astype(int)
        items = fit_df["item_idx"].to_numpy().astype(int)
        ratings = fit_df["rating"].to_numpy().astype(float)
        counts = np.bincount(items, minlength=n_items).astype(float)
        warm = counts > 0
        active = np.bincount(users, minlength=n_users).astype(float)
        means = np.divide(
            np.bincount(items, weights=ratings, minlength=n_items),
            counts,
            out=np.zeros(n_items),
            where=warm,
        )
        self.category_to_code = {
            value: i + 1 for i, value in enumerate(sorted(set(category[warm])))
        }
        categories = np.array(
            [self.category_to_code.get(value, 0) for value in category]
        )
        n_categories = len(self.category_to_code) + 1
        dominant = np.zeros(n_users, dtype=np.int64)
        affinity = np.zeros(n_users)
        positive = ratings >= 4
        if positive.any():
            pairs, pair_counts = np.unique(
                np.c_[users[positive], categories[items[positive]]],
                axis=0,
                return_counts=True,
            )
            order = np.lexsort((pairs[:, 1], -pair_counts, pairs[:, 0]))
            pairs, pair_counts = pairs[order], pair_counts[order]
            first = np.r_[True, pairs[1:, 0] != pairs[:-1, 0]]
            best_users = pairs[first, 0]
            dominant[best_users] = pairs[first, 1]
            totals = np.bincount(users[positive], minlength=n_users)
            affinity[best_users] = pair_counts[first] / totals[best_users]
        self.dominant_category_, self.affinity_ = dominant, affinity
        self.item_fields = [categories, np.zeros(n_items, int), np.zeros(n_items, int)]
        self.item_values = [
            np.ones(n_items),
            _standardize(np.log1p(counts), warm),
            _standardize(means, warm),
        ]
        self.user_fields = [np.zeros(n_users, int), dominant, np.zeros(n_users, int)]
        self.user_values = [
            _standardize(np.log1p(active), active > 0),
            np.ones(n_users),
            _standardize(affinity, active > 0),
        ]
        self.sizes += [n_categories, 1, 1, 1, n_categories, 1]
        self.offsets = np.r_[0, np.cumsum(self.sizes)].astype(np.int64)
        return self.item_fields, self.user_fields, self.sizes

    def _encode(self, users, items):
        users, items = np.asarray(users, dtype=int), np.asarray(items, dtype=int)
        codes = (
            [users, items]
            + [field[items] for field in self.item_fields]
            + [field[users] for field in self.user_fields]
        )
        values = (
            [np.ones(len(users)), np.ones(len(users))]
            + [field[items] for field in self.item_values]
            + [field[users] for field in self.user_values]
        )
        return (
            np.column_stack(codes).astype(np.int64) + self.offsets[:-1],
            np.column_stack(values).astype(np.float32),
        )

    def fit(self, interactions, n_users=None, n_items=None, metadata=None):
        fit_df = interactions
        if fit_df.is_empty() or not (fit_df["rating"] >= 4).any():
            raise ValueError("FM needs at least one positive training interaction")
        super().fit(interactions, n_users, n_items, metadata)
        self.item_meta = metadata
        self._build(fit_df, self.n_users, self.n_items)
        # The ID-only variant uses just the user and item features.
        users, items, labels = sample_training_pairs(
            fit_df, self.warm, self.neg_per_pos, self.seed
        )
        indices, x = self._encode(users, items)
        self.model = FactorizationMachine(int(self.offsets[-1]), self.d, self.seed)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.l2, fused=True
        )
        loss_fn = torch.nn.BCEWithLogitsLoss()
        rng = np.random.default_rng(self.seed)
        with training_threads(torch_model=True):
            for epoch in range(self.epochs):
                order = rng.permutation(len(labels))
                # Keep an epoch's shuffled tensors in RAM; batches are contiguous views.
                index = torch.from_numpy(indices[order])
                values = torch.from_numpy(x[order])
                target = torch.from_numpy(labels[order])
                for start in range(0, len(order), self.batch_size):
                    batch = slice(start, start + self.batch_size)
                    optimizer.zero_grad(set_to_none=True)
                    loss = loss_fn(
                        self.model(index[batch], values[batch]), target[batch]
                    )
                    loss.backward()
                    optimizer.step()
                print(f"FM, epoch {epoch + 1}/{self.epochs}")
        self._score_cache = None
        return self

    def _vectors(self):
        w = self.model.w.weight.detach().numpy().ravel()
        v = self.model.v.weight.detach().numpy()

        def group(ids, fields, values, positions):
            linear_term = np.zeros(len(ids))
            summed = np.zeros((len(ids), self.d))
            squares = np.zeros(len(ids))
            for codes, weight, position in zip(fields, values, positions, strict=True):
                vector = v[self.offsets[position] + codes] * weight[:, None]
                linear_term += w[self.offsets[position] + codes] * weight
                summed += vector
                squares += (vector**2).sum(1)
            return linear_term + 0.5 * ((summed**2).sum(1) - squares), summed

        u = np.arange(self.n_users)
        i = np.arange(self.n_items)
        ub, uv = group(
            u,
            [u] + self.user_fields,
            [np.ones(len(u))] + self.user_values,
            [0] + list(range(2 + len(self.item_fields), len(self.sizes))),
        )
        ib, iv = group(
            i,
            [i] + self.item_fields,
            [np.ones(len(i))] + self.item_values,
            [1] + list(range(2, 2 + len(self.item_fields))),
        )
        return ub, uv, ib, iv

    def _score(self, user_idx, timestamp=None):
        if getattr(self, "_score_cache", None) is None:
            self._score_cache = self._vectors()
        cache = self._score_cache
        assert cache is not None
        ub, uv, ib, iv = cache
        return float(self.model.w0.detach()) + ub[user_idx] + ib + iv @ uv[user_idx]

    def _score_batch(self, users, timestamps=None):
        if getattr(self, "_score_cache", None) is None:
            self._score_cache = self._vectors()
        ub, uv, ib, iv = self._score_cache
        scores = uv[users] @ iv.T
        scores += ub[users, None]
        scores += ib[None, :]
        scores += float(self.model.w0.detach())
        return scores
