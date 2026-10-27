import numpy as np
import polars as pl
from scipy import sparse

from ._batch import topk_rows


class BaseRecommender:
    def fit(self, interactions, n_users=None, n_items=None, metadata=None):
        self.n_users = n_users or int(interactions["user_idx"].max()) + 1
        self.n_items = n_items or (
            len(metadata)
            if metadata is not None
            else int(interactions["item_idx"].max()) + 1
        )
        users = interactions["user_idx"].to_numpy()
        items = interactions["item_idx"].to_numpy()
        self.seen = sparse.csr_matrix(
            (np.ones(len(interactions), dtype=bool), (users, items)),
            shape=(self.n_users, self.n_items),
        )
        self.seen.sort_indices()
        self.warm = np.zeros(self.n_items, dtype=bool)
        self.warm[items] = True
        positive_items = interactions.filter(pl.col("rating") >= 4)[
            "item_idx"
        ].to_numpy()
        self.popularity = np.bincount(positive_items, minlength=self.n_items).astype(
            float
        )
        return self

    def _positive_matrix(self, interactions):
        positive = interactions.filter(pl.col("rating") >= 4)
        return sparse.csr_matrix(
            (
                np.ones(len(positive)),
                (positive["user_idx"].to_numpy(), positive["item_idx"].to_numpy()),
            ),
            shape=(self.n_users, self.n_items),
        )

    def _score(self, user_idx, timestamp=None):
        raise NotImplementedError

    def recommend(self, user_idx, k=10, timestamp=None):
        if k <= 0:
            return np.array([], dtype=int)
        scores = self.popularity.copy()
        seen = np.array([], dtype=int)
        if 0 <= user_idx < self.n_users:
            start, end = self.seen.indptr[user_idx : user_idx + 2]
            seen = self.seen.indices[start:end]
            if len(seen):
                scores = np.asarray(self._score(user_idx, timestamp)).ravel()
        scores = np.where(self.warm, scores, -np.inf)
        scores[seen] = -np.inf
        candidates = np.flatnonzero(np.isfinite(scores))
        if not len(candidates):
            return np.array([], dtype=int)
        if len(candidates) > k:
            values = scores[candidates]
            cutoff = np.partition(values, len(values) - k)[len(values) - k]
            above = candidates[values > cutoff]
            tied = candidates[values == cutoff][: k - len(above)]
            candidates = np.r_[above, tied]
        return candidates[np.lexsort((candidates, -scores[candidates]))]

    def _score_batch(self, users, timestamps=None):
        """Compatibility fallback for custom models; built-in models override this."""
        times = [None] * len(users) if timestamps is None else timestamps
        return np.stack([
            np.asarray(self._score(int(u), t)).ravel()
            for u, t in zip(users, times, strict=True)
        ])

    def recommend_batch(self, users, k=10, timestamps=None):
        """Score a user batch, returning a matrix of item IDs padded with -1.

        Each row follows recommend's filtering, cold-user fallback and tie order.
        Callers can bound peak RAM by splitting users into batches.
        """
        users = np.asarray(users, dtype=np.int64)
        if users.ndim != 1:
            raise ValueError("users must be a one-dimensional array")
        if timestamps is not None:
            timestamps = np.broadcast_to(np.asarray(timestamps), users.shape)
        if k <= 0 or not len(users):
            return np.full((len(users), max(0, min(k, self.n_items))), -1, dtype=np.int64)
        scores = np.broadcast_to(self.popularity, (len(users), self.n_items)).copy()
        valid = (users >= 0) & (users < self.n_users)
        rows = np.flatnonzero(valid)
        known_users = users[rows]
        active = (self.seen.indptr[known_users + 1] - self.seen.indptr[known_users]) > 0
        active_rows = rows[active]
        if len(active_rows):
            times = None if timestamps is None else timestamps[active_rows]
            scores[active_rows] = self._score_batch(users[active_rows], times)
        scores[:, ~self.warm] = -np.inf
        seen = self.seen[known_users].tocoo()
        scores[rows[seen.row], seen.col] = -np.inf
        scores[~np.isfinite(scores)] = -np.inf
        return topk_rows(scores, k)
