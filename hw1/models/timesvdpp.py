import numpy as np
from scipy import sparse
from numba import njit, prange

from ._runtime import training_threads
from .base import BaseRecommender

MILLISECONDS_PER_DAY = 86_400_000.0


def time_deviation(t, t_bar, beta):
    delta = np.asarray(t, dtype=float) - t_bar
    return np.sign(delta) * np.abs(delta) ** beta


def time_bins(t, edges):
    if edges[0] == edges[-1]:
        return np.zeros(len(t), dtype=int)
    return np.searchsorted(edges[1:-1], t, side="right")


@njit(cache=True, parallel=True)
def _prepare_time(indptr, t, n_bins, use_time, beta):
    n_users = len(indptr) - 1
    t_bar = np.zeros(n_users)
    edges = np.zeros((n_users, n_bins + 1))
    varying = np.zeros(n_users, dtype=np.bool_)
    bins = np.zeros(len(t), dtype=np.int64)
    dev = np.zeros(len(t))
    for u in prange(n_users):
        start, end = indptr[u], indptr[u + 1]
        if start == end:
            continue
        values = np.sort(t[start:end])
        t_bar[u] = t[start:end].mean()
        for b in range(n_bins + 1):
            position = (end - start - 1) * (b / n_bins)
            lower = int(position)
            upper = min(lower + 1, len(values) - 1)
            fraction = position - lower
            edges[u, b] = values[lower] + fraction * (values[upper] - values[lower])
        varying[u] = use_time and values[0] != values[-1]
        if varying[u]:
            for row in range(start, end):
                bins[row] = np.searchsorted(edges[u, 1:-1], t[row], side="right")
                delta = t[row] - t_bar[u]
                dev[row] = np.sign(delta) * abs(delta) ** beta
    return t_bar, edges, varying, bins, dev


@njit(cache=True)
def _train_epoch(
    order,
    indptr,
    i,
    r_ui,
    bins,
    dev,
    N_indptr,
    N_indices,
    mu,
    b_u,
    b_i,
    alpha,
    a,
    p,
    q,
    y,
    b_u_bin,
    varying,
    lr,
    reg,
):
    # Keep the seeded user order: parallel user updates would race on shared q/y.
    d = p.shape[1]
    for u in order:
        start, end = indptr[u], indptr[u + 1]
        count = end - start
        N_start, N_end = N_indptr[u], N_indptr[u + 1]
        N_count = N_end - N_start
        history = np.zeros(d)
        if N_count:
            for offset in range(N_start, N_end):
                for f in range(d):
                    history[f] += y[N_indices[offset], f] / np.sqrt(N_count)
        old_q = q[i[start:end]].copy()
        old_b_i = b_i[i[start:end]].copy()
        effective = np.empty((count, d))
        error = np.empty(count)
        gradient = np.zeros(d)
        drift_gradient = np.zeros(d)
        bin_gradient = np.zeros(b_u_bin.shape[1])
        for row in range(count):
            idx = start + row
            r_hat = (
                mu + b_u[u] + old_b_i[row] + alpha[u] * dev[idx] + b_u_bin[u, bins[idx]]
            )
            for f in range(d):
                effective[row, f] = p[u, f] + history[f] + dev[idx] * a[u, f]
                r_hat += old_q[row, f] * effective[row, f]
            error[row] = r_ui[idx] - r_hat
            bin_gradient[bins[idx]] += error[row] / count
            for f in range(d):
                gradient[f] += error[row] * old_q[row, f] / count
                drift_gradient[f] += error[row] * dev[idx] * old_q[row, f] / count
        b_u[u] += lr * (error.mean() - reg * b_u[u])
        if varying[u]:
            alpha_gradient = 0.0
            for row in range(count):
                alpha_gradient += error[row] * dev[start + row] / count
            alpha[u] += lr * 0.01 * (alpha_gradient - reg * alpha[u])
            for f in range(d):
                a[u, f] += lr * 0.01 * (drift_gradient[f] - reg * a[u, f])
            for b in range(b_u_bin.shape[1]):
                b_u_bin[u, b] += lr * (bin_gradient[b] - reg * b_u_bin[u, b])
        for f in range(d):
            p[u, f] += lr * (gradient[f] - reg * p[u, f])
        for row in range(count):
            item = i[start + row]
            b_i[item] = old_b_i[row] + lr * (error[row] - reg * old_b_i[row]) / count
            for f in range(d):
                q[item, f] = (
                    old_q[row, f]
                    + lr
                    * (error[row] * effective[row, f] - reg * old_q[row, f])
                    / count
                )
        for offset in range(N_start, N_end):
            j = N_indices[offset]
            for f in range(d):
                y[j, f] += lr * (gradient[f] / np.sqrt(N_count) - reg * y[j, f])


class TimeSVDpp(BaseRecommender):
    def __init__(
        self,
        n_factors=32,
        epochs=10,
        lr=0.005,
        reg=0.05,
        beta=0.4,
        n_bins=5,
        seed=42,
        use_time=True,
    ):
        self.n_factors, self.epochs, self.lr, self.reg = n_factors, epochs, lr, reg
        self.beta, self.n_bins, self.seed = beta, n_bins, seed
        self.use_time = use_time

    def fit(self, interactions, n_users=None, n_items=None, metadata=None):
        if interactions.is_empty() or self.n_bins < 1:
            raise ValueError("Ratings and at least one time bin are required")
        super().fit(interactions, n_users, n_items, metadata)
        rng = np.random.default_rng(self.seed)
        d = self.n_factors
        self.mu = float(interactions["rating"].mean())
        self.b_u = np.zeros(self.n_users)
        self.b_i = np.zeros(self.n_items)
        self.alpha = np.zeros(self.n_users)
        self.a = np.zeros((self.n_users, d))
        self.p = 0.05 * rng.standard_normal((self.n_users, d))
        self.q = 0.05 * rng.standard_normal((self.n_items, d))
        self.y = np.zeros((self.n_items, d))
        self.b_u_bin = np.zeros((self.n_users, self.n_bins))
        # Contiguous arrays replace one Polars frame/Python tuple per user.
        ordered = interactions.sort("user_idx", maintain_order=True)
        users = ordered["user_idx"].to_numpy()
        i = ordered["item_idx"].to_numpy().astype(np.int64)
        r_ui = ordered["rating"].to_numpy().astype(float)
        t = ordered["timestamp"].to_numpy().astype(float) / MILLISECONDS_PER_DAY
        indptr = np.r_[0, np.cumsum(np.bincount(users, minlength=self.n_users))]
        N = self._positive_matrix(interactions)
        self.N = [N.indices[N.indptr[u] : N.indptr[u + 1]] for u in range(self.n_users)]
        active_users = np.flatnonzero(np.diff(indptr))
        with training_threads(parallel=True):
            self.t_bar, self.user_bins, self.time_varying, bins, dev = _prepare_time(
                indptr, t, self.n_bins, self.use_time, self.beta
            )
            for epoch in range(self.epochs):
                _train_epoch(
                    rng.permutation(active_users),
                    indptr,
                    i,
                    r_ui,
                    bins,
                    dev,
                    N.indptr,
                    N.indices,
                    self.mu,
                    self.b_u,
                    self.b_i,
                    self.alpha,
                    self.a,
                    self.p,
                    self.q,
                    self.y,
                    self.b_u_bin,
                    self.time_varying,
                    self.lr,
                    self.reg,
                )
                print(f"timeSVD++, epoch {epoch + 1}/{self.epochs}")
        self._history_cache = None
        return self

    def _time_context(self, u, t):
        if not self.time_varying[u]:
            return np.zeros(len(t), dtype=int), np.zeros(len(t))
        return (
            time_bins(t, self.user_bins[u]),
            time_deviation(t, self.t_bar[u], self.beta),
        )

    def _history_term(self, u):
        N_u = self.N[u]
        if not len(N_u):
            return np.zeros(self.n_factors)
        return self.y[N_u].sum(axis=0) / np.sqrt(len(N_u))

    def _score(self, user_idx, timestamp=None):
        t = (
            self.t_bar[user_idx]
            if timestamp is None
            else timestamp / MILLISECONDS_PER_DAY
        )
        bins, dev = self._time_context(user_idx, np.array([t]))
        effective = (
            self.p[user_idx] + self._history_term(user_idx) + self.a[user_idx] * dev[0]
        )
        return (
            self.mu
            + self.b_u[user_idx]
            + self.b_i
            + self.alpha[user_idx] * dev[0]
            + self.b_u_bin[user_idx, bins[0]]
            + self.q @ effective
        )

    def _score_batch(self, users, timestamps=None):
        if getattr(self, "_history_cache", None) is None:
            # Construct the implicit history matrix once, including for older pickles.
            counts = np.fromiter(map(len, self.N), dtype=np.int64, count=self.n_users)
            indptr = np.r_[0, np.cumsum(counts)]
            indices = np.concatenate(self.N)
            N = sparse.csr_matrix(
                (np.ones(len(indices)), indices, indptr),
                shape=(self.n_users, self.n_items),
            )
            self._history_cache = (N @ self.y) / np.sqrt(np.maximum(counts, 1))[:, None]
        t = self.t_bar[users] if timestamps is None else timestamps / MILLISECONDS_PER_DAY
        delta = t - self.t_bar[users]
        dev = np.sign(delta) * np.abs(delta) ** self.beta
        dev = np.where(self.time_varying[users], dev, 0.0)
        bins = (t[:, None] >= self.user_bins[users, 1:-1]).sum(axis=1)
        bins = np.where(self.time_varying[users], bins, 0)
        effective = self.p[users] + self._history_cache[users] + self.a[users] * dev[:, None]
        scores = effective @ self.q.T
        scores += self.b_i[None, :]
        scores += (
            self.mu + self.b_u[users] + self.alpha[users] * dev + self.b_u_bin[users, bins]
        )[:, None]
        return scores
