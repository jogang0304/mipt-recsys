import numpy as np
from numba import njit, prange

from ._runtime import training_threads
from .base import BaseRecommender


@njit(cache=True, parallel=True)
def _als_step(indptr, indices, Y, X, alpha, lam):
    # Each row has its own linear system; only the fixed Y is shared.
    YtY = Y.T @ Y
    d = Y.shape[1]
    for u in prange(len(indptr) - 1):
        start, end = indptr[u], indptr[u + 1]
        if start == end:
            X[u] = 0
            continue
        A = YtY.copy()
        b = np.zeros(d)
        for j in range(d):
            A[j, j] += lam
        for offset in range(start, end):
            y_i = Y[indices[offset]]
            for j in range(d):
                b[j] += (1 + alpha) * y_i[j]
                for k in range(d):
                    A[j, k] += alpha * y_i[j] * y_i[k]
        X[u] = np.linalg.solve(A, b)


class ImplicitALS(BaseRecommender):
    def __init__(self, n_factors=32, alpha=20, lam=0.05, n_iters=10, seed=42):
        self.n_factors = n_factors
        self.alpha = alpha
        self.lam = lam
        self.n_iters = n_iters
        self.seed = seed

    def fit(self, interactions, n_users=None, n_items=None, metadata=None):
        super().fit(interactions, n_users, n_items, metadata)
        P = self._positive_matrix(interactions)
        P.data[:] = 1
        P_T = P.T.tocsr()
        rng = np.random.default_rng(self.seed)
        self.X = np.zeros((self.n_users, self.n_factors))
        self.Y = 0.05 * rng.standard_normal((self.n_items, self.n_factors))
        with training_threads(parallel=True):
            for iteration in range(self.n_iters):
                _als_step(P.indptr, P.indices, self.Y, self.X, self.alpha, self.lam)
                _als_step(P_T.indptr, P_T.indices, self.X, self.Y, self.alpha, self.lam)
                print(f"iALS, iteration {iteration + 1}/{self.n_iters}")
        return self

    def _score(self, user_idx, timestamp=None):
        return self.Y @ self.X[user_idx]

    def _score_batch(self, users, timestamps=None):
        return self.X[users] @ self.Y.T
