import numpy as np
from numba import njit, prange
from scipy import sparse

from ._runtime import training_threads
from .base import BaseRecommender
from ._batch import sparse_sparse_scores


@njit(cache=True, parallel=True)
def _knn_neighbors(n_ij, n_i, k, lam, min_overlap):
    n_items = len(n_i)
    neighbours = np.zeros((n_items, k), dtype=np.int64)
    weights = np.zeros((n_items, k))
    for i in prange(n_items):
        if k == 0:
            continue
        s = np.zeros(n_items)
        for j in range(n_items):
            overlap = n_ij[i, j]
            if (
                i != j
                and overlap >= min_overlap
                and overlap > 0
                and n_i[i] > 0
                and n_i[j] > 0
            ):
                s[j] = (overlap / np.sqrt(n_i[i] * n_i[j])) * (
                    overlap / (overlap + lam)
                )
        cutoff = np.partition(s, n_items - k)[n_items - k]
        selected = np.empty(k, dtype=np.int64)
        count = 0
        for j in range(n_items):
            if s[j] > cutoff:
                selected[count] = j
                count += 1
        for j in range(n_items):
            if count == k:
                break
            if s[j] == cutoff:
                selected[count] = j
                count += 1
        # Stable tie ordering matches the original full argsort (lowest item ID first).
        order = np.argsort(-s[selected], kind="mergesort")
        for pos in range(k):
            neighbours[i, pos] = selected[order[pos]]
            weights[i, pos] = s[selected[order[pos]]]
    return neighbours, weights


class ItemKNN(BaseRecommender):
    def __init__(self, k=50, lam=10, min_overlap=3):
        self.k = k
        self.lam = lam
        self.min_overlap = min_overlap

    def fit(self, interactions, n_users=None, n_items=None, metadata=None):
        super().fit(interactions, n_users, n_items, metadata)
        self.X = self._positive_matrix(interactions)
        self.X.data[:] = 1
        n_ij = (self.X.T @ self.X).toarray().astype(float)
        n_i = np.diag(n_ij).copy()
        with training_threads(parallel=True):
            neighbours, values = _knn_neighbors(
                n_ij, n_i, min(self.k, self.n_items), self.lam, self.min_overlap
            )
        rows = np.repeat(np.arange(self.n_items), neighbours.shape[1])
        cols, values = neighbours.ravel(), values.ravel()
        keep = values > 0
        self.s = sparse.csr_matrix(
            (values[keep], (rows[keep], cols[keep])), shape=(self.n_items, self.n_items)
        )
        self._scoring_matrix = self.s.T.tocsr()
        return self

    def _score(self, user_idx, timestamp=None):
        return (self.X.getrow(user_idx) @ self._scoring_matrix).toarray().ravel()

    def _score_batch(self, users, timestamps=None):
        s = self._scoring_matrix
        return sparse_sparse_scores(
            users, self.X.indptr, self.X.indices, self.X.data,
            s.indptr, s.indices, s.data, self.n_items,
        )
