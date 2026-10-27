import numpy as np
from scipy.linalg import cho_factor, cho_solve

from ._runtime import training_threads
from .base import BaseRecommender
from ._batch import sparse_dense_scores


class EASE(BaseRecommender):
    def __init__(self, lam=100):
        self.lam = lam

    def fit(self, interactions, n_users=None, n_items=None, metadata=None):
        super().fit(interactions, n_users, n_items, metadata)
        self.X = self._positive_matrix(interactions).astype(np.float64).tocsr()
        gram = (self.X.T @ self.X).toarray()
        gram.flat[:: len(gram) + 1] += self.lam
        with training_threads():
            if self.lam > 0:
                factor = cho_factor(gram, overwrite_a=True, check_finite=False)
                Q = cho_solve(
                    factor, np.eye(self.n_items), overwrite_b=True, check_finite=False
                )
            else:
                Q = np.linalg.inv(gram)
        self.B = -Q / np.diag(Q)[None, :]
        np.fill_diagonal(self.B, 0)
        self.B = self.B.astype(np.float32)
        self.X = self.X.astype(np.float32)
        return self

    def _score(self, user_idx, timestamp=None):
        return self.X.getrow(user_idx) @ self.B

    def _score_batch(self, users, timestamps=None):
        return sparse_dense_scores(users, self.X.indptr, self.X.indices, self.X.data, self.B)
