"""Parallel kernels for sparse batch scoring and deterministic top-k selection."""

import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True)
def sparse_dense_scores(users, indptr, indices, data, B):
    scores = np.zeros((len(users), B.shape[1]), dtype=B.dtype)
    for row in prange(len(users)):
        u = users[row]
        for offset in range(indptr[u], indptr[u + 1]):
            i = indices[offset]
            for j in range(B.shape[1]):
                scores[row, j] += data[offset] * B[i, j]
    return scores


@njit(cache=True, parallel=True)
def sparse_sparse_scores(users, indptr, indices, data, s_indptr, s_indices, s_data, n_items):
    scores = np.zeros((len(users), n_items))
    for row in prange(len(users)):
        u = users[row]
        for offset in range(indptr[u], indptr[u + 1]):
            i = indices[offset]
            for neighbour in range(s_indptr[i], s_indptr[i + 1]):
                scores[row, s_indices[neighbour]] += data[offset] * s_data[neighbour]
    return scores


@njit(cache=True, parallel=True)
def topk_rows(scores, k):
    """Return up to k item IDs per row, with -1 padding and stable item-ID ties."""
    n_rows, n_items = scores.shape
    k = min(k, n_items)
    top = np.full((n_rows, k), -1, dtype=np.int64)
    for row in prange(n_rows):
        if k == 0:
            continue
        values = scores[row]
        count = 0
        for j in range(n_items):
            if np.isfinite(values[j]):
                count += 1
        size = min(k, count)
        if size == 0:
            continue
        cutoff = np.partition(values, n_items - size)[n_items - size]
        selected = np.empty(size, dtype=np.int64)
        found = 0
        for j in range(n_items):
            if values[j] > cutoff:
                selected[found] = j
                found += 1
        for j in range(n_items):
            if found == size:
                break
            if values[j] == cutoff:
                selected[found] = j
                found += 1
        order = np.argsort(-values[selected], kind="mergesort")
        for j in range(size):
            top[row, j] = selected[order[j]]
    return top
