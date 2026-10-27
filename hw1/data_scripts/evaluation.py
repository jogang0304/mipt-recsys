import os

import numpy as np
import polars as pl

from models._runtime import training_threads


def prepare_evaluation(fit, secret, n_users, n_items, positive_rating=4):
    warm = np.zeros(n_items, dtype=bool)
    warm[fit["item_idx"].to_numpy()] = True
    target = np.full(n_users, -1, dtype=int)
    ratings = np.zeros(n_users)
    query_time = np.zeros(n_users)
    users = secret["user_idx"].to_numpy()
    target[users] = secret["item_idx"].to_numpy()
    ratings[users] = secret["rating"].to_numpy()
    query_time[users] = secret["timestamp"].to_numpy()
    eligible = (
        (target >= 0) & (ratings >= positive_rating) & warm[np.maximum(target, 0)]
    )
    return {
        "warm": warm,
        "target": target,
        "query_time": query_time,
        "users": np.flatnonzero(eligible),
    }


def evaluation_report(evaluation, heldout):
    return pl.DataFrame(
        {
            "metric": [
                "users",
                "below relevance threshold",
                "cold target item",
                "eligible for evaluation",
            ],
            "value": [
                len(heldout),
                int((heldout["rating"] < 4).sum()),
                int((~evaluation["warm"][heldout["item_idx"].to_numpy()]).sum()),
                len(evaluation["users"]),
            ],
        }
    )


def ndcg_at_k(rank, k=10):
    return 1 / np.log2(rank + 1) if rank is not None and 1 <= rank <= k else 0.0


def recall_at_k(rank, k=10):
    return float(rank is not None and 1 <= rank <= k)


def _evaluation_batch_size(n_items):
    """Budget for scores, model output and indexing temporaries, not the full test set."""
    if "RECSYS_EVAL_MEMORY_MB" in os.environ:
        budget = int(os.environ["RECSYS_EVAL_MEMORY_MB"]) * 1024**2
        if budget <= 0:
            raise ValueError("RECSYS_EVAL_MEMORY_MB must be positive")
    else:
        budget = 512 * 1024**2
        if hasattr(os, "sysconf"):
            available = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
            budget = min(budget, max(1, available // 8))
    return max(1, min(4096, budget // max(24 * n_items, 1)))


def ranking_metrics(model, evaluation, users=None, k=10, *, batch_size=None):
    """Evaluate batches on all CPU cores without materializing all user/item scores.

    The result and per-user nullable ranks follow the original scalar evaluator:
    empty recommendation lists are excluded from metric averages and rank rows.
    """
    users = np.asarray(
        evaluation["users"]
        if users is None
        else np.intersect1d(evaluation["users"], users),
        dtype=np.int64,
    )
    if batch_size is None:
        batch_size = _evaluation_batch_size(model.n_items)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    ranks = np.zeros(len(users), dtype=np.int64)
    nonempty = np.zeros(len(users), dtype=bool)
    recommended = np.zeros(model.n_items, dtype=bool)
    with training_threads():
        for start in range(0, len(users), batch_size):
            end = min(start + batch_size, len(users))
            batch_users = users[start:end]
            top = model.recommend_batch(
                batch_users, k=k, timestamps=evaluation["query_time"][batch_users]
            )
            valid = top >= 0
            nonempty[start:end] = valid.any(axis=1)
            recommended[top[valid]] = True
            if top.shape[1]:
                hits = valid & (top == evaluation["target"][batch_users, None])
                ranks[start:end] = np.where(
                    hits.any(axis=1), hits.argmax(axis=1) + 1, 0
                )
    valid_ranks = ranks[nonempty]
    hit_ranks = valid_ranks[valid_ranks > 0]
    count = len(valid_ranks)
    warm_count = np.count_nonzero(evaluation["warm"])
    metrics = {
        f"NDCG@{k}": float(np.sum(1 / np.log2(hit_ranks + 1)) / count)
        if count
        else np.nan,
        f"Recall@{k}": len(hit_ranks) / count if count else np.nan,
        "coverage": np.count_nonzero(recommended) / warm_count
        if warm_count
        else np.nan,
        "eligible_users": count,
        "empty_candidates": len(users) - count,
    }
    details = pl.DataFrame(
        {"user_idx": users[nonempty], "rank": valid_ranks}
    ).with_columns(
        pl.when(pl.col("rank") > 0).then(pl.col("rank")).otherwise(None).alias("rank")
    )
    return metrics, details
