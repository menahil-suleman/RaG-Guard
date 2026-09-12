"""
retrievalbench/metrics.py — Standard information-retrieval evaluation metrics.

All functions share the same call signature:

    metric(retrieved_ids, relevant_ids, k) -> float

Parameters
──────────
retrieved_ids : list[str]
    Ordered list of chunk_ids returned by a retriever (rank 1 = index 0).
relevant_ids  : list[str]
    Ground-truth relevant chunk_ids for the query.
k             : int
    Cut-off depth.

Metrics implemented
───────────────────
recall_at_k     — fraction of relevant items found in top-k.
                  Binary for single-relevant-item queries.
mrr             — mean reciprocal rank (k is ignored; full list used).
ndcg_at_k       — normalised discounted cumulative gain at k.
                  Uses binary relevance (1 if in relevant_ids, else 0).

Aggregation helpers
───────────────────
compute_query_metrics   — evaluates all metrics for a single query.
aggregate_metrics       — averages per-query results across the test set.
"""

from __future__ import annotations

import math
from collections import defaultdict


# ─────────────────────────────────────────────────────────────────────────────
# Core metric functions
# ─────────────────────────────────────────────────────────────────────────────

def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float:
    """
    Recall@k — proportion of relevant items that appear in the top-k results.

    For queries with a single relevant item (the common case here) this is
    equivalent to a binary hit/miss indicator.

    Returns a float in [0, 1].
    """
    if not relevant_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    retrieved_at_k = set(retrieved_ids[:k])
    hits = len(retrieved_at_k & relevant_set)
    return hits / len(relevant_set)


def mrr(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int | None = None,
) -> float:
    """
    Mean Reciprocal Rank (MRR) — reciprocal of the rank of the first
    relevant item.

    ``k`` is accepted for API consistency but is ignored; the full retrieved
    list is used (which matches standard MRR definition).

    Returns a float in [0, 1].  Returns 0 if no relevant item is found.
    """
    relevant_set = set(relevant_ids)
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float:
    """
    nDCG@k — Normalised Discounted Cumulative Gain at depth k.

    Uses binary relevance: gain is 1 for relevant items, 0 otherwise.
    The ideal DCG (IDCG) is computed from the sorted ideal ranking, capped
    at k positions.

    Returns a float in [0, 1].
    """
    if not relevant_ids:
        return 0.0

    relevant_set = set(relevant_ids)
    n_relevant = len(relevant_ids)

    # Actual DCG
    dcg = 0.0
    for rank, cid in enumerate(retrieved_ids[:k], start=1):
        if cid in relevant_set:
            dcg += 1.0 / math.log2(rank + 1)

    # Ideal DCG — place all relevant items at the top positions
    ideal_positions = min(n_relevant, k)
    idcg = sum(1.0 / math.log2(pos + 1) for pos in range(1, ideal_positions + 1))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


# ─────────────────────────────────────────────────────────────────────────────
# Per-query metric bundle
# ─────────────────────────────────────────────────────────────────────────────

def compute_query_metrics(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k_values: list[int],
) -> dict[str, float]:
    """
    Compute all metrics for a single query.

    Returns a dict with keys like:
        "recall@1", "recall@3", "recall@5", "recall@10",
        "mrr",
        "ndcg@1",  "ndcg@3",  "ndcg@5",  "ndcg@10"
    """
    result: dict[str, float] = {}

    for k in k_values:
        result[f"recall@{k}"] = recall_at_k(retrieved_ids, relevant_ids, k)
        result[f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, relevant_ids, k)

    result["mrr"] = mrr(retrieved_ids, relevant_ids)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation across the test set
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_metrics(per_query_results: list[dict[str, float]]) -> dict[str, float]:
    """
    Average per-query metric dicts into a single aggregate dict.

    Parameters
    ----------
    per_query_results : list of dicts, one per query, as returned by
        compute_query_metrics().

    Returns
    -------
    dict mapping metric name → mean value across all queries.
    """
    if not per_query_results:
        return {}

    totals: dict[str, float] = defaultdict(float)
    for row in per_query_results:
        for key, val in row.items():
            totals[key] += val

    n = len(per_query_results)
    return {k: v / n for k, v in totals.items()}
