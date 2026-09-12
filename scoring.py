"""
scoring.py — RAGGuard scoring functions.

Re-exports Recall@k, MRR, and nDCG@k directly from RetrievalBench so the
rest of RAGGuard can import them from a single local module without
duplicating any logic.

Usage
─────
    from scoring import recall_at_k, mrr, ndcg_at_k
    from scoring import compute_query_metrics, aggregate_metrics
"""

# Re-export unchanged implementations from RetrievalBench
from retrievalbench.metrics import (       # noqa: F401
    recall_at_k,
    mrr,
    ndcg_at_k,
    compute_query_metrics,
    aggregate_metrics,
)

__all__ = [
    "recall_at_k",
    "mrr",
    "ndcg_at_k",
    "compute_query_metrics",
    "aggregate_metrics",
]
