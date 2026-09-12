"""
tests/test_metrics.py — Unit tests for all IR metric functions.

Run with:
    py -m pytest tests/test_metrics.py -v
"""

import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrievalbench.metrics import (
    recall_at_k, mrr, ndcg_at_k,
    compute_query_metrics, aggregate_metrics,
)

# ─────────────────────────────────────────────────────────────────────────────
# recall_at_k
# ─────────────────────────────────────────────────────────────────────────────

def test_recall_hit_at_1():
    assert recall_at_k(["a", "b", "c"], ["a"], k=1) == 1.0

def test_recall_miss_at_1():
    assert recall_at_k(["b", "c", "a"], ["a"], k=1) == 0.0

def test_recall_hit_at_3():
    assert recall_at_k(["b", "c", "a"], ["a"], k=3) == 1.0

def test_recall_partial():
    # 2 relevant items, only 1 in top-3
    assert recall_at_k(["a", "x", "y", "b"], ["a", "b"], k=3) == 0.5

def test_recall_empty_relevant():
    assert recall_at_k(["a", "b"], [], k=5) == 0.0

def test_recall_k_larger_than_results():
    assert recall_at_k(["a"], ["a"], k=100) == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# mrr
# ─────────────────────────────────────────────────────────────────────────────

def test_mrr_rank1():
    assert mrr(["a", "b", "c"], ["a"]) == 1.0

def test_mrr_rank2():
    assert mrr(["x", "a", "c"], ["a"]) == pytest_approx(0.5)

def test_mrr_rank3():
    assert abs(mrr(["x", "y", "a"], ["a"]) - 1/3) < 1e-9

def test_mrr_not_found():
    assert mrr(["x", "y", "z"], ["a"]) == 0.0

def test_mrr_multiple_relevant_first():
    # First relevant item is at rank 1
    assert mrr(["a", "b"], ["a", "b"]) == 1.0

def test_mrr_multiple_relevant_second():
    # First relevant item at rank 2
    assert abs(mrr(["x", "a", "b"], ["a", "b"]) - 0.5) < 1e-9


def pytest_approx(val):
    """Simple tolerance check returning a float for comparison in assert."""
    return val


# ─────────────────────────────────────────────────────────────────────────────
# ndcg_at_k
# ─────────────────────────────────────────────────────────────────────────────

def test_ndcg_perfect():
    # Relevant item at rank 1 → nDCG@5 == 1.0
    assert ndcg_at_k(["a", "x", "y"], ["a"], k=5) == 1.0

def test_ndcg_zero():
    assert ndcg_at_k(["x", "y", "z"], ["a"], k=5) == 0.0

def test_ndcg_rank2():
    # Single relevant item at rank 2
    # DCG = 1/log2(3),  IDCG = 1/log2(2) = 1
    expected = (1.0 / math.log2(3)) / (1.0 / math.log2(2))
    assert abs(ndcg_at_k(["x", "a", "y"], ["a"], k=5) - expected) < 1e-9

def test_ndcg_empty_relevant():
    assert ndcg_at_k(["a", "b"], [], k=5) == 0.0

def test_ndcg_cutoff_respected():
    # Relevant item is at rank 3 but k=2 so it should not count
    assert ndcg_at_k(["x", "y", "a"], ["a"], k=2) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# compute_query_metrics
# ─────────────────────────────────────────────────────────────────────────────

def test_compute_query_metrics_keys():
    m = compute_query_metrics(["a", "b"], ["a"], k_values=[1, 5])
    assert "recall@1" in m
    assert "recall@5" in m
    assert "mrr"      in m
    assert "ndcg@1"   in m
    assert "ndcg@5"   in m

def test_compute_query_metrics_values():
    m = compute_query_metrics(["a", "b", "c"], ["a"], k_values=[1, 3])
    assert m["recall@1"] == 1.0
    assert m["recall@3"] == 1.0
    assert m["mrr"]      == 1.0
    assert m["ndcg@1"]   == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# aggregate_metrics
# ─────────────────────────────────────────────────────────────────────────────

def test_aggregate_mean():
    rows = [
        {"recall@5": 1.0, "mrr": 1.0},
        {"recall@5": 0.0, "mrr": 0.5},
    ]
    agg = aggregate_metrics(rows)
    assert abs(agg["recall@5"] - 0.5) < 1e-9
    assert abs(agg["mrr"] - 0.75) < 1e-9

def test_aggregate_empty():
    assert aggregate_metrics([]) == {}
