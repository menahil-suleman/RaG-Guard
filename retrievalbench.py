"""
retrievalbench.py — Single-file convenience module.

Exposes the primary public API so callers can do:

    from retrievalbench import evaluate
    results = evaluate(configurations, test_set)

or run a quick benchmark with sensible defaults:

    python retrievalbench.py --corpus data/corpus.json \
                              --test_set data/test_set.json \
                              --top_k 10

This file is intentionally a thin re-export layer; all logic lives in
the retrievalbench/ package.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# ── Public re-exports ─────────────────────────────────────────────────────────
from retrievalbench.evaluate import evaluate, save_results, EvaluationResults  # noqa: F401
from retrievalbench.metrics import (                                             # noqa: F401
    recall_at_k, mrr, ndcg_at_k,
    compute_query_metrics, aggregate_metrics,
)
from retrievalbench.store import VectorStore                                     # noqa: F401
from retrievalbench.retrieval import (                                           # noqa: F401
    DenseRetriever, HybridRetriever, HybridRerankedRetriever,
)
from retrievalbench.visualise import (                                           # noqa: F401
    print_summary_table, save_all_plots,
    plot_metrics_bar, plot_latency_tradeoff,
)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _build_default_configurations(store: VectorStore, cross_encoder_model: str, alpha: float):
    dense    = DenseRetriever(store=store)
    hybrid   = HybridRetriever(store=store, alpha=alpha)
    reranked = HybridRerankedRetriever(
        hybrid=hybrid,
        cross_encoder_model=cross_encoder_model,
        rerank_candidates=20,
    )
    return [dense, hybrid, reranked]


def main() -> None:
    from config import (
        EMBEDDING_MODEL, CROSS_ENCODER_MODEL,
        TOP_K, K_VALUES, HYBRID_ALPHA,
        DATA_DIR, RESULTS_DIR, PLOTS_DIR,
        CORPUS_FILE, TEST_SET_FILE, RESULTS_FILE, RESULTS_CSV,
    )

    parser = argparse.ArgumentParser(
        description="RetrievalBench — run the full retrieval quality benchmark"
    )
    parser.add_argument("--corpus",   default=CORPUS_FILE,   help="Path to corpus.json")
    parser.add_argument("--test_set", default=TEST_SET_FILE, help="Path to test_set.json")
    parser.add_argument("--top_k",    default=TOP_K,         type=int)
    parser.add_argument("--results",  default=RESULTS_FILE,  help="Output JSON path")
    parser.add_argument("--csv",      default=RESULTS_CSV,   help="Output CSV path")
    parser.add_argument("--plots_dir",default=PLOTS_DIR,     help="Directory for plots")
    parser.add_argument("--no_rerank", action="store_true",  help="Skip reranking (faster)")
    args = parser.parse_args()

    with open(args.corpus, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    with open(args.test_set, "r", encoding="utf-8") as f:
        test_set = json.load(f)

    print(f"Corpus: {len(corpus)} chunks | Test set: {len(test_set)} queries")

    store = VectorStore(model_name=EMBEDDING_MODEL, cache_dir=DATA_DIR)
    store.build(corpus)

    cfgs = _build_default_configurations(store, CROSS_ENCODER_MODEL, HYBRID_ALPHA)
    if args.no_rerank:
        cfgs = [c for c in cfgs if c.name != "hybrid_reranked"]

    results = evaluate(cfgs, test_set, top_k=args.top_k, k_values=K_VALUES)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)
    save_results(results, args.results, args.csv)
    save_all_plots(results, args.plots_dir)
    print_summary_table(results, k_values=K_VALUES)


if __name__ == "__main__":
    main()
