"""
RetrievalBench — Central configuration.

All tuneable constants live here so the rest of the codebase
never contains magic numbers.
"""

# ── Models ────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"          # fast, 384-dim bi-encoder
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # reranker

# ── Retrieval knobs ───────────────────────────────────────────────────────────
TOP_K = 10          # candidates retrieved before reranking
RERANK_TOP_N = 10   # top-N after reranking (kept == TOP_K for fair comparison)

# Hybrid score: final_score = ALPHA * dense_score + (1 - ALPHA) * bm25_score
HYBRID_ALPHA = 0.5

# ── Evaluation thresholds ─────────────────────────────────────────────────────
K_VALUES = [1, 3, 5, 10]   # k values for Recall@k and nDCG@k

# ── Paths ─────────────────────────────────────────────────────────────────────
import os

ROOT_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(ROOT_DIR, "data")
RESULTS_DIR    = os.path.join(ROOT_DIR, "results")
PLOTS_DIR      = os.path.join(ROOT_DIR, "plots")
CORPUS_FILE    = os.path.join(DATA_DIR, "corpus.json")
TEST_SET_FILE  = os.path.join(DATA_DIR, "test_set.json")
RESULTS_FILE   = os.path.join(RESULTS_DIR, "results.json")
RESULTS_CSV    = os.path.join(RESULTS_DIR, "results.csv")
