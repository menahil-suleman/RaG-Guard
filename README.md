# RAGGuard

**Self-monitoring and auto-rollback layer for RAG retrieval pipelines**

RAGGuard builds directly on top of [RetrievalBench](retrievalbench/) and answers a question that a one-time benchmark cannot: *what happens to retrieval quality when the pipeline's inputs degrade, and can the system detect that and protect itself without human intervention?*

It does this by deliberately injecting two types of data drift, measuring the quality drop across three retrieval configurations, and automatically reverting to a known-good configuration when any metric falls below an acceptable threshold — with a full, append-only event log of every decision.

---

## Background

RetrievalBench established clean-condition baselines for three retrieval configurations (dense-only, hybrid dense+BM25, hybrid+cross-encoder reranking) using standard IR metrics — Recall@k, MRR, and nDCG@k — on a 40-chunk medical knowledge base with 59 labelled queries.

RAGGuard reuses those scoring functions unchanged and asks: *how much do those numbers fall when something goes wrong upstream?*

---

## How it works

### 1. Drift injection

Two scenarios simulate realistic failure modes:

**Noisy embeddings** — Gaussian noise (σ = `NOISE_LEVEL`) is added to each query embedding before retrieval. This simulates a garbled transcript, a typo-heavy query, or an upstream encoding glitch that corrupts the vector before it reaches the store. Each retriever's full pipeline is applied to the noisy vector — dense search uses the distorted embedding, BM25 still runs on the raw query text, and the cross-encoder rescores using the raw query string. This means the three configurations respond to noise differently, which is the point.

**Corrupted chunks** — A fraction (`CORRUPTION_RATE`) of document chunks is permanently removed from the vector store before retrieval runs, simulating partial data-loss or an ingestion failure in the knowledge base. A separate degraded vector store is built from the reduced corpus and each retriever is wired to it independently.

### 2. Evaluation under drift

After each drift scenario, all three retrieval configurations are evaluated with the same Recall@k, MRR, and nDCG@k functions used in the clean baseline. Per-query metrics are aggregated and compared against the clean numbers to produce degradation deltas.

### 3. Rollback mechanism

After every drifted evaluation run, each primary metric is checked against a configurable minimum threshold:

- If the score is **above threshold** — pipeline is healthy, no action taken.
- If the score is **below threshold and the fallback config is healthy** — a `[ROLLBACK]` event is logged and the system reverts to the last known-good configuration.
- If the score is **below threshold and the fallback config is also below threshold** — an `[ALL_CONFIGS_BREACHED]` event is logged instead, accurately reflecting that no safe configuration is available rather than falsely reporting a successful revert.

Every event — breach, rollback, or all-configs-breached — is appended to `results/rollback_log.json` with a timestamp, the metric name, measured score, threshold, configuration, and scenario.

---

## Key findings

| Scenario | Dense | Hybrid | Hybrid + Reranked |
|---|---|---|---|
| Clean (baseline) | Recall@5 = 1.00, MRR = 0.97 | Recall@5 = 1.00, MRR = 0.97 | Recall@5 = 1.00, MRR = 0.97 |
| Noisy embeddings (σ=0.25) | Recall@5 = 0.85, MRR = 0.66 | Recall@5 = 0.98, MRR = 0.94 | Recall@5 = 1.00, MRR = 0.97 |
| Corrupted chunks (30% removed) | Recall@5 = 0.64, MRR = 0.64 | Recall@5 = 0.64, MRR = 0.63 | Recall@5 = 0.64, MRR = 0.64 |

**What this shows:**

- BM25 partially shields hybrid retrieval from noisy embeddings — because BM25 uses raw query text, not embeddings, it anchors ranking even when the dense vector is corrupted.
- The cross-encoder fully recovers quality under embedding noise at this noise level, because it rescores using the raw query string with no dependence on the embedding at all.
- Chunk corruption hits all three configurations equally hard — when a relevant document is simply gone from the store, no ranking algorithm can recover it. This is the correct outcome and validates that the rollback mechanism fires `ALL_CONFIGS_BREACHED` rather than pretending to recover.
- The most robust configuration under noise is **not** the same as the configuration with the highest raw performance under clean conditions — a finding that a single-condition benchmark would miss entirely.

---

## Project structure

```
.
├── scoring.py          # Recall@k, MRR, nDCG@k — re-exported from RetrievalBench
├── drift.py            # noisy_embeddings() and corrupt_chunks()
├── rollback.py         # check_threshold(), log_rollback_event(), trigger_rollback()
├── run_experiment.py   # Full orchestrator — all tuning constants at the top
├── config.py           # Shared config (models, paths, k-values)
├── retrievalbench/     # RetrievalBench package (unchanged)
│   ├── retrieval.py    # DenseRetriever, HybridRetriever, HybridRerankedRetriever
│   ├── metrics.py      # Recall@k, MRR, nDCG@k implementations
│   ├── store.py        # In-memory vector store with embedding cache
│   ├── evaluate.py     # Batch evaluation runner and data structures
│   └── visualise.py    # Plots and summary tables
├── data/
│   ├── corpus.json              # 40-chunk medical knowledge base
│   ├── test_set.json            # 59 labelled queries with ground-truth chunk IDs
│   └── embeddings_cache.npy    # Cached bi-encoder embeddings (auto-generated)
├── results/
│   ├── pg_results.csv           # Clean / noisy / corrupted metrics for all configs
│   └── rollback_log.json        # Append-only log of every rollback event
└── plots/
    └── degradation.png          # Grouped bar chart: clean vs. drifted per config
```

---

## Running it

### Install dependencies

```bash
pip install -r requirements.txt
```

Dependencies: `sentence-transformers`, `rank-bm25`, `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `tqdm`. No new dependencies beyond the RetrievalBench baseline.

### Run the full experiment

```bash
python run_experiment.py
```

On first run the bi-encoder embeds all corpus chunks and caches them to `data/embeddings_cache.npy`. Subsequent runs load the cache instantly. If a clean baseline from a previous RetrievalBench run exists at `results/results.json` it is reused, skipping the expensive cross-encoder pass for the clean phase.

### Tune the experiment

All configurable constants are at the top of `run_experiment.py` — nothing is hardcoded inside functions:

| Constant | Default | What it controls |
|---|---|---|
| `NOISE_LEVEL` | `0.25` | Gaussian σ added to query embeddings |
| `CORRUPTION_RATE` | `0.30` | Fraction of corpus chunks removed |
| `NOISE_SEED` | `42` | RNG seed for embedding noise |
| `CORRUPTION_SEED` | `42` | RNG seed for chunk removal |
| `ROLLBACK_THRESHOLDS` | `recall@5 ≥ 0.70, mrr ≥ 0.65, ndcg@5 ≥ 0.68` | Minimum acceptable scores before rollback fires |

### Output files

| File | Contents |
|---|---|
| `results/pg_results.csv` | One row per configuration × scenario, with all metrics and degradation deltas |
| `results/rollback_log.json` | Append-only JSON array of every breach and rollback event across all runs |
| `plots/degradation.png` | Grouped bar chart comparing clean vs. noisy vs. corrupted per config, with threshold reference lines |

---

## Possible extensions

- Additional drift scenarios: embedding model version drift, query distribution shift over time, index staleness.
- Larger and more diverse test sets to reduce variance in the degradation measurements.
- A live monitoring loop that checks metrics on a rolling window of production queries rather than a one-off experiment.
- Automatic threshold calibration from historical performance instead of manually chosen cutoffs.
