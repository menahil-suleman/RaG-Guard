"""
run_experiment.py — RAGGuard full experiment orchestrator.

Flow
────
1. Load corpus and test set.
2. Build vector store (uses embedding cache when available).
3. Instantiate all three retrieval configurations (dense, hybrid, hybrid+reranked).
4. Run CLEAN baseline evaluation — reuses existing results/results.json when
   present so the heavy cross-encoder pass only runs once.
5. Run both DRIFT scenarios (noisy embeddings, corrupted chunks) across all
   three configurations.
6. For each drifted run, check every primary metric against its threshold;
   trigger and log a rollback whenever a breach is detected.
7. Save a unified results table (results/pg_results.csv) and one degradation
   comparison plot (plots/degradation.png).
8. Print a summary table to stdout.

Configurable constants
──────────────────────
All tuning knobs are at the top of this file — never buried inside functions.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# ── RetrievalBench imports (unchanged) ────────────────────────────────────────
from config import (
    EMBEDDING_MODEL, CROSS_ENCODER_MODEL,
    TOP_K, RERANK_TOP_N, HYBRID_ALPHA,
    K_VALUES, DATA_DIR, RESULTS_DIR, PLOTS_DIR,
    CORPUS_FILE, TEST_SET_FILE, RESULTS_FILE, RESULTS_CSV,
)
from retrievalbench.store import VectorStore
from retrievalbench.retrieval import (
    DenseRetriever, HybridRetriever, HybridRerankedRetriever,
)
from scoring import compute_query_metrics, aggregate_metrics

# ── RAGGuard imports ──────────────────────────────────────────────────────────
from drift import noisy_embeddings, corrupt_chunks
from rollback import check_threshold, log_rollback_event, trigger_rollback

# =============================================================================
# CONFIGURABLE CONSTANTS — change these, not the functions below
# =============================================================================

# --- Drift scenario parameters -----------------------------------------------

# Gaussian noise standard deviation added to query embeddings (Scenario A).
# 0.0 = no noise;  0.15 = moderate;  0.40 = heavy
NOISE_LEVEL: float = 0.25

# Fraction of corpus chunks to remove (Scenario B, "remove" mode).
# 0.0 = none removed;  0.30 = 30 % of chunks deleted
CORRUPTION_RATE: float = 0.30

# Random seeds for reproducibility
NOISE_SEED: int = 42
CORRUPTION_SEED: int = 42

# --- Rollback thresholds ------------------------------------------------------
# A score *below* the threshold fires a rollback event.
# Keys must match metric names produced by compute_query_metrics().
ROLLBACK_THRESHOLDS: dict[str, float] = {
    "recall@5":  0.70,   # primary guard metric — corruption drops to ~0.64, triggering rollback
    "mrr":       0.65,   # noisy drops to ~0.66, near-miss; corruption drops to ~0.63, triggering
    "ndcg@5":    0.68,   # noisy drops to ~0.69, near-miss; corruption drops to ~0.64, triggering
}

# --- Metrics to display in the degradation plot -------------------------------
PLOT_METRICS: list[str] = ["recall@5", "mrr", "ndcg@5"]

# --- Output paths (override config defaults for RAGGuard outputs) -------------
PG_RESULTS_CSV   = os.path.join(RESULTS_DIR, "pg_results.csv")
PG_DEGRADATION_PLOT = os.path.join(PLOTS_DIR, "degradation.png")
PG_ROLLBACK_LOG  = os.path.join(RESULTS_DIR, "rollback_log.json")

# =============================================================================
# Internal helpers
# =============================================================================

def _banner(text: str) -> None:
    width = 64
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def _section(text: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {text}")
    print("─" * 60)


# =============================================================================
# Core evaluation routine
# =============================================================================

def _get_store(cfg) -> VectorStore:
    """Dig out the VectorStore from any of the three retriever types."""
    if hasattr(cfg, "store"):
        return cfg.store
    if hasattr(cfg, "hybrid"):
        return cfg.hybrid.store
    raise TypeError(f"Cannot extract VectorStore from {type(cfg)}")


def _evaluate_config(
    cfg,
    test_set: list[dict],
    top_k: int,
    k_values: list[int],
    noisy_rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """
    Run *cfg* over *test_set* and return aggregate metrics.

    If *noisy_rng* is provided, Gaussian noise is injected into the query
    embedding by temporarily patching ``store.embed_query`` to return a
    noisy vector for that one call.  Crucially, ``cfg.retrieve()`` is still
    called normally for every configuration, so:

    - DenseRetriever   → dense search uses the noisy vector.
    - HybridRetriever  → dense half uses the noisy vector; BM25 runs on the
                         raw query string (correct: BM25 doesn't use embeddings).
    - HybridReranked   → same as hybrid for stage-1; cross-encoder rescores
                         the hybrid candidates (also correct).

    This means the three configurations genuinely diverge under noise because
    each config's own logic (BM25 blending, cross-encoder reranking) is
    actually applied.

    Returns a dict mapping metric name → mean score.
    """
    store = _get_store(cfg)
    per_query_metrics: list[dict[str, float]] = []

    for item in test_set:
        query = item["query"]
        relevant_ids = item["relevant_chunk_ids"]

        if noisy_rng is not None:
            # Compute the noisy vector once for this query.
            clean_vec = store.embed_query(query)
            noisy_vec = noisy_embeddings(clean_vec, NOISE_LEVEL, rng=noisy_rng)

            # Patch embed_query on the store so that cfg.retrieve() —
            # whichever config it is — uses the noisy vector for the dense
            # component while still running its own BM25 / reranking logic.
            original_embed_query = store.embed_query
            store.embed_query = lambda q: noisy_vec  # noqa: B023
            try:
                raw = cfg.retrieve(query, top_k=top_k)
            finally:
                store.embed_query = original_embed_query  # always restore
        else:
            raw = cfg.retrieve(query, top_k=top_k)

        retrieved_ids = [cid for cid, _ in raw]
        m = compute_query_metrics(retrieved_ids, relevant_ids, k_values)
        per_query_metrics.append(m)

    agg = aggregate_metrics(per_query_metrics)
    return {k: round(v, 4) for k, v in agg.items()}


# =============================================================================
# Rollback check helper
# =============================================================================

def _check_and_rollback(
    cfg_name: str,
    metrics: dict[str, float],
    last_good_config: str,
    last_good_metrics: dict[str, float],
    scenario: str,
) -> tuple[bool, str]:
    """
    Check all threshold metrics for *cfg_name* under *scenario*.

    If a breach is detected:
    - If *last_good_config* is itself also below threshold on the same metric,
      log an ALL_CONFIGS_BREACHED event instead of a rollback — there is no
      safe configuration to revert to.
    - If *last_good_config* is above threshold, log a true ROLLBACK event.

    Returns (rollback_triggered: bool, active_config: str).
    """
    rollback_triggered = False
    active_config = cfg_name
    ts = datetime.now(tz=timezone.utc).isoformat()

    for metric_name, threshold in ROLLBACK_THRESHOLDS.items():
        score = metrics.get(metric_name, 0.0)
        if check_threshold(metric_name, score, threshold):
            print(
                f"  [BREACH] {cfg_name} | {scenario} | "
                f"{metric_name}={score:.4f} < threshold={threshold}"
            )
            log_rollback_event(
                metric_name=metric_name,
                score=score,
                threshold=threshold,
                timestamp=ts,
                extra={
                    "configuration": cfg_name,
                    "scenario": scenario,
                },
            )

            if not rollback_triggered:
                # Check whether the fallback config is itself above threshold
                fallback_score = last_good_metrics.get(metric_name, 0.0)
                fallback_healthy = fallback_score >= threshold

                if fallback_healthy:
                    # True rollback — revert to a config that IS above threshold
                    active_config = trigger_rollback(
                        current_config=cfg_name,
                        last_good_config=last_good_config,
                        reason=f"{metric_name}={score:.4f} < {threshold}",
                        extra={"scenario": scenario},
                    )
                    rollback_triggered = True
                else:
                    # The fallback also breached — no safe config available
                    fallback_score_str = f"{fallback_score:.4f}"
                    print(
                        f"  [ALL_CONFIGS_BREACHED] No safe fallback: "
                        f"{last_good_config} {metric_name}={fallback_score_str} "
                        f"also < threshold={threshold}"
                    )
                    log_rollback_event(
                        metric_name=metric_name,
                        score=score,
                        threshold=threshold,
                        timestamp=ts,
                        extra={
                            "event_type": "ALL_CONFIGS_BREACHED",
                            "configuration": cfg_name,
                            "scenario": scenario,
                            "fallback_config": last_good_config,
                            "fallback_score": fallback_score,
                        },
                    )
                    rollback_triggered = True  # breach was handled (no revert possible)

    return rollback_triggered, active_config


# =============================================================================
# Degradation plot
# =============================================================================

def _save_degradation_plot(
    rows: list[dict],
    output_path: str,
) -> None:
    """
    Grouped bar chart comparing clean vs. noisy vs. corrupted per configuration.

    X-axis : metrics (recall@5, mrr, ndcg@5 …)
    Groups : one bar-cluster per configuration
    Colours: clean=solid, noisy=hatched, corrupted=hatched differently
    """
    cfg_names = sorted({r["configuration"] for r in rows})
    scenarios = ["clean", "noisy", "corrupted"]

    palette = {
        "dense":            "#4C72B0",
        "hybrid":           "#DD8452",
        "hybrid_reranked":  "#55A868",
    }
    hatch_map = {"clean": "", "noisy": "//", "corrupted": "xx"}
    alpha_map  = {"clean": 0.95, "noisy": 0.65, "corrupted": 0.55}

    n_metrics = len(PLOT_METRICS)
    n_series = len(cfg_names) * len(scenarios)
    bar_width = 0.8 / n_series
    x = np.arange(n_metrics)

    fig, ax = plt.subplots(figsize=(max(12, n_metrics * 2.5), 6))

    series_idx = 0
    legend_handles = []
    for cfg in cfg_names:
        color = palette.get(cfg, "#777777")
        for scenario in scenarios:
            # Find matching row
            match = [
                r for r in rows
                if r["configuration"] == cfg and r["scenario"] == scenario
            ]
            vals = [match[0].get(m, 0.0) if match else 0.0 for m in PLOT_METRICS]

            offset = (series_idx - n_series / 2 + 0.5) * bar_width
            bars = ax.bar(
                x + offset,
                vals,
                width=bar_width * 0.92,
                color=color,
                alpha=alpha_map[scenario],
                hatch=hatch_map[scenario],
                edgecolor="white" if scenario == "clean" else color,
                linewidth=0.6,
                zorder=3,
            )
            label = f"{cfg} / {scenario}"
            legend_handles.append(
                plt.Rectangle(
                    (0, 0), 1, 1,
                    color=color,
                    alpha=alpha_map[scenario],
                    hatch=hatch_map[scenario],
                    label=label,
                )
            )
            series_idx += 1

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in PLOT_METRICS], fontsize=10)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_title(
        "RAGGuard — Retrieval Quality: Clean vs. Drifted",
        fontsize=13, fontweight="bold",
    )
    ax.legend(
        handles=legend_handles,
        fontsize=7,
        loc="upper right",
        ncol=3,
        framealpha=0.85,
    )
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    # Threshold reference lines
    for metric_name, threshold in ROLLBACK_THRESHOLDS.items():
        if metric_name in PLOT_METRICS:
            xi = PLOT_METRICS.index(metric_name)
            ax.hlines(
                threshold,
                xi - 0.45,
                xi + 0.45,
                colors="red",
                linestyles="--",
                linewidth=1.2,
                zorder=4,
            )
            ax.text(
                xi + 0.46,
                threshold + 0.01,
                f"threshold={threshold}",
                color="red",
                fontsize=7,
                va="bottom",
            )

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Degradation plot saved → {output_path}")


# =============================================================================
# Main experiment
# =============================================================================

def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    _banner("RAGGuard — Self-Monitoring & Auto-Rollback for RAG")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    _section("1 / 6  Loading corpus and test set")
    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        corpus: list[dict] = json.load(f)
    with open(TEST_SET_FILE, "r", encoding="utf-8") as f:
        test_set: list[dict] = json.load(f)
    print(f"  Corpus  : {len(corpus)} chunks")
    print(f"  Test set: {len(test_set)} queries")

    # ── 2. Build vector store (clean) ─────────────────────────────────────────
    _section("2 / 6  Building clean vector store")
    clean_store = VectorStore(model_name=EMBEDDING_MODEL, cache_dir=DATA_DIR)
    clean_store.build(corpus, show_progress=True)

    # ── 3. Instantiate retrieval configurations ───────────────────────────────
    _section("3 / 6  Instantiating retrieval configurations")
    dense    = DenseRetriever(store=clean_store)
    hybrid   = HybridRetriever(store=clean_store, alpha=HYBRID_ALPHA)
    reranked = HybridRerankedRetriever(
        hybrid=hybrid,
        cross_encoder_model=CROSS_ENCODER_MODEL,
        rerank_candidates=max(TOP_K * 2, 20),
    )
    configurations = [dense, hybrid, reranked]
    print(f"  Configurations: {[c.name for c in configurations]}")

    # ── 4. Clean baseline ─────────────────────────────────────────────────────
    _section("4 / 6  Clean baseline evaluation")
    clean_results: dict[str, dict[str, float]] = {}

    # Re-use existing RetrievalBench results when available to avoid
    # re-running the expensive cross-encoder pass.
    baseline_loaded = False
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, "r", encoding="utf-8") as f:
                rb_data = json.load(f)
            for cfg_entry in rb_data.get("configurations", []):
                name = cfg_entry["configuration"]
                agg  = cfg_entry.get("aggregate", {})
                if agg:
                    clean_results[name] = {k: round(v, 4) for k, v in agg.items()}
            if clean_results:
                print(f"  Loaded cached baseline from {RESULTS_FILE}")
                print(f"  Configurations found: {list(clean_results.keys())}")
                baseline_loaded = True
        except (json.JSONDecodeError, KeyError):
            pass

    if not baseline_loaded:
        print("  No cached baseline found — running fresh evaluation…")
        for cfg in configurations:
            print(f"    Evaluating {cfg.name}…", end=" ", flush=True)
            t0 = time.perf_counter()
            agg = _evaluate_config(cfg, test_set, TOP_K, K_VALUES)
            elapsed = time.perf_counter() - t0
            clean_results[cfg.name] = agg
            print(f"done ({elapsed:.1f}s)")

    print("\n  Clean baseline metrics:")
    for cfg_name, agg in clean_results.items():
        recall5 = agg.get("recall@5", float("nan"))
        mrr_val = agg.get("mrr", float("nan"))
        ndcg5   = agg.get("ndcg@5", float("nan"))
        print(f"    {cfg_name:<22}  recall@5={recall5:.4f}  mrr={mrr_val:.4f}  ndcg@5={ndcg5:.4f}")

    # ── 5. Drift evaluation ───────────────────────────────────────────────────
    _section("5 / 6  Drift scenario evaluation + rollback checks")

    # The "last good config" is whichever clean configuration had the best recall@5
    last_good_config = max(
        clean_results,
        key=lambda n: clean_results[n].get("recall@5", 0.0),
    )
    print(f"  Last-good-config (highest clean recall@5): {last_good_config!r}")
    print(f"\n  Drift parameters:")
    print(f"    Noise level      : {NOISE_LEVEL}  (Gaussian σ on embedding)")
    print(f"    Corruption rate  : {CORRUPTION_RATE * 100:.0f}%  chunks removed")
    print(f"    Rollback thresholds: {ROLLBACK_THRESHOLDS}")

    noisy_results: dict[str, dict[str, float]] = {}
    corrupted_results: dict[str, dict[str, float]] = {}
    rollback_events: list[dict] = []  # collect summary for final table

    # ── Scenario A: Noisy embeddings ──────────────────────────────────────────
    print("\n  [Scenario A] Noisy embeddings")
    noisy_rng = np.random.default_rng(NOISE_SEED)
    for cfg in configurations:
        print(f"    {cfg.name}…", end=" ", flush=True)
        t0 = time.perf_counter()
        agg = _evaluate_config(
            cfg, test_set, TOP_K, K_VALUES,
            noisy_rng=np.random.default_rng(NOISE_SEED),  # fresh rng per config
        )
        elapsed = time.perf_counter() - t0
        noisy_results[cfg.name] = agg
        recall5 = agg.get("recall@5", float("nan"))
        mrr_val = agg.get("mrr", float("nan"))
        ndcg5   = agg.get("ndcg@5", float("nan"))
        print(
            f"recall@5={recall5:.4f}  mrr={mrr_val:.4f}  "
            f"ndcg@5={ndcg5:.4f}  ({elapsed:.1f}s)"
        )

        triggered, active = _check_and_rollback(
            cfg.name, agg, last_good_config,
            last_good_metrics=noisy_results.get(last_good_config, clean_results.get(last_good_config, {})),
            scenario="noisy"
        )
        if triggered:
            rollback_events.append({
                "scenario": "noisy",
                "configuration": cfg.name,
                "reverted_to": active,
                "true_rollback": (active != cfg.name),
            })

    # ── Scenario B: Corrupted chunks ──────────────────────────────────────────
    print("\n  [Scenario B] Corrupted chunks (remove mode)")
    corrupted_corpus = corrupt_chunks(corpus, CORRUPTION_RATE, mode="remove", seed=CORRUPTION_SEED)
    print(f"    Original corpus : {len(corpus)} chunks")
    print(f"    Corrupted corpus: {len(corrupted_corpus)} chunks "
          f"({len(corpus) - len(corrupted_corpus)} removed)")

    # Build a separate vector store on the corrupted corpus
    corrupted_store = VectorStore(model_name=EMBEDDING_MODEL, cache_dir=None)
    # Encode only the reduced corpus (no cache — it's a degraded snapshot)
    corrupted_store.build(corrupted_corpus, show_progress=False)

    # Wire new retrievers onto the corrupted store
    c_dense   = DenseRetriever(store=corrupted_store)
    c_hybrid  = HybridRetriever(store=corrupted_store, alpha=HYBRID_ALPHA)
    c_reranked = HybridRerankedRetriever(
        hybrid=c_hybrid,
        cross_encoder_model=CROSS_ENCODER_MODEL,
        rerank_candidates=max(TOP_K * 2, 20),
    )
    cfg_map_corrupted = {
        "dense":           c_dense,
        "hybrid":          c_hybrid,
        "hybrid_reranked": c_reranked,
    }

    for cfg_name, cfg in cfg_map_corrupted.items():
        print(f"    {cfg_name}…", end=" ", flush=True)
        t0 = time.perf_counter()
        agg = _evaluate_config(cfg, test_set, TOP_K, K_VALUES)
        elapsed = time.perf_counter() - t0
        corrupted_results[cfg_name] = agg
        recall5 = agg.get("recall@5", float("nan"))
        mrr_val = agg.get("mrr", float("nan"))
        ndcg5   = agg.get("ndcg@5", float("nan"))
        print(
            f"recall@5={recall5:.4f}  mrr={mrr_val:.4f}  "
            f"ndcg@5={ndcg5:.4f}  ({elapsed:.1f}s)"
        )

        triggered, active = _check_and_rollback(
            cfg_name, agg, last_good_config,
            last_good_metrics=corrupted_results.get(last_good_config, clean_results.get(last_good_config, {})),
            scenario="corrupted"
        )
        if triggered:
            rollback_events.append({
                "scenario": "corrupted",
                "configuration": cfg_name,
                "reverted_to": active,
                "true_rollback": (active != cfg_name),
            })

    # ── 6. Save results & plot ─────────────────────────────────────────────────
    _section("6 / 6  Saving results and generating degradation plot")

    # Build unified rows list
    all_rows: list[dict] = []
    for cfg_name in [c.name for c in configurations]:
        for scenario, result_dict in [
            ("clean",     clean_results),
            ("noisy",     noisy_results),
            ("corrupted", corrupted_results),
        ]:
            agg = result_dict.get(cfg_name, {})
            row = {
                "configuration": cfg_name,
                "scenario": scenario,
                "noise_level": NOISE_LEVEL if scenario == "noisy" else 0.0,
                "corruption_rate": CORRUPTION_RATE if scenario == "corrupted" else 0.0,
            }
            # Add all available metrics
            for metric_key, val in agg.items():
                row[metric_key] = round(val, 4)
            all_rows.append(row)

    df = pd.DataFrame(all_rows)

    # Add degradation columns (clean - drifted) for the primary metrics
    clean_lookup = {
        cfg_name: clean_results.get(cfg_name, {})
        for cfg_name in [c.name for c in configurations]
    }
    for metric in PLOT_METRICS:
        col_delta = f"delta_{metric.replace('@', '_at_')}"
        df[col_delta] = df.apply(
            lambda r: (
                round(
                    clean_lookup.get(r["configuration"], {}).get(metric, 0.0)
                    - r.get(metric, 0.0),
                    4,
                )
                if r["scenario"] != "clean"
                else 0.0
            ),
            axis=1,
        )

    df.to_csv(PG_RESULTS_CSV, index=False)
    print(f"  Results CSV saved → {PG_RESULTS_CSV}")

    _save_degradation_plot(all_rows, PG_DEGRADATION_PLOT)

    # ── 7. Summary table ──────────────────────────────────────────────────────
    _banner("RAGGuard — Results Summary")

    display_metrics = PLOT_METRICS
    col_w = 16

    header = (
        f"{'Configuration':<22}"
        f"{'Scenario':<14}"
        + "".join(f"{m.upper():<{col_w}}" for m in display_metrics)
        + "".join(
            f"{'Δ' + m.upper():<{col_w}}"
            for m in display_metrics
        )
    )
    sep = "─" * len(header)
    print(header)
    print(sep)

    for _, row in df.iterrows():
        cfg_label = str(row["configuration"])
        scenario  = str(row["scenario"])
        metric_vals = "".join(
            f"{row.get(m, float('nan')):>{col_w}.4f}"
            for m in display_metrics
        )
        delta_col = "".join(
            f"{row.get('delta_' + m.replace('@', '_at_'), 0.0):>{col_w}.4f}"
            if scenario != "clean" else f"{'—':>{col_w}}"
            for m in display_metrics
        )
        print(f"{cfg_label:<22}{scenario:<14}{metric_vals}{delta_col}")
        if scenario == "corrupted":
            print(sep)

    # ── 8. Rollback summary ───────────────────────────────────────────────────
    print(f"\n  Rollback events this run: {len(rollback_events)}")
    for evt in rollback_events:
        if evt.get("true_rollback"):
            tag = "ROLLBACK         "
            suffix = f"→ reverted to {evt['reverted_to']!r}"
        else:
            tag = "ALL_CONFIGS_BREACHED"
            suffix = "(no safe fallback available)"
        print(
            f"    • [{tag}] scenario={evt['scenario']:<12}  "
            f"config={evt['configuration']:<22}  {suffix}"
        )

    if os.path.exists(PG_ROLLBACK_LOG):
        with open(PG_ROLLBACK_LOG, "r", encoding="utf-8") as f:
            all_log = json.load(f)
        print(f"  Total events in rollback_log.json: {len(all_log)}")

    print(f"\n  All output files:")
    print(f"    {PG_RESULTS_CSV}")
    print(f"    {PG_DEGRADATION_PLOT}")
    print(f"    {PG_ROLLBACK_LOG}")
    print("\nDone.\n")


if __name__ == "__main__":
    main()
