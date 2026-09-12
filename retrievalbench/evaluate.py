"""
retrievalbench/evaluate.py — Main experiment runner.

Public API
──────────
    evaluate(configurations, test_set, top_k, k_values) -> EvaluationResults

The function runs every retriever against every query in the test set,
measures latency, computes all IR metrics, and returns a structured results
object that can be inspected, serialised, or passed to the visualisation layer.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict

from .metrics import compute_query_metrics, aggregate_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QueryResult:
    """Outcome of running one retriever on one query."""
    query_id: str
    query: str
    configuration: str
    retrieved_ids: list[str]
    relevant_ids: list[str]
    metrics: dict[str, float]
    latency_ms: float           # wall-clock time for the retrieve() call


@dataclass
class ConfigurationResult:
    """Aggregate results for one retriever across the full test set."""
    configuration: str
    per_query: list[QueryResult] = field(default_factory=list)
    aggregate: dict[str, float] = field(default_factory=dict)
    mean_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0


@dataclass
class EvaluationResults:
    """Top-level container returned by evaluate()."""
    configurations: list[ConfigurationResult] = field(default_factory=list)

    # ── Convenience accessors ─────────────────────────────────────────────────

    def as_dict(self) -> dict:
        """Serialise to a plain dict (JSON-safe)."""
        return asdict(self)

    def summary_table(self) -> list[dict]:
        """
        Return one dict per configuration with aggregate metrics + latency.
        Suitable for pandas.DataFrame or printing.
        """
        rows = []
        for cfg in self.configurations:
            row = {"configuration": cfg.configuration}
            row.update(cfg.aggregate)
            row["mean_latency_ms"] = round(cfg.mean_latency_ms, 2)
            row["p50_latency_ms"] = round(cfg.p50_latency_ms, 2)
            row["p95_latency_ms"] = round(cfg.p95_latency_ms, 2)
            rows.append(row)
        return rows

    def get(self, configuration_name: str) -> ConfigurationResult | None:
        """Look up a ConfigurationResult by name."""
        for cfg in self.configurations:
            if cfg.configuration == configuration_name:
                return cfg
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation function
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    configurations: list,
    test_set: list[dict],
    top_k: int = 10,
    k_values: list[int] | None = None,
    verbose: bool = True,
) -> EvaluationResults:
    """
    Run all configurations against the full test set and collect metrics.

    Parameters
    ----------
    configurations : list
        A list of retriever objects (DenseRetriever, HybridRetriever,
        HybridRerankedRetriever or any object with a ``.name`` attribute and a
        ``retrieve(query, top_k) -> list[(chunk_id, score)]`` method).
    test_set : list[dict]
        Each dict must have keys: ``query_id``, ``query``,
        ``relevant_chunk_ids``.
    top_k : int
        Number of results to request from each retriever.
    k_values : list[int] | None
        Cut-off depths for Recall@k and nDCG@k.  Defaults to [1, 3, 5, 10].
    verbose : bool
        Print progress to stdout.

    Returns
    -------
    EvaluationResults
    """
    import numpy as np

    if k_values is None:
        k_values = [1, 3, 5, 10]

    results = EvaluationResults()

    for cfg in configurations:
        cfg_name = cfg.name
        if verbose:
            print(f"\n{'─'*60}")
            print(f"  Evaluating: {cfg_name}  ({len(test_set)} queries, top_k={top_k})")
            print(f"{'─'*60}")

        per_query: list[QueryResult] = []
        latencies: list[float] = []

        for i, item in enumerate(test_set):
            query_id = item["query_id"]
            query = item["query"]
            relevant_ids = item["relevant_chunk_ids"]

            # ── Timed retrieval call ──────────────────────────────────────────
            t0 = time.perf_counter()
            ranked = cfg.retrieve(query, top_k=top_k)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(latency_ms)

            retrieved_ids = [cid for cid, _ in ranked]

            # ── Metrics for this query ────────────────────────────────────────
            m = compute_query_metrics(retrieved_ids, relevant_ids, k_values)

            qr = QueryResult(
                query_id=query_id,
                query=query,
                configuration=cfg_name,
                retrieved_ids=retrieved_ids,
                relevant_ids=relevant_ids,
                metrics=m,
                latency_ms=round(latency_ms, 3),
            )
            per_query.append(qr)

            if verbose and (i + 1) % 10 == 0:
                print(f"    {i+1}/{len(test_set)} queries done "
                      f"(last latency: {latency_ms:.1f} ms)")

        # ── Aggregate ─────────────────────────────────────────────────────────
        agg = aggregate_metrics([qr.metrics for qr in per_query])
        # Round for readability
        agg = {k: round(v, 4) for k, v in agg.items()}

        lat_arr = np.array(latencies)
        cfg_result = ConfigurationResult(
            configuration=cfg_name,
            per_query=per_query,
            aggregate=agg,
            mean_latency_ms=round(float(lat_arr.mean()), 2),
            p50_latency_ms=round(float(np.percentile(lat_arr, 50)), 2),
            p95_latency_ms=round(float(np.percentile(lat_arr, 95)), 2),
        )
        results.configurations.append(cfg_result)

        if verbose:
            print(f"\n  Results for {cfg_name}:")
            for metric, val in sorted(agg.items()):
                print(f"    {metric:<20} {val:.4f}")
            print(f"  Latency → mean={cfg_result.mean_latency_ms:.1f}ms  "
                  f"p50={cfg_result.p50_latency_ms:.1f}ms  "
                  f"p95={cfg_result.p95_latency_ms:.1f}ms")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_results(results: EvaluationResults, json_path: str, csv_path: str) -> None:
    """Persist the full results dict as JSON and a summary CSV."""
    import pandas as pd

    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    # Full JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results.as_dict(), f, indent=2, ensure_ascii=False)
    print(f"Full results saved → {json_path}")

    # Summary CSV
    df = pd.DataFrame(results.summary_table())
    df.to_csv(csv_path, index=False)
    print(f"Summary CSV saved  → {csv_path}")


def load_results_json(json_path: str) -> dict:
    """Load a previously saved results JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)
