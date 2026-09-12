"""
retrievalbench/visualise.py — Plots and printed tables.

Functions
─────────
print_summary_table     Pretty-print the aggregate results table.
plot_metrics_bar        Grouped bar chart: metric × configuration.
plot_latency_tradeoff   Scatter/line plot: accuracy vs latency.
save_all_plots          Convenience wrapper that saves both plots to disk.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from .evaluate import EvaluationResults

# ── Consistent colour palette per configuration ───────────────────────────────
_PALETTE = {
    "dense":            "#4C72B0",   # blue
    "hybrid":           "#DD8452",   # orange
    "hybrid_reranked":  "#55A868",   # green
}
_DEFAULT_COLOR = "#777777"


def _color(cfg_name: str) -> str:
    return _PALETTE.get(cfg_name, _DEFAULT_COLOR)


def _cfg_label(cfg_name: str) -> str:
    return {
        "dense":           "Dense-only",
        "hybrid":          "Hybrid",
        "hybrid_reranked": "Hybrid + Reranked",
    }.get(cfg_name, cfg_name)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Console summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_summary_table(results: EvaluationResults, k_values: list[int] | None = None) -> None:
    """Print a formatted ASCII table of aggregate metrics."""
    if k_values is None:
        k_values = [1, 3, 5, 10]

    # Build column headers
    metric_cols = (
        [f"recall@{k}" for k in k_values]
        + ["mrr"]
        + [f"ndcg@{k}"   for k in k_values]
    )
    header_cols = ["Configuration"] + metric_cols + ["mean_latency_ms"]
    col_w = 18

    sep = "─" * (col_w * len(header_cols))
    print(f"\n{'RetrievalBench — Results Summary':^{len(sep)}}")
    print(sep)
    print("".join(h.ljust(col_w) for h in header_cols))
    print(sep)

    for cfg in results.configurations:
        row_vals = [_cfg_label(cfg.configuration)]
        for col in metric_cols:
            val = cfg.aggregate.get(col, float("nan"))
            row_vals.append(f"{val:.4f}")
        row_vals.append(f"{cfg.mean_latency_ms:.1f} ms")
        print("".join(v.ljust(col_w) for v in row_vals))

    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Grouped bar chart — metrics by configuration
# ─────────────────────────────────────────────────────────────────────────────

def plot_metrics_bar(
    results: EvaluationResults,
    metrics_to_plot: list[str] | None = None,
    output_path: str | None = None,
    figsize: tuple = (14, 5),
) -> plt.Figure:
    """
    Grouped bar chart showing Recall@k, MRR, and nDCG@k
    for all three configurations side by side.

    Parameters
    ----------
    results : EvaluationResults
    metrics_to_plot : list[str] | None
        Subset of metric names.  Defaults to a sensible selection.
    output_path : str | None
        If given, save the figure to this path.
    figsize : tuple
    """
    if metrics_to_plot is None:
        metrics_to_plot = ["recall@1", "recall@5", "recall@10",
                           "mrr",
                           "ndcg@1",  "ndcg@5",  "ndcg@10"]

    cfgs = results.configurations
    n_cfgs = len(cfgs)
    n_metrics = len(metrics_to_plot)

    x = np.arange(n_metrics)
    bar_width = 0.8 / n_cfgs
    offsets = np.linspace(-(n_cfgs - 1) / 2, (n_cfgs - 1) / 2, n_cfgs) * bar_width

    fig, ax = plt.subplots(figsize=figsize)

    for cfg, offset in zip(cfgs, offsets):
        vals = [cfg.aggregate.get(m, 0.0) for m in metrics_to_plot]
        bars = ax.bar(
            x + offset,
            vals,
            width=bar_width * 0.9,
            color=_color(cfg.configuration),
            label=_cfg_label(cfg.configuration),
            zorder=3,
        )
        # Value labels on top of bars
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{val:.3f}",
                ha="center", va="bottom",
                fontsize=7, color="#333333",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [m.upper().replace("_", "\n") for m in metrics_to_plot],
        fontsize=9,
    )
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_title("Retrieval Quality — All Configurations", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Metrics bar chart saved → {output_path}")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. Accuracy vs latency tradeoff scatter plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_latency_tradeoff(
    results: EvaluationResults,
    accuracy_metric: str = "recall@10",
    output_path: str | None = None,
    figsize: tuple = (7, 5),
) -> plt.Figure:
    """
    Scatter plot placing each configuration at (mean_latency_ms, accuracy).
    A dashed line connects them to show the tradeoff curve.

    Parameters
    ----------
    results : EvaluationResults
    accuracy_metric : str
        Which metric to use on the y-axis (default: recall@10).
    output_path : str | None
        If given, save the figure here.
    figsize : tuple
    """
    cfgs = results.configurations

    latencies   = [c.mean_latency_ms for c in cfgs]
    accuracies  = [c.aggregate.get(accuracy_metric, 0.0) for c in cfgs]
    labels      = [_cfg_label(c.configuration) for c in cfgs]
    colors      = [_color(c.configuration) for c in cfgs]

    fig, ax = plt.subplots(figsize=figsize)

    # Dashed line connecting the points (sorted by latency)
    order = sorted(range(len(latencies)), key=lambda i: latencies[i])
    ax.plot(
        [latencies[i] for i in order],
        [accuracies[i] for i in order],
        linestyle="--", color="#aaaaaa", linewidth=1.2, zorder=1,
    )

    # Scatter points
    for lat, acc, label, color in zip(latencies, accuracies, labels, colors):
        ax.scatter(lat, acc, s=180, color=color, zorder=3, edgecolors="white", linewidths=1.5)
        ax.annotate(
            label,
            (lat, acc),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=9,
            color=color,
        )

    ax.set_xlabel("Mean Latency per Query (ms)", fontsize=11)
    ax.set_ylabel(accuracy_metric.upper(), fontsize=11)
    ax.set_title("Accuracy vs. Latency Tradeoff", fontsize=13, fontweight="bold")

    # Add padding so labels aren't clipped
    x_margin = max(latencies) * 0.25 if latencies else 10
    y_margin = 0.05
    ax.set_xlim(0, max(latencies) + x_margin)
    ax.set_ylim(min(accuracies) - y_margin, min(1.0, max(accuracies) + y_margin))

    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Latency tradeoff plot saved → {output_path}")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. Latency distribution box plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_latency_distribution(
    results: EvaluationResults,
    output_path: str | None = None,
    figsize: tuple = (7, 5),
) -> plt.Figure:
    """
    Box plot of per-query latency distributions for each configuration.
    Shows spread and outliers, complementing the mean/p95 summary numbers.
    """
    cfgs = results.configurations
    data    = [[qr.latency_ms for qr in cfg.per_query] for cfg in cfgs]
    labels  = [_cfg_label(cfg.configuration) for cfg in cfgs]
    colors  = [_color(cfg.configuration) for cfg in cfgs]

    fig, ax = plt.subplots(figsize=figsize)

    bp = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        medianprops=dict(color="white", linewidth=2),
        flierprops=dict(marker="o", markersize=3, alpha=0.5),
        zorder=2,
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_ylabel("Latency per Query (ms)", fontsize=11)
    ax.set_title("Per-Query Latency Distribution", fontsize=13, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Latency distribution plot saved → {output_path}")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5. Convenience wrapper
# ─────────────────────────────────────────────────────────────────────────────

def save_all_plots(results: EvaluationResults, plots_dir: str) -> None:
    """Generate and save all three standard plots."""
    plot_metrics_bar(
        results,
        output_path=os.path.join(plots_dir, "metrics_bar.png"),
    )
    plot_latency_tradeoff(
        results,
        output_path=os.path.join(plots_dir, "latency_tradeoff.png"),
    )
    plot_latency_distribution(
        results,
        output_path=os.path.join(plots_dir, "latency_distribution.png"),
    )
