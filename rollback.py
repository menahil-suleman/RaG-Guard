"""
rollback.py — RAGGuard threshold-based rollback mechanism.

Three public functions implement the self-monitoring logic:

check_threshold(metric_name, score, threshold) -> bool
    Returns True when the score is *below* the threshold (i.e. a breach
    has been detected and rollback should be triggered).

log_rollback_event(metric_name, score, threshold, timestamp)
    Appends a structured record to results/rollback_log.json so that every
    rollback event is permanently auditable.

trigger_rollback(current_config, last_good_config) -> str
    Performs the revert, logs the decision, and returns the name of the
    configuration that was restored.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

# ─────────────────────────────────────────────────────────────────────────────
# Path for the persistent rollback event log
# ─────────────────────────────────────────────────────────────────────────────

_ROOT = os.path.dirname(os.path.abspath(__file__))
ROLLBACK_LOG_PATH = os.path.join(_ROOT, "results", "rollback_log.json")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Threshold check
# ─────────────────────────────────────────────────────────────────────────────

def check_threshold(
    metric_name: str,
    score: float,
    threshold: float,
) -> bool:
    """
    Return True if *score* is strictly below *threshold* (breach detected).

    Parameters
    ----------
    metric_name : str
        Human-readable metric name used only for logging; not evaluated here.
    score : float
        The measured metric value for the current run.
    threshold : float
        The minimum acceptable value.

    Returns
    -------
    bool
        True  → score < threshold  → rollback should be triggered.
        False → score ≥ threshold  → pipeline is healthy.
    """
    return score < threshold


# ─────────────────────────────────────────────────────────────────────────────
# 2. Event logging
# ─────────────────────────────────────────────────────────────────────────────

def log_rollback_event(
    metric_name: str,
    score: float,
    threshold: float,
    timestamp: str | None = None,
    extra: dict | None = None,
) -> dict:
    """
    Append a rollback event record to the persistent JSON log file.

    The log file is created if it does not exist.  All events accumulate as
    a JSON array so the full history is always available.

    Parameters
    ----------
    metric_name : str
        The metric that triggered the rollback (e.g. "recall@5").
    score : float
        The measured score that breached the threshold.
    threshold : float
        The threshold value that was not met.
    timestamp : str | None
        ISO-8601 timestamp string.  Defaults to *now* (UTC) if None.
    extra : dict | None
        Optional additional metadata (configuration name, drift scenario,
        noise level, etc.) to store alongside the standard fields.

    Returns
    -------
    dict
        The event record that was written (useful for inline inspection).
    """
    if timestamp is None:
        timestamp = datetime.now(tz=timezone.utc).isoformat()

    event: dict = {
        "timestamp": timestamp,
        "metric_name": metric_name,
        "score": round(float(score), 6),
        "threshold": round(float(threshold), 6),
        "breach_delta": round(float(threshold - score), 6),
    }
    if extra:
        event.update(extra)

    # Load existing log or start fresh
    os.makedirs(os.path.dirname(ROLLBACK_LOG_PATH), exist_ok=True)
    if os.path.exists(ROLLBACK_LOG_PATH):
        with open(ROLLBACK_LOG_PATH, "r", encoding="utf-8") as f:
            try:
                log: list = json.load(f)
            except json.JSONDecodeError:
                log = []
    else:
        log = []

    log.append(event)

    with open(ROLLBACK_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    return event


# ─────────────────────────────────────────────────────────────────────────────
# 3. Rollback trigger
# ─────────────────────────────────────────────────────────────────────────────

def trigger_rollback(
    current_config: str,
    last_good_config: str,
    reason: str = "",
    extra: dict | None = None,
) -> str:
    """
    Revert to *last_good_config* and record the decision in the rollback log.

    In this in-process implementation the "revert" is represented as
    returning the name of the configuration that should be used going
    forward.  The caller is responsible for substituting the actual
    retriever object; this function handles logging and reporting only.

    Parameters
    ----------
    current_config : str
        The configuration that triggered the rollback (the failing one).
    last_good_config : str
        The configuration to revert to.
    reason : str
        Free-text reason string recorded in the log.
    extra : dict | None
        Additional metadata passed through to log_rollback_event().

    Returns
    -------
    str
        The name of the restored (last good) configuration.
    """
    timestamp = datetime.now(tz=timezone.utc).isoformat()

    log_entry: dict = {
        "event_type": "rollback",
        "reverted_from": current_config,
        "reverted_to": last_good_config,
        "reason": reason,
    }
    if extra:
        log_entry.update(extra)

    # Reuse log_rollback_event with a synthetic metric name to keep the
    # log format consistent; score/threshold are 0/0 as placeholders when
    # called via trigger_rollback (the threshold breach is already logged
    # separately by the caller with log_rollback_event).
    log_rollback_event(
        metric_name="[rollback_trigger]",
        score=0.0,
        threshold=0.0,
        timestamp=timestamp,
        extra=log_entry,
    )

    print(
        f"  [ROLLBACK] {current_config!r} → {last_good_config!r}  "
        f"({reason or 'threshold breach'})"
    )
    return last_good_config
