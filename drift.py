"""
drift.py — RAGGuard drift simulation scenarios.

Two functions produce controlled degradation of the retrieval pipeline's
inputs so we can measure how much quality drops and whether the rollback
mechanism fires correctly.

Functions
─────────
noisy_embeddings(query_embedding, noise_level)
    Adds Gaussian noise to a query embedding before it is passed to the
    vector store, simulating garbled transcripts, heavy typos, or an
    upstream encoding glitch.

corrupt_chunks(chunk_store, corruption_rate)
    Returns a modified copy of a corpus chunk list with a given fraction
    of chunks either removed or shuffled, simulating partial data-loss or
    an ingestion failure in the knowledge base.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from retrievalbench.store import VectorStore


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 1 — Noisy query embeddings
# ─────────────────────────────────────────────────────────────────────────────

def noisy_embeddings(
    query_embedding: np.ndarray,
    noise_level: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Add controlled Gaussian noise to a query embedding.

    The noise is drawn from N(0, noise_level²) and added element-wise to
    the embedding.  The result is NOT re-normalised so that the magnitude
    distortion is also present — this produces a more realistic degradation
    than pure directional noise.

    Parameters
    ----------
    query_embedding : np.ndarray, shape (dim,)
        The original L2-normalised query embedding from the bi-encoder.
    noise_level : float
        Standard deviation of the Gaussian noise.  Typical values:
          0.05  — mild disturbance, barely perceptible
          0.20  — moderate noise, noticeable quality drop
          0.50  — heavy noise, most signal destroyed
    rng : np.random.Generator | None
        Optional random generator for reproducibility.  If None, a default
        numpy generator is used (non-deterministic).

    Returns
    -------
    np.ndarray, shape (dim,)
        Noisy (unnormalised) query embedding.
    """
    if rng is None:
        rng = np.random.default_rng()

    noise = rng.normal(loc=0.0, scale=noise_level, size=query_embedding.shape)
    return query_embedding + noise.astype(query_embedding.dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 2 — Corrupted document chunks
# ─────────────────────────────────────────────────────────────────────────────

def corrupt_chunks(
    chunks: list[dict],
    corruption_rate: float,
    mode: str = "remove",
    seed: int | None = None,
) -> list[dict]:
    """
    Return a degraded copy of the corpus chunk list.

    Two corruption modes are available:

    "remove"  — The given fraction of chunks is deleted entirely,
                simulating a partial data-loss or ingestion failure.
                The remaining chunks are returned in their original order.

    "shuffle" — The given fraction of chunks has its ``text`` field replaced
                with the text from a randomly selected *other* chunk,
                simulating misaligned or scrambled ingestion (e.g. pages
                written to the wrong record).  Chunk IDs are preserved so
                that retrieval still returns IDs, but the content no longer
                matches — causing silent quality degradation.

    Parameters
    ----------
    chunks : list[dict]
        Original corpus, each dict having at least ``chunk_id`` and ``text``.
    corruption_rate : float
        Fraction of chunks to corrupt, in [0, 1].  0.10 means 10 %.
    mode : str
        "remove" (default) or "shuffle".
    seed : int | None
        Optional random seed for reproducibility.

    Returns
    -------
    list[dict]
        A new list (deep-ish copy) with the requested corruption applied.
        The original list is never modified.

    Raises
    ------
    ValueError
        If corruption_rate is outside [0, 1] or mode is unrecognised.
    """
    if not 0.0 <= corruption_rate <= 1.0:
        raise ValueError(f"corruption_rate must be in [0, 1], got {corruption_rate}")
    if mode not in ("remove", "shuffle"):
        raise ValueError(f"mode must be 'remove' or 'shuffle', got {mode!r}")

    rng = random.Random(seed)
    n = len(chunks)
    n_corrupt = int(round(n * corruption_rate))

    if n_corrupt == 0:
        return list(chunks)  # nothing to do

    # Indices to corrupt, chosen without replacement
    corrupt_indices = set(rng.sample(range(n), n_corrupt))

    if mode == "remove":
        return [chunk for i, chunk in enumerate(chunks) if i not in corrupt_indices]

    # mode == "shuffle": replace text of targeted chunks with a random other
    # chunk's text, keeping the chunk_id intact (the mismatch is the corruption)
    result = [dict(chunk) for chunk in chunks]  # shallow copy of each dict
    all_indices = list(range(n))
    for i in corrupt_indices:
        # Pick a replacement index that is different from i
        candidates = [j for j in all_indices if j != i]
        j = rng.choice(candidates)
        result[i] = dict(chunks[i])          # ensure we have our own dict
        result[i]["text"] = chunks[j]["text"] # corrupt the text
    return result
