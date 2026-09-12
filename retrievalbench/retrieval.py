"""
retrievalbench/retrieval.py — The three retrieval configurations.

Each configuration exposes a single callable with the signature:

    retrieve(query: str, top_k: int) -> list[tuple[str, float]]

returning (chunk_id, score) pairs sorted descending.

Configurations
──────────────
DenseRetriever          Pure bi-encoder cosine similarity (VectorStore).
HybridRetriever         Linear interpolation of dense score and BM25 score.
HybridRerankedRetriever Hybrid candidates re-scored by a cross-encoder.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .store import VectorStore


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _min_max_normalise(scores: list[float]) -> list[float]:
    """Normalise a list of scores to [0, 1]."""
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [0.5] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


# ─────────────────────────────────────────────────────────────────────────────
# Configuration 1 — Dense-only
# ─────────────────────────────────────────────────────────────────────────────

class DenseRetriever:
    """Pure semantic vector search using the bi-encoder VectorStore."""

    name = "dense"

    def __init__(self, store: "VectorStore") -> None:
        self.store = store

    def retrieve(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """
        Embed the query and return the top-k chunks ranked by cosine similarity.

        Returns
        -------
        list of (chunk_id, score) sorted descending.
        """
        query_vec = self.store.embed_query(query)
        return self.store.search(query_vec, top_k)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration 2 — Hybrid (Dense + BM25)
# ─────────────────────────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Linear interpolation of dense and BM25 scores.

    Both score lists are min-max normalised to [0, 1] before blending
    so that differences in scale do not favour one modality.

    final_score = alpha * dense_score_norm + (1 - alpha) * bm25_score_norm
    """

    name = "hybrid"

    def __init__(
        self,
        store: "VectorStore",
        alpha: float = 0.5,
    ) -> None:
        self.store = store
        self.alpha = alpha
        self._bm25 = None  # built lazily

    def _get_bm25(self):
        """Build BM25 index on first use."""
        if self._bm25 is None:
            from rank_bm25 import BM25Okapi  # type: ignore
            tokenised = [text.lower().split() for text in self.store.texts]
            self._bm25 = BM25Okapi(tokenised)
        return self._bm25

    def retrieve(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """
        Combine dense similarity and BM25 scores and return the top-k chunks.

        Strategy
        --------
        1. Get dense scores for ALL corpus chunks (full cosine similarity pass).
        2. Get BM25 scores for ALL corpus chunks.
        3. Min-max normalise both score arrays independently.
        4. Blend: final = alpha * dense_norm + (1-alpha) * bm25_norm.
        5. Return top-k by descending final score.
        """
        chunk_ids = self.store.chunk_ids
        n = len(chunk_ids)

        # Dense scores — full pass
        query_vec = self.store.embed_query(query)
        dense_pairs = self.store.search(query_vec, top_k=n)
        dense_map = {cid: score for cid, score in dense_pairs}

        # BM25 scores — full corpus
        bm25 = self._get_bm25()
        tokens = query.lower().split()
        bm25_raw = bm25.get_scores(tokens)  # ndarray length n

        # Assemble aligned arrays
        dense_scores = np.array([dense_map[cid] for cid in chunk_ids])
        bm25_scores = bm25_raw  # already aligned with store.chunk_ids

        # Normalise
        dense_norm = _min_max_normalise(dense_scores.tolist())
        bm25_norm = _min_max_normalise(bm25_scores.tolist())

        # Blend
        final_scores = [
            self.alpha * d + (1.0 - self.alpha) * b
            for d, b in zip(dense_norm, bm25_norm)
        ]

        # Sort and return top-k
        ranked = sorted(
            zip(chunk_ids, final_scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# Configuration 3 — Hybrid + Cross-encoder Reranking
# ─────────────────────────────────────────────────────────────────────────────

class HybridRerankedRetriever:
    """
    Two-stage retrieval: hybrid candidate generation followed by
    cross-encoder reranking.

    Stage 1: Hybrid retrieval fetches ``rerank_candidates`` candidates.
    Stage 2: A cross-encoder model scores each (query, chunk) pair jointly,
             then returns the top_k from those candidates by cross-encoder score.

    The cross-encoder score is more accurate than the bi-encoder similarity
    because it can attend to both the query and passage simultaneously, but is
    too slow to apply to the full corpus — hence the two-stage design.
    """

    name = "hybrid_reranked"

    def __init__(
        self,
        hybrid: HybridRetriever,
        cross_encoder_model: str,
        rerank_candidates: int = 20,
    ) -> None:
        self.hybrid = hybrid
        self.cross_encoder_model = cross_encoder_model
        self.rerank_candidates = rerank_candidates
        self._cross_encoder = None

    def _get_cross_encoder(self):
        if self._cross_encoder is None:
            from sentence_transformers.cross_encoder import CrossEncoder  # type: ignore
            self._cross_encoder = CrossEncoder(self.cross_encoder_model)
        return self._cross_encoder

    def retrieve(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """
        Stage 1: Hybrid retrieves ``rerank_candidates`` chunks.
        Stage 2: Cross-encoder rescores them; top_k returned by CE score.

        Returns
        -------
        list of (chunk_id, cross_encoder_score) sorted descending.
        """
        # Stage 1 — fetch more candidates than we need
        n_candidates = max(top_k, self.rerank_candidates)
        candidates = self.hybrid.retrieve(query, top_k=n_candidates)

        # Stage 2 — cross-encoder scoring
        ce = self._get_cross_encoder()
        candidate_ids = [cid for cid, _ in candidates]
        texts = [self.hybrid.store.text_for(cid) for cid in candidate_ids]
        pairs = [[query, text] for text in texts]

        ce_scores = ce.predict(pairs, show_progress_bar=False)
        # Sigmoid to map logits to (0, 1) — not strictly required but keeps
        # scores interpretable alongside the hybrid scores.
        ce_scores_sig = [1.0 / (1.0 + math.exp(-float(s))) for s in ce_scores]

        # Sort by cross-encoder score descending
        ranked = sorted(
            zip(candidate_ids, ce_scores_sig),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]
