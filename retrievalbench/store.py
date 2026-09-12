"""
retrievalbench/store.py — In-memory vector store backed by sentence-transformers.

Provides:
    VectorStore   — embeds corpus chunks and answers cosine-similarity queries.

Design notes
────────────
• Uses sentence-transformers bi-encoder for all embedding work.
• All embeddings are L2-normalised so cosine similarity == dot product,
  allowing a simple matrix multiply for batch retrieval.
• No pgvector dependency is required; the in-memory store is intentionally
  portable so the benchmark can run anywhere without a database server.
  The interface is the same one you would wire to pgvector in production:
  embed a query → retrieve top-k (chunk_id, score) pairs.
• The store serialises embeddings to disk (data/embeddings_cache.npy +
  data/embeddings_ids.json) so the model is only called once per corpus.
"""

from __future__ import annotations

import json
import os
import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

# Lazy import so the module loads even if sentence-transformers is absent
# (unit tests mock the model).
_model_cache: dict = {}


def _get_model(model_name: str):
    """Return a cached SentenceTransformer instance."""
    if model_name not in _model_cache:
        from sentence_transformers import SentenceTransformer  # type: ignore
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


class VectorStore:
    """
    Lightweight in-memory vector store.

    Parameters
    ----------
    model_name : str
        HuggingFace model id passed to SentenceTransformer.
    cache_dir : str | None
        Directory to cache computed embeddings.  If None, caching is disabled.
    """

    def __init__(self, model_name: str, cache_dir: str | None = None) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._chunk_ids: list[str] = []
        self._texts: list[str] = []
        self._embeddings: np.ndarray | None = None  # shape (n_chunks, dim)

    # ── Building ──────────────────────────────────────────────────────────────

    def build(self, chunks: list[dict], show_progress: bool = True) -> None:
        """
        Embed all corpus chunks and store them.

        Parameters
        ----------
        chunks : list of dicts with at least ``chunk_id`` and ``text`` keys.
        show_progress : bool
            Show a tqdm progress bar during encoding.
        """
        self._chunk_ids = [c["chunk_id"] for c in chunks]
        self._texts = [c["text"] for c in chunks]

        cache_emb_path = (
            os.path.join(self.cache_dir, "embeddings_cache.npy")
            if self.cache_dir
            else None
        )
        cache_ids_path = (
            os.path.join(self.cache_dir, "embeddings_ids.json")
            if self.cache_dir
            else None
        )

        # Load from cache if it exists and IDs match
        if cache_emb_path and os.path.exists(cache_emb_path) and os.path.exists(cache_ids_path):
            with open(cache_ids_path, "r", encoding="utf-8") as f:
                cached_ids = json.load(f)
            if cached_ids == self._chunk_ids:
                self._embeddings = np.load(cache_emb_path)
                print(f"  [VectorStore] Loaded cached embeddings ({self._embeddings.shape})")
                return

        model = _get_model(self.model_name)
        print(f"  [VectorStore] Encoding {len(self._texts)} chunks with '{self.model_name}'…")
        t0 = time.perf_counter()
        emb = model.encode(
            self._texts,
            batch_size=32,
            show_progress_bar=show_progress,
            normalize_embeddings=True,   # L2-norm → cosine = dot product
            convert_to_numpy=True,
        )
        elapsed = time.perf_counter() - t0
        print(f"  [VectorStore] Encoding done in {elapsed:.2f}s  shape={emb.shape}")
        self._embeddings = emb

        # Persist cache
        if cache_emb_path and self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            np.save(cache_emb_path, emb)
            with open(cache_ids_path, "w", encoding="utf-8") as f:
                json.dump(self._chunk_ids, f)
            print(f"  [VectorStore] Embeddings cached to {cache_emb_path}")

    # ── Querying ──────────────────────────────────────────────────────────────

    def embed_query(self, query: str) -> np.ndarray:
        """Return an L2-normalised 1-D embedding for a single query string."""
        model = _get_model(self.model_name)
        vec = model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vec[0]  # shape (dim,)

    def search(self, query_vec: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        """
        Return the top-k most similar chunks by cosine similarity.

        Parameters
        ----------
        query_vec : np.ndarray, shape (dim,)
            L2-normalised query embedding.
        top_k : int

        Returns
        -------
        list of (chunk_id, score) sorted descending by score.
        """
        if self._embeddings is None:
            raise RuntimeError("VectorStore.build() must be called before search().")

        # (n_chunks,) — dot product == cosine because both sides are normalised
        scores = self._embeddings @ query_vec

        # Partial sort for efficiency: top_k indices descending
        if top_k >= len(scores):
            sorted_idx = np.argsort(scores)[::-1]
        else:
            # argpartition is O(n), then sort only top_k
            part = np.argpartition(scores, -top_k)[-top_k:]
            sorted_idx = part[np.argsort(scores[part])[::-1]]

        return [(self._chunk_ids[i], float(scores[i])) for i in sorted_idx]

    # ── Accessors ─────────────────────────────────────────────────────────────

    @property
    def chunk_ids(self) -> list[str]:
        return list(self._chunk_ids)

    @property
    def texts(self) -> list[str]:
        return list(self._texts)

    def text_for(self, chunk_id: str) -> str:
        """Return the raw text for a given chunk_id."""
        idx = self._chunk_ids.index(chunk_id)
        return self._texts[idx]
