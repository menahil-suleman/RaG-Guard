# RAGGuard — Short Description

## One-liner (GitHub repo description)
Self-monitoring and auto-rollback layer for RAG retrieval pipelines — injects data drift, measures quality degradation, and reverts to a known-good configuration when metrics breach threshold.

---

## Short paragraph (portfolio / project page)

RAGGuard is a robustness-testing and self-monitoring system for RAG retrieval pipelines, built on top of a prior retrieval benchmarking project. It simulates two realistic failure modes — noisy query embeddings and corrupted document chunks — and measures how much retrieval quality (Recall@k, MRR, nDCG@k) degrades across three retrieval configurations: dense-only, hybrid dense+BM25, and hybrid with cross-encoder reranking. When any metric falls below a configurable threshold, the system automatically reverts to the last known-good configuration and logs the event with a full audit trail. When no safe configuration exists, it logs an `ALL_CONFIGS_BREACHED` event rather than falsely reporting a successful recovery. The project surfaces a finding that a clean-condition benchmark alone would miss: the configuration most robust to input noise is not the same as the one with the highest raw performance.

---

## CV line

Built RAGGuard, a self-monitoring and auto-rollback layer for RAG retrieval pipelines: injected controlled data drift (noisy embeddings, corrupted document chunks), measured degradation in Recall@k, MRR, and nDCG@k across three retrieval configurations (dense, hybrid BM25+dense, hybrid+cross-encoder reranking), and implemented a threshold-based rollback mechanism that reverts to a known-good configuration with full JSON event logging.

---

## Technical summary (for a research poster or abstract)

We present RAGGuard, a lightweight monitoring layer for retrieval-augmented generation pipelines that evaluates retrieval robustness under two controlled drift scenarios: Gaussian noise applied to query embeddings and random removal of document chunks from the knowledge base. Three retrieval configurations — dense vector search, hybrid dense+BM25, and hybrid with cross-encoder reranking — are evaluated under each scenario using Recall@k, MRR, and nDCG@k. An automatic rollback mechanism checks post-drift scores against configurable thresholds and reverts to a previously validated configuration when a breach is detected, logging every decision to a persistent audit trail. Results show that BM25 partially shields hybrid retrieval from embedding noise, while the cross-encoder provides near-complete immunity at moderate noise levels due to its independence from query embeddings at reranking time. Chunk corruption degrades all three configurations equally, correctly triggering an ALL_CONFIGS_BREACHED state rather than a false recovery.
