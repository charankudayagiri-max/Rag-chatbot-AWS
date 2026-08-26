"""
retriever.py — Advanced Retrieval Pipeline

CONCEPTS DEMONSTRATED
─────────────────────
  What is a Retriever, Similarity Search, Top-K, Score Threshold,
         Dense Retrieval, Sparse Retrieval, BM25, Keyword Search,
         Hybrid Search, MMR (Maximum Marginal Relevance),
         Metadata Filtering, Retriever Pipeline, Precision, Recall, Latency.

Pipeline stage 7:
    Vector Store → **Retriever** → Context Engine → Prompt Builder → LLM

RETRIEVAL PIPELINE
──────────────────
  1. Dense search  — vector similarity (cosine) via ChromaDB.
  2. Sparse search — BM25 keyword matching.
  3. Hybrid merge  — combine dense + sparse scores with weights.
  4. Threshold     — drop results below score threshold.
  5. MMR           — re-rank for diversity while maintaining relevance.
"""

from __future__ import annotations

import time
import numpy as np
from rank_bm25 import BM25Okapi

from app.ai.embeddings import generate_query_embedding, cosine_similarity
from app.ai.vector_store import query_documents, get_document_count, get_all_documents
from app.config import (
    TOP_K_RESULTS,
    SCORE_THRESHOLD,
    RETRIEVAL_STRATEGY,
    MMR_ENABLED,
    MMR_LAMBDA,
    BM25_WEIGHT,
    DENSE_WEIGHT,
    logger,
)


# ═══════════════════════════════════════════════════════════════════════════
# Retrieval Metrics
# ═══════════════════════════════════════════════════════════════════════════

class RetrievalMetrics:
    """
    Track retrieval pipeline performance metrics.

    Attributes
    ----------
    total_candidates : int
        Total documents considered.
    dense_results : int
        Results from dense vector search.
    sparse_results : int
        Results from BM25 keyword search.
    after_threshold : int
        Results remaining after score threshold.
    after_mmr : int
        Final results after MMR diversity re-ranking.
    strategy : str
        The retrieval strategy used.
    latency_ms : float
        Total retrieval time in milliseconds.
    dense_latency_ms : float
        Dense search latency.
    sparse_latency_ms : float
        Sparse search latency.
    """

    def __init__(self) -> None:
        self.total_candidates = 0
        self.dense_results = 0
        self.sparse_results = 0
        self.after_threshold = 0
        self.after_mmr = 0
        self.strategy = ""
        self.latency_ms = 0.0
        self.dense_latency_ms = 0.0
        self.sparse_latency_ms = 0.0

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "total_candidates": self.total_candidates,
            "dense_results": self.dense_results,
            "sparse_results": self.sparse_results,
            "after_threshold": self.after_threshold,
            "after_mmr": self.after_mmr,
            "latency_ms": round(self.latency_ms, 2),
            "dense_latency_ms": round(self.dense_latency_ms, 2),
            "sparse_latency_ms": round(self.sparse_latency_ms, 2),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Dense Retrieval
# ═══════════════════════════════════════════════════════════════════════════

def _dense_retrieve(
    query_embedding: list[float],
    top_k: int,
) -> list[dict]:
    """
    Perform dense vector similarity search.

    Uses the embedding of the query to find the most similar
    document embeddings in ChromaDB via cosine distance.

    Returns list of dicts with: content, source, score, metadata.
    """
    results = query_documents(query_embedding, n_results=top_k)

    retrieved: list[dict] = []
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for doc, meta, dist in zip(documents, metadatas, distances):
        # Convert cosine distance to cosine similarity (1.0 - distance)
        similarity = 1.0 - dist
        retrieved.append({
            "content": doc,
            "source": meta.get("source", "unknown"),
            "score": round(similarity, 4),
            "retrieval_type": "dense",
            "metadata": meta,
        })

    return retrieved


# ═══════════════════════════════════════════════════════════════════════════
# Sparse Retrieval — BM25
# ═══════════════════════════════════════════════════════════════════════════

def _sparse_retrieve(
    query: str,
    top_k: int,
) -> list[dict]:
    """
    Perform sparse keyword retrieval using BM25 (Best Matching 25).

    BM25 is a ranking function that scores documents based on
    term frequency and inverse document frequency.  Unlike dense
    retrieval (which captures semantic meaning), BM25 excels at
    matching exact keywords and rare terms.

    Steps:
      1. Load all documents from the vector store.
      2. Tokenize documents and query.
      3. Compute BM25 scores.
      4. Return top-k results.
    """
    all_docs = get_all_documents()

    if not all_docs["documents"]:
        return []

    documents = all_docs["documents"]
    metadatas = all_docs["metadatas"]

    # Tokenize (simple whitespace + lowercasing)
    tokenized_docs = [doc.lower().split() for doc in documents]
    tokenized_query = query.lower().split()

    # Build BM25 index
    bm25 = BM25Okapi(tokenized_docs)
    scores = bm25.get_scores(tokenized_query)

    # Get top-k indices
    top_indices = np.argsort(scores)[::-1][:top_k]

    retrieved: list[dict] = []
    # Normalize scores to 0-1 range
    max_score = float(max(scores)) if max(scores) > 0 else 1.0

    for idx in top_indices:
        score = float(scores[idx])
        if score <= 0:
            continue

        normalized_score = score / max_score
        retrieved.append({
            "content": documents[idx],
            "source": metadatas[idx].get("source", "unknown"),
            "score": round(normalized_score, 4),
            "retrieval_type": "sparse",
            "metadata": metadatas[idx],
        })

    return retrieved


# ═══════════════════════════════════════════════════════════════════════════
# Hybrid Search
# ═══════════════════════════════════════════════════════════════════════════

def _hybrid_merge(
    dense_results: list[dict],
    sparse_results: list[dict],
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
) -> list[dict]:
    """
    Merge dense and sparse retrieval results with weighted scoring.

    Hybrid search combines the semantic understanding of dense
    retrieval with the keyword precision of sparse retrieval.

    The final score for each document is:
        score = (dense_weight × dense_score) + (sparse_weight × sparse_score)

    Documents that appear in both result sets get bonus from both scores.
    """
    # Build a map of content → combined result
    merged: dict[str, dict] = {}

    for result in dense_results:
        key = result["content"][:200]  # Use first 200 chars as key
        merged[key] = {
            "content": result["content"],
            "source": result["source"],
            "dense_score": result["score"],
            "sparse_score": 0.0,
            "retrieval_type": "hybrid",
            "metadata": result["metadata"],
        }

    for result in sparse_results:
        key = result["content"][:200]
        if key in merged:
            # Boost: appears in both dense and sparse results
            merged[key]["sparse_score"] = result["score"]
            merged[key]["retrieval_type"] = "hybrid (both)"
        else:
            merged[key] = {
                "content": result["content"],
                "source": result["source"],
                "dense_score": 0.0,
                "sparse_score": result["score"],
                "retrieval_type": "hybrid (sparse-only)",
                "metadata": result["metadata"],
            }

    # Compute combined scores
    results: list[dict] = []
    for item in merged.values():
        combined_score = (
            dense_weight * item["dense_score"]
            + sparse_weight * item["sparse_score"]
        )
        results.append({
            "content": item["content"],
            "source": item["source"],
            "score": round(combined_score, 4),
            "retrieval_type": item["retrieval_type"],
            "metadata": item["metadata"],
        })

    # Sort by combined score descending
    results.sort(key=lambda r: r["score"], reverse=True)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Score Threshold
# ═══════════════════════════════════════════════════════════════════════════

def _apply_threshold(
    results: list[dict],
    threshold: float,
) -> list[dict]:
    """
    Filter results below the score threshold.

    This prevents low-quality chunks from polluting the context
    and potentially causing hallucinations.
    """
    filtered = [r for r in results if r["score"] >= threshold]

    if len(filtered) < len(results):
        logger.info(
            "🔍 Threshold filter: %d → %d results (threshold=%.2f)",
            len(results), len(filtered), threshold,
        )

    return filtered


# ═══════════════════════════════════════════════════════════════════════════
# MMR — Maximum Marginal Relevance
# ═══════════════════════════════════════════════════════════════════════════

def _apply_mmr(
    results: list[dict],
    query_embedding: list[float],
    top_k: int,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Re-rank results using Maximum Marginal Relevance.

    MMR balances two objectives:
      1. Relevance — how similar each result is to the query.
      2. Diversity — how different each result is from already-selected results.

    The trade-off is controlled by lambda_param:
      • lambda=1.0 — pure relevance (no diversity).
      • lambda=0.5 — equal balance.
      • lambda=0.0 — pure diversity (ignore relevance).

    Algorithm:
      1. Start with the most relevant result.
      2. For each remaining candidate, compute:
         MMR = λ × Sim(candidate, query) - (1-λ) × max(Sim(candidate, selected))
      3. Select the candidate with the highest MMR score.
      4. Repeat until top_k results are selected.
    """
    if len(results) <= 1:
        return results[:top_k]

    # Lazy import to avoid circular dependency
    from app.ai.embeddings import generate_embeddings

    # Get embeddings for all result contents
    contents = [r["content"] for r in results]
    doc_embeddings = generate_embeddings(contents)

    # Track selected and remaining indices
    selected_indices: list[int] = []
    remaining_indices: list[int] = list(range(len(results)))

    # Step 1: Select the most relevant result first
    best_idx = 0
    selected_indices.append(best_idx)
    remaining_indices.remove(best_idx)

    # Step 2-4: Iteratively select based on MMR
    while len(selected_indices) < top_k and remaining_indices:
        best_mmr = float("-inf")
        best_remaining = remaining_indices[0]

        for idx in remaining_indices:
            # Relevance: similarity to query
            relevance = cosine_similarity(doc_embeddings[idx], query_embedding)

            # Diversity: max similarity to any already-selected document
            max_sim_to_selected = max(
                cosine_similarity(doc_embeddings[idx], doc_embeddings[sel_idx])
                for sel_idx in selected_indices
            )

            # MMR score
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected

            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_remaining = idx

        selected_indices.append(best_remaining)
        remaining_indices.remove(best_remaining)

    # Return results in MMR order
    mmr_results = [results[i] for i in selected_indices]

    logger.info(
        "🔀 MMR re-ranking: %d → %d results (λ=%.1f)",
        len(results), len(mmr_results), lambda_param,
    )

    return mmr_results


# ═══════════════════════════════════════════════════════════════════════════
# Main Retrieval Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def retrieve_context(
    query: str,
    top_k: int | None = None,
    strategy: str | None = None,
) -> tuple[list[dict], RetrievalMetrics]:
    """
    Execute the full retrieval pipeline for a user query.

    Pipeline stages:
      1. Dense search   — vector similarity via ChromaDB.
      2. Sparse search  — BM25 keyword matching (if hybrid).
      3. Hybrid merge   — combine scores (if hybrid).
      4. Threshold       — filter low-score results.
      5. MMR            — diversity re-ranking (if enabled).

    Parameters
    ----------
    query : str
        The user's question.
    top_k : int | None
        Number of results. Defaults to config TOP_K_RESULTS.
    strategy : str | None
        Retrieval strategy. Defaults to config RETRIEVAL_STRATEGY.
        Valid: dense, sparse, hybrid.

    Returns
    -------
    tuple[list[dict], RetrievalMetrics]
        (retrieved_chunks, metrics) where each chunk has:
        content, source, score, retrieval_type.
    """
    k = top_k or TOP_K_RESULTS
    strat = strategy or RETRIEVAL_STRATEGY
    metrics = RetrievalMetrics()
    metrics.strategy = strat
    metrics.total_candidates = get_document_count()

    start_time = time.perf_counter()

    # No knowledge ingested yet
    if metrics.total_candidates == 0:
        logger.info("📭 No documents in vector store — skipping retrieval")
        return [], metrics

    logger.info("🔍 Retrieving context for: %s (strategy=%s)", query[:80], strat)

    # ── Stage 1: Dense retrieval ─────────────────────────────
    dense_start = time.perf_counter()
    query_embedding = generate_query_embedding(query)

    dense_results = []
    if strat in ("dense", "hybrid"):
        # Fetch more than top_k for post-processing
        fetch_k = k * 3 if MMR_ENABLED else k
        dense_results = _dense_retrieve(query_embedding, top_k=fetch_k)
        metrics.dense_results = len(dense_results)

    metrics.dense_latency_ms = (time.perf_counter() - dense_start) * 1000

    # ── Stage 2: Sparse retrieval (BM25) ─────────────────────
    sparse_results = []
    if strat in ("sparse", "hybrid"):
        sparse_start = time.perf_counter()
        sparse_results = _sparse_retrieve(query, top_k=k * 2)
        metrics.sparse_results = len(sparse_results)
        metrics.sparse_latency_ms = (time.perf_counter() - sparse_start) * 1000

    # ── Stage 3: Merge results ───────────────────────────────
    if strat == "hybrid" and dense_results and sparse_results:
        results = _hybrid_merge(
            dense_results, sparse_results,
            dense_weight=DENSE_WEIGHT,
            sparse_weight=BM25_WEIGHT,
        )
    elif strat == "sparse":
        results = sparse_results
    else:
        results = dense_results

    # ── Stage 4: Score threshold ─────────────────────────────
    results = _apply_threshold(results, SCORE_THRESHOLD)
    metrics.after_threshold = len(results)

    # ── Stage 5: MMR diversity re-ranking ────────────────────
    if MMR_ENABLED and len(results) > 1:
        results = _apply_mmr(
            results, query_embedding,
            top_k=k,
            lambda_param=MMR_LAMBDA,
        )
    else:
        results = results[:k]

    metrics.after_mmr = len(results)
    metrics.latency_ms = (time.perf_counter() - start_time) * 1000

    logger.info(
        "✅ Retrieved %d chunks | strategy=%s | latency=%.1fms",
        len(results), strat, metrics.latency_ms,
    )

    return results, metrics
