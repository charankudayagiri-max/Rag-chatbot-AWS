"""
embeddings.py — Vector Embedding Generator

CONCEPTS DEMONSTRATED
─────────────────────
  What are Embeddings, Semantic Meaning, Vector Representation,
         High-dimensional Space, Cosine Similarity, Dot Product,
         Euclidean Distance, Dense Embeddings, Embedding Models,
         Dimensions, Caching, Cost Optimization.

Pipeline stage 5:
    Loader → Parser → Cleaner → Chunker → **Embeddings** → Vector Store

WHAT IS AN EMBEDDING?
─────────────────────
An embedding is a list of numbers (a vector) that captures the *meaning*
of a piece of text.  Similar texts produce similar vectors.

WHY LOCAL EMBEDDINGS?
─────────────────────
We use sentence-transformers (runs on your CPU/GPU) instead of an
external API.  Benefits:
  • Free — no API costs.
  • Fast — no network latency.
  • Private — your data never leaves your machine.
  • Educational — you can inspect the actual vectors.
"""

from __future__ import annotations

import math
from functools import lru_cache
from collections import OrderedDict

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL, EMBEDDING_CACHE_ENABLED, EMBEDDING_CACHE_SIZE, logger


# ═══════════════════════════════════════════════════════════════════════════
# Cached Model (Singleton)
# ═══════════════════════════════════════════════════════════════════════════

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Load or return the cached embedding model."""
    global _model
    if _model is None:
        logger.info("🧠 Loading embedding model: %s ...", EMBEDDING_MODEL)
        _model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info(
            "✅ Embedding model loaded (dimension=%d)",
            _model.get_sentence_embedding_dimension(),
        )
    return _model


# ═══════════════════════════════════════════════════════════════════════════
# Embedding Cache
# ═══════════════════════════════════════════════════════════════════════════

class EmbeddingCache:
    """
    LRU cache for embedding vectors.

    Avoids re-computing embeddings for text we've already encoded.
    This is particularly useful for repeated queries or when the
    same document chunks are queried multiple times.

    Uses an OrderedDict for O(1) LRU eviction.
    """

    def __init__(self, max_size: int = 1000) -> None:
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._max_size = max_size
        self._hits = 0
        self._misses = 0

    def get(self, text: str) -> list[float] | None:
        """Look up a cached embedding. Returns None on miss."""
        if text in self._cache:
            self._hits += 1
            # Move to end (most recently used)
            self._cache.move_to_end(text)
            return self._cache[text]
        self._misses += 1
        return None

    def put(self, text: str, embedding: list[float]) -> None:
        """Store an embedding in the cache."""
        if text in self._cache:
            self._cache.move_to_end(text)
        else:
            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)  # Remove oldest
            self._cache[text] = embedding

    @property
    def stats(self) -> dict:
        """Return cache statistics."""
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 4) if total > 0 else 0.0,
        }


_cache = EmbeddingCache(max_size=EMBEDDING_CACHE_SIZE) if EMBEDDING_CACHE_ENABLED else None


# ═══════════════════════════════════════════════════════════════════════════
# Similarity Metrics
# ═══════════════════════════════════════════════════════════════════════════

def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    Cosine similarity measures the angle between two vectors,
    ignoring their magnitude.  Range: [-1, 1] where 1 = identical.

    This is the most common similarity metric for text embeddings
    because it normalizes for document length.
    """
    a = np.array(vec_a)
    b = np.array(vec_b)
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def dot_product(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute dot product between two vectors.

    Dot product combines both direction and magnitude.
    Higher values indicate more similarity, but the scale
    depends on vector magnitudes.
    """
    return float(np.dot(np.array(vec_a), np.array(vec_b)))


def euclidean_distance(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute Euclidean distance between two vectors.

    Euclidean distance measures straight-line distance in
    high-dimensional space.  Smaller = more similar.
    """
    return float(np.linalg.norm(np.array(vec_a) - np.array(vec_b)))


# ═══════════════════════════════════════════════════════════════════════════
# Embedding Info
# ═══════════════════════════════════════════════════════════════════════════

def get_embedding_info() -> dict:
    """
    Return detailed information about the embedding model and cache.

    Returns
    -------
    dict
        Model name, dimension, type, and cache statistics.
    """
    model = _get_model()
    info = {
        "model_name": EMBEDDING_MODEL,
        "dimensions": model.get_sentence_embedding_dimension(),
        "type": "dense",
        "max_sequence_length": model.max_seq_length,
        "cache_enabled": EMBEDDING_CACHE_ENABLED,
    }
    if _cache is not None:
        info["cache_stats"] = _cache.stats
    return info


# ═══════════════════════════════════════════════════════════════════════════
# Core Embedding Functions
# ═══════════════════════════════════════════════════════════════════════════

def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Convert a list of text strings into embedding vectors.

    Parameters
    ----------
    texts : list[str]
        The text chunks to embed.

    Returns
    -------
    list[list[float]]
        A list of embedding vectors (each is a list of floats).
    """
    model = _get_model()
    logger.info("🔢 Generating embeddings for %d texts", len(texts))

    if _cache is not None and EMBEDDING_CACHE_ENABLED:
        # Check cache first, batch-encode only misses
        results: list[list[float] | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, text in enumerate(texts):
            cached = _cache.get(text)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        if uncached_texts:
            new_embeddings = model.encode(uncached_texts, show_progress_bar=False)
            for idx, emb in zip(uncached_indices, new_embeddings):
                emb_list = emb.tolist()
                results[idx] = emb_list
                _cache.put(texts[idx], emb_list)

        logger.info(
            "✅ Generated %d embeddings (cache hits: %d, computed: %d)",
            len(texts), len(texts) - len(uncached_texts), len(uncached_texts),
        )
        return results  # type: ignore
    else:
        # No cache — batch encode everything
        embeddings = model.encode(texts, show_progress_bar=False)
        result = [emb.tolist() for emb in embeddings]
        logger.info("✅ Generated %d embeddings", len(result))
        return result


def generate_query_embedding(query: str) -> list[float]:
    """
    Convert a single query string into an embedding vector.

    Parameters
    ----------
    query : str
        The user's question.

    Returns
    -------
    list[float]
        The embedding vector for the query.
    """
    if _cache is not None and EMBEDDING_CACHE_ENABLED:
        cached = _cache.get(query)
        if cached is not None:
            return cached

    model = _get_model()
    embedding = model.encode(query, show_progress_bar=False)
    emb_list = embedding.tolist()

    if _cache is not None and EMBEDDING_CACHE_ENABLED:
        _cache.put(query, emb_list)

    return emb_list
