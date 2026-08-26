"""
vector_store.py — ChromaDB Vector Database Wrapper

CONCEPTS DEMONSTRATED
─────────────────────
  Why SQL isn't enough, What is a Vector Database, Vector Storage,
         Nearest Neighbor Search, Approximate NN (HNSW), Metadata Filtering,
         Collections, Namespaces, Persistence.

Pipeline stage 6:
    Loader → Parser → Cleaner → Chunker → Embeddings → **Vector Store**

WHY CHROMADB?
─────────────
  • Runs locally — no server to manage.
  • Persists to disk — survives server restarts.
  • HNSW index — fast approximate nearest neighbor search.
  • Metadata filtering — query by source, date, chunk index.
  • Beginner-friendly API.
"""

from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone

import chromadb
from app.config import CHROMA_PERSIST_DIR, logger

# ── Cached Client & Collection ───────────────────────────────────────────
_client: chromadb.ClientAPI | None = None
_collection: chromadb.Collection | None = None

COLLECTION_NAME = "knowledge_base"


def _get_client() -> chromadb.ClientAPI:
    """Get or create the ChromaDB persistent client."""
    global _client
    if _client is None:
        logger.info("💾 Initialising ChromaDB at: %s", CHROMA_PERSIST_DIR)
        _client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return _client


def _get_collection() -> chromadb.Collection:
    """
    Get or create the ChromaDB collection.

    The collection uses HNSW (Hierarchical Navigable Small World) index
    with cosine distance, which is the standard for text similarity.
    """
    global _collection

    if _collection is None:
        client = _get_client()
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        count = _collection.count()
        logger.info(
            "✅ ChromaDB ready — collection '%s' has %d documents",
            COLLECTION_NAME, count,
        )

        # Persistence verification
        if count > 0:
            logger.info("💾 Persistence verified: %d documents loaded from disk", count)

    return _collection


# ═══════════════════════════════════════════════════════════════════════════
# Document ID Generation
# ═══════════════════════════════════════════════════════════════════════════

def _generate_doc_id(content: str, source: str) -> str:
    """
    Generate a deterministic document ID from content hash.

    Uses SHA-256 hash of content + source to detect duplicates.
    If the same text from the same source is ingested twice,
    it gets the same ID and ChromaDB will skip it.
    """
    hash_input = f"{source}::{content}"
    return hashlib.sha256(hash_input.encode("utf-8")).hexdigest()[:16]


# ═══════════════════════════════════════════════════════════════════════════
# Add Documents
# ═══════════════════════════════════════════════════════════════════════════

def add_documents(
    chunks: list[str],
    embeddings: list[list[float]],
    source_url: str,
) -> int:
    """
    Store text chunks with their embeddings in the vector database.

    Each chunk is stored with rich metadata:
      • source      — the URL or file path this chunk came from.
      • chunk_index — position of this chunk in the document.
      • ingested_at — ISO timestamp when this chunk was stored.
      • char_count  — number of characters in the chunk.

    Parameters
    ----------
    chunks : list[str]
        The text chunks to store.
    embeddings : list[list[float]]
        The corresponding embedding vectors.
    source_url : str
        The URL these chunks came from (stored as metadata).

    Returns
    -------
    int
        The number of chunks stored.
    """
    collection = _get_collection()

    # Generate content-hash IDs for deduplication
    ids = [_generate_doc_id(chunk, source_url) for chunk in chunks]

    # Rich metadata for each chunk
    ingested_at = datetime.now(timezone.utc).isoformat()
    metadatas = [
        {
            "source": source_url,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "ingested_at": ingested_at,
            "char_count": len(chunk),
        }
        for i, chunk in enumerate(chunks)
    ]

    # Check for existing documents to avoid duplicates
    existing = set()
    try:
        existing_docs = collection.get(ids=ids)
        existing = set(existing_docs["ids"]) if existing_docs["ids"] else set()
    except Exception:
        pass

    # Filter out duplicates
    new_ids, new_chunks, new_embeddings, new_metadatas = [], [], [], []
    for doc_id, chunk, emb, meta in zip(ids, chunks, embeddings, metadatas):
        if doc_id not in existing:
            new_ids.append(doc_id)
            new_chunks.append(chunk)
            new_embeddings.append(emb)
            new_metadatas.append(meta)

    skipped = len(chunks) - len(new_ids)
    if skipped > 0:
        logger.info("⏭️ Skipped %d duplicate chunks", skipped)

    if new_ids:
        collection.add(
            ids=new_ids,
            documents=new_chunks,
            embeddings=new_embeddings,
            metadatas=new_metadatas,
        )

    logger.info(
        "✅ Stored %d chunks from %s (skipped %d dupes, total: %d)",
        len(new_ids), source_url, skipped, collection.count(),
    )
    return len(new_ids)


# ═══════════════════════════════════════════════════════════════════════════
# Query Documents
# ═══════════════════════════════════════════════════════════════════════════

def query_documents(
    query_embedding: list[float],
    n_results: int = 5,
    metadata_filter: dict | None = None,
) -> dict:
    """
    Find the most similar documents to a query embedding.

    Uses HNSW (Hierarchical Navigable Small World) approximate
    nearest neighbor search for fast retrieval.

    Parameters
    ----------
    query_embedding : list[float]
        The embedding vector of the user's question.
    n_results : int
        Number of top results to return.
    metadata_filter : dict | None
        Optional ChromaDB where clause for metadata filtering.
        Example: {"source": "https://example.com"}

    Returns
    -------
    dict
        ChromaDB query results with documents, metadatas, and distances.
    """
    collection = _get_collection()

    if collection.count() == 0:
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    query_params = {
        "query_embeddings": [query_embedding],
        "n_results": min(n_results, collection.count()),
    }

    # Metadata filtering
    if metadata_filter:
        query_params["where"] = metadata_filter
        logger.info("🔍 Querying with metadata filter: %s", metadata_filter)

    results = collection.query(**query_params)

    logger.info(
        "🔍 Retrieved %d results from vector store",
        len(results["documents"][0]),
    )
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Get All Documents (for BM25 retrieval)
# ═══════════════════════════════════════════════════════════════════════════

def get_all_documents() -> dict:
    """
    Retrieve all documents from the collection.

    Used by BM25 sparse retrieval to build the keyword index.

    Returns
    -------
    dict
        All documents with their IDs, contents, and metadatas.
    """
    collection = _get_collection()
    if collection.count() == 0:
        return {"ids": [], "documents": [], "metadatas": []}

    return collection.get(include=["documents", "metadatas"])


# ═══════════════════════════════════════════════════════════════════════════
# Statistics
# ═══════════════════════════════════════════════════════════════════════════

def get_document_count() -> int:
    """Return the total number of documents in the vector store."""
    return _get_collection().count()


def get_sources() -> list[str]:
    """Return a list of unique source URLs in the knowledge base."""
    collection = _get_collection()
    if collection.count() == 0:
        return []

    all_data = collection.get(include=["metadatas"])
    sources = set()
    for meta in all_data["metadatas"]:
        if "source" in meta:
            sources.add(meta["source"])

    return sorted(sources)


def get_vector_store_stats() -> dict:
    """
    Return detailed statistics about the vector store.

    Returns
    -------
    dict
        Comprehensive stats including document count, sources,
        collection metadata, and per-source chunk counts.
    """
    collection = _get_collection()
    count = collection.count()

    if count == 0:
        return {
            "total_chunks": 0,
            "unique_sources": 0,
            "sources": [],
            "collection_name": COLLECTION_NAME,
            "persist_directory": CHROMA_PERSIST_DIR,
            "index_type": "HNSW",
            "distance_metric": "cosine",
        }

    all_data = collection.get(include=["metadatas"])

    # Per-source statistics
    source_stats: dict[str, dict] = {}
    for meta in all_data["metadatas"]:
        source = meta.get("source", "unknown")
        if source not in source_stats:
            source_stats[source] = {
                "url": source,
                "chunk_count": 0,
                "total_chars": 0,
                "ingested_at": meta.get("ingested_at", "unknown"),
            }
        source_stats[source]["chunk_count"] += 1
        source_stats[source]["total_chars"] += meta.get("char_count", 0)

    return {
        "total_chunks": count,
        "unique_sources": len(source_stats),
        "sources": list(source_stats.values()),
        "collection_name": COLLECTION_NAME,
        "persist_directory": CHROMA_PERSIST_DIR,
        "index_type": "HNSW",
        "distance_metric": "cosine",
    }


# ═══════════════════════════════════════════════════════════════════════════
# Collection Management
# ═══════════════════════════════════════════════════════════════════════════

def clear_collection() -> None:
    """Delete all indexed chunks in the ChromaDB collection."""
    collection = _get_collection()
    if collection.count() > 0:
        all_data = collection.get()
        ids = all_data["ids"]
        collection.delete(ids=ids)
        logger.info("🗑️ Cleared all %d documents from ChromaDB.", len(ids))


def delete_source(source_url: str) -> int:
    """
    Delete all chunks from a specific source URL.

    Parameters
    ----------
    source_url : str
        The source URL whose chunks should be deleted.

    Returns
    -------
    int
        Number of chunks deleted.
    """
    collection = _get_collection()
    if collection.count() == 0:
        return 0

    # Find documents with this source
    all_data = collection.get(include=["metadatas"])
    ids_to_delete = [
        doc_id
        for doc_id, meta in zip(all_data["ids"], all_data["metadatas"])
        if meta.get("source") == source_url
    ]

    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
        logger.info(
            "🗑️ Deleted %d chunks from source: %s",
            len(ids_to_delete), source_url,
        )

    return len(ids_to_delete)
