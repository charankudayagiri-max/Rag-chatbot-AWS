"""
knowledge_service.py — Knowledge Ingestion Orchestrator (Async Engine)

CONCEPTS DEMONSTRATED
─────────────────────
  Indexing Pipeline (URL/File → Load → Parse → Clean → Chunk → Embed → Store).
         Asynchronous background execution using FastAPI threads and PostgreSQL tasks.
  PDF/Document parsing.
  Multi-strategy chunking.
"""

from __future__ import annotations

import time
import uuid
import json
from datetime import datetime
from contextlib import contextmanager

from app.database import SessionLocal
from app.db_models import IngestionTask
from app.ai.loader import load_url
from app.ai.parser import parse_html
from app.ai.cleaner import clean_text
from app.ai.chunker import chunk_text, evaluate_chunks
from app.ai.embeddings import generate_embeddings
from app.ai.vector_store import (
    add_documents, get_sources, get_document_count,
    get_vector_store_stats,
)
from app.config import logger, PIPELINE_METRICS_ENABLED, CHUNKING_STRATEGY


# ═══════════════════════════════════════════════════════════════════════════
# Database Context Helper
# ═══════════════════════════════════════════════════════════════════════════

@contextmanager
def get_db_session():
    """Context manager for local self-contained database connections."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as exc:
        db.rollback()
        logger.exception("Database operation failed in knowledge service:")
        raise exc
    finally:
        db.close()


# ═══════════════════════════════════════════════════════════════════════════
# Ingestion Pipeline Metrics
# ═══════════════════════════════════════════════════════════════════════════

class IngestionMetrics:
    """Track per-stage metrics for the indexing pipeline."""

    def __init__(self) -> None:
        self.load_ms = 0.0
        self.parse_ms = 0.0
        self.clean_ms = 0.0
        self.chunk_ms = 0.0
        self.embed_ms = 0.0
        self.store_ms = 0.0
        self.total_ms = 0.0

        # Size metrics
        self.raw_html_chars = 0
        self.parsed_chars = 0
        self.cleaned_chars = 0
        self.chunks_created = 0
        self.embeddings_generated = 0
        self.chunks_stored = 0

        # Chunk quality
        self.chunk_metrics: dict = {}
        self.chunking_strategy = ""

    def to_dict(self) -> dict:
        return {
            "timing": {
                "load_ms": round(self.load_ms, 2),
                "parse_ms": round(self.parse_ms, 2),
                "clean_ms": round(self.clean_ms, 2),
                "chunk_ms": round(self.chunk_ms, 2),
                "embed_ms": round(self.embed_ms, 2),
                "store_ms": round(self.store_ms, 2),
                "total_ms": round(self.total_ms, 2),
            },
            "sizes": {
                "raw_html_chars": self.raw_html_chars,
                "parsed_chars": self.parsed_chars,
                "cleaned_chars": self.cleaned_chars,
                "chunks_created": self.chunks_created,
                "embeddings_generated": self.embeddings_generated,
                "chunks_stored": self.chunks_stored,
            },
            "chunking": {
                "strategy": self.chunking_strategy,
                "metrics": self.chunk_metrics,
            },
        }


# ═══════════════════════════════════════════════════════════════════════════
# Task Operations
# ═══════════════════════════════════════════════════════════════════════════

def initiate_ingestion_task(source: str) -> str:
    """Create a new IngestionTask in the database with status 'processing'."""
    task_id = str(uuid.uuid4())
    with get_db_session() as db:
        task = IngestionTask(
            id=task_id,
            source=source,
            status="processing",
            created_at=datetime.utcnow()
        )
        db.add(task)
        logger.info("🎬 Initiated background task: %s (source=%s)", task_id, source)
    return task_id


def get_task_status(task_id: str) -> dict | None:
    """Retrieve status and metadata of an ingestion task."""
    with get_db_session() as db:
        task = db.query(IngestionTask).filter(IngestionTask.id == task_id).first()
        return task.to_dict() if task else None


# ═══════════════════════════════════════════════════════════════════════════
# Asynchronous Background Ingestion Executors
# ═══════════════════════════════════════════════════════════════════════════

def run_async_website_ingestion(task_id: str, url: str) -> None:
    """
    Background worker that runs the full website ingestion pipeline.
    Catches errors and updates PostgreSQL task state dynamically.
    """
    metrics = IngestionMetrics()
    metrics.chunking_strategy = CHUNKING_STRATEGY
    total_start = time.perf_counter()
    logger.info("🚀 Background Ingestion start for URL: %s (task=%s)", url, task_id)

    try:
        # Stage 1: Load URL
        t = time.perf_counter()
        html = load_url(url)
        metrics.load_ms = (time.perf_counter() - t) * 1000
        metrics.raw_html_chars = len(html)

        # Stage 2: Parse HTML
        t = time.perf_counter()
        text = parse_html(html)
        metrics.parse_ms = (time.perf_counter() - t) * 1000
        metrics.parsed_chars = len(text) if text else 0

        if not text or len(text.strip()) < 50:
            raise ValueError(
                f"Could not extract meaningful text from {url}. "
                "The page may be empty, JavaScript-rendered, or blocked."
            )

        # Stage 3: Clean Text
        t = time.perf_counter()
        cleaned = clean_text(text)
        metrics.clean_ms = (time.perf_counter() - t) * 1000
        metrics.cleaned_chars = len(cleaned) if cleaned else 0

        if not cleaned or len(cleaned.strip()) < 50:
            raise ValueError(f"After cleaning, not enough text remained from {url}.")

        # Stage 4: Chunk Text
        t = time.perf_counter()
        chunks = chunk_text(cleaned)
        metrics.chunk_ms = (time.perf_counter() - t) * 1000
        metrics.chunks_created = len(chunks)
        metrics.chunk_metrics = evaluate_chunks(chunks)

        if not chunks:
            raise ValueError(f"Could not split content into chunks for {url}.")

        # Stage 5: Generate Embeddings
        t = time.perf_counter()
        embeddings = generate_embeddings(chunks)
        metrics.embed_ms = (time.perf_counter() - t) * 1000
        metrics.embeddings_generated = len(embeddings)

        # Stage 6: Store in ChromaDB
        t = time.perf_counter()
        stored_count = add_documents(chunks, embeddings, source_url=url)
        metrics.store_ms = (time.perf_counter() - t) * 1000
        metrics.chunks_stored = stored_count

        metrics.total_ms = (time.perf_counter() - total_start) * 1000

        # Success - update database
        with get_db_session() as db:
            task = db.query(IngestionTask).filter(IngestionTask.id == task_id).first()
            if task:
                task.status = "completed"
                task.chunks_created = stored_count
                task.completed_at = datetime.utcnow()
                if PIPELINE_METRICS_ENABLED:
                    task.metrics = json.dumps(metrics.to_dict())
        logger.info("🎉 Background Ingestion SUCCESS for task: %s (%d chunks)", task_id, stored_count)

    except Exception as exc:
        logger.exception("❌ Background Ingestion FAILED for task %s:", task_id)
        # Update database with error logs
        with get_db_session() as db:
            task = db.query(IngestionTask).filter(IngestionTask.id == task_id).first()
            if task:
                task.status = "failed"
                task.error_message = str(exc)
                task.completed_at = datetime.utcnow()


def run_async_file_ingestion(task_id: str, filename: str, content_bytes: bytes) -> None:
    """
    Background worker that runs the full file ingestion pipeline.
    """
    from app.ai.file_loader import load_file

    metrics = IngestionMetrics()
    metrics.chunking_strategy = CHUNKING_STRATEGY
    total_start = time.perf_counter()
    logger.info("🚀 Background Ingestion start for File: %s (task=%s)", filename, task_id)

    try:
        # Stage 1: Load file text
        t = time.perf_counter()
        text, file_type = load_file(filename, content_bytes)
        metrics.load_ms = (time.perf_counter() - t) * 1000
        metrics.parsed_chars = len(text)

        if not text or len(text.strip()) < 10:
            raise ValueError(f"Could not extract meaningful text from {filename}.")

        # Stage 2: Clean Text
        t = time.perf_counter()
        cleaned = clean_text(text)
        metrics.clean_ms = (time.perf_counter() - t) * 1000
        metrics.cleaned_chars = len(cleaned) if cleaned else 0

        if not cleaned or len(cleaned.strip()) < 10:
            raise ValueError(f"After cleaning, not enough text remained from {filename}.")

        # Stage 3: Chunk Text
        t = time.perf_counter()
        chunks = chunk_text(cleaned)
        metrics.chunk_ms = (time.perf_counter() - t) * 1000
        metrics.chunks_created = len(chunks)
        metrics.chunk_metrics = evaluate_chunks(chunks)

        if not chunks:
            raise ValueError(f"Could not split file content into chunks for {filename}.")

        # Stage 4: Generate Embeddings
        t = time.perf_counter()
        embeddings = generate_embeddings(chunks)
        metrics.embed_ms = (time.perf_counter() - t) * 1000
        metrics.embeddings_generated = len(embeddings)

        # Stage 5: Store in ChromaDB
        source_name = f"file://{filename}"
        t = time.perf_counter()
        stored_count = add_documents(chunks, embeddings, source_url=source_name)
        metrics.store_ms = (time.perf_counter() - t) * 1000
        metrics.chunks_stored = stored_count

        metrics.total_ms = (time.perf_counter() - total_start) * 1000

        # Success - update database
        with get_db_session() as db:
            task = db.query(IngestionTask).filter(IngestionTask.id == task_id).first()
            if task:
                task.status = "completed"
                task.chunks_created = stored_count
                task.completed_at = datetime.utcnow()
                if PIPELINE_METRICS_ENABLED:
                    task.metrics = json.dumps(metrics.to_dict())
        logger.info("🎉 Background Ingestion SUCCESS for file task: %s (%d chunks)", task_id, stored_count)

    except Exception as exc:
        logger.exception("❌ Background Ingestion FAILED for file task %s:", task_id)
        # Update database with error logs
        with get_db_session() as db:
            task = db.query(IngestionTask).filter(IngestionTask.id == task_id).first()
            if task:
                task.status = "failed"
                task.error_message = str(exc)
                task.completed_at = datetime.utcnow()


# ═══════════════════════════════════════════════════════════════════════════
# Deprecated Synchronous Wrappers (for test compatibility)
# ═══════════════════════════════════════════════════════════════════════════

def ingest_website(url: str) -> dict:
    """Wrapper that executes synchronously for testing backward compatibility."""
    task_id = initiate_ingestion_task(url)
    run_async_website_ingestion(task_id, url)
    res = get_task_status(task_id)
    return {
        "status": res["status"],
        "url": url,
        "chunks_created": res["chunks_created"],
        "message": f"Successfully learned {res['chunks_created']} knowledge chunks from {url}",
        "metrics": res.get("metrics")
    }


def ingest_file(filename: str, content_bytes: bytes) -> dict:
    """Wrapper that executes file pipeline synchronously for testing."""
    task_id = initiate_ingestion_task(f"file://{filename}")
    run_async_file_ingestion(task_id, filename, content_bytes)
    res = get_task_status(task_id)
    return {
        "status": res["status"],
        "filename": filename,
        "file_type": "pdf" if filename.lower().endswith(".pdf") else "txt",
        "chunks_created": res["chunks_created"],
        "message": f"Successfully learned {res['chunks_created']} knowledge chunks from {filename}",
        "metrics": res.get("metrics")
    }


# ═══════════════════════════════════════════════════════════════════════════
# Knowledge Base Status
# ═══════════════════════════════════════════════════════════════════════════

def get_knowledge_status() -> dict:
    """Return current knowledge base status."""
    sources = get_sources()
    count = get_document_count()
    return {
        "total_chunks": count,
        "sources": sources,
        "has_knowledge": count > 0,
    }


def get_knowledge_stats() -> dict:
    """Return detailed vector store statistics."""
    return get_vector_store_stats()


def clear_knowledge_base() -> None:
    """Clear all items in the vector database."""
    from app.ai.vector_store import clear_collection
    clear_collection()
