"""
chat_service.py — Chat Orchestrator (Three-Pipeline Architecture)

CONCEPTS DEMONSTRATED
─────────────────────
  Complete RAG Architecture, Retrieval Pipeline, Generation Pipeline,
         Embedding Pipeline, Knowledge Base, Retriever, Prompt Builder,
         Generator, Advantages, Limitations, Failure Cases.

Responsibility: Coordinate the RAG chat workflow across three pipelines.

PIPELINE ARCHITECTURE
──────────────────────────────
  1. Retrieval Pipeline:
     Query → Embed → Dense Search → BM25 → Hybrid Merge → Threshold → MMR

  2. Generation Pipeline:
     Retrieved Chunks → Context Filter → Prompt Build → Context Assemble →
     Context Budget → LLM Call → Response

  3. Diagnostics:
     Confidence scoring, knowledge gap detection, retrieval quality assessment.

This service bridges the API layer and the AI Pipeline layer.
"""

from __future__ import annotations

import time

from app.ai.retriever import retrieve_context, RetrievalMetrics
from app.ai.prompt_builder import build_prompt
from app.ai.context_engine import assemble_context, filter_chunks, ContextBudget
from app.ai.diagnostics import (
    assess_retrieval_quality,
    detect_knowledge_gap,
    compute_confidence_score,
)
from app.ai.vector_store import get_document_count
from app.llm import get_llm_response, get_llm_response_stream
from app.config import logger, MODEL_NAME, PIPELINE_METRICS_ENABLED


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline Metrics
# ═══════════════════════════════════════════════════════════════════════════

class PipelineMetrics:
    """Track timing for each stage of the RAG pipeline."""

    def __init__(self) -> None:
        self.retrieval_ms = 0.0
        self.context_ms = 0.0
        self.generation_ms = 0.0
        self.total_ms = 0.0
        self.retrieval_details: dict = {}
        self.context_budget: dict = {}
        self.diagnostics: dict = {}

    def to_dict(self) -> dict:
        return {
            "retrieval_ms": round(self.retrieval_ms, 2),
            "context_ms": round(self.context_ms, 2),
            "generation_ms": round(self.generation_ms, 2),
            "total_ms": round(self.total_ms, 2),
            "retrieval_details": self.retrieval_details,
            "context_budget": self.context_budget,
            "diagnostics": self.diagnostics,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Non-Streaming Response
# ═══════════════════════════════════════════════════════════════════════════

def generate_chat_response(
    user_prompt: str,
    conversation_history: list[dict[str, str]] | None = None,
) -> dict:
    """
    Generate an AI response using the full RAG pipeline.

    Three-Pipeline Flow:
      Pipeline 1 — RETRIEVAL: Query → Embed → Search → Filter → MMR
      Pipeline 2 — GENERATION: Context → Budget → Compress → LLM → Response
      Pipeline 3 — DIAGNOSTICS: Quality → Confidence → Gap Detection

    Parameters
    ----------
    user_prompt : str
        The user's question.
    conversation_history : list[dict] | None
        Previous messages for multi-turn context.

    Returns
    -------
    dict
        Result with keys: response, sources, has_knowledge, metrics,
        confidence, diagnostics.
    """
    pipeline = PipelineMetrics()
    total_start = time.perf_counter()

    has_knowledge = get_document_count() > 0

    # ── Pipeline 1: RETRIEVAL ────────────────────────────────
    retrieval_start = time.perf_counter()

    if has_knowledge:
        retrieved_chunks, retrieval_metrics = retrieve_context(user_prompt)
        # Apply context-level filtering
        retrieved_chunks = filter_chunks(retrieved_chunks)
    else:
        retrieved_chunks = []
        retrieval_metrics = RetrievalMetrics()

    pipeline.retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
    pipeline.retrieval_details = retrieval_metrics.to_dict()

    # ── Pipeline 2: GENERATION ───────────────────────────────
    context_start = time.perf_counter()

    # Build system prompt with context
    system_prompt = build_prompt(retrieved_chunks)

    # Assemble context with budget management
    messages, budget = assemble_context(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        retrieved_chunks=retrieved_chunks,
        conversation_history=conversation_history,
    )

    pipeline.context_ms = (time.perf_counter() - context_start) * 1000
    pipeline.context_budget = budget.to_dict()

    # Call LLM
    gen_start = time.perf_counter()
    response_text = get_llm_response(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        conversation_history=conversation_history,
    )
    pipeline.generation_ms = (time.perf_counter() - gen_start) * 1000

    # ── Pipeline 3: DIAGNOSTICS ──────────────────────────────
    confidence = compute_confidence_score(retrieved_chunks, has_knowledge)
    knowledge_gap = detect_knowledge_gap(retrieved_chunks, has_knowledge)
    quality = assess_retrieval_quality(retrieved_chunks)

    pipeline.diagnostics = {
        "confidence": confidence,
        "knowledge_gap": knowledge_gap,
        "retrieval_quality": quality,
    }

    pipeline.total_ms = (time.perf_counter() - total_start) * 1000

    # Format sources for the frontend
    sources = [
        {
            "content": chunk["content"][:200] + "..." if len(chunk["content"]) > 200 else chunk["content"],
            "source": chunk["source"],
            "score": chunk["score"],
            "retrieval_type": chunk.get("retrieval_type", "dense"),
        }
        for chunk in retrieved_chunks
    ]

    logger.info(
        "💬 Chat response | knowledge=%s | sources=%d | confidence=%d/5 | time=%.0fms",
        has_knowledge, len(sources), confidence, pipeline.total_ms,
    )

    return {
        "response": response_text,
        "sources": sources,
        "has_knowledge": has_knowledge,
        "confidence": confidence,
        "diagnostics": knowledge_gap,
        "metrics": pipeline.to_dict() if PIPELINE_METRICS_ENABLED else None,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Streaming Response
# ═══════════════════════════════════════════════════════════════════════════

def generate_chat_response_stream(
    user_prompt: str,
    conversation_history: list[dict[str, str]] | None = None,
):
    """
    Generate a streamed AI response using the RAG pipeline.

    Streaming Architecture:
      1. First yield: metadata (sources, model, diagnostics, metrics).
      2. Content yields: individual tokens as they arrive from the LLM.
      3. The API layer converts these into SSE/NDJSON for the frontend.

    Yields
    ------
    dict
        Dictionary packet with 'type' field and corresponding payload.
    """
    pipeline = PipelineMetrics()
    total_start = time.perf_counter()

    has_knowledge = get_document_count() > 0

    # ── Pipeline 1: RETRIEVAL ────────────────────────────────
    retrieval_start = time.perf_counter()

    if has_knowledge:
        retrieved_chunks, retrieval_metrics = retrieve_context(user_prompt)
        retrieved_chunks = filter_chunks(retrieved_chunks)
    else:
        retrieved_chunks = []
        retrieval_metrics = RetrievalMetrics()

    pipeline.retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
    pipeline.retrieval_details = retrieval_metrics.to_dict()

    # ── Pipeline 2: GENERATION (setup) ───────────────────────
    context_start = time.perf_counter()
    system_prompt = build_prompt(retrieved_chunks)

    messages, budget = assemble_context(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        retrieved_chunks=retrieved_chunks,
        conversation_history=conversation_history,
    )

    pipeline.context_ms = (time.perf_counter() - context_start) * 1000
    pipeline.context_budget = budget.to_dict()

    # ── Pipeline 3: DIAGNOSTICS ──────────────────────────────
    confidence = compute_confidence_score(retrieved_chunks, has_knowledge)
    knowledge_gap = detect_knowledge_gap(retrieved_chunks, has_knowledge)
    quality = assess_retrieval_quality(retrieved_chunks)

    pipeline.diagnostics = {
        "confidence": confidence,
        "knowledge_gap": knowledge_gap,
        "retrieval_quality": quality,
    }

    # Format sources
    sources = [
        {
            "content": chunk["content"][:200] + "..." if len(chunk["content"]) > 200 else chunk["content"],
            "source": chunk["source"],
            "score": chunk["score"],
            "retrieval_type": chunk.get("retrieval_type", "dense"),
        }
        for chunk in retrieved_chunks
    ]

    # Yield metadata first (everything the frontend needs before streaming starts)
    yield {
        "type": "metadata",
        "has_knowledge": has_knowledge,
        "sources": sources,
        "model": MODEL_NAME,
        "confidence": confidence,
        "diagnostics": knowledge_gap,
        "retrieval_quality": quality,
        "metrics": pipeline.to_dict() if PIPELINE_METRICS_ENABLED else None,
        "context_budget": budget.to_dict() if PIPELINE_METRICS_ENABLED else None,
    }

    # ── Stream content tokens ────────────────────────────────
    gen_start = time.perf_counter()
    token_generator = get_llm_response_stream(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        conversation_history=conversation_history,
    )

    for token in token_generator:
        yield {
            "type": "content",
            "delta": token,
        }

    pipeline.generation_ms = (time.perf_counter() - gen_start) * 1000
    pipeline.total_ms = (time.perf_counter() - total_start) * 1000
