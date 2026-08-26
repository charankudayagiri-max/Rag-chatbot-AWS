"""
diagnostics.py — RAG Diagnostic Module

CONCEPTS DEMONSTRATED
─────────────────────
  Why RAG Exists, Hallucinations, Knowledge Cutoff,
         Context Window, Static vs Dynamic Knowledge.
  Failure Cases, Limitations.

This module provides functions to assess the quality of RAG responses
and detect common failure modes:
  • Low retrieval confidence (chunks aren't relevant enough).
  • Knowledge gaps (query is outside the knowledge base scope).
  • Potential hallucination risk (low-quality context may mislead).
"""

from __future__ import annotations

from app.config import SCORE_THRESHOLD, logger


def assess_retrieval_quality(
    retrieved_chunks: list[dict],
    threshold: float | None = None,
) -> dict:
    """
    Evaluate whether retrieved chunks are relevant enough to answer.

    Metrics:
      • avg_score       — average similarity score across chunks.
      • max_score       — best matching chunk score.
      • min_score       — worst matching chunk score.
      • above_threshold — how many chunks exceed the quality threshold.
      • quality_level   — human-readable assessment (high/medium/low/none).

    Parameters
    ----------
    retrieved_chunks : list[dict]
        Chunks with 'score' field from the retriever.
    threshold : float | None
        Quality threshold. Defaults to SCORE_THRESHOLD.

    Returns
    -------
    dict
        Assessment metrics.
    """
    t = threshold or SCORE_THRESHOLD

    if not retrieved_chunks:
        return {
            "avg_score": 0.0,
            "max_score": 0.0,
            "min_score": 0.0,
            "above_threshold": 0,
            "total_chunks": 0,
            "quality_level": "none",
            "confidence": 0,
        }

    scores = [c.get("score", 0) for c in retrieved_chunks]
    avg = sum(scores) / len(scores)
    max_s = max(scores)
    min_s = min(scores)
    above = sum(1 for s in scores if s >= t)

    # Determine quality level
    if avg >= 0.7 and above == len(scores):
        quality = "high"
        confidence = 5
    elif avg >= 0.5 and above >= len(scores) * 0.5:
        quality = "medium"
        confidence = 3
    elif max_s >= 0.4:
        quality = "low"
        confidence = 2
    else:
        quality = "very_low"
        confidence = 1

    return {
        "avg_score": round(avg, 4),
        "max_score": round(max_s, 4),
        "min_score": round(min_s, 4),
        "above_threshold": above,
        "total_chunks": len(scores),
        "quality_level": quality,
        "confidence": confidence,
    }


def detect_knowledge_gap(
    retrieved_chunks: list[dict],
    has_knowledge: bool,
) -> dict:
    """
    Detect whether the query falls outside the knowledge base scope.

    Knowledge gap scenarios:
      1. No knowledge loaded at all → suggest ingesting content.
      2. Knowledge exists but no relevant chunks found → topic mismatch.
      3. Low-quality matches → partial gap.

    Returns
    -------
    dict
        Gap detection results with type, message, and suggestions.
    """
    if not has_knowledge:
        return {
            "has_gap": True,
            "gap_type": "no_knowledge",
            "message": (
                "No knowledge has been loaded yet. "
                "The response is based on the model's general training data, "
                "which may be outdated or inaccurate for your specific topic."
            ),
            "suggestions": [
                "Paste a website URL in the sidebar to load knowledge.",
                "Upload a PDF or text file with relevant content.",
            ],
            "hallucination_risk": "high",
        }

    if not retrieved_chunks:
        return {
            "has_gap": True,
            "gap_type": "topic_mismatch",
            "message": (
                "Knowledge is loaded, but none of it matches your question. "
                "The model may answer from its general training, which could "
                "be inaccurate for this specific topic."
            ),
            "suggestions": [
                "Try rephrasing your question.",
                "Load additional sources covering this topic.",
            ],
            "hallucination_risk": "high",
        }

    quality = assess_retrieval_quality(retrieved_chunks)

    if quality["quality_level"] in ("low", "very_low"):
        return {
            "has_gap": True,
            "gap_type": "partial_match",
            "message": (
                f"Found {quality['total_chunks']} related chunks, but relevance "
                f"is low (avg score: {quality['avg_score']:.1%}). "
                f"The answer may not fully address your question."
            ),
            "suggestions": [
                "Load more sources on this specific topic.",
                "Try asking a more specific question.",
            ],
            "hallucination_risk": "medium",
        }

    return {
        "has_gap": False,
        "gap_type": None,
        "message": (
            f"Good match: {quality['above_threshold']} of "
            f"{quality['total_chunks']} chunks are highly relevant "
            f"(avg score: {quality['avg_score']:.1%})."
        ),
        "suggestions": [],
        "hallucination_risk": "low",
    }


def compute_confidence_score(
    retrieved_chunks: list[dict],
    has_knowledge: bool,
) -> int:
    """
    Compute an overall confidence score (1-5) for the RAG response.

    Scale:
      5 — High confidence: multiple highly relevant sources.
      4 — Good confidence: relevant sources found.
      3 — Medium: some relevant context, may have gaps.
      2 — Low: weak matches, higher hallucination risk.
      1 — Very low: no knowledge or no relevant chunks.

    Returns
    -------
    int
        Confidence score from 1 to 5.
    """
    if not has_knowledge or not retrieved_chunks:
        return 1

    quality = assess_retrieval_quality(retrieved_chunks)
    return quality["confidence"]
