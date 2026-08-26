"""
prompt_builder.py — RAG Prompt Constructor

CONCEPTS DEMONSTRATED
─────────────────────
  Context Assembly, Context Ordering.
  Prompt Builder stage in the Generation Pipeline.

Pipeline stage 8:
    Retriever → **Prompt Builder** → Context Engine → LLM

This module sits between retrieval and the LLM call.
It takes the raw retrieved chunks and formats them into a structured
context block that the LLM can reason over.
"""

from app.prompts import get_rag_prompt, get_general_prompt
from app.config import logger


def build_prompt(retrieved_chunks: list[dict]) -> str:
    """
    Build the system prompt based on available context.

    If retrieved_chunks is empty (no knowledge), returns the general
    assistant prompt.  Otherwise, formats the RAG prompt with context.

    Context Ordering:
      Sources are ordered by relevance (highest first) with clear
      delimiters and source numbers for citation.

    Parameters
    ----------
    retrieved_chunks : list[dict]
        List of dicts with keys: content, source, score.

    Returns
    -------
    str
        The complete system prompt ready for the LLM.
    """
    if not retrieved_chunks:
        logger.info("📋 No context available — using general prompt")
        return get_general_prompt()

    # Format each chunk with its source for attribution
    context_parts: list[str] = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        score_pct = f"{chunk.get('score', 0):.0%}"
        retrieval_type = chunk.get("retrieval_type", "dense")
        context_parts.append(
            f"--- Source {i}: {chunk['source']} "
            f"(relevance: {score_pct}, type: {retrieval_type}) ---\n"
            f"{chunk['content']}\n"
        )

    context_block = "\n".join(context_parts)

    logger.info(
        "📋 Built RAG prompt with %d sources (%d chars of context)",
        len(retrieved_chunks),
        len(context_block),
    )

    return get_rag_prompt(context_block)
