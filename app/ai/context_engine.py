"""
context_engine.py — Context Engineering Module

CONCEPTS DEMONSTRATED
─────────────────────
  Prompt vs Context, Dynamic Context, Context Assembly,
         Context Ordering, Context Filtering, Context Budget,
         Context Compression.

  Bridge Concept: "Everything that reaches the model is Context."

This module is the single place where all context components — system
prompt, RAG sources, conversation history, and user query — are
assembled into the final messages array that the LLM receives.

ARCHITECTURE
────────────
  1. Assemble  — gather all context components.
  2. Filter    — remove duplicates, low-relevance sources.
  3. Order     — system → RAG → history → user (optimal order).
  4. Budget    — allocate tokens proportionally across categories.
  5. Compress  — summarize older conversation to save tokens.
  6. Build     — produce the final messages list.
"""

from __future__ import annotations

from app.chat_history import count_tokens, count_message_tokens
from app.config import (
    CONTEXT_BUDGET_TOKENS,
    CONTEXT_RAG_RATIO,
    CONTEXT_HISTORY_RATIO,
    CONTEXT_SYSTEM_RATIO,
    CONTEXT_COMPRESSION_ENABLED,
    SCORE_THRESHOLD,
    logger,
)


# ═══════════════════════════════════════════════════════════════════════════
# Context Budget Allocation
# ═══════════════════════════════════════════════════════════════════════════

class ContextBudget:
    """
    Token budget allocator for context components.

    Distributes a total token budget across three categories:
      • System prompt (instructions + RAG context)
      • Conversation history
      • User query (reserved, never trimmed)

    The ratios are configurable via environment variables.
    """

    def __init__(self, total_budget: int | None = None) -> None:
        self.total = total_budget or CONTEXT_BUDGET_TOKENS
        self.system_budget = int(self.total * CONTEXT_SYSTEM_RATIO)
        self.rag_budget = int(self.total * CONTEXT_RAG_RATIO)
        self.history_budget = int(self.total * CONTEXT_HISTORY_RATIO)

        # Track actual usage
        self.system_used = 0
        self.rag_used = 0
        self.history_used = 0
        self.user_used = 0

    @property
    def total_used(self) -> int:
        return self.system_used + self.rag_used + self.history_used + self.user_used

    @property
    def remaining(self) -> int:
        return max(0, self.total - self.total_used)

    def to_dict(self) -> dict:
        """Return budget info for API response / metrics."""
        return {
            "total_budget": self.total,
            "system": {"budget": self.system_budget, "used": self.system_used},
            "rag": {"budget": self.rag_budget, "used": self.rag_used},
            "history": {"budget": self.history_budget, "used": self.history_used},
            "user": {"used": self.user_used},
            "total_used": self.total_used,
            "remaining": self.remaining,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Context Filtering
# ═══════════════════════════════════════════════════════════════════════════

def filter_chunks(
    chunks: list[dict],
    score_threshold: float | None = None,
) -> list[dict]:
    """
    Filter retrieved chunks to remove low-quality and duplicate content.

    Filtering steps:
      1. Remove chunks below the score threshold.
      2. Remove near-duplicate chunks (same content from different runs).

    Parameters
    ----------
    chunks : list[dict]
        Retrieved chunks with keys: content, source, score.
    score_threshold : float | None
        Minimum similarity score. Defaults to config SCORE_THRESHOLD.

    Returns
    -------
    list[dict]
        Filtered chunks, sorted by relevance (highest first).
    """
    threshold = score_threshold if score_threshold is not None else SCORE_THRESHOLD

    # Step 1: Score threshold
    filtered = [c for c in chunks if c.get("score", 0) >= threshold]

    if len(filtered) < len(chunks):
        logger.info(
            "🔍 Context filter: %d → %d chunks (threshold=%.2f)",
            len(chunks), len(filtered), threshold,
        )

    # Step 2: Deduplicate by content similarity (exact match for now)
    seen_content: set[str] = set()
    deduped: list[dict] = []
    for chunk in filtered:
        # Use first 200 chars as dedup key
        key = chunk["content"][:200].strip().lower()
        if key not in seen_content:
            seen_content.add(key)
            deduped.append(chunk)

    if len(deduped) < len(filtered):
        logger.info(
            "🔍 Context dedup: %d → %d chunks",
            len(filtered), len(deduped),
        )

    # Sort by score descending (best first)
    deduped.sort(key=lambda c: c.get("score", 0), reverse=True)
    return deduped


# ═══════════════════════════════════════════════════════════════════════════
# Context Compression
# ═══════════════════════════════════════════════════════════════════════════

def compress_history(
    messages: list[dict[str, str]],
    token_budget: int,
) -> list[dict[str, str]]:
    """
    Compress conversation history to fit within a token budget.

    Compression strategy:
      1. Keep the most recent messages (they're most relevant).
      2. For older messages, truncate content to key sentences.
      3. If still over budget, drop oldest messages entirely.

    Parameters
    ----------
    messages : list[dict]
        The full conversation history.
    token_budget : int
        Maximum tokens allowed for history.

    Returns
    -------
    list[dict]
        Compressed history within the token budget.
    """
    if not messages:
        return []

    current_tokens = count_message_tokens(messages)

    # Already within budget
    if current_tokens <= token_budget:
        return messages

    if not CONTEXT_COMPRESSION_ENABLED:
        # Just trim from the front (simple sliding window)
        return _sliding_window_trim(messages, token_budget)

    logger.info(
        "📦 Compressing history: %d tokens → %d budget",
        current_tokens, token_budget,
    )

    result = messages.copy()

    # Phase 1: Truncate older messages (keep last 4 messages intact)
    keep_recent = min(4, len(result))
    recent = result[-keep_recent:]
    older = result[:-keep_recent]

    compressed_older: list[dict[str, str]] = []
    for msg in older:
        content = msg["content"]
        # Truncate to first 100 chars + "..."
        if count_tokens(content) > 50:
            sentences = content.split(". ")
            truncated = ". ".join(sentences[:2])
            if len(truncated) < len(content):
                truncated += "..."
            compressed_older.append({"role": msg["role"], "content": truncated})
        else:
            compressed_older.append(msg)

    result = compressed_older + recent

    # Phase 2: If still over budget, drop oldest messages
    result = _sliding_window_trim(result, token_budget)

    logger.info(
        "📦 History compressed: %d → %d tokens (%d messages)",
        current_tokens, count_message_tokens(result), len(result),
    )

    return result


def _sliding_window_trim(
    messages: list[dict[str, str]],
    token_budget: int,
) -> list[dict[str, str]]:
    """Keep the most recent messages that fit within the token budget."""
    result = messages.copy()
    while len(result) > 0 and count_message_tokens(result) > token_budget:
        result.pop(0)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Context Assembly
# ═══════════════════════════════════════════════════════════════════════════

def assemble_context(
    system_prompt: str,
    user_prompt: str,
    retrieved_chunks: list[dict],
    conversation_history: list[dict[str, str]] | None = None,
) -> tuple[list[dict[str, str]], ContextBudget]:
    """
    Assemble all context into the final messages array for the LLM.

    This is THE central function — the bridge concept:
    "Everything that reaches the model is Context."

    Context Ordering:
      1. System prompt (base instructions)
      2. RAG context (injected into system prompt)
      3. Conversation history (compressed if needed)
      4. User query (never trimmed)

    Parameters
    ----------
    system_prompt : str
        The base system prompt (with {context} placeholder or plain).
    user_prompt : str
        The current user question.
    retrieved_chunks : list[dict]
        Filtered retrieved chunks from the vector store.
    conversation_history : list[dict] | None
        Previous conversation messages.

    Returns
    -------
    tuple[list[dict], ContextBudget]
        (messages_array, budget_info)
    """
    budget = ContextBudget()

    # ── 1. Account for user query (never trimmed) ────────────
    budget.user_used = count_tokens(user_prompt) + 4

    # ── 2. Account for system prompt ─────────────────────────
    budget.system_used = count_tokens(system_prompt) + 4

    # ── 3. Apply RAG context budget ──────────────────────────
    # Filter chunks that fit within the RAG token budget
    rag_chunks = []
    rag_tokens = 0
    for chunk in retrieved_chunks:
        chunk_tokens = count_tokens(chunk["content"])
        if rag_tokens + chunk_tokens <= budget.rag_budget:
            rag_chunks.append(chunk)
            rag_tokens += chunk_tokens
        else:
            logger.info(
                "📦 RAG budget exhausted: keeping %d of %d chunks",
                len(rag_chunks), len(retrieved_chunks),
            )
            break
    budget.rag_used = rag_tokens

    # ── 4. Compress conversation history ─────────────────────
    history = conversation_history or []
    compressed_history = compress_history(history, budget.history_budget)
    budget.history_used = count_message_tokens(compressed_history)

    # ── 5. Build final messages array ──
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
    ]

    # Add compressed history
    if compressed_history:
        messages.extend(compressed_history)

    # User query always goes last
    messages.append({"role": "user", "content": user_prompt})

    logger.info(
        "📋 Context assembled | messages=%d | budget: %s",
        len(messages),
        f"system={budget.system_used} rag={budget.rag_used} "
        f"history={budget.history_used} user={budget.user_used} "
        f"total={budget.total_used}/{budget.total}",
    )

    return messages, budget
