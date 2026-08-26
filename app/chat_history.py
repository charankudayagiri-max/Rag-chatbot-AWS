"""
chat_history.py — Session-Based Conversation Memory (MySQL Engine)

CONCEPTS DEMONSTRATED
─────────────────────
  Multi-turn Conversation, Stateless APIs → Stateful Applications,
         Chat History, Sliding Window, Message Pruning, Session Management.
  Context Budget (token-aware trimming).

LLMs are stateless — they have zero memory between API calls.
This module stores conversation history persistently in a relational
database using SQLAlchemy, so we can resend the full context with each request.

PRODUCTION FEATURES
───────────────────
  • PostgreSQL Persistence — survives restarts, scales horizontally.
  • Token-aware pruning    — trims by token count, not just message count.
  • Session metadata       — tracks creation time, message count, tokens.
  • Sliding window         — keeps the most recent messages within budget.
  • Self-contained TX      — uses SQLAlchemy context managers to isolate DB logic.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from contextlib import contextmanager
import tiktoken

from app.database import SessionLocal
from app.db_models import ChatSession, ChatMessage
from app.config import MAX_HISTORY_LENGTH, MAX_HISTORY_TOKENS, logger

# ── Token Counter ──────────────────────────────────────────────────

try:
    _tokenizer = tiktoken.get_encoding("cl100k_base")
except Exception:
    _tokenizer = None
    logger.warning("⚠️ tiktoken not available — falling back to word-count estimation")


def count_tokens(text: str) -> int:
    """
    Count the number of tokens in a text string.

    Uses tiktoken for accuracy, falls back to word-count / 0.75 estimate.
    """
    if _tokenizer is not None:
        return len(_tokenizer.encode(text))
    return int(len(text.split()) / 0.75)


def count_message_tokens(messages: list[dict[str, str]]) -> int:
    """Count total tokens across a list of messages."""
    total = 0
    for msg in messages:
        total += count_tokens(msg.get("content", "")) + 4
    return total


# ── Database Transaction Helper ──────────────────────────────────────────

@contextmanager
def get_db_session():
    """
    Provide a transactional scope around a series of operations.
    Automatically commits on success, rollbacks on error, and closes the session.
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as exc:
        db.rollback()
        logger.exception("Database transaction failed:")
        raise exc
    finally:
        db.close()


# ── Database-Backed Session Store ──────────────

def create_session() -> str:
    """
    Create a new persistent session in the database and return its UUID.
    """
    session_id = str(uuid.uuid4())
    with get_db_session() as db:
        session = ChatSession(id=session_id)
        db.add(session)
        logger.info("📝 Created new DB session: %s", session_id)
    return session_id


def get_history(session_id: str) -> list[dict[str, str]]:
    """
    Return the conversation history for a session sorted by message creation.
    """
    with get_db_session() as db:
        messages = (
            db.query(ChatMessage)
            .filter(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.id.asc())
            .all()
        )
        return [{"role": msg.role, "content": msg.content} for msg in messages]


def add_message(session_id: str, role: str, content: str) -> None:
    """
    Append a message to the database and apply intelligent pruning.

    Pruning Strategy:
      1. First, check message count against MAX_HISTORY_LENGTH.
      2. Then, check token count against MAX_HISTORY_TOKENS.
      3. Always keep at least the most recent user-assistant pair.
      4. Prune oldest messages first (sliding window).
    """
    token_count = count_tokens(content)

    with get_db_session() as db:
        # Retrieve or create session metadata
        session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if session is None:
            session = ChatSession(id=session_id)
            db.add(session)
            db.flush()

        # Add message
        msg = ChatMessage(
            session_id=session_id,
            role=role,
            content=content,
            token_count=token_count,
        )
        db.add(msg)

        # Update metadata counters
        session.total_messages_ever += 1
        session.total_tokens_ever += token_count
        session.last_activity = datetime.utcnow()
        db.flush()

        # Prune older messages in this transaction block
        _prune_session_db(db, session_id)


def clear_history(session_id: str) -> bool:
    """Clear history for a session. Returns True if the session existed."""
    with get_db_session() as db:
        session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if session is None:
            return False

        # Cascade delete is handled at ORM/DB foreign key level,
        # but let's delete messages explicitly for safety.
        db.query(ChatMessage).filter(ChatMessage.session_id == session_id).delete()
        logger.info("🗑️ Cleared DB history for session: %s", session_id)
        return True


def list_sessions() -> list[dict]:
    """
    List all active persistent sessions.
    """
    with get_db_session() as db:
        sessions = (
            db.query(ChatSession)
            .order_by(ChatSession.last_activity.desc())
            .all()
        )
        return [s.to_dict() for s in sessions]


def get_session_info(session_id: str) -> dict | None:
    """Return metadata for a specific session, or None if not found."""
    with get_db_session() as db:
        session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        return session.to_dict() if session else None


# ── Pruning DB Logic ────────────────────────────────

def _prune_session_db(db, session_id: str) -> None:
    """
    Apply sliding window pruning inside the database session.

    Two-phase pruning:
      Phase 1: Message count limit (coarse).
      Phase 2: Token budget limit (fine-grained, token budget).

    Preserves messages in pairs (user + assistant) to maintain coherent context.
    """
    # Fetch all current messages
    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.id.asc())
        .all()
    )

    if not messages:
        return

    # Phase 1: Message count limit (coarse)
    max_items = MAX_HISTORY_LENGTH * 2  # pairs
    if len(messages) > max_items:
        to_remove_count = len(messages) - max_items
        # Identify oldest N message IDs
        ids_to_delete = [msg.id for msg in messages[:to_remove_count]]

        db.query(ChatMessage).filter(ChatMessage.id.in_(ids_to_delete)).delete(
            synchronize_session=False
        )

        # Refetch remaining messages for Phase 2
        messages = messages[to_remove_count:]
        logger.info(
            "✂️ [DB Count] Pruned %d messages from session %s",
            to_remove_count, session_id,
        )

    # Phase 2: Token budget limit (fine-grained)
    current_tokens = sum((msg.token_count or 0) + 4 for msg in messages)
    if current_tokens > MAX_HISTORY_TOKENS:
        ids_to_delete = []
        token_sum = current_tokens

        idx = 0
        # Loop to find how many pairs need to be deleted from the start
        while len(messages) - idx > 2 and token_sum > MAX_HISTORY_TOKENS:
            # We delete a pair (user + assistant)
            m1 = messages[idx]
            m2 = messages[idx + 1]

            ids_to_delete.extend([m1.id, m2.id])
            token_sum -= ((m1.token_count or 0) + 4 + (m2.token_count or 0) + 4)
            idx += 2

        if ids_to_delete:
            db.query(ChatMessage).filter(ChatMessage.id.in_(ids_to_delete)).delete(
                synchronize_session=False
            )
            logger.info(
                "✂️ [DB Tokens] Pruned %d messages from session %s (kept %d msgs, token sum %d)",
                len(ids_to_delete), session_id, len(messages) - len(ids_to_delete), token_sum,
            )
