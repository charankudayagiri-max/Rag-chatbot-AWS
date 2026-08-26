"""
db_models.py — SQLAlchemy Entity Schemas

Maps Python classes to PostgreSQL tables for conversation sessions and messages.
"""

from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from app.database import Base


class ChatSession(Base):
    """
    Tracks metadata for a conversation session.
    """
    __tablename__ = "chat_sessions"

    id = Column(String(36), primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_activity = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    total_messages_ever = Column(Integer, default=0)
    total_tokens_ever = Column(Integer, default=0)

    # Cascade deletes to all associated messages
    messages = relationship(
        "ChatMessage",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ChatMessage.id.asc()",
    )

    def to_dict(self) -> dict:
        """Convert metadata to a serializable dictionary."""
        # Calculate current tokens in relational history
        current_tokens = sum(msg.token_count or 0 for msg in self.messages)
        return {
            "session_id": self.id,
            "message_count": len(self.messages),
            "total_messages_ever": self.total_messages_ever,
            "total_tokens_ever": self.total_tokens_ever,
            "current_tokens": current_tokens,
            "created_at": self.created_at.timestamp() if self.created_at else datetime.utcnow().timestamp(),
            "last_activity": self.last_activity.timestamp() if self.last_activity else datetime.utcnow().timestamp(),
        }


class ChatMessage(Base):
    """
    Stores individual messages in a conversation.
    """
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_id = Column(String(36), ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    token_count = Column(Integer, default=0)

    # Reference back to parent session
    session = relationship("ChatSession", back_populates="messages")

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
        }


class IngestionTask(Base):
    """
    Tracks state of background indexing pipelines.
    """
    __tablename__ = "ingestion_tasks"

    id = Column(String(36), primary_key=True, index=True)
    source = Column(String(500), nullable=False)
    status = Column(String(20), default="processing", nullable=False)  # processing, completed, failed
    chunks_created = Column(Integer, default=0, nullable=False)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    metrics = Column(Text, nullable=True)  # JSON-serialized performance metrics

    def to_dict(self) -> dict:
        import json
        metrics_dict = None
        if self.metrics:
            try:
                metrics_dict = json.loads(self.metrics)
            except Exception:
                pass
        return {
            "task_id": self.id,
            "source": self.source,
            "status": self.status,
            "chunks_created": self.chunks_created,
            "error_message": self.error_message,
            "created_at": self.created_at.timestamp() if self.created_at else None,
            "completed_at": self.completed_at.timestamp() if self.completed_at else None,
            "metrics": metrics_dict,
        }
