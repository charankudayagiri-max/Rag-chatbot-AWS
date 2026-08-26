"""
models.py — Pydantic Data Models

CONCEPTS DEMONSTRATED
─────────────────────
  JSON Requests, JSON Responses — Pydantic validates and
         documents every payload shape.
  Pipeline metrics, retrieval metrics, context budget models.

Defines the shape of every JSON payload that flows in and out of the API.
"""

from pydantic import BaseModel, Field, field_validator

class ChatRequest(BaseModel):
    """JSON body sent to POST /chat."""

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="The user's question or instruction.",
        examples=["What does this website say about pricing?"],
    )

    session_id: str | None = Field(
        default=None,
        description=(
            "Session ID for conversation history. "
            "Omit on first message to create a new session."
        ),
    )

    @field_validator('prompt')
    @classmethod
    def validate_prompt_not_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Prompt cannot be empty or whitespace only.")
        return value

class SourceInfo(BaseModel):
    """A single retrieved knowledge chunk shown to the user."""

    content: str = Field(..., description="The text of the retrieved chunk.")
    source: str = Field(..., description="The URL this chunk came from.")
    score: float = Field(..., description="Similarity score (higher = more similar).")
    retrieval_type: str = Field(
        default="dense",
        description="How this chunk was retrieved: dense, sparse, or hybrid.",
    )

class ChatResponse(BaseModel):
    """JSON body returned by POST /chat."""

    response: str = Field(..., description="The AI-generated answer.")
    model: str = Field(..., description="The LLM model used.")
    response_time: float = Field(..., description="Time taken in seconds.")
    session_id: str = Field(..., description="Session ID for this conversation.")
    sources: list[SourceInfo] = Field(
        default_factory=list,
        description="Knowledge chunks used to answer (RAG sources).",
    )
    has_knowledge: bool = Field(
        default=False,
        description="Whether the vector store has any ingested knowledge.",
    )
    confidence: int = Field(
        default=1,
        description="Confidence score (1-5) based on retrieval quality.",
    )

# Learn (Knowledge Ingestion) Models

class LearnRequest(BaseModel):
    """JSON body sent to POST /learn."""

    url: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="The website URL to learn from.",
        examples=["https://example.com/about"],
    )

    @field_validator("url")
    @classmethod
    def validate_url_format(cls, value: str) -> str:
        value = value.strip()
        if not value.startswith(("http://", "https://")):
            raise ValueError(
                "URL must start with http:// or https://"
            )
        return value

class LearnResponse(BaseModel):
    """JSON body returned by POST /learn."""

    task_id: str = Field(..., description="The asynchronous task tracking ID.")
    status: str = Field(..., description="Task status (processing, completed, failed).")
    url: str = Field(..., description="The URL submitted for learning.")
    message: str = Field(..., description="Confirmation details.")

class FileUploadResponse(BaseModel):
    """JSON body returned by POST /upload."""

    task_id: str = Field(..., description="The asynchronous task tracking ID.")
    status: str = Field(..., description="Task status.")
    filename: str = Field(..., description="The uploaded filename.")
    file_type: str = Field(..., description="Detected file type (pdf/txt/md).")
    message: str = Field(..., description="Confirmation details.")

class TaskStatusResponse(BaseModel):
    """JSON body returned by GET /tasks/{task_id}."""

    task_id: str = Field(..., description="Unique task identifier.")
    source: str = Field(..., description="Source resource URL or filepath.")
    status: str = Field(..., description="Current status of background run.")
    chunks_created: int = Field(0, description="Total chunks ingested.")
    error_message: str | None = Field(None, description="Detailed error logs if task failed.")
    created_at: float | None = Field(None, description="Task creation epoch timestamp.")
    completed_at: float | None = Field(None, description="Task completion epoch timestamp.")
    metrics: dict | None = Field(None, description="Timing and sizing metrics if completed.")

# Session Models
class ClearRequest(BaseModel):
    """JSON body sent to POST /clear."""

    session_id: str = Field(
        ..., description="The session ID to clear."
    )