"""
main.py — FastAPI Application Entry Point

Creates the FastAPI app, includes the API router, and serves the frontend.

TO RUN:
    uvicorn app.main:app --reload

Then open:
    http://127.0.0.1:8000        — Chat UI
    http://127.0.0.1:8000/docs   — Swagger API docs
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from app.api import router
from app.config import TEMPLATE_DIR, logger

# ── Create the FastAPI Application ────────────────────────────────────────
app = FastAPI(
    title="RAG Chatbot API",
    description=(
        "A production-ready RAG (Retrieval-Augmented Generation) chatbot "
        "built from scratch without LangChain.\n\n"
        "**Features:**\n"
        "- Learn from website URLs\n"
        "- Answer questions using ingested knowledge\n"
        "- Conversation memory with session management\n"
        "- Source citations for every answer\n"
        "- Powered by Groq LLM and local embeddings\n"
    ),
    version="1.0.0",
)

# ── Include API Routes ───────────────────────────────────────────────────
app.include_router(router)

@app.on_event("startup")
async def startup_event():
    from app.database import init_db
    init_db()

logger.info("🚀 RAG Chatbot API is ready")


# ── Serve the Chat UI ────────────────────────────────────────────────────
@app.get(
    "/",
    response_class=HTMLResponse,
    summary="Chat UI",
    description="Serves the interactive RAG chatbot interface.",
    include_in_schema=False,
)
async def chat_ui() -> HTMLResponse:
    """Serve the chat frontend."""
    html_path = TEMPLATE_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
