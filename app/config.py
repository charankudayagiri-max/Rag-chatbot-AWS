"""
config.py — Centralised Application Configuration

This module loads environment variables and provides typed configuration
constants for every layer of the application.

WHY THIS FILE EXISTS
────────────────────
Scattering os.getenv() calls across many files makes it nearly impossible
to know what the application depends on.  Centralising configuration here
means every module simply imports what it needs, and the defaults plus
validation live in one place.

CONCEPTS DEMONSTRATED
─────────────────────
  API Keys, Environment Variables, Provider Configuration
  Conversation History Settings
  Context Budget Parameters
  Chunking Strategy Selection
  Retrieval Strategy Parameters
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env file ────────────────────────────────────────────────────────
load_dotenv()

# ── Fix Windows SSL issue ────────────────────────────────────────────────
os.environ.pop("SSLKEYLOGFILE", None)

# ── Project Paths ─────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).parent.parent
TEMPLATE_DIR: Path = Path(__file__).parent / "templates"
UPLOAD_DIR: Path = PROJECT_ROOT / "data" / "uploads"

# ── LLM Provider ────────────────────
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq")
GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL: str | None = os.getenv("OPENAI_BASE_URL")
MODEL_NAME: str = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")

# ── LLM Generation Parameters ────────────────────────────────────────────
TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))
TOP_P: float = float(os.getenv("TOP_P", "1.0"))
MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "1024"))

# ── Embedding Settings ───────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_CACHE_ENABLED: bool = os.getenv("EMBEDDING_CACHE_ENABLED", "true").lower() == "true"
EMBEDDING_CACHE_SIZE: int = int(os.getenv("EMBEDDING_CACHE_SIZE", "1000"))

# ── Chunking Settings ────────────────────────────
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
CHUNKING_STRATEGY: str = os.getenv("CHUNKING_STRATEGY", "recursive")
# Valid: recursive, sentence, paragraph, semantic

# ── Retrieval Settings ────────────────────────
TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "5"))
SCORE_THRESHOLD: float = float(os.getenv("SCORE_THRESHOLD", "0.3"))
RETRIEVAL_STRATEGY: str = os.getenv("RETRIEVAL_STRATEGY", "hybrid")
# Valid: dense, sparse, hybrid
MMR_ENABLED: bool = os.getenv("MMR_ENABLED", "true").lower() == "true"
MMR_LAMBDA: float = float(os.getenv("MMR_LAMBDA", "0.7"))
BM25_WEIGHT: float = float(os.getenv("BM25_WEIGHT", "0.3"))
DENSE_WEIGHT: float = float(os.getenv("DENSE_WEIGHT", "0.7"))

# ── Context Engineering ──────────────────────────────────────────
CONTEXT_BUDGET_TOKENS: int = int(os.getenv("CONTEXT_BUDGET_TOKENS", "6000"))
CONTEXT_RAG_RATIO: float = float(os.getenv("CONTEXT_RAG_RATIO", "0.4"))
CONTEXT_HISTORY_RATIO: float = float(os.getenv("CONTEXT_HISTORY_RATIO", "0.3"))
CONTEXT_SYSTEM_RATIO: float = float(os.getenv("CONTEXT_SYSTEM_RATIO", "0.3"))
CONTEXT_COMPRESSION_ENABLED: bool = os.getenv(
    "CONTEXT_COMPRESSION_ENABLED", "true"
).lower() == "true"

# ── Conversation History ─────────────────────────────────────────
MAX_HISTORY_LENGTH: int = int(os.getenv("MAX_HISTORY_LENGTH", "50"))
MAX_HISTORY_TOKENS: int = int(os.getenv("MAX_HISTORY_TOKENS", "2000"))

# ── Vector Database ──────────────────────────────────────────────
CHROMA_PERSIST_DIR: str = os.getenv(
    "CHROMA_PERSIST_DIR",
    str(PROJECT_ROOT / "data" / "chroma"),
)

# ── Relational Database (SQLAlchemy / PostgreSQL) ───────────────────────
DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "mysql+pymysql://root:@localhost:3306/rag_chatbot",
)

# ── Pipeline Metrics ─────────────────────────────────────────────
PIPELINE_METRICS_ENABLED: bool = os.getenv(
    "PIPELINE_METRICS_ENABLED", "true"
).lower() == "true"

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger: logging.Logger = logging.getLogger("rag-chatbot")
