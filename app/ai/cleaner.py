"""
cleaner.py — Text Normaliser

Responsibility: Clean and normalise extracted text.
Nothing else.

Pipeline stage 3:
    Loader → Parser → **Cleaner** → Chunker → Embeddings → Vector Store

Operations:
  • Collapse multiple whitespace/newlines
  • Remove non-printable characters
  • Strip leading/trailing whitespace
  • Remove very short lines (< 10 chars)
"""

import re
from app.config import logger


def clean_text(text: str) -> str:
    """
    Normalise extracted text for chunking.

    Parameters
    ----------
    text : str
        The raw extracted text from the parser.

    Returns
    -------
    str
        Cleaned, normalised text ready for chunking.
    """
    logger.info("🧹 Cleaning text (%d chars)", len(text))

    # Remove non-printable / control characters (keep newlines and tabs)
    text = re.sub(r"[^\S\n\t]+", " ", text)

    # Collapse 3+ consecutive newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove lines that are too short to be meaningful
    lines = text.split("\n")
    cleaned_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        # Keep headings (start with #) and lines >= 10 chars
        if stripped.startswith("#") or len(stripped) >= 10:
            cleaned_lines.append(stripped)

    cleaned = "\n".join(cleaned_lines)

    # Final strip
    cleaned = cleaned.strip()

    logger.info("✅ Cleaned to %d characters", len(cleaned))
    return cleaned
