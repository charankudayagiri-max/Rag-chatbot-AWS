"""
file_loader.py — Local File Content Loader

CONCEPTS DEMONSTRATED
─────────────────────
  Company PDFs, Employee Handbooks, Financial Statements —
         real-world documents that need RAG.
  PDF Chunking, Markdown Chunking.

Supports:
  • PDF files   — extracts text page-by-page using pdfplumber.
  • TXT files   — reads plain text directly.
  • Markdown    — reads .md files as plain text.

The extracted text feeds into the same pipeline:
    File → **Loader** → Cleaner → Chunker → Embeddings → Vector Store
"""

from __future__ import annotations

from pathlib import Path
from app.config import logger


def load_file(file_path: str, content_bytes: bytes | None = None) -> tuple[str, str]:
    """
    Extract text content from a local file.

    Parameters
    ----------
    file_path : str
        Path or filename of the file.
    content_bytes : bytes | None
        Raw file content (for uploaded files). If None, reads from disk.

    Returns
    -------
    tuple[str, str]
        (extracted_text, file_type) where file_type is pdf/txt/md.

    Raises
    ------
    ValueError
        If the file type is not supported or no text could be extracted.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _load_pdf(path, content_bytes), "pdf"
    elif suffix in (".txt", ".text"):
        return _load_text(path, content_bytes), "txt"
    elif suffix in (".md", ".markdown"):
        return _load_text(path, content_bytes), "md"
    else:
        raise ValueError(
            f"Unsupported file type: '{suffix}'. "
            f"Supported: .pdf, .txt, .md"
        )


def _load_pdf(path: Path, content_bytes: bytes | None = None) -> str:
    """
    Extract text from a PDF file using pdfplumber.

    Extracts text page-by-page and joins with double newlines
    to preserve page structure for downstream chunking.
    """
    import pdfplumber
    import io

    logger.info("📄 Loading PDF: %s", path.name)

    pages_text: list[str] = []

    if content_bytes:
        pdf_file = io.BytesIO(content_bytes)
    else:
        pdf_file = str(path)

    with pdfplumber.open(pdf_file) as pdf:
        total_pages = len(pdf.pages)
        logger.info("📄 PDF has %d pages", total_pages)

        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and text.strip():
                pages_text.append(text.strip())

    if not pages_text:
        raise ValueError(
            f"Could not extract text from PDF: {path.name}. "
            "The PDF may be image-based (scanned) or encrypted."
        )

    extracted = "\n\n".join(pages_text)
    logger.info(
        "✅ Extracted %d characters from %d pages of %s",
        len(extracted), len(pages_text), path.name,
    )
    return extracted


def _load_text(path: Path, content_bytes: bytes | None = None) -> str:
    """Load a plain text or markdown file."""
    logger.info("📄 Loading text file: %s", path.name)

    if content_bytes:
        # Try common encodings
        for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                text = content_bytes.decode(encoding)
                break
            except (UnicodeDecodeError, LookupError):
                continue
        else:
            raise ValueError(f"Could not decode file: {path.name}")
    else:
        text = path.read_text(encoding="utf-8")

    if not text or len(text.strip()) < 10:
        raise ValueError(f"File is empty or too short: {path.name}")

    logger.info("✅ Loaded %d characters from %s", len(text), path.name)
    return text


def get_supported_extensions() -> list[str]:
    """Return list of supported file extensions."""
    return [".pdf", ".txt", ".text", ".md", ".markdown"]
