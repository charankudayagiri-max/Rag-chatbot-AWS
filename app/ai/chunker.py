"""
chunker.py — Multi-Strategy Text Splitter

CONCEPTS DEMONSTRATED
─────────────────────
  Why Chunking, Chunk Size, Chunk Overlap, Sentence Chunking,
         Paragraph Chunking, Recursive Chunking, Semantic Chunking,
         Sliding Window, Metadata, Chunk Evaluation.

Pipeline stage 4:
    Loader → Parser → Cleaner → **Chunker** → Embeddings → Vector Store

WHY CHUNKING?
─────────────
LLMs have limited context windows.  Embedding models work best on
smaller text passages.  Splitting text into chunks lets us:
  1. Create focused embeddings for each passage.
  2. Retrieve only the most relevant passages for a query.
  3. Stay within token limits.

WHY OVERLAP?
────────────
If a sentence spans two chunks without overlap, it gets split and
neither chunk captures the full meaning.  Overlap ensures continuity.

STRATEGIES
──────────
  • recursive   — Split by paragraph → sentence → character (default).
  • sentence    — Split by sentence boundaries.
  • paragraph   — Split by paragraph boundaries (\\n\\n).
  • semantic    — Group sentences by semantic similarity (embedding-based).
"""

from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod

from app.config import CHUNK_SIZE, CHUNK_OVERLAP, CHUNKING_STRATEGY, logger

# ═══════════════════════════════════════════════════════════════════════════
# Chunk Metadata
# ═══════════════════════════════════════════════════════════════════════════

class ChunkMetadata:
    """
    Metadata attached to each text chunk for tracing and evaluation.

    Attributes
    ----------
    chunk_index : int
        Position of this chunk in the sequence.
    total_chunks : int
        Total number of chunks produced.
    char_start : int
        Start character offset in the original text.
    char_end : int
        End character offset in the original text.
    strategy : str
        The chunking strategy that produced this chunk.
    char_count : int
        Number of characters in this chunk.
    """

    def __init__(
        self,
        chunk_index: int,
        total_chunks: int,
        char_start: int,
        char_end: int,
        strategy: str,
    ) -> None:
        self.chunk_index = chunk_index
        self.total_chunks = total_chunks
        self.char_start = char_start
        self.char_end = char_end
        self.strategy = strategy
        self.char_count = char_end - char_start

    def to_dict(self) -> dict:
        return {
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "strategy": self.strategy,
            "char_count": self.char_count,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Chunk Result Container
# ═══════════════════════════════════════════════════════════════════════════

class ChunkResult:
    """A chunk of text with its metadata."""

    def __init__(self, text: str, metadata: ChunkMetadata) -> None:
        self.text = text
        self.metadata = metadata


# ═══════════════════════════════════════════════════════════════════════════
# Abstract Chunking Strategy
# ═══════════════════════════════════════════════════════════════════════════

class ChunkingStrategy(ABC):
    """Abstract base for all chunking strategies."""

    @abstractmethod
    def chunk(
        self, text: str, chunk_size: int, chunk_overlap: int,
    ) -> list[str]:
        """Split text into a list of string chunks."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name for metadata."""
        ...


# ═══════════════════════════════════════════════════════════════════════════
# Recursive Chunking
# ═══════════════════════════════════════════════════════════════════════════

class RecursiveChunker(ChunkingStrategy):
    """
    Split by paragraph → sentence → character boundaries with overlap.

    This is the most commonly used strategy.  It tries to preserve
    natural text boundaries:
      1. First, try to break at paragraph boundaries (\\n\\n).
      2. If no paragraph break found, try sentence boundaries (. ).
      3. As a last resort, split at the exact character position.
    """

    @property
    def name(self) -> str:
        return "recursive"

    def chunk(
        self, text: str, chunk_size: int, chunk_overlap: int,
    ) -> list[str]:
        if not text or not text.strip():
            return []

        if len(text) <= chunk_size:
            return [text]

        chunks: list[str] = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # If not the last chunk, try to break at a boundary
            if end < len(text):
                # Prefer paragraph boundary
                paragraph_break = text.rfind("\n\n", start, end)
                if paragraph_break > start + chunk_size // 2:
                    end = paragraph_break

                # Else try sentence boundary
                elif (sentence_break := text.rfind(". ", start, end)) > start + chunk_size // 2:
                    end = sentence_break + 1  # include the period

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Advance by chunk length minus overlap
            start = start + max((end - start) - chunk_overlap, 1)
            if start >= len(text):
                break

        return chunks


# ═══════════════════════════════════════════════════════════════════════════
# Sentence Chunking
# ═══════════════════════════════════════════════════════════════════════════

class SentenceChunker(ChunkingStrategy):
    """
    Split text into chunks at sentence boundaries.

    Groups consecutive sentences until the chunk_size limit is reached,
    then starts a new chunk with overlap sentences from the previous chunk.
    """

    # Regex to split on sentence-ending punctuation followed by space or newline
    SENTENCE_PATTERN = re.compile(r'(?<=[.!?])\s+')

    @property
    def name(self) -> str:
        return "sentence"

    def chunk(
        self, text: str, chunk_size: int, chunk_overlap: int,
    ) -> list[str]:
        if not text or not text.strip():
            return []

        if len(text) <= chunk_size:
            return [text]

        sentences = self.SENTENCE_PATTERN.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return [text]

        chunks: list[str] = []
        current_chunk_sentences: list[str] = []
        current_length = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            if current_length + sentence_len > chunk_size and current_chunk_sentences:
                # Save current chunk
                chunks.append(" ".join(current_chunk_sentences))

                # Calculate overlap: keep last N sentences that fit in overlap
                overlap_sentences: list[str] = []
                overlap_len = 0
                for s in reversed(current_chunk_sentences):
                    if overlap_len + len(s) <= chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_len += len(s)
                    else:
                        break

                current_chunk_sentences = overlap_sentences
                current_length = sum(len(s) for s in current_chunk_sentences)

            current_chunk_sentences.append(sentence)
            current_length += sentence_len

        # Don't forget the last chunk
        if current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))

        return chunks


# ═══════════════════════════════════════════════════════════════════════════
# Paragraph Chunking
# ═══════════════════════════════════════════════════════════════════════════

class ParagraphChunker(ChunkingStrategy):
    """
    Split text at paragraph boundaries (double newlines).

    Groups consecutive paragraphs until chunk_size is reached.
    If a single paragraph exceeds chunk_size, it falls back to
    recursive chunking for that paragraph.
    """

    @property
    def name(self) -> str:
        return "paragraph"

    def chunk(
        self, text: str, chunk_size: int, chunk_overlap: int,
    ) -> list[str]:
        if not text or not text.strip():
            return []

        if len(text) <= chunk_size:
            return [text]

        paragraphs = text.split("\n\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if not paragraphs:
            return [text]

        chunks: list[str] = []
        current_parts: list[str] = []
        current_length = 0

        for paragraph in paragraphs:
            para_len = len(paragraph)

            # If a single paragraph is too large, use recursive chunker as fallback
            if para_len > chunk_size:
                # Save current chunk first
                if current_parts:
                    chunks.append("\n\n".join(current_parts))
                    current_parts = []
                    current_length = 0

                # Sub-chunk the large paragraph
                recursive = RecursiveChunker()
                sub_chunks = recursive.chunk(paragraph, chunk_size, chunk_overlap)
                chunks.extend(sub_chunks)
                continue

            if current_length + para_len > chunk_size and current_parts:
                chunks.append("\n\n".join(current_parts))

                # Overlap: keep last paragraph if it fits
                if current_parts and len(current_parts[-1]) <= chunk_overlap:
                    current_parts = [current_parts[-1]]
                    current_length = len(current_parts[0])
                else:
                    current_parts = []
                    current_length = 0

            current_parts.append(paragraph)
            current_length += para_len

        if current_parts:
            chunks.append("\n\n".join(current_parts))

        return chunks


# ═══════════════════════════════════════════════════════════════════════════
# Semantic Chunking
# ═══════════════════════════════════════════════════════════════════════════

class SemanticChunker(ChunkingStrategy):
    """
    Group sentences by semantic similarity using embeddings.

    Algorithm:
      1. Split text into sentences.
      2. Embed each sentence.
      3. Compare consecutive sentence embeddings.
      4. When similarity drops below a threshold, start a new chunk.
      5. Merge small chunks with neighbours.

    This is SLOWER than other strategies because it requires
    embedding every sentence, but produces the most semantically
    coherent chunks.
    """

    @property
    def name(self) -> str:
        return "semantic"

    def chunk(
        self, text: str, chunk_size: int, chunk_overlap: int,
    ) -> list[str]:
        if not text or not text.strip():
            return []

        if len(text) <= chunk_size:
            return [text]

        # Split into sentences
        sentence_pattern = re.compile(r'(?<=[.!?])\s+')
        sentences = sentence_pattern.split(text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

        if len(sentences) <= 1:
            return [text]

        # Lazy import to avoid circular dependency
        from app.ai.embeddings import generate_embeddings, cosine_similarity

        # Embed all sentences
        logger.info("🔬 Semantic chunking: embedding %d sentences", len(sentences))
        embeddings = generate_embeddings(sentences)

        # Find breakpoints where consecutive similarity drops
        similarities: list[float] = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)

        # Dynamic threshold: mean - 1 stddev
        if similarities:
            import numpy as np
            mean_sim = float(np.mean(similarities))
            std_sim = float(np.std(similarities))
            threshold = max(mean_sim - std_sim, 0.3)
        else:
            threshold = 0.5

        # Group sentences into chunks at breakpoints
        groups: list[list[str]] = [[sentences[0]]]
        for i, sentence in enumerate(sentences[1:], start=0):
            if i < len(similarities) and similarities[i] < threshold:
                groups.append([sentence])
            else:
                groups[-1].append(sentence)

        # Merge small groups and enforce chunk_size
        chunks: list[str] = []
        current = ""
        for group in groups:
            group_text = " ".join(group)
            if len(current) + len(group_text) <= chunk_size:
                current = (current + " " + group_text).strip()
            else:
                if current:
                    chunks.append(current)
                current = group_text

        if current:
            chunks.append(current)

        return chunks


# ═══════════════════════════════════════════════════════════════════════════
# Strategy Factory
# ═══════════════════════════════════════════════════════════════════════════

_STRATEGIES: dict[str, type[ChunkingStrategy]] = {
    "recursive": RecursiveChunker,
    "sentence": SentenceChunker,
    "paragraph": ParagraphChunker,
    "semantic": SemanticChunker,
}


def get_chunker(strategy: str | None = None) -> ChunkingStrategy:
    """
    Get a chunking strategy instance by name.

    Parameters
    ----------
    strategy : str | None
        Strategy name. Defaults to config CHUNKING_STRATEGY.
        Valid: recursive, sentence, paragraph, semantic.

    Returns
    -------
    ChunkingStrategy
        The requested strategy instance.
    """
    name = strategy or CHUNKING_STRATEGY
    if name not in _STRATEGIES:
        raise ValueError(
            f"Unknown chunking strategy: '{name}'. "
            f"Valid: {', '.join(_STRATEGIES.keys())}"
        )
    return _STRATEGIES[name]()


# ═══════════════════════════════════════════════════════════════════════════
# Chunk Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_chunks(chunks: list[str]) -> dict:
    """
    Compute quality metrics for a set of chunks.

    Metrics:
      • count         — number of chunks.
      • avg_size      — average character count.
      • min_size      — smallest chunk.
      • max_size      — largest chunk.
      • size_variance — standard deviation of sizes.
      • total_chars   — total characters across all chunks.

    Returns
    -------
    dict
        Evaluation metrics.
    """
    if not chunks:
        return {"count": 0}

    sizes = [len(c) for c in chunks]
    avg = sum(sizes) / len(sizes)
    variance = (sum((s - avg) ** 2 for s in sizes) / len(sizes)) ** 0.5

    return {
        "count": len(chunks),
        "avg_size": round(avg, 1),
        "min_size": min(sizes),
        "max_size": max(sizes),
        "size_variance": round(variance, 1),
        "total_chars": sum(sizes),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Public API (backwards-compatible)
# ═══════════════════════════════════════════════════════════════════════════

def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    strategy: str | None = None,
) -> list[str]:
    """
    Split text into overlapping chunks using the configured strategy.

    Parameters
    ----------
    text : str
        The cleaned text to split.
    chunk_size : int | None
        Max characters per chunk. Defaults to config CHUNK_SIZE.
    chunk_overlap : int | None
        Overlap between chunks. Defaults to config CHUNK_OVERLAP.
    strategy : str | None
        Chunking strategy name. Defaults to config CHUNKING_STRATEGY.

    Returns
    -------
    list[str]
        A list of text chunks.
    """
    size = chunk_size or CHUNK_SIZE
    overlap = chunk_overlap or CHUNK_OVERLAP

    chunker = get_chunker(strategy)

    logger.info(
        "✂️ Chunking %d chars (strategy=%s, size=%d, overlap=%d)",
        len(text), chunker.name, size, overlap,
    )

    start_time = time.time()
    chunks = chunker.chunk(text, size, overlap)
    duration = time.time() - start_time

    metrics = evaluate_chunks(chunks)
    logger.info(
        "✅ Created %d chunks in %.3fs | avg=%s, min=%s, max=%s",
        metrics["count"], duration,
        metrics.get("avg_size", 0),
        metrics.get("min_size", 0),
        metrics.get("max_size", 0),
    )

    return chunks
