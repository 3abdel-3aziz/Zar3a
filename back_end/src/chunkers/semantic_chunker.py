"""
semantic_chunker.py
───────────────────────────────────────────────────────────────────────────────
Article-aware + recursive-character-fallback chunker for Arabic legal documents.

Strategy (two-pass):
  Pass 1 – Structural split:
      Split on recognised article/section headers (المادة, مادة, Article).
      Every recognised header becomes a separate chunk so article-level
      diffs and citation work out of the box.

  Pass 2 – Recursive character fallback (NEW):
      If Pass 1 yields ≤1 chunk (i.e. no structural headers were found, or the
      entire document ended up as one giant preamble), run a recursive
      character splitter instead.  This splitter tries paragraph → sentence →
      word boundaries in sequence and guarantees that every chunk stays inside
      the configured size window with a configurable overlap.

      This covers three real-world failure modes:
        a) OCR output that looks like plain paragraphs with no Markdown headers.
        b) Documents where the TextFormatter missed the header pattern.
        c) Non-legal documents (wiki scrape, ministry announcements) that have
           no article structure at all.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Union

from pydantic import BaseModel, Field

# pyrefly: ignore [missing-import]
from .base_chunker import BaseChunker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ChunkMetadata(BaseModel):
    """Strongly-typed metadata attached to every chunk."""

    type: str  # 'preamble' | 'article' | 'section' | 'paragraph' | 'recursive'
    header: Optional[str] = None
    article_number: Optional[Union[int, str]] = None
    word_count: int = 0
    language: str = "en"


class Chunk(BaseModel):
    """A single document chunk produced by the SemanticChunker."""

    doc_id: Union[int, str]
    chunk_index: int
    content: str
    metadata: ChunkMetadata


# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

# Arabic article headers: "### المادة 12", "مادة (٦)", "المادة الأولى", etc.
_ARABIC_ARTICLE_NUMBER_RE = re.compile(
    r"(?:الماد[ةه]|ماد[ةه])\s*\(?\s*([0-9\u0660-\u0669]+)\)?",
    re.IGNORECASE,
)

# English article headers
_ENGLISH_ARTICLE_NUMBER_RE = re.compile(
    r"\(?\s*Article\s*\(?\s*(\d+)\s*\)?",
    re.IGNORECASE,
)

# Arabic Unicode
_ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+")

# Eastern-Arabic → Western digit translation table
_EASTERN_TO_WESTERN = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

# Separators used by the recursive character splitter, in priority order.
# We try each one in sequence until chunks fit within the target window.
_SPLIT_SEPARATORS: List[str] = [
    "\n\n",   # paragraph boundary
    ".\n",    # sentence ending followed by newline
    "،\n",    # Arabic comma + newline
    ".\u0020",  # sentence ending + space
    "،\u0020",  # Arabic comma + space
    "\n",     # any newline
    "،",      # Arabic comma alone
    ".",      # any period
    " ",      # word boundary (last resort)
    "",       # character (absolute fallback)
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _extract_article_number(header: str) -> Optional[Union[int, str]]:
    """Return the numeric article number from *header*, or ``None``."""
    m = _ARABIC_ARTICLE_NUMBER_RE.search(header)
    if m:
        raw = m.group(1)
        try:
            return int(raw.translate(_EASTERN_TO_WESTERN))
        except ValueError:
            return raw

    m = _ENGLISH_ARTICLE_NUMBER_RE.search(header)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return m.group(1)

    return None


def _detect_language(text: str) -> str:
    """Return ``'ar'`` if the text contains Arabic characters, else ``'en'``."""
    return "ar" if _ARABIC_RE.search(text) else "en"


def _word_count(text: str) -> int:
    """Fast whitespace-based word count."""
    return len(text.split())


def _classify_header_type(header: str) -> str:
    """Classify an article header as 'article' or 'section'."""
    section_re = re.compile(
        r"(?:الباب|باب|الفصل|فصل|القسم|قسم|الجزء|جزء)",
        re.IGNORECASE,
    )
    if section_re.search(header):
        return "section"
    return "article"


# ---------------------------------------------------------------------------
# Recursive character splitter (no external deps)
# ---------------------------------------------------------------------------


def _recursive_split(
    text: str,
    separators: List[str],
    chunk_size: int,
    chunk_overlap: int,
) -> List[str]:
    """
    Splits *text* into chunks no larger than *chunk_size* characters,
    with *chunk_overlap* characters of prefix context.

    Tries each separator in *separators* in order.  Once a separator
    produces pieces that individually fit within *chunk_size*, those pieces
    are merged greedily until the next merge would exceed *chunk_size*,
    then a new chunk starts (beginning with the last *chunk_overlap* chars
    of the preceding chunk as context).
    """
    # Base case: already fits
    if len(text) <= chunk_size:
        stripped = text.strip()
        return [stripped] if stripped else []

    # Try separators in priority order
    for sep in separators:
        if sep == "":
            # Absolute character-level fallback
            pieces = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size - chunk_overlap)]
            return [p.strip() for p in pieces if p.strip()]

        raw_pieces = text.split(sep)
        # If the split produced >1 pieces, use this separator
        if len(raw_pieces) > 1:
            # Re-attach the separator to all pieces except the last
            pieces_with_sep: List[str] = []
            for i, piece in enumerate(raw_pieces):
                if i < len(raw_pieces) - 1:
                    pieces_with_sep.append(piece + sep)
                else:
                    pieces_with_sep.append(piece)

            # Sub-split any piece that still exceeds chunk_size
            good_pieces: List[str] = []
            remaining_seps = separators[separators.index(sep) + 1 :]
            for piece in pieces_with_sep:
                if len(piece) > chunk_size and remaining_seps:
                    good_pieces.extend(_recursive_split(piece, remaining_seps, chunk_size, chunk_overlap))
                else:
                    good_pieces.append(piece)

            # Merge small pieces into chunks
            chunks: List[str] = []
            current = ""
            for piece in good_pieces:
                if not piece.strip():
                    continue
                if len(current) + len(piece) <= chunk_size:
                    current += piece
                else:
                    if current.strip():
                        chunks.append(current.strip())
                    # Start new chunk with overlap context
                    if chunk_overlap > 0 and current:
                        overlap_text = current[-chunk_overlap:]
                        current = overlap_text + piece
                    else:
                        current = piece

            if current.strip():
                chunks.append(current.strip())

            return [c for c in chunks if c.strip()]

    # Fallback: return as single chunk
    return [text.strip()] if text.strip() else []


# ---------------------------------------------------------------------------
# SemanticChunker
# ---------------------------------------------------------------------------


class SemanticChunker(BaseChunker):
    """
    Article-aware semantic chunker for legal/regulatory Markdown documents.

    Pass 1: Splits on article/section headers (المادة, مادة, Article).
    Pass 2: If no structure found, applies recursive character splitting
            with configurable chunk_size and chunk_overlap to guarantee
            all documents are chunked even without explicit headers.
    """

    # Default article header pattern
    _DEFAULT_ARTICLE_PATTERN: re.Pattern = re.compile(
        r"(?m)"
        r"^((?:#{1,6}\s*|\*\*)?[\(\（]?\s*(?!Page\s+\d)(?:ب?(?:المادة|مادة)|Article)\s*[\(\（]?\s*(?:\d+|[٠-٩]+|[أ-ي]+)?\s*[\)\）]?\s*(?:\((?:المادة|مادة)\s+[أ-ي]+\))?\s*:?(?:\*\*)?\s*:?\s*$"
        r"|\*\*\s*(?:ب?(?:المادة|مادة)|Article)\s*[\(\（]\s*(?:\d+|[٠-٩]+|[أ-ي]+)\s*[\)\）]\s*:?\s*\*\*)",
        re.IGNORECASE,
    )

    _DEFAULT_PAGE_SEP_PATTERN: re.Pattern = re.compile(
        r"(?m)^\s*(?:#{1,6}\s*)?---\s*\n+\s*(?:#{1,6}\s*)?Page\s+\d*\s*\n+\d*\s*\n*"
    )

    def __init__(
        self,
        header_pattern: Optional[re.Pattern] = None,
        page_sep_pattern: Optional[re.Pattern] = None,
        *,
        # Recursive fallback parameters
        fallback_chunk_size: int = 700,
        fallback_chunk_overlap: int = 120,
        fallback_min_chars: int = 80,
    ) -> None:
        """
        Args:
            header_pattern:       Override the default article-header regex.
            page_sep_pattern:     Override the default page-separator cleanup regex.
            fallback_chunk_size:  Target character length for recursive fallback chunks.
                                  Chosen to fit ~100-120 Arabic words (good for
                                  multilingual-e5-large's 512-token limit).
            fallback_chunk_overlap: Characters of context carried over between
                                  consecutive fallback chunks.
            fallback_min_chars:   Discard fallback chunks shorter than this.
        """
        self._pattern = header_pattern or self._DEFAULT_ARTICLE_PATTERN
        self._page_sep_pattern = page_sep_pattern or self._DEFAULT_PAGE_SEP_PATTERN
        self._fallback_chunk_size = fallback_chunk_size
        self._fallback_chunk_overlap = fallback_chunk_overlap
        self._fallback_min_chars = fallback_min_chars

    # ------------------------------------------------------------------
    # BaseChunker contract
    # ------------------------------------------------------------------

    def create_chunks(self, full_text: str, doc_id: str) -> List[dict]:
        """
        Split *full_text* on article headers (Pass 1).
        Fall back to recursive character splitting (Pass 2) when no
        structural headers are detected.

        Returns:
            List[dict]: Serialised Chunk model dicts.
        """
        if not isinstance(full_text, str) or not full_text.strip():
            raise ValueError(
                f"full_text must be a non-empty string, got {type(full_text).__name__!r}: {full_text!r}"
            )
        if not isinstance(doc_id, (int, str)):
            raise ValueError(
                f"doc_id must be an integer or string, got {type(doc_id).__name__!r}: {doc_id!r}"
            )

        # 1. Page-separator cleanup
        full_text = self._page_sep_pattern.sub("", full_text)

        # 2. Pass 1 — structural header split
        chunks = self._structural_split(full_text, doc_id)

        # 3. Pass 2 — recursive fallback when no article structure found
        #    Trigger when: only one chunk produced (the whole doc became preamble),
        #    OR that single chunk is extremely large (> 2× fallback window)
        if len(chunks) <= 1:
            sole_chunk = chunks[0] if chunks else None
            sole_is_huge = (
                sole_chunk is not None
                and len(sole_chunk["content"]) > self._fallback_chunk_size * 2
            )
            if sole_chunk is None or sole_is_huge:
                logger.info(
                    "SemanticChunker: doc_id=%s – structural split yielded %d chunk(s). "
                    "Activating recursive character fallback (chunk_size=%d, overlap=%d).",
                    doc_id,
                    len(chunks),
                    self._fallback_chunk_size,
                    self._fallback_chunk_overlap,
                )
                chunks = self._recursive_fallback_split(full_text, doc_id)

        logger.info(
            "SemanticChunker: doc_id=%s – %d total chunks produced.",
            doc_id,
            len(chunks),
        )
        return chunks

    # ------------------------------------------------------------------
    # Internal: Pass 1 — structural header split
    # ------------------------------------------------------------------

    def _structural_split(self, full_text: str, doc_id: str) -> List[dict]:
        """Split on article/section headers using the compiled regex."""
        parts = self._pattern.split(full_text)

        chunks: List[Chunk] = []
        idx = 0
        article_count = 0
        preamble_count = 0

        # Preamble — text before the first header
        preamble = parts[0].strip()
        if preamble:
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    chunk_index=idx,
                    content=preamble,
                    metadata=ChunkMetadata(
                        type="preamble",
                        header=None,
                        article_number=None,
                        word_count=_word_count(preamble),
                        language=_detect_language(preamble),
                    ),
                )
            )
            idx += 1
            preamble_count += 1

        # Article / section chunks
        remaining = parts[1:]
        it = iter(range(len(remaining)))
        for i in it:
            header = remaining[i]
            try:
                j = next(it)
            except StopIteration:
                logger.warning(
                    "SemanticChunker: doc_id=%s – header at position %d has no body; skipping: %r",
                    doc_id,
                    i,
                    header[:80],
                )
                break

            body_raw = remaining[j]
            if not isinstance(header, str) or not isinstance(body_raw, str):
                continue

            header = header.strip()
            body = body_raw.strip()
            content = f"{header}\n\n{body}" if body else header

            chunk_type = _classify_header_type(header)
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    chunk_index=idx,
                    content=content,
                    metadata=ChunkMetadata(
                        type=chunk_type,
                        header=header,
                        article_number=_extract_article_number(header),
                        word_count=_word_count(content),
                        language=_detect_language(header),
                    ),
                )
            )
            idx += 1
            article_count += 1

        logger.debug(
            "SemanticChunker._structural_split: doc_id=%s – %d chunks "
            "(%d articles/sections, %d preamble).",
            doc_id,
            len(chunks),
            article_count,
            preamble_count,
        )
        return [c.model_dump() for c in chunks]

    # ------------------------------------------------------------------
    # Internal: Pass 2 — recursive character fallback
    # ------------------------------------------------------------------

    def _recursive_fallback_split(self, full_text: str, doc_id: str) -> List[dict]:
        """
        Splits *full_text* recursively using paragraph → sentence → word
        boundaries until all chunks fit within *fallback_chunk_size*.
        """
        raw_chunks = _recursive_split(
            full_text,
            _SPLIT_SEPARATORS,
            self._fallback_chunk_size,
            self._fallback_chunk_overlap,
        )

        chunks: List[dict] = []
        for idx, content in enumerate(raw_chunks):
            content = content.strip()
            if len(content) < self._fallback_min_chars:
                # Merge tiny tail fragments into the previous chunk
                if chunks:
                    chunks[-1]["content"] += " " + content
                    chunks[-1]["metadata"]["word_count"] = _word_count(chunks[-1]["content"])
                continue

            chunk = Chunk(
                doc_id=doc_id,
                chunk_index=idx,
                content=content,
                metadata=ChunkMetadata(
                    type="recursive",
                    header=None,
                    article_number=None,
                    word_count=_word_count(content),
                    language=_detect_language(content),
                ),
            )
            chunks.append(chunk.model_dump())

        # Re-index after any merges
        for new_idx, ch in enumerate(chunks):
            ch["chunk_index"] = new_idx

        logger.debug(
            "SemanticChunker._recursive_fallback_split: doc_id=%s – %d chunks.",
            doc_id,
            len(chunks),
        )
        return chunks