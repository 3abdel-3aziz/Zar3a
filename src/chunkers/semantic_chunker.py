import logging
import re
import unicodedata
from typing import List, Optional, Union

from pydantic import BaseModel, Field

# pyrefly: ignore [missing-import]
from .base_chunker import BaseChunker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ChunkMetadata(BaseModel):
    """Strongly-typed metadata attached to every chunk."""

    type: str  # 'preamble' | 'article'
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
# Helpers
# ---------------------------------------------------------------------------

# Matches Arabic article headers: "### المادة 12", "ماده (٦)", etc.
_ARABIC_ARTICLE_NUMBER_RE = re.compile(
    r"(?:الماد[ةه]|ماد[ةه])\s*\(?\s*([0-9\u0660-\u0669]+)\)?",
    re.IGNORECASE,
)

# Matches English article headers in multiple formats:
#   "Article 5"  |  "Article (4)"  |  "(Article 1)"  |  "## Article 1 Some title"
_ENGLISH_ARTICLE_NUMBER_RE = re.compile(
    r"\(?\s*Article\s*\(?\s*(\d+)\s*\)?",
    re.IGNORECASE,
)

# Combined convenience alias used by _extract_article_number
_ARTICLE_NUMBER_RE = re.compile(
    r"(?:Article|الماد[ةه]|ماد[ةه])\s*\(?\s*([0-9\u0660-\u0669\w]+)\)?",
    re.IGNORECASE,
)


# Detects Arabic Unicode characters.
_ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+")


def _extract_article_number(header: str) -> Optional[Union[int, str]]:
    """Return the numeric article number from *header*, or ``None``.

    Resolution order:
    1. Arabic patterns  (e.g. ``مادة (٦)``  or  ``المادة 12``)
    2. English patterns (e.g. ``Article 5``, ``Article (4)``, ``(Article 1)``)
    """
    # -- Arabic first (preserves existing behaviour) -----------------------
    m = _ARABIC_ARTICLE_NUMBER_RE.search(header)
    if m:
        raw = m.group(1)
        # Convert Eastern-Arabic digits to Western digits
        eastern = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
        try:
            return int(raw.translate(eastern))
        except ValueError:
            return raw

    # -- English next -------------------------------------------------------
    m = _ENGLISH_ARTICLE_NUMBER_RE.search(header)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return m.group(1)

    return None


def _detect_language(header: str) -> str:
    """Return ``'ar'`` if the header contains Arabic characters, else ``'en'``."""
    return "ar" if _ARABIC_RE.search(header) else "en"


def _word_count(text: str) -> int:
    """Fast whitespace-based word count."""
    return len(text.split())


# ---------------------------------------------------------------------------
# SemanticChunker
# ---------------------------------------------------------------------------


class SemanticChunker(BaseChunker):
    """
    Article-aware semantic chunker for legal/regulatory Markdown documents.

    Splits the full Markdown text exactly at article headers, so every
    chunk maps 1-to-1 with one legal article.  This is the ideal input
    for the Diff Engine because it can detect precise article-level
    additions, modifications, and deletions.

    Recognised header patterns (configurable via ``header_pattern``):
        - Arabic:  ``### المادة 5``  or  ``### مادة الأولى``
        - English: ``### Article 5``

    Any text that appears before the first article header is kept as a
    preamble chunk (chunk_index=0, type='preamble').

    Patterns are loaded from ``config/document_processor_config.yaml``
    (``processing_pipeline.semantic_chunker.*``) and compiled at import
    time.  Pass explicit ``re.Pattern`` objects to override them at
    runtime (useful in tests).
    """

    # ------------------------------------------------------------------
    # Default patterns – loaded from config at module level so the YAML
    # is the single source of truth.
    # ------------------------------------------------------------------
    _DEFAULT_ARTICLE_PATTERN: re.Pattern = re.compile(
        r"(?m)"
        r"^((?:#{1,6}\s*|\*\*)?[\(\（]?\s*(?!Page\s+\d)(?:ب?(?:المادة|مادة)|Article)\s*[\(\（]?\s*(?:\d+|[٠-٩]+|[أ-ي]+)?\s*[\)\）]?\s*(?:\((?:المادة|مادة)\s+[أ-ي]+\))?\s*:?(?:\*\*)?\s*:?\s*$"
        r"|\*\*\s*(?:ب?(?:المادة|مادة)|Article)\s*[\(\（]\s*(?:\d+|[٠-٩]+|[أ-ي]+)\s*[\)\）]\s*:?\s*\*\*)",
        re.IGNORECASE,
    )
    _DEFAULT_PAGE_SEP_PATTERN: re.Pattern = re.compile(
        r"(?m)^\s*(?:#{1,6}\s*)?---\s*\n+\s*(?:#{1,6}\s*)?Page\s+\d*\s*\n+\d*\s*\n*"
    )

    # ------------------------------------------------------------------

    def __init__(
        self,
        header_pattern: Optional[re.Pattern] = None,
        page_sep_pattern: Optional[re.Pattern] = None,
    ) -> None:
        """
        Args:
            header_pattern (re.Pattern | None): Override the default article-header
                regex.  Must use exactly one capture group so that ``re.split()``
                keeps the delimiter in the resulting list.
            page_sep_pattern (re.Pattern | None): Override the default page-separator
                cleanup regex.
        """
        self._pattern = header_pattern or self._DEFAULT_ARTICLE_PATTERN
        self._page_sep_pattern = page_sep_pattern or self._DEFAULT_PAGE_SEP_PATTERN

    # ------------------------------------------------------------------
    # BaseChunker contract
    # ------------------------------------------------------------------

    def create_chunks(self, full_text: str, doc_id: str) -> List[dict]:
        """
        Split *full_text* on article headers.

        Args:
            full_text (str): The complete Markdown string from the extractor.
            doc_id (int):    Parent document identifier (must be a positive int).

        Returns:
            List[Chunk]: Pydantic ``Chunk`` objects with enriched metadata.

        Raises:
            ValueError: If *full_text* is not a non-empty string, or if
                        *doc_id* is not a positive integer.
        """
        # ------------------------------------------------------------------
        # 1. Input validation
        # ------------------------------------------------------------------
        if not isinstance(full_text, str) or not full_text.strip():
            raise ValueError(
                "full_text must be a non-empty string, "
                f"got {type(full_text).__name__!r}: {full_text!r}"
            )
        if not isinstance(doc_id, (int, str)):
            raise ValueError(
                f"doc_id must be an integer or string, got {type(doc_id).__name__!r}: {doc_id!r}"
            )

        logger.debug("SemanticChunker: Splitting doc_id=%s on article headers…", doc_id)

        # ------------------------------------------------------------------
        # 2. Page-separator cleanup (hardened regex from config)
        # ------------------------------------------------------------------
        full_text = self._page_sep_pattern.sub("", full_text)

        # ------------------------------------------------------------------
        # 3. Split on article headers
        # re.split with a capturing group keeps the delimiters in the list:
        # [text_before_first_header, header, body, header, body, …]
        # ------------------------------------------------------------------
        parts = self._pattern.split(full_text)

        chunks: List[Chunk] = []
        idx = 0
        article_count = 0
        preamble_count = 0

        # ------------------------------------------------------------------
        # 4. Preamble – everything before the first header
        # ------------------------------------------------------------------
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
                        language="en",
                    ),
                )
            )
            idx += 1
            preamble_count += 1

        # ------------------------------------------------------------------
        # 5. Article chunks – header/body pairs
        # ------------------------------------------------------------------
        remaining = parts[1:]
        it = iter(range(len(remaining)))
        for i in it:
            header = remaining[i]

            # Guard against malformed splits (unexpected list structure)
            try:
                j = next(it)
            except StopIteration:
                logger.warning(
                    "SemanticChunker: doc_id=%s – header at position %d has no "
                    "corresponding body; skipping: %r",
                    doc_id,
                    i,
                    header[:80],
                )
                break

            body_raw = remaining[j]
            if not isinstance(header, str) or not isinstance(body_raw, str):
                logger.warning(
                    "SemanticChunker: doc_id=%s – unexpected type at positions "
                    "(%d, %d): header=%r body=%r; skipping.",
                    doc_id,
                    i,
                    j,
                    type(header),
                    type(body_raw),
                )
                continue

            header = header.strip()
            body = body_raw.strip()
            if not body:
                logger.warning(
                    "SemanticChunker: doc_id=%s – article chunk at index %d has an empty body "
                    "(header: %r). This may indicate a PDF extraction issue.",
                    doc_id,
                    idx,
                    header,
                )
            content = f"{header}\n\n{body}" if body else header

            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    chunk_index=idx,
                    content=content,
                    metadata=ChunkMetadata(
                        type="article",
                        header=header,
                        article_number=_extract_article_number(header),
                        word_count=_word_count(content),
                        language=_detect_language(header),
                    ),
                )
            )
            idx += 1
            article_count += 1

        logger.info(
            "SemanticChunker: doc_id=%s – %d chunks created "
            "(%d articles, %d preamble).",
            doc_id,
            len(chunks),
            article_count,
            preamble_count,
        )
        
        return [c.model_dump() for c in chunks]