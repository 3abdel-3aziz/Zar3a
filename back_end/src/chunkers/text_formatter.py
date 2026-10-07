# src/chunkers/text_formatter.py

import re
import unicodedata


class TextFormatter:
    """
    Standardizes and cleans text extracted from OCR engines.
    Provides different levels of cleaning based on the source quality.

    Key improvements:
    - Preserves paragraph structure (double newlines) instead of collapsing them.
    - Promotes Arabic section/chapter headers (باب, فصل, قسم) to Markdown headings.
    - Handles both Eastern-Arabic and Western-Arabic numerals in patterns.
    - Normalises Unicode to NFC before any regex work.
    """

    # Legal article header: مادة / المادة followed by a number
    _ARTICLE_RE = re.compile(
        r"^((?:ب)?(?:المادة|مادة))\s*(\(?\s*(?:\d+|[٠-٩]+|[أ-ي]+)\s*\)?)",
        re.MULTILINE,
    )

    # Section / chapter headers — promoted to ## (h2)
    _SECTION_RE = re.compile(
        r"^((?:الباب|باب|الفصل|فصل|القسم|قسم|الجزء|جزء)(?:\s+(?:\d+|[٠-٩]+|[أ-ي]+|الأول|الثاني|الثالث|الرابع|الخامس))?(?:\s*[:\-–—])?)",
        re.MULTILINE,
    )

    # Collapses 3+ consecutive blank lines to exactly two (one blank line separator)
    _MULTI_BLANK_RE = re.compile(r"\n{3,}")

    # Trailing/leading whitespace on every line (but preserves the newline itself)
    _LINE_TRAILING_RE = re.compile(r"[ \t]+$", re.MULTILINE)

    def light_format(self, text: str) -> str:
        """
        Applies minimal formatting for high-quality Markdown (e.g., from Mistral).
        - Promotes untagged article lines to ### headers.
        - Promotes section/chapter lines to ## headers.
        - Preserves all paragraph separators.
        """
        text = unicodedata.normalize("NFC", text)
        text = self._LINE_TRAILING_RE.sub("", text)
        text = self._MULTI_BLANK_RE.sub("\n\n", text)

        lines = text.split("\n")
        out: list[str] = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                out.append("")
                continue

            if stripped.startswith("#"):
                # Already a markdown header — keep as-is
                out.append(line)
            elif self._ARTICLE_RE.match(stripped):
                out.append(f"### {stripped}")
            elif self._SECTION_RE.match(stripped):
                out.append(f"## {stripped}")
            else:
                out.append(line)

        return "\n".join(out)

    def full_clean_format(self, text: str) -> str:
        """
        Heavy cleaning for raw OCR output (Surya / EasyOCR).
        - Strips OCR noise characters.
        - Fixes whitespace WITHOUT discarding paragraph structure.
        - Promotes article and section headers to Markdown headings.

        Critical fix: preserves blank-line paragraph separators so that the
        SemanticChunker's regex fallback can detect sentence / paragraph
        boundaries correctly.
        """
        text = unicodedata.normalize("NFC", text)

        # Remove obvious OCR noise: lone box-drawing / replacement chars
        text = re.sub(r"[\ufffd\u25a0\u25a1\u2588\u2591\u2592\u2593]+", " ", text)

        # Normalise Windows line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Strip trailing whitespace per line (preserves line structure)
        text = self._LINE_TRAILING_RE.sub("", text)

        # Collapse 3+ blank lines → single blank line (preserves paragraphs)
        text = self._MULTI_BLANK_RE.sub("\n\n", text).strip()

        # Collapse multiple spaces within a line into one
        text = re.sub(r"[ \t]{2,}", " ", text)

        lines = text.split("\n")
        out: list[str] = []

        for line in lines:
            stripped = line.strip()

            if not stripped:
                # Preserve the blank line (paragraph separator)
                out.append("")
                continue

            if stripped.startswith("#"):
                out.append(stripped)
            elif self._ARTICLE_RE.match(stripped):
                out.append(f"### {stripped}")
            elif self._SECTION_RE.match(stripped):
                out.append(f"## {stripped}")
            else:
                out.append(stripped)

        return "\n".join(out)