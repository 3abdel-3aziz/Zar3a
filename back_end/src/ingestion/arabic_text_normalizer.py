"""
arabic_text_normalizer.py
─────────────────────────────────────────────────────────────────────────────
Corrects reversed/mirrored Arabic text produced by Surya OCR 0.7.x on CPU.

Problem
───────
Surya OCR scans pages left-to-right regardless of script direction.
For Arabic (RTL) content it outputs both:
  - Individual Arabic words with characters reversed
  - Word tokens placed in LTR (wrong) order within each line

Example corruption:
  OCR output:  قحلملا وتويك لوكتورب يلع
  Expected:    على بروتوكول كيوتو الملحق

Fix strategy (token-aware reversal)
────────────────────────────────────
For each predominantly-Arabic line:
  1. Tokenise into (Arabic-word | number | whitespace | punct) tokens.
  2. Reverse characters WITHIN each Arabic-word token.
  3. Reverse the ORDER of the entire token list.
  4. Re-join tokens.

Digits/spaces/punctuation are never character-reversed, preserving numbers
like "2008" or "119" correctly.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import List

logger = logging.getLogger(__name__)

# ── Optional dependency guards ────────────────────────────────────────────────

try:
    import arabic_reshaper as _arabic_reshaper
    _RESHAPER_AVAILABLE = True
except ImportError:
    _arabic_reshaper = None  # type: ignore[assignment]
    _RESHAPER_AVAILABLE = False
    logger.debug(
        "arabic-reshaper not installed — character-form normalisation disabled. "
        "Install via `uv add arabic-reshaper`."
    )

# ── Regex helpers ─────────────────────────────────────────────────────────────

# Matches one or more Arabic / Arabic-Extended / Arabic-Supplement chars
_ARABIC_CHAR_RE = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]+"
)

# Tokeniser: splits a line into semantically-distinct token types while
# preserving all content so a round-trip join reproduces the original string.
_TOKEN_RE = re.compile(
    r"("
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]+"
    r"|\d+[\.,]?\d*"
    r"|\s+"
    r"|[^\s]"
    r")"
)


# ── Core helpers ──────────────────────────────────────────────────────────────


def _is_predominantly_arabic(text: str) -> bool:
    """Return True when >50% of alphabetic characters in *text* are Arabic."""
    arabic_count = sum(len(m) for m in _ARABIC_CHAR_RE.findall(text))
    alpha_count = sum(1 for ch in text if ch.isalpha())
    return alpha_count > 0 and arabic_count / alpha_count > 0.5


def _fix_reversed_arabic_line(line: str) -> str:
    """
    Correct a single reversed Arabic OCR line.

    Token-aware algorithm:
      1. Tokenise into (Arabic-word | number | whitespace | punct) segments.
      2. Reverse characters *within* each Arabic-word token.
      3. Reverse the *order* of the entire token list.
      4. Re-join.

    Non-Arabic tokens (digits, punctuation, spaces) are never character-reversed.
    """
    tokens: List[str] = _TOKEN_RE.findall(line)
    fixed: List[str] = []

    for tok in tokens:
        if _ARABIC_CHAR_RE.match(tok):
            fixed.append(tok[::-1])  # reverse chars within Arabic word
        else:
            fixed.append(tok)       # digit/space/punct: keep as-is

    # Reverse overall token order to restore RTL word order
    fixed.reverse()
    return "".join(fixed)


# Pre-compiled Unicode-safe regexes for reversal detection
# NOTE: \b word boundaries are intentionally AVOIDED here because Python's `re`
# module exhibits catastrophic backtracking on Arabic Unicode with \b, causing
# multi-minute hangs on documents with hundreds of chunks.
_RE_TEH_MARBUTA_START = re.compile(r"(?<![\u0600-\u06FF])ة[\u0600-\u06FF]{1,}")
_RE_REVERSED_PREPS = re.compile(r"(?<![\u0600-\u06FF])(?:يف|نم|ىلع|نع|ىلإ|نأ)(?![\u0600-\u06FF])")
_RE_ENDING_IN_LA = re.compile(r"[\u0600-\u06FF]{2,}لا(?![\u0600-\u06FF])")
_RE_ALIF_LAM_START = re.compile(r"(?<![\u0600-\u06FF])ال[\u0600-\u06FF]{2,}")
_RE_FORWARD_PREPS = re.compile(
    r"(?<![\u0600-\u06FF])(?:في|من|على|عن|إلى|مع|أن|أو|بعد|رقم|قرار|قانون|المجلس|الدولة|البيئة)(?![\u0600-\u06FF])"
)
_RE_TEH_MARBUTA_END = re.compile(r"[\u0600-\u06FF]{1,}ة(?![\u0600-\u06FF])")


def is_reversed_arabic_line(text: str) -> bool:
    """
    Detects whether an Arabic line is genuinely reversed.
    In normal Arabic, words start with 'ال' or prefixes and never start with
    teh marbuta (ة). In reversed Arabic, words start with 'ة' and end with 'لا'.

    Uses pre-compiled, \\b-free Unicode-safe regexes to avoid catastrophic
    backtracking on Arabic text in CPython's re module.
    """
    if not text.strip():
        return False

    # Fast length guard — skip very short fragments
    stripped = text.strip()
    if len(stripped) < 10:
        return False

    # Words starting with Teh Marbuta (impossible in standard forward Arabic)
    teh_marbuta_start = len(_RE_TEH_MARBUTA_START.findall(stripped))
    # Common Arabic prepositions reversed: يف (في), نم (من), ىلع (على), نع (عن), ىلإ (إلى)
    reversed_preps = len(_RE_REVERSED_PREPS.findall(stripped))
    # Words ending in "لا" (could be reversed "ال-")
    ending_in_la = len(_RE_ENDING_IN_LA.findall(stripped))

    backward_score = teh_marbuta_start * 3 + reversed_preps * 2 + ending_in_la

    # Early exit — if no backward signals, skip forward scoring
    if backward_score == 0:
        return False

    # Forward signals:
    alif_lam_start = len(_RE_ALIF_LAM_START.findall(stripped))
    forward_preps = len(_RE_FORWARD_PREPS.findall(stripped))
    teh_marbuta_end = len(_RE_TEH_MARBUTA_END.findall(stripped))

    forward_score = alif_lam_start * 2 + forward_preps * 2 + teh_marbuta_end

    return backward_score > forward_score and backward_score >= 2


def _clean_tatweel_and_noise(text: str) -> str:
    """
    Strips kashida/tatweel artifacts and OCR line noise.
    """
    # Replace repeated tatweel / underscore fillers with space
    text = re.sub(r"[ـ_]{2,}", " ", text)
    # Remove single tatweels within words
    text = re.sub(r"ـ+", "", text)
    return text


def _reshape(text: str) -> str:
    """
    Apply arabic-reshaper to normalise character presentation forms.

    Returns the input unchanged if arabic-reshaper is not installed.
    """
    if not _RESHAPER_AVAILABLE or _arabic_reshaper is None:
        return text
    try:
        return _arabic_reshaper.reshape(text)
    except Exception as exc:  # noqa: BLE001
        logger.debug("arabic-reshaper failed on text snippet: %s", exc)
        return text


# ── Public API ────────────────────────────────────────────────────────────────


def normalize_arabic_text(
    text: str,
    *,
    fix_reversal: bool = True,
    apply_reshaper: bool = False,
) -> str:
    """
    Normalize Arabic OCR text for downstream storage and embedding.

    Parameters
    ----------
    text:
        Raw OCR text (may contain reversed Arabic lines from Surya OCR).
    fix_reversal:
        When True (default), detects and applies the token-aware line reversal
        fix ONLY to lines that test positive as reversed Arabic. Forward Arabic
        lines remain completely untouched to preserve natural word order.
    apply_reshaper:
        When True, passes Arabic lines through arabic-reshaper to
        normalise character presentation forms.

    Returns
    -------
    str
        Corrected text in logical Unicode order, ready for SemanticChunker
        and multilingual embedding models.
    """
    if not text or not text.strip():
        return text

    # NFC normalisation: decomposes and recomposes Unicode combining chars
    text = unicodedata.normalize("NFC", text)
    text = _clean_tatweel_and_noise(text)

    lines = text.splitlines(keepends=True)
    result: List[str] = []

    for raw_line in lines:
        stripped = raw_line.rstrip("\n\r")
        ending = raw_line[len(stripped):]

        if not stripped:
            result.append(raw_line)
            continue

        if _is_predominantly_arabic(stripped):
            # Only reverse if the line actually exhibits backward text patterns
            if fix_reversal and is_reversed_arabic_line(stripped):
                stripped = _fix_reversed_arabic_line(stripped)
            if apply_reshaper:
                stripped = _reshape(stripped)

        result.append(stripped + ending)

    return "".join(result)
