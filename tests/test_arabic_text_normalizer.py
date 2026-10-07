"""
tests/test_arabic_text_normalizer.py

Unit tests for arabic_text_normalizer.py verifying:
  - Token-aware reversal of mirrored Arabic OCR lines
  - Preservation of numbers, punctuation, and English text
  - Normalization of text passed into ingestion parsing
"""

import pytest
from back_end.src.ingestion.arabic_text_normalizer import (
    normalize_arabic_text,
    _fix_reversed_arabic_line,
    _is_predominantly_arabic,
)


def test_is_predominantly_arabic():
    assert _is_predominantly_arabic("قحلملا وتويك لوكتورب يلع") is True
    assert _is_predominantly_arabic("Hello World 123") is False
    assert _is_predominantly_arabic("123456") is False
    assert _is_predominantly_arabic("") is False


def test_fix_reversed_arabic_line():
    # Mirrored: قحلملا وتويك لوكتورب يلع
    # Correct:  علي بروتكول كيوتو الملحق
    corrupted = "قحلملا وتويك لوكتورب يلع"
    fixed = _fix_reversed_arabic_line(corrupted)
    assert fixed == "علي بروتكول كيوتو الملحق"


def test_normalize_arabic_preserves_numbers_and_punctuation():
    # With legal article number e.g. "مادة (119)" reversed
    corrupted = "(119) ةدام"
    fixed = normalize_arabic_text(corrupted)
    # The Arabic word "ةدام" is reversed to "مادة", number "(119)" is kept intact
    assert "مادة" in fixed
    assert "119" in fixed


def test_normalize_arabic_ignores_english():
    text = "Regulation standard (ISO 9001) for environmental safety."
    normalized = normalize_arabic_text(text)
    assert normalized == text


def test_normalize_arabic_empty_or_whitespace():
    assert normalize_arabic_text("") == ""
    assert normalize_arabic_text("   \n\t  ") == "   \n\t  "
