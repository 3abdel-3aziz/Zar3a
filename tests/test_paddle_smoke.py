"""
tests/test_paddle_smoke.py

Smoke & verification tests for PaddleOCR as the default OCR engine.

Checks:
  1. Dependency imports  – paddleocr, pymupdf (fitz), pypdf all importable.
  2. PaddleOCR backend   – `paddle` core package importable & version logged.
  3. Default engine      – MistralPDFProcessor defaults to 'paddle' without
                           touching the Mistral API or any API key.
  4. PaddleOCRExtractor  – lazy initialization succeeds (no actual OCR call
                           needed; we just confirm the instance is created with
                           the correct language setting).
  5. No-Mistral guard    – verifying that running with no MISTRAL_API_KEY env var
                           does NOT raise an error when engine='paddle'.
  6. LocalPDFTextExtractor – still available as a zero-API pre-check layer.
"""

import sys
import logging
import pytest
from pathlib import Path
from unittest.mock import MagicMock

# ── Project root on sys.path ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("Zar3a.Test.PaddleSmoke")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Dependency Import Check
# ═══════════════════════════════════════════════════════════════════════════════

def test_paddleocr_importable():
    """paddleocr must be installed and importable."""
    import paddleocr
    assert hasattr(paddleocr, "PaddleOCR"), (
        "paddleocr.PaddleOCR class not found – is the package installed correctly?"
    )
    logger.info("paddleocr version: %s", paddleocr.__version__)


def test_pymupdf_importable():
    """pymupdf (fitz) must be installed and importable."""
    import pymupdf  # noqa: F401
    import fitz     # noqa: F401  (legacy alias)
    logger.info("pymupdf (fitz) importable – version: %s", pymupdf.__version__)


def test_pypdf_importable():
    """pypdf must be installed and importable."""
    import pypdf  # noqa: F401
    logger.info("pypdf importable – version: %s", pypdf.__version__)


def test_paddlepaddle_backend_importable():
    """
    paddle (the PaddlePaddle backend) must be installed.
    If this fails: `uv add paddlepaddle`
    """
    try:
        import paddle
        version = paddle.__version__
        logger.info("paddlepaddle (paddle) version: %s", version)
    except ModuleNotFoundError:
        pytest.fail(
            "PaddlePaddle backend ('paddle') is NOT installed.\n"
            "Fix: uv add paddlepaddle"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Default Engine = 'paddle' (no API key required)
# ═══════════════════════════════════════════════════════════════════════════════

def test_default_engine_is_paddle(tmp_path, monkeypatch):
    """
    MistralPDFProcessor must default to 'paddle' even when no MISTRAL_API_KEY
    is set. It should NOT raise any error about missing API keys.
    """
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.delenv("MISTRAL_API_KEYS", raising=False)
    monkeypatch.delenv("OCR_ENGINE", raising=False)

    from src.ingestion.ocr_loader import MistralPDFProcessor

    processor = MistralPDFProcessor(output_dir=tmp_path, inter_file_delay=0.0)

    assert processor.ocr_engine == "paddle", (
        f"Expected ocr_engine='paddle', got '{processor.ocr_engine}'"
    )
    assert processor.paddle_extractor is not None, (
        "PaddleOCRExtractor sub-component was not initialized."
    )
    assert processor.ocr_client is None, (
        "MistralOCRClient should be None when paddle engine is selected."
    )
    assert processor.pool_manager is None, (
        "APIKeyPoolManager should be None when paddle engine is selected."
    )

    logger.info("Default engine is 'paddle'. Mistral API client was NOT initialized.")


def test_no_api_key_no_error_with_paddle(tmp_path, monkeypatch):
    """
    Explicitly confirm that missing MISTRAL_API_KEY does NOT raise ValueError
    when ocr_engine is 'paddle' (or unset).
    """
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.delenv("MISTRAL_API_KEYS", raising=False)

    from src.ingestion.ocr_loader import MistralPDFProcessor

    try:
        processor = MistralPDFProcessor(
            ocr_engine="paddle",
            output_dir=tmp_path,
            inter_file_delay=0.0,
        )
    except ValueError as exc:
        pytest.fail(
            f"MistralPDFProcessor raised ValueError with paddle engine: {exc}"
        )

    assert processor.ocr_engine == "paddle"
    logger.info("No ValueError raised with engine='paddle' and no API key in environment.")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PaddleOCRExtractor Lazy-Init Smoke Test
# ═══════════════════════════════════════════════════════════════════════════════

def test_paddle_extractor_initializes_correctly(tmp_path, monkeypatch):
    """
    Verifies PaddleOCRExtractor is created with the correct lang='ar' setting
    and that _ocr is eagerly initialized (singleton pattern) – NOT None after __init__.
    """
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)

    from src.ingestion.ocr_loader import MistralPDFProcessor

    processor = MistralPDFProcessor(
        ocr_engine="paddle",
        output_dir=tmp_path,
        inter_file_delay=0.0,
        paddle_lang="ar",
    )

    extractor = processor.paddle_extractor
    assert extractor is not None
    assert extractor.lang == "ar", (
        f"PaddleOCRExtractor.lang expected 'ar', got '{extractor.lang}'"
    )
    # Singleton eager init: _ocr must be set (not None) after __init__
    assert extractor._ocr is not None, (
        "PaddleOCR instance should be set immediately in __init__ (singleton pattern)."
    )

    logger.info(
        "PaddleOCRExtractor configured: lang='%s', _ocr initialized eagerly (singleton).",
        extractor.lang,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. LocalPDFTextExtractor is Active as Pre-Check Layer
# ═══════════════════════════════════════════════════════════════════════════════

def test_local_text_extractor_active_with_paddle(tmp_path, monkeypatch):
    """
    LocalPDFTextExtractor must always be active regardless of engine choice,
    as it acts as a zero-API, zero-OCR pre-check layer.
    """
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)

    from src.ingestion.ocr_loader import MistralPDFProcessor, LocalPDFTextExtractor

    processor = MistralPDFProcessor(
        ocr_engine="paddle",
        output_dir=tmp_path,
        inter_file_delay=0.0,
    )

    assert isinstance(processor.local_extractor, LocalPDFTextExtractor), (
        "processor.local_extractor is not a LocalPDFTextExtractor instance."
    )
    assert processor.local_extractor.is_pypdf_available(), (
        "pypdf is not available – LocalPDFTextExtractor will not function."
    )

    logger.info("LocalPDFTextExtractor is active and pypdf is available.")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Full process_pdf with Mocked PaddleOCR (no GPU / model download needed)
# ═══════════════════════════════════════════════════════════════════════════════

def test_process_pdf_paddle_engine_mock(tmp_path, monkeypatch):
    """
    Smoke test for the full process_pdf() flow using the paddle engine.
    PaddleOCRExtractor.extract_text is mocked to avoid model download.
    Confirms: cache miss -> paddle OCR called -> result saved -> cache hit on second call.
    """
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)

    dummy_pdf = tmp_path / "test_doc.pdf"
    dummy_pdf.write_bytes(b"%PDF-1.4 fake arabic content")

    from src.ingestion.ocr_loader import MistralPDFProcessor

    processor = MistralPDFProcessor(
        ocr_engine="paddle",
        output_dir=tmp_path,
        inter_file_delay=0.0,
    )

    mock_ocr_result = {
        "pages": [{"index": 0, "markdown": "نص عربي تجريبي"}],
        "source": "paddle_ocr_local",
    }

    # Mock PaddleOCRExtractor.extract_text to bypass model loading
    processor.paddle_extractor.extract_text = MagicMock(return_value=mock_ocr_result)

    # First call: cache miss -> paddle OCR
    result1 = processor.process_pdf(dummy_pdf)
    assert result1["pages"][0]["markdown"] == "نص عربي تجريبي"
    assert processor.paddle_extractor.extract_text.call_count == 1
    assert processor.stats["paddle_ocr_hits"] == 1
    assert processor.stats["cache_hits"] == 0
    logger.info("First call: paddle OCR executed (call_count=1, paddle_ocr_hits=1).")

    # Second call: cache hit
    result2 = processor.process_pdf(dummy_pdf)
    assert result2 == result1
    assert processor.paddle_extractor.extract_text.call_count == 1  # not called again
    assert processor.stats["cache_hits"] == 1
    logger.info("Second call: cache hit (call_count still 1, cache_hits=1).")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Stats Tracking for Paddle Engine
# ═══════════════════════════════════════════════════════════════════════════════

def test_stats_structure_paddle(tmp_path, monkeypatch):
    """Confirms all expected stat keys are present when using paddle engine."""
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)

    from src.ingestion.ocr_loader import MistralPDFProcessor

    processor = MistralPDFProcessor(
        ocr_engine="paddle",
        output_dir=tmp_path,
        inter_file_delay=0.0,
    )

    stats = processor.get_processing_stats()
    expected_keys = {"cache_hits", "local_text_hits", "paddle_ocr_hits", "ocr_api_hits", "failures"}
    assert expected_keys.issubset(stats.keys()), (
        f"Missing stats keys. Got: {set(stats.keys())}"
    )
    assert all(v == 0 for v in stats.values()), (
        "All stats should be 0 at initialization."
    )
    logger.info("Stats structure valid: %s", stats)
