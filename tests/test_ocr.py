"""
tests/test_ocr.py

Unit and integration tests for src/ingestion/ocr_loader.py:
  - LocalPDFTextExtractor (Digital PDF pre-check)
  - APIKeyPoolManager (Key rotation & cooldown tracking)
  - MistralOCRClient (Backoff + Jitter + Header parsing)
  - MistralPDFProcessor (Facade coordinator & caching)
"""

import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root directory is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.ocr_loader import (
    LocalPDFTextExtractor,
    APIKeyPoolManager,
    MistralOCRClient,
    MistralPDFProcessor,
)


@pytest.fixture
def mock_pdf_path(tmp_path):
    pdf_file = tmp_path / "sample.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 dummy pdf binary content")
    return pdf_file


def test_local_pdf_text_extractor_low_threshold(mock_pdf_path):
    extractor = LocalPDFTextExtractor(min_char_threshold=100)
    # Binary dummy PDF has no valid text layer in pypdf
    res = extractor.extract_text(mock_pdf_path)
    assert res is None


def test_api_key_pool_manager():
    keys = ["key_111111", "key_222222"]
    pool = APIKeyPoolManager(api_keys=keys, default_cooldown_seconds=60.0)

    assert pool.get_num_keys() == 2
    assert pool.mask_key(0) == "...1111"
    assert pool.get_available_key_index() == 0

    # Mark key 0 as rate limited
    pool.mark_key_rate_limited(0, cooldown_seconds=30.0)
    assert pool.get_available_key_index() == 1

    # Mark key 1 as rate limited
    pool.mark_key_rate_limited(1, cooldown_seconds=30.0)
    assert pool.is_pool_exhausted() is True
    assert pool.get_pool_cooldown_remaining() > 0.0


def test_ocr_caching_logic(mock_pdf_path, tmp_path, monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "fake_key_12345")
    cache_dir = tmp_path / "processed_ocr"

    # Explicitly request mistral engine – this test exercises ocr_client internals.
    processor = MistralPDFProcessor(output_dir=cache_dir, ocr_engine="mistral", inter_file_delay=0.0)
    assert processor.is_ocr_cached(mock_pdf_path) is False

    # Mock ocr_client.execute_ocr to return a dummy response
    processor.ocr_client.execute_ocr = MagicMock(return_value={"pages": [{"markdown": "Extracted OCR"}]})

    # First call: hits OCR execution
    res1 = processor.process_pdf(mock_pdf_path)
    assert res1["pages"][0]["markdown"] == "Extracted OCR"
    assert processor.ocr_client.execute_ocr.call_count == 1
    assert processor.stats["ocr_api_hits"] == 1

    # Second call: hits cache
    res2 = processor.process_pdf(mock_pdf_path)
    assert res2 == res1
    assert processor.ocr_client.execute_ocr.call_count == 1
    assert processor.stats["cache_hits"] == 1


def test_ocr_client_backoff_and_jitter(mock_pdf_path, tmp_path, monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "fake_key_12345")
    mock_sleep = MagicMock()
    monkeypatch.setattr("src.ingestion.ocr_loader.time.sleep", mock_sleep)

    # Explicitly request mistral engine – this test exercises pool_manager & backoff.
    processor = MistralPDFProcessor(
        output_dir=tmp_path,
        ocr_engine="mistral",
        request_delay=0.1,
        max_retries=2,
        initial_backoff=1.0,
        inter_file_delay=0.0,
    )

    mock_response = MagicMock()
    mock_response.model_dump.return_value = {"pages": [{"markdown": "Retry Success"}]}
    rate_limit_error = Exception("HTTP 429 Rate limit exceeded. Try again in 5 s")

    mock_client_instance = MagicMock()
    mock_client_instance.ocr.process = MagicMock(side_effect=[rate_limit_error, mock_response])

    with patch.object(processor.pool_manager, "create_client", return_value=mock_client_instance):
        res = processor.ocr_client.execute_ocr(mock_pdf_path)
        assert res == {"pages": [{"markdown": "Retry Success"}]}
        assert mock_client_instance.ocr.process.call_count == 2


def test_processor_stats_tracking(mock_pdf_path, tmp_path, monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "fake_key_12345")
    processor = MistralPDFProcessor(output_dir=tmp_path, inter_file_delay=0.0)

    stats = processor.get_processing_stats()
    assert "cache_hits" in stats
    assert "local_text_hits" in stats
    assert "ocr_api_hits" in stats
    assert "failures" in stats