import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock

# Ensure project root directory is in sys.path when running via python directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# pyrefly: ignore [missing-import]
from src.ingestion.ocr_loader import MistralPDFProcessor


@pytest.fixture
def mock_pdf_path(tmp_path):
    """
    Fixture to create a dummy PDF file for testing purposes.
    """
    pdf_file = tmp_path / "sample.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 dummy pdf content")
    return pdf_file

def test_ocr_caching_logic(mock_pdf_path, tmp_path):
    """
    Test that the OCR processor checks and handles cache correctly 
    without hitting the real API (using mocking).
    """
    cache_dir = tmp_path / "processed_ocr"
    
    # Mock the Mistral client so it doesn't require a real API key for this test
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("MISTRAL_API_KEY", "fake_key_for_testing")
        
        processor = MistralPDFProcessor(output_dir=cache_dir, inter_file_delay=0.0)
        
        # Initially, it should not be cached
        assert processor.is_ocr_cached(mock_pdf_path) is False

        # Mock the internal API call method to return a dummy dictionary response
        processor._call_mistral_api = MagicMock(return_value={"pages": [{"markdown": "Extracted text"}]})
        
        # First process call (should hit the "API")
        result1 = processor.process_pdf(mock_pdf_path)
        assert result1["pages"][0]["markdown"] == "Extracted text"
        processor._call_mistral_api.assert_called_once()

        # Second process call (should load from local cache file and NOT call API again)
        result2 = processor.process_pdf(mock_pdf_path)
        assert result2 == result1
        # Call count should still be 1 because it loaded from cache!
        processor._call_mistral_api.assert_called_once()

def test_integration_loader_with_ocr(mock_pdf_path, tmp_path):
    """
    Integration Test: Verify that LocalPDFLoader output can flow 
    smoothly into the MistralPDFOCRProcessor workflow.
    """
    cache_dir = tmp_path / "processed_ocr"
    
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("MISTRAL_API_KEY", "fake_key_for_testing")
        
        # 1. Load the PDF using LocalPDFLoader (assuming it takes directory or file path)
        # Here we test passing the file path directly to the processor as received from a loader
        processor = MistralPDFProcessor(output_dir=cache_dir, inter_file_delay=0.0)
        processor._call_mistral_api = MagicMock(return_value={"status": "success", "text": "mocked ocr"})

        # Simulate getting the path from LocalPDFLoader
        pdf_path = mock_pdf_path
        
        # 2. Process it
        ocr_result = processor.process_pdf(pdf_path)
        
        # 3. Assertions
        assert ocr_result["status"] == "success"
        assert processor.is_ocr_cached(pdf_path) is True


def test_call_mistral_api_base64_payload(mock_pdf_path, tmp_path):
    """
    Test that _call_mistral_api reads the PDF in binary mode, encodes it into a
    Base64 Data URI, and passes the formatted payload to client.ocr.process.
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("MISTRAL_API_KEY", "fake_key_for_testing")
        processor = MistralPDFProcessor(output_dir=tmp_path)

        mock_response = MagicMock()
        mock_response.model_dump.return_value = {"pages": [{"markdown": "OCR text"}]}
        processor.client.ocr.process = MagicMock(return_value=mock_response)

        res = processor._call_mistral_api(mock_pdf_path)

        assert res == {"pages": [{"markdown": "OCR text"}]}
        processor.client.ocr.process.assert_called_once()
        call_kwargs = processor.client.ocr.process.call_args.kwargs
        assert call_kwargs["model"] == "mistral-ocr-latest"
        doc_payload = call_kwargs["document"]
        assert doc_payload["type"] == "document_url"
        assert doc_payload["document_url"].startswith("data:application/pdf;base64,")


def test_call_mistral_api_rate_limit_retry(mock_pdf_path, tmp_path, monkeypatch):
    """
    Test that _call_mistral_api catches HTTP 429 rate limit errors, sleeps for backoff duration,
    and succeeds when a retry succeeds.
    """
    monkeypatch.setenv("MISTRAL_API_KEY", "fake_key_for_testing")
    mock_sleep = MagicMock()
    monkeypatch.setattr("src.ingestion.ocr_loader.time.sleep", mock_sleep)

    processor = MistralPDFProcessor(
        output_dir=tmp_path,
        request_delay=0.1,
        max_retries=3,
        initial_backoff=1.0,
        backoff_factor=2.0
    )

    mock_response = MagicMock()
    mock_response.model_dump.return_value = {"pages": [{"markdown": "OCR retry text"}]}

    # Simulate 2 rate limit failures (HTTP 429) then 1 success
    rate_limit_error = Exception("API error occurred: Status 429. Body: Rate limit exceeded")
    processor.client.ocr.process = MagicMock(side_effect=[rate_limit_error, rate_limit_error, mock_response])

    res = processor._call_mistral_api(mock_pdf_path)

    assert res == {"pages": [{"markdown": "OCR retry text"}]}
    assert processor.client.ocr.process.call_count == 3
    # Check that time.sleep was called for backoff 1.0s, 2.0s, and request_delay 0.1s
    sleep_calls = [call[0][0] for call in mock_sleep.call_args_list]
    assert 1.0 in sleep_calls
    assert 2.0 in sleep_calls
    assert 0.1 in sleep_calls


def test_call_mistral_api_rate_limit_exceeded(mock_pdf_path, tmp_path, monkeypatch):
    """
    Test that _call_mistral_api raises exception after exhausting max retries on HTTP 429 errors.
    """
    monkeypatch.setenv("MISTRAL_API_KEY", "fake_key_for_testing")
    mock_sleep = MagicMock()
    monkeypatch.setattr("src.ingestion.ocr_loader.time.sleep", mock_sleep)

    processor = MistralPDFProcessor(
        output_dir=tmp_path,
        request_delay=0.0,
        max_retries=2,
        initial_backoff=1.0,
        backoff_factor=2.0
    )

    rate_limit_error = Exception("Status 429 Rate limit exceeded")
    processor.client.ocr.process = MagicMock(side_effect=[rate_limit_error, rate_limit_error])

    with pytest.raises(Exception, match="429"):
        processor._call_mistral_api(mock_pdf_path)

    assert processor.client.ocr.process.call_count == 2


def test_key_parsing_and_rotation(mock_pdf_path, tmp_path, monkeypatch):
    """
    Test parsing comma-separated keys from environment and rotating keys upon HTTP 429 errors.
    """
    from unittest.mock import patch

    raw_env_keys = " key_one_1111 , key_two_2222 , key_three_3333 "
    monkeypatch.setenv("MISTRAL_API_KEY", raw_env_keys)
    mock_sleep = MagicMock()
    monkeypatch.setattr("src.ingestion.ocr_loader.time.sleep", mock_sleep)

    mock_response = MagicMock()
    mock_response.model_dump.return_value = {"pages": [{"markdown": "Rotated text"}]}
    rate_limit_err = Exception("HTTP 429 Too Many Requests")

    mock_client = MagicMock()
    mock_client.ocr.process = MagicMock(side_effect=[rate_limit_err, mock_response])

    with patch("src.ingestion.ocr_loader.Mistral", return_value=mock_client):
        processor = MistralPDFProcessor(output_dir=tmp_path, request_delay=0.0)

        assert len(processor.api_keys) == 3
        assert processor.api_keys == ["key_one_1111", "key_two_2222", "key_three_3333"]
        assert processor.current_key_index == 0

        res = processor._call_mistral_api(mock_pdf_path)

        assert res == {"pages": [{"markdown": "Rotated text"}]}
        assert processor.current_key_index == 1
        assert processor.api_keys[processor.current_key_index] == "key_two_2222"


def test_pool_exhaustion_safety_delay(mock_pdf_path, tmp_path, monkeypatch):
    """
    Test that when all keys in the pool fail consecutively, safety delay is invoked before next round.
    """
    from unittest.mock import patch

    keys = ["keyA_aaaa", "keyB_bbbb"]
    mock_sleep = MagicMock()
    monkeypatch.setattr("src.ingestion.ocr_loader.time.sleep", mock_sleep)

    mock_response = MagicMock()
    mock_response.model_dump.return_value = {"pages": [{"markdown": "After safety delay"}]}
    rate_limit_err = Exception("429 rate limit exceeded")

    mock_client = MagicMock()
    mock_client.ocr.process = MagicMock(side_effect=[rate_limit_err, rate_limit_err, mock_response])

    with patch("src.ingestion.ocr_loader.Mistral", return_value=mock_client):
        processor = MistralPDFProcessor(
            api_key=keys,
            output_dir=tmp_path,
            request_delay=0.0,
            max_retries=2,
            safety_delay=45.0
        )

        res = processor._call_mistral_api(mock_pdf_path)

        assert res == {"pages": [{"markdown": "After safety delay"}]}
        sleep_calls = [call[0][0] for call in mock_sleep.call_args_list]
        assert 45.0 in sleep_calls


def test_inter_file_delay_enforcement(mock_pdf_path, tmp_path, monkeypatch):
    """
    Test that process_pdf enforces inter-file safety delay (default 8s) after API processing.
    """
    monkeypatch.setenv("MISTRAL_API_KEY", "fake_key")
    mock_sleep = MagicMock()
    monkeypatch.setattr("src.ingestion.ocr_loader.time.sleep", mock_sleep)

    processor = MistralPDFProcessor(output_dir=tmp_path, request_delay=0.0, inter_file_delay=8.0)
    processor._call_mistral_api = MagicMock(return_value={"pages": [{"markdown": "PDF text"}]})

    res = processor.process_pdf(mock_pdf_path)

    assert res == {"pages": [{"markdown": "PDF text"}]}
    mock_sleep.assert_called_with(8.0)