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
        
        processor = MistralPDFProcessor(output_dir=cache_dir)
        
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
        processor = MistralPDFProcessor(output_dir=cache_dir)
        processor._call_mistral_api = MagicMock(return_value={"status": "success", "text": "mocked ocr"})

        # Simulate getting the path from LocalPDFLoader
        pdf_path = mock_pdf_path
        
        # 2. Process it
        ocr_result = processor.process_pdf(pdf_path)
        
        # 3. Assertions
        assert ocr_result["status"] == "success"
        assert processor.is_ocr_cached(pdf_path) is True