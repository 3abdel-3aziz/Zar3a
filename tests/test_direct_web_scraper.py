import os
import sys
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from pydantic import ValidationError

# Ensure project root directory is in sys.path when running via python directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# pyrefly: ignore [missing-import]
from src.ingestion.web_scraper_direct import DirectWebScraper, LawDocumentSchema


# ============================================================================
# 1. LawDocumentSchema Tests
# ============================================================================

def test_law_document_schema_valid():
    """Test that LawDocumentSchema instantiates correctly with required fields and defaults."""
    doc = LawDocumentSchema(
        title="Law No. 119 of 2008",
        content="Main body text and regulatory provisions."
    )
    assert doc.title == "Law No. 119 of 2008"
    assert doc.content == "Main body text and regulatory provisions."
    assert doc.metadata == {}


def test_law_document_schema_with_metadata():
    """Test that LawDocumentSchema handles custom metadata correctly."""
    doc = LawDocumentSchema(
        title="Law No. 119 of 2008",
        content="Main body text",
        metadata={"year": 2008, "articles_count": 50}
    )
    assert doc.metadata["year"] == 2008
    assert doc.metadata["articles_count"] == 50


def test_law_document_schema_missing_required_fields():
    """Test that LawDocumentSchema raises ValidationError when required fields are missing."""
    with pytest.raises(ValidationError):
        LawDocumentSchema(title="Only Title")


# ============================================================================
# 2. DirectWebScraper Initialization & API Key Validation Tests
# ============================================================================

def test_scraper_init_missing_api_key(monkeypatch):
    """Test that DirectWebScraper raises ValueError if MISTRAL_API_KEY is not set."""
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    with pytest.raises(ValueError, match="MISTRAL_API_KEY environment variable is missing"):
        DirectWebScraper()


def test_scraper_init_creates_directory(monkeypatch, tmp_path):
    """Test that DirectWebScraper initializes successfully and creates output_dir if missing."""
    monkeypatch.setenv("MISTRAL_API_KEY", "fake_test_key")
    target_dir = tmp_path / "custom_output"
    assert not target_dir.exists()

    scraper = DirectWebScraper(output_dir=target_dir, model_name="mistral-small-latest")
    assert target_dir.exists()
    assert scraper.output_dir == target_dir
    assert scraper.model_name == "mistral-small-latest"


# ============================================================================
# 3. Extraction Strategy Configuration Tests
# ============================================================================

def test_create_extraction_strategy(monkeypatch, tmp_path):
    """Test that _create_extraction_strategy builds an LLMExtractionStrategy instance."""
    monkeypatch.setenv("MISTRAL_API_KEY", "fake_test_key")
    scraper = DirectWebScraper(output_dir=tmp_path, model_name="mistral-small-latest")
    
    strategy = scraper._create_extraction_strategy()
    assert strategy.llm_config.provider == "mistral/mistral-small-latest"
    assert strategy.llm_config.api_token == "fake_test_key"
    assert strategy.extract_type == "schema"


# ============================================================================
# 4. Storage Functionality Tests
# ============================================================================

def test_store_final_output(monkeypatch, tmp_path):
    """Test storing the final JSON output payload into the target directory."""
    monkeypatch.setenv("MISTRAL_API_KEY", "fake_test_key")
    scraper = DirectWebScraper(output_dir=tmp_path)

    sample_data = {
        "title": "Law No. 119 of 2008",
        "content": "Sample content text.",
        "metadata": {"source": "test"}
    }
    filename = "test_law_output"
    output_path = scraper.store_final_output(sample_data, filename)

    assert output_path.exists()
    assert output_path.name == "test_law_output.json"

    with open(output_path, "r", encoding="utf-8") as f:
        stored_payload = json.load(f)

    assert stored_payload["source_type"] == "direct_web_scraping_llm"
    assert stored_payload["data"] == sample_data


# ============================================================================
# 5. Fetch Page Content Async Tests (Mocked)
# ============================================================================

@pytest.mark.anyio
async def test_fetch_page_content_success(monkeypatch, tmp_path):
    """Test _fetch_page_content when AsyncWebCrawler succeeds."""
    monkeypatch.setenv("MISTRAL_API_KEY", "fake_test_key")
    scraper = DirectWebScraper(output_dir=tmp_path)

    mock_run_result = MagicMock()
    mock_run_result.success = True
    mock_run_result.extracted_content = json.dumps({
        "title": "Law No. 119",
        "content": "Extracted text content"
    })

    mock_crawler_instance = AsyncMock()
    mock_crawler_instance.arun = AsyncMock(return_value=mock_run_result)
    mock_crawler_instance.__aenter__ = AsyncMock(return_value=mock_crawler_instance)
    mock_crawler_instance.__aexit__ = AsyncMock(return_value=None)

    with patch("src.ingestion.web_scraper_direct.AsyncWebCrawler", return_value=mock_crawler_instance):
        content = await scraper._fetch_page_content("https://example.com/law")

    assert content is not None
    parsed = json.loads(content)
    assert parsed["title"] == "Law No. 119"


@pytest.mark.anyio
async def test_fetch_page_content_failure(monkeypatch, tmp_path):
    """Test _fetch_page_content when AsyncWebCrawler fails."""
    monkeypatch.setenv("MISTRAL_API_KEY", "fake_test_key")
    scraper = DirectWebScraper(output_dir=tmp_path)

    mock_run_result = MagicMock()
    mock_run_result.success = False
    mock_run_result.error_message = "HTTP 404 Not Found"

    mock_crawler_instance = AsyncMock()
    mock_crawler_instance.arun = AsyncMock(return_value=mock_run_result)
    mock_crawler_instance.__aenter__ = AsyncMock(return_value=mock_crawler_instance)
    mock_crawler_instance.__aexit__ = AsyncMock(return_value=None)

    with patch("src.ingestion.web_scraper_direct.AsyncWebCrawler", return_value=mock_crawler_instance):
        content = await scraper._fetch_page_content("https://example.com/invalid")

    assert content is None


# ============================================================================
# 6. process_url Pipeline Orchestration Tests (Mocked)
# ============================================================================

@pytest.mark.anyio
async def test_process_url_success_single_dict(monkeypatch, tmp_path):
    """Test process_url when _fetch_page_content returns a JSON dict string."""
    monkeypatch.setenv("MISTRAL_API_KEY", "fake_test_key")
    scraper = DirectWebScraper(output_dir=tmp_path)

    extracted_dict = {"title": "Law 119", "content": "Articles provisions..."}
    
    with patch.object(scraper, "_fetch_page_content", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = json.dumps(extracted_dict)
        saved_path = await scraper.process_url("https://example.com", "out_dict")

    assert saved_path is not None
    assert saved_path.exists()
    with open(saved_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["data"]["title"] == "Law 119"


@pytest.mark.anyio
async def test_process_url_success_list_of_dicts(monkeypatch, tmp_path):
    """Test process_url when _fetch_page_content returns a JSON array string."""
    monkeypatch.setenv("MISTRAL_API_KEY", "fake_test_key")
    scraper = DirectWebScraper(output_dir=tmp_path)

    extracted_list = [{"title": "Law 119 List Item", "content": "List content"}]
    
    with patch.object(scraper, "_fetch_page_content", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = json.dumps(extracted_list)
        saved_path = await scraper.process_url("https://example.com", "out_list")

    assert saved_path is not None
    assert saved_path.exists()
    with open(saved_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["data"]["title"] == "Law 119 List Item"


@pytest.mark.anyio
async def test_process_url_fetch_failed(monkeypatch, tmp_path):
    """Test process_url when _fetch_page_content returns None."""
    monkeypatch.setenv("MISTRAL_API_KEY", "fake_test_key")
    scraper = DirectWebScraper(output_dir=tmp_path)

    with patch.object(scraper, "_fetch_page_content", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = None
        saved_path = await scraper.process_url("https://example.com", "out_fail")

    assert saved_path is None


@pytest.mark.anyio
async def test_process_url_invalid_json(monkeypatch, tmp_path):
    """Test process_url when _fetch_page_content returns invalid JSON string."""
    monkeypatch.setenv("MISTRAL_API_KEY", "fake_test_key")
    scraper = DirectWebScraper(output_dir=tmp_path)

    with patch.object(scraper, "_fetch_page_content", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = "INVALID_JSON_RESPONSE"
        saved_path = await scraper.process_url("https://example.com", "out_bad")

    assert saved_path is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))

