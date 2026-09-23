import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

# Ensure project root directory is in sys.path when running via python directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# pyrefly: ignore [missing-import]
from src.ingestion.web_scraper_pdf import PDFDownloaderScraper


def test_init_default_dir(tmp_path, monkeypatch):
    """Test PDFDownloaderScraper initialization with default directory."""
    monkeypatch.setattr("src.ingestion.web_scraper_pdf.DATA_DIR", tmp_path)
    scraper = PDFDownloaderScraper()
    assert scraper.output_dir == tmp_path / "Arabic_Data"
    assert scraper.output_dir.exists()


def test_init_custom_dir(tmp_path):
    """Test PDFDownloaderScraper initialization with custom directory."""
    custom_dir = tmp_path / "custom_pdf_dir"
    assert not custom_dir.exists()
    scraper = PDFDownloaderScraper(output_dir=custom_dir)
    assert custom_dir.exists()
    assert scraper.output_dir == custom_dir


def test_download_pdf_success(tmp_path):
    """Test downloading a binary PDF file successfully using stream mock."""
    scraper = PDFDownloaderScraper(output_dir=tmp_path)
    
    fake_pdf_data = b"%PDF-1.4 Fake PDF Content Header and Bytes"
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.iter_content = MagicMock(return_value=[fake_pdf_data])

    with patch("src.ingestion.web_scraper_pdf.requests.get", return_value=mock_response):
        downloaded_path = scraper.download_pdf("https://example.com/test_doc.pdf", "test_doc")

    assert downloaded_path is not None
    assert downloaded_path.exists()
    assert downloaded_path.name == "test_doc.pdf"
    assert downloaded_path.read_bytes() == fake_pdf_data


def test_download_pdf_failure(tmp_path):
    """Test handling download errors gracefully when requests fail."""
    scraper = PDFDownloaderScraper(output_dir=tmp_path)

    with patch("src.ingestion.web_scraper_pdf.requests.get", side_effect=Exception("Connection refused")):
        downloaded_path = scraper.download_pdf("https://example.com/failed.pdf", "failed_doc")

    assert downloaded_path is None


def test_download_pdf_skips_existing(tmp_path):
    """Test smart skipping when target PDF file already exists and is non-empty."""
    scraper = PDFDownloaderScraper(output_dir=tmp_path)
    existing_file = tmp_path / "already_downloaded.pdf"
    existing_file.write_bytes(b"%PDF existing content")

    with patch("src.ingestion.web_scraper_pdf.requests.get") as mock_get:
        result_path = scraper.download_pdf("https://example.com/already_downloaded.pdf", "already_downloaded.pdf")

    assert result_path == existing_file
    assert result_path.exists()
    assert result_path.read_bytes() == b"%PDF existing content"
    mock_get.assert_not_called()


def test_download_pdf_redownloads_empty_file(tmp_path):
    """Test re-downloading when target PDF file exists but is empty (0 bytes)."""
    scraper = PDFDownloaderScraper(output_dir=tmp_path)
    empty_file = tmp_path / "empty_doc.pdf"
    empty_file.write_bytes(b"")

    fake_pdf_data = b"%PDF-1.4 Fresh Bytes"
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.iter_content = MagicMock(return_value=[fake_pdf_data])

    with patch("src.ingestion.web_scraper_pdf.requests.get", return_value=mock_response) as mock_get:
        result_path = scraper.download_pdf("https://example.com/empty_doc.pdf", "empty_doc.pdf")

    assert result_path == empty_file
    assert result_path.read_bytes() == fake_pdf_data
    mock_get.assert_called_once()


def test_extract_links_from_html():
    """Test extracting item links and identifying PDF links from raw list page HTML."""
    # Pass /detail/ as a recognised prefix so /detail/doc2 is treated as a detail page
    scraper = PDFDownloaderScraper(detail_url_prefixes=["/content/", "/node/", "/detail/"])
    html_content = """
    <html>
        <body>
            <a href="/doc1.pdf">Document One (PDF)</a>
            <a href="/detail/doc2">Document Two Detail Page</a>
            <a href="javascript:void(0)">Ignored JS Link</a>
        </body>
    </html>
    """
    items = scraper.extract_links_from_html(html_content, "https://example.com")

    assert len(items) == 2
    assert items[0]["url"] == "https://example.com/doc1.pdf"
    assert items[0]["is_pdf"] is True
    assert items[1]["url"] == "https://example.com/detail/doc2"
    assert items[1]["is_pdf"] is False


def test_find_pdf_download_url():
    """Test finding direct PDF download URLs or buttons on detail page HTML."""
    scraper = PDFDownloaderScraper()

    # Case 1: Direct link ending in .pdf
    html_direct = '<html><body><a href="/downloads/file.pdf">Download PDF File</a></body></html>'
    pdf_url = scraper.find_pdf_download_url(html_direct, "https://example.com/detail")
    assert pdf_url == "https://example.com/downloads/file.pdf"

    # Case 2: Arabic button text or class
    html_arabic = '<html><body><a href="/download_action?id=123" class="btn-download">تحميل المستند</a></body></html>'
    pdf_url_arabic = scraper.find_pdf_download_url(html_arabic, "https://example.com/detail")
    assert pdf_url_arabic == "https://example.com/download_action?id=123"


def test_find_next_page_url():
    """Test parsing pagination 'Next' link from HTML."""
    scraper = PDFDownloaderScraper()
    html_pagination = """
    <div class="pagination">
        <a href="/list?page=1">1</a>
        <a href="/list?page=2" rel="next">التالي</a>
    </div>
    """
    next_url = scraper.find_next_page_url(html_pagination, "https://example.com/list?page=1")
    assert next_url == "https://example.com/list?page=2"


@pytest.mark.anyio
async def test_scrape_and_download_workflow_mocked(tmp_path):
    """Test full multi-page orchestration workflow with mocked crawler and downloader."""
    scraper = PDFDownloaderScraper(output_dir=tmp_path)

    page1_html = """
    <html>
        <body>
            <a href="/doc1.pdf">Doc 1 PDF</a>
            <a href="/list?page=2" rel="next">Next Page</a>
        </body>
    </html>
    """
    page2_html = """
    <html>
        <body>
            <a href="/doc2.pdf">Doc 2 PDF</a>
        </body>
    </html>
    """

    async def mock_fetch_page(crawler, url):
        if "page=2" in url:
            return page2_html
        return page1_html

    with patch.object(scraper, "_fetch_page", side_effect=mock_fetch_page):
        with patch.object(scraper, "download_pdf", side_effect=lambda url, name: tmp_path / f"{name}.pdf"):
            # Create dummy files when download_pdf is invoked
            def fake_download(url, name):
                p = tmp_path / name
                p.write_bytes(b"%PDF dummy")
                return p
            scraper.download_pdf = fake_download

            downloaded_files = await scraper.scrape_and_download_workflow("https://example.com/list", max_pages=2)

    assert len(downloaded_files) == 2
    for p in downloaded_files:
        assert p.exists()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
