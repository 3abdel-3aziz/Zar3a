import sys
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

# Ensure project root directory is in sys.path when running via python directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# pyrefly: ignore [missing-import]
from src.ingestion.web_scraper_pdf import PDFDownloaderScraper


# Target URL constant specified in requirement
TARGET_MANSHURAT_URL = "https://manshurat.org/taxonomy/term/56?sort=search_api_aggregation_1%20DESC"


def test_pdf_downloader_scraper_init_default_dir(tmp_path, monkeypatch):
    """Test PDFDownloaderScraper initializes with central Data/Arabic_Data path."""
    monkeypatch.setattr("src.ingestion.web_scraper_pdf.DATA_DIR", tmp_path)
    scraper = PDFDownloaderScraper()
    assert scraper.output_dir == tmp_path / "Arabic_Data"
    assert scraper.output_dir.exists()


def test_manshurat_taxonomy_link_extraction():
    """Test extracting node/detail links from simulated Manshurat taxonomy term HTML."""
    scraper = PDFDownloaderScraper()
    manshurat_list_html = """
    <!DOCTYPE html>
    <html lang="ar">
        <body>
            <div class="views-row">
                <h2><a href="/content/sdr-qnwn-tnzym-hyz-lhywnt">قانون تنظيم حيازة الحيوانات الخطرة</a></h2>
            </div>
            <div class="views-row">
                <h2><a href="/node/74812">استراتيجية تغير المناخ 2050</a></h2>
            </div>
            <ul class="pager">
                <li class="pager-current">1</li>
                <li class="pager-next"><a href="/taxonomy/term/56?sort=search_api_aggregation_1%20DESC&page=1">التالي ›</a></li>
            </ul>
        </body>
    </html>
    """
    items = scraper.extract_links_from_html(manshurat_list_html, TARGET_MANSHURAT_URL)

    # Should extract 2 detail links (/content/... and /node/74812) and exclude pagination (/taxonomy/term/56...)
    assert len(items) == 2
    urls = [item["url"] for item in items]
    assert "https://manshurat.org/content/sdr-qnwn-tnzym-hyz-lhywnt" in urls
    assert "https://manshurat.org/node/74812" in urls


def test_manshurat_detail_pdf_url_discovery():
    """Test discovering PDF download button links on Manshurat detail pages (/node/74812)."""
    scraper = PDFDownloaderScraper()
    manshurat_detail_html = """
    <!DOCTYPE html>
    <html>
        <body>
            <div class="field-item">
                <a href="/file/87393/download?token=vGEM_pV-" class="btn btn-default">Download</a>
            </div>
        </body>
    </html>
    """
    pdf_url = scraper.find_pdf_download_url(manshurat_detail_html, "https://manshurat.org/node/74812")
    assert pdf_url == "https://manshurat.org/file/87393/download?token=vGEM_pV-"


def test_manshurat_pagination_next_url():
    """Test discovering next page pagination links on Manshurat list pages."""
    scraper = PDFDownloaderScraper()
    manshurat_pager_html = """
    <ul class="pager">
        <li class="pager-item"><a href="/taxonomy/term/56?sort=search_api_aggregation_1%20DESC&page=0">1</a></li>
        <li class="pager-next"><a href="/taxonomy/term/56?sort=search_api_aggregation_1%20DESC&page=1">التالي ›</a></li>
    </ul>
    """
    next_url = scraper.find_next_page_url(manshurat_pager_html, TARGET_MANSHURAT_URL)
    assert next_url == "https://manshurat.org/taxonomy/term/56?sort=search_api_aggregation_1%20DESC&page=1"


def test_chunked_pdf_download_success(tmp_path):
    """Test downloading binary PDF content with chunked streaming into output directory."""
    scraper = PDFDownloaderScraper(output_dir=tmp_path)
    fake_pdf_stream = [b"%PDF-1.5 Chunk 1 ", b"Chunk 2 Binary Content ", b"EOF"]

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.iter_content = MagicMock(return_value=fake_pdf_stream)

    pdf_url = "https://manshurat.org/file/87393/download?token=vGEM_pV-"
    filename = "manshurat_law_74812.pdf"

    with patch("src.ingestion.web_scraper_pdf.requests.get", return_value=mock_response):
        downloaded_path = scraper.download_pdf(pdf_url, filename)

    assert downloaded_path is not None
    assert downloaded_path.exists()
    assert downloaded_path.name == "manshurat_law_74812.pdf"
    assert downloaded_path.read_bytes() == b"".join(fake_pdf_stream)


def test_chunked_pdf_download_failure(tmp_path):
    """Test error handling when binary PDF download request fails."""
    scraper = PDFDownloaderScraper(output_dir=tmp_path)

    with patch("src.ingestion.web_scraper_pdf.requests.get", side_effect=Exception("HTTP 500 Server Error")):
        downloaded_path = scraper.download_pdf("https://manshurat.org/file/999/download", "failed_doc.pdf")

    assert downloaded_path is None


@pytest.mark.anyio
async def test_manshurat_multi_page_workflow_mocked(tmp_path):
    """Test multi-page pagination, navigation, and PDF downloading workflow end-to-end with mocks."""
    scraper = PDFDownloaderScraper(output_dir=tmp_path)

    page1_html = """
    <html>
        <body>
            <div class="views-row"><a href="/node/101">Law 101</a></div>
            <ul class="pager"><li class="pager-next"><a href="/taxonomy/term/56?page=1">التالي</a></li></ul>
        </body>
    </html>
    """
    detail101_html = """
    <html>
        <body>
            <a href="/file/101/download?token=abc">Download</a>
        </body>
    </html>
    """
    page2_html = """
    <html>
        <body>
            <div class="views-row"><a href="/node/102">Law 102</a></div>
        </body>
    </html>
    """
    detail102_html = """
    <html>
        <body>
            <a href="/file/102/download?token=xyz">Download</a>
        </body>
    </html>
    """

    async def mock_fetch_page(crawler, url):
        if "node/101" in url:
            return detail101_html
        elif "node/102" in url:
            return detail102_html
        elif "page=1" in url:
            return page2_html
        return page1_html

    with patch.object(scraper, "_fetch_page", side_effect=mock_fetch_page):
        def fake_download(pdf_url, filename):
            p = tmp_path / filename
            p.write_bytes(b"%PDF Fake Content")
            return p

        with patch.object(scraper, "download_pdf", side_effect=fake_download):
            downloaded_files = await scraper.scrape_and_download_workflow(TARGET_MANSHURAT_URL, max_pages=2)

    assert len(downloaded_files) == 2
    assert all(f.exists() for f in downloaded_files)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
