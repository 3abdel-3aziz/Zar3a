import os
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Set
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler

# Importing the central configuration path from config file
# pyrefly: ignore [missing-import]
from config import DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


class PDFDownloaderScraper:
    """
    A specialized scraper designed for multi-page navigation, extracting titles/links,
    interacting with detail pages, finding PDF download buttons, and storing them
    centrally in Data/Arabic_Data.
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        detail_url_prefixes: Optional[List[str]] = None,
    ):
        """
        Initializes the PDFDownloaderScraper with a secure output directory pointing to Data/Arabic_Data.

        Args:
            output_dir: Directory where downloaded PDFs will be stored.
            detail_url_prefixes: Path prefixes that identify document detail pages
                (e.g. ``["/content/", "/node/"]``).  Only links matching at least
                one prefix — or direct ``.pdf`` links — will be followed.  Defaults
                to a sensible set for manshurat.org.
        """
        self.output_dir = output_dir or (Path(DATA_DIR) / "Arabic_Data")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Path prefixes that indicate a document detail page worth crawling.
        self.detail_url_prefixes: List[str] = detail_url_prefixes or [
            "/content/",
            "/node/",
        ]

        logger.info(f"PDFDownloaderScraper initialized. PDFs will be saved to: {self.output_dir}")

    def download_pdf(self, pdf_url: str, filename: str) -> Optional[Path]:
        """
        Downloads a binary PDF file from a given URL and stores it in the target directory.

        Skips the download entirely and returns the existing path if a non-empty file with
        the same name already exists in ``self.output_dir``.
        """
        if not filename.lower().endswith(".pdf"):
            filename += ".pdf"

        file_path = self.output_dir / filename

        # ── Idempotency guard ─────────────────────────────────────────────────
        if file_path.exists() and file_path.stat().st_size > 0:
            logger.info(f"File already exists, skipping download: {file_path}")
            return file_path
        # ─────────────────────────────────────────────────────────────────────

        try:
            response = requests.get(pdf_url, stream=True, timeout=120)
            response.raise_for_status()

            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            logger.info(f"Successfully downloaded PDF to: {file_path}")
            return file_path

        except Exception as e:
            logger.error(f"Failed to download PDF from {pdf_url}. Error: {e}")
            return None

    def extract_links_from_html(self, html_content: str, base_url: str) -> List[Dict[str, str]]:
        """
        Parses raw HTML content to extract document detail links and direct PDF URLs.

        Only links whose path starts with one of ``self.detail_url_prefixes`` or
        whose path ends with ``.pdf`` are returned.  Pagination navigation links
        (``rel="next/prev"``, Arabic "التالي"/"السابق" text) are always skipped.
        """
        soup = BeautifulSoup(html_content, "html.parser")
        extracted_items: List[Dict[str, str]] = []
        seen_urls: Set[str] = set()

        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"].strip()
            if not href or href.startswith("javascript:") or href.startswith("#"):
                continue

            rel = a_tag.get("rel", [])
            rel_str = " ".join(rel).lower() if isinstance(rel, list) else str(rel).lower()
            text = a_tag.get_text(strip=True)

            # Skip pagination navigation links
            if (
                "next" in rel_str
                or "prev" in rel_str
                or "التالي" in text
                or "السابق" in text
            ):
                continue

            full_url = urljoin(base_url, href)
            parsed_path = urlparse(full_url).path

            is_direct_pdf = parsed_path.lower().endswith(".pdf")
            is_detail_page = any(
                parsed_path.startswith(prefix) for prefix in self.detail_url_prefixes
            )

            # Only collect URLs that are document detail pages or direct PDF links
            if not is_direct_pdf and not is_detail_page:
                continue

            if full_url in seen_urls:
                continue
            seen_urls.add(full_url)

            title = text.strip() or a_tag.get("title", "").strip() or "Untitled"

            extracted_items.append({
                "url": full_url,
                "title": title,
                "is_pdf": is_direct_pdf,
            })

        return extracted_items

    def find_pdf_download_url(self, html_content: str, base_url: str) -> Optional[str]:
        """
        Inspects a detail page HTML content to locate direct PDF download link/button.
        """
        soup = BeautifulSoup(html_content, "html.parser")

        # 1. Look for <a> tags with .pdf in href
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"].strip()
            full_url = urljoin(base_url, href)
            if full_url.lower().endswith(".pdf"):
                return full_url

        # 2. Look for buttons or links containing download or pdf keywords
        for a_tag in soup.find_all(["a", "button"], href=True):
            href = a_tag.get("href", "").strip()
            text = a_tag.get_text(strip=True).lower()
            tag_class = " ".join(a_tag.get("class", [])).lower()
            tag_id = a_tag.get("id", "").lower()

            keywords = ["pdf", "download", "تحميل"]
            if any(k in href.lower() or k in text or k in tag_class or k in tag_id for k in keywords):
                if href:
                    return urljoin(base_url, href)

        return None

    def find_next_page_url(self, html_content: str, base_url: str) -> Optional[str]:
        """
        Finds the pagination 'Next' link or next page URL from list page HTML.
        """
        soup = BeautifulSoup(html_content, "html.parser")

        for a_tag in soup.find_all("a", href=True):
            rel = a_tag.get("rel", [])
            if isinstance(rel, list):
                rel = " ".join(rel).lower()
            else:
                rel = str(rel).lower()

            text = a_tag.get_text(strip=True).lower()
            tag_class = " ".join(a_tag.get("class", [])).lower()

            if "next" in rel or "next" in text or "التالي" in text or "next" in tag_class:
                return urljoin(base_url, a_tag["href"].strip())

        return None

    async def _fetch_page(self, crawler: AsyncWebCrawler, url: str) -> Optional[str]:
        """
        Fetches page HTML content asynchronously using Crawl4AI.
        """
        try:
            result = await crawler.arun(url=url, bypass_cache=True)
            if result.success:
                return result.html
            else:
                logger.error(f"Failed to fetch page at {url}: {result.error_message}")
                return None
        except Exception as e:
            logger.error(f"Error fetching page at {url}: {e}")
            return None

    async def scrape_and_download_workflow(self, start_url: str, max_pages: int = 5) -> List[Path]:
        """
        Main orchestration workflow for pagination, detail page navigation, finding PDF links,
        and executing downloads into Data/Arabic_Data.
        """
        logger.info(f"Starting PDF scraping workflow from: {start_url} (Max pages: {max_pages})")
        downloaded_files: List[Path] = []
        current_url: Optional[str] = start_url
        pages_visited = 0
        visited_urls: Set[str] = set()

        async with AsyncWebCrawler(verbose=False) as crawler:
            while current_url and pages_visited < max_pages:
                if current_url in visited_urls:
                    logger.info(f"URL already visited: {current_url}. Stopping pagination loop.")
                    break

                visited_urls.add(current_url)
                pages_visited += 1
                logger.info(f"Processing page {pages_visited}/{max_pages}: {current_url}")

                html_content = await self._fetch_page(crawler, current_url)
                if not html_content:
                    logger.warning(f"Skipping unreadable page: {current_url}")
                    break

                items = self.extract_links_from_html(html_content, current_url)
                logger.info(f"Extracted {len(items)} items/links from page {pages_visited}")

                for idx, item in enumerate(items, 1):
                    item_url = item["url"]
                    item_title = item["title"]

                    safe_title = "".join(
                        c if c.isalnum() or c in (" ", "_", "-") else "_" for c in item_title
                    ).strip()
                    safe_title = safe_title or f"document_p{pages_visited}_i{idx}"
                    filename = f"{safe_title}.pdf"
                    expected_path = self.output_dir / filename

                    if expected_path.exists() and expected_path.stat().st_size > 0:
                        logger.info(f"File already exists, skipping download: {expected_path}")
                        downloaded_files.append(expected_path)
                        continue

                    if item["is_pdf"]:
                        pdf_url = item_url
                    else:
                        detail_html = await self._fetch_page(crawler, item_url)
                        if not detail_html:
                            continue
                        pdf_url = self.find_pdf_download_url(detail_html, item_url)

                    if pdf_url:
                        downloaded_path = self.download_pdf(pdf_url, filename)
                        if downloaded_path:
                            downloaded_files.append(downloaded_path)

                current_url = self.find_next_page_url(html_content, current_url)

        logger.info(f"PDF scraping workflow completed. Downloaded {len(downloaded_files)} PDF(s).")
        return downloaded_files
