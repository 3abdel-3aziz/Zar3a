"""
ingestion_pipeline.py

Central orchestrator for the Zar3a data ingestion pipeline.

Pipeline stages:
  1. Direct text ingestion   – scrape WikiSource via LLM extraction (DirectWebScraper)
  2. Multi-page PDF download – iterate Manshurat taxonomy pages (PDFDownloaderScraper)
  3. OCR processing          – run Mistral OCR on every downloaded PDF (MistralPDFProcessor)

All JSON outputs land in OUTPUT_JSON_DIR (Data/processed_json/).
All raw PDFs land in ARABIC_DATA_DIR  (Data/Arabic_Data/).
"""

import sys
import json
import asyncio
import logging
from pathlib import Path
from typing import List, Optional

from .arabic_text_normalizer import normalize_arabic_text

# ── Path bootstrap ────────────────────────────────────────────────────────────
# Ensures the project root (D:\ITI\Zar3a) is on sys.path so that `config`
# and `src.*` are importable whether this file is run directly
# (`python src/ingestion/ingestion_pipeline.py`) or imported by pytest / main.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# ─────────────────────────────────────────────────────────────────────────────


# pyrefly: ignore [missing-import]
from config import (
    OUTPUT_JSON_DIR,
    ARABIC_DATA_DIR,
    DIRECT_TEXT_URL,
    PAGINATION_PDF_URL,
)

# pyrefly: ignore [missing-import]
from src.ingestion.web_scraper_direct import DirectWebScraper

# pyrefly: ignore [missing-import]
from src.ingestion.web_scraper_pdf import PDFDownloaderScraper

# pyrefly: ignore [missing-import]
from src.ingestion.local_pdf_loader import LocalPDFLoader

# pyrefly: ignore [missing-import]
from src.ingestion.ocr_loader import MistralPDFProcessor


# ── Module-level logger ───────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("Zar3a.IngestionPipeline")


# ── Stage helpers ─────────────────────────────────────────────────────────────


async def run_direct_text_ingestion(
    url: str = DIRECT_TEXT_URL,
    output_filename: str = "law_119_2008_wikisource",
    output_dir: Optional[Path] = None,
) -> Optional[Path]:
    """
    Stage 1 – Direct text ingestion.

    Scrapes the given URL using ``DirectWebScraper`` with LLM extraction and
    saves the structured JSON result into the central JSON output directory.

    Returns the path to the saved JSON file, or ``None`` on failure.
    """
    effective_output_dir = output_dir or OUTPUT_JSON_DIR

    logger.info("=== Stage 1: Direct Text Ingestion ===")
    logger.info(f"Target URL : {url}")
    logger.info(f"Output dir : {effective_output_dir}")

    try:
        scraper = DirectWebScraper(output_dir=effective_output_dir)
        saved_path = await scraper.process_url(url, output_filename)

        if saved_path:
            logger.info(f"Stage 1 complete. JSON saved to: {saved_path}")
        else:
            logger.warning("Stage 1 returned no output – page may have failed to scrape.")

        return saved_path

    except Exception as exc:
        logger.error(f"Stage 1 failed with an unexpected error: {exc}")
        return None


async def run_pdf_download(
    url: str = PAGINATION_PDF_URL,
    pdf_output_dir: Optional[Path] = None,
    max_pages: int = 5,
) -> List[Path]:
    """
    Stage 2 – Multi-page PDF download.

    Iterates through paginated listing pages starting at ``url`` using
    ``PDFDownloaderScraper``.  Each discovered PDF is downloaded and stored
    in ``pdf_output_dir`` (defaults to ``ARABIC_DATA_DIR``).

    Returns the list of downloaded PDF ``Path`` objects.
    """
    effective_pdf_dir = pdf_output_dir or ARABIC_DATA_DIR

    logger.info("=== Stage 2: Multi-Page PDF Download ===")
    logger.info(f"Target URL : {url}")
    logger.info(f"PDF dir    : {effective_pdf_dir}")
    logger.info(f"Max pages  : {max_pages}")

    try:
        scraper = PDFDownloaderScraper(output_dir=effective_pdf_dir)
        downloaded = await scraper.scrape_and_download_workflow(url, max_pages=max_pages)
        logger.info(f"Stage 2 complete. {len(downloaded)} PDF(s) downloaded.")
        return downloaded

    except Exception as exc:
        logger.error(f"Stage 2 failed with an unexpected error: {exc}")
        return []


def run_ocr_processing(
    pdf_dir: Optional[Path] = None,
    json_output_dir: Optional[Path] = None,
    ocr_engine: Optional[str] = None,
    surya_backend: Optional[str] = None,
) -> List[Path]:
    """
    Stage 3 – OCR processing.

    Scans ``pdf_dir`` (defaults to ``ARABIC_DATA_DIR``) for PDF files,
    runs each through ``MistralPDFProcessor`` (defaulting to 'surya' OCR engine),
    and stores the resulting JSON cache files in ``json_output_dir`` (defaults to ``OUTPUT_JSON_DIR``).

    Idempotent: if a JSON cache file already exists for a PDF it is counted as
    a cache hit and returned immediately without re-running OCR.

    Returns the list of JSON ``Path`` objects that were produced or already cached.
    """
    effective_pdf_dir = pdf_dir or ARABIC_DATA_DIR
    effective_json_dir = json_output_dir or OUTPUT_JSON_DIR

    logger.info("=== Stage 3: OCR Processing ===")
    logger.info(f"PDF source : {effective_pdf_dir}")
    logger.info(f"JSON out   : {effective_json_dir}")

    try:
        loader = LocalPDFLoader(data_dir=effective_pdf_dir)
        batch = loader.load_folder_batch()

        if not batch:
            logger.info("Stage 3: No PDF files found to process.")
            return []

        logger.info(f"Stage 3: {len(batch)} PDF(s) found in source directory.")
        ocr_processor = MistralPDFProcessor(
            output_dir=effective_json_dir,
            ocr_engine=ocr_engine,
            surya_backend=surya_backend,
        )
        produced_jsons: List[Path] = []
        pre_cached_count = 0

        for item in batch:
            pdf_path = Path(item["file_path"])

            # ── Idempotency pre-flight check ──────────────────────────────────
            # If the JSON output already exists, skip OCR entirely and collect
            # the cached path. This makes repeated pipeline runs a no-op for
            # already-processed files.
            json_path = ocr_processor._get_expected_json_path(pdf_path)
            if json_path.exists():
                pre_cached_count += 1
                logger.info(
                    "Stage 3 [SKIP] Already cached – skipping OCR for: %s -> %s",
                    pdf_path.name,
                    json_path.name,
                )
                produced_jsons.append(json_path)
                continue
            # ──────────────────────────────────────────────────────────────────

            try:
                ocr_processor.process_pdf(pdf_path)
                produced_jsons.append(json_path)
            except Exception as exc:
                logger.error(f"OCR failed for {pdf_path.name}: {exc}")
                continue

        stats = ocr_processor.get_processing_stats()
        logger.info(
            "Stage 3 complete. %d JSON file(s) available "
            "(pre-cached: %d, cache_hits: %d, local_text: %d, surya_ocr: %d, paddle_ocr: %d, api_ocr: %d, failures: %d).",
            len(produced_jsons),
            pre_cached_count,
            stats.get("cache_hits", 0),
            stats.get("local_text_hits", 0),
            stats.get("surya_ocr_hits", 0),
            stats.get("paddle_ocr_hits", 0),
            stats.get("ocr_api_hits", 0),
            stats.get("failures", 0),
        )
        return produced_jsons

    except Exception as exc:
        logger.error(f"Stage 3 failed with an unexpected error: {exc}")
        return []


# ── Pipeline entry-point ──────────────────────────────────────────────────────


async def run_full_pipeline(
    direct_text_url: str = DIRECT_TEXT_URL,
    pagination_pdf_url: str = PAGINATION_PDF_URL,
    pdf_max_pages: int = 5,
    pdf_output_dir: Optional[Path] = None,
    json_output_dir: Optional[Path] = None,
    skip_scraping: bool = False,
    ocr_engine: Optional[str] = None,
    surya_backend: Optional[str] = None,
) -> dict:
    """
    Full ingestion pipeline orchestrator.

    Executes stages 1 → 2 → 3 in sequence (or skips stages 1 & 2 if skip_scraping=True)
    and returns a summary dict.
    """
    effective_json_dir = json_output_dir or OUTPUT_JSON_DIR
    effective_pdf_dir = pdf_output_dir or ARABIC_DATA_DIR

    logger.info("========================================")
    logger.info("   Zar3a Ingestion Pipeline – START    ")
    logger.info("========================================")

    direct_json: Optional[Path] = None
    downloaded_pdfs: List[Path] = []

    if not skip_scraping:
        # Stage 1
        direct_json = await run_direct_text_ingestion(
            url=direct_text_url,
            output_dir=effective_json_dir,
        )

        # Stage 2
        downloaded_pdfs = await run_pdf_download(
            url=pagination_pdf_url,
            pdf_output_dir=effective_pdf_dir,
            max_pages=pdf_max_pages,
        )
    else:
        logger.info("Skipping Stage 1 (Direct Text Ingestion) & Stage 2 (PDF Download) as requested.")
        logger.info(f"Proceeding directly to Stage 3 (OCR) using local PDF files in: {effective_pdf_dir}")

    # Stage 3
    ocr_jsons = run_ocr_processing(
        pdf_dir=effective_pdf_dir,
        json_output_dir=effective_json_dir,
        ocr_engine=ocr_engine,
        surya_backend=surya_backend,
    )

    summary = {
        "direct_text_json": direct_json,
        "downloaded_pdfs": downloaded_pdfs,
        "ocr_jsons": ocr_jsons,
    }

    logger.info("========================================")
    logger.info("   Zar3a Ingestion Pipeline – DONE     ")
    logger.info(f"   Direct JSON  : {direct_json}")
    logger.info(f"   PDFs Scraped : {len(downloaded_pdfs)}")
    logger.info(f"   OCR JSONs    : {len(ocr_jsons)}")
    logger.info("========================================")

    return summary


# ── Data Normalization & Loader Helpers ─────────────────────────────────────

def extract_text_from_json_data(raw_data: dict) -> str:
    """
    Extracts plain text or markdown from arbitrary JSON schemas
    (Direct web scraping schema, Mistral OCR schema, Surya OCR schema, or
    generic dictionary) and applies Arabic text normalisation to fix the
    reversed/mirrored character output produced by Surya OCR on CPU.
    """
    if not isinstance(raw_data, dict):
        return ""

    raw_text: str = ""

    # 1. Direct Web Scraper format: {"source_type": "...", "data": {"title": "...", "content": "..."}}
    if "data" in raw_data and isinstance(raw_data["data"], dict):
        nested_data = raw_data["data"]
        content = nested_data.get("content") or nested_data.get("full_text") or nested_data.get("text")
        if content and isinstance(content, str):
            raw_text = content

    # 2. Direct keys in root
    if not raw_text:
        if "content" in raw_data and isinstance(raw_data["content"], str):
            raw_text = raw_data["content"]
        elif "full_text" in raw_data and isinstance(raw_data["full_text"], str):
            raw_text = raw_data["full_text"]
        elif "text" in raw_data and isinstance(raw_data["text"], str):
            raw_text = raw_data["text"]

    # 3. Mistral / Surya OCR page format: {"pages": [{"markdown": "..."}, ...]}
    if not raw_text and "pages" in raw_data and isinstance(raw_data["pages"], list):
        markdown_pages: List[str] = []
        for page in raw_data["pages"]:
            if isinstance(page, dict) and "markdown" in page and isinstance(page["markdown"], str):
                markdown_pages.append(page["markdown"])
        if markdown_pages:
            raw_text = "\n\n".join(markdown_pages)

    if not raw_text:
        return ""

    # ── Arabic normalisation ────────────────────────────────────────────────
    # Fix reversed/mirrored Arabic characters produced by Surya OCR on CPU.
    # normalize_arabic_text() is a no-op for non-Arabic (English/numbers-only)
    # lines, so this is safe to apply unconditionally to all document sources.
    return normalize_arabic_text(raw_text)


def parse_single_json_document(json_path: Path) -> dict:
    """
    Reads a single JSON file and extracts normalized document dictionary:
    {
        "doc_id": "...",
        "file_name": "...",
        "file_path": "...",
        "full_text": "..."
    }
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file does not exist: {json_path}")

    doc_id = json_path.stem
    file_name = json_path.name
    file_path = str(json_path.resolve())

    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    full_text = extract_text_from_json_data(raw_data)
    return {
        "doc_id": doc_id,
        "file_name": file_name,
        "file_path": file_path,
        "full_text": full_text,
    }


def load_and_normalize_ingested_documents(
    json_dir: Optional[Path] = None,
) -> List[dict]:
    """
    Reads all JSON output files from json_dir (defaults to OUTPUT_JSON_DIR),
    normalizes their schemas into a clean list of document dictionaries ready for consumption:

    [
        {
            "doc_id": "...",
            "file_name": "...",
            "file_path": "...",
            "full_text": "..."
        },
        ...
    ]
    """
    effective_dir = Path(json_dir or OUTPUT_JSON_DIR)
    if not effective_dir.exists() or not effective_dir.is_dir():
        logger.warning("JSON output directory '%s' does not exist or is not a directory.", effective_dir)
        return []

    json_files = list(effective_dir.glob("*.json"))
    if not json_files:
        logger.warning("No JSON files found in '%s' to load.", effective_dir)
        return []

    logger.info("Loading and normalizing %d JSON document(s) from '%s'...", len(json_files), effective_dir)
    documents: List[dict] = []

    for json_path in json_files:
        try:
            doc_dict = parse_single_json_document(json_path)
            if doc_dict["full_text"].strip():
                documents.append(doc_dict)
            else:
                logger.warning("Skipping '%s' as it contains no extractable text.", json_path.name)
        except Exception as exc:
            logger.error("Failed to read/normalize JSON file '%s': %s", json_path.name, exc)

    logger.info("Successfully normalized %d valid document(s).", len(documents))
    return documents


# ── CLI execution ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    asyncio.run(run_full_pipeline())
