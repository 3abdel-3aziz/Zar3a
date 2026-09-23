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
import asyncio
import logging
from pathlib import Path
from typing import List, Optional

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
) -> List[Path]:
    """
    Stage 3 – OCR processing.

    Scans ``pdf_dir`` (defaults to ``ARABIC_DATA_DIR``) for PDF files,
    runs each through ``MistralPDFProcessor``, and stores the resulting
    JSON cache files in ``json_output_dir`` (defaults to ``OUTPUT_JSON_DIR``).

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

        logger.info(f"Stage 3: {len(batch)} PDF(s) queued for OCR.")
        ocr_processor = MistralPDFProcessor(output_dir=effective_json_dir)
        produced_jsons: List[Path] = []

        for item in batch:
            pdf_path = Path(item["file_path"])
            try:
                ocr_processor.process_pdf(pdf_path)
                json_path = ocr_processor._get_expected_json_path(pdf_path)
                produced_jsons.append(json_path)
            except Exception as exc:
                logger.error(f"OCR failed for {pdf_path.name}: {exc}")
                continue

        logger.info(f"Stage 3 complete. {len(produced_jsons)} JSON file(s) produced.")
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
) -> dict:
    """
    Full ingestion pipeline orchestrator.

    Executes stages 1 → 2 → 3 in sequence and returns a summary dict:

    .. code-block:: python

        {
            "direct_text_json": Path | None,   # Stage 1 output
            "downloaded_pdfs" : [Path, ...],   # Stage 2 outputs
            "ocr_jsons"       : [Path, ...],   # Stage 3 outputs
        }
    """
    effective_json_dir = json_output_dir or OUTPUT_JSON_DIR
    effective_pdf_dir = pdf_output_dir or ARABIC_DATA_DIR

    logger.info("========================================")
    logger.info("   Zar3a Ingestion Pipeline – START    ")
    logger.info("========================================")

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

    # Stage 3
    ocr_jsons = run_ocr_processing(
        pdf_dir=effective_pdf_dir,
        json_output_dir=effective_json_dir,
    )

    summary = {
        "direct_text_json": direct_json,
        "downloaded_pdfs": downloaded_pdfs,
        "ocr_jsons": ocr_jsons,
    }

    logger.info("========================================")
    logger.info("   Zar3a Ingestion Pipeline – DONE     ")
    logger.info(f"   Direct JSON  : {direct_json}")
    logger.info(f"   PDFs         : {len(downloaded_pdfs)}")
    logger.info(f"   OCR JSONs    : {len(ocr_jsons)}")
    logger.info("========================================")

    return summary


# ── CLI execution ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    asyncio.run(run_full_pipeline())
