# ── Critical: set PaddlePaddle flags before ANY import ───────────────────────
# paddle/paddleocr/paddlex read these at C++ runtime initialization.
# Must appear before the docstring to guarantee they fire first.
import os
os.environ["FLAGS_use_mkldnn"] = "0"       # disable Intel OneDNN → prevents
os.environ["FLAGS_enable_pir_api"] = "0"   # disable PIR optimizer  → "ConvertPirAttribute2RuntimeAttribute" crash
# ─────────────────────────────────────────────────────────────────────────────

"""
main_orchestrator.py

High-Level End-to-End Orchestrator ("Maestro") for Zar3a Project.
Follows Clean Architecture & Separation of Concerns:
  - Database schema initialization (src.database.connection).
  - Ingestion pipeline trigger & normalized data fetching (src.ingestion.ingestion_pipeline).
  - Semantic chunking orchestration (src.chunkers.ChunkingPipeline).
  - Database persistence & duplicate filtering (src.database.operations).

Runnable via:
  uv run python main_orchestrator.py
"""


import sys
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Project imports
# pyrefly: ignore [missing-import]
from config import OUTPUT_JSON_DIR
from src.database.connection import engine, SessionLocal, Base
from src.database.operations import save_document_with_chunks
from src.chunkers.ChunkingPipeline import ChunkingPipeline
from src.ingestion.ingestion_pipeline import (
    run_full_pipeline,
    load_and_normalize_ingested_documents,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("Zar3a.MainOrchestrator")


# ── 1. Database Schema Initialization ────────────────────────────────────────

def init_database_tables() -> None:
    """
    Responsibility: Ensures all database tables (documents, document_chunks) exist.
    """
    logger.info("Initializing database schema...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database schema initialized successfully.")


# ── 2. Ingestion Trigger Stage ───────────────────────────────────────────────

async def run_ingestion_stage(
    pdf_max_pages: int = 5,
    json_output_dir: Optional[Path] = None,
    skip_scraping: bool = False,
    ocr_engine: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Responsibility: Triggers the Ingestion Pipeline to scrape text and process OCR.
    """
    logger.info("=== STEP 1: Executing Ingestion Pipeline ===")
    summary = await run_full_pipeline(
        pdf_max_pages=pdf_max_pages,
        json_output_dir=json_output_dir or OUTPUT_JSON_DIR,
        skip_scraping=skip_scraping,
        ocr_engine=ocr_engine,
    )
    logger.info("Ingestion execution complete. Summary: %s", summary)
    return summary


# ── 3. Single Document Processing ────────────────────────────────────────────

def process_single_document(
    doc_data: Dict[str, Any],
    chunker: ChunkingPipeline,
    db_session: Any,
) -> Tuple[int, int]:
    """
    Responsibility: Takes a normalized document dictionary, chunks it, and saves to DB.
    
    Expected doc_data keys:
      - doc_id: str
      - file_name: str
      - file_path: str
      - full_text: str
      
    Returns: (inserted_count, skipped_count)
    """
    doc_id = doc_data["doc_id"]
    file_name = doc_data["file_name"]
    file_path = doc_data["file_path"]
    full_text = doc_data["full_text"]

    logger.info("Processing document doc_id='%s' (%s)", doc_id, file_name)

    if not full_text or not full_text.strip():
        logger.warning("Document '%s' has empty text content. Skipping.", doc_id)
        return 0, 0

    # Execute semantic chunking pipeline
    chunks = chunker.run(full_text=full_text, doc_id=doc_id, cleaning_mode="full")
    if not chunks:
        logger.warning("No chunks generated for doc_id='%s'.", doc_id)
        return 0, 0

    # Persist document and chunks atomically in DB
    _, num_inserted, num_skipped = save_document_with_chunks(
        db=db_session,
        doc_id=doc_id,
        file_name=file_name,
        file_path=file_path,
        chunks_data=chunks,
    )
    return num_inserted, num_skipped


# ── 4. Chunking & Database Storage Stage ──────────────────────────────────────

def run_chunking_and_storage_stage(
    documents: List[Dict[str, Any]],
) -> Dict[str, int]:
    """
    Responsibility: Receives normalized documents, orchestrates chunking, and persists to DB.
    Returns aggregated run statistics.
    """
    logger.info("=== STEP 2: Executing Chunking & Database Storage Stage ===")
    
    if not documents:
        logger.warning("No normalized documents available to process.")
        return {"processed_docs": 0, "inserted_chunks": 0, "skipped_chunks": 0}

    logger.info("Processing %d normalized document(s)...", len(documents))

    chunker = ChunkingPipeline()
    total_inserted = 0
    total_skipped = 0
    processed_docs = 0

    with SessionLocal() as db_session:
        for doc_data in documents:
            try:
                inserted, skipped = process_single_document(
                    doc_data=doc_data,
                    chunker=chunker,
                    db_session=db_session,
                )
                total_inserted += inserted
                total_skipped += skipped
                processed_docs += 1
            except Exception as exc:
                logger.error(
                    "Failed to process document '%s': %s",
                    doc_data.get("doc_id"),
                    exc,
                    exc_info=True,
                )

    stats = {
        "processed_docs": processed_docs,
        "inserted_chunks": total_inserted,
        "skipped_chunks": total_skipped,
    }
    logger.info("Chunking & Storage stage completed stats: %s", stats)
    return stats


# ── 5. Full Orchestrator Flow ("Maestro") ─────────────────────────────────────

async def main_orchestrator_flow(
    skip_ingestion: bool = False,
    skip_scraping: bool = False,
    pdf_max_pages: int = 5,
    json_output_dir: Optional[Path] = None,
    ocr_engine: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Top-level Orchestrator pipeline.
    Coordinates high-level stages: DB Init -> Ingestion -> Normalization -> Chunking & DB Storage.
    """
    logger.info("==================================================")
    logger.info("        ZAR3A END-TO-END ORCHESTRATOR           ")
    logger.info("==================================================")

    # 1. Initialize database schema
    init_database_tables()

    # 2. Trigger Ingestion Stage (unless skipped)
    if not skip_ingestion:
        await run_ingestion_stage(
            pdf_max_pages=pdf_max_pages,
            json_output_dir=json_output_dir,
            skip_scraping=skip_scraping,
            ocr_engine=ocr_engine,
        )
    else:
        logger.info("Skipping Ingestion Stage (both scraping & OCR) as requested.")

    # 3. Fetch normalized documents from Ingestion module
    normalized_docs = load_and_normalize_ingested_documents(json_output_dir)

    # 4. Orchestrate Chunking & Database Storage Stage
    storage_stats = run_chunking_and_storage_stage(documents=normalized_docs)

    logger.info("==================================================")
    logger.info("        ZAR3A ORCHESTRATION COMPLETE             ")
    logger.info("  Total Documents Processed : %d", storage_stats["processed_docs"])
    logger.info("  Total Chunks Inserted     : %d", storage_stats["inserted_chunks"])
    logger.info("  Total Duplicates Skipped  : %d", storage_stats["skipped_chunks"])
    logger.info("==================================================")

    return storage_stats


# ── CLI Entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Zar3a End-to-End Orchestrator")
    parser.add_argument(
        "--skip-scraping",
        "--ocr-only",
        action="store_true",
        dest="skip_scraping",
        help="Skip web scraping & PDF downloads; run OCR on local PDFs in Data/Arabic_Data directly.",
    )
    parser.add_argument(
        "--skip-ingestion",
        action="store_true",
        help="Skip web scraping & OCR entirely; process existing JSON files only.",
    )
    parser.add_argument(
        "--ocr-engine",
        type=str,
        choices=["paddle", "mistral"],
        default=None,
        help="OCR engine to use: 'paddle' (offline default) or 'mistral' (API).",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=5,
        help="Max PDF pagination pages to scrape during ingestion (default: 5).",
    )

    args = parser.parse_args()

    asyncio.run(
        main_orchestrator_flow(
            skip_ingestion=args.skip_ingestion,
            skip_scraping=args.skip_scraping,
            pdf_max_pages=args.max_pages,
            ocr_engine=args.ocr_engine,
        )
    )
