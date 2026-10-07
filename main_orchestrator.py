import os
import sys
import time

# ── 0. Real-time Unbuffered Output & Immediate-Flushing Logger ───────────────
os.environ["PYTHONUNBUFFERED"] = "1"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

print("[Zar3a Orchestrator] Bootstrapping runtime environment (Surya OCR dedicated)...", flush=True)

# 1. PyTorch initialization for Surya OCR
print("  [1/5] Initializing PyTorch runtime...", flush=True)
t_start = time.time()
try:
    import torch
    cuda_status = f"CUDA available ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "CPU mode"
    print(f"        -> PyTorch ready ({time.time() - t_start:.2f}s, {cuda_status})", flush=True)
except Exception as exc:
    print(f"        -> PyTorch import warning: {exc}", flush=True)

# 3. Path setup
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
BACKEND_DIR = PROJECT_ROOT / "back_end"
for path_entry in [str(PROJECT_ROOT), str(BACKEND_DIR)]:
    if path_entry not in sys.path:
        sys.path.insert(0, path_entry)

# 4. Logger setup with immediate flush
import logging

class FlushingStreamHandler(logging.StreamHandler):
    """StreamHandler that flushes immediately after every emitted log record."""
    def emit(self, record):
        super().emit(record)
        self.flush()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[FlushingStreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger("Zar3a.MainOrchestrator")
logger.setLevel(logging.INFO)
logging.getLogger().setLevel(logging.INFO)

# 2. Database models
print("  [2/5] Connecting to database models (SQLAlchemy)...", flush=True)
t_sub = time.time()
from config import OUTPUT_JSON_DIR
from src.database.connection import engine, SessionLocal, Base
from src.database.operations import save_document_with_chunks
print(f"        -> Database connector ready ({time.time() - t_sub:.2f}s)", flush=True)

# 3. Chunking
print("  [3/5] Loading semantic chunking pipeline...", flush=True)
t_sub = time.time()
from src.chunkers.ChunkingPipeline import ChunkingPipeline
print(f"        -> Chunking pipeline ready ({time.time() - t_sub:.2f}s)", flush=True)

# 4. Ingestion pipeline
print("  [4/5] Loading ingestion & Surya OCR modules...", flush=True)
t_sub = time.time()
from src.ingestion.ingestion_pipeline import (
    run_full_pipeline,
    load_and_normalize_ingested_documents,
)
print(f"        -> Ingestion & OCR pipeline ready ({time.time() - t_sub:.2f}s)", flush=True)

# 5. RAG & Vector Store
print("  [5/5] Loading RAG vector pipeline (Qdrant & Embedder)...", flush=True)
t_sub = time.time()
from src.rag_database.rag_pipeline import RAGIngestionPipeline
print(f"        -> RAG vector pipeline ready ({time.time() - t_sub:.2f}s)", flush=True)

# Guarantee root logger remains at INFO
logging.getLogger().setLevel(logging.INFO)
logger.setLevel(logging.INFO)
print("[Zar3a Orchestrator] All dependencies loaded successfully.\n", flush=True)

import asyncio
from typing import List, Dict, Any, Tuple, Optional





# ── 1. Database Schema Initialization ────────────────────────────────────────

def init_database_tables() -> None:
    """
    Responsibility: Ensures all database tables (documents, document_chunks) exist.
    """
    print("[Orchestrator] Step 1: Initializing database schema...", flush=True)
    logger.info("Initializing database schema...")
    Base.metadata.create_all(bind=engine)
    print("[Orchestrator] Step 1: Database schema initialized successfully.", flush=True)
    logger.info("Database schema initialized successfully.")


# ── 2. Ingestion Trigger Stage ───────────────────────────────────────────────
#    Post-OCR Stage 1: OCR → Structured JSON
#    The ingestion pipeline runs OCR on PDFs, formats the raw OCR output into
#    structured JSON, and stores it in the designated OUTPUT_JSON_DIR.
#    This stage is handled by run_full_pipeline() which delegates to
#    MistralPDFProcessor (now supporting 'surya', 'paddle', 'mistral' engines).

async def run_ingestion_stage(
    pdf_max_pages: int = 5,
    json_output_dir: Optional[Path] = None,
    skip_scraping: bool = False,
    ocr_engine: str = "surya",
    surya_backend: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Responsibility: Triggers the Ingestion Pipeline to scrape text and process OCR.

    Post-OCR Stage 1 is embedded here:
      - Raw OCR output is automatically formatted/parsed into structured JSON
      - JSON files are placed in their designated location (OUTPUT_JSON_DIR)
    """
    print(f"[Orchestrator] Step 2: Triggering Ingestion Pipeline (OCR engine: '{ocr_engine}', surya_backend: '{surya_backend or 'auto'}')...", flush=True)
    logger.info("=== STEP 1: Executing Ingestion Pipeline (OCR → Structured JSON) ===")
    summary = await run_full_pipeline(
        pdf_max_pages=pdf_max_pages,
        json_output_dir=json_output_dir or OUTPUT_JSON_DIR,
        skip_scraping=skip_scraping,
        ocr_engine=ocr_engine,
        surya_backend=surya_backend,
    )
    print("[Orchestrator] Step 2: Ingestion Pipeline complete.", flush=True)
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

    Post-OCR Stage 2 & 3 for a single document:
      Stage 2: Pass the document's full_text through the semantic chunking mechanism.
      Stage 3: Store the generated chunks into the primary relational database.

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

    # Post-OCR Stage 2: Execute semantic chunking pipeline
    chunks = chunker.run(full_text=full_text, doc_id=doc_id, cleaning_mode="full")
    if not chunks:
        logger.warning("No chunks generated for doc_id='%s'.", doc_id)
        return 0, 0

    # Post-OCR Stage 3: Persist document and chunks atomically in DB
    _, num_inserted, num_skipped = save_document_with_chunks(
        db=db_session,
        doc_id=doc_id,
        file_name=file_name,
        file_path=file_path,
        chunks_data=chunks,
    )
    return num_inserted, num_skipped


# ── 4. Chunking & Database Storage Stage ──────────────────────────────────────
#    Post-OCR Stage 2: JSON Text → Semantic Chunks
#    Post-OCR Stage 3: Chunks → Relational Database

def run_chunking_and_storage_stage(
    documents: List[Dict[str, Any]],
) -> Dict[str, int]:
    """
    Responsibility: Receives normalized documents, orchestrates chunking, and persists to DB.

    Post-OCR Stages 2 & 3 (batch):
      Stage 2: Pass each document's text through the semantic chunking mechanism.
      Stage 3: Store the generated chunks into the primary relational/document database.

    Returns aggregated run statistics.
    """
    logger.info("=== STEP 2: Executing Chunking & Database Storage Stage ===")
    logger.info("  (Post-OCR Stage 2: JSON Text → Semantic Chunks)")
    logger.info("  (Post-OCR Stage 3: Chunks → Relational Database)")
    
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


# ── 5. Vector Embedding & Qdrant Upsert Stage ─────────────────────────────────
#    Post-OCR Stage 4: DB Records → Embeddings → Qdrant

def run_vector_embedding_stage(
    batch_size: int = 50,
    max_iterations: int = 100,
) -> Dict[str, int]:
    """
    Post-OCR Stage 4: Fetch un-embedded records from the relational database,
    generate dense vector embeddings, and upsert them into Qdrant vector database.

    This stage bridges the relational DB and the vector DB by:
      1. Querying document_chunks where is_embedded=False
      2. Generating embeddings via RAGEmbedder (multilingual-e5-large)
      3. Upserting the vectors + metadata payloads into Qdrant
      4. Marking the chunks as is_embedded=True in the relational DB

    Processes in batches until no more un-embedded chunks remain (or max_iterations).

    Args:
        batch_size: Number of chunks to process per iteration.
        max_iterations: Safety cap to prevent infinite loops.

    Returns:
        Dict with total_embedded count.
    """
    logger.info("=== STEP 3: Executing Vector Embedding & Qdrant Upsert Stage ===")
    logger.info("  (Post-OCR Stage 4: DB Records → Embeddings → Qdrant)")

    total_embedded = 0

    try:
        rag_pipeline = RAGIngestionPipeline()

        with SessionLocal() as db_session:
            for iteration in range(1, max_iterations + 1):
                processed = rag_pipeline.process_unembedded_chunks(
                    db=db_session,
                    batch_size=batch_size,
                )

                if processed == 0:
                    logger.info(
                        "Vector embedding complete. No more un-embedded chunks found "
                        "(after %d iteration(s), %d total embedded).",
                        iteration, total_embedded,
                    )
                    break

                total_embedded += processed
                logger.info(
                    "Vector embedding iteration %d: embedded %d chunk(s) (total: %d).",
                    iteration, processed, total_embedded,
                )

        if total_embedded == 0:
            logger.info("No un-embedded chunks found. Vector embedding stage is a no-op.")

    except Exception as exc:
        logger.error(
            "Vector embedding stage encountered an error: %s", exc, exc_info=True,
        )

    stats = {"total_embedded": total_embedded}
    logger.info("Vector Embedding & Qdrant Upsert stage completed: %s", stats)
    return stats


# ── 6. Full Orchestrator Flow ("Maestro") ─────────────────────────────────────

async def main_orchestrator_flow(
    skip_ingestion: bool = False,
    skip_scraping: bool = False,
    skip_vectorization: bool = False,
    pdf_max_pages: int = 5,
    json_output_dir: Optional[Path] = None,
    ocr_engine: str = "surya",
    embedding_batch_size: int = 50,
    surya_backend: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Top-level Orchestrator pipeline.
    Coordinates the full post-OCR workflow:
      DB Init → Ingestion (OCR → JSON) → Normalization → Chunking & DB Storage → Embedding & Qdrant

    Post-OCR Pipeline Stages:
      Stage 1: OCR → Structured JSON          [handled in run_ingestion_stage]
      Stage 2: JSON Text → Semantic Chunks     [handled in run_chunking_and_storage_stage]
      Stage 3: Chunks → Relational Database    [handled in run_chunking_and_storage_stage]
      Stage 4: DB Records → Embeddings → Qdrant [handled in run_vector_embedding_stage]
    """
    print("\n" + "=" * 50, flush=True)
    print("        ZAR3A END-TO-END ORCHESTRATOR           ", flush=True)
    print("=" * 50 + "\n", flush=True)
    logger.info("==================================================")
    logger.info("        ZAR3A END-TO-END ORCHESTRATOR           ")
    logger.info("==================================================")

    # 1. Initialize database schema
    init_database_tables()

    # 2. Trigger Ingestion Stage (unless skipped)
    #    Post-OCR Stage 1: OCR → Structured JSON
    if not skip_ingestion:
        await run_ingestion_stage(
            pdf_max_pages=pdf_max_pages,
            json_output_dir=json_output_dir,
            skip_scraping=skip_scraping,
            ocr_engine=ocr_engine,
            surya_backend=surya_backend,
        )
    else:
        print("[Orchestrator] Skipping Ingestion Stage as requested.", flush=True)
        logger.info("Skipping Ingestion Stage (both scraping & OCR) as requested.")

    # 3. Fetch normalized documents from Ingestion module
    print("[Orchestrator] Step 3: Fetching and normalizing ingested documents...", flush=True)
    normalized_docs = load_and_normalize_ingested_documents(json_output_dir)
    print(f"[Orchestrator] Step 3: Found {len(normalized_docs)} document(s) ready for chunking.", flush=True)

    # 4. Orchestrate Chunking & Database Storage Stage
    print("[Orchestrator] Step 4: Running Chunking & Database Storage stage...", flush=True)
    storage_stats = run_chunking_and_storage_stage(documents=normalized_docs)
    print(f"[Orchestrator] Step 4: Storage complete. Chunks inserted: {storage_stats.get('inserted_chunks', 0)}, skipped: {storage_stats.get('skipped_chunks', 0)}", flush=True)

    # 5. Orchestrate Vector Embedding & Qdrant Upsert Stage
    embedding_stats = {"total_embedded": 0}
    if not skip_vectorization:
        print("[Orchestrator] Step 5: Running Vector Embedding & Qdrant Upsert stage...", flush=True)
        embedding_stats = run_vector_embedding_stage(
            batch_size=embedding_batch_size,
        )
        print(f"[Orchestrator] Step 5: Vector embedding complete. Total embedded: {embedding_stats.get('total_embedded', 0)}", flush=True)
    else:
        print("[Orchestrator] Skipping Vector Embedding & Qdrant stage as requested.", flush=True)
        logger.info("Skipping Vector Embedding & Qdrant stage as requested.")

    print("\n" + "=" * 50, flush=True)
    print("        ZAR3A ORCHESTRATION COMPLETE             ", flush=True)
    print(f"  Total Documents Processed : {storage_stats['processed_docs']}", flush=True)
    print(f"  Total Chunks Inserted     : {storage_stats['inserted_chunks']}", flush=True)
    print(f"  Total Duplicates Skipped  : {storage_stats['skipped_chunks']}", flush=True)
    print(f"  Total Chunks Embedded     : {embedding_stats['total_embedded']}", flush=True)
    print("=" * 50 + "\n", flush=True)

    return {
        **storage_stats,
        **embedding_stats,
    }


# Convenient alias for backwards compatibility and clean imports
run_orchestrator = main_orchestrator_flow



# ── CLI Entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    print("[CLI Entrypoint] Parsing command line arguments...", flush=True)

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
        "--skip-vectorization",
        action="store_true",
        help="Skip the vector embedding & Qdrant upsert stage.",
    )
    parser.add_argument(
        "--ocr-engine",
        type=str,
        choices=["surya", "paddle", "mistral"],
        default="surya",
        help="OCR engine to use (default: 'surya', the dedicated local VLM engine).",
    )
    parser.add_argument(
        "--surya-backend",
        type=str,
        choices=["vllm", "llamacpp"],
        default=None,
        help="Inference backend for Surya OCR: 'vllm' (Docker GPU) or 'llamacpp' (CPU/local GGUF).",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=5,
        help="Max PDF pagination pages to scrape during ingestion (default: 5).",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=50,
        help="Batch size for vector embedding stage (default: 50).",
    )

    args = parser.parse_args()

    print(
        f"[CLI Entrypoint] Launching pipeline: ocr_engine='{args.ocr_engine}', "
        f"surya_backend='{args.surya_backend}', skip_scraping={args.skip_scraping}, "
        f"skip_ingestion={args.skip_ingestion}, skip_vectorization={args.skip_vectorization}\n",
        flush=True,
    )

    asyncio.run(
        main_orchestrator_flow(
            skip_ingestion=args.skip_ingestion,
            skip_scraping=args.skip_scraping,
            skip_vectorization=args.skip_vectorization,
            pdf_max_pages=args.max_pages,
            ocr_engine=args.ocr_engine,
            embedding_batch_size=args.embedding_batch_size,
            surya_backend=args.surya_backend,
        )
    )

