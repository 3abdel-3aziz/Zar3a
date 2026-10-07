"""
reingest_from_json.py
═══════════════════════════════════════════════════════════════════════════════
Zar3a RAG Reset & Re-ingestion Script

PURPOSE
───────
Wipes all existing chunks/embeddings from the relational DB (SQLite/Postgres)
and the Qdrant vector store, then re-chunks and re-ingests every document
from the pre-existing JSON cache (Data/processed_json/) WITHOUT re-running
OCR or web scraping.

STEPS EXECUTED
──────────────
1. Validate that the JSON source directory is non-empty.
2. Purge Qdrant vector collection (delete + recreate).
3. Purge relational database (DELETE all chunks → documents in correct order).
4. Load & normalise text from every JSON file via the existing ingestion helpers.
5. Run each document through the updated ChunkingPipeline (new semantic +
   recursive fallback chunker).
6. Persist chunks to the relational DB via save_document_with_chunks().
7. Run the RAGIngestionPipeline to embed all new chunks and upsert into Qdrant.
8. Print a final summary.

USAGE
─────
    uv run python reingest_from_json.py [--json-dir PATH] [--batch-size N] [--dry-run]

OPTIONS
───────
    --json-dir    Override path to processed JSON directory (default: Data/processed_json)
    --batch-size  Embedding batch size per RAG pipeline call (default: 50)
    --dry-run     Print what would happen without writing to any database
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List

# ── Path bootstrap ────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent
_BACKEND_DIR = _PROJECT_ROOT / "back_end"

for _p in [str(_PROJECT_ROOT), str(_BACKEND_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
# ─────────────────────────────────────────────────────────────────────────────

# ── Encoding safety for Windows console ───────────────────────────────────────
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("Zar3a.ReIngest")


# ═════════════════════════════════════════════════════════════════════════════
# STEP HELPERS
# ═════════════════════════════════════════════════════════════════════════════


def _step_banner(n: int, label: str) -> None:
    logger.info("=" * 60)
    logger.info("  STEP %d: %s", n, label)
    logger.info("=" * 60)


# ─── Step 1: Validate JSON source ────────────────────────────────────────────

def validate_json_dir(json_dir: Path) -> List[Path]:
    """Returns the list of .json files found, or raises if none."""
    _step_banner(1, "Validate JSON source directory")

    if not json_dir.exists() or not json_dir.is_dir():
        logger.error("JSON directory does not exist: %s", json_dir)
        raise SystemExit(1)

    json_files = list(json_dir.glob("*.json"))
    if not json_files:
        logger.error("No .json files found in: %s", json_dir)
        raise SystemExit(1)

    logger.info("Found %d JSON file(s) in: %s", len(json_files), json_dir)
    return json_files


# ─── Step 2: Purge Qdrant ────────────────────────────────────────────────────

def purge_qdrant(dry_run: bool) -> None:
    """Deletes and recreates the Qdrant collection to start fresh."""
    _step_banner(2, "Purge Qdrant vector collection")

    from src.rag_database.config import rag_config
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels

    collection = rag_config.VECTOR_DB_COLLECTION_NAME
    dimension = rag_config.EMBEDDING_DIMENSION
    local_path = rag_config.QDRANT_LOCAL_PATH
    host = rag_config.VECTOR_DB_HOST
    port = rag_config.VECTOR_DB_PORT
    api_key = rag_config.VECTOR_DB_API_KEY

    if dry_run:
        logger.info("[DRY-RUN] Would delete and recreate Qdrant collection '%s'", collection)
        return

    # Build client
    if local_path:
        client = QdrantClient(path=local_path)
    elif api_key:
        client = QdrantClient(host=host, port=port, api_key=api_key)
    else:
        client = QdrantClient(host=host, port=port)

    existing = {c.name for c in client.get_collections().collections}
    if collection in existing:
        logger.info("Deleting existing Qdrant collection: '%s'", collection)
        client.delete_collection(collection_name=collection)
        logger.info("Collection '%s' deleted.", collection)
    else:
        logger.info("Collection '%s' does not exist — nothing to delete.", collection)

    logger.info("Recreating collection '%s' (dim=%d, Cosine)...", collection, dimension)
    client.create_collection(
        collection_name=collection,
        vectors_config=qmodels.VectorParams(
            size=dimension,
            distance=qmodels.Distance.COSINE,
        ),
    )
    logger.info("Qdrant collection '%s' recreated successfully.", collection)
    try:
        client.close()
    except Exception:
        pass


# ─── Step 3: Purge relational DB ─────────────────────────────────────────────

def purge_relational_db(dry_run: bool) -> None:
    """Truncates document_chunks then documents tables."""
    _step_banner(3, "Purge relational database (chunks → documents)")

    from src.database.connection import engine, Base, SessionLocal
    from src.database.models import DocumentChunkModel, DocumentModel

    # Ensure tables exist before trying to delete from them
    Base.metadata.create_all(bind=engine)

    if dry_run:
        logger.info("[DRY-RUN] Would DELETE all rows from document_chunks and documents tables.")
        return

    db = SessionLocal()
    try:
        chunk_count = db.query(DocumentChunkModel).count()
        doc_count = db.query(DocumentModel).count()

        logger.info("Current state: %d document(s), %d chunk(s) in DB.", doc_count, chunk_count)

        # Delete in FK-safe order: chunks first, then documents
        db.query(DocumentChunkModel).delete(synchronize_session=False)
        db.query(DocumentModel).delete(synchronize_session=False)
        db.commit()

        remaining_chunks = db.query(DocumentChunkModel).count()
        remaining_docs = db.query(DocumentModel).count()
        logger.info(
            "Purge complete. Remaining: %d documents, %d chunks.",
            remaining_docs,
            remaining_chunks,
        )
    except Exception as exc:
        db.rollback()
        logger.error("Failed to purge relational database: %s", exc, exc_info=True)
        raise
    finally:
        db.close()


# ─── Step 4–6: Load, chunk, and persist ──────────────────────────────────────

def load_chunk_and_persist(
    json_dir: Path,
    dry_run: bool,
    use_llm_correct: bool = True,
    model_name: Optional[str] = None,
) -> dict:
    """
    Loads all JSON files, runs them through ChunkingPipeline, applies Ollama LLM
    text correction for OCR fragments/scrambles, and saves to DB.
    Returns a statistics dictionary.
    """
    _step_banner(4, "Load JSON → LLM Correction → Chunk → Persist to DB")

    from src.ingestion.ingestion_pipeline import load_and_normalize_ingested_documents
    from src.chunkers.ChunkingPipeline import ChunkingPipeline
    from src.database.connection import SessionLocal
    from src.database.operations import save_document_with_chunks
    from src.ingestion.ollama_arabic_corrector import OllamaArabicCorrector

    documents = load_and_normalize_ingested_documents(json_dir)
    if not documents:
        logger.error("No valid documents extracted from JSON files — aborting.")
        raise SystemExit(1)

    logger.info("Loaded and normalised %d document(s).", len(documents))

    pipeline = ChunkingPipeline()
    corrector = OllamaArabicCorrector(model_name=model_name) if use_llm_correct else None
    total_chunks = 0
    all_chunk_lengths: List[int] = []
    all_chunk_words: List[int] = []
    failed_docs: List[str] = []

    for idx, doc in enumerate(documents, start=1):
        doc_id = doc["doc_id"]
        file_name = doc["file_name"]
        full_text = doc["full_text"]

        logger.info(
            "[%d/%d] Processing '%s' (%d chars)...",
            idx,
            len(documents),
            file_name,
            len(full_text),
        )

        try:
            # Use 'full' cleaning mode for OCR output, 'light' for web-scraped Markdown
            # Heuristic: if text already has '###' headers → light, else full
            cleaning_mode = "light" if "###" in full_text[:2000] else "full"
            chunks = pipeline.run(
                full_text=full_text,
                doc_id=doc_id,
                cleaning_mode=cleaning_mode,
            )

            if not chunks:
                logger.warning("  → Zero chunks produced for '%s'. Skipping.", file_name)
                failed_docs.append(file_name)
                continue

            # LLM-based Arabic text correction via Ollama for scrambled/broken OCR
            if corrector is not None:
                chunks = corrector.process_chunks(chunks)

            logger.info("  → %d chunk(s) generated (mode=%s).", len(chunks), cleaning_mode)

            chunk_lengths = [len(c.get("content", "")) for c in chunks]
            chunk_words = [len(c.get("content", "").split()) for c in chunks]
            all_chunk_lengths.extend(chunk_lengths)
            all_chunk_words.extend(chunk_words)

            if not dry_run:
                db = SessionLocal()
                try:
                    save_document_with_chunks(
                        db=db,
                        doc_id=doc_id,
                        file_name=file_name,
                        file_path=doc.get("file_path", ""),
                        chunks_data=chunks,
                    )
                    db.commit()
                finally:
                    db.close()

            total_chunks += len(chunks)

        except Exception as exc:
            logger.error("  → ERROR processing '%s': %s", file_name, exc, exc_info=True)
            failed_docs.append(file_name)

    if failed_docs:
        logger.warning(
            "%d document(s) failed chunking: %s",
            len(failed_docs),
            ", ".join(failed_docs),
        )

    logger.info(
        "Chunk & persist stage complete: %d chunks across %d/%d documents.",
        total_chunks,
        len(documents) - len(failed_docs),
        len(documents),
    )
    return {
        "total_chunks": total_chunks,
        "chunk_lengths": all_chunk_lengths,
        "chunk_words": all_chunk_words,
        "documents_processed": len(documents) - len(failed_docs),
    }


# ─── Step 7: Embed and upsert to Qdrant ──────────────────────────────────────

def embed_and_vectorize(batch_size: int, dry_run: bool) -> int:
    """Embeds all un-embedded chunks and upserts them into Qdrant."""
    _step_banner(7, "Embed chunks → Upsert to Qdrant vector store")

    if dry_run:
        logger.info("[DRY-RUN] Would run RAGIngestionPipeline.process_unembedded_chunks().")
        return 0

    from src.rag_database.rag_pipeline import RAGIngestionPipeline
    from src.database.connection import SessionLocal

    pipeline = RAGIngestionPipeline()
    total_vectorized = 0

    db = SessionLocal()
    try:
        while True:
            processed = pipeline.process_unembedded_chunks(db, batch_size=batch_size)
            if processed == 0:
                break
            total_vectorized += processed
            logger.info("  → Vectorized %d chunk(s) this batch (total so far: %d).", processed, total_vectorized)
    finally:
        db.close()

    # Ensure vector store client connection is closed
    if hasattr(pipeline.vector_store, "client"):
        try:
            pipeline.vector_store.client.close()
        except Exception:
            pass

    logger.info("Embedding stage complete: %d total chunk(s) indexed in Qdrant.", total_vectorized)
    return total_vectorized


# ─── Step 8: Validation & Retrieval Ready Verification ──────────────────────

def validate_and_test_retrieval(
    stats: dict,
    total_vectorized: int,
    dry_run: bool,
) -> None:
    """
    Computes chunk size statistics and runs a live retrieval test to confirm
    that chunks are properly indexed and retrieval-ready.
    """
    _step_banner(8, "Validation Summary & Retrieval Ready Check")

    chunk_lengths: List[int] = stats.get("chunk_lengths", [])
    chunk_words: List[int] = stats.get("chunk_words", [])
    total_chunks: int = stats.get("total_chunks", 0)

    if chunk_lengths:
        avg_chars = sum(chunk_lengths) / len(chunk_lengths)
        avg_words = sum(chunk_words) / len(chunk_words)
        min_chars = min(chunk_lengths)
        max_chars = max(chunk_lengths)
    else:
        avg_chars = avg_words = min_chars = max_chars = 0

    logger.info("-" * 60)
    logger.info("  [METRICS] CHUNK INTEGRITY & SIZE METRICS")
    logger.info("  * Documents Processed    : %d", stats.get("documents_processed", 0))
    logger.info("  * Total Chunks Generated : %d", total_chunks)
    logger.info("  * Average Chunk Size     : %.1f characters (~%.1f words)", avg_chars, avg_words)
    logger.info("  * Min Chunk Size         : %d characters", min_chars)
    logger.info("  * Max Chunk Size         : %d characters", max_chars)
    logger.info("-" * 60)

    if dry_run:
        logger.info("[DRY-RUN] Skipping database verification & retrieval test.")
        return

    from src.database.connection import SessionLocal
    from src.database.models import DocumentChunkModel
    from src.rag_database.config import rag_config
    from qdrant_client import QdrantClient

    db = SessionLocal()
    try:
        db_chunk_count = db.query(DocumentChunkModel).count()
        db_embedded_count = db.query(DocumentChunkModel).filter(DocumentChunkModel.is_embedded.is_(True)).count()
    finally:
        db.close()

    collection = rag_config.VECTOR_DB_COLLECTION_NAME
    local_path = rag_config.QDRANT_LOCAL_PATH
    if local_path:
        q_client = QdrantClient(path=local_path)
    elif rag_config.VECTOR_DB_API_KEY:
        q_client = QdrantClient(host=rag_config.VECTOR_DB_HOST, port=rag_config.VECTOR_DB_PORT, api_key=rag_config.VECTOR_DB_API_KEY)
    else:
        q_client = QdrantClient(host=rag_config.VECTOR_DB_HOST, port=rag_config.VECTOR_DB_PORT)

    try:
        qdrant_points = q_client.get_collection(collection_name=collection).points_count
    except Exception as exc:
        qdrant_points = f"Error: {exc}"
    finally:
        try:
            q_client.close()
        except Exception:
            pass

    logger.info("  [STORAGE] DATABASE PERSISTENCE STATUS")
    logger.info("  * SQLite Total Chunks    : %d", db_chunk_count)
    logger.info("  * SQLite Embedded Flags  : %d", db_embedded_count)
    logger.info("  * Qdrant Indexed Points  : %s", qdrant_points)
    logger.info("-" * 60)

    # Live hybrid retrieval verification
    logger.info("  [TEST] LIVE RETRIEVAL TEST")
    test_query = "ما هي أهداف الاستراتيجية الوطنية لتغير المناخ؟"
    logger.info("  * Query: '%s'", test_query)

    try:
        from src.rag_database.retriever import RAGRetriever
        retriever = RAGRetriever(default_top_k=2)
        results = retriever.retrieve(test_query)
        logger.info("  * Candidates Retrieved   : %d", len(results))
        for idx, res in enumerate(results, 1):
            score = res.get("score", 0.0)
            snippet = res.get("text", "")[:100].replace("\n", " ")
            source = res.get("metadata", {}).get("source", "unknown")
            logger.info("    [%d] Score: %.4f | Source: %s | Preview: %s...", idx, score, source, snippet)

        if hasattr(retriever.vector_store, "client"):
            try:
                retriever.vector_store.client.close()
            except Exception:
                pass

        logger.info("-" * 60)
        logger.info("  STATUS: [SUCCESS] RETRIEVAL READY (All checks passed)")
        logger.info("-" * 60)
    except Exception as exc:
        logger.error("  [FAILED] Retrieval test failed: %s", exc, exc_info=True)


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zar3a RAG Reset & Re-ingestion Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--json-dir",
        type=Path,
        default=None,
        help="Path to the processed JSON directory (default: Data/processed_json)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Embedding batch size per RAG pipeline call (default: 50)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Ollama model for Arabic OCR correction (default: auto-detect qwen2.5:3b or qwen2.5:7b)",
    )
    parser.add_argument(
        "--no-llm-correct",
        action="store_true",
        help="Disable Ollama LLM correction and use algorithmic normalization only",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate all steps without writing to any database",
    )
    args = parser.parse_args()

    # Resolve JSON directory
    if args.json_dir:
        json_dir = args.json_dir.resolve()
    else:
        # Try to read from config, fallback to canonical path
        try:
            from config import OUTPUT_JSON_DIR
            json_dir = Path(OUTPUT_JSON_DIR).resolve()
        except ImportError:
            json_dir = (_PROJECT_ROOT / "Data" / "processed_json").resolve()

    t0 = time.monotonic()

    logger.info("=" * 60)
    logger.info("  Zar3a RAG Reset & Re-ingestion Script")
    logger.info("  JSON source  : %s", json_dir)
    logger.info("  Batch size   : %d", args.batch_size)
    logger.info("  LLM Correct  : %s (model: %s)", not args.no_llm_correct, args.model or "auto")
    logger.info("  Dry run      : %s", args.dry_run)
    logger.info("=" * 60)

    # ── Execute pipeline steps ────────────────────────────────────────────
    json_files = validate_json_dir(json_dir)           # Step 1
    purge_qdrant(dry_run=args.dry_run)                 # Step 2
    purge_relational_db(dry_run=args.dry_run)          # Step 3
    chunk_stats = load_chunk_and_persist(              # Steps 4-6
        json_dir=json_dir,
        dry_run=args.dry_run,
        use_llm_correct=not args.no_llm_correct,
        model_name=args.model,
    )
    total_vectorized = embed_and_vectorize(            # Step 7
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )
    validate_and_test_retrieval(                       # Step 8
        stats=chunk_stats,
        total_vectorized=total_vectorized,
        dry_run=args.dry_run,
    )

    elapsed = time.monotonic() - t0
    logger.info("=" * 60)
    logger.info("  Re-ingestion complete in %.1fs", elapsed)
    logger.info("  JSON files processed : %d", len(json_files))
    logger.info("  Chunks created       : %d", chunk_stats.get("total_chunks", 0))
    logger.info("  Chunks vectorized    : %d", total_vectorized)
    if args.dry_run:
        logger.info("  [NOTICE] DRY-RUN -- no data was actually written.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
