"""
resume_ingestion.py

Resumes embedding and uploading only the un-embedded chunks that are already
stored in SQLite (is_embedded=False).  Skips JSON re-loading and re-chunking.

Run:
    uv run python resume_ingestion.py
"""

import sys
import logging
import time
from pathlib import Path

# ── path setup ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "back_end"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ResumeIngestion")

from src.database.connection import SessionLocal
from src.database.models import DocumentChunkModel, DocumentModel
from src.rag_database.embedder import RAGEmbedder
from src.rag_database.vector_store import VectorStore

BATCH_SIZE = 16          # chunks embedded + upserted per iteration


def main() -> None:
    db = SessionLocal()
    try:
        total = db.query(DocumentChunkModel).count()
        remaining = (
            db.query(DocumentChunkModel)
            .filter(DocumentChunkModel.is_embedded == False)  # noqa: E712
            .count()
        )
        logger.info(
            "SQLite totals — total: %d | already embedded: %d | remaining: %d",
            total, total - remaining, remaining,
        )

        if remaining == 0:
            logger.info("✅ All chunks are already embedded. Nothing to do.")
            return

        # ── initialise components ───────────────────────────────────────────
        logger.info("Loading embedding model …")
        embedder = RAGEmbedder()

        logger.info("Connecting to Qdrant …")
        vector_store = VectorStore()

        # ── resume loop ─────────────────────────────────────────────────────
        embedded_count = 0
        failed_count   = 0
        start_time     = time.time()

        while True:
            # Join with DocumentModel to get file metadata
            batch = (
                db.query(DocumentChunkModel, DocumentModel)
                .join(DocumentModel, DocumentChunkModel.document_id == DocumentModel.id)
                .filter(DocumentChunkModel.is_embedded == False)  # noqa: E712
                .limit(BATCH_SIZE)
                .all()
            )
            if not batch:
                break

            texts   = [chunk.content     for chunk, _doc in batch]
            chunks  = [chunk             for chunk, _doc in batch]
            docs    = [doc               for _chunk, doc in batch]

            vectors = embedder.embed_texts(texts, is_query=False)

            points = []
            for chunk, doc, vector in zip(chunks, docs, vectors):
                points.append({
                    "id":     chunk.id,
                    "vector": vector,
                    "payload": {
                        "chunk_text":    chunk.content,
                        "document_name": doc.file_name,
                        "source_file":   doc.file_path,
                        "chunk_index":   chunk.chunk_index,
                        "chunk_hash":    chunk.chunk_hash,
                    },
                })

            ok = vector_store.upsert_chunks(points)
            if ok:
                for chunk in chunks:
                    chunk.is_embedded = True
                db.commit()
                embedded_count += len(chunks)
            else:
                logger.error(
                    "Upsert failed for batch starting at chunk id=%s", chunks[0].id
                )
                failed_count += len(chunks)
                db.rollback()

            elapsed = time.time() - start_time
            rate    = embedded_count / elapsed if elapsed > 0 else 0
            eta_s   = (remaining - embedded_count) / rate if rate > 0 else float("inf")
            logger.info(
                "Progress: %d/%d embedded | %.1f chunks/s | ETA ~%.0f s",
                embedded_count, remaining, rate, eta_s,
            )

        # ── summary ─────────────────────────────────────────────────────────
        elapsed = time.time() - start_time
        logger.info(
            "✅ Done — embedded %d chunks in %.1f s (%d failed).",
            embedded_count, elapsed, failed_count,
        )

    finally:
        db.close()


if __name__ == "__main__":
    main()
