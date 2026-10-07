"""
backfill_chunk_text.py
======================
One-time migration script to backfill the `chunk_text` alias key in existing
Qdrant points that were stored with only the `content` key.

After the retriever.py fix (reading "content" || "chunk_text"), this backfill
is only needed for legacy compatibility, but it's good practice to run once.

Usage:
    .venv\\Scripts\\python.exe backfill_chunk_text.py
"""
from __future__ import annotations

import sys
import logging
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
_BACKEND_DIR = _PROJECT_ROOT / "back_end"
for p in [str(_PROJECT_ROOT), str(_BACKEND_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Zar3a.Backfill")

from src.rag_database.config import rag_config
from qdrant_client import QdrantClient
from qdrant_client.http.models import SetPayload

def backfill():
    client = QdrantClient(path=rag_config.QDRANT_LOCAL_PATH)
    collection = rag_config.VECTOR_DB_COLLECTION_NAME

    info = client.get_collection(collection)
    total_points = info.points_count
    logger.info("Collection '%s' has %d points. Starting backfill...", collection, total_points)

    offset = None
    updated = 0
    skipped = 0
    batch_size = 100

    while True:
        records, next_offset = client.scroll(
            collection_name=collection,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not records:
            break

        for rec in records:
            payload = rec.payload or {}
            content = payload.get("content", "")
            chunk_text = payload.get("chunk_text", "")

            if content and not chunk_text:
                # Backfill chunk_text from content
                client.set_payload(
                    collection_name=collection,
                    payload={"chunk_text": content},
                    points=[rec.id],
                )
                updated += 1
            else:
                skipped += 1

        offset = next_offset
        logger.info("Progress: updated=%d, skipped=%d...", updated, skipped)
        if not next_offset:
            break

    logger.info("Backfill complete. Updated: %d, Skipped/already-set: %d", updated, skipped)


if __name__ == "__main__":
    backfill()
