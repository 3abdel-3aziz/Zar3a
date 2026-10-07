"""
inspect_payload_schema.py
Inspect the actual payload keys of points that have content, to understand
the schema differences between the old and new ingestion runs.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "back_end"))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.rag_database.config import rag_config
from qdrant_client import QdrantClient

client = QdrantClient(path=rag_config.QDRANT_LOCAL_PATH)
collection = rag_config.VECTOR_DB_COLLECTION_NAME

offset = None
found_samples = {}
while len(found_samples) < 5:
    records, next_offset = client.scroll(
        collection_name=collection, limit=100, offset=offset,
        with_payload=True, with_vectors=False,
    )
    if not records:
        break
    for rec in records:
        p = rec.payload or {}
        txt = str(p.get("content", "") or p.get("chunk_text", "")).strip()
        if txt and str(rec.id) not in found_samples:
            found_samples[str(rec.id)] = {
                "keys": sorted(p.keys()),
                "content_preview": txt[:100],
                "file_name": p.get("file_name"),
                "doc_id": p.get("doc_id"),
            }
    offset = next_offset
    if not next_offset:
        break

for pid, info in found_samples.items():
    print(f"\nPoint ID: {pid}")
    print(f"  Keys: {info['keys']}")
    print(f"  file_name: {info['file_name']}")
    print(f"  doc_id: {info['doc_id']}")
    print(f"  content: {repr(info['content_preview'])}")
