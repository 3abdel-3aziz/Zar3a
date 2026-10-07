"""
summarize_qdrant_content.py
Summarize the content quality across all files in Qdrant.
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
stats = {}  # file_name -> {"empty": int, "nonempty": int, "sample": str}

while True:
    records, next_offset = client.scroll(
        collection_name=collection, limit=200, offset=offset,
        with_payload=True, with_vectors=False,
    )
    if not records:
        break
    for rec in records:
        p = rec.payload or {}
        fn = p.get("file_name") or "UNKNOWN"
        txt = str(p.get("content", "") or p.get("chunk_text", "")).strip()
        if fn not in stats:
            stats[fn] = {"empty": 0, "nonempty": 0, "sample": ""}
        if txt:
            stats[fn]["nonempty"] += 1
            if not stats[fn]["sample"]:
                stats[fn]["sample"] = txt[:100]
        else:
            stats[fn]["empty"] += 1
    offset = next_offset
    if not next_offset:
        break

print(f"{'File':<55} {'Empty':>7} {'NonEmpty':>9} {'Sample'}")
print("-" * 110)
for fn in sorted(stats.keys()):
    s = stats[fn]
    sample = repr(s["sample"][:60]) if s["sample"] else "(no text)"
    print(f"{fn:<55} {s['empty']:>7} {s['nonempty']:>9}   {sample}")
