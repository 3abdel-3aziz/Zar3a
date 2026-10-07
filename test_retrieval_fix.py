"""
test_retrieval.py — Quick verification that the retriever now returns actual text.
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

from src.rag_database.retriever import RAGRetriever

retriever = RAGRetriever()

queries = [
    "المجلس الوطني للتغيرات المناخية",
    "قانون الموارد المائية والري 147",
    "الاستراتيجية الوطنية لتغير المناخ 2050",
]

for q in queries:
    print(f"\n{'='*60}")
    print(f"Query: {q}")
    results = retriever.retrieve(q, top_k=3)
    print(f"  => {len(results)} chunk(s) retrieved")
    for i, r in enumerate(results, 1):
        content = r.get("content", "")
        score = r.get("score", 0)
        file_name = r.get("metadata", {}).get("file_name", "?")
        print(f"  [{i}] file={file_name} | score={score:.4f} | content_len={len(content)}")
        if content.strip():
            print(f"      Preview: {repr(content[:150])}")
        else:
            print(f"      ⚠️ EMPTY CONTENT")
