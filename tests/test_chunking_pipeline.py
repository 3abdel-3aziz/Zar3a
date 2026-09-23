"""
tests/test_chunking_pipeline.py
Smoke-test for the end-to-end ChunkingPipeline.

Run with:
    uv run pytest tests/test_chunking_pipeline.py -v
or simply:
    uv run python tests/test_chunking_pipeline.py
"""

import json
import sys
import logging
from pathlib import Path

# ---------------------------------------------------------------------------
# Make sure the project root is on sys.path so relative imports resolve.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# ---------------------------------------------------------------------------
# Import the pipeline (this validates all internal imports as well)
# ---------------------------------------------------------------------------
# pyrefly: ignore [missing-import]
from src.chunkers.ChunkingPipeline import ChunkingPipeline

# ---------------------------------------------------------------------------
# Synthetic Arabic legal text that mirrors real OCR output
# ---------------------------------------------------------------------------
SAMPLE_TEXT = """\
بسم الله الرحمن الرحيم

هذا قانون زراعي للاختبار.

### مادة 1

تسري أحكام هذا القانون على جميع الأراضي الزراعية في جمهورية مصر العربية.

### مادة 2

يُحظر استخدام الأراضي الزراعية في غير أغراضها الزراعية إلا بترخيص.

### مادة (٣)

يُعاقب على مخالفة أحكام المادة السابقة بالغرامة المنصوص عليها في هذا القانون.
"""


def test_pipeline_smoke() -> None:
    """Verifies that the pipeline produces correctly structured output."""
    pipeline = ChunkingPipeline()
    chunks = pipeline.run(full_text=SAMPLE_TEXT, doc_id="test_doc_001", cleaning_mode="light")

    assert chunks, "Pipeline returned no chunks — check SemanticChunker regex patterns."

    print(f"\n  Pipeline produced {len(chunks)} chunk(s).\n")

    required_keys = {"doc_id", "chunk_index", "content", "metadata", "chunk_id", "chunk_hash", "source"}
    for i, chunk in enumerate(chunks):
        missing = required_keys - chunk.keys()
        assert not missing, f"Chunk {i} is missing keys: {missing}"

        print(f"--- Chunk {i} ---")
        print(f"  chunk_id   : {chunk['chunk_id']}")
        print(f"  chunk_hash : {chunk['chunk_hash'][:16]}...")
        print(f"  type       : {chunk['metadata'].get('type') if isinstance(chunk['metadata'], dict) else chunk['metadata']}")
        print(f"  words      : {chunk.get('metadata', {}).get('word_count', '?') if isinstance(chunk.get('metadata'), dict) else '?'}")
        print(f"  content[:60]: {chunk['content'][:60].strip()!r}")
        print()

    print("  All required keys are present on every chunk.")


def test_folder_processing() -> None:
    """Tests process_folder() against Data/processed_json/ if it contains .md files."""
    folder = ROOT / "Data" / "processed_json"
    md_files = list(folder.glob("*.md"))

    if not md_files:
        print(f"\n  No .md files found in {folder} — skipping folder test.\n")
        return

    pipeline = ChunkingPipeline()
    results = pipeline.process_folder(folder_path=folder, cleaning_mode="full")

    assert results, "process_folder() returned empty results."
    print(f"\n  Folder processing produced results for {len(results)} document(s).")
    for doc_id, chunks in results.items():
        print(f"    {doc_id}: {len(chunks)} chunk(s)")


if __name__ == "__main__":
    print("=" * 60)
    print("Running ChunkingPipeline smoke test…")
    print("=" * 60)
    test_pipeline_smoke()
    test_folder_processing()
    print("=" * 60)
    print("All tests passed.")
    print("=" * 60)
