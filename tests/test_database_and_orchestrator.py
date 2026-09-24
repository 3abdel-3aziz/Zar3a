"""
tests/test_database_and_orchestrator.py

Unit and integration tests for src/database/operations.py, ingestion normalization,
and main_orchestrator.py. Uses in-memory SQLite for testing database operations safely.
"""

import sys
import json
import pytest
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Bootstrap project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database.models import Base, DocumentModel, DocumentChunkModel
from src.database.operations import (
    validate_document_inputs,
    validate_chunk_data,
    get_document_by_doc_id,
    get_existing_chunk_identifiers,
    insert_document,
    insert_chunks_batch,
    save_document_with_chunks,
    filter_duplicate_chunks,
    get_unembedded_chunks,
    update_chunk_embedding_status,
    update_chunk_vector_id,
    get_chunks_with_metadata_for_vectorization,
)
from src.ingestion.ingestion_pipeline import (
    extract_text_from_json_data,
    parse_single_json_document,
    load_and_normalize_ingested_documents,
)
from main_orchestrator import (
    process_single_document,
    main_orchestrator_flow,
)
from src.chunkers.ChunkingPipeline import ChunkingPipeline


@pytest.fixture
def in_memory_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


def test_validate_document_inputs():
    validate_document_inputs("doc_1", "file.json", "/path/to/file")
    with pytest.raises(ValueError):
        validate_document_inputs("", "file.json", "/path/to/file")


def test_validate_chunk_data():
    valid_chunk = {"chunk_hash": "hash123", "chunk_text": "Sample text"}
    validate_chunk_data(valid_chunk)

    with pytest.raises(ValueError):
        validate_chunk_data({"chunk_hash": ""})


def test_insert_and_query_document(in_memory_db):
    doc = insert_document(in_memory_db, "doc_test_1", "test.json", "/tmp/test.json")
    assert doc.id is not None
    assert doc.doc_id == "doc_test_1"

    fetched = get_document_by_doc_id(in_memory_db, "doc_test_1")
    assert fetched is not None
    assert fetched.id == doc.id


def test_save_document_with_chunks_duplicate_filtering(in_memory_db):
    chunks_data = [
        {"chunk_hash": "hash_a", "chunk_text": "Text A", "chunk_id": "id_a", "chunk_index": 0},
        {"chunk_hash": "hash_b", "chunk_text": "Text B", "chunk_id": "id_b", "chunk_index": 1},
    ]

    doc, inserted, skipped = save_document_with_chunks(
        db=in_memory_db,
        doc_id="doc_dupe_test",
        file_name="dupe.json",
        file_path="/tmp/dupe.json",
        chunks_data=chunks_data,
    )
    assert inserted == 2
    assert skipped == 0

    # Try saving document again with overlapping chunks
    new_chunks_data = [
        {"chunk_hash": "hash_a", "chunk_text": "Text A", "chunk_id": "id_a", "chunk_index": 0},
        {"chunk_hash": "hash_c", "chunk_text": "Text C", "chunk_id": "id_c", "chunk_index": 2},
    ]

    doc2, inserted2, skipped2 = save_document_with_chunks(
        db=in_memory_db,
        doc_id="doc_dupe_test",
        file_name="dupe.json",
        file_path="/tmp/dupe.json",
        chunks_data=new_chunks_data,
    )
    assert inserted2 == 1  # Only hash_c inserted
    assert skipped2 == 1   # hash_a skipped as duplicate


def test_extract_text_from_json_data():
    direct_scraping = {
        "source_type": "direct_web_scraping_llm",
        "data": {"title": "Law 119", "content": "Article 1 content"}
    }
    assert extract_text_from_json_data(direct_scraping) == "Article 1 content"

    ocr_data = {
        "pages": [{"markdown": "Page 1 text"}, {"markdown": "Page 2 text"}]
    }
    assert extract_text_from_json_data(ocr_data) == "Page 1 text\n\nPage 2 text"


def test_parse_single_json_document(tmp_path):
    json_file = tmp_path / "test_doc.json"
    payload = {"data": {"content": "Sample legal article body text."}}
    json_file.write_text(json.dumps(payload), encoding="utf-8")

    doc_dict = parse_single_json_document(json_file)
    assert doc_dict["doc_id"] == "test_doc"
    assert doc_dict["file_name"] == "test_doc.json"
    assert doc_dict["full_text"] == "Sample legal article body text."


def test_load_and_normalize_ingested_documents(tmp_path):
    json_file = tmp_path / "doc_1.json"
    payload = {"data": {"content": "Normalized text content."}}
    json_file.write_text(json.dumps(payload), encoding="utf-8")

    docs = load_and_normalize_ingested_documents(tmp_path)
    assert len(docs) == 1
    assert docs[0]["doc_id"] == "doc_1"
    assert docs[0]["full_text"] == "Normalized text content."


@pytest.mark.anyio
async def test_main_orchestrator_flow(tmp_path, monkeypatch):
    import uuid

    unique_token = uuid.uuid4().hex[:8]
    unique_doc_id = f"doc_test_flow_{unique_token}"
    sample_file = tmp_path / f"{unique_doc_id}.json"
    sample_payload = {
        "data": {
            "title": "Law 119",
            "content": f"Article 1: Sample legal text {unique_token} for testing chunking pipeline.\n\nArticle 2: All previous regulations are hereby repealed."
        }
    }
    sample_file.write_text(json.dumps(sample_payload, ensure_ascii=True), encoding="utf-8")

    stats = await main_orchestrator_flow(skip_ingestion=True, json_output_dir=tmp_path)
    assert stats["processed_docs"] == 1
    assert stats["inserted_chunks"] > 0


def test_vectorization_helpers(in_memory_db):
    chunks_data = [
        {"chunk_hash": "v_hash_1", "chunk_text": "Sample text for chunk 1", "chunk_id": "vec_1", "chunk_index": 0},
        {"chunk_hash": "v_hash_2", "chunk_text": "Sample text for chunk 2", "chunk_id": "vec_2", "chunk_index": 1},
    ]

    doc, inserted, _ = save_document_with_chunks(
        db=in_memory_db,
        doc_id="vec_test_doc",
        file_name="vec.json",
        file_path="/tmp/vec.json",
        chunks_data=chunks_data,
    )
    assert inserted == 2

    # 1. Test get_unembedded_chunks
    unembedded = get_unembedded_chunks(in_memory_db, batch_size=10)
    assert len(unembedded) == 2
    chunk1 = unembedded[0]

    # 2. Test get_chunks_with_metadata_for_vectorization
    payloads = get_chunks_with_metadata_for_vectorization(in_memory_db, batch_size=10)
    assert len(payloads) == 2
    assert payloads[0]["doc_id"] == "vec_test_doc"
    assert payloads[0]["file_name"] == "vec.json"
    assert "content" in payloads[0]

    # 3. Test update_chunk_vector_id
    success_v = update_chunk_vector_id(in_memory_db, chunk1.id, "qdrant_vec_uuid_100")
    assert success_v is True
    assert chunk1.vector_id == "qdrant_vec_uuid_100"

    # 4. Test update_chunk_embedding_status
    success_e = update_chunk_embedding_status(in_memory_db, chunk_id=chunk1.id, is_embedded=True)
    assert success_e is True

    # Verify unembedded chunks now count is 1
    remaining = get_unembedded_chunks(in_memory_db, batch_size=10)
    assert len(remaining) == 1

