"""
src/database/operations.py

Database Operations Layer for Zar3a project.
Implements granular database queries, record creations, duplicate chunk checks,
and atomic transaction workflows following the Single Responsibility Principle (SRP).
"""

import logging
from typing import List, Set, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from src.database.models import DocumentModel, DocumentChunkModel

logger = logging.getLogger("Zar3a.DatabaseOperations")


# ── 1. Input Validation & Data Mapping Helpers ───────────────────────────────

def validate_document_inputs(doc_id: str, file_name: str, file_path: str) -> None:
    """
    Validates document metadata before database operations.
    Raises ValueError if required fields are empty or invalid.
    """
    if not doc_id or not isinstance(doc_id, str) or not doc_id.strip():
        raise ValueError("doc_id must be a non-empty string.")
    if not file_name or not isinstance(file_name, str) or not file_name.strip():
        raise ValueError("file_name must be a non-empty string.")
    if not file_path or not isinstance(file_path, str) or not file_path.strip():
        raise ValueError("file_path must be a non-empty string.")


def validate_chunk_data(chunk_data: Dict[str, Any]) -> None:
    """
    Validates a single chunk dictionary structure.
    Raises ValueError if mandatory fields are missing.
    """
    if not isinstance(chunk_data, dict):
        raise ValueError(f"Chunk data must be a dictionary, got {type(chunk_data)}.")

    chunk_hash = chunk_data.get("chunk_hash")
    if not chunk_hash or not isinstance(chunk_hash, str) or not chunk_hash.strip():
        raise ValueError(f"Chunk dictionary missing valid 'chunk_hash': {chunk_data}")

    content = chunk_data.get("chunk_text") or chunk_data.get("content")
    if not content or not isinstance(content, str) or not content.strip():
        raise ValueError(f"Chunk dictionary missing valid content/chunk_text: {chunk_data}")


def map_dict_to_chunk_model(document_id: int, chunk_data: Dict[str, Any], index: int) -> DocumentChunkModel:
    """
    Maps a raw chunk dictionary to a DocumentChunkModel SQLAlchemy instance.
    """
    validate_chunk_data(chunk_data)
    
    content = chunk_data.get("chunk_text") or chunk_data.get("content", "")
    chunk_hash = chunk_data["chunk_hash"]
    vector_id = chunk_data.get("chunk_id") or chunk_data.get("vector_id")

    return DocumentChunkModel(
        document_id=document_id,
        content=content,
        vector_id=vector_id,
        chunk_index=chunk_data.get("chunk_index", index),
        chunk_hash=chunk_hash,
        is_embedded=chunk_data.get("is_embedded", False),
    )


# ── 2. Query / Check Operations (Read-Only) ───────────────────────────────────

def get_document_by_doc_id(db: Session, doc_id: str) -> Optional[DocumentModel]:
    """
    Queries the database for an existing DocumentModel record by doc_id.
    """
    return db.query(DocumentModel).filter(DocumentModel.doc_id == doc_id).first()


def get_existing_chunk_identifiers(
    db: Session, 
    chunks_data: List[Dict[str, Any]]
) -> Tuple[Set[str], Set[str]]:
    """
    Bulk queries the database to check which chunk_hashes AND vector_ids already exist.
    Returns (Set of existing hashes, Set of existing vector_ids).
    """
    if not chunks_data:
        return set(), set()

    input_hashes = [c["chunk_hash"] for c in chunks_data if c.get("chunk_hash")]
    input_vector_ids = [
        c.get("chunk_id") or c.get("vector_id")
        for c in chunks_data
        if (c.get("chunk_id") or c.get("vector_id"))
    ]

    existing_hashes: Set[str] = set()
    if input_hashes:
        records = (
            db.query(DocumentChunkModel.chunk_hash)
            .filter(DocumentChunkModel.chunk_hash.in_(input_hashes))
            .all()
        )
        existing_hashes = {rec[0] for rec in records if rec[0]}

    existing_vector_ids: Set[str] = set()
    if input_vector_ids:
        records = (
            db.query(DocumentChunkModel.vector_id)
            .filter(DocumentChunkModel.vector_id.in_(input_vector_ids))
            .all()
        )
        existing_vector_ids = {rec[0] for rec in records if rec[0]}

    return existing_hashes, existing_vector_ids


# Alias for backward compatibility
get_existing_chunk_hashes = get_existing_chunk_identifiers


def is_chunk_hash_exists(db: Session, chunk_hash: str) -> bool:
    """
    Checks if a single chunk_hash already exists in the document_chunks table.
    """
    return db.query(
        db.query(DocumentChunkModel).filter(DocumentChunkModel.chunk_hash == chunk_hash).exists()
    ).scalar()


# ── 3. Granular Write Operations (Write-Only) ─────────────────────────────────

def insert_document(db: Session, doc_id: str, file_name: str, file_path: str) -> DocumentModel:
    """
    Creates and adds a new DocumentModel record to the session.
    Flushes session to assign primary key ID without committing transaction.
    """
    validate_document_inputs(doc_id, file_name, file_path)

    doc_model = DocumentModel(
        doc_id=doc_id,
        file_name=file_name,
        file_path=file_path,
    )
    db.add(doc_model)
    db.flush()  # Populates doc_model.id
    logger.debug("Created DocumentModel(id=%s, doc_id='%s')", doc_model.id, doc_id)
    return doc_model


def insert_chunks_batch(
    db: Session, 
    document_id: int, 
    chunks_data: List[Dict[str, Any]]
) -> List[DocumentChunkModel]:
    """
    Maps and bulk adds chunk records to the session.
    Does not issue commit; callers orchestrate full transaction commit.
    """
    if not chunks_data:
        return []

    chunk_models: List[DocumentChunkModel] = []
    for idx, chunk_dict in enumerate(chunks_data):
        chunk_model = map_dict_to_chunk_model(document_id, chunk_dict, index=idx)
        chunk_models.append(chunk_model)

    db.add_all(chunk_models)
    db.flush()
    logger.debug("Added %d DocumentChunkModel records for document_id=%s", len(chunk_models), document_id)
    return chunk_models


# ── 4. Pure Functional Filters ───────────────────────────────────────────────

def filter_duplicate_chunks(
    existing_hashes: Set[str],
    existing_vector_ids: Set[str],
    chunks_data: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Filters out chunk dicts whose chunk_hash OR vector_id already exists in the database.
    Returns (new_chunks, skipped_count).
    """
    new_chunks: List[Dict[str, Any]] = []
    skipped_count = 0

    for chunk in chunks_data:
        ch_hash = chunk.get("chunk_hash")
        vec_id = chunk.get("chunk_id") or chunk.get("vector_id")

        is_hash_dupe = bool(ch_hash and ch_hash in existing_hashes)
        is_vec_dupe = bool(vec_id and vec_id in existing_vector_ids)

        if is_hash_dupe or is_vec_dupe:
            skipped_count += 1
        else:
            new_chunks.append(chunk)

    return new_chunks, skipped_count


# ── 5. High-Level Composite Transaction Workflow ──────────────────────────────

def save_document_with_chunks(
    db: Session,
    doc_id: str,
    file_name: str,
    file_path: str,
    chunks_data: List[Dict[str, Any]],
) -> Tuple[DocumentModel, int, int]:
    """
    High-level composite transaction pipeline for saving a document and its chunks.
    Coordinates validation, document creation/retrieval, duplicate checking, batch insertion,
    and atomic commit/rollback.

    Args:
        db: SQLAlchemy database session.
        doc_id: Unique document string identifier.
        file_name: Origin file name.
        file_path: Path to the original or cached source file.
        chunks_data: List of chunk dictionaries produced by ChunkingPipeline.

    Returns:
        Tuple[DocumentModel, int, int]: (document instance, inserted_chunks_count, skipped_chunks_count)
    """
    validate_document_inputs(doc_id, file_name, file_path)

    try:
        # 1. Fetch existing document or insert new one
        doc_model = get_document_by_doc_id(db, doc_id)
        if not doc_model:
            doc_model = insert_document(db, doc_id, file_name, file_path)

        if not chunks_data:
            db.commit()
            return doc_model, 0, 0

        # 2. Extract hashes and vector IDs to check existing records in DB
        existing_hashes, existing_vector_ids = get_existing_chunk_identifiers(db, chunks_data)

        # 3. Filter out duplicates
        new_chunks, skipped_count = filter_duplicate_chunks(
            existing_hashes, existing_vector_ids, chunks_data
        )

        # 4. Batch insert new chunks
        inserted_models = insert_chunks_batch(db, doc_model.id, new_chunks)

        # 5. Commit atomic transaction
        db.commit()
        db.refresh(doc_model)

        logger.info(
            "Saved document doc_id='%s' (ID=%d): %d chunks inserted, %d chunks skipped (duplicates).",
            doc_id,
            doc_model.id,
            len(inserted_models),
            skipped_count,
        )
        return doc_model, len(inserted_models), skipped_count

    except Exception as exc:
        db.rollback()
        logger.error("Transaction failed for doc_id='%s'. Rolling back. Error: %s", doc_id, exc)
        raise SQLAlchemyError(f"Failed to save document '{doc_id}' with chunks: {exc}") from exc


# ── 6. Embedding & Vector Database Operations ───────────────────────────────

def get_unembedded_chunks(db: Session, batch_size: int = 50) -> List[DocumentChunkModel]:
    """
    Retrieves a batch of DocumentChunkModel records where is_embedded is False.

    Args:
        db: SQLAlchemy database session.
        batch_size: Maximum number of records to retrieve (default: 50).

    Returns:
        List[DocumentChunkModel]: List of unembedded chunk model instances.
    """
    return (
        db.query(DocumentChunkModel)
        .filter(DocumentChunkModel.is_embedded == False)
        .limit(batch_size)
        .all()
    )


def update_chunk_embedding_status(
    db: Session,
    chunk_id: Optional[int] = None,
    chunk_hash: Optional[str] = None,
    vector_id: Optional[str] = None,
    is_embedded: bool = True,
) -> bool:
    """
    Updates the is_embedded boolean flag for a specific chunk identified by primary key id,
    chunk_hash, or vector_id. Handles commits and rollbacks safely.

    Args:
        db: SQLAlchemy database session.
        chunk_id: Primary key ID of the chunk.
        chunk_hash: SHA-256 hash string of the chunk.
        vector_id: Deterministic vector ID of the chunk.
        is_embedded: Target boolean status (default: True).

    Returns:
        bool: True if a chunk record was found and updated successfully, False otherwise.
    """
    if chunk_id is None and not chunk_hash and not vector_id:
        logger.warning("update_chunk_embedding_status called without any chunk identifier.")
        return False

    try:
        query = db.query(DocumentChunkModel)
        if chunk_id is not None:
            query = query.filter(DocumentChunkModel.id == chunk_id)
        elif chunk_hash:
            query = query.filter(DocumentChunkModel.chunk_hash == chunk_hash)
        elif vector_id:
            query = query.filter(DocumentChunkModel.vector_id == vector_id)

        chunk = query.first()
        if not chunk:
            logger.warning(
                "No chunk found matching identifiers (chunk_id=%s, chunk_hash=%s, vector_id=%s).",
                chunk_id,
                chunk_hash,
                vector_id,
            )
            return False

        chunk.is_embedded = is_embedded
        db.commit()
        logger.debug("Updated chunk ID=%d is_embedded status to %s.", chunk.id, is_embedded)
        return True

    except Exception as exc:
        db.rollback()
        logger.error("Failed to update chunk embedding status: %s", exc)
        return False


def update_chunk_vector_id(db: Session, chunk_id: int, vector_id: str) -> bool:
    """
    Updates the vector_id field for a specific chunk after it has been indexed in the Vector Database.

    Args:
        db: SQLAlchemy database session.
        chunk_id: Primary key ID of the target chunk.
        vector_id: Vector database identifier string to assign.

    Returns:
        bool: True if updated successfully, False otherwise.
    """
    if not chunk_id or not vector_id:
        logger.warning("update_chunk_vector_id requires non-null chunk_id and vector_id.")
        return False

    try:
        chunk = db.query(DocumentChunkModel).filter(DocumentChunkModel.id == chunk_id).first()
        if not chunk:
            logger.warning("No chunk found with id=%s to update vector_id.", chunk_id)
            return False

        chunk.vector_id = vector_id
        db.commit()
        logger.debug("Updated chunk ID=%d with vector_id='%s'.", chunk_id, vector_id)
        return True

    except Exception as exc:
        db.rollback()
        logger.error("Failed to update vector_id for chunk ID=%d: %s", chunk_id, exc)
        return False


def get_chunks_with_metadata_for_vectorization(
    db: Session, 
    batch_size: int = 50
) -> List[Dict[str, Any]]:
    """
    Performs a SQL JOIN between DocumentChunkModel and DocumentModel to fetch unembedded chunks
    (is_embedded == False) along with parent document metadata.

    Args:
        db: SQLAlchemy database session.
        batch_size: Maximum number of records to retrieve (default: 50).

    Returns:
        List[Dict[str, Any]]: List of dictionary payloads formatted for Vector DB ingestion.
    """
    try:
        results = (
            db.query(DocumentChunkModel, DocumentModel)
            .join(DocumentModel, DocumentChunkModel.document_id == DocumentModel.id)
            .filter(DocumentChunkModel.is_embedded == False)
            .limit(batch_size)
            .all()
        )

        payloads: List[Dict[str, Any]] = []
        for chunk, doc in results:
            payloads.append({
                "chunk_id": chunk.id,
                "vector_id": chunk.vector_id,
                "chunk_hash": chunk.chunk_hash,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "is_embedded": chunk.is_embedded,
                "doc_id": doc.doc_id,
                "file_name": doc.file_name,
                "file_path": doc.file_path,
                "created_at": doc.created_at,
            })

        logger.debug("Fetched %d chunk payload(s) for vectorization.", len(payloads))
        return payloads

    except Exception as exc:
        logger.error("Failed to fetch chunks with metadata for vectorization: %s", exc)
        return []
