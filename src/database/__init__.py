from .connection import Base, engine, SessionLocal, get_db
from .models import DocumentModel, DocumentChunkModel
from .operations import (
    save_document_with_chunks,
    get_document_by_doc_id,
    get_existing_chunk_identifiers,
    insert_document,
    insert_chunks_batch,
    get_unembedded_chunks,
    update_chunk_embedding_status,
    update_chunk_vector_id,
    get_chunks_with_metadata_for_vectorization,
)

# Alias for backwards compatibility
get_existing_chunk_hashes = get_existing_chunk_identifiers

__all__ = [
    "Base",
    "engine",
    "SessionLocal",
    "get_db",
    "DocumentModel",
    "DocumentChunkModel",
    "save_document_with_chunks",
    "get_document_by_doc_id",
    "get_existing_chunk_identifiers",
    "get_existing_chunk_hashes",
    "insert_document",
    "insert_chunks_batch",
    "get_unembedded_chunks",
    "update_chunk_embedding_status",
    "update_chunk_vector_id",
    "get_chunks_with_metadata_for_vectorization",
]
