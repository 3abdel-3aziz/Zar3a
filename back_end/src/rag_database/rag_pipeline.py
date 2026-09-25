"""
src/rag_database/rag_pipeline.py

RAG Ingestion Pipeline for Zar3a project.
Orchestrates fetching un-embedded chunks from PostgreSQL, generating embeddings,
upserting them into Qdrant Vector Store, and updating their status back in the database.
"""

import logging
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session

from src.database.operations import (
    get_chunks_with_metadata_for_vectorization,
    update_chunk_embedding_status,
    update_chunk_vector_id,
)
from src.rag_database.embedder import RAGEmbedder  
from src.rag_database.vector_store import VectorStore
from src.rag_database.config import rag_config

logger = logging.getLogger("Zar3a.RAGPipeline")


class RAGIngestionPipeline:
    """
    Manages the end-to-end ingestion pipeline from relational database to vector database.
    """

    def __init__(self, embedder: Optional[RAGEmbedder] = None, vector_store: Optional[VectorStore] = None):
        logger.info("Initializing RAGIngestionPipeline...")
        self.embedder = embedder if embedder else RAGEmbedder()
        self.vector_store = vector_store if vector_store else VectorStore()

    def process_unembedded_chunks(self, db: Session, batch_size: Optional[int] = None) -> int:
        """
        Fetches a batch of un-embedded chunks from PostgreSQL, generates embeddings,
        upserts them to Qdrant, and marks them as embedded in the database.

        Args:
            db: SQLAlchemy database session.
            batch_size: Number of chunks to process in this batch (defaults to config).

        Returns:
            int: Total number of successfully processed and indexed chunks.
        """
        limit = batch_size if batch_size else rag_config.RAG_BATCH_SIZE

        # 1. Fetch un-embedded chunks with metadata using DB operations
        chunks_payloads = get_chunks_with_metadata_for_vectorization(db, batch_size=limit)
        if not chunks_payloads:
            logger.debug("No un-embedded chunks found to process.")
            return 0

        logger.info("Fetched %d un-embedded chunk(s) from database.", len(chunks_payloads))

        # 2. Extract texts for embedding generation
        texts_to_embed = [item["content"] for item in chunks_payloads]

        try:
            # 3. Generate embeddings using RAGEmbedder (is_query=False for passages)
            logger.debug("Generating embeddings for %d text(s)...", len(texts_to_embed))
            embeddings = self.embedder.embed_texts(texts_to_embed, is_query=False)

            if len(embeddings) != len(chunks_payloads):
                logger.error("Mismatch between number of texts (%d) and generated embeddings (%d).", 
                             len(chunks_payloads), len(embeddings))
                return 0

            # 4. Prepare points data for Qdrant Vector Store upsert
            points_data: List[Dict[str, Any]] = []
            for idx, item in enumerate(chunks_payloads):
                chunk_id = item["chunk_id"]
                # Use chunk_id or generate a consistent vector identifier string if needed
                vector_id = item.get("vector_id") or f"vec_{chunk_id}"
                
                points_data.append({
                    "id": chunk_id,  # Using relational chunk ID as Qdrant point ID for easy mapping
                    "vector": embeddings[idx],
                    "payload": {
                        "chunk_id": chunk_id,
                        "vector_id": vector_id,
                        "chunk_hash": item["chunk_hash"],
                        "chunk_index": item["chunk_index"],
                        "content": item["content"],
                        "doc_id": item["doc_id"],
                        "file_name": item["file_name"],
                        "file_path": item["file_path"],
                        "created_at": str(item["created_at"]),
                    }
                })

            # 5. Upsert points into Qdrant Vector Store
            success = self.vector_store.upsert_chunks(points_data)
            if not success:
                logger.error("Failed to upsert points batch into Qdrant Vector Store.")
                return 0

            # 6. Update chunk status back in the relational database
            # Note: each helper calls db.commit() individually — acceptable for small
            # batches; refactor to bulk update if batch sizes grow significantly.
            processed_count = 0
            for item, point in zip(chunks_payloads, points_data):
                chunk_id = item["chunk_id"]
                vector_id = point["payload"]["vector_id"]  # reuse the already-computed value

                update_chunk_vector_id(db, chunk_id=chunk_id, vector_id=vector_id)
                status_updated = update_chunk_embedding_status(db, chunk_id=chunk_id, is_embedded=True)

                if status_updated:
                    processed_count += 1

            logger.info(
                "Successfully processed, vectorized, and indexed %d/%d chunk(s).",
                processed_count,
                len(chunks_payloads),
            )
            return processed_count

        except Exception as exc:
            logger.error("Error occurred during RAG ingestion pipeline execution: %s", exc)
            db.rollback()  # Prevent a dirty session from being reused after a mid-batch failure
            return 0