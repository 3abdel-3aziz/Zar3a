"""
src/rag_database/vector_store.py

Flexible Vector Database module using Qdrant.
Automatically switches between Local Persistent Mode and Client-Server (Docker) Mode 
based on configuration.
"""

import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

from src.rag_database.config import rag_config

logger = logging.getLogger("Zar3a.VectorStore")


class VectorStore:
    """
    Manages interactions with Qdrant Vector Database supporting both Local and Server modes.
    """

    def __init__(self):
        self.collection_name = rag_config.VECTOR_DB_COLLECTION_NAME
        self.dimension = rag_config.EMBEDDING_DIMENSION
        self.local_path = rag_config.QDRANT_LOCAL_PATH

        self.host = rag_config.VECTOR_DB_HOST
        self.port = rag_config.VECTOR_DB_PORT
        self.api_key = rag_config.VECTOR_DB_API_KEY

        try:
            # Flexible Initialization: Local Path vs Client-Server Mode
            if self.local_path:
                logger.info("Initializing Qdrant in LOCAL mode using path: '%s'", self.local_path)
                self.client = QdrantClient(path=self.local_path)
            else:
                logger.info(
                    "Initializing Qdrant in CLIENT-SERVER mode for host='%s', port=%d",
                    self.host, self.port
                )
                if self.api_key:
                    self.client = QdrantClient(host=self.host, port=self.port, api_key=self.api_key)
                else:
                    self.client = QdrantClient(host=self.host, port=self.port)
            
            # Ensure collection exists
            self._ensure_collection_exists()

        except Exception as exc:
            logger.error("Failed to initialize Qdrant VectorStore: %s", exc)
            raise RuntimeError(f"Could not connect to Qdrant: {exc}") from exc

    def _ensure_collection_exists(self) -> None:
        """
        Checks if the collection exists in Qdrant. If not, creates it with 
        the correct vector size (1024) and Cosine distance metric.
        """
        try:
            collections = self.client.get_collections().collections
            exists = any(col.name == self.collection_name for col in collections)

            if not exists:
                logger.info("Collection '%s' does not exist. Creating new collection...", self.collection_name)
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.dimension,
                        distance=models.Distance.COSINE
                    )
                )
                logger.info("Successfully created collection '%s' with dimension %d.", self.collection_name, self.dimension)
            else:
                logger.debug("Collection '%s' already exists.", self.collection_name)

        except UnexpectedResponse as exc:
            logger.error("Error checking/creating Qdrant collection: %s", exc)
            raise
        except Exception as exc:
            logger.error("Unexpected error in _ensure_collection_exists: %s", exc)
            raise

    def upsert_chunks(self, points_data: List[Dict[str, Any]]) -> bool:
        """
        Upserts a batch of points (vectors + payloads) into the Qdrant collection.
        """
        if not points_data:
            return True

        try:
            points = [
                models.PointStruct(
                    id=item["id"],
                    vector=item["vector"],
                    payload=item["payload"]
                )
                for item in points_data
            ]

            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.debug("Successfully upserted %d point(s) to Qdrant.", len(points))
            return True

        except Exception as exc:
            logger.error("Failed to upsert points to VectorStore: %s", exc)
            return False

    def search(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Performs vector similarity search against the collection.
        Uses query_points() — the current qdrant-client API (>=1.7).
        """
        try:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
            )

            results = [
                {
                    "id": hit.id,
                    "score": hit.score,
                    "payload": hit.payload,
                }
                for hit in response.points
            ]

            logger.debug("Vector search returned %d result(s).", len(results))
            return results

        except Exception as exc:
            logger.error("Error during vector similarity search: %s", exc)
            return []

    def scroll(self, limit: int = 100, offset: Optional[Any] = None) -> List[Dict[str, Any]]:
        """
        Retrieves points by scrolling through the collection.
        Useful for building lexical indices or exporting points.
        """
        try:
            records, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            results = [
                {
                    "id": rec.id,
                    "score": 0.0,
                    "payload": rec.payload or {},
                }
                for rec in records
            ]
            logger.debug("VectorStore scroll returned %d point(s).", len(results))
            return results

        except Exception as exc:
            logger.error("Error during vector store scroll: %s", exc)
            return []
