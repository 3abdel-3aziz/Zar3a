"""
src/rag_database/vector_store.py

Flexible Vector Database module using Qdrant.
Automatically switches between Local Persistent Mode and Client-Server (Docker) Mode
based on configuration.

IMPORTANT – Singleton Design:
  In LOCAL mode (QDRANT_LOCAL_PATH set), Qdrant grabs an exclusive OS-level lock on the
  storage folder. Only ONE QdrantClient instance may exist per process. This module
  enforces that guarantee via a module-level singleton so that multiple code paths
  (API dependencies, RAG retriever, ingestion pipeline) all share the same client.
"""

import atexit
import logging
from threading import Lock
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

from src.rag_database.config import rag_config

logger = logging.getLogger("Zar3a.VectorStore")

# ---------------------------------------------------------------------------
# Module-level singleton for the QdrantClient (local mode safety)
# ---------------------------------------------------------------------------
_qdrant_client_singleton: Optional[QdrantClient] = None
_qdrant_client_lock: Lock = Lock()


def _get_or_create_qdrant_client() -> QdrantClient:
    """
    Returns the process-wide QdrantClient singleton, creating it lazily.

    In LOCAL mode this prevents the 'Storage folder already accessed by another
    instance of Qdrant client' error, since we only ever open the folder once.
    In client-server mode the singleton is still used for efficiency but the
    locking guarantee is not strictly required.
    """
    global _qdrant_client_singleton
    with _qdrant_client_lock:
        if _qdrant_client_singleton is None:
            local_path = rag_config.QDRANT_LOCAL_PATH
            if local_path:
                logger.info(
                    "Creating Qdrant singleton client in LOCAL mode using path: '%s'",
                    local_path,
                )
                _qdrant_client_singleton = QdrantClient(path=local_path)
            else:
                host = rag_config.VECTOR_DB_HOST
                port = rag_config.VECTOR_DB_PORT
                api_key = rag_config.VECTOR_DB_API_KEY
                logger.info(
                    "Creating Qdrant singleton client in CLIENT-SERVER mode for host='%s', port=%d",
                    host,
                    port,
                )
                if api_key:
                    _qdrant_client_singleton = QdrantClient(
                        host=host, port=port, api_key=api_key
                    )
                else:
                    _qdrant_client_singleton = QdrantClient(host=host, port=port)

            # Register a graceful shutdown hook so the lock is released cleanly
            atexit.register(_close_qdrant_client)

    return _qdrant_client_singleton


def _close_qdrant_client() -> None:
    """Gracefully closes the singleton Qdrant client on process exit."""
    global _qdrant_client_singleton
    with _qdrant_client_lock:
        if _qdrant_client_singleton is not None:
            try:
                _qdrant_client_singleton.close()
                logger.info("Qdrant singleton client closed.")
            except Exception:
                pass
            finally:
                _qdrant_client_singleton = None


def reset_qdrant_singleton() -> None:
    """Force-closes and resets the singleton (useful for tests or post-reingest)."""
    _close_qdrant_client()


class VectorStore:
    """
    Manages interactions with Qdrant Vector Database supporting both Local and Server modes.

    Always delegates to the process-wide singleton client so that only one file
    lock is held in local persistent mode.
    """

    def __init__(self):
        self.collection_name = rag_config.VECTOR_DB_COLLECTION_NAME
        self.dimension = rag_config.EMBEDDING_DIMENSION

        try:
            self.client = _get_or_create_qdrant_client()
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
                logger.info(
                    "Collection '%s' does not exist. Creating...", self.collection_name
                )
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.dimension,
                        distance=models.Distance.COSINE,
                    ),
                )
                logger.info(
                    "Created collection '%s' (dim=%d, Cosine).",
                    self.collection_name,
                    self.dimension,
                )
            else:
                logger.debug("Collection '%s' already exists.", self.collection_name)

        except UnexpectedResponse as exc:
            logger.error("Error checking/creating Qdrant collection: %s", exc)
            raise
        except Exception as exc:
            logger.error("Unexpected error in _ensure_collection_exists: %s", exc)
            raise

    def upsert_chunks(self, points_data: List[Dict[str, Any]]) -> bool:
        """Upserts a batch of points (vectors + payloads) into the Qdrant collection."""
        if not points_data:
            return True

        try:
            points = [
                models.PointStruct(
                    id=item["id"],
                    vector=item["vector"],
                    payload=item["payload"],
                )
                for item in points_data
            ]
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            logger.debug("Upserted %d point(s) to Qdrant.", len(points))
            return True

        except Exception as exc:
            logger.error("Failed to upsert points to VectorStore: %s", exc)
            return False

    def search(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Performs vector similarity search against the collection."""
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

    def scroll(
        self, limit: int = 100, offset: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """Retrieves points by scrolling through the collection."""
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