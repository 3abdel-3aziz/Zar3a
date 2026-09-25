"""
src/rag_database

RAG and Vector Database package for the Zar3a project.
Provides embedding generation, Qdrant vector storage, ingestion pipeline,
and hybrid retrieval with Reciprocal Rank Fusion (RRF).
"""

from typing import TYPE_CHECKING
from src.rag_database.config import rag_config, RAGSettings

if TYPE_CHECKING:
    from src.rag_database.embedder import RAGEmbedder
    from src.rag_database.vector_store import VectorStore
    from src.rag_database.rag_pipeline import RAGIngestionPipeline
    from src.rag_database.retriever import RAGRetriever

__all__ = [
    "rag_config",
    "RAGSettings",
    "RAGEmbedder",
    "VectorStore",
    "RAGIngestionPipeline",
    "RAGRetriever",
]


def __getattr__(name: str):
    if name == "RAGEmbedder":
        from src.rag_database.embedder import RAGEmbedder
        return RAGEmbedder
    if name == "VectorStore":
        from src.rag_database.vector_store import VectorStore
        return VectorStore
    if name == "RAGIngestionPipeline":
        from src.rag_database.rag_pipeline import RAGIngestionPipeline
        return RAGIngestionPipeline
    if name == "RAGRetriever":
        from src.rag_database.retriever import RAGRetriever
        return RAGRetriever
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
