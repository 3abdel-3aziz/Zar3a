"""
src/api/routers/knowledge.py

FastAPI router exposing the Zar3a RAG Vector Search & Knowledge Base.
Provides RESTful endpoints to query environmental laws, forestry regulations, and greening standards.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from api.dependencies import get_rag_retriever
from rag_database.retriever import RAGRetriever

logger = logging.getLogger("Zar3a.API.Knowledge")

router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class KnowledgeSearchRequest(BaseModel):
    """Input payload for searching legal and environmental document chunks."""

    query: str = Field(
        ...,
        min_length=1,
        description="Query text regarding environmental laws, setback rules, or tree preservation.",
        examples=["ما هي ضوابط زراعة الأشجار على الأرصفة في قانون البيئة؟"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of most relevant document chunks to retrieve (1-20).",
    )


class DocumentChunkItem(BaseModel):
    """Individual document chunk returned from the Qdrant RAG vector store."""

    chunk_id: Optional[str] = Field(None, description="Unique chunk hash or identifier")
    content: str = Field(..., description="Text passage excerpt from the legal/environmental document")
    score: float = Field(..., description="Hybrid Reciprocal Rank Fusion (RRF) similarity score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata (file name, section, source)")


class KnowledgeSearchResponse(BaseModel):
    """Structured response containing retrieved document passages."""

    query: str = Field(..., description="The query string executed")
    total_results: int = Field(..., description="Number of document passages retrieved")
    results: List[DocumentChunkItem] = Field(..., description="Ordered list of candidate passages")


# ---------------------------------------------------------------------------
# Route Handlers
# ---------------------------------------------------------------------------

@router.post(
    "/documents/search",
    response_model=KnowledgeSearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Search Legal & Environmental Documents",
    description="Performs hybrid dense-sparse vector search against Qdrant to retrieve authoritative excerpts.",
)
def search_documents(
    payload: KnowledgeSearchRequest,
    retriever: RAGRetriever = Depends(get_rag_retriever),
) -> KnowledgeSearchResponse:
    """
    Executes hybrid RAG retrieval using dense embeddings + BM25 reciprocal rank fusion.
    """
    try:
        raw_chunks = retriever.retrieve(query=payload.query, top_k=payload.top_k)
        items: List[DocumentChunkItem] = []

        for chunk in raw_chunks:
            items.append(
                DocumentChunkItem(
                    chunk_id=chunk.get("chunk_id"),
                    content=str(chunk.get("content", "")),
                    score=round(float(chunk.get("score", 0.0)), 4),
                    metadata=chunk.get("metadata", {}) or {},
                )
            )

        return KnowledgeSearchResponse(
            query=payload.query,
            total_results=len(items),
            results=items,
        )
    except Exception as exc:
        logger.error("RAG retrieval failed for query '%s': %s", payload.query, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Knowledge retrieval failed: {exc}",
        ) from exc


@router.get(
    "/documents",
    response_model=KnowledgeSearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Query Knowledge Documents (GET)",
    description="Convenience query endpoint to retrieve legal passages via standard query parameters.",
)
def query_documents(
    q: str = Query(..., min_length=1, description="Search query string"),
    top_k: int = Query(5, ge=1, le=20, description="Maximum number of passages to return"),
    retriever: RAGRetriever = Depends(get_rag_retriever),
) -> KnowledgeSearchResponse:
    """
    HTTP GET interface for searching the knowledge base.
    """
    return search_documents(payload=KnowledgeSearchRequest(query=q, top_k=top_k), retriever=retriever)
