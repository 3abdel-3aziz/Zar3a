"""
src/rag_database/retriever.py

Hybrid Retriever module for the Zar3a RAG system.
Combines dense semantic vector search (multilingual E5 embeddings via Qdrant)
with sparse lexical search (BM25Okapi) using Reciprocal Rank Fusion (RRF).

Designed for production use and easy integration into agentic workflows.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from rank_bm25 import BM25Okapi

from src.rag_database.config import rag_config
from src.rag_database.vector_store import VectorStore

if TYPE_CHECKING:
    from src.rag_database.embedder import RAGEmbedder

logger = logging.getLogger("Zar3a.RAGRetriever")


class RAGRetriever:
    """
    Hybrid Retriever that combines dense vector similarity from Qdrant
    and lexical BM25 ranking over candidate passages using Reciprocal Rank Fusion (RRF).

    Key Features:
      - Uses RAGEmbedder with E5 query-prefixing ('query: ') for optimal dense retrieval.
      - Fetches an expanded candidate pool from Qdrant to ensure high semantic recall.
      - Computes BM25Okapi scores over candidates for exact keyword / entity precision.
      - Fuses dense and sparse rankings using Reciprocal Rank Fusion (RRF).
      - Returns clean, agentic-ready structured dictionaries with content, score, and metadata.
    """

    def __init__(
        self,
        embedder: Optional[RAGEmbedder] = None,
        vector_store: Optional[VectorStore] = None,
        rrf_k: Optional[int] = None,
        default_top_k: Optional[int] = None,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        candidate_multiplier: Optional[int] = None,
    ):
        """
        Initializes the Hybrid Retriever.

        Args:
            embedder: Pre-initialized RAGEmbedder instance. If None, instantiates a new one.
            vector_store: Pre-initialized VectorStore instance. If None, instantiates a new one.
            rrf_k: Smoothing constant for Reciprocal Rank Fusion (standard default: 60).
            default_top_k: Default number of top chunks to return (default: 5).
            dense_weight: Relative weight factor for dense vector ranking in RRF.
            sparse_weight: Relative weight factor for BM25 lexical ranking in RRF.
            candidate_multiplier: Multiplier to expand candidate pool fetched from vector store.
        """
        self._embedder = embedder
        self._vector_store = vector_store
        self.rrf_k = rrf_k if rrf_k is not None else getattr(rag_config, "RRF_K", 60)
        self.default_top_k = (
            default_top_k if default_top_k is not None else getattr(rag_config, "RETRIEVER_TOP_K", 5)
        )
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.candidate_multiplier = (
            candidate_multiplier
            if candidate_multiplier is not None
            else getattr(rag_config, "CANDIDATE_POOL_MULTIPLIER", 4)
        )

        logger.info(
            "Initialized RAGRetriever (rrf_k=%d, default_top_k=%d, dense_weight=%.2f, sparse_weight=%.2f)",
            self.rrf_k,
            self.default_top_k,
            self.dense_weight,
            self.sparse_weight,
        )

    @property
    def embedder(self) -> "RAGEmbedder":
        """Lazily initialize embedder if not provided at construction."""
        if self._embedder is None:
            logger.info("Initializing default RAGEmbedder for retriever...")
            from src.rag_database.embedder import RAGEmbedder
            self._embedder = RAGEmbedder()
        return self._embedder


    @property
    def vector_store(self) -> VectorStore:
        """Lazily initialize vector store if not provided at construction."""
        if self._vector_store is None:
            logger.info("Initializing default VectorStore for retriever...")
            self._vector_store = VectorStore()
        return self._vector_store

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Tokenizes text into words/subwords with support for multilingual (Arabic/English) text.

        Uses Unicode word character matching, lowercasing, and whitespace normalization.
        """
        if not text:
            return []
        tokens = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
        return tokens

    @staticmethod
    def compute_rrf_score(
        dense_rank: Optional[int],
        sparse_rank: Optional[int],
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ) -> float:
        """
        Calculates the Reciprocal Rank Fusion (RRF) score for a single document.

        Formula:
            score = (dense_weight / (k + rank_dense)) + (sparse_weight / (k + rank_sparse))

        Args:
            dense_rank: 1-based rank from dense retrieval (or None if unranked).
            sparse_rank: 1-based rank from sparse retrieval (or None if unranked).
            rrf_k: RRF smoothing constant.
            dense_weight: Multiplier weight for dense retrieval.
            sparse_weight: Multiplier weight for sparse retrieval.

        Returns:
            Calculated float RRF score.
        """
        score = 0.0
        if dense_rank is not None and dense_rank > 0:
            score += dense_weight / (rrf_k + dense_rank)
        if sparse_rank is not None and sparse_rank > 0:
            score += sparse_weight / (rrf_k + sparse_rank)
        return score

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Performs hybrid retrieval for a given query string using Dense + BM25 RRF.

        Execution Pipeline:
          1. Validates query input.
          2. Generates query dense vector using RAGEmbedder with 'query: ' prefix.
          3. Queries Qdrant VectorStore for candidate chunks (top_k * candidate_multiplier).
          4. Computes BM25Okapi lexical scores across the retrieved candidates.
          5. Computes RRF scores combining dense and lexical ranks.
          6. Formats and returns top_k results as clean, agentic-ready dictionaries.

        Args:
            query: The user query string.
            top_k: Maximum number of chunks to return (defaults to self.default_top_k).
            score_threshold: Optional minimum RRF score threshold to filter low-relevance results.

        Returns:
            List of structured result dictionaries containing:
              - chunk_id: Unique identifier for the chunk.
              - content: The text content of the chunk.
              - score: Final fused RRF score.
              - metadata: Dict with doc_id, file_name, file_path, chunk_index, etc.
              - dense_score: Raw cosine similarity score from Qdrant.
              - dense_rank: 1-based rank from dense retrieval.
              - sparse_score: Raw BM25 score.
              - sparse_rank: 1-based rank from BM25 retrieval (or None if no keyword match).
        """
        if not query or not query.strip():
            logger.warning("Empty or whitespace-only query received. Returning empty results.")
            return []

        clean_query = query.strip()
        limit = top_k if top_k is not None else self.default_top_k
        if limit <= 0:
            logger.warning("Invalid top_k (%d) specified; must be > 0.", limit)
            return []

        # Candidate pool size to retrieve from vector store
        candidate_limit = max(limit * self.candidate_multiplier, 20)

        try:
            # 1. Generate query embedding with 'query: ' prefix (handled by embedder.embed_query)
            logger.debug("Generating query embedding for: '%s'", clean_query[:80])
            query_vector = self.embedder.embed_query(clean_query)
            if not query_vector:
                logger.error("Failed to generate embedding vector for query.")
                return []

            # 2. Fetch candidate chunks from Qdrant Vector Store
            logger.debug("Querying Qdrant for top %d dense candidates...", candidate_limit)
            dense_candidates = self.vector_store.search(query_vector=query_vector, top_k=candidate_limit)

            if not dense_candidates:
                logger.info("No candidates returned from vector store for query: '%s'", clean_query[:80])
                return []

            logger.debug("Retrieved %d candidate chunks from Qdrant.", len(dense_candidates))

            # 3. Dense ranking (already sorted descending by cosine similarity score)
            # Map candidate_id -> (candidate_dict, dense_rank, dense_score)
            candidates_map: Dict[Any, Dict[str, Any]] = {}
            for rank_idx, item in enumerate(dense_candidates, start=1):
                c_id = item.get("id")
                candidates_map[c_id] = {
                    "item": item,
                    "dense_rank": rank_idx,
                    "dense_score": float(item.get("score", 0.0)),
                    "sparse_rank": None,
                    "sparse_score": 0.0,
                }

            # 4. Lexical / BM25 Ranking over Candidate Texts
            query_tokens = self.tokenize(clean_query)
            corpus_tokens: List[List[str]] = []
            id_order: List[Any] = []

            for item in dense_candidates:
                payload = item.get("payload") or {}
                # NOTE: ingestion pipeline stores text under "content" key
                text = payload.get("content", "") or payload.get("chunk_text", "")
                tokens = self.tokenize(text)
                corpus_tokens.append(tokens)
                id_order.append(item.get("id"))

            # Calculate BM25 scores if we have non-empty query and corpus tokens
            has_tokens = query_tokens and any(len(doc_t) > 0 for doc_t in corpus_tokens)
            if has_tokens:
                bm25 = BM25Okapi(corpus_tokens)
                raw_bm25_scores = bm25.get_scores(query_tokens)

                # Pair IDs with BM25 scores
                scored_sparse: List[Tuple[Any, float]] = [
                    (id_order[i], float(raw_bm25_scores[i]))
                    for i in range(len(id_order))
                ]

                # Filter only candidates with positive BM25 scores for sparse ranking
                # Chunks with 0 BM25 score have zero query term overlap and should not receive sparse rank
                positive_sparse = [pair for pair in scored_sparse if pair[1] > 0.0]
                positive_sparse.sort(key=lambda x: x[1], reverse=True)

                for sparse_rank, (c_id, b_score) in enumerate(positive_sparse, start=1):
                    if c_id in candidates_map:
                        candidates_map[c_id]["sparse_rank"] = sparse_rank
                        candidates_map[c_id]["sparse_score"] = b_score

                # Also record raw BM25 scores for zero-scored items
                for c_id, b_score in scored_sparse:
                    if c_id in candidates_map and candidates_map[c_id]["sparse_rank"] is None:
                        candidates_map[c_id]["sparse_score"] = b_score
            else:
                logger.debug("Query or corpus has no alphanumeric tokens; skipping BM25 scoring.")

            # 5. Calculate RRF scores and assemble result list
            fused_results: List[Dict[str, Any]] = []
            for c_id, data in candidates_map.items():
                item = data["item"]
                payload = item.get("payload") or {}

                dense_rank = data["dense_rank"]
                sparse_rank = data["sparse_rank"]
                dense_score = data["dense_score"]
                sparse_score = data["sparse_score"]

                rrf_score = self.compute_rrf_score(
                    dense_rank=dense_rank,
                    sparse_rank=sparse_rank,
                    rrf_k=self.rrf_k,
                    dense_weight=self.dense_weight,
                    sparse_weight=self.sparse_weight,
                )

                chunk_id = payload.get("chunk_id")
                if chunk_id is None:
                    chunk_id = item.get("id")

                # NOTE: ingestion pipeline stores text under "content" key;
                # fall back to "chunk_text" for any legacy points ingested
                # before this schema was standardised.
                chunk_text = (
                    payload.get("content", "")
                    or payload.get("chunk_text", "")
                    or ""
                )

                result_entry = {
                    "chunk_id": chunk_id,
                    "content": chunk_text,
                    "score": round(rrf_score, 6),
                    "metadata": {
                        "doc_id": payload.get("doc_id"),
                        "file_name": payload.get("file_name"),
                        "file_path": payload.get("file_path"),
                        "chunk_index": payload.get("chunk_index"),
                        "chunk_hash": payload.get("chunk_hash"),
                        "vector_id": payload.get("vector_id"),
                        "created_at": payload.get("created_at"),
                    },
                    "dense_score": round(dense_score, 4),
                    "dense_rank": dense_rank,
                    "sparse_score": round(sparse_score, 4),
                    "sparse_rank": sparse_rank,
                }
                fused_results.append(result_entry)

            # Sort fused results by RRF score descending (tie-break with dense_score)
            fused_results.sort(
                key=lambda x: (x["score"], x["dense_score"]),
                reverse=True,
            )

            # Apply score threshold if specified
            if score_threshold is not None:
                fused_results = [r for r in fused_results if r["score"] >= score_threshold]

            # Slice to requested top_k limit
            final_results = fused_results[:limit]
            logger.info(
                "Retrieved %d relevant chunk(s) for query: '%s' (top score: %.5f)",
                len(final_results),
                clean_query[:50],
                final_results[0]["score"] if final_results else 0.0,
            )
            return final_results

        except Exception as exc:
            logger.error("Error during hybrid retrieval for query '%s': %s", clean_query[:50], exc, exc_info=True)
            return []

    def retrieve_as_context(self, query: str, top_k: Optional[int] = None) -> str:
        """
        Helper method for LLM prompting: retrieves top chunks and formats them
        into a single, clean markdown context string suitable for prompt injection.

        Args:
            query: The user query string.
            top_k: Number of chunks to include.

        Returns:
            Formatted context string.
        """
        results = self.retrieve(query, top_k=top_k)
        if not results:
            return "No relevant context found in knowledge base."

        context_blocks = []
        for i, res in enumerate(results, start=1):
            meta = res.get("metadata", {})
            file_name = meta.get("file_name") or "Unknown"
            chunk_idx = meta.get("chunk_index")
            chunk_info = f" (Chunk {chunk_idx})" if chunk_idx is not None else ""
            header = f"[Source {i}: {file_name}{chunk_info} | Relevance Score: {res['score']:.4f}]"
            context_blocks.append(f"{header}\n{res['content']}")

        return "\n\n---\n\n".join(context_blocks)

    def as_tool_spec(self) -> Dict[str, Any]:
        """
        Returns a standard OpenAI/LangChain function tool specification
        for binding this retriever directly into an agentic workflow.
        """
        return {
            "type": "function",
            "function": {
                "name": "retrieve_agricultural_knowledge",
                "description": (
                    "Retrieves relevant agricultural, climate, soil, and crop management "
                    "knowledge passages from the Zar3a RAG database based on a user query."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query, topic, or question in English or Arabic.",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "The maximum number of relevant chunks to retrieve (default: 5).",
                            "default": self.default_top_k,
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    def __call__(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Makes the instance directly callable like a function."""
        return self.retrieve(query, top_k=top_k)
