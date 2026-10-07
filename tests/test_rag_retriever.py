"""
tests/test_rag_retriever.py

Comprehensive unit tests for RAGRetriever hybrid retrieval and RRF scoring.
"""

from unittest.mock import MagicMock
import pytest

from src.rag_database.retriever import RAGRetriever


@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    # Mock embed_query returning a dummy vector
    embedder.embed_query.return_value = [0.1] * 1024
    return embedder


@pytest.fixture
def sample_candidates():
    return [
        {
            "id": 1,
            "score": 0.95,
            "payload": {
                "chunk_id": 101,
                "chunk_text": "Water conservation in agriculture through drip irrigation and mulching.",
                "content": "Water conservation in agriculture through drip irrigation and mulching.",
                "doc_id": 10,
                "file_name": "irrigation_guide.pdf",
                "file_path": "data/irrigation_guide.pdf",
                "chunk_index": 0,
                "chunk_hash": "hash_101",
                "vector_id": "vec_101",
                "created_at": "2026-09-24T00:00:00",
            },
        },
        {
            "id": 2,
            "score": 0.85,
            "payload": {
                "chunk_id": 102,
                "chunk_text": "Citrus tree cultivation in arid and semi-arid climates.",
                "content": "Citrus tree cultivation in arid and semi-arid climates.",
                "doc_id": 11,
                "file_name": "citrus_trees.pdf",
                "file_path": "data/citrus_trees.pdf",
                "chunk_index": 1,
                "chunk_hash": "hash_102",
                "vector_id": "vec_102",
                "created_at": "2026-09-24T00:00:00",
            },
        },
        {
            "id": 3,
            "score": 0.75,
            "payload": {
                "chunk_id": 103,
                "chunk_text": "الزراعة المستدامة وإدارة الموارد المائية في مصر في بيئة جافة.",
                "content": "الزراعة المستدامة وإدارة الموارد المائية في مصر في بيئة جافة.",
                "doc_id": 12,
                "file_name": "egypt_agriculture.pdf",
                "file_path": "data/egypt_agriculture.pdf",
                "chunk_index": 2,
                "chunk_hash": "hash_103",
                "vector_id": "vec_103",
                "created_at": "2026-09-24T00:00:00",
            },
        },
    ]


@pytest.fixture
def mock_vector_store(sample_candidates):
    store = MagicMock()
    store.search.return_value = sample_candidates
    return store


@pytest.fixture
def retriever(mock_embedder, mock_vector_store):
    return RAGRetriever(
        embedder=mock_embedder,
        vector_store=mock_vector_store,
        rrf_k=60,
        default_top_k=2,
    )


class TestRAGRetrieverInitialization:
    def test_init_with_defaults(self, mock_embedder, mock_vector_store):
        r = RAGRetriever(embedder=mock_embedder, vector_store=mock_vector_store)
        assert r.rrf_k == 60
        assert r.default_top_k == 5
        assert r.dense_weight == 1.0
        assert r.sparse_weight == 1.0

    def test_init_with_custom_values(self, mock_embedder, mock_vector_store):
        r = RAGRetriever(
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            rrf_k=50,
            default_top_k=10,
            dense_weight=1.5,
            sparse_weight=0.8,
            candidate_multiplier=3,
        )
        assert r.rrf_k == 50
        assert r.default_top_k == 10
        assert r.dense_weight == 1.5
        assert r.sparse_weight == 0.8
        assert r.candidate_multiplier == 3


class TestRAGRetrieverTokenization:
    def test_tokenize_english(self):
        tokens = RAGRetriever.tokenize("Drip irrigation saves 40% water!")
        assert "drip" in tokens
        assert "irrigation" in tokens
        assert "saves" in tokens
        assert "40" in tokens
        assert "water" in tokens

    def test_tokenize_arabic(self):
        tokens = RAGRetriever.tokenize("إدارة الموارد المائية في مصر 2026")
        assert "إدارة" in tokens
        assert "الموارد" in tokens
        assert "المائية" in tokens
        assert "مصر" in tokens
        assert "2026" in tokens

    def test_tokenize_empty(self):
        assert RAGRetriever.tokenize("") == []
        assert RAGRetriever.tokenize("   ") == []


class TestRRFComputation:
    def test_compute_rrf_score_dense_and_sparse(self):
        score = RAGRetriever.compute_rrf_score(
            dense_rank=1,
            sparse_rank=1,
            rrf_k=60,
            dense_weight=1.0,
            sparse_weight=1.0,
        )
        expected = (1.0 / 61) + (1.0 / 61)
        assert abs(score - expected) < 1e-6

    def test_compute_rrf_score_dense_only(self):
        score = RAGRetriever.compute_rrf_score(
            dense_rank=2,
            sparse_rank=None,
            rrf_k=60,
        )
        expected = 1.0 / 62
        assert abs(score - expected) < 1e-6

    def test_compute_rrf_score_unranked(self):
        assert RAGRetriever.compute_rrf_score(None, None) == 0.0


class TestRAGRetrieverRetrieve:
    def test_retrieve_empty_query(self, retriever):
        assert retriever.retrieve("") == []
        assert retriever.retrieve("   ") == []

    def test_retrieve_success(self, retriever, mock_embedder, mock_vector_store):
        results = retriever.retrieve("drip irrigation water", top_k=2)

        # Check embedder called with clean query
        mock_embedder.embed_query.assert_called_once_with("drip irrigation water")

        # Check vector store search called
        mock_vector_store.search.assert_called_once()

        # Check top_k limit respected
        assert len(results) == 2

        # Check chunk 101 ranks highest due to exact match on "drip irrigation water"
        first = results[0]
        assert first["chunk_id"] == 101
        assert "irrigation" in first["content"].lower()
        assert first["metadata"]["doc_id"] == 10
        assert first["metadata"]["file_name"] == "irrigation_guide.pdf"
        assert first["dense_score"] == 0.95
        assert first["dense_rank"] == 1
        assert first["sparse_rank"] == 1
        assert first["score"] > 0.0

    def test_retrieve_arabic_query(self, retriever, mock_vector_store):
        results = retriever.retrieve("الموارد المائية في مصر", top_k=3)
        assert len(results) == 3

        # Chunk 103 should get high sparse score for Arabic keyword overlap
        chunk_103 = next((r for r in results if r["chunk_id"] == 103), None)
        assert chunk_103 is not None
        assert chunk_103["sparse_score"] > 0
        assert chunk_103["sparse_rank"] is not None

    def test_retrieve_empty_candidates(self, retriever, mock_vector_store):
        mock_vector_store.search.return_value = []
        results = retriever.retrieve("unknown query")
        assert results == []

    def test_retrieve_score_threshold(self, retriever):
        # A very high score threshold should filter out results
        results = retriever.retrieve("irrigation", top_k=5, score_threshold=1.0)
        assert results == []

    def test_retrieve_as_context(self, retriever):
        context = retriever.retrieve_as_context("irrigation", top_k=1)
        assert "irrigation_guide.pdf" in context
        assert "Water conservation" in context

    def test_as_tool_spec(self, retriever):
        spec = retriever.as_tool_spec()
        assert spec["type"] == "function"
        assert spec["function"]["name"] == "retrieve_agricultural_knowledge"
        assert "parameters" in spec["function"]
        assert "query" in spec["function"]["parameters"]["properties"]

    def test_callable_interface(self, retriever):
        results = retriever("drip irrigation", top_k=1)
        assert len(results) == 1
        assert results[0]["chunk_id"] == 101
