"""
tests/test_api_knowledge.py

Tests for the /api/v1/knowledge RAG vector search endpoints using TestClient.
"""

from unittest.mock import MagicMock

from fastapi.testclient import TestClient
import pytest

from src.api.dependencies import get_rag_retriever
from src.api.main import app

client = TestClient(app)


def test_search_documents_post_success():
    """Verifies document search endpoint returns formatted passages with scores."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        {
            "chunk_id": "chunk-101",
            "content": "يحظر قطع الأشجار في الطرق العامة دون ترخيص من الجهة الإدارية المختصة.",
            "score": 0.8845,
            "metadata": {"file_name": "law_4_1994.pdf", "article": "37"},
        }
    ]

    app.dependency_overrides[get_rag_retriever] = lambda: mock_retriever

    payload = {"query": "قوانين قطع الأشجار", "top_k": 2}
    response = client.post("/api/v1/knowledge/documents/search", json=payload)

    app.dependency_overrides.clear()

    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "قوانين قطع الأشجار"
    assert data["total_results"] == 1
    assert len(data["results"]) == 1
    item = data["results"][0]
    assert item["chunk_id"] == "chunk-101"
    assert "يحظر قطع" in item["content"]
    assert item["metadata"]["article"] == "37"


def test_query_documents_get_success():
    """Verifies HTTP GET query interface for document search."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        {
            "chunk_id": "chunk-202",
            "content": "معايير مسافات زراعة الأشجار من كابلات الكهرباء والمياه.",
            "score": 0.7950,
            "metadata": {"source": "guidelines.pdf"},
        }
    ]

    app.dependency_overrides[get_rag_retriever] = lambda: mock_retriever

    response = client.get("/api/v1/knowledge/documents", params={"q": "مسافات المرافق", "top_k": 3})

    app.dependency_overrides.clear()

    assert response.status_code == 200
    data = response.json()
    assert data["total_results"] == 1
    assert data["results"][0]["chunk_id"] == "chunk-202"


def test_search_documents_validation_errors():
    """Verifies that an empty query fails validation."""
    response = client.post("/api/v1/knowledge/documents/search", json={"query": ""})
    assert response.status_code == 422

    response = client.get("/api/v1/knowledge/documents", params={"q": ""})
    assert response.status_code == 422
