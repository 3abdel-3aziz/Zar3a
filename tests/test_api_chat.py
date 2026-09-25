"""
tests/test_api_chat.py

Tests for the /api/v1/chat LangGraph multi-agent endpoints using TestClient.
"""

from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
import pytest

from src.api.dependencies import get_agent_graph
from src.api.main import app

client = TestClient(app)


def test_send_chat_message_success_with_mock_graph():
    """Verifies sending a chat message to the LangGraph workflow and receiving structured synthesis."""
    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {
        "final_response": "أفضل الأشجار لشارع ضيق في القاهرة هي النيم والكاسيا نودوزا لتوفير الظل وجذورها الآمنة.",
        "routes": ["knowledge", "climate_recommendation"],
        "rag_context": "[Document 1: law.pdf] ضوابط الأرصفة",
        "climate_metrics": {"predicted_temp_delta_c": -2.1, "ndvi_improvement": 0.15},
        "candidate_species": [
            {"name_ar": "نيم", "name_en": "Neem", "cooling_effect_score": 8}
        ],
        "language": "ar",
    }

    app.dependency_overrides[get_agent_graph] = lambda: mock_graph

    payload = {
        "message": "أريد زراعة شجرة لشارع ضيق في القاهرة",
        "thread_id": "test-session-42",
        "language": "ar",
    }
    response = client.post("/api/v1/chat/messages", json=payload)

    app.dependency_overrides.clear()

    assert response.status_code == 200
    data = response.json()
    assert data["thread_id"] == "test-session-42"
    assert "النيم" in data["response"]
    assert data["routes"] == ["knowledge", "climate_recommendation"]
    assert data["climate_metrics"]["predicted_temp_delta_c"] == -2.1
    assert len(data["candidate_species"]) == 1
    assert data["language"] == "ar"


def test_send_chat_message_generates_thread_id_if_omitted():
    """Verifies that thread_id is auto-generated if omitted from request payload."""
    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {
        "final_response": "رد افتراضي",
        "routes": ["knowledge"],
        "language": "ar",
    }

    app.dependency_overrides[get_agent_graph] = lambda: mock_graph

    response = client.post("/api/v1/chat/messages", json={"message": "سؤال عام"})

    app.dependency_overrides.clear()

    assert response.status_code == 200
    data = response.json()
    assert data["thread_id"] is not None
    assert len(data["thread_id"]) > 0


def test_send_chat_message_validation_error():
    """Verifies that an empty message string fails validation with 422."""
    response = client.post("/api/v1/chat/messages", json={"message": ""})
    assert response.status_code == 422
