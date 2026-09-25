"""
tests/test_router.py

Unit and integration tests for the supervisor router node in `src/agent/router.py`.
Tests schema validation, routing decisions, fallback behavior, error handling,
and compatibility with LangGraph and `AgentState`.
"""

from unittest.mock import MagicMock
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import ValidationError

from src.agent.router import (
    FALLBACK_ROUTES,
    VALID_ROUTES,
    RouterOutput,
    route_query,
    set_router_chain,
)
from src.agent.state import AgentState


# ===========================================================================
# 1. Pydantic RouterOutput Schema Tests
# ===========================================================================

def test_router_output_valid_single_knowledge():
    output = RouterOutput(routes=["knowledge"])
    assert output.routes == ["knowledge"]


def test_router_output_valid_single_climate():
    output = RouterOutput(routes=["climate_recommendation"])
    assert output.routes == ["climate_recommendation"]


def test_router_output_valid_both_routes():
    output = RouterOutput(routes=["knowledge", "climate_recommendation"])
    assert "knowledge" in output.routes
    assert "climate_recommendation" in output.routes


def test_router_output_invalid_route_rejected():
    with pytest.raises(ValidationError):
        RouterOutput(routes=["invalid_route"])


def test_router_output_json_schema():
    schema = RouterOutput.model_json_schema()
    assert "routes" in schema["properties"]
    assert schema["properties"]["routes"]["type"] == "array"
    items = schema["properties"]["routes"]["items"]
    assert set(items["enum"]) == {"knowledge", "climate_recommendation"}


# ===========================================================================
# 2. route_query Unit Tests with Mock LLM
# ===========================================================================

def test_route_query_knowledge_branch():
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = RouterOutput(routes=["knowledge"])

    state: AgentState = {
        "messages": [HumanMessage(content="ما هي قوانين البيئة والتشجير في مصر؟")],
        "routes": [],
        "rag_context": None,
        "climate_metrics": None,
        "candidate_species": None,
        "language": "ar",
        "final_response": None,
    }

    result = route_query(state, llm_chain=mock_chain)
    assert result == {"routes": ["knowledge"]}
    assert mock_chain.invoke.called


def test_route_query_climate_recommendation_branch():
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = RouterOutput(routes=["climate_recommendation"])

    state: AgentState = {
        "messages": [HumanMessage(content="نريد تقليل درجة الحرارة وترشيح أشجار تتحمل الجفاف")],
        "routes": [],
        "rag_context": None,
        "climate_metrics": None,
        "candidate_species": None,
        "language": "ar",
        "final_response": None,
    }

    result = route_query(state, llm_chain=mock_chain)
    assert result == {"routes": ["climate_recommendation"]}


def test_route_query_compound_both_routes():
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = RouterOutput(routes=["knowledge", "climate_recommendation"])

    state: AgentState = {
        "messages": [
            HumanMessage(
                content="ما هي الاشتراطات القانونية لزراعة الأشجار في الشوارع وما هي أفضل أشجار للتظليل؟"
            )
        ],
        "routes": [],
        "rag_context": None,
        "climate_metrics": None,
        "candidate_species": None,
        "language": "ar",
        "final_response": None,
    }

    result = route_query(state, llm_chain=mock_chain)
    assert result["routes"] == ["knowledge", "climate_recommendation"]


def test_route_query_empty_messages_fallback():
    state: AgentState = {
        "messages": [],
        "routes": [],
        "rag_context": None,
        "climate_metrics": None,
        "candidate_species": None,
        "language": "ar",
        "final_response": None,
    }

    result = route_query(state)
    assert result == {"routes": FALLBACK_ROUTES}


def test_route_query_empty_query_text_fallback():
    state: AgentState = {
        "messages": [HumanMessage(content="   ")],
        "routes": [],
        "rag_context": None,
        "climate_metrics": None,
        "candidate_species": None,
        "language": "ar",
        "final_response": None,
    }

    result = route_query(state)
    assert result == {"routes": FALLBACK_ROUTES}


def test_route_query_llm_exception_fallback():
    mock_chain = MagicMock()
    mock_chain.invoke.side_effect = RuntimeError("OpenAI API rate limit or network timeout")

    state: AgentState = {
        "messages": [HumanMessage(content="ما هي الأشجار المقترحة؟")],
        "routes": [],
        "rag_context": None,
        "climate_metrics": None,
        "candidate_species": None,
        "language": "ar",
        "final_response": None,
    }

    result = route_query(state, llm_chain=mock_chain)
    assert result == {"routes": FALLBACK_ROUTES}


def test_route_query_empty_llm_routes_fallback():
    mock_chain = MagicMock()
    # Mock returning object with empty routes list
    mock_result = MagicMock()
    mock_result.routes = []
    mock_chain.invoke.return_value = mock_result

    state: AgentState = {
        "messages": [HumanMessage(content="سؤال غامض")],
        "routes": [],
        "rag_context": None,
        "climate_metrics": None,
        "candidate_species": None,
        "language": "ar",
        "final_response": None,
    }

    result = route_query(state, llm_chain=mock_chain)
    assert result == {"routes": FALLBACK_ROUTES}


def test_route_query_with_dict_messages():
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = RouterOutput(routes=["knowledge"])

    state = {
        "messages": [
            {"role": "user", "content": "ما هي القوانين المتعلقة بقطع الأشجار؟"}
        ],
        "routes": [],
        "rag_context": None,
        "climate_metrics": None,
        "candidate_species": None,
        "language": "ar",
        "final_response": None,
    }

    result = route_query(state, llm_chain=mock_chain)
    assert result == {"routes": ["knowledge"]}


# ===========================================================================
# 3. LangGraph StateGraph Integration Test
# ===========================================================================

def test_route_query_inside_langgraph():
    from langgraph.graph import StateGraph, START, END

    mock_chain = MagicMock()
    mock_chain.invoke.return_value = RouterOutput(routes=["climate_recommendation"])

    def router_step(state: AgentState) -> dict:
        return route_query(state, llm_chain=mock_chain)

    workflow = StateGraph(AgentState)
    workflow.add_node("router", router_step)
    workflow.add_edge(START, "router")
    workflow.add_edge("router", END)
    app = workflow.compile()

    initial_state: AgentState = {
        "messages": [HumanMessage(content="Recommend shade trees for Giza")],
        "routes": [],
        "rag_context": None,
        "climate_metrics": None,
        "candidate_species": None,
        "language": "en",
        "final_response": None,
    }

    final_state = app.invoke(initial_state)
    assert final_state["routes"] == ["climate_recommendation"]
    assert len(final_state["messages"]) == 1
