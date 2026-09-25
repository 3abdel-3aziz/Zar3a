"""
tests/test_agent_graph.py

Comprehensive unit and integration tests for the Zar3a LangGraph workflow (`src/agent/graph.py`).
Tests graph structure, conditional edge routing, parallel execution, fan-in convergence,
multi-turn persistence via MemorySaver, and end-to-end execution.
"""

from unittest.mock import MagicMock
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.agent.graph import (
    compiled_graph,
    create_agent_graph,
    route_destinations,
)
from src.agent.state import AgentState


# ===========================================================================
# 1. route_destinations Unit Tests
# ===========================================================================

def test_route_destinations_knowledge_only():
    state: AgentState = {
        "messages": [],
        "routes": ["knowledge"],
        "rag_context": None,
        "climate_metrics": None,
        "candidate_species": None,
        "language": "ar",
        "final_response": None,
    }
    assert route_destinations(state) == ["knowledge"]


def test_route_destinations_climate_only():
    state: AgentState = {
        "messages": [],
        "routes": ["climate_recommendation"],
        "rag_context": None,
        "climate_metrics": None,
        "candidate_species": None,
        "language": "ar",
        "final_response": None,
    }
    assert route_destinations(state) == ["climate_recommendation"]


def test_route_destinations_both_parallel():
    state: AgentState = {
        "messages": [],
        "routes": ["knowledge", "climate_recommendation"],
        "rag_context": None,
        "climate_metrics": None,
        "candidate_species": None,
        "language": "ar",
        "final_response": None,
    }
    destinations = route_destinations(state)
    assert len(destinations) == 2
    assert "knowledge" in destinations
    assert "climate_recommendation" in destinations


def test_route_destinations_empty_or_unknown_fallback():
    state: AgentState = {
        "messages": [],
        "routes": ["invalid_path"],
        "rag_context": None,
        "climate_metrics": None,
        "candidate_species": None,
        "language": "ar",
        "final_response": None,
    }
    destinations = route_destinations(state)
    assert set(destinations) == {"knowledge", "climate_recommendation"}


# ===========================================================================
# 2. Graph Structural & Compilation Tests
# ===========================================================================

def test_compiled_graph_nodes():
    assert compiled_graph is not None
    # Verify graph contains all 4 required operational nodes
    node_names = set(compiled_graph.get_graph().nodes.keys())
    assert "router" in node_names
    assert "knowledge" in node_names
    assert "climate_recommendation" in node_names
    assert "synthesizer" in node_names


def test_create_agent_graph_custom_checkpointer():
    from langgraph.checkpoint.memory import MemorySaver

    custom_saver = MemorySaver()
    graph = create_agent_graph(checkpointer=custom_saver)
    assert graph.checkpointer is custom_saver


# ===========================================================================
# 3. Execution Pipeline Tests (with Fast Mocked Nodes)
# ===========================================================================

def test_parallel_fanout_and_convergence_in_graph():
    """Validates that parallel routes merge cleanly into the synthesizer."""
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import StateGraph, START, END

    def mock_router(state: AgentState) -> dict:
        return {"routes": ["knowledge", "climate_recommendation"]}

    def mock_knowledge(state: AgentState) -> dict:
        return {
            "rag_context": "[Doc 1: law.pdf] نصوص قانونية",
            "messages": [AIMessage(content="تم استرجاع السند القانوني.")],
        }

    def mock_climate(state: AgentState) -> dict:
        return {
            "climate_metrics": {"predicted_temp_delta_c": -2.2},
            "candidate_species": [{"name_ar": "نيم", "name_en": "Neem"}],
            "messages": [AIMessage(content="تم ترشيح أشجار النيم.")],
        }

    def mock_synth(state: AgentState) -> dict:
        summary = (
            f"تقرير مدمج: {state.get('rag_context')} مع خفض حرارة "
            f"{state.get('climate_metrics', {}).get('predicted_temp_delta_c')}"
        )
        return {
            "final_response": summary,
            "messages": [AIMessage(content=summary)],
        }

    wf = StateGraph(AgentState)
    wf.add_node("router", mock_router)
    wf.add_node("knowledge", mock_knowledge)
    wf.add_node("climate_recommendation", mock_climate)
    wf.add_node("synthesizer", mock_synth)

    wf.add_edge(START, "router")
    wf.add_conditional_edges("router", route_destinations, ["knowledge", "climate_recommendation"])
    wf.add_edge("knowledge", "synthesizer")
    wf.add_edge("climate_recommendation", "synthesizer")
    wf.add_edge("synthesizer", END)

    memory = MemorySaver()
    app = wf.compile(checkpointer=memory)

    initial_state: AgentState = {
        "messages": [HumanMessage(content="أريد اشتراطات وترشيحات التظليل")],
        "routes": [],
        "rag_context": None,
        "climate_metrics": None,
        "candidate_species": None,
        "language": "ar",
        "final_response": None,
    }

    config = {"configurable": {"thread_id": "thread-multi-turn-1"}}
    output = app.invoke(initial_state, config=config)

    # 1. Routes updated by router
    assert set(output["routes"]) == {"knowledge", "climate_recommendation"}

    # 2. Both knowledge and climate node outputs merged into state
    assert output["rag_context"] is not None
    assert "Doc 1" in output["rag_context"]
    assert output["climate_metrics"]["predicted_temp_delta_c"] == -2.2
    assert output["candidate_species"][0]["name_ar"] == "نيم"

    # 3. Final synthesis generated
    assert output["final_response"] is not None
    assert "تقرير مدمج" in output["final_response"]
    assert len(output["messages"]) >= 3


def test_multi_turn_memory_saver_persistence():
    """Validates that MemorySaver preserves conversational state across turns."""
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import StateGraph, START, END

    turn_count = 0

    def mock_node(state: AgentState) -> dict:
        nonlocal turn_count
        turn_count += 1
        return {
            "final_response": f"Turn {turn_count} processed",
            "messages": [AIMessage(content=f"Response {turn_count}")],
        }

    wf = StateGraph(AgentState)
    wf.add_node("processor", mock_node)
    wf.add_edge(START, "processor")
    wf.add_edge("processor", END)

    app = wf.compile(checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "user-session-42"}}

    # Turn 1
    state1 = app.invoke(
        {
            "messages": [HumanMessage(content="Hello turn 1")],
            "routes": [],
            "rag_context": None,
            "climate_metrics": None,
            "candidate_species": None,
            "language": "ar",
            "final_response": None,
        },
        config=cfg,
    )
    assert len(state1["messages"]) == 2  # HumanMessage + AIMessage

    # Turn 2 using SAME thread_id
    state2 = app.invoke(
        {
            "messages": [HumanMessage(content="Followup turn 2")],
        },
        config=cfg,
    )
    # Total messages should now be 4 (Turn 1 human & ai + Turn 2 human & ai)
    assert len(state2["messages"]) == 4
    assert state2["messages"][-1].content == "Response 2"
