"""
tests/test_climate_rec_node.py

Unit and integration tests for `src/agent/nodes/climate_rec_node.py`.
Tests constraint extraction, microclimate ML impact prediction,
tree recommendation tool execution, bilingual synthesis, and LangGraph integration.
"""

from unittest.mock import MagicMock
import os
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.agent.nodes.climate_rec_node import (
    ExtractedTreeConstraints,
    _extract_constraints_heuristic,
    _generate_fallback_synthesis,
    climate_recommendation_node,
    execute_climate_impact,
    extract_user_constraints,
)
from src.agent.state import AgentState


@pytest.fixture(autouse=True)
def fast_climate_timeout(monkeypatch):
    """Ensures unit tests do not hang on remote MLflow network calls."""
    monkeypatch.setenv("CLIMATE_ML_TIMEOUT", "0.05")


def test_heuristic_constraint_extraction_arabic_narrow_street():
    text = "أريد أشجار تظليل لشارع ضيق تستهلك مياه قليلة وتقلل الحرارة"
    constraints = _extract_constraints_heuristic(text)
    assert constraints.narrow_street is True
    assert constraints.water_requirement == "Low"
    assert constraints.min_cooling_score >= 5


def test_heuristic_constraint_extraction_english_high_water():
    text = "Recommend 5 trees for a wide park with medium water and high cooling"
    constraints = _extract_constraints_heuristic(text)
    assert constraints.narrow_street is False
    assert constraints.water_requirement == "Medium"
    assert constraints.top_n == 5
    assert constraints.min_cooling_score >= 5


def test_extract_user_constraints_with_mock_chain():
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = ExtractedTreeConstraints(
        narrow_street=True,
        water_requirement="Low",
        min_cooling_score=8,
        top_n=4,
    )
    result = extract_user_constraints("أي استفسار", extractor_chain=mock_chain)
    assert result.narrow_street is True
    assert result.min_cooling_score == 8
    assert result.top_n == 4


# ===========================================================================
# 2. Climate Impact Execution Tests
# ===========================================================================

def test_execute_climate_impact_fallback():
    constraints = ExtractedTreeConstraints(min_cooling_score=6, top_n=3)
    metrics = execute_climate_impact(constraints)
    assert "predicted_temp_delta_c" in metrics
    assert metrics["predicted_temp_delta_c"] < 0
    assert "ndvi_improvement" in metrics
    assert "cooling_confidence" in metrics


def test_execute_climate_impact_custom_predictor():
    mock_predictor = MagicMock()
    mock_predictor.return_value = {
        "target_variable": "mean_temperature",
        "predicted_value": 29.8,
    }
    constraints = ExtractedTreeConstraints(min_cooling_score=7, top_n=3)
    metrics = execute_climate_impact(constraints, custom_predictor=mock_predictor)
    assert metrics["predicted_mean_temp_c"] == 29.8
    assert metrics["status"] == "ml_model_predicted"
    mock_predictor.assert_called_once()


# ===========================================================================
# 3. Fallback Synthesis Tests
# ===========================================================================

def test_generate_fallback_synthesis_arabic():
    constraints = ExtractedTreeConstraints(narrow_street=True, water_requirement="Low", min_cooling_score=6)
    metrics = {"predicted_temp_delta_c": -2.1, "ndvi_improvement": 0.14}
    candidates = [
        {
            "name_ar": "نيم",
            "name_en": "Neem",
            "category": "ظل",
            "water_requirement": "Low",
            "cooling_effect_score": 8,
            "final_score": 19.5,
        }
    ]
    summary = _generate_fallback_synthesis(constraints, metrics, candidates, language="ar")
    assert "نيم" in summary
    assert "Neem" in summary
    assert "2.1 درجة مئوية" in summary
    assert "جذور عميقة آمنة" in summary


def test_generate_fallback_synthesis_english():
    constraints = ExtractedTreeConstraints(narrow_street=False, water_requirement="Medium", min_cooling_score=5)
    metrics = {"predicted_temp_delta_c": -1.8, "ndvi_improvement": 0.12}
    candidates = [
        {
            "name_ar": "سنط عربي",
            "name_en": "Acacia",
            "category": "Shade",
            "water_requirement": "Medium",
            "cooling_effect_score": 7,
            "final_score": 18.0,
        }
    ]
    summary = _generate_fallback_synthesis(constraints, metrics, candidates, language="en")
    assert "Acacia" in summary
    assert "1.8°C" in summary


# ===========================================================================
# 4. climate_recommendation_node Unit & Functional Tests
# ===========================================================================

def test_climate_recommendation_node_end_to_end_real_engine():
    """Runs climate_recommendation_node against the actual recommendation engine."""
    state: AgentState = {
        "messages": [
            HumanMessage(
                content="أريد ترشيح 3 أشجار لشارع ضيق تتحمل الجفاف وتوفر تظليل عالي"
            )
        ],
        "routes": ["climate_recommendation"],
        "rag_context": None,
        "climate_metrics": None,
        "candidate_species": None,
        "language": "ar",
        "final_response": None,
    }

    result = climate_recommendation_node(state)

    # 1. Check climate_metrics
    assert "climate_metrics" in result
    assert result["climate_metrics"]["predicted_temp_delta_c"] < 0

    # 2. Check candidate_species
    assert "candidate_species" in result
    candidates = result["candidate_species"]
    assert isinstance(candidates, list)
    assert len(candidates) > 0
    first_tree = candidates[0]
    assert "name_ar" in first_tree
    assert "name_en" in first_tree
    assert "cooling_effect_score" in first_tree

    # 3. Check messages output
    assert "messages" in result
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)
    assert len(result["messages"][0].content) > 50


def test_climate_recommendation_node_with_injected_mocks():
    mock_tool = MagicMock()
    mock_tool.return_value = '[{"tree_id": 5, "name_ar": "كاسيا", "name_en": "Cassia", "category": "Shade", "water_requirement": "Low", "cooling_effect_score": 8, "final_score": 20.0}]'

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="تقرير التوصيات للأشجار: كاسيا مناسبة تماماً.")

    state: AgentState = {
        "messages": [HumanMessage(content="Recommend trees for Cairo")],
        "routes": ["climate_recommendation"],
        "rag_context": None,
        "climate_metrics": None,
        "candidate_species": None,
        "language": "ar",
        "final_response": None,
    }

    result = climate_recommendation_node(
        state,
        recommendation_tool=mock_tool,
        llm=mock_llm,
    )

    assert len(result["candidate_species"]) == 1
    assert result["candidate_species"][0]["name_en"] == "Cassia"
    assert "تقرير التوصيات للأشجار" in result["messages"][0].content
    mock_tool.assert_called_once()
    mock_llm.invoke.assert_called_once()


# ===========================================================================
# 5. LangGraph StateGraph Integration
# ===========================================================================

def test_climate_recommendation_node_in_langgraph_pipeline():
    from langgraph.graph import StateGraph, START, END

    workflow = StateGraph(AgentState)
    workflow.add_node("climate_recommendation", climate_recommendation_node)
    workflow.add_edge(START, "climate_recommendation")
    workflow.add_edge("climate_recommendation", END)
    graph = workflow.compile()

    initial_state: AgentState = {
        "messages": [HumanMessage(content="Recommend shade trees with low water consumption in Giza")],
        "routes": ["climate_recommendation"],
        "rag_context": None,
        "climate_metrics": None,
        "candidate_species": None,
        "language": "en",
        "final_response": None,
    }

    output = graph.invoke(initial_state)

    # Messages appended (initial HumanMessage + new AIMessage)
    assert len(output["messages"]) == 2
    assert isinstance(output["messages"][-1], AIMessage)

    # State keys updated
    assert output["climate_metrics"] is not None
    assert output["candidate_species"] is not None
    assert len(output["candidate_species"]) > 0
