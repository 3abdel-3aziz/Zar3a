"""
tests/test_synthesizer_node.py

Unit and integration tests for `src/agent/nodes/synthesizer_node.py`.
Tests aggregation of RAG legal context and ML tree recommendations,
fallback formatting, UTF-8 Arabic support, and LangGraph integration.
"""

from unittest.mock import MagicMock
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.agent.nodes.synthesizer_node import (
    _generate_fallback_synthesis,
    synthesizer_node,
)
from src.agent.state import AgentState


# ===========================================================================
# 1. Fallback Synthesis Tests
# ===========================================================================

def test_fallback_synthesis_both_domains_arabic():
    rag_ctx = "[Document 1: law_4_1994.pdf] تنص المادة على حماية المسطحات الخضراء."
    metrics = {"predicted_temp_delta_c": -2.3, "ndvi_improvement": 0.15}
    candidates = [
        {
            "name_ar": "نيم",
            "name_en": "Neem",
            "category": "ظل",
            "water_requirement": "Low",
            "cooling_effect_score": 8,
            "final_score": 19.8,
        }
    ]

    res = _generate_fallback_synthesis(
        query="ما هي اشتراطات وأفضل أشجار التظليل؟",
        rag_context=rag_ctx,
        climate_metrics=metrics,
        candidate_species=candidates,
        language="ar",
    )

    assert "الضوابط والاشتراطات القانونية" in res
    assert "law_4_1994.pdf" in res
    assert "2.3 درجة مئوية" in res
    assert "نيم" in res
    assert "Neem" in res
    assert "إرشادات الزراعة" in res


def test_fallback_synthesis_english():
    rag_ctx = "[Document 1: green_code.pdf] Public street setback must be 1.5m."
    metrics = {"predicted_temp_delta_c": -1.8, "ndvi_improvement": 0.12}
    candidates = [
        {
            "name_ar": "صنوبر حلبي",
            "name_en": "Aleppo Pine",
            "category": "Shade",
            "water_requirement": "Low",
            "cooling_effect_score": 7,
            "final_score": 18.5,
        }
    ]

    res = _generate_fallback_synthesis(
        query="Recommend legal shade trees",
        rag_context=rag_ctx,
        climate_metrics=metrics,
        candidate_species=candidates,
        language="en",
    )

    assert "Legal & Regulatory Guidelines" in res
    assert "green_code.pdf" in res
    assert "-1.8°C" in res
    assert "Aleppo Pine" in res


# ===========================================================================
# 2. synthesizer_node Unit Tests with Mocks
# ===========================================================================

def test_synthesizer_node_with_mock_llm():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(
        content="## تقرير متكامل\nبناءً على قانون البيئة، يُوصى بزراعة شجرة النيم لخفض الحرارة بمقدار 2.5 درجة."
    )

    state: AgentState = {
        "messages": [HumanMessage(content="أريد ترشيح شجرة قانونية لشارع سكني")],
        "routes": ["knowledge", "climate_recommendation"],
        "rag_context": "[Doc 1: law.pdf] أحكام التشجير",
        "climate_metrics": {"predicted_temp_delta_c": -2.5},
        "candidate_species": [{"name_ar": "نيم", "name_en": "Neem"}],
        "language": "ar",
        "final_response": None,
    }

    result = synthesizer_node(state, llm=mock_llm)

    assert "final_response" in result
    assert "messages" in result
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)
    assert result["final_response"] == result["messages"][0].content
    assert "تقرير متكامل" in result["final_response"]
    mock_llm.invoke.assert_called_once()


def test_synthesizer_node_llm_failure_uses_fallback():
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = RuntimeError("OpenAI API unavailable")

    state: AgentState = {
        "messages": [HumanMessage(content="أريد معرفة الأشجار القانونية")],
        "routes": ["knowledge", "climate_recommendation"],
        "rag_context": "[Doc 1: law.pdf] مادة 10",
        "climate_metrics": {"predicted_temp_delta_c": -1.9},
        "candidate_species": [{"name_ar": "كاسيا", "name_en": "Cassia"}],
        "language": "ar",
        "final_response": None,
    }

    result = synthesizer_node(state, llm=mock_llm)

    assert "final_response" in result
    assert "كاسيا" in result["final_response"]
    assert "مادة 10" in result["final_response"]
    assert len(result["messages"]) == 1


# ===========================================================================
# 3. LangGraph StateGraph Integration
# ===========================================================================

def test_synthesizer_node_in_langgraph_pipeline():
    from langgraph.graph import StateGraph, START, END

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(
        content="## ملخص نهائي\nالتوصية المتكاملة معتمدة وجاهزة للتطبيق."
    )

    def node_wrapper(state: AgentState) -> dict:
        return synthesizer_node(state, llm=mock_llm)

    workflow = StateGraph(AgentState)
    workflow.add_node("synthesizer", node_wrapper)
    workflow.add_edge(START, "synthesizer")
    workflow.add_edge("synthesizer", END)
    graph = workflow.compile()

    initial_state: AgentState = {
        "messages": [HumanMessage(content="ما هي التوصية النهائية؟")],
        "routes": ["knowledge", "climate_recommendation"],
        "rag_context": "اشتراطات قانونية موثقة",
        "climate_metrics": {"predicted_temp_delta_c": -2.0},
        "candidate_species": [{"name_ar": "نيم", "name_en": "Neem"}],
        "language": "ar",
        "final_response": None,
    }

    output = graph.invoke(initial_state)

    # 1. messages length updated (initial + synthesized AIMessage)
    assert len(output["messages"]) == 2
    assert isinstance(output["messages"][-1], AIMessage)

    # 2. final_response populated
    assert output["final_response"] is not None
    assert "ملخص نهائي" in output["final_response"]
