"""
tests/test_knowledge_node.py

Unit and integration tests for `src/agent/nodes/knowledge_node.py`.
Tests context formatting, citation generation, fallback handling, error resilience,
and integration within LangGraph and `AgentState`.
"""

from unittest.mock import MagicMock
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.agent.nodes.knowledge_node import (
    _generate_fallback_synthesis,
    format_rag_context,
    knowledge_node,
)
from src.agent.state import AgentState


# ===========================================================================
# 1. format_rag_context Tests
# ===========================================================================

def test_format_rag_context_empty():
    res = format_rag_context([])
    assert "No relevant legal or environmental documents found" in res


def test_format_rag_context_with_chunks():
    chunks = [
        {
            "chunk_id": 101,
            "content": "تنص المادة 4 من قانون البيئة على حماية المسطحات الخضراء.",
            "score": 0.8912,
            "metadata": {
                "file_name": "law_4_1994.pdf",
                "chunk_index": 2,
            },
        },
        {
            "chunk_id": 102,
            "content": "Decree 102 sets sidewalk tree setback requirements to 1.5 meters.",
            "score": 0.7654,
            "metadata": {
                "file_name": "decree_102.pdf",
                "chunk_index": 0,
            },
        },
    ]

    context_str = format_rag_context(chunks)
    assert "[Document 1: law_4_1994.pdf (Chunk 2) | Score: 0.8912]" in context_str
    assert "المادة 4" in context_str
    assert "[Document 2: decree_102.pdf (Chunk 0) | Score: 0.7654]" in context_str
    assert "1.5 meters" in context_str
    assert "---" in context_str


# ===========================================================================
# 2. _generate_fallback_synthesis Tests
# ===========================================================================

def test_generate_fallback_synthesis_empty_arabic():
    res = _generate_fallback_synthesis(query="قانون البيئة", chunks=[], language="ar")
    assert "لم يتم العثور على وثائق" in res


def test_generate_fallback_synthesis_empty_english():
    res = _generate_fallback_synthesis(query="law regulations", chunks=[], language="en")
    assert "No relevant legal or ecological documentation was found" in res


def test_generate_fallback_synthesis_with_chunks():
    chunks = [
        {
            "content": "اشتراطات التشجير تتطلب مسافة 2 متر عن المرافق العامة.",
            "metadata": {"file_name": "guidelines.pdf", "chunk_index": 1},
        }
    ]
    res_ar = _generate_fallback_synthesis(query="اشتراطات", chunks=chunks, language="ar")
    assert "guidelines.pdf (Chunk 1)" in res_ar
    assert "2 متر" in res_ar


# ===========================================================================
# 3. knowledge_node Unit Tests with Mocks
# ===========================================================================

def test_knowledge_node_success():
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        {
            "chunk_id": 1,
            "content": "يمنع قطع الأشجار المعمرة في الطرق العامة إلا بتصريح رسمي.",
            "score": 0.95,
            "metadata": {"file_name": "law_forestry.pdf", "chunk_index": 0},
        }
    ]

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(
        content="وفقاً للمادة المذكورة في [Document 1: law_forestry.pdf]، يُحظر قطع الأشجار دون تصريح."
    )

    state: AgentState = {
        "messages": [HumanMessage(content="هل يجوز قطع شجرة أمام منزلي؟")],
        "routes": ["knowledge"],
        "rag_context": None,
        "climate_metrics": None,
        "candidate_species": None,
        "language": "ar",
        "final_response": None,
    }

    result = knowledge_node(state, retriever=mock_retriever, llm=mock_llm)

    assert "rag_context" in result
    assert "law_forestry.pdf" in result["rag_context"]
    assert "messages" in result
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)
    assert "يُحظر قطع الأشجار" in result["messages"][0].content
    mock_retriever.retrieve.assert_called_once()
    mock_llm.invoke.assert_called_once()


def test_knowledge_node_empty_messages():
    state: AgentState = {
        "messages": [],
        "routes": ["knowledge"],
        "rag_context": None,
        "climate_metrics": None,
        "candidate_species": None,
        "language": "ar",
        "final_response": None,
    }

    result = knowledge_node(state)
    assert "rag_context" in result
    assert "messages" in result
    assert len(result["messages"]) == 1
    assert "لم يتم استلام استفسار واضح" in result["messages"][0].content


def test_knowledge_node_retriever_failure_graceful_handling():
    mock_retriever = MagicMock()
    mock_retriever.retrieve.side_effect = RuntimeError("Qdrant connection refused")

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(
        content="لم تتوفر وثائق كافية للرد على استفسارك."
    )

    state: AgentState = {
        "messages": [HumanMessage(content="ما هي القوانين؟")],
        "routes": ["knowledge"],
        "rag_context": None,
        "climate_metrics": None,
        "candidate_species": None,
        "language": "ar",
        "final_response": None,
    }

    result = knowledge_node(state, retriever=mock_retriever, llm=mock_llm)
    assert "rag_context" in result
    assert "No relevant legal or environmental documents found" in result["rag_context"]
    assert len(result["messages"]) == 1


def test_knowledge_node_llm_failure_uses_fallback_synthesis():
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        {
            "chunk_id": 1,
            "content": "تنص اللائحة على ضرورة الحفاظ على الأشجار.",
            "score": 0.88,
            "metadata": {"file_name": "regulations.pdf", "chunk_index": 1},
        }
    ]

    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = RuntimeError("OpenAI rate limit or network error")

    state: AgentState = {
        "messages": [HumanMessage(content="ما هي اللائحة؟")],
        "routes": ["knowledge"],
        "rag_context": None,
        "climate_metrics": None,
        "candidate_species": None,
        "language": "ar",
        "final_response": None,
    }

    result = knowledge_node(state, retriever=mock_retriever, llm=mock_llm)
    assert "rag_context" in result
    assert "regulations.pdf" in result["rag_context"]
    assert len(result["messages"]) == 1
    assert "regulations.pdf" in result["messages"][0].content


def test_knowledge_node_with_dict_messages():
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="Info retrieved.")

    state = {
        "messages": [{"role": "user", "content": "What is law 4?"}],
        "routes": ["knowledge"],
        "rag_context": None,
        "climate_metrics": None,
        "candidate_species": None,
        "language": "en",
        "final_response": None,
    }

    result = knowledge_node(state, retriever=mock_retriever, llm=mock_llm)
    assert "rag_context" in result
    assert len(result["messages"]) == 1


# ===========================================================================
# 4. LangGraph StateGraph Integration
# ===========================================================================

def test_knowledge_node_in_langgraph_pipeline():
    from langgraph.graph import StateGraph, START, END

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        {
            "chunk_id": 1,
            "content": "Trees on public sidewalks must leave 1.2m clear width.",
            "score": 0.92,
            "metadata": {"file_name": "sidewalk_code.pdf", "chunk_index": 0},
        }
    ]

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(
        content="According to [Document 1: sidewalk_code.pdf], 1.2m clear width is required."
    )

    def node_wrapper(state: AgentState) -> dict:
        return knowledge_node(state, retriever=mock_retriever, llm=mock_llm)

    workflow = StateGraph(AgentState)
    workflow.add_node("knowledge", node_wrapper)
    workflow.add_edge(START, "knowledge")
    workflow.add_edge("knowledge", END)
    graph = workflow.compile()

    initial_state: AgentState = {
        "messages": [HumanMessage(content="What are sidewalk clearance rules?")],
        "routes": ["knowledge"],
        "rag_context": None,
        "climate_metrics": None,
        "candidate_species": None,
        "language": "en",
        "final_response": None,
    }

    output = graph.invoke(initial_state)

    # State update checks:
    # 1. messages length should now be 2 (initial HumanMessage + new AIMessage appended via add_messages)
    assert len(output["messages"]) == 2
    assert isinstance(output["messages"][-1], AIMessage)
    assert "sidewalk_code.pdf" in output["messages"][-1].content
    # 2. rag_context updated with formatted string
    assert output["rag_context"] is not None
    assert "sidewalk_code.pdf" in output["rag_context"]
