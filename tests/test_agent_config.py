"""
tests/test_agent_config.py

Unit tests for the clean, minimal configuration and language defaults in `src.agent.config`
and state creation in `src.agent.state`.
"""

from langchain_core.messages import HumanMessage

from src.agent.config import (
    AGENT_LLM_MODEL,
    CLIMATE_LLM_MODEL,
    CLIMATE_TEMPERATURE,
    DEFAULT_LANGUAGE,
    KNOWLEDGE_LLM_MODEL,
    KNOWLEDGE_TEMPERATURE,
    ROUTER_LLM_MODEL,
    ROUTER_TEMPERATURE,
    SYNTHESIZER_LLM_MODEL,
    SYNTHESIZER_TEMPERATURE,
    TEXT_ENCODING,
)
from src.agent.state import AgentState, create_initial_state


def test_config_default_values():
    """Verifies that all default config values are properly typed and populated for local quantized Qwen."""
    assert DEFAULT_LANGUAGE == "ar"
    assert TEXT_ENCODING.lower() == "utf-8"
    assert AGENT_LLM_MODEL in ("qwen2.5:3b", "ollama/qwen2.5:7b")
    assert ROUTER_LLM_MODEL in ("qwen2.5:3b", "ollama/qwen2.5:7b")
    assert KNOWLEDGE_LLM_MODEL in ("qwen2.5:3b", "ollama/qwen2.5:7b")
    assert CLIMATE_LLM_MODEL in ("qwen2.5:3b", "ollama/qwen2.5:7b")
    assert SYNTHESIZER_LLM_MODEL in ("qwen2.5:3b", "ollama/qwen2.5:7b")

    # Temperatures
    assert ROUTER_TEMPERATURE == 0.0
    assert KNOWLEDGE_TEMPERATURE == 0.0
    assert CLIMATE_TEMPERATURE == 0.0
    assert SYNTHESIZER_TEMPERATURE == 0.1


def test_create_initial_state_defaults():
    """Verifies that create_initial_state guarantees Arabic language by default."""
    state = create_initial_state()
    assert state["language"] == "ar"
    assert state["messages"] == []
    assert state["routes"] == []
    assert state["rag_context"] is None
    assert state["climate_metrics"] is None
    assert state["candidate_species"] is None
    assert state["final_response"] is None


def test_create_initial_state_with_custom_inputs():
    """Verifies that create_initial_state accepts custom initial values."""
    msg = HumanMessage(content="أريد زراعة شجرة في شارع ضيق بالقاهرة")
    state = create_initial_state(
        messages=[msg],
        language="ar",
        routes=["climate_recommendation"],
    )
    assert len(state["messages"]) == 1
    assert state["messages"][0].content == "أريد زراعة شجرة في شارع ضيق بالقاهرة"
    assert state["language"] == "ar"
    assert state["routes"] == ["climate_recommendation"]
