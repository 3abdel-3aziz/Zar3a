"""
src/agent/state.py

Shared state definition for the Zar3a multi-agent LangGraph workflow.

This TypedDict is the single source of truth that flows between all nodes
in the graph — the router, RAG retrieval node, climate recommendation node,
and the final response synthesizer node.

Field reference:
  messages           — Full conversation history (auto-merged via add_messages reducer).
  routes             — Router-selected destination(s): "knowledge", "climate_recommendation", or both.
  rag_context        — Retrieved text passages from the Qdrant RAG vector store.
  climate_metrics    — Structured outputs from the climate ML model (temperature impact, NDVI, etc.).
  candidate_species  — Ranked tree recommendations from the TreeRecommendationEngine.
  language           — Detected or declared interface language ('ar' | 'en').
  final_response     — The synthesized, user-facing answer produced by the response node.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

from src.agent.config import DEFAULT_LANGUAGE


class AgentState(TypedDict):
    """
    Shared state container for the Zar3a LangGraph multi-agent workflow.

    All nodes read from and write updates to this dictionary.
    LangGraph merges partial updates automatically — fields not included
    in a node's return value remain unchanged in the graph's state.

    Fields
    ------
    messages : Annotated[list, add_messages]
        The full conversation history between the user and the agent system.
        Uses the ``add_messages`` reducer so each node can append new messages
        (HumanMessage, AIMessage, ToolMessage, etc.) without overwriting earlier turns.

    routes : List[str]
        The routing decision(s) produced by the router node.
        Valid values: ``"knowledge"``, ``"climate_recommendation"``, or both.
        Consumed by the conditional edge logic that fans the graph out to
        the appropriate specialist node(s).

    rag_context : Optional[str]
        Retrieved text from the Qdrant vector store, formatted and ready
        to inject into an LLM prompt as grounding context.
        Populated by the RAG retrieval node; ``None`` when the route
        does not include ``"knowledge"``.

    climate_metrics : Optional[Dict[str, Any]]
        Structured output from the climate ML model, e.g.::

            {
                "predicted_temp_delta_c": -2.3,
                "ndvi_improvement": 0.12,
                "cooling_confidence": 0.87,
            }

        Populated by the climate node; ``None`` when not triggered.

    candidate_species : Optional[List[Dict[str, Any]]]
        Ordered list of recommended tree species returned by the
        ``TreeRecommendationEngine``, e.g.::

            [
                {
                    "tree_id": 7,
                    "name_ar": "نيم",
                    "name_en": "Neem",
                    "category": "Shade",
                    "water_requirement": "Medium",
                    "cooling_effect_score": 8,
                    "final_score": 19.08,
                },
                ...
            ]

        Populated by the climate recommendation node; ``None`` when not triggered.

    language : str
        The active interface language for the conversation.
        Accepted values: ``'ar'`` (Arabic, default) or ``'en'`` (English).
        Used by the response synthesizer to format the final answer in the
        correct language and script direction.

    final_response : Optional[str]
        The synthesized, user-facing answer assembled by the final response node.
        This is the string returned to the front-end / API layer once the
        graph reaches the ``END`` node.
        ``None`` until the synthesis step completes.
    """

    # --- Conversation history (LangGraph reducer merges new messages) ---
    messages: Annotated[List[BaseMessage], add_messages]

    # --- Routing ---
    routes: List[str]

    # --- RAG Retrieval ---
    rag_context: Optional[str]

    # --- Climate ML Model Outputs ---
    climate_metrics: Optional[Dict[str, Any]]

    # --- Tree Recommendation Engine Outputs ---
    candidate_species: Optional[List[Dict[str, Any]]]

    # --- Language Control ---
    language: str

    # --- Final Synthesized Output ---
    final_response: Optional[str]


def create_initial_state(
    messages: Optional[List[BaseMessage]] = None,
    language: str = DEFAULT_LANGUAGE,
    routes: Optional[List[str]] = None,
    rag_context: Optional[str] = None,
    climate_metrics: Optional[Dict[str, Any]] = None,
    candidate_species: Optional[List[Dict[str, Any]]] = None,
    final_response: Optional[str] = None,
) -> AgentState:
    """
    Convenience factory to create a properly initialized AgentState.
    
    Guarantees that `language` defaults to Arabic ('ar') as defined in `config.py`.
    """
    return {
        "messages": messages or [],
        "routes": routes or [],
        "rag_context": rag_context,
        "climate_metrics": climate_metrics,
        "candidate_species": candidate_species,
        "language": language,
        "final_response": final_response,
    }
