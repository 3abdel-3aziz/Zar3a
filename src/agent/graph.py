"""
src/agent/graph.py

Main LangGraph workflow compilation for the Zar3a Multi-Agent Platform.

Ties together:
  - AgentState: Shared TypedDict conversation state.
  - Router Node (`router`): Evaluates user query and determines execution path(s).
  - Knowledge Node (`knowledge`): Queries Qdrant RAG store for environmental laws/regulations.
  - Climate Node (`climate_recommendation`): ML microclimate prediction and tree filtering.
  - Synthesizer Node (`synthesizer`): Merges legal citations and quantitative recommendations.
  - Multi-turn state persistence via `MemorySaver` checkpointer.

Graph Flow:
    START
      │
      ▼
   [router]
      │
      ├───────────────────────────────┐
      ▼ (if "knowledge")              ▼ (if "climate_recommendation")
 [knowledge]              [climate_recommendation]
      │                               │
      └───────────────┬───────────────┘
                      ▼
                [synthesizer]
                      │
                      ▼
                     END
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agent.nodes.climate_rec_node import climate_recommendation_node
from src.agent.nodes.knowledge_node import knowledge_node
from src.agent.nodes.synthesizer_node import synthesizer_node
from src.agent.router import FALLBACK_ROUTES, VALID_ROUTES, route_query
from src.agent.state import AgentState

# ---------------------------------------------------------------------------
# Logger Configuration (Strict INFO level, no DEBUG)
# ---------------------------------------------------------------------------
logger = logging.getLogger("Zar3a.AgentGraph")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Conditional Edge Router Function
# ---------------------------------------------------------------------------

def route_destinations(state: AgentState) -> Sequence[str]:
    """
    Evaluates the chosen `routes` in `AgentState` and returns destination node(s).

    Supports dynamic parallel fan-out:
      - Returns `["knowledge"]` for regulatory-only queries.
      - Returns `["climate_recommendation"]` for tree/climate-only queries.
      - Returns `["knowledge", "climate_recommendation"]` for compound or ambiguous
        queries, triggering concurrent parallel execution of both pipelines.

    Args:
        state: Current `AgentState` containing the `routes` list.

    Returns:
        Sequence[str]: List of downstream node names to execute.
    """
    selected_routes = state.get("routes", [])
    destinations: List[str] = []

    for route in selected_routes:
        if route in VALID_ROUTES and route not in destinations:
            destinations.append(route)

    # Safe fallback if empty, missing, or unrecognized
    if not destinations:
        logger.warning(
            "Graph conditional edge found no valid routes in state (%s). Defaulting to parallel execution of both.",
            selected_routes,
        )
        destinations = list(FALLBACK_ROUTES)

    logger.info("Graph conditional edge fanning out to: %s", destinations)
    return destinations


# ---------------------------------------------------------------------------
# Graph Builder & Compilation
# ---------------------------------------------------------------------------

def create_agent_graph(
    checkpointer: Optional[BaseCheckpointSaver] = None,
) -> CompiledStateGraph:
    """
    Constructs and compiles the Zar3a LangGraph multi-agent workflow.

    Args:
        checkpointer: Optional LangGraph checkpoint saver (defaults to in-memory `MemorySaver`).
                      Pass an external checkpointer (e.g. PostgresSaver) for persistent multi-turn storage.

    Returns:
        CompiledStateGraph: Fully compiled, executable LangGraph instance.
    """
    logger.info("Constructing Zar3a StateGraph...")

    # 1. Initialize StateGraph with the shared AgentState
    workflow = StateGraph(AgentState)

    # 2. Register all specialist nodes
    workflow.add_node("router", route_query)
    workflow.add_node("knowledge", knowledge_node)
    workflow.add_node("climate_recommendation", climate_recommendation_node)
    workflow.add_node("synthesizer", synthesizer_node)

    # 3. Configure Entry Point Edge
    workflow.add_edge(START, "router")

    # 4. Configure Conditional Fan-out from Router
    workflow.add_conditional_edges(
        "router",
        route_destinations,
        ["knowledge", "climate_recommendation"],
    )

    # 5. Converge specialist pipelines into the Master Synthesizer
    workflow.add_edge("knowledge", "synthesizer")
    workflow.add_edge("climate_recommendation", "synthesizer")

    # 6. End of workflow
    workflow.add_edge("synthesizer", END)

    # 7. Attach checkpointer for multi-turn conversational persistence
    active_checkpointer = checkpointer if checkpointer is not None else MemorySaver()

    compiled = workflow.compile(checkpointer=active_checkpointer)
    logger.info("Zar3a StateGraph compiled successfully with checkpointer: %s", type(active_checkpointer).__name__)
    return compiled


# ---------------------------------------------------------------------------
# Module-level Compiled Instance (Default Production Export)
# ---------------------------------------------------------------------------
compiled_graph: CompiledStateGraph = create_agent_graph()
