"""
src/api/dependencies.py

Shared dependencies and service providers for the Zar3a FastAPI application.
Follows Dependency Injection (DI) principles for modularity, testability, and thin handlers.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from agent.graph import compiled_graph
from climate_ml.src.predictor import predict_urban_temperature
from rag_database.retriever import RAGRetriever
from recommendation_system.engine import TreeRecommendationEngine

logger = logging.getLogger("Zar3a.API.Dependencies")

# Module singletons
_retriever_singleton: Optional[RAGRetriever] = None


def get_recommendation_engine() -> TreeRecommendationEngine:
    """
    Dependency provider for the TreeRecommendationEngine singleton instance.
    Loads and caches the master Egyptian tree species dataset.
    """
    return TreeRecommendationEngine()


def get_rag_retriever() -> RAGRetriever:
    """
    Dependency provider for the RAGRetriever instance.
    Provides hybrid dense-sparse vector search against Qdrant.
    """
    global _retriever_singleton
    if _retriever_singleton is None:
        logger.info("Initializing RAGRetriever dependency singleton...")
        _retriever_singleton = RAGRetriever()
    return _retriever_singleton


def get_climate_predictor() -> Callable[..., Any]:
    """
    Dependency provider for the microclimate urban temperature predictor.
    """
    return predict_urban_temperature


def get_agent_graph() -> Any:
    """
    Dependency provider for the compiled LangGraph workflow instance with MemorySaver checkpointer.
    """
    return compiled_graph
