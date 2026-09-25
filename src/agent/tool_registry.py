"""
src/agent/tool_registry.py

Centralized Tool Registry for the Zar3a Multi-Agent LangGraph Workflow.

Provides direct, decoupled access to the underlying production tools,
ML models, and retrieval systems without duplicate or mocked logic:
  1. Tree Recommendation Tool & Input Schema:
     - `recommend_trees_tool` (`src.recommendation_system.tool`)
     - `TreeFilterInput` (`src.recommendation_system.tool`)
  2. Climate ML Predictor & Feature Schema:
     - `predict_urban_temperature` (`src.climate_ml.src.predictor`)
     - `ClimatePredictionInput` (`src.climate_ml.src.predictor`)
  3. Knowledge / Legal RAG Vector Retriever:
     - `RAGRetriever` (`src.rag_database.retriever`)
"""

from __future__ import annotations

from src.climate_ml.src.predictor import (
    ClimatePredictionInput,
    predict_urban_temperature,
)
from src.rag_database.retriever import RAGRetriever
from src.recommendation_system.tool import (
    TreeFilterInput,
    get_recommend_trees_tool_spec,
    recommend_trees_tool,
)

__all__ = [
    # Tree Recommendation System
    "recommend_trees_tool",
    "TreeFilterInput",
    "get_recommend_trees_tool_spec",
    # Microclimate ML Prediction
    "predict_urban_temperature",
    "ClimatePredictionInput",
    # Knowledge / Legal RAG Vector Retriever
    "RAGRetriever",
]
