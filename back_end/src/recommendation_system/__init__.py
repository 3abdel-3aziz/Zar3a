"""
src/recommendation_system

Urban tree recommendation package for the Zar3a project.
Provides the recommendation engine and agentic tool wrapper.
"""

from src.recommendation_system.engine import TreeRecommendationEngine
from src.recommendation_system.tool import (
    TreeFilterInput,
    recommend_trees_tool,
    get_recommend_trees_tool_spec,
    engine,
)

__all__ = [
    "TreeRecommendationEngine",
    "TreeFilterInput",
    "recommend_trees_tool",
    "get_recommend_trees_tool_spec",
    "engine",
]
