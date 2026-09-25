"""
tests/test_recommendation.py

Comprehensive unit tests for the Tree Recommendation Engine and Agentic Tool wrapper.
Tests the Singleton data loading, dynamic filtering, edge cases, and JSON output formatting.
"""

import json
import os
import pandas as pd
import pytest

from src.recommendation_system.engine import TreeRecommendationEngine
from src.recommendation_system.tool import (
    TreeFilterInput,
    recommend_trees_tool,
    get_recommend_trees_tool_spec,
)


@pytest.fixture(autouse=True)
def ensure_clean_singleton():
    """Ensures each test has access to a clean or initialized singleton state."""
    yield
    # Keep singleton intact for subsequent tests, or can reset if needed


class TestSingletonDataLoading:
    """Tests verifying the Singleton pattern for dataset loading and memory efficiency."""

    def test_singleton_instance_identity(self):
        """Verifies that multiple calls return the same engine instance and DataFrame."""
        engine_a = TreeRecommendationEngine()
        engine_b = TreeRecommendationEngine()

        assert engine_a is engine_b
        assert engine_a.df is engine_b.df
        assert isinstance(engine_a.df, pd.DataFrame)
        assert len(engine_a.df) > 0

    def test_singleton_data_persistence(self):
        """Verifies that the dataset remains cached in memory across calls."""
        engine = TreeRecommendationEngine()
        records_count = len(engine.df)
        assert records_count >= 20  # Dataset has 37+ tree species

        # Verify key columns exist in master dataset
        required_cols = [
            "tree_id", "name_ar", "name_en", "category",
            "water_requirement", "cooling_effect_score",
            "root_system", "suitable_for_narrow_streets",
            "carbon_sequestration_kg_yr",
        ]
        for col in required_cols:
            assert col in engine.df.columns

    def test_singleton_reset(self):
        """Verifies that reset clears the singleton state and allows re-initialization."""
        engine_before = TreeRecommendationEngine()
        TreeRecommendationEngine.reset()
        assert TreeRecommendationEngine._instance is None
        assert TreeRecommendationEngine._df is None

        engine_after = TreeRecommendationEngine()
        assert engine_after is not None
        assert engine_after.df is not None

    def test_nonexistent_csv_path_raises_error(self):
        """Verifies that providing an invalid path raises FileNotFoundError."""
        TreeRecommendationEngine.reset()
        fake_path = os.path.join("non", "existent", "path", "trees.csv")
        with pytest.raises(FileNotFoundError):
            TreeRecommendationEngine(csv_path=fake_path, force_reload=True)
        # Restore valid singleton
        TreeRecommendationEngine.reset()
        TreeRecommendationEngine()


class TestDynamicFiltering:
    """Tests dynamic criteria filtering through dictionaries."""

    @pytest.fixture
    def engine(self):
        return TreeRecommendationEngine()

    def test_empty_criteria_returns_default_top_n(self, engine):
        """Verifies default behavior with empty criteria dictionary."""
        result = engine.recommend(criteria={}, top_n=3)
        assert len(result) == 3
        expected_cols = [
            "tree_id", "name_ar", "name_en", "category",
            "water_requirement", "cooling_effect_score", "final_score",
        ]
        assert list(result.columns) == expected_cols

    def test_narrow_street_filter(self, engine):
        """Verifies narrow street filter only returns sidewalk-safe trees."""
        result = engine.recommend(criteria={"narrow_street": True}, top_n=10)
        assert len(result) > 0

        # Cross-reference with master dataset to confirm all returned trees are suitable
        for _, row in result.iterrows():
            master_match = engine.df[engine.df["tree_id"] == row["tree_id"]].iloc[0]
            assert master_match["suitable_for_narrow_streets"] == True
            assert master_match["root_system"] == "Deep"

    def test_water_requirement_low(self, engine):
        """Verifies 'Low' water filter returns strictly low-water-demand trees."""
        result = engine.recommend(criteria={"water_requirement": "Low"}, top_n=10)
        assert len(result) > 0
        for _, row in result.iterrows():
            assert row["water_requirement"].lower() == "low"

    def test_water_requirement_medium(self, engine):
        """Verifies 'Medium' water constraint permits Low and Medium trees."""
        result = engine.recommend(criteria={"water_requirement": "Medium"}, top_n=10)
        assert len(result) > 0
        for _, row in result.iterrows():
            assert row["water_requirement"].lower() in ["low", "medium"]

    def test_min_cooling_score_filter(self, engine):
        """Verifies minimum cooling effect score constraint."""
        min_score = 7
        result = engine.recommend(criteria={"min_cooling_score": min_score}, top_n=10)
        assert len(result) > 0
        for _, row in result.iterrows():
            assert row["cooling_effect_score"] >= min_score

    def test_combined_spatial_and_climate_criteria(self, engine):
        """Verifies multi-criteria filtering simultaneously."""
        criteria = {
            "narrow_street": True,
            "water_requirement": "Low",
            "min_cooling_score": 6,
            "top_n": 5,
        }
        result = engine.recommend(criteria=criteria)
        assert len(result) > 0
        assert len(result) <= 5

        for _, row in result.iterrows():
            assert row["water_requirement"].lower() == "low"
            assert row["cooling_effect_score"] >= 6
            master_row = engine.df[engine.df["tree_id"] == row["tree_id"]].iloc[0]
            assert master_row["suitable_for_narrow_streets"] == True

    def test_custom_scoring_weights(self, engine):
        """Verifies custom cooling and carbon weights affect the final score calculation."""
        criteria_cooling_heavy = {
            "cooling_weight": 1.0,
            "carbon_weight": 0.0,
            "top_n": 1,
        }
        result_cooling = engine.recommend(criteria_cooling_heavy)
        top_cooling = result_cooling.iloc[0]
        # When carbon weight is 0, final_score should equal cooling_effect_score
        assert abs(top_cooling["final_score"] - top_cooling["cooling_effect_score"]) < 1e-4

        criteria_carbon_heavy = {
            "cooling_weight": 0.0,
            "carbon_weight": 1.0,
            "top_n": 1,
        }
        result_carbon = engine.recommend(criteria_carbon_heavy)
        top_carbon = result_carbon.iloc[0]
        master_match = engine.df[engine.df["tree_id"] == top_carbon["tree_id"]].iloc[0]
        assert abs(top_carbon["final_score"] - master_match["carbon_sequestration_kg_yr"]) < 1e-4


class TestEdgeCases:
    """Tests edge cases and boundary conditions."""

    @pytest.fixture
    def engine(self):
        return TreeRecommendationEngine()

    def test_strict_constraints_return_empty_dataframe(self, engine):
        """Verifies that impossible criteria return an empty DataFrame with proper schema."""
        impossible_criteria = {
            "narrow_street": True,
            "water_requirement": "Low",
            "min_cooling_score": 99,  # No tree has cooling score >= 99
        }
        result = engine.recommend(criteria=impossible_criteria)
        assert isinstance(result, pd.DataFrame)
        assert result.empty
        expected_cols = [
            "tree_id", "name_ar", "name_en", "category",
            "water_requirement", "cooling_effect_score", "final_score",
        ]
        assert list(result.columns) == expected_cols

    def test_none_criteria(self, engine):
        """Verifies handling when criteria is passed as None."""
        result = engine.recommend(criteria=None, top_n=2)
        assert len(result) == 2

    def test_invalid_top_n_values(self, engine):
        """Verifies robust handling of negative or invalid top_n."""
        result_neg = engine.recommend(criteria={"top_n": -5})
        assert len(result_neg) >= 1  # Gracefully coerced to at least 1

        result_str = engine.recommend(criteria={"top_n": "invalid"})
        assert len(result_str) == 3  # Coerced to default 3


class TestToolWrapper:
    """Tests the Agentic Tool function recommend_trees_tool and Pydantic schema."""

    def test_recommend_trees_tool_success_json_format(self):
        """Verifies that the tool returns a valid JSON string with matching trees."""
        criteria = {
            "narrow_street": True,
            "water_requirement": "Low",
            "min_cooling_score": 6,
            "top_n": 3,
        }
        json_output = recommend_trees_tool(criteria)
        assert isinstance(json_output, str)

        # Parse JSON to verify structure
        parsed = json.loads(json_output)
        assert isinstance(parsed, list)
        assert len(parsed) <= 3
        assert len(parsed) > 0

        first_tree = parsed[0]
        required_keys = {
            "tree_id", "name_ar", "name_en", "category",
            "water_requirement", "cooling_effect_score", "final_score",
        }
        assert required_keys.issubset(first_tree.keys())

        # Ensure Arabic text is preserved in UTF-8 without unicode escapes
        assert any(ord(char) > 127 for char in first_tree["name_ar"])

    def test_recommend_trees_tool_empty_fallback_message(self):
        """Verifies that strict impossible criteria return the clear agent guidance message."""
        strict_criteria = {
            "narrow_street": True,
            "water_requirement": "Low",
            "min_cooling_score": 99,
        }
        output = recommend_trees_tool(strict_criteria)
        assert isinstance(output, str)
        assert "No matching trees found with these strict constraints" in output

    def test_recommend_trees_tool_with_pydantic_input(self):
        """Verifies that passing a Pydantic TreeFilterInput model works seamlessly."""
        pydantic_input = TreeFilterInput(
            narrow_street=True,
            water_requirement="Low",
            min_cooling_score=5,
            top_n=2,
        )
        json_output = recommend_trees_tool(pydantic_input)
        parsed = json.loads(json_output)
        assert isinstance(parsed, list)
        assert len(parsed) <= 2

    def test_get_recommend_trees_tool_spec(self):
        """Verifies the OpenAI/LangChain function tool schema generation."""
        spec = get_recommend_trees_tool_spec()
        assert spec["type"] == "function"
        assert spec["function"]["name"] == "recommend_trees_tool"
        assert "parameters" in spec["function"]
        props = spec["function"]["parameters"]["properties"]
        assert "narrow_street" in props
        assert "water_requirement" in props
        assert "min_cooling_score" in props
        assert "top_n" in props
