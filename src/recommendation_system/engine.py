"""
src/recommendation_system/engine.py

Tree Recommendation Engine for urban tree selection in Egypt.
Implements a Singleton pattern for dataset loading and dynamic dictionary-based filtering
with flexible weighted multi-criteria ranking.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional
import pandas as pd

# Configure logger with strict minimum logging level INFO
logger = logging.getLogger("Zar3a.TreeRecommendationEngine")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class TreeRecommendationEngine:
    """
    A scalable, production-ready recommendation engine for urban tree selection 
    in Egypt using a Singleton data loading pattern and dynamic dictionary filtering.
    """
    _instance: Optional[TreeRecommendationEngine] = None
    _df: Optional[pd.DataFrame] = None
    _loaded_path: Optional[str] = None

    def __new__(cls, *args: Any, **kwargs: Any) -> TreeRecommendationEngine:
        """Ensures a single instance of TreeRecommendationEngine exists (Singleton pattern)."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, csv_path: Optional[str] = None, force_reload: bool = False):
        """
        Initializes the recommendation engine and loads the master dataset once.

        Args:
            csv_path: Optional custom path to trees_data.csv.
            force_reload: If True, forces reloading the CSV dataset from disk.
        """
        # Load dataset only if not already loaded or force_reload is requested
        if TreeRecommendationEngine._df is None or force_reload:
            if csv_path is None:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                csv_path = os.path.join(current_dir, "data", "trees_data.csv")

            if not os.path.exists(csv_path):
                logger.error("Dataset file not found at path: %s", csv_path)
                raise FileNotFoundError(f"Dataset not found at path: {csv_path}")

            logger.info("Loading tree recommendation dataset from: %s", csv_path)
            TreeRecommendationEngine._df = pd.read_csv(csv_path)
            TreeRecommendationEngine._loaded_path = csv_path
            logger.info(
                "Successfully loaded %d tree records into memory.",
                len(TreeRecommendationEngine._df),
            )
        else:
            logger.info(
                "Reusing in-memory dataset (%d records, Singleton pattern active).",
                len(TreeRecommendationEngine._df),
            )

        self.df: pd.DataFrame = TreeRecommendationEngine._df

    @classmethod
    def reset(cls) -> None:
        """Utility method to reset the singleton state, primarily used for isolated testing."""
        cls._instance = None
        cls._df = None
        cls._loaded_path = None
        logger.info("TreeRecommendationEngine singleton state reset.")

    def recommend(self, criteria: Optional[Dict[str, Any]] = None, top_n: int = 3) -> pd.DataFrame:
        """
        Filters the master tree dataset dynamically based on an input criteria dictionary 
        and ranks the results using a weighted scoring formula.

        Args:
            criteria (dict, optional): Dictionary containing filtering parameters:
                - narrow_street (bool): Filter for sidewalk-safe deep roots.
                - water_requirement (str): 'Low', 'Medium', or 'High'.
                - min_cooling_score (int/float): Minimum required cooling score.
                - cooling_weight (float, default=0.6): Weight for cooling effect in ranking.
                - carbon_weight (float, default=0.4): Weight for carbon sequestration in ranking.
                - top_n (int, optional): Overrides top_n limit.
            top_n (int): Number of top recommendations to return (default: 3).

        Returns:
            pd.DataFrame: Filtered and ranked tree recommendations with schema:
                ['tree_id', 'name_ar', 'name_en', 'category', 
                 'water_requirement', 'cooling_effect_score', 'final_score']
        """
        criteria = criteria or {}
        limit = criteria.get("top_n", top_n)
        try:
            limit = max(1, int(limit))
        except (ValueError, TypeError):
            limit = 3

        logger.info(
            "Evaluating recommendation request with criteria: %s (limit: %d)",
            criteria,
            limit,
        )

        df = self.df.copy()

        # 1. Dynamic Scalable Filtering
        # Filter A: Narrow street constraint (requires sidewalk-safe root system)
        narrow_val = criteria.get("narrow_street")
        if narrow_val is True or (isinstance(narrow_val, str) and narrow_val.strip().lower() in ("true", "1", "yes")):
            df = df[df["suitable_for_narrow_streets"] == True]
            logger.info("Applied 'narrow_street' filter -> %d candidate trees remaining.", len(df))

        # Filter B: Water requirement availability constraint
        if "water_requirement" in criteria and criteria["water_requirement"]:
            water_val = str(criteria["water_requirement"]).strip().lower()
            if water_val == "low":
                df = df[df["water_requirement"].str.lower() == "low"]
                logger.info("Applied 'water_requirement' == 'Low' filter -> %d candidate trees remaining.", len(df))
            elif water_val == "medium":
                df = df[df["water_requirement"].str.lower().isin(["low", "medium"])]
                logger.info("Applied 'water_requirement' in ['Low', 'Medium'] filter -> %d candidate trees remaining.", len(df))
            elif water_val == "high":
                df = df[df["water_requirement"].str.lower().isin(["low", "medium", "high"])]
                logger.info("Applied 'water_requirement' in ['Low', 'Medium', 'High'] filter -> %d candidate trees remaining.", len(df))
            else:
                logger.warning("Unrecognized water_requirement value '%s'. Filter skipped.", water_val)

        # Filter C: Minimum cooling effect score
        if "min_cooling_score" in criteria and criteria["min_cooling_score"] is not None:
            try:
                min_cooling = float(criteria["min_cooling_score"])
                df = df[df["cooling_effect_score"] >= min_cooling]
                logger.info("Applied 'min_cooling_score' >= %.1f filter -> %d candidate trees remaining.", min_cooling, len(df))
            except (ValueError, TypeError):
                logger.warning("Invalid min_cooling_score value '%s'. Filter skipped.", criteria["min_cooling_score"])

        # Check for empty results after filtering
        if df.empty:
            logger.warning(
                "No matching trees found with criteria: %s. Returning empty DataFrame.",
                criteria,
            )
            return pd.DataFrame(columns=[
                "tree_id", "name_ar", "name_en", "category", 
                "water_requirement", "cooling_effect_score", "final_score"
            ])

        # 2. Flexible Scoring Weights Calculation
        try:
            cooling_weight = float(criteria.get("cooling_weight", 0.6))
            carbon_weight = float(criteria.get("carbon_weight", 0.4))
        except (ValueError, TypeError):
            cooling_weight, carbon_weight = 0.6, 0.4

        df["final_score"] = (
            (df["cooling_effect_score"] * cooling_weight) + 
            (df["carbon_sequestration_kg_yr"] * carbon_weight)
        ).round(2)

        # 3. Sort and Return Top-N Results
        result = df.sort_values(by="final_score", ascending=False).head(limit)
        
        logger.info(
            "Successfully ranked recommendations. Top match: '%s' (Score: %.2f)",
            result.iloc[0]["name_en"],
            result.iloc[0]["final_score"],
        )

        return result[[
            "tree_id", "name_ar", "name_en", "category", 
            "water_requirement", "cooling_effect_score", "final_score"
        ]]