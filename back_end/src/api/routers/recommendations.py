"""
src/api/routers/recommendations.py

FastAPI router exposing the Zar3a Tree Recommendation Engine.
Provides RESTful endpoints for tree species recommendations and inventory exploration.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from api.dependencies import get_recommendation_engine
from recommendation_system.engine import TreeRecommendationEngine

logger = logging.getLogger("Zar3a.API.Recommendations")

router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class TreeRecommendationRequest(BaseModel):
    """Input payload for generating tailored tree recommendations."""

    narrow_street: bool = Field(
        default=False,
        description="True if planting in a narrow corridor requiring deep roots that preserve sidewalks and underground utilities.",
    )
    water_requirement: Literal["Low", "Medium", "High"] = Field(
        default="Low",
        description="Water availability constraint. Options: 'Low', 'Medium', or 'High'.",
    )
    min_cooling_score: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Minimum desired cooling effect score from 1 (mild) to 10 (maximum shade/cooling).",
    )
    top_n: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Number of top candidate species to return (1-20).",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "narrow_street": True,
                "water_requirement": "Low",
                "min_cooling_score": 7,
                "top_n": 3,
            }
        }
    }


class TreeSpeciesItem(BaseModel):
    """Structured representation of a single recommended tree species."""

    tree_id: Optional[int] = Field(None, description="Unique identifier for the tree record")
    name_ar: str = Field(..., description="Arabic name of the tree")
    name_en: str = Field(..., description="English/Scientific name of the tree")
    category: Optional[str] = Field(None, description="Tree category (e.g. Shade, Aesthetic, Windbreak)")
    root_type: Optional[str] = Field(None, description="Root system type (e.g. Deep, Shallow, Invasive)")
    water_requirement: Optional[str] = Field(None, description="Water requirements")
    growth_rate: Optional[str] = Field(None, description="Tree growth rate")
    canopy_spread_m: Optional[float] = Field(None, description="Canopy diameter in meters")
    cooling_effect_score: Optional[int] = Field(None, description="Microclimate cooling score (1-10)")
    air_purification_score: Optional[int] = Field(None, description="Air pollution filtration score (1-10)")
    maintenance_cost_score: Optional[int] = Field(None, description="Maintenance cost score")
    final_score: Optional[float] = Field(None, description="Composite weighted suitability score")


class TreeRecommendationResponse(BaseModel):
    """Response containing ranked tree recommendations."""

    total_results: int = Field(..., description="Number of returned candidate species")
    criteria: Dict[str, Any] = Field(..., description="Criteria applied for filtering")
    recommendations: List[TreeSpeciesItem] = Field(..., description="Ranked list of recommended trees")


class SpeciesListResponse(BaseModel):
    """Response listing available species in the database."""

    total: int = Field(..., description="Total number of species returned")
    species: List[TreeSpeciesItem] = Field(..., description="List of tree records")


# ---------------------------------------------------------------------------
# Route Handlers
# ---------------------------------------------------------------------------

@router.post(
    "",
    response_model=TreeRecommendationResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate Tree Recommendations",
    description="Filters and scores tree species based on spatial constraints, water availability, and cooling needs.",
)
def generate_recommendations(
    payload: TreeRecommendationRequest,
    engine: TreeRecommendationEngine = Depends(get_recommendation_engine),
) -> TreeRecommendationResponse:
    """
    Executes the tree recommendation algorithm against the master dataset.
    """
    criteria = {
        "narrow_street": payload.narrow_street,
        "water_requirement": payload.water_requirement,
        "min_cooling_score": payload.min_cooling_score,
    }

    try:
        recommendations_df = engine.recommend(criteria=criteria, top_n=payload.top_n)
        records = recommendations_df.to_dict(orient="records")
        return TreeRecommendationResponse(
            total_results=len(records),
            criteria=criteria,
            recommendations=[TreeSpeciesItem.model_validate(rec) for rec in records],
        )
    except Exception as exc:
        logger.error("Error executing recommendations: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate recommendations: {exc}",
        ) from exc


@router.get(
    "/species",
    response_model=SpeciesListResponse,
    status_code=status.HTTP_200_OK,
    summary="List Tree Species",
    description="Retrieves available tree species with optional filters for category and water needs.",
)
def list_species(
    category: Optional[str] = Query(None, description="Filter by category (e.g. Shade, Aesthetic)"),
    water_requirement: Optional[str] = Query(None, description="Filter by water requirement (Low, Medium, High)"),
    narrow_street_safe: Optional[bool] = Query(None, description="Filter only species suitable for narrow streets"),
    engine: TreeRecommendationEngine = Depends(get_recommendation_engine),
) -> SpeciesListResponse:
    """
    Returns inventory of available tree species in the master database.
    """
    try:
        df = engine.df.copy()
        if category:
            df = df[df["category"].str.lower() == category.lower()]
        if water_requirement:
            df = df[df["water_requirement"].str.lower() == water_requirement.lower()]
        if narrow_street_safe is not None:
            df = df[df["suitable_for_narrow_streets"] == narrow_street_safe]

        records = df.to_dict(orient="records")
        return SpeciesListResponse(
            total=len(records),
            species=[TreeSpeciesItem.model_validate(rec) for rec in records],
        )
    except Exception as exc:
        logger.error("Error listing species: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve tree species: {exc}",
        ) from exc


@router.get(
    "/species/{tree_id}",
    response_model=TreeSpeciesItem,
    status_code=status.HTTP_200_OK,
    summary="Get Tree Species by ID",
    description="Retrieves full profile details for a specific tree species.",
)
def get_species_by_id(
    tree_id: int,
    engine: TreeRecommendationEngine = Depends(get_recommendation_engine),
) -> TreeSpeciesItem:
    """
    Fetches a single tree record by its unique ID.
    """
    try:
        matched = engine.df[engine.df["tree_id"] == tree_id]
        if matched.empty:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Tree species with ID {tree_id} not found.",
            )
        record = matched.iloc[0].to_dict()
        return TreeSpeciesItem(**record)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error retrieving species ID %d: %s", tree_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve species profile: {exc}",
        ) from exc
