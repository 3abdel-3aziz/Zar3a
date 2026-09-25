"""
src/api/routers/climate.py

FastAPI router exposing the Zar3a Urban Microclimate ML Predictor.
Provides RESTful endpoints for temperature regression and cooling impact analysis.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from api.dependencies import get_climate_predictor
from climate_ml.src.predictor import ClimatePredictionInput

logger = logging.getLogger("Zar3a.API.Climate")

router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class ClimatePredictionRequest(BaseModel):
    """Input features for microclimate temperature regression."""

    lat: float = Field(
        ...,
        ge=20.0,
        le=35.0,
        description="Geographic latitude in Egypt (e.g. 30.0444 for Cairo)",
        examples=[30.0444],
    )
    lon: float = Field(
        ...,
        ge=24.0,
        le=38.0,
        description="Geographic longitude in Egypt (e.g. 31.2357 for Cairo)",
        examples=[31.2357],
    )
    greenery_area: float = Field(
        default=5000.0,
        ge=0.0,
        description="Total surface area of parks, gardens, and vegetation in square meters",
        examples=[5000.0],
    )
    greenery_density: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Ratio of greenery area to total area (0.0 - 1.0)",
        examples=[0.20],
    )
    building_area: float = Field(
        default=25000.0,
        ge=0.0,
        description="Total footprint area of buildings in square meters",
        examples=[25000.0],
    )
    building_density: float = Field(
        default=0.40,
        ge=0.0,
        le=1.0,
        description="Ratio of building footprint to total area (0.0 - 1.0)",
        examples=[0.40],
    )
    avg_building_levels: float = Field(
        default=4.0,
        ge=1.0,
        description="Average height/floors of nearby buildings",
        examples=[4.0],
    )
    road_density: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Ratio of paved asphalt surfaces (0.0 - 1.0)",
        examples=[0.05],
    )
    distance_to_water: float = Field(
        default=1200.0,
        ge=0.0,
        description="Distance to nearest significant water body (e.g. Nile) in meters",
        examples=[1200.0],
    )


class ClimatePredictionResponse(BaseModel):
    """Output results of urban microclimate temperature regression."""

    target_variable: str = Field(..., description="Target predicted metric (e.g. mean_temperature)")
    predicted_mean_temp_c: float = Field(..., description="Predicted local surface temperature in °C")
    cooling_potential_c: Optional[float] = Field(None, description="Estimated cooling potential under maximum canopy cover")
    features: Dict[str, Any] = Field(..., description="Input feature payload used for model inference")
    status: str = Field("success", description="Prediction execution status")


# ---------------------------------------------------------------------------
# Route Handlers
# ---------------------------------------------------------------------------

@router.post(
    "/predictions",
    response_model=ClimatePredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict Urban Microclimate Temperature",
    description="Invokes the ML regression model to forecast local surface temperatures and cooling potential.",
)
def predict_climate(
    payload: ClimatePredictionRequest,
    predictor: Callable[..., Any] = Depends(get_climate_predictor),
) -> ClimatePredictionResponse:
    """
    Executes the trained ML model against urban morphology and greenery features.
    """
    feature_dict = payload.model_dump()
    try:
        # Validate against the internal ML model schema
        validated_input = ClimatePredictionInput(**feature_dict)
        pred_result = predictor(validated_input)

        predicted_val = float(pred_result.get("predicted_value", 32.5))
        cooling_val = pred_result.get("cooling_potential_c")

        return ClimatePredictionResponse(
            target_variable=pred_result.get("target_variable", "mean_temperature"),
            predicted_mean_temp_c=round(predicted_val, 2),
            cooling_potential_c=round(float(cooling_val), 2) if cooling_val is not None else None,
            features=feature_dict,
            status="success",
        )
    except Exception as exc:
        logger.error("Climate ML prediction failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Climate prediction failed: {exc}",
        ) from exc
