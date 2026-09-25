"""ClimatePredictor Microservice Module for Zar3a.

Production-grade, agent-ready inference service for urban climate and heat predictions.
Provides robust Pydantic schemas, MLflow model caching, schema validation, and structured outputs.
"""

import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from pydantic import BaseModel, ConfigDict, Field, model_validator

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ClimatePredictionInput(BaseModel):
    """Input feature schema for urban climate and temperature regression.

    Exposed to LLM agents, LangGraph/LangChain tools, and FastAPI endpoints
    with full typing, field descriptions, and validation constraints.
    """

    model_config = ConfigDict(extra="ignore")

    lat: float = Field(
        ...,
        description="Latitude of the geographic grid cell or location (e.g., 30.0444 for Cairo).",
        ge=20.0,
        le=35.0,
        examples=[30.0444],
    )
    lon: float = Field(
        ...,
        description="Longitude of the geographic grid cell or location (e.g., 31.2357 for Cairo).",
        ge=24.0,
        le=38.0,
        examples=[31.2357],
    )
    greenery_area: float = Field(
        default=0.0,
        description="Total surface area occupied by greenery (parks, gardens, vegetation) in square meters.",
        ge=0.0,
        examples=[12500.0],
    )
    greenery_density: float = Field(
        default=0.0,
        description="Ratio of greenery area to total district/cell area (range: 0.0 to 1.0).",
        ge=0.0,
        le=1.0,
        examples=[0.15],
    )
    building_area: float = Field(
        default=0.0,
        description="Total footprint area occupied by building structures in square meters.",
        ge=0.0,
        examples=[45000.0],
    )
    building_density: float = Field(
        default=0.0,
        description="Ratio of building footprint area to total district/cell area (range: 0.0 to 1.0).",
        ge=0.0,
        le=1.0,
        examples=[0.45],
    )
    avg_building_levels: float = Field(
        default=1.0,
        description="Average number of building storeys/floors within the grid cell.",
        ge=1.0,
        le=100.0,
        examples=[4.5],
    )
    road_density: float = Field(
        default=0.0,
        description="Total road length divided by district/cell area (transport density proxy).",
        ge=0.0,
        examples=[0.025],
    )
    distance_to_water: float = Field(
        default=0.0,
        description="Distance in meters/degrees to the nearest surface water body (e.g., Nile River).",
        ge=0.0,
        examples=[1200.0],
    )
    poi_density: float = Field(
        default=0.0,
        description="Density of points of interest (commercial amenities, offices, shops) per unit area.",
        ge=0.0,
        examples=[0.005],
    )
    bare_ground_density: float = Field(
        default=0.0,
        description="Proportion of cell area covered by unshaded bare ground, sand, or desert (range: 0.0 to 1.0).",
        ge=0.0,
        le=1.0,
        examples=[0.2],
    )
    ndvi_mean: float = Field(
        default=0.15,
        description="Mean Normalized Difference Vegetation Index (NDVI), typically between -0.1 and 0.8 in urban Cairo.",
        ge=-1.0,
        le=1.0,
        examples=[0.18],
    )
    elevation: float = Field(
        default=40.0,
        description="Elevation / altitude in meters above sea level.",
        ge=-50.0,
        le=3000.0,
        examples=[65.0],
    )
    nighttime_lights_intensity: float = Field(
        default=15.0,
        description="Proxy index for nocturnal human and economic activity derived from satellite nighttime lights.",
        ge=0.0,
        examples=[22.5],
    )
    green_building_ratio: Optional[float] = Field(
        default=None,
        description="Ratio of green surface area to building footprint area. If omitted, computed automatically.",
        ge=0.0,
        examples=[0.27],
    )

    @model_validator(mode="after")
    def compute_derived_metrics(self) -> "ClimatePredictionInput":
        """Computes derived features if not explicitly provided."""
        if self.green_building_ratio is None:
            self.green_building_ratio = self.greenery_area / (self.building_area + 1e-6)
        return self

    def to_dataframe(self) -> pd.DataFrame:
        """Converts input into a single-row pandas DataFrame matching model training features."""
        data = self.model_dump()
        return pd.DataFrame([data])


class ClimatePredictionOutput(BaseModel):
    """Structured response schema optimized for Agent tool interpretation and REST APIs."""

    model_config = ConfigDict(extra="ignore")

    status: str = Field(
        ...,
        description="Execution status ('success' or 'error').",
        examples=["success"],
    )
    target_variable: str = Field(
        default="mean_temperature",
        description="The predicted climatic variable name.",
        examples=["mean_temperature"],
    )
    predicted_value: float = Field(
        ...,
        description="The estimated numerical target prediction.",
        examples=[28.45],
    )
    unit: str = Field(
        default="celsius",
        description="Measurement unit for the prediction.",
        examples=["celsius"],
    )
    experiment_name: str = Field(
        ...,
        description="MLflow experiment name from which the model was loaded.",
        examples=["tree_baseline"],
    )
    run_id: Optional[str] = Field(
        default=None,
        description="Unique MLflow run identifier for lineage and auditability.",
        examples=["511d469b0c624e5a9302dd95480c8f7d"],
    )
    model_uri: Optional[str] = Field(
        default=None,
        description="MLflow model artifact URI used for inference.",
        examples=["runs:/511d469b0c624e5a9302dd95480c8f7d/model"],
    )
    timestamp: str = Field(
        ...,
        description="ISO 8601 UTC timestamp of when prediction was computed.",
        examples=["2026-09-24T11:45:00Z"],
    )
    features_used: List[str] = Field(
        default_factory=list,
        description="List of feature column names fed to the regressor model.",
    )


class ClimatePredictor:
    """Production-grade ML inference engine for climate regression.

    Features:
    - Lazy loading with in-memory caching to eliminate redundant MLflow artifact downloads.
    - Automatic lineage resolution and latest model discovery from MLflow experiments.
    - Full compatibility with LangChain/LangGraph Agent tools and FastAPI endpoints.
    - Batch and single-sample inference modes.
    """

    def __init__(
        self,
        experiment_name: str = "tree_baseline",
        model_uri: Optional[str] = None,
        run_id: Optional[str] = None,
        tracking_uri: Optional[str] = None,
    ):
        self.experiment_name = experiment_name
        self.model_uri = model_uri
        self.run_id = run_id
        self._model: Optional[Any] = None

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

    def get_model(self) -> Any:
        """Retrieves the cached model or dynamically loads it from MLflow."""
        if self._model is not None:
            return self._model

        # 1. Direct model URI provided
        if self.model_uri:
            logger.info(f"Loading model artifact from explicit URI: {self.model_uri}")
            try:
                self._model = mlflow.pyfunc.load_model(self.model_uri)
                return self._model
            except Exception as e:
                raise RuntimeError(f"Failed to load model from provided model_uri '{self.model_uri}': {e}") from e

        # 2. Specific run_id provided
        if self.run_id:
            resolved_uri = f"runs:/{self.run_id}/model"
            logger.info(f"Loading model artifact from specified run_id: {self.run_id} ({resolved_uri})")
            try:
                self._model = mlflow.pyfunc.load_model(resolved_uri)
                self.model_uri = resolved_uri
                return self._model
            except Exception as e:
                raise RuntimeError(f"Failed to load model from run_id '{self.run_id}': {e}") from e

        # 3. Dynamic lookup of latest run in MLflow experiment
        logger.info(f"Searching for latest model artifact in MLflow experiment: '{self.experiment_name}'")
        client = MlflowClient()
        try:
            experiment = client.get_experiment_by_name(self.experiment_name)
        except Exception as e:
            raise RuntimeError(f"MLflow connection error when searching for experiment '{self.experiment_name}': {e}") from e

        if not experiment:
            raise FileNotFoundError(
                f"MLflow experiment '{self.experiment_name}' does not exist. "
                "Ensure that the training pipeline has been executed at least once."
            )

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=5,
        )
        if not runs:
            raise FileNotFoundError(
                f"No runs found in experiment '{self.experiment_name}'. "
                "Run 'python src/climate_ml/train_pipeline.py' to produce a model artifact."
            )

        # Select the latest finished or active run with model artifact
        target_run = None
        for r in runs:
            if r.info.status in ("FINISHED", "RUNNING"):
                target_run = r
                break
        if target_run is None:
            target_run = runs[0]

        self.run_id = target_run.info.run_id
        self.model_uri = f"runs:/{self.run_id}/model"
        logger.info(f"Resolved latest model: {self.model_uri} (Run status: {target_run.info.status})")

        try:
            self._model = mlflow.pyfunc.load_model(self.model_uri)
            return self._model
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model artifact at '{self.model_uri}'. "
                f"Underlying error: {e}"
            ) from e

    def predict(
        self,
        input_data: Union[ClimatePredictionInput, Dict[str, Any]],
        as_dict: bool = False,
    ) -> Union[ClimatePredictionOutput, Dict[str, Any]]:
        """Computes prediction for a single geospatial location or feature vector.

        Args:
            input_data: A ClimatePredictionInput instance or a dictionary of features.
            as_dict: If True, returns a plain python dictionary (ideal for LangChain/Agent tools).
                     If False (default), returns the validated ClimatePredictionOutput model.

        Returns:
            ClimatePredictionOutput instance or dict with prediction results.
        """
        # Validate into Pydantic model if raw dict is passed
        if isinstance(input_data, dict):
            validated_input = ClimatePredictionInput(**input_data)
        elif isinstance(input_data, ClimatePredictionInput):
            validated_input = input_data
        else:
            raise TypeError(
                f"Expected ClimatePredictionInput or dict, got {type(input_data).__name__}"
            )

        model = self.get_model()
        df_features = validated_input.to_dataframe()

        try:
            preds = model.predict(df_features)
            predicted_value = float(preds[0])
        except Exception as e:
            logger.error(f"Inference computation failed: {e}")
            raise RuntimeError(f"Model prediction failed on input features: {e}") from e

        output = ClimatePredictionOutput(
            status="success",
            target_variable="mean_temperature",
            predicted_value=round(predicted_value, 4),
            unit="celsius",
            experiment_name=self.experiment_name,
            run_id=self.run_id,
            model_uri=self.model_uri,
            timestamp=datetime.now(timezone.utc).isoformat(),
            features_used=list(df_features.columns),
        )

        return output.model_dump() if as_dict else output

    def predict_batch(
        self,
        inputs: List[Union[ClimatePredictionInput, Dict[str, Any]]],
        as_dict: bool = False,
    ) -> Union[List[ClimatePredictionOutput], List[Dict[str, Any]]]:
        """Performs batch prediction over a list of inputs efficiently.

        Args:
            inputs: List of ClimatePredictionInput instances or feature dictionaries.
            as_dict: Whether to return dictionaries instead of Pydantic models.

        Returns:
            List of ClimatePredictionOutput instances or dicts.
        """
        if not inputs:
            return []

        validated_inputs = [
            i if isinstance(i, ClimatePredictionInput) else ClimatePredictionInput(**i)
            for i in inputs
        ]

        # Combine into single DataFrame for vectorized prediction
        dfs = [i.to_dataframe() for i in validated_inputs]
        batch_df = pd.concat(dfs, ignore_index=True)

        model = self.get_model()
        try:
            preds = model.predict(batch_df)
        except Exception as e:
            logger.error(f"Batch inference failed: {e}")
            raise RuntimeError(f"Batch prediction failed: {e}") from e

        now_str = datetime.now(timezone.utc).isoformat()
        results = [
            ClimatePredictionOutput(
                status="success",
                target_variable="mean_temperature",
                predicted_value=round(float(p), 4),
                unit="celsius",
                experiment_name=self.experiment_name,
                run_id=self.run_id,
                model_uri=self.model_uri,
                timestamp=now_str,
                features_used=list(batch_df.columns),
            )
            for p in preds
        ]

        return [r.model_dump() for r in results] if as_dict else results


def predict_urban_temperature(
    input_data: Union[ClimatePredictionInput, Dict[str, Any]],
    experiment_name: str = "tree_baseline",
) -> Dict[str, Any]:
    """Agent tool entrypoint for predicting urban mean temperature.

    Easily wrapped in LangChain `@tool` or LangGraph node:
    ```python
    from langchain.tools import tool

    @tool
    def get_temperature_prediction(input_data: ClimatePredictionInput) -> dict:
        \"\"\"Predicts mean urban temperature for coordinates and spatial features.\"\"\"
        return predict_urban_temperature(input_data)
    ```
    """
    predictor = ClimatePredictor(experiment_name=experiment_name)
    result = predictor.predict(input_data, as_dict=True)
    return result if isinstance(result, dict) else result.model_dump()
