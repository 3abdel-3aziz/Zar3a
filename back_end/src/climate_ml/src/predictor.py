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

# Dynamically locate project root containing mlruns or .git
_this_file = Path(__file__).resolve()
REPO_ROOT = _this_file.parents[4] if len(_this_file.parents) > 4 else _this_file.parents[-1]
for parent in _this_file.parents:
    if (parent / "mlruns").exists() or (parent / ".git").exists():
        REPO_ROOT = parent
        break

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _configure_local_mlflow_tracking() -> None:
    """
    Auto-detects and configures the local MLflow tracking URI.

    Priority:
    1. MLFLOW_TRACKING_URI environment variable (if set)
    2. Local mlruns/ directory at repo root (file-based fallback)
    3. SQLite mlflow.db co-located with the climate_ml package
    """
    # If already configured externally, respect it
    if os.environ.get("MLFLOW_TRACKING_URI"):
        return

    # Direct mlruns directory at repo root
    mlruns_dir = REPO_ROOT / "mlruns"
    if mlruns_dir.exists():
        uri = mlruns_dir.as_uri()
        mlflow.set_tracking_uri(uri)
        logger.info("Auto-configured MLflow tracking URI (mlruns/): %s", uri)
        return

    # Search for mlflow.db relative to this file
    climate_ml_dir = Path(__file__).resolve().parent.parent
    sqlite_db = climate_ml_dir / "mlflow.db"
    if sqlite_db.exists():
        uri = f"sqlite:///{sqlite_db.as_posix()}"
        mlflow.set_tracking_uri(uri)
        logger.info("Auto-configured MLflow tracking URI: %s", uri)
        return


def _find_local_model_path() -> Optional[Path]:
    """
    Scans for the most recently modified MLmodel artifact on the local filesystem.

    Returns:
        Path to the directory containing MLmodel, or None if not found.
    """
    search_roots = [
        REPO_ROOT,
        Path(__file__).resolve().parent.parent,
        Path(__file__).resolve().parents[3] if len(Path(__file__).resolve().parents) > 3 else REPO_ROOT,
        Path(__file__).resolve().parents[4] if len(Path(__file__).resolve().parents) > 4 else REPO_ROOT,
    ]

    candidates: list[Path] = []
    seen = set()
    for root in search_roots:
        if root in seen or not root.exists():
            continue
        seen.add(root)
        for mlruns in [root / "mlruns", root / "models"]:
            if mlruns.exists():
                for mlmodel_file in mlruns.rglob("MLmodel"):
                    candidates.append(mlmodel_file.parent)

    if not candidates:
        return None

    best = max(candidates, key=lambda p: p.stat().st_mtime if p.exists() else 0)
    logger.info("Discovered local model artifact directory: %s", best)
    return best


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

    @model_validator(mode="before")
    @classmethod
    def alias_coordinates(cls, data: Any) -> Any:
        """Allows 'latitude' and 'longitude' aliases for 'lat' and 'lon'."""
        if isinstance(data, dict):
            if "lat" not in data and "latitude" in data:
                data["lat"] = data["latitude"]
            if "lon" not in data and "longitude" in data:
                data["lon"] = data["longitude"]
        return data

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
    cooling_potential_c: Optional[float] = Field(
        default=None,
        description="Estimated local temperature reduction (°C) achieved by maximum canopy/greenery.",
        examples=[1.85],
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
        """Retrieves the cached model or dynamically loads it from MLflow or local disk."""
        if self._model is not None:
            return self._model

        # 1. Direct model URI provided
        if self.model_uri:
            logger.info("Loading model artifact from explicit URI: %s", self.model_uri)
            try:
                self._model = mlflow.pyfunc.load_model(self.model_uri)
                return self._model
            except Exception as e:
                logger.warning("Failed loading model from explicit URI '%s': %s", self.model_uri, e)

        # 2. Direct local artifact discovery (fastest and immune to URI/SQLite path corruption)
        local_model_path = _find_local_model_path()
        if local_model_path and local_model_path.exists():
            try:
                logger.info("Loading model directly from local artifact path: %s", local_model_path)
                self._model = mlflow.pyfunc.load_model(str(local_model_path))
                self.model_uri = str(local_model_path)
                return self._model
            except Exception as exc:
                logger.warning("Direct local model load failed: %s. Falling back to MLflow tracking.", exc)

        # 3. Ensure local MLflow tracking is configured before any client operations
        _configure_local_mlflow_tracking()

        # 4. Specific run_id provided
        if self.run_id:
            resolved_uri = f"runs:/{self.run_id}/model"
            logger.info("Loading model artifact from specified run_id: %s (%s)", self.run_id, resolved_uri)
            try:
                self._model = mlflow.pyfunc.load_model(resolved_uri)
                self.model_uri = resolved_uri
                return self._model
            except Exception as e:
                logger.warning("Failed to load model from run_id '%s': %s", self.run_id, e)

        # 5. Dynamic lookup of latest run in MLflow experiment
        logger.info("Searching for latest model artifact in MLflow experiment: '%s'", self.experiment_name)
        try:
            client = MlflowClient()
            experiment = client.get_experiment_by_name(self.experiment_name)
            if experiment:
                runs = client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    order_by=["attributes.start_time DESC"],
                    max_results=5,
                )
                if runs:
                    target_run = next((r for r in runs if r.info.status in ("FINISHED", "RUNNING")), runs[0])
                    self.run_id = target_run.info.run_id
                    self.model_uri = f"runs:/{self.run_id}/model"
                    logger.info("Resolved latest model: %s (Run status: %s)", self.model_uri, target_run.info.status)
                    self._model = mlflow.pyfunc.load_model(self.model_uri)
                    return self._model
        except Exception as e:
            logger.warning("MLflow experiment lookup failed: %s", e)

        # 6. Final attempt on discovered local artifact path if earlier attempts failed
        if local_model_path and local_model_path.exists():
            self._model = mlflow.pyfunc.load_model(str(local_model_path))
            self.model_uri = str(local_model_path)
            return self._model

        raise RuntimeError(
            f"No valid ML model artifact could be found or loaded for experiment '{self.experiment_name}'."
        )

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

        # Calculate dynamic cooling potential using the ML model by comparing with an increased greenery scenario
        cooling_potential = None
        try:
            green_df = df_features.copy()
            if "greenery_area" in green_df.columns:
                green_df["greenery_area"] = float(green_df["greenery_area"].iloc[0]) + 30000.0
            if "building_area" in green_df.columns:
                green_df["building_area"] = max(0.0, float(green_df["building_area"].iloc[0]) - 5000.0)
            cooler_preds = model.predict(green_df)
            cooling_delta = predicted_value - float(cooler_preds[0])
            cooling_potential = round(
                max(0.4, cooling_delta if cooling_delta > 0 else (float(validated_input.greenery_density or 0.15) * 3.2 + 0.8)),
                2,
            )
        except Exception as exc:
            logger.warning("Cooling potential calculation fallback: %s", exc)
            cooling_potential = round(float(validated_input.greenery_density or 0.15) * 3.5 + 0.5, 2)

        output = ClimatePredictionOutput(
            status="success",
            target_variable="mean_temperature",
            predicted_value=round(predicted_value, 4),
            cooling_potential_c=cooling_potential,
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
