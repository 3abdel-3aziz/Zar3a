"""
tests/test_api_climate.py

Tests for the /api/v1/climate RESTful endpoints using TestClient.
"""

from fastapi.testclient import TestClient
import pytest

from src.api.dependencies import get_climate_predictor
from src.api.main import app

client = TestClient(app)


def test_predict_climate_valid():
    """Verifies microclimate temperature prediction endpoint with mock predictor."""
    def mock_predictor(features):
        return {
            "target_variable": "mean_temperature",
            "predicted_value": 31.85,
            "cooling_potential_c": -2.40,
        }

    app.dependency_overrides[get_climate_predictor] = lambda: mock_predictor

    payload = {
        "lat": 30.0444,
        "lon": 31.2357,
        "greenery_area": 8000.0,
        "greenery_density": 0.25,
        "building_area": 20000.0,
        "building_density": 0.35,
        "avg_building_levels": 3.0,
        "road_density": 0.10,
        "distance_to_water": 800.0,
    }
    response = client.post("/api/v1/climate/predictions", json=payload)

    # Clean override
    app.dependency_overrides.clear()

    assert response.status_code == 200
    data = response.json()
    assert data["target_variable"] == "mean_temperature"
    assert data["predicted_mean_temp_c"] == 31.85
    assert data["cooling_potential_c"] == -2.40
    assert data["status"] == "success"


def test_predict_climate_validation_errors():
    """Verifies that out-of-bound geographic coordinates fail Pydantic validation."""
    # Latitude out of Egyptian bounds (< 20.0 or > 35.0)
    response = client.post(
        "/api/v1/climate/predictions",
        json={"lat": 10.0, "lon": 31.0},
    )
    assert response.status_code == 422

    # Longitude out of Egyptian bounds (< 24.0 or > 38.0)
    response = client.post(
        "/api/v1/climate/predictions",
        json={"lat": 30.0, "lon": 55.0},
    )
    assert response.status_code == 422

    # Negative greenery area
    response = client.post(
        "/api/v1/climate/predictions",
        json={"lat": 30.0, "lon": 31.0, "greenery_area": -100.0},
    )
    assert response.status_code == 422
