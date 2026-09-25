"""
tests/test_api_recommendations.py

Tests for the /api/v1/recommendations RESTful endpoints using TestClient.
"""

from fastapi.testclient import TestClient
import pytest

from src.api.main import app

client = TestClient(app)


def test_generate_recommendations_valid():
    """Verifies that valid filter criteria return ranked tree recommendations."""
    payload = {
        "narrow_street": True,
        "water_requirement": "Low",
        "min_cooling_score": 5,
        "top_n": 3,
    }
    response = client.post("/api/v1/recommendations", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "total_results" in data
    assert "recommendations" in data
    assert len(data["recommendations"]) <= 3
    if data["recommendations"]:
        first = data["recommendations"][0]
        assert "name_ar" in first
        assert "name_en" in first
        assert "cooling_effect_score" in first


def test_generate_recommendations_validation_errors():
    """Verifies that invalid payload parameters trigger 422 Unprocessable Entity."""
    # min_cooling_score > 10
    response = client.post(
        "/api/v1/recommendations",
        json={"min_cooling_score": 15},
    )
    assert response.status_code == 422

    # invalid water_requirement literal
    response = client.post(
        "/api/v1/recommendations",
        json={"water_requirement": "SuperHigh"},
    )
    assert response.status_code == 422

    # top_n < 1
    response = client.post(
        "/api/v1/recommendations",
        json={"top_n": 0},
    )
    assert response.status_code == 422


def test_list_species():
    """Verifies that GET /api/v1/recommendations/species returns tree inventory."""
    response = client.get("/api/v1/recommendations/species")
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "species" in data
    assert data["total"] > 0
    assert len(data["species"]) == data["total"]


def test_list_species_with_filters():
    """Verifies filtering species by category and water requirement."""
    response = client.get(
        "/api/v1/recommendations/species",
        params={"water_requirement": "Low"},
    )
    assert response.status_code == 200
    data = response.json()
    for item in data["species"]:
        assert item["water_requirement"] == "Low"


def test_get_species_by_id_success():
    """Verifies retrieving a specific tree species by its ID."""
    # List first to get a valid ID
    list_res = client.get("/api/v1/recommendations/species")
    valid_id = list_res.json()["species"][0]["tree_id"]

    response = client.get(f"/api/v1/recommendations/species/{valid_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["tree_id"] == valid_id
    assert "name_ar" in data
    assert "name_en" in data


def test_get_species_by_id_not_found():
    """Verifies that an invalid ID returns 404 Not Found."""
    response = client.get("/api/v1/recommendations/species/999999")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()
