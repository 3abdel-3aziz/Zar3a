"""
tests/test_api_main.py

Tests for the FastAPI main application entry point, health checks, root metadata, and CORS.
"""

from fastapi.testclient import TestClient
import pytest

from src.api.main import app

client = TestClient(app)


def test_root_endpoint():
    """Verifies that root '/' returns 200 OK and expected platform branding."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "platform" in data
    assert "Zar3a" in data["platform"]
    assert data["version"] == "1.0.0"


def test_health_check_endpoints():
    """Verifies that both /health and /api/v1/health report healthy status."""
    for path in ["/health", "/api/v1/health"]:
        response = client.get(path)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "zar3a-api"


def test_nonexistent_endpoint_returns_404():
    """Verifies that an unknown endpoint returns 404 Not Found."""
    response = client.get("/api/v1/unknown_resource")
    assert response.status_code == 404


def test_cors_headers_present():
    """Verifies that CORS middleware attaches appropriate headers on preflight requests."""
    response = client.options(
        "/api/v1/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "*"
