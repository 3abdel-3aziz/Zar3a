"""
src/api/main.py

FastAPI Application Entry Point for the Zar3a Platform.
Exposes RESTful endpoints for tree recommendations, microclimate ML predictions,
legal/environmental RAG knowledge search, and multi-agent LangGraph chat.

Security layers:
  1. CORS Middleware (cross-origin resource sharing)
  2. Rate Limiting via slowapi — 60 requests / minute per IP
  3. Security Headers Middleware — X-Content-Type-Options, X-Frame-Options, X-XSS-Protection
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from api.routers import (
    chat_router,
    climate_router,
    knowledge_router,
    recommendations_router,
)

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger("Zar3a.API.Main")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Rate Limiter — shared instance, importable by individual routers if needed
# Default: 60 requests per minute per IP address
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])


# ---------------------------------------------------------------------------
# Security Headers Middleware
# ---------------------------------------------------------------------------
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Injects hardened security headers into every HTTP response.

    X-Content-Type-Options: nosniff
        Prevents MIME-type sniffing — forces browser to respect declared Content-Type.
    X-Frame-Options: DENY
        Blocks page from being embedded in an iframe (clickjacking protection).
    X-XSS-Protection: 1; mode=block
        Activates the built-in XSS filter in legacy browsers and blocks rendering on attack.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        return response


# ---------------------------------------------------------------------------
# Application Lifespan (Startup / Shutdown)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Handles application startup and shutdown lifecycle events.
    Pre-warms datasets or connections if needed.
    """
    logger.info("Initializing Zar3a Urban Greening & Resilience API...")
    try:
        # Pre-warm recommendation engine dataset
        from recommendation_system.engine import TreeRecommendationEngine
        TreeRecommendationEngine()
        logger.info("TreeRecommendationEngine initialized and ready.")
    except Exception as exc:
        logger.warning("Recommendation engine pre-warm warning: %s", exc)

    yield

    logger.info("Shutting down Zar3a API...")


# ---------------------------------------------------------------------------
# FastAPI Application Factory
# ---------------------------------------------------------------------------
def create_app() -> FastAPI:
    """
    Constructs and configures the FastAPI application instance with:
      1. CORS Middleware
      2. Rate Limiting — 60 req/min per IP (slowapi + SlowAPIMiddleware)
      3. Security Headers — X-Content-Type-Options, X-Frame-Options, X-XSS-Protection
    """
    app = FastAPI(
        title="Zar3a Urban Resilience & Greening API",
        version="1.0.0",
        description=(
            "RESTful API for the Zar3a platform in Egypt. Integrates:\n"
            "- Multi-Agent LangGraph Chat Workflow (`/api/v1/chat`)\n"
            "- Tree Species Recommendation Engine (`/api/v1/recommendations`)\n"
            "- Urban Microclimate ML Temperature Predictor (`/api/v1/climate`)\n"
            "- Environmental Law & Forestry RAG Search (`/api/v1/knowledge`)\n"
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ------------------------------------------------------------------
    # 1. CORS Middleware
    #    NOTE: Replace "*" with explicit origins in production,
    #    e.g. allow_origins=["https://zar3a.eg", "https://app.zar3a.eg"]
    # ------------------------------------------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # 2. Rate Limiting — 60 requests / minute per IP address (slowapi)
    # ------------------------------------------------------------------
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]
    app.add_middleware(SlowAPIMiddleware)

    # ------------------------------------------------------------------
    # 3. Security Headers Middleware
    # ------------------------------------------------------------------
    app.add_middleware(SecurityHeadersMiddleware)

    # ------------------------------------------------------------------
    # Feature Routers — /api/v1 versioned prefix
    # ------------------------------------------------------------------
    app.include_router(chat_router, prefix="/api/v1/chat", tags=["Agent Chat"])
    app.include_router(recommendations_router, prefix="/api/v1/recommendations", tags=["Tree Recommendations"])
    app.include_router(climate_router, prefix="/api/v1/climate", tags=["Climate Prediction"])
    app.include_router(knowledge_router, prefix="/api/v1/knowledge", tags=["Knowledge & RAG"])

    # ------------------------------------------------------------------
    # Health & Root Endpoints
    # ------------------------------------------------------------------
    @app.get(
        "/health",
        status_code=status.HTTP_200_OK,
        tags=["System"],
        summary="System Health Check",
    )
    @app.get(
        "/api/v1/health",
        status_code=status.HTTP_200_OK,
        tags=["System"],
        summary="API v1 Health Check",
    )
    def health_check() -> Dict[str, str]:
        """Returns API operational status."""
        return {
            "status": "healthy",
            "service": "zar3a-api",
            "version": "1.0.0",
        }

    @app.get(
        "/",
        status_code=status.HTTP_200_OK,
        tags=["System"],
        summary="API Root",
    )
    def root() -> Dict[str, Any]:
        """Returns platform root metadata."""
        return {
            "platform": "Zar3a (زرعة)",
            "message": "Welcome to the Zar3a Urban Resilience & Greening API",
            "documentation": "/docs",
            "version": "1.0.0",
        }

    return app


# Default application instance (used by uvicorn / gunicorn)
app: FastAPI = create_app()
