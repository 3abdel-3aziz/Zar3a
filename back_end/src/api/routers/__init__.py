"""
src/api/routers package

Exports all feature-based API routers for the Zar3a platform.
"""

from api.routers.chat import router as chat_router
from api.routers.climate import router as climate_router
from api.routers.knowledge import router as knowledge_router
from api.routers.recommendations import router as recommendations_router

__all__ = [
    "chat_router",
    "climate_router",
    "knowledge_router",
    "recommendations_router",
]
