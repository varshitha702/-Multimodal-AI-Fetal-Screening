"""FastAPI dependencies."""
from __future__ import annotations

from fastapi import Request

from backend.app.core.config import Settings
from backend.app.core.errors import ModelNotLoadedError
from backend.app.services.ctg_service import CTGService
from backend.app.services.model_service import ModelService


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def get_model_service(request: Request) -> ModelService:
    return request.app.state.model_service


def require_model(request: Request) -> ModelService:
    """503 when the model failed to load at startup (missing or broken checkpoint)."""
    service: ModelService = request.app.state.model_service
    if not service.ready:
        raise ModelNotLoadedError(f"Model is not loaded. {service.load_error or ''}".strip())
    return service


def get_ctg_service(request: Request) -> CTGService:
    return request.app.state.ctg_service


def require_ctg(request: Request) -> CTGService:
    """503 when the CTG model failed to load at startup."""
    service: CTGService = request.app.state.ctg_service
    if not service.ready:
        raise ModelNotLoadedError(f"CTG model is not loaded. {service.load_error or ''}".strip())
    return service
