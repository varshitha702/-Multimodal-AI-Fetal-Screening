"""FastAPI application factory.

Run from the repository root:
    uvicorn backend.app.main:app --reload --port 8000
OpenAPI docs: http://localhost:8000/docs
"""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.v1.routes import router
from backend.app.core.config import Settings, get_settings
from backend.app.core.errors import register_exception_handlers
from backend.app.core.logging import configure_logging
from backend.app.schemas.prediction import DISCLAIMER
from backend.app.services.model_service import ModelService

log = logging.getLogger("fetalai.api")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Load the model ONCE at startup; every request reuses it.
        service = ModelService(settings)
        service.load()
        app.state.model_service = service
        yield
        log.info("shutdown")

    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        description=f"Fetal head segmentation and head-circumference measurement (HC18 U-Net).\n\n**{DISCLAIMER}**",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_log(request: Request, call_next):
        request_id = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
        t0 = time.perf_counter()
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        log.info("request", extra={"request_id": request_id, "method": request.method,
                                   "path": request.url.path, "status": response.status_code,
                                   "ms": round((time.perf_counter() - t0) * 1000, 1)})
        return response

    register_exception_handlers(app)
    app.include_router(router)
    return app


app = create_app()
