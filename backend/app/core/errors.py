"""Domain errors and their mapping to HTTP responses.

Every error response has the same shape: ``{"error": {"code": ..., "message": ...}}``.
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

log = logging.getLogger("fetalai.errors")


class APIError(Exception):
    status_code = status.HTTP_400_BAD_REQUEST
    code = "bad_request"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class InvalidImageError(APIError):
    status_code = status.HTTP_400_BAD_REQUEST
    code = "invalid_image"


class FileTooLargeError(APIError):
    status_code = status.HTTP_413_CONTENT_TOO_LARGE
    code = "file_too_large"


class UnsupportedMediaTypeError(APIError):
    status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
    code = "unsupported_media_type"


class ModelNotLoadedError(APIError):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "model_not_loaded"


def _body(code: str, message: str, details=None) -> dict:
    err = {"code": code, "message": message}
    if details is not None:
        err["details"] = details
    return {"error": err}


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(APIError)
    async def handle_api_error(request: Request, exc: APIError):
        log.warning("request rejected", extra={"path": request.url.path, "code": exc.code, "detail": exc.message})
        return JSONResponse(status_code=exc.status_code, content=_body(exc.code, exc.message))

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError):
        details = [{"field": ".".join(str(p) for p in e["loc"]), "message": e["msg"]} for e in exc.errors()]
        log.warning("validation failed", extra={"path": request.url.path, "details": details})
        return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                            content=_body("validation_error", "Invalid request parameters", details))
