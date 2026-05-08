from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, Request
from fastapi.exception_handlers import (
    http_exception_handler,
    request_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.chat import router as chat_router
from app.api.chat import v1_router as chat_v1_router
from app.api.report import router as report_router
from app.api.report import v1_router as report_v1_router
from app.api.schemas import api_error, api_success
from app.api.upload import router as upload_router
from app.api.upload import v1_router as upload_v1_router
from app.config import settings
from app.dependencies import request_id_context


def _is_v1_request(request: Request) -> bool:
    return request.url.path.startswith("/api/v1")


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", None) or "unknown"


def _check_writable_directory(path: Path) -> dict[str, Any]:
    path.mkdir(parents=True, exist_ok=True)
    marker = path / f".ready-{uuid.uuid4().hex}.tmp"
    marker.write_text("ok", encoding="utf-8")
    marker.unlink(missing_ok=True)
    return {"path": str(path), "writable": True}


def _readiness_payload() -> dict[str, Any]:
    checks = {
        "raw_docs_dir": _check_writable_directory(settings.raw_docs_dir),
        "processed_dir": _check_writable_directory(settings.processed_dir),
        "reports_dir": _check_writable_directory(settings.reports_dir),
        "chroma_persist_dir": _check_writable_directory(settings.chroma_persist_dir),
        "config": {
            "upload_max_bytes": settings.upload_max_bytes,
            "top_k": settings.top_k,
            "tool_max_calls": settings.tool_max_calls,
            "search_max_results": settings.search_max_results,
            "openai_configured": bool(settings.openai_api_key),
            "chroma_collection_name": settings.chroma_collection_name,
        },
    }
    return {"ready": True, "checks": checks}


def create_app() -> FastAPI:
    settings.ensure_directories()
    app = FastAPI(title=settings.app_name, version=settings.app_version)

    @app.middleware("http")
    async def add_request_id(request: Request, call_next: Any) -> Any:
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request.state.request_id = request_id
        token = request_id_context.set(request_id)
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            request_id_context.reset(token)

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(request: Request, exc: StarletteHTTPException) -> Response:
        if not _is_v1_request(request):
            return await http_exception_handler(request, exc)
        message = exc.detail if isinstance(exc.detail, str) else "Request failed"
        return JSONResponse(
            status_code=exc.status_code,
            content=api_error(
                code="http_error",
                message=message,
                details=jsonable_encoder(exc.detail),
                request_id=_request_id(request),
            ),
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_exception(
        request: Request,
        exc: RequestValidationError,
    ) -> Response:
        if not _is_v1_request(request):
            return await request_validation_exception_handler(request, exc)
        return JSONResponse(
            status_code=422,
            content=api_error(
                code="validation_error",
                message="Request validation failed",
                details=jsonable_encoder(exc.errors()),
                request_id=_request_id(request),
            ),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_exception(request: Request, exc: Exception) -> JSONResponse:
        if _is_v1_request(request):
            return JSONResponse(
                status_code=500,
                content=api_error(
                    code="internal_error",
                    message="Internal server error",
                    request_id=_request_id(request),
                ),
            )
        return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})

    @app.get("/health", tags=["system"])
    def health() -> dict[str, str]:
        return {"status": "ok", "app": settings.app_name}

    v1_system_router = APIRouter()

    @v1_system_router.get("/health", tags=["system"])
    def health_v1(request: Request) -> dict[str, Any]:
        return api_success(
            {"status": "ok", "app": settings.app_name, "version": settings.app_version},
            _request_id(request),
        )

    @v1_system_router.get("/ready", tags=["system"])
    def ready_v1(request: Request) -> Any:
        try:
            return api_success(_readiness_payload(), _request_id(request))
        except Exception as exc:
            return JSONResponse(
                status_code=503,
                content=api_error(
                    code="not_ready",
                    message="Readiness checks failed",
                    details={"error": str(exc)},
                    request_id=_request_id(request),
                ),
            )

    app.include_router(upload_router, prefix="/api", tags=["upload"])
    app.include_router(chat_router, prefix="/api", tags=["chat"])
    app.include_router(report_router, prefix="/api", tags=["report"])
    app.include_router(v1_system_router, prefix="/api/v1")
    app.include_router(upload_v1_router, prefix="/api/v1", tags=["upload"])
    app.include_router(chat_v1_router, prefix="/api/v1", tags=["chat"])
    app.include_router(report_v1_router, prefix="/api/v1", tags=["report"])
    return app


app = create_app()
