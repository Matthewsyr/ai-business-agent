from __future__ import annotations

from fastapi import FastAPI

from app.api.chat import router as chat_router
from app.api.report import router as report_router
from app.api.upload import router as upload_router
from app.config import settings


def create_app() -> FastAPI:
    settings.ensure_directories()
    app = FastAPI(title=settings.app_name, version=settings.app_version)

    @app.get("/health", tags=["system"])
    def health() -> dict[str, str]:
        return {"status": "ok", "app": settings.app_name}

    app.include_router(upload_router, prefix="/api", tags=["upload"])
    app.include_router(chat_router, prefix="/api", tags=["chat"])
    app.include_router(report_router, prefix="/api", tags=["report"])
    return app


app = create_app()
