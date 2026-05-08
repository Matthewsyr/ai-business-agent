from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.api.chat import _response_to_dict
from app.api.schemas import api_success
from app.config import settings
from app.dependencies import get_agent, get_request_id

router = APIRouter()
v1_router = APIRouter()


class ReportRequest(BaseModel):
    topic: str = Field(..., min_length=2)
    use_web: bool = False
    sql_query: str | None = None
    excel_path: str | None = None


def _create_report(request: ReportRequest) -> dict[str, Any]:
    response = get_agent().run(
        question=request.topic,
        use_web=request.use_web,
        sql_query=request.sql_query,
        excel_path=request.excel_path,
        generate_report=True,
    )
    return _response_to_dict(response)


def _list_reports() -> dict[str, list[str]]:
    settings.ensure_directories()
    files = sorted(
        str(path)
        for path in settings.reports_dir.glob("*")
        if path.suffix.lower() in {".md", ".docx"}
    )
    return {"reports": files[: settings.reports_max_files]}


@router.post("/report")
def create_report(request: ReportRequest) -> dict[str, Any]:
    return _create_report(request)


@router.get("/reports")
def list_reports() -> dict[str, list[str]]:
    return _list_reports()


@v1_router.post("/report")
def create_report_v1(request: ReportRequest) -> dict[str, Any]:
    return api_success(_create_report(request), get_request_id())


@v1_router.get("/reports")
def list_reports_v1() -> dict[str, Any]:
    return api_success(_list_reports(), get_request_id())
