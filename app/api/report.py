from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.config import settings
from app.dependencies import get_agent

router = APIRouter()


class ReportRequest(BaseModel):
    topic: str = Field(..., min_length=2)
    use_web: bool = False
    sql_query: str | None = None
    excel_path: str | None = None


@router.post("/report")
def create_report(request: ReportRequest) -> dict[str, object]:
    response = get_agent().run(
        question=request.topic,
        use_web=request.use_web,
        sql_query=request.sql_query,
        excel_path=request.excel_path,
        generate_report=True,
    )
    return asdict(response)


@router.get("/reports")
def list_reports() -> dict[str, list[str]]:
    settings.ensure_directories()
    files = sorted(
        str(path)
        for path in settings.reports_dir.glob("*")
        if path.suffix.lower() in {".md", ".docx"}
    )
    return {"reports": files}
