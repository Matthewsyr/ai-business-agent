from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.dependencies import get_agent

router = APIRouter()


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=2)
    use_web: bool = False
    sql_query: str | None = None
    excel_path: str | None = None
    generate_report: bool = False


@router.post("/chat")
def chat(request: ChatRequest) -> dict[str, object]:
    response = get_agent().run(
        question=request.question,
        use_web=request.use_web,
        sql_query=request.sql_query,
        excel_path=request.excel_path,
        generate_report=request.generate_report,
    )
    return asdict(response)
