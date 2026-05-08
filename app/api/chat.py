from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, cast

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.api.schemas import api_success
from app.dependencies import get_agent, get_request_id

router = APIRouter()
v1_router = APIRouter()


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=2)
    use_web: bool = False
    sql_query: str | None = None
    excel_path: str | None = None
    generate_report: bool = False


def _response_to_dict(response: Any) -> dict[str, Any]:
    if is_dataclass(response):
        return asdict(cast(Any, response))
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if isinstance(response, dict):
        return response
    return dict(response)


def _chat(request: ChatRequest) -> dict[str, Any]:
    response = get_agent().run(
        question=request.question,
        use_web=request.use_web,
        sql_query=request.sql_query,
        excel_path=request.excel_path,
        generate_report=request.generate_report,
    )
    return _response_to_dict(response)


@router.post("/chat")
def chat(request: ChatRequest) -> dict[str, Any]:
    return _chat(request)


@v1_router.post("/chat")
def chat_v1(request: ChatRequest) -> dict[str, Any]:
    return api_success(_chat(request), get_request_id())
