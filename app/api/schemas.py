from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ApiError(BaseModel):
    code: str
    message: str
    details: Any = None


class ApiResponse(BaseModel):
    success: bool
    data: Any = None
    error: ApiError | None = None
    request_id: str = Field(..., min_length=1)


def api_success(data: Any, request_id: str) -> dict[str, Any]:
    return ApiResponse(success=True, data=data, request_id=request_id).model_dump()


def api_error(
    code: str,
    message: str,
    request_id: str,
    details: Any = None,
) -> dict[str, Any]:
    return ApiResponse(
        success=False,
        error=ApiError(code=code, message=message, details=details),
        request_id=request_id,
    ).model_dump()
