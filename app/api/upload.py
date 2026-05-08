from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.api.schemas import api_success
from app.config import settings
from app.dependencies import get_request_id, get_retriever

try:
    from rag.loader import SUPPORTED_SUFFIXES
except ImportError:
    SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf", ".docx"}


router = APIRouter()
v1_router = APIRouter()


def _validate_suffix(filename: str | None) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Supported: {sorted(SUPPORTED_SUFFIXES)}",
        )
    return suffix


async def _save_upload(file: UploadFile) -> Path:
    settings.ensure_directories()
    safe_name = Path(file.filename or "uploaded.txt").name
    target_path = settings.raw_docs_dir / safe_name
    bytes_written = 0

    try:
        with target_path.open("wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                bytes_written += len(chunk)
                if bytes_written > settings.upload_max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Upload exceeds maximum size of {settings.upload_max_bytes} bytes",
                    )
                buffer.write(chunk)
    except Exception:
        if target_path.exists():
            target_path.unlink(missing_ok=True)
        raise

    return target_path


async def _upload_document(file: UploadFile) -> dict[str, Any]:
    _validate_suffix(file.filename)
    target_path = await _save_upload(file)
    ingest_result = get_retriever().ingest_file(target_path)
    return {
        "filename": target_path.name,
        "path": str(target_path),
        "chunks_added": ingest_result["chunks_added"],
        "documents_loaded": ingest_result["documents_loaded"],
    }


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)) -> dict[str, Any]:
    return await _upload_document(file)


@v1_router.post("/upload")
async def upload_document_v1(file: UploadFile = File(...)) -> dict[str, Any]:
    return api_success(await _upload_document(file), get_request_id())
