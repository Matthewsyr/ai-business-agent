from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import settings
from app.dependencies import get_retriever
from rag.loader import SUPPORTED_SUFFIXES

router = APIRouter()


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)) -> dict[str, object]:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Supported: {sorted(SUPPORTED_SUFFIXES)}",
        )

    settings.ensure_directories()
    safe_name = Path(file.filename or "uploaded.txt").name
    target_path = settings.raw_docs_dir / safe_name
    with target_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    retriever = get_retriever()
    ingest_result = retriever.ingest_file(target_path)
    return {
        "filename": safe_name,
        "path": str(target_path),
        "chunks_added": ingest_result["chunks_added"],
        "documents_loaded": ingest_result["documents_loaded"],
    }
