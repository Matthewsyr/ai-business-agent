from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


BASE_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    app_name: str = "AI Business Analysis Agent"
    app_version: str = "0.1.0"
    base_dir: Path = BASE_DIR
    data_dir: Path = BASE_DIR / "data"
    raw_docs_dir: Path = BASE_DIR / "data" / "raw_docs"
    processed_dir: Path = BASE_DIR / "data" / "processed"
    reports_dir: Path = BASE_DIR / "data" / "reports"
    vector_store_path: Path = BASE_DIR / "data" / "processed" / "vector_store.json"
    sqlite_path: Path = BASE_DIR / "data" / "business.sqlite3"
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "256"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "900"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))
    top_k: int = int(os.getenv("RAG_TOP_K", "5"))
    search_enabled: bool = _bool_env("SEARCH_ENABLED", False)
    search_max_results: int = int(os.getenv("SEARCH_MAX_RESULTS", "5"))

    def ensure_directories(self) -> None:
        for path in (
            self.raw_docs_dir,
            self.processed_dir,
            self.reports_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
