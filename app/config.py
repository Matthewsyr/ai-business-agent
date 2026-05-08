from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator

try:
    from pydantic_settings import (
        BaseSettings as PydanticBaseSettings,
        SettingsConfigDict as PydanticSettingsConfigDict,
    )
except ImportError:

    def PydanticSettingsConfigDict(**kwargs: Any) -> dict[str, Any]:  # type: ignore[no-redef]
        return kwargs

    class PydanticBaseSettings(BaseModel):  # type: ignore[no-redef]
        """Small import-safe fallback until pydantic-settings is installed."""

        def __init__(self, **data: Any) -> None:
            env_data: dict[str, str] = {}
            for name, field in self.__class__.model_fields.items():
                for env_name in _env_names(name, field):
                    if env_name in os.environ:
                        env_data[name] = os.environ[env_name]
                        break
            env_data.update(data)
            super().__init__(**env_data)


BASE_DIR = Path(__file__).resolve().parents[1]


def _env_names(name: str, field: Any) -> list[str]:
    names: list[str] = []
    alias = getattr(field, "validation_alias", None)
    choices = getattr(alias, "choices", None)
    if isinstance(alias, str):
        names.extend([alias, alias.lower()])
    elif choices:
        for choice in choices:
            if isinstance(choice, str):
                names.extend([choice, choice.lower()])
    names.extend([name, name.upper(), name.lower()])
    return list(dict.fromkeys(names))


class Settings(PydanticBaseSettings):
    model_config = PydanticSettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    app_name: str = "AI Business Analysis Agent"
    app_version: str = "0.1.0"

    base_dir: Path = BASE_DIR
    data_dir: Path = BASE_DIR / "data"
    raw_docs_dir: Path = BASE_DIR / "data" / "raw_docs"
    processed_dir: Path = BASE_DIR / "data" / "processed"
    reports_dir: Path = BASE_DIR / "data" / "reports"
    vector_store_path: Path = BASE_DIR / "data" / "processed" / "vector_store.json"
    sqlite_path: Path = BASE_DIR / "data" / "business.sqlite3"

    chroma_persist_dir: Path = BASE_DIR / "data" / "processed" / "chroma"
    chroma_collection_name: str = "business_documents"
    chroma_host: str | None = None
    chroma_port: int = 8000
    chroma_ssl: bool = False

    openai_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_API_KEY", "LLM_API_KEY", "EMBEDDING_API_KEY"),
    )
    openai_base_url: str | None = Field(
        default="https://api.openai.com/v1",
        validation_alias=AliasChoices("OPENAI_BASE_URL", "LLM_BASE_URL", "EMBEDDING_BASE_URL"),
    )
    openai_model: str | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_MODEL", "CHAT_MODEL", "LLM_MODEL"),
    )
    openai_embedding_model: str | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_EMBEDDING_MODEL", "EMBEDDING_MODEL"),
    )
    llm_timeout_seconds: float = 30.0
    llm_temperature: float = 0.2
    embedding_batch_size: int = 64

    embedding_dim: int = 256
    chunk_size: int = 900
    chunk_overlap: int = 120
    top_k: int = Field(default=5, validation_alias=AliasChoices("TOP_K", "RAG_TOP_K"))

    upload_max_bytes: int = Field(
        default=10 * 1024 * 1024,
        validation_alias=AliasChoices("UPLOAD_MAX_BYTES", "MAX_UPLOAD_BYTES"),
    )

    search_enabled: bool = False
    search_max_results: int = 5
    tool_timeout_seconds: float = 30.0
    tool_max_calls: int = 6
    sql_row_limit: int = 500
    excel_max_rows: int = 10000
    reports_max_files: int = 200

    @field_validator(
        "base_dir",
        "data_dir",
        "raw_docs_dir",
        "processed_dir",
        "reports_dir",
        "vector_store_path",
        "sqlite_path",
        "chroma_persist_dir",
        mode="before",
    )
    @classmethod
    def _coerce_path(cls, value: str | Path) -> Path:
        return Path(value)

    @field_validator(
        "openai_api_key",
        "openai_base_url",
        "openai_model",
        "openai_embedding_model",
        "chroma_host",
        mode="before",
    )
    @classmethod
    def _blank_to_none(cls, value: str | None) -> str | None:
        if value == "":
            return None
        return value

    @model_validator(mode="after")
    def _derive_data_paths(self) -> Settings:
        fields_set = self.model_fields_set
        if "data_dir" in fields_set:
            if "raw_docs_dir" not in fields_set:
                self.raw_docs_dir = self.data_dir / "raw_docs"
            if "processed_dir" not in fields_set:
                self.processed_dir = self.data_dir / "processed"
            if "reports_dir" not in fields_set:
                self.reports_dir = self.data_dir / "reports"

        if "processed_dir" in fields_set or "data_dir" in fields_set:
            if "vector_store_path" not in fields_set:
                self.vector_store_path = self.processed_dir / "vector_store.json"
            if "chroma_persist_dir" not in fields_set:
                self.chroma_persist_dir = self.processed_dir / "chroma"
        return self

    @model_validator(mode="after")
    def _validate_limits(self) -> Settings:
        positive_ints = {
            "embedding_dim": self.embedding_dim,
            "chunk_size": self.chunk_size,
            "top_k": self.top_k,
            "embedding_batch_size": self.embedding_batch_size,
            "upload_max_bytes": self.upload_max_bytes,
            "search_max_results": self.search_max_results,
            "tool_max_calls": self.tool_max_calls,
            "sql_row_limit": self.sql_row_limit,
            "excel_max_rows": self.excel_max_rows,
            "reports_max_files": self.reports_max_files,
            "chroma_port": self.chroma_port,
        }
        for name, value in positive_ints.items():
            if value <= 0:
                raise ValueError(f"{name} must be greater than 0")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be greater than or equal to 0")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.llm_timeout_seconds <= 0 or self.tool_timeout_seconds <= 0:
            raise ValueError("timeouts must be greater than 0")
        if not 0 <= self.llm_temperature <= 2:
            raise ValueError("llm_temperature must be between 0 and 2")
        return self

    def ensure_directories(self) -> None:
        for path in (
            self.raw_docs_dir,
            self.processed_dir,
            self.reports_dir,
            self.chroma_persist_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
