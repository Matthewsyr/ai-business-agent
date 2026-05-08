from __future__ import annotations

from pathlib import Path
from typing import Any

from app.config import Settings


def test_settings_defaults_are_import_safe_without_openai_key(monkeypatch: Any) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    settings = Settings()

    assert settings.openai_api_key is None
    assert settings.openai_model is None
    assert settings.openai_embedding_model is None
    assert settings.upload_max_bytes > 0
    assert settings.top_k > 0


def test_settings_reads_legacy_and_new_environment_names(monkeypatch: Any) -> None:
    monkeypatch.setenv("RAG_TOP_K", "7")
    monkeypatch.setenv("UPLOAD_MAX_BYTES", "12345")
    monkeypatch.setenv("SEARCH_ENABLED", "true")
    monkeypatch.setenv("TOOL_MAX_CALLS", "3")
    monkeypatch.setenv("CHAT_MODEL", "chat-model")
    monkeypatch.setenv("EMBEDDING_MODEL", "embedding-model")

    settings = Settings()

    assert settings.top_k == 7
    assert settings.upload_max_bytes == 12345
    assert settings.search_enabled is True
    assert settings.tool_max_calls == 3
    assert settings.openai_model == "chat-model"
    assert settings.openai_embedding_model == "embedding-model"


def test_data_dir_derives_child_paths(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    settings = Settings()

    assert settings.data_dir == tmp_path
    assert settings.raw_docs_dir == tmp_path / "raw_docs"
    assert settings.processed_dir == tmp_path / "processed"
    assert settings.reports_dir == tmp_path / "reports"
    assert settings.vector_store_path == tmp_path / "processed" / "vector_store.json"
    assert settings.chroma_persist_dir == tmp_path / "processed" / "chroma"
