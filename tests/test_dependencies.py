from __future__ import annotations

from pathlib import Path
from typing import Any

from app import dependencies


def configure_temp_settings(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.setattr(dependencies.settings, "base_dir", tmp_path)
    monkeypatch.setattr(dependencies.settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(dependencies.settings, "raw_docs_dir", tmp_path / "data" / "raw_docs")
    monkeypatch.setattr(dependencies.settings, "processed_dir", tmp_path / "data" / "processed")
    monkeypatch.setattr(dependencies.settings, "reports_dir", tmp_path / "data" / "reports")
    monkeypatch.setattr(
        dependencies.settings,
        "vector_store_path",
        tmp_path / "data" / "processed" / "vector_store.json",
    )
    monkeypatch.setattr(
        dependencies.settings, "sqlite_path", tmp_path / "data" / "business.sqlite3"
    )
    monkeypatch.setattr(dependencies.settings, "chroma_persist_dir", tmp_path / "data" / "chroma")
    monkeypatch.setattr(dependencies.settings, "openai_api_key", None)
    monkeypatch.setattr(dependencies.settings, "openai_model", None)
    monkeypatch.setattr(dependencies.settings, "openai_embedding_model", None)
    dependencies.get_retriever.cache_clear()
    dependencies.get_agent.cache_clear()


def test_default_dependencies_use_local_fallback_stack(monkeypatch: Any, tmp_path: Path) -> None:
    configure_temp_settings(monkeypatch, tmp_path)

    retriever = dependencies.get_retriever()
    agent = dependencies.get_agent()

    assert retriever.search("anything") == []
    response = agent.run("What should we analyze?")
    assert response.question == "What should we analyze?"
    assert response.sources == []


def test_explicit_fallback_classes_are_structured(tmp_path: Path) -> None:
    missing_file = tmp_path / "missing.txt"
    existing_file = tmp_path / "doc.txt"
    existing_file.write_text("hello", encoding="utf-8")

    retriever = dependencies.FallbackRetriever()
    agent = dependencies.FallbackAgent()

    assert retriever.ingest_file(missing_file)["documents_loaded"] == 0
    assert retriever.ingest_file(existing_file)["documents_loaded"] == 1
    assert retriever.search("question") == []

    response = agent.run("question")
    assert response.answer == "Analysis services are not available in this runtime."
    assert response.metrics["task_completion_rate"] == 0.0
