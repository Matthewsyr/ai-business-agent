from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from app.api import upload as upload_api
from app.api.chat import _response_to_dict
from app.config import settings
from app.main import app


class StubRetriever:
    def ingest_file(self, path: Path) -> dict[str, Any]:
        return {
            "documents_loaded": 1,
            "chunks_created": 1,
            "chunks_added": 1,
            "vector_store_size": 1,
        }


class ModelDumpResponse:
    def model_dump(self) -> dict[str, str]:
        return {"answer": "model"}


def test_health_and_reports_routes_preserve_legacy_bodies() -> None:
    client = TestClient(app)

    health = client.get("/health")
    reports = client.get("/api/reports")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert reports.status_code == 200
    assert "reports" in reports.json()
    assert "success" not in reports.json()


def test_v1_health_ready_and_reports_use_response_envelope() -> None:
    client = TestClient(app)

    health = client.get("/api/v1/health", headers={"X-Request-ID": "test-request"})
    ready = client.get("/api/v1/ready")
    reports = client.get("/api/v1/reports")

    assert health.status_code == 200
    assert health.headers["X-Request-ID"] == "test-request"
    assert health.json()["success"] is True
    assert health.json()["request_id"] == "test-request"
    assert health.json()["data"]["status"] == "ok"

    assert ready.status_code == 200
    assert ready.json()["success"] is True
    assert ready.json()["data"]["ready"] is True
    assert "checks" in ready.json()["data"]

    assert reports.status_code == 200
    assert reports.json()["success"] is True
    assert "reports" in reports.json()["data"]


def test_v1_validation_errors_use_response_envelope() -> None:
    client = TestClient(app)

    response = client.post("/api/v1/chat", json={"question": ""})

    assert response.status_code == 422
    body = response.json()
    assert body["success"] is False
    assert body["error"]["code"] == "validation_error"
    assert body["request_id"]


def test_v1_not_found_errors_use_response_envelope() -> None:
    client = TestClient(app)

    response = client.get("/api/v1/missing")

    assert response.status_code == 404
    body = response.json()
    assert body["success"] is False
    assert body["error"]["code"] == "http_error"


def test_legacy_validation_errors_keep_detail_body() -> None:
    client = TestClient(app)

    response = client.post("/api/chat", json={"question": ""})

    assert response.status_code == 422
    assert "detail" in response.json()
    assert "success" not in response.json()


def test_response_to_dict_handles_common_response_shapes() -> None:
    assert _response_to_dict({"answer": "dict"}) == {"answer": "dict"}
    assert _response_to_dict(ModelDumpResponse()) == {"answer": "model"}


def test_upload_v1_uses_envelope_and_keeps_legacy_upload_body(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(settings, "raw_docs_dir", tmp_path / "raw_docs")
    monkeypatch.setattr(settings, "processed_dir", tmp_path / "processed")
    monkeypatch.setattr(settings, "reports_dir", tmp_path / "reports")
    monkeypatch.setattr(settings, "chroma_persist_dir", tmp_path / "chroma")
    monkeypatch.setattr(upload_api, "get_retriever", lambda: StubRetriever())

    client = TestClient(app)

    legacy = client.post(
        "/api/upload",
        files={"file": ("sample.txt", b"hello", "text/plain")},
    )
    v1 = client.post(
        "/api/v1/upload",
        files={"file": ("sample-v1.txt", b"hello", "text/plain")},
    )

    assert legacy.status_code == 200
    assert legacy.json()["filename"] == "sample.txt"
    assert "success" not in legacy.json()

    assert v1.status_code == 200
    assert v1.json()["success"] is True
    assert v1.json()["data"]["filename"] == "sample-v1.txt"
