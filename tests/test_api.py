from fastapi.testclient import TestClient

from app.main import app


def test_health_and_reports_routes() -> None:
    client = TestClient(app)

    health = client.get("/health")
    reports = client.get("/api/reports")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert reports.status_code == 200
    assert "reports" in reports.json()
