from fastapi.testclient import TestClient

from backend.config import PROJECT_ROOT
from backend.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "project": "PolyGen AI",
    }


def test_generation_status():
    response = client.get("/generation/status")

    assert response.status_code == 200
    assert response.json()["available"] is False
    assert response.json()["model"] == "shap-e"


def test_generate_unavailable():
    response = client.post(
        "/generate",
        json={"prompt": "A medieval sword"},
    )

    assert response.status_code == 503
    assert "not available" in response.json()["detail"]


def test_generate_endpoint_rejects_invalid_prompt():
    response = client.post(
        "/generate",
        json={"prompt": "Hi"},
    )

    assert response.status_code == 422


