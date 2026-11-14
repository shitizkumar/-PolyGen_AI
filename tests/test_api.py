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


def test_generate_endpoint_success(monkeypatch):
    expected_path = PROJECT_ROOT / "artifacts" / "test_model.obj"

    def mock_generate_mesh(prompt):
        assert prompt == "A medieval sword"
        return expected_path

    monkeypatch.setattr(
        "backend.main.generate_mesh",
        mock_generate_mesh,
    )

    response = client.post(
        "/generate",
        json={"prompt": "A medieval sword"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    assert response.json()["model_path"] == "artifacts/test_model.obj"


def test_generate_endpoint_rejects_invalid_prompt():
    response = client.post(
        "/generate",
        json={"prompt": "Hi"},
    )

    assert response.status_code == 422


def test_generate_endpoint_handles_generation_error(monkeypatch):
    def mock_generate_mesh(prompt):
        raise RuntimeError("Model generation failed")

    monkeypatch.setattr(
        "backend.main.generate_mesh",
        mock_generate_mesh,
    )

    response = client.post(
        "/generate",
        json={"prompt": "A medieval sword"},
    )

    assert response.status_code == 500
    assert "3D generation failed" in response.json()["detail"]
