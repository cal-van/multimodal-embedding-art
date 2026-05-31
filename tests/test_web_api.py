import pytest
from fastapi.testclient import TestClient

from embedding_art.web.app import app
from embedding_art.web.services.job_manager import job_manager


@pytest.fixture
def client(monkeypatch):
    # Clear jobs before each test
    job_manager._jobs.clear()

    async def mock_start_job(job_id: str) -> None:
        pass

    monkeypatch.setattr(job_manager, "start_job", mock_start_job)
    return TestClient(app)


def test_cors_allowed_origin(client):
    """Test that allowed origins work."""
    response = client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert response.headers["access-control-allow-credentials"] == "true"


def test_cors_forbidden_origin(client):
    """Test that disallowed origins are not reflected."""
    response = client.get("/health", headers={"Origin": "http://evil.com"})
    assert response.status_code == 200
    # Should NOT have the allow-origin header matching the request
    assert (
        "access-control-allow-origin" not in response.headers
        or response.headers["access-control-allow-origin"] != "http://evil.com"
    )


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "embedding-art"}


def test_create_job(client):
    response = client.post(
        "/jobs/", json={"target_text": ["test_concept"], "output_modality": "image"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert data["status"] in ["queued", "running", "completed"]
    assert data["progress"] >= 0.0


def test_create_job_invalid_input(client):
    response = client.post("/jobs/", json={"target_text": []})
    assert response.status_code == 400


def test_list_jobs(client):
    # Create a job first
    client.post("/jobs/", json={"target_text": ["list_test"]})

    response = client.get("/jobs/")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 1


def test_get_job(client):
    # Create a job
    create_res = client.post("/jobs/", json={"target_text": ["get_test"]})
    job_id = create_res.json()["id"]

    response = client.get(f"/jobs/{job_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == job_id


def test_get_nonexistent_job(client):
    response = client.get("/jobs/non-existent-id")
    assert response.status_code == 404
