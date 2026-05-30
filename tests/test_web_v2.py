"""
Tests for the v2 Web UI API.

Covers:
- encoder_name field in CreateJobRequest schema
- default encoder fallback to "imagebind"
- encoder_name in JobResponse
- compare endpoint schema validation
- similarity_weight and feature_matching_weight in CreateJobRequest
- engine initialization via create_default_registry / EmbeddingArtEngine.from_registry
- progress callback sends loss breakdown components

All tests use FastAPI's TestClient and mock out the engine/job_manager so no
optimization actually runs.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
from fastapi.testclient import TestClient

from embedding_art.web.app import app
from embedding_art.web.services.job_manager import job_manager


@pytest.fixture(autouse=True)
def clear_jobs():
    """Reset the job store before every test."""
    job_manager._jobs.clear()
    yield
    job_manager._jobs.clear()


@pytest.fixture
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_render_result() -> MagicMock:
    """Build a fake RenderResult that looks like a real one."""
    result = MagicMock()
    result.output = torch.zeros(1, 3, 64, 64)
    result.final_similarity = 0.9
    result.history = MagicMock()
    result.history.final_similarity = 0.9
    return result


# ---------------------------------------------------------------------------
# 1. CreateJobRequest schema — encoder_name field
# ---------------------------------------------------------------------------


def test_create_job_with_encoder_name(client):
    """encoder_name is accepted and stored on the job."""
    with patch.object(job_manager, "start_job") as mock_start:
        mock_start.return_value = None  # skip actual execution
        response = client.post(
            "/jobs/",
            json={
                "target_text": ["ocean"],
                "output_modality": "image",
                "encoder_name": "siglip2-so400m",
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["encoder_name"] == "siglip2-so400m"


# ---------------------------------------------------------------------------
# 2. CreateJobRequest schema — default encoder is "imagebind"
# ---------------------------------------------------------------------------


def test_create_job_default_encoder(client):
    """When encoder_name is omitted, the response reports the v3 canonical
    encoder (``languagebind``)."""
    with patch.object(job_manager, "start_job") as mock_start:
        mock_start.return_value = None
        response = client.post(
            "/jobs/",
            json={"target_text": ["ocean"]},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["encoder_name"] == "languagebind"


# ---------------------------------------------------------------------------
# 3. JobResponse includes encoder_name
# ---------------------------------------------------------------------------


def test_job_response_includes_encoder(client):
    """GET /jobs/{id} includes encoder_name in the response body."""
    with patch.object(job_manager, "start_job") as mock_start:
        mock_start.return_value = None
        create_res = client.post(
            "/jobs/",
            json={"target_text": ["river"], "encoder_name": "clap-general"},
        )

    job_id = create_res.json()["id"]
    response = client.get(f"/jobs/{job_id}")

    assert response.status_code == 200
    assert response.json()["encoder_name"] == "clap-general"


# ---------------------------------------------------------------------------
# 4. Compare endpoint — schema accepts a list of encoder names
# ---------------------------------------------------------------------------


def test_compare_endpoint_accepts_encoder_list(client):
    """POST /jobs/compare validates encoder_names and returns a job."""
    with patch.object(job_manager, "start_job") as mock_start:
        mock_start.return_value = None
        response = client.post(
            "/jobs/compare",
            json={
                "target_text": "forest",
                "output_modality": "image",
                "encoder_names": ["imagebind", "siglip2-so400m"],
                "steps": 10,
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert data["status"] in ("queued", "running", "completed")


def test_compare_endpoint_requires_at_least_two_encoders(client):
    """compare endpoint rejects requests with fewer than 2 encoder names."""
    response = client.post(
        "/jobs/compare",
        json={
            "target_text": "forest",
            "encoder_names": ["imagebind"],  # only one — invalid
            "steps": 10,
        },
    )
    assert response.status_code == 422


def test_compare_endpoint_requires_encoder_names(client):
    """compare endpoint returns 422 when encoder_names is missing."""
    response = client.post(
        "/jobs/compare",
        json={"target_text": "forest", "steps": 10},
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# 5. CreateJobRequest — similarity_weight and feature_matching_weight
# ---------------------------------------------------------------------------


def test_create_job_with_loss_weights(client):
    """similarity_weight and feature_matching_weight are accepted and round-trip."""
    with patch.object(job_manager, "start_job") as mock_start:
        mock_start.return_value = None
        response = client.post(
            "/jobs/",
            json={
                "target_text": ["goldfish"],
                "similarity_weight": 2.0,
                "feature_matching_weight": 0.1,
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["similarity_weight"] == pytest.approx(2.0)
    assert data["feature_matching_weight"] == pytest.approx(0.1)


def test_create_job_default_loss_weights(client):
    """Default loss weights are 1.0 / 0.5 when omitted from request."""
    with patch.object(job_manager, "start_job") as mock_start:
        mock_start.return_value = None
        response = client.post(
            "/jobs/",
            json={"target_text": ["flamingo"]},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["similarity_weight"] == pytest.approx(1.0)
    assert data["feature_matching_weight"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 6. Engine init uses create_default_registry + from_registry
# ---------------------------------------------------------------------------


def test_get_engine_uses_registry():
    """get_engine() builds the engine via create_default_registry + from_registry."""
    # Reset the global engine so get_engine() re-initialises.
    import embedding_art.web.services.job_manager as jm

    original = jm._engine
    jm._engine = None

    try:
        mock_registry = MagicMock()
        mock_encoder = MagicMock()
        mock_registry.load.return_value = mock_encoder

        mock_engine = MagicMock()

        with (
            patch(
                "embedding_art.web.services.job_manager.create_default_registry",
                return_value=mock_registry,
            ) as mock_create_registry,
            patch(
                "embedding_art.core.engine.EmbeddingArtEngine.from_registry",
                return_value=mock_engine,
            ) as mock_from_registry,
        ):
            engine = jm.get_engine()

        mock_create_registry.assert_called_once()
        mock_from_registry.assert_called_once()
        # Verify from_registry was called with our registry and languagebind as default
        call_kwargs = mock_from_registry.call_args
        assert call_kwargs[1].get("default_encoder") == "languagebind" or (
            len(call_kwargs[0]) >= 2 and call_kwargs[0][1] == "languagebind"
        )
        assert engine is mock_engine
    finally:
        jm._engine = original


# ---------------------------------------------------------------------------
# 7. Job execution uses engine.render() and sends breakdown in progress messages
# ---------------------------------------------------------------------------


def test_job_execution_uses_render_and_broadcasts_breakdown(tmp_path, monkeypatch):
    """_run_job_sync calls engine.render() and broadcasts loss breakdown components."""
    import asyncio

    import embedding_art.web.services.job_manager as jm

    # Arrange: build a job with known properties
    job = jm.Job(
        id="test-job-id",
        target_text=["whale"],
        output_modality="image",
        encoder_name="imagebind",
        similarity_weight=1.0,
        feature_matching_weight=0.5,
    )

    # Build a fake RenderResult
    render_result = _make_render_result()

    # Patch the engine so render() is a no-op that returns our fake result
    mock_engine = MagicMock()
    mock_engine.render.return_value = render_result

    # Capture broadcast calls
    broadcast_calls: list[dict] = []

    def fake_broadcast(msg: dict):
        broadcast_calls.append(msg)

    # Monkeypatch get_engine
    monkeypatch.setattr(jm, "get_engine", lambda: mock_engine)

    # Monkeypatch outputs directory to a temp dir
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir(exist_ok=True)

    # Run the job synchronously (no real event loop needed for the broadcast mock)
    loop = asyncio.new_event_loop()
    try:
        # Patch asyncio.run_coroutine_threadsafe to capture messages synchronously
        import asyncio as asyncio_mod

        def fake_run_coro(coro, _loop):
            # Drain the coroutine — extract the message from broadcast_to_job
            # by calling the real coroutine on our loop
            future = loop.run_until_complete(coro)
            return future

        with patch.object(asyncio_mod, "run_coroutine_threadsafe", side_effect=fake_run_coro):
            jm._run_job_sync_direct(job, loop, fake_broadcast)
    finally:
        loop.close()

    # Assert engine.render was called
    mock_engine.render.assert_called_once()
    call_kwargs = mock_engine.render.call_args
    # encoder_name must be forwarded
    assert call_kwargs[1].get("encoder_name") == "imagebind"

    # Assert the job completed successfully
    assert job.status == jm.JobStatus.COMPLETED


def test_progress_callback_message_includes_components():
    """The progress callback sends 'components' dict with named loss terms."""
    import embedding_art.web.services.job_manager as jm

    captured: list[dict] = []

    def fake_broadcast(msg: dict):
        captured.append(msg)

    # Build a breakdown
    from embedding_art.core.render_result import LossBreakdown

    breakdown = LossBreakdown(
        total=torch.tensor(-0.82),
        components={
            "similarity": torch.tensor(-0.85),
            "feature_matching": torch.tensor(0.03),
        },
    )

    # Call the helper that builds progress messages
    jm._broadcast_progress(
        broadcast=fake_broadcast,
        job_id="abc",
        step=5,
        total_steps=10,
        breakdown=breakdown,
    )

    assert len(captured) == 1
    msg = captured[0]
    assert msg["type"] == "progress"
    assert msg["progress"] == pytest.approx(0.5)  # step=5, total=10
    assert msg["step"] == 5
    assert "loss" in msg
    assert "components" in msg
    assert "similarity" in msg["components"]
    assert "feature_matching" in msg["components"]
    assert msg["components"]["similarity"] == pytest.approx(-0.85)
    assert msg["components"]["feature_matching"] == pytest.approx(0.03)


# ---------------------------------------------------------------------------
# 8. Backward compatibility — old requests without new fields still work
# ---------------------------------------------------------------------------


def test_old_request_without_encoder_name_still_works(client):
    """Requests that omit encoder_name/loss weights remain valid.

    Default flips to ``languagebind`` post-v3 but the field stays
    optional so the request body itself remains backward-compatible.
    """
    with patch.object(job_manager, "start_job") as mock_start:
        mock_start.return_value = None
        response = client.post(
            "/jobs/",
            json={"target_text": ["jellyfish"], "output_modality": "image"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["encoder_name"] == "languagebind"
    assert data["similarity_weight"] == pytest.approx(1.0)
    assert data["feature_matching_weight"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 9. Showcase endpoint (v3)
# ---------------------------------------------------------------------------


class TestShowcaseEndpoint:
    """POST /jobs/showcase mirrors the ``embed-art showcase`` CLI."""

    def test_minimal_request_is_accepted(self, client):
        with patch.object(job_manager, "start_job") as mock_start:
            mock_start.return_value = None
            response = client.post(
                "/jobs/showcase",
                json={"target_text": "thunder"},
            )

        assert response.status_code == 200, response.text
        data = response.json()
        assert data["kind"] == "showcase"
        assert data["encoder_name"] == "languagebind"
        # Default modalities are all four, default tracks is honest only.
        assert set(data["modalities"]) == {"image", "audio", "video", "text"}
        assert data["tracks"] == ["honest"]

    def test_dual_track_request_is_accepted(self, client):
        with patch.object(job_manager, "start_job") as mock_start:
            mock_start.return_value = None
            response = client.post(
                "/jobs/showcase",
                json={
                    "target_text": "ocean",
                    "tracks": ["honest", "natural"],
                    "modalities": ["image"],
                    "image_backbone": "sd35",
                    "autocast_dtype": "bf16",
                    "steps": 50,
                    "seed": 7,
                },
            )

        assert response.status_code == 200, response.text
        data = response.json()
        assert data["tracks"] == ["honest", "natural"]
        assert data["modalities"] == ["image"]

    def test_empty_target_text_rejected(self, client):
        response = client.post(
            "/jobs/showcase",
            json={"target_text": "   "},
        )
        assert response.status_code == 400

    def test_invalid_track_rejected_at_schema_level(self, client):
        response = client.post(
            "/jobs/showcase",
            json={"target_text": "thunder", "tracks": ["bogus"]},
        )
        assert response.status_code == 422

    def test_realism_request_is_accepted_and_surfaced(self, client):
        with patch.object(job_manager, "start_job") as mock_start:
            mock_start.return_value = None
            response = client.post(
                "/jobs/showcase",
                json={"target_text": "thunder", "realism": 0.3, "modalities": ["image"]},
            )
        assert response.status_code == 200, response.text
        assert response.json()["realism"] == 0.3

    def test_realism_out_of_range_rejected(self, client):
        for bad in (2.0, -0.1):
            response = client.post(
                "/jobs/showcase",
                json={"target_text": "thunder", "realism": bad},
            )
            assert response.status_code == 422, f"realism={bad} should 422"

    def test_realism_with_explicit_tracks_rejected(self, client):
        response = client.post(
            "/jobs/showcase",
            json={"target_text": "thunder", "realism": 0.5, "tracks": ["honest", "natural"]},
        )
        assert response.status_code == 422

    def test_invalid_compile_mode_rejected(self, client):
        response = client.post(
            "/jobs/showcase",
            json={"target_text": "thunder", "compile_mode": "rm -rf"},
        )
        assert response.status_code == 422

    def test_invalid_autocast_dtype_rejected(self, client):
        response = client.post(
            "/jobs/showcase",
            json={"target_text": "thunder", "autocast_dtype": "float128"},
        )
        assert response.status_code == 422

    def test_sae_path_traversal_rejected(self, client):
        response = client.post(
            "/jobs/showcase",
            json={"target_text": "thunder", "sae_path": "../../etc/passwd"},
        )
        assert response.status_code == 422

    def test_compile_mode_and_realism_forwarded_to_impl(self):
        """Regression guard: the bug where _run_showcase_job dropped compile_mode.
        Asserts compile_mode AND realism reach _showcase_impl."""
        from embedding_art.web.services import job_manager as jm

        job = job_manager.create_showcase_job(
            target_text="goldfish",
            modalities=["text"],
            compile_mode="reduce-overhead",
            realism=0.3,
        )
        # _run_showcase_job does a local `from ...showcase import _showcase_impl`,
        # so patch it at the source module, not on job_manager.
        with patch("embedding_art.cli.commands.showcase._showcase_impl") as mock_impl:
            jm._run_showcase_job(job, broadcast=lambda _msg: None)
        assert mock_impl.call_count == 1
        kwargs = mock_impl.call_args.kwargs
        assert kwargs["compile_mode"] == "reduce-overhead"
        assert kwargs["realism"] == 0.3

    def test_get_returns_manifest_url_when_complete(self, client):
        """``GET /jobs/{id}`` reports manifest_url after the showcase runs."""
        with patch.object(job_manager, "start_job") as mock_start:
            mock_start.return_value = None
            create_res = client.post(
                "/jobs/showcase",
                json={"target_text": "thunder"},
            )
        job_id = create_res.json()["id"]

        # Mutate the in-memory job to simulate completion (so we don't
        # have to actually run the showcase).
        job = job_manager.get_job(job_id)
        assert job is not None
        job.manifest_path = f"/outputs/showcase/{job_id}/manifest.json"
        job.result_path = job.manifest_path

        get_res = client.get(f"/jobs/{job_id}")
        assert get_res.status_code == 200
        data = get_res.json()
        assert data["manifest_url"] == job.manifest_path
        assert data["result_url"] == job.manifest_path
        assert data["kind"] == "showcase"
