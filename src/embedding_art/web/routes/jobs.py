from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, field_validator

from embedding_art.web.services.job_manager import Job, JobStatus, job_manager
from embedding_art.web.sockets import manager

router = APIRouter(prefix="/jobs", tags=["jobs"])


class CreateJobRequest(BaseModel):
    target_text: list[str] = []
    output_modality: str = "image"
    encoder_name: str = "languagebind"
    steps: int = 500
    learning_rate: float = 0.1
    seed: int | None = None
    similarity_weight: float = 1.0
    feature_matching_weight: float = 0.5


class CreateShowcaseRequest(BaseModel):
    """v3 four-modality showcase request.

    Mirrors the ``embed-art showcase`` CLI surface.
    """

    target_text: str
    modalities: list[str] = ["image", "audio", "video", "text"]
    encoder_name: str = "languagebind"
    image_backbone: str = "sd35"
    audio_backbone: str = "stable-audio-open"
    video_backbone: str = "ltx-video"
    tracks: list[str] = ["honest"]
    autocast_dtype: str = "fp32"
    interpret: bool = True
    evaluate: bool = True
    sae_path: str | None = None
    steps: int = 200
    seed: int | None = None

    @field_validator("tracks")
    @classmethod
    def _validate_tracks(cls, v: list[str]) -> list[str]:
        valid = {"honest", "natural"}
        bad = [t for t in v if t not in valid]
        if bad:
            raise ValueError(f"Unknown tracks {bad}; valid tracks: {sorted(valid)}")
        if not v:
            raise ValueError("At least one track required")
        return v


class CompareJobRequest(BaseModel):
    target_text: str
    target_weight: float = 1.0
    output_modality: str = "image"
    encoder_names: list[str]
    steps: int = 500

    @field_validator("encoder_names")
    @classmethod
    def _require_at_least_two_encoders(cls, v: list[str]) -> list[str]:
        if len(v) < 2:
            raise ValueError("encoder_names must contain at least 2 encoder names")
        return v


class JobResponse(BaseModel):
    id: str
    status: JobStatus
    progress: float
    error: str | None
    logs: list[str]
    result_url: str | None = None
    encoder_name: str = "languagebind"
    similarity_weight: float = 1.0
    feature_matching_weight: float = 0.5
    kind: str = "single"
    manifest_url: str | None = None
    tracks: list[str] = []
    modalities: list[str] = []
    target_text: list[str] = []
    output_modality: str = "image"


@router.post("/", response_model=JobResponse)
async def create_job(request: CreateJobRequest) -> JobResponse:
    """Create a new optimization job."""
    if not request.target_text:
        raise HTTPException(status_code=400, detail="At least one target text required")

    job = job_manager.create_job(
        target_text=request.target_text,
        output_modality=request.output_modality,
        encoder_name=request.encoder_name,
        similarity_weight=request.similarity_weight,
        feature_matching_weight=request.feature_matching_weight,
    )
    await job_manager.start_job(job.id)

    return _map_job_to_response(job)


@router.post("/compare", response_model=JobResponse)
async def create_compare_job(request: CompareJobRequest) -> JobResponse:
    """Create a comparison job that renders the same concept with multiple encoders."""
    job = job_manager.create_compare_job(
        target_text=request.target_text,
        output_modality=request.output_modality,
        encoder_names=request.encoder_names,
        steps=request.steps,
    )
    await job_manager.start_job(job.id)

    return _map_job_to_response(job)


@router.post("/showcase", response_model=JobResponse)
async def create_showcase_job(request: CreateShowcaseRequest) -> JobResponse:
    """Create a v3 four-modality showcase job.

    Renders the target concept across the requested modalities in a single
    shared encoder space and writes a manifest summarising the bundle.
    """
    if not request.target_text.strip():
        raise HTTPException(status_code=400, detail="target_text must not be empty")

    job = job_manager.create_showcase_job(
        target_text=request.target_text,
        modalities=request.modalities,
        encoder_name=request.encoder_name,
        image_backbone=request.image_backbone,
        audio_backbone=request.audio_backbone,
        video_backbone=request.video_backbone,
        tracks=request.tracks,
        autocast_dtype=request.autocast_dtype,
        interpret=request.interpret,
        evaluate=request.evaluate,
        sae_path=request.sae_path,
        steps=request.steps,
        seed=request.seed,
    )
    await job_manager.start_job(job.id)
    return _map_job_to_response(job)


@router.get("/", response_model=list[JobResponse])
async def list_jobs() -> list[JobResponse]:
    """List all jobs."""
    jobs = job_manager.list_jobs()
    return [_map_job_to_response(j) for j in jobs]


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(job_id: str) -> JobResponse:
    """Get job details."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return _map_job_to_response(job)


@router.websocket("/{job_id}/ws")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for job updates."""
    await manager.connect(websocket, job_id)

    # Send current state immediately
    job = job_manager.get_job(job_id)
    if job:
        init_msg = {
            "type": "init",
            "status": job.status,
            "progress": job.progress,
            "logs": job.logs[-50:] if job.logs else [],
            "error": job.error,
        }
        try:
            await websocket.send_json(init_msg)
        except Exception as e:
            print(f"Failed to send init message: {e}")
            manager.disconnect(websocket, job_id)
            return

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, job_id)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket, job_id)


def _map_job_to_response(job: Job) -> JobResponse:
    return JobResponse(
        id=job.id,
        status=job.status,
        progress=job.progress,
        error=job.error,
        logs=job.logs[-5:] if job.logs else [],
        result_url=job.result_path,
        encoder_name=job.encoder_name,
        similarity_weight=job.similarity_weight,
        feature_matching_weight=job.feature_matching_weight,
        kind=job.kind,
        manifest_url=job.manifest_path,
        tracks=job.tracks if job.kind == "showcase" else [],
        modalities=job.modalities if job.kind == "showcase" else [],
        target_text=job.target_text,
        output_modality=job.output_modality,
    )
