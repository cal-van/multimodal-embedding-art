from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from embedding_art.web.services.job_manager import job_manager, Job, JobStatus
from embedding_art.web.sockets import manager

router = APIRouter(prefix="/jobs", tags=["jobs"])


class CreateJobRequest(BaseModel):
    target_text: list[str] = []
    output_modality: str = "image"


class JobResponse(BaseModel):
    id: str
    status: JobStatus
    progress: float
    error: str | None
    logs: list[str]
    result_url: str | None = None


@router.post("/", response_model=JobResponse)
async def create_job(request: CreateJobRequest) -> JobResponse:
    """Create a new optimization job."""
    if not request.target_text:
        raise HTTPException(status_code=400, detail="At least one target text required")
    
    job = job_manager.create_job(request.target_text, request.output_modality)
    # Start immediately for now (Phase 1 mock)
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
        # Construct initial state message
        init_msg = {
            "type": "init",
            "status": job.status,
            "progress": job.progress,
            "logs": job.logs[-50:] if job.logs else [],
            "error": job.error
        }
        try:
            await websocket.send_json(init_msg)
        except Exception as e:
            # Client might have disconnected immediately
            print(f"Failed to send init message: {e}")
            manager.disconnect(websocket, job_id)
            return
        
    try:
        while True:
            # Keep connection alive.
            # In the future, we could accept commands here (e.g. stop job).
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
        logs=job.logs[-5:] if job.logs else [],  # Return last 5 logs for summary
        result_url=job.result_path,
    )
