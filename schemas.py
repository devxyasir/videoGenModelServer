from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ── Auth ──────────────────────────────────────────────────────────────────────

class TokenRequest(BaseModel):
    api_key: str = Field(..., description="Static API key to exchange for a JWT")


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds


# ── Job Submission ─────────────────────────────────────────────────────────────

class GenerationRequest(BaseModel):
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description=(
            "Detailed chronological description of motion and scene. "
            "Start with the action, be literal and cinematic."
        ),
        examples=["A majestic eagle soaring above snow-capped mountains, "
                  "sunlight glinting off its wings, slow cinematic pan."],
    )
    negative_prompt: Optional[str] = Field(
        default="worst quality, inconsistent motion, blurry, jittery, distorted",
        max_length=300,
    )
    height: int = Field(default=512, ge=64, le=1024, multiple_of=32)
    width: int = Field(default=768, ge=64, le=1024, multiple_of=32)
    num_frames: int = Field(default=25, ge=9, le=257, description="Must be 8k+1 (9,17,25,33…)")
    fps: int = Field(default=8, ge=1, le=30)
    guidance_scale: float = Field(default=3.0, ge=1.0, le=20.0)
    num_inference_steps: int = Field(default=50, ge=10, le=100)
    seed: Optional[int] = Field(default=None, ge=0)

    @field_validator("num_frames")
    @classmethod
    def validate_num_frames(cls, v: int) -> int:
        # LTX requires (N-1) % 8 == 0
        if (v - 1) % 8 != 0:
            raise ValueError("num_frames must satisfy (n-1) % 8 == 0  (e.g. 9, 17, 25, 33, 49…)")
        return v


class ImageToVideoRequest(GenerationRequest):
    """Submitted as multipart/form-data — image file handled separately."""
    pass


class TextToVideoRequest(GenerationRequest):
    pass


# ── Job Status ────────────────────────────────────────────────────────────────

class JobStatusResponse(BaseModel):
    job_id: str
    task_id: Optional[str]
    status: str
    job_type: str
    prompt: str
    height: int
    width: int
    num_frames: int
    fps: int
    seed: Optional[int]
    output_url: Optional[str]
    error_message: Optional[str]
    duration_seconds: Optional[float]
    queue_position: Optional[int]
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime]

    model_config = {"from_attributes": True}


class JobSubmittedResponse(BaseModel):
    job_id: str
    task_id: str
    status: str
    message: str
    estimated_wait_seconds: Optional[int] = None
    poll_url: str


class JobListResponse(BaseModel):
    jobs: list[JobStatusResponse]
    total: int
    page: int
    page_size: int


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "error"]
    version: str
    model_loaded: bool
    gpu_available: bool
    queue_depth: int
    active_workers: int
    redis_connected: bool
    uptime_seconds: float
