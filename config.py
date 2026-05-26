from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Server ────────────────────────────────────────────────────────────────
    app_env: Literal["development", "production"] = "production"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_workers: int = 1
    log_level: str = "info"

    # ── Security ──────────────────────────────────────────────────────────────
    secret_key: str = "CHANGE_ME"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440
    api_keys: str = ""  # comma-separated

    @property
    def api_keys_list(self) -> List[str]:
        return [k.strip() for k in self.api_keys.split(",") if k.strip()]

    # ── Redis / Celery ─────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # ── Model ─────────────────────────────────────────────────────────────────
    model_repo: str = "Symphone/ltx-video-2b-v0.9-fp8"
    model_variant: Literal["fp8", "mixed"] = "fp8"
    hf_token: str = ""
    model_cache_dir: str = "./model_cache"
    device: str = "cuda"
    torch_dtype: str = "bfloat16"

    # ── Inference Defaults ────────────────────────────────────────────────────
    default_height: int = 512
    default_width: int = 768
    default_num_frames: int = 25
    default_fps: int = 8
    default_guidance_scale: float = 3.0
    default_num_inference_steps: int = 50
    max_concurrent_jobs: int = 2

    # ── Storage ───────────────────────────────────────────────────────────────
    output_dir: str = "./storage/outputs"
    upload_dir: str = "./storage/uploads"
    max_upload_size_mb: int = 20
    output_ttl_hours: int = 24

    # ── Optional S3 ───────────────────────────────────────────────────────────
    use_s3: bool = False
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"
    s3_bucket: str = ""

    # ── Rate Limiting ─────────────────────────────────────────────────────────
    rate_limit_per_minute: int = 10
    rate_limit_per_hour: int = 100

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = "sqlite+aiosqlite:///./ltx_jobs.db"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
