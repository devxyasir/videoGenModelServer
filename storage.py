"""
Storage service.

Handles saving generated videos to local disk and optionally uploading to S3.
Also provides a cleanup task to remove old files.
"""
from __future__ import annotations

import asyncio
import os
import time
import uuid
from pathlib import Path
from typing import Optional, Tuple

import aiofiles

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


def _ensure_dirs() -> None:
    Path(settings.output_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)


_ensure_dirs()


# ─────────────────────────────────────────────────────────────────────────────
# Upload helpers
# ─────────────────────────────────────────────────────────────────────────────

async def save_upload(data: bytes, suffix: str = ".png") -> Tuple[str, str]:
    """Persist an uploaded file. Returns (file_id, abs_path)."""
    file_id = str(uuid.uuid4())
    filename = f"{file_id}{suffix}"
    abs_path = str(Path(settings.upload_dir) / filename)
    async with aiofiles.open(abs_path, "wb") as f:
        await f.write(data)
    logger.info("storage.upload_saved", file_id=file_id, path=abs_path)
    return file_id, abs_path


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────

def output_path_for_job(job_id: str) -> str:
    return str(Path(settings.output_dir) / f"{job_id}.mp4")


async def upload_to_s3(local_path: str, job_id: str) -> Optional[str]:
    """Upload to S3 and return the public URL, or None on failure."""
    if not settings.use_s3:
        return None
    try:
        import boto3

        s3 = boto3.client(
            "s3",
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
        )
        key = f"ltx-videos/{job_id}.mp4"
        await asyncio.to_thread(
            s3.upload_file,
            local_path,
            settings.s3_bucket,
            key,
            ExtraArgs={"ContentType": "video/mp4"},
        )
        url = f"https://{settings.s3_bucket}.s3.{settings.aws_region}.amazonaws.com/{key}"
        logger.info("storage.s3_uploaded", job_id=job_id, url=url)
        return url
    except Exception as exc:
        logger.error("storage.s3_upload_failed", job_id=job_id, error=str(exc))
        return None


def build_local_url(job_id: str, base_url: str) -> str:
    return f"{base_url.rstrip('/')}/api/v1/jobs/{job_id}/download"


# ─────────────────────────────────────────────────────────────────────────────
# Cleanup
# ─────────────────────────────────────────────────────────────────────────────

async def cleanup_old_files() -> int:
    """Delete output videos older than OUTPUT_TTL_HOURS. Returns count deleted."""
    cutoff = time.time() - settings.output_ttl_hours * 3600
    deleted = 0
    output_dir = Path(settings.output_dir)
    for f in output_dir.glob("*.mp4"):
        if f.stat().st_mtime < cutoff:
            try:
                f.unlink()
                deleted += 1
            except OSError:
                pass
    if deleted:
        logger.info("storage.cleanup", deleted=deleted)
    return deleted
