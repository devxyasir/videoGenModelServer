"""
Celery application — single module imported by both the API and worker processes.
"""
from __future__ import annotations

from celery import Celery

from app.core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "ltx_video",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.workers.video_worker"],
)

celery_app.conf.update(
    # ── Serialization ──────────────────────────────────────────────────────
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # ── Time limits ────────────────────────────────────────────────────────
    task_soft_time_limit=600,    # 10 min soft limit  → raises SoftTimeLimitExceeded
    task_time_limit=660,         # 11 min hard limit  → SIGKILL

    # ── Retry / ack behaviour ──────────────────────────────────────────────
    task_acks_late=True,         # ack only after success / explicit failure
    task_reject_on_worker_lost=True,
    task_max_retries=2,

    # ── Concurrency ────────────────────────────────────────────────────────
    # Controlled per-worker at launch time with --concurrency flag.
    # Default: 1 per GPU worker to avoid VRAM contention.
    worker_prefetch_multiplier=1,   # pull 1 task at a time per worker
    worker_max_tasks_per_child=50,  # restart worker after N tasks (VRAM leak guard)

    # ── Result expiry ──────────────────────────────────────────────────────
    result_expires=86400,  # 24 h

    # ── Routing ────────────────────────────────────────────────────────────
    task_routes={
        "app.workers.video_worker.generate_video": {"queue": "gpu"},
        "app.workers.video_worker.cleanup_files": {"queue": "maintenance"},
    },

    # ── Beat schedule (periodic tasks) ────────────────────────────────────
    beat_schedule={
        "cleanup-old-outputs": {
            "task": "app.workers.video_worker.cleanup_files",
            "schedule": 3600.0,  # every hour
        },
    },
    timezone="UTC",
)
