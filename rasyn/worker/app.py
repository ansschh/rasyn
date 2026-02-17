"""Celery application configuration."""

from __future__ import annotations

import os

from celery import Celery

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

app = Celery(
    "rasyn.worker",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["rasyn.worker.tasks"],
)

app.conf.update(
    # GPU tasks are long-running — only prefetch 1 at a time
    worker_prefetch_multiplier=1,
    # Acknowledge after task completes (not before)
    task_acks_late=True,
    # Reject tasks on worker shutdown so they get requeued
    task_reject_on_worker_lost=True,
    # Serialization
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    # Timezone
    timezone="UTC",
    enable_utc=True,
    # Result expiry
    result_expires=3600,  # 1 hour
)
