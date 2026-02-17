"""Redis pub/sub helper for SSE streaming of job events."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import AsyncIterator

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")


def _get_channel(job_id: str) -> str:
    return f"job:{job_id}"


async def get_redis() -> aioredis.Redis:
    """Get an async Redis connection."""
    return aioredis.from_url(REDIS_URL, decode_responses=True)


def publish_event_sync(job_id: str, kind: str, message: str = "", data: dict | None = None) -> None:
    """Publish an event synchronously (for use in Celery worker)."""
    import redis as sync_redis

    r = sync_redis.from_url(REDIS_URL, decode_responses=True)
    event = {
        "kind": kind,
        "message": message,
        "data": data or {},
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    r.publish(_get_channel(str(job_id)), json.dumps(event))
    r.close()


async def publish_event(job_id: str, kind: str, message: str = "", data: dict | None = None) -> None:
    """Publish an event asynchronously."""
    r = await get_redis()
    event = {
        "kind": kind,
        "message": message,
        "data": data or {},
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    await r.publish(_get_channel(str(job_id)), json.dumps(event))
    await r.aclose()


async def subscribe_events(job_id: str) -> AsyncIterator[str]:
    """Subscribe to job events and yield SSE-formatted strings.

    Yields lines like:
        event: model_running
        data: {"kind": "model_running", "message": "...", "data": {...}}

    """
    r = await get_redis()
    pubsub = r.pubsub()
    channel = _get_channel(str(job_id))
    await pubsub.subscribe(channel)

    try:
        async for msg in pubsub.listen():
            if msg["type"] != "message":
                continue
            raw = msg["data"]
            try:
                event = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue

            kind = event.get("kind", "info")
            yield f"event: {kind}\ndata: {json.dumps(event)}\n\n"

            # Terminal events — stop streaming
            if kind in ("completed", "failed"):
                break
    finally:
        await pubsub.unsubscribe(channel)
        await pubsub.aclose()
        await r.aclose()
