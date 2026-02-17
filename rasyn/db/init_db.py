"""Database initialization — create tables on startup."""

from __future__ import annotations

import logging

from rasyn.db.engine import async_engine
from rasyn.db.models import Base

logger = logging.getLogger(__name__)


async def init_db() -> None:
    """Create all tables if they don't exist."""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables initialized.")
