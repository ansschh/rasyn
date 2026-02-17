"""SQLAlchemy engine configuration for async (API) and sync (worker) usage."""

from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+asyncpg://rasyn:rasyn@localhost:5432/rasyn",
)

# Async engine for FastAPI (asyncpg)
async_engine = create_async_engine(
    DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    echo=False,
)

async_session = sessionmaker(
    async_engine, class_=AsyncSession, expire_on_commit=False
)

# Sync engine for Celery worker (psycopg2)
_sync_url = DATABASE_URL.replace("+asyncpg", "+psycopg2")
SYNC_DATABASE_URL = os.environ.get("DATABASE_URL_SYNC", _sync_url)

sync_engine = create_engine(
    SYNC_DATABASE_URL,
    pool_size=2,
    max_overflow=5,
    echo=False,
)
