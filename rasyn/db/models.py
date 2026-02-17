"""SQLAlchemy ORM models for the Chemistry OS."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Jobs & Events (Slice 1)
# ---------------------------------------------------------------------------

class Job(Base):
    __tablename__ = "jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    smiles = Column(Text, nullable=False)
    job_type = Column(String(32), nullable=False, default="RETRO_PLAN")
    status = Column(
        Enum("queued", "running", "completed", "failed", name="job_status"),
        nullable=False,
        default="queued",
    )
    result = Column(JSONB, nullable=True)
    config = Column(JSONB, nullable=True)  # top_k, models, constraints
    error = Column(Text, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    events = relationship("JobEvent", back_populates="job", order_by="JobEvent.id")
    routes = relationship("Route", back_populates="job", order_by="Route.rank")


class JobEvent(Base):
    __tablename__ = "job_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(
        UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False
    )
    kind = Column(String(64), nullable=False)
    data = Column(JSONB, nullable=True)
    ts = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    job = relationship("Job", back_populates="events")

    __table_args__ = (Index("ix_job_events_job_id", "job_id"),)


# ---------------------------------------------------------------------------
# Routes & Reactions (Slice 3 — multi-step planning)
# ---------------------------------------------------------------------------

class Route(Base):
    __tablename__ = "routes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(
        UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False
    )
    route_id = Column(String(64), nullable=False)
    rank = Column(Integer, nullable=False, default=1)
    tree = Column(JSONB, nullable=False)  # Full route tree as JSON
    score = Column(Float, nullable=True)
    score_breakdown = Column(JSONB, nullable=True)
    num_steps = Column(Integer, nullable=True)
    all_purchasable = Column(Boolean, default=False)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    job = relationship("Job", back_populates="routes")
    reactions = relationship("Reaction", back_populates="route", order_by="Reaction.step_number")


class Reaction(Base):
    __tablename__ = "reactions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    route_id = Column(Integer, ForeignKey("routes.id", ondelete="CASCADE"), nullable=True)
    step_number = Column(Integer, nullable=True)
    product_smiles = Column(Text, nullable=False)
    reactants_smiles = Column(Text, nullable=False)  # dot-separated
    reaction_smiles = Column(Text, nullable=True)     # reactants>>product
    reaction_class = Column(String(64), nullable=True)
    conditions = Column(JSONB, nullable=True)
    model_source = Column(String(32), nullable=True)
    confidence = Column(Float, nullable=True)
    roundtrip_score = Column(Float, nullable=True)
    outcome = Column(String(32), nullable=True)  # success/failure/null
    actual_yield = Column(Float, nullable=True)
    failure_reason = Column(Text, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    route = relationship("Route", back_populates="reactions")


# ---------------------------------------------------------------------------
# Sourcing Cache (Slice 5)
# ---------------------------------------------------------------------------

class SourcingCache(Base):
    __tablename__ = "sourcing_cache"

    id = Column(Integer, primary_key=True, autoincrement=True)
    smiles = Column(Text, nullable=False, index=True)
    vendor = Column(String(64), nullable=False)
    catalog_number = Column(String(128), nullable=True)
    price = Column(String(64), nullable=True)
    pack_size = Column(String(64), nullable=True)
    available = Column(Boolean, default=False)
    lead_time = Column(String(64), nullable=True)
    url = Column(Text, nullable=True)
    queried_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        Index("ix_sourcing_cache_smiles", "smiles"),
    )


# ---------------------------------------------------------------------------
# Experiments & Samples (Slice 7 — Execute)
# ---------------------------------------------------------------------------

class Experiment(Base):
    __tablename__ = "experiments"

    id = Column(String(32), primary_key=True)  # EXP-YYMMDD-XXXX
    job_id = Column(
        UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="SET NULL"), nullable=True
    )
    route_id = Column(String(64), nullable=True)
    step_number = Column(Integer, nullable=True)
    product_smiles = Column(Text, nullable=True)
    reaction_name = Column(String(128), nullable=True)
    scale = Column(String(32), default="0.5 mmol")
    protocol = Column(JSONB, nullable=True)  # Full protocol data
    reagents = Column(JSONB, nullable=True)  # Reagent table
    workup = Column(JSONB, nullable=True)  # Workup checklist
    status = Column(String(32), default="planned")  # planned, in_progress, completed, failed
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    samples = relationship("Sample", back_populates="experiment", order_by="Sample.id")


class Sample(Base):
    __tablename__ = "samples"

    id = Column(String(32), primary_key=True)  # RSN-YYMMDD-XXXX
    experiment_id = Column(
        String(32), ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False
    )
    label = Column(String(128), nullable=True)
    sample_type = Column(String(32), nullable=True)  # reaction_mixture, crude, purified, fraction
    planned_analysis = Column(JSONB, nullable=True)  # ["LCMS", "HPLC", "1H NMR"]
    status = Column(String(32), default="pending")  # pending, submitted, analyzed
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    experiment = relationship("Experiment", back_populates="samples")
    analysis_files = relationship("AnalysisFile", back_populates="sample")


# ---------------------------------------------------------------------------
# Analysis Files (Slice 8 — Analyze)
# ---------------------------------------------------------------------------

class AnalysisFile(Base):
    __tablename__ = "analysis_files"

    id = Column(Integer, primary_key=True, autoincrement=True)
    file_id = Column(String(32), nullable=False, unique=True)  # FILE-XXXXXX
    sample_id = Column(
        String(32), ForeignKey("samples.id", ondelete="SET NULL"), nullable=True
    )
    filename = Column(Text, nullable=False)
    instrument = Column(String(16), nullable=False)  # LCMS, HPLC, NMR, IR
    file_path = Column(Text, nullable=True)  # S3 or local path
    file_size = Column(Integer, nullable=True)
    status = Column(String(16), default="pending")  # pending, interpreted, anomaly
    interpretation = Column(JSONB, nullable=True)  # Full AnalysisResult
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    sample = relationship("Sample", back_populates="analysis_files")
