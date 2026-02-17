"""Analyze API routes — instrument file upload, parsing, interpretation.

Slice 8: POST /analyze/upload, GET /analyze/{task_id}
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse

from rasyn.api.schemas_v2 import AnalysisBatchResult, AnalysisFileResult

logger = logging.getLogger(__name__)
router = APIRouter(tags=["analyze"])

# Upload directory (local storage; S3 in production)
UPLOAD_DIR = Path(os.environ.get("RASYN_UPLOAD_DIR", "/tmp/rasyn-uploads"))


@router.post("/analyze/upload")
async def upload_and_analyze(
    files: list[UploadFile] = File(...),
    expected_product_smiles: str = Form(default=""),
    expected_mw: float | None = Form(default=None),
    sample_id: str = Form(default=""),
):
    """Upload instrument files for automated analysis.

    Accepts LCMS (.mzML, .raw), NMR (.fid, .jdx), HPLC (.csv), and IR files.
    Returns interpretation with conversion, purity, impurities, and anomalies.
    """
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for f in files:
        # Save to disk
        file_id = uuid.uuid4().hex[:8]
        safe_name = f.filename.replace("/", "_").replace("\\", "_") if f.filename else f"upload_{file_id}"
        dest = UPLOAD_DIR / f"{file_id}_{safe_name}"
        content = await f.read()
        dest.write_bytes(content)
        saved_paths.append(str(dest))

    if not saved_paths:
        raise HTTPException(status_code=400, detail="No files uploaded")

    # Run analysis
    try:
        from rasyn.modules.analyze import analyze_batch

        result = analyze_batch(
            saved_paths,
            expected_product_smiles=expected_product_smiles or None,
            expected_mw=expected_mw,
        )

        # Save results to database
        try:
            _save_analysis_results(result.get("files", []), sample_id)
        except Exception as e:
            logger.warning(f"Failed to save analysis to DB: {e}")

        return result

    except Exception as e:
        logger.exception(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)[:200]}")


@router.post("/analyze/interpret")
async def interpret_data(
    instrument: str = Form(...),
    data: str = Form(...),  # JSON string of parsed data
    expected_mw: float | None = Form(default=None),
    expected_product_smiles: str = Form(default=""),
):
    """Interpret pre-parsed instrument data (no file upload needed).

    Useful for re-interpreting data or interpreting data parsed client-side.
    """
    import json

    try:
        parsed_data = json.loads(data)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON data")

    try:
        from rasyn.modules.analyze import interpret_lcms, interpret_hplc, interpret_nmr

        if instrument.upper() == "LCMS":
            result = interpret_lcms(parsed_data, expected_mw=expected_mw)
        elif instrument.upper() == "HPLC":
            result = interpret_hplc(parsed_data)
        elif instrument.upper() == "NMR":
            result = interpret_nmr(parsed_data)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown instrument: {instrument}")

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Interpretation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.get("/analyze/{file_id}")
async def get_analysis(file_id: str):
    """Get analysis results for a specific file."""
    try:
        from rasyn.db.engine import async_session
        from rasyn.db.models import AnalysisFile
        from sqlalchemy import select

        async with async_session() as session:
            result = await session.execute(
                select(AnalysisFile).where(AnalysisFile.file_id == file_id)
            )
            row = result.scalar_one_or_none()

        if not row:
            raise HTTPException(status_code=404, detail="Analysis file not found")

        return {
            "id": row.file_id,
            "filename": row.filename,
            "instrument": row.instrument,
            "sampleId": row.sample_id or "",
            "status": row.status,
            "interpretation": row.interpretation,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to get analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.get("/analyze/sample/{sample_id}")
async def get_sample_analyses(sample_id: str):
    """Get all analysis results for a sample."""
    try:
        from rasyn.db.engine import async_session
        from rasyn.db.models import AnalysisFile
        from sqlalchemy import select

        async with async_session() as session:
            result = await session.execute(
                select(AnalysisFile).where(AnalysisFile.sample_id == sample_id)
            )
            rows = result.scalars().all()

        return {
            "sample_id": sample_id,
            "files": [
                {
                    "id": r.file_id,
                    "filename": r.filename,
                    "instrument": r.instrument,
                    "status": r.status,
                    "interpretation": r.interpretation,
                }
                for r in rows
            ],
            "total": len(rows),
        }

    except Exception as e:
        logger.exception(f"Failed to get sample analyses: {e}")
        raise HTTPException(status_code=500, detail=str(e)[:200])


def _save_analysis_results(files: list[dict], sample_id: str = "") -> None:
    """Save analysis results to database."""
    from sqlalchemy.orm import sessionmaker
    from rasyn.db.engine import sync_engine
    from rasyn.db.models import AnalysisFile

    SessionLocal = sessionmaker(bind=sync_engine)
    session = SessionLocal()
    try:
        for f in files:
            row = AnalysisFile(
                file_id=f.get("id", f"FILE-{uuid.uuid4().hex[:6].upper()}"),
                sample_id=sample_id or f.get("sampleId") or None,
                filename=f.get("filename", ""),
                instrument=f.get("instrument", "Unknown"),
                file_size=0,
                status=f.get("status", "pending"),
                interpretation=f.get("interpretation"),
            )
            session.add(row)
        session.commit()
    except Exception as e:
        session.rollback()
        raise
    finally:
        session.close()
