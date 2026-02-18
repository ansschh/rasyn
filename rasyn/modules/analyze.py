"""Analyze module — instrument file parsing, interpretation, anomaly detection.

Slice 8: Auto-ingest LCMS/HPLC/NMR data and generate interpretations.

Parsers gracefully degrade if optional dependencies (pyOpenMS, nmrglue) are
not installed — they return structured data from CSV/mzML fallbacks.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# File type detection
# ---------------------------------------------------------------------------

INSTRUMENT_EXTENSIONS = {
    ".mzml": "LCMS",
    ".mzxml": "LCMS",
    ".raw": "LCMS",
    ".d": "LCMS",
    ".cdf": "LCMS",
    ".fid": "NMR",
    ".jdx": "NMR",
    ".dx": "NMR",
    ".csv": "HPLC",  # Default for CSV; could also be LCMS
    ".txt": "HPLC",
    ".spc": "IR",
    ".spa": "IR",
}


def detect_instrument(filename: str) -> str:
    """Detect instrument type from file extension."""
    ext = Path(filename).suffix.lower()
    return INSTRUMENT_EXTENSIONS.get(ext, "Unknown")


# ---------------------------------------------------------------------------
# LCMS parser
# ---------------------------------------------------------------------------

def parse_lcms(filepath: str) -> dict:
    """Parse LCMS data from mzML or CSV.

    Returns:
        {spectra: [{rt, mz_list, intensity_list}], chromatogram: {rt[], tic[]}, metadata: {}}
    """
    ext = Path(filepath).suffix.lower()

    if ext in (".mzml", ".mzxml"):
        return _parse_lcms_mzml(filepath)
    elif ext in (".csv", ".txt"):
        return _parse_lcms_csv(filepath)
    else:
        return _parse_lcms_generic(filepath)


def _parse_lcms_mzml(filepath: str) -> dict:
    """Parse mzML using pyOpenMS if available."""
    try:
        from pyopenms import MSExperiment, MzMLFile

        exp = MSExperiment()
        MzMLFile().load(filepath, exp)

        spectra = []
        tic_rt, tic_int = [], []

        for spec in exp:
            rt = spec.getRT()
            mz, intensity = spec.get_peaks()
            spectra.append({
                "rt": round(rt, 2),
                "mz_list": mz.tolist()[:100],  # Limit for JSON size
                "intensity_list": intensity.tolist()[:100],
                "ms_level": spec.getMSLevel(),
            })
            tic_rt.append(round(rt, 2))
            tic_int.append(float(intensity.sum()))

        return {
            "spectra": spectra[:50],  # Limit total spectra
            "chromatogram": {"rt": tic_rt, "tic": tic_int},
            "metadata": {"num_spectra": len(spectra), "format": "mzML"},
        }

    except ImportError:
        logger.warning("pyOpenMS not available — returning minimal LCMS parse")
        return {"spectra": [], "chromatogram": {"rt": [], "tic": []},
                "metadata": {"format": "mzML", "error": "pyOpenMS not installed"}}


def _parse_lcms_csv(filepath: str) -> dict:
    """Parse LCMS data from a CSV file."""
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return {"spectra": [], "chromatogram": {"rt": [], "tic": []}, "metadata": {}}

    headers = {h.lower().strip() for h in rows[0].keys()}

    # Try to find RT and intensity columns
    rt_col = next((h for h in rows[0].keys() if "rt" in h.lower() or "time" in h.lower()), None)
    int_col = next((h for h in rows[0].keys() if "int" in h.lower() or "abs" in h.lower() or "tic" in h.lower()), None)
    mz_col = next((h for h in rows[0].keys() if "mz" in h.lower() or "mass" in h.lower()), None)

    spectra = []
    rt_list, tic_list = [], []

    for row in rows[:1000]:
        rt = float(row[rt_col]) if rt_col and row.get(rt_col) else 0
        intensity = float(row[int_col]) if int_col and row.get(int_col) else 0
        rt_list.append(round(rt, 2))
        tic_list.append(intensity)

        if mz_col and row.get(mz_col):
            spectra.append({
                "rt": round(rt, 2),
                "mz_list": [float(row[mz_col])],
                "intensity_list": [intensity],
            })

    return {
        "spectra": spectra[:50],
        "chromatogram": {"rt": rt_list, "tic": tic_list},
        "metadata": {"num_rows": len(rows), "format": "CSV"},
    }


def _parse_lcms_generic(filepath: str) -> dict:
    """Fallback parser for unknown LCMS formats."""
    file_size = os.path.getsize(filepath) if os.path.exists(filepath) else 0
    return {
        "spectra": [],
        "chromatogram": {"rt": [], "tic": []},
        "metadata": {
            "format": Path(filepath).suffix,
            "file_size": file_size,
            "note": "Binary format — requires vendor software or pyOpenMS",
        },
    }


# ---------------------------------------------------------------------------
# NMR parser
# ---------------------------------------------------------------------------

def parse_nmr(filepath: str) -> dict:
    """Parse NMR data (Bruker .fid, JCAMP-DX .jdx, etc.)."""
    ext = Path(filepath).suffix.lower()

    if ext in (".jdx", ".dx"):
        return _parse_nmr_jdx(filepath)
    elif ext == ".fid" or os.path.isdir(filepath):
        return _parse_nmr_bruker(filepath)
    else:
        return {"ppm": [], "intensity": [], "peaks": [],
                "metadata": {"format": ext, "note": "Unsupported NMR format"}}


def _parse_nmr_jdx(filepath: str) -> dict:
    """Parse JCAMP-DX NMR files."""
    ppm = []
    intensity = []
    metadata = {}

    with open(filepath, "r") as f:
        data_section = False
        for line in f:
            line = line.strip()
            if line.startswith("##"):
                key_val = line[2:].split("=", 1)
                if len(key_val) == 2:
                    key, val = key_val[0].strip(), key_val[1].strip()
                    metadata[key] = val
                    if key == "XYDATA" or key == "DATA TABLE":
                        data_section = True
                        continue
                    elif key.startswith("END"):
                        data_section = False
            elif data_section and line:
                parts = line.split()
                try:
                    if len(parts) >= 2:
                        ppm.append(float(parts[0]))
                        intensity.append(float(parts[1]))
                except ValueError:
                    continue

    # Peak picking (simple threshold)
    peaks = _pick_peaks_simple(ppm, intensity)

    return {
        "ppm": ppm[:10000],
        "intensity": intensity[:10000],
        "peaks": peaks,
        "metadata": metadata,
    }


def _parse_nmr_bruker(filepath: str) -> dict:
    """Parse Bruker NMR data using nmrglue if available."""
    try:
        import nmrglue as ng
        import numpy as np

        if os.path.isdir(filepath):
            dic, data = ng.bruker.read(filepath)
        else:
            dic, data = ng.bruker.read(os.path.dirname(filepath))

        # Process: zero-fill + FFT + autophase
        data = ng.proc_base.zf_size(data, 32768)
        data = ng.proc_base.fft(data)
        try:
            data = ng.proc_autophase.autops(data, "acme")
        except Exception:
            pass

        real_data = data.real
        ppm = np.linspace(12, -2, len(real_data)).tolist()
        peaks = _pick_peaks_simple(ppm, real_data.tolist())

        return {
            "ppm": ppm,
            "intensity": real_data.tolist(),
            "peaks": peaks,
            "metadata": {"format": "Bruker", "num_points": len(data)},
        }

    except ImportError:
        logger.warning("nmrglue not available — returning minimal NMR parse")
        return {"ppm": [], "intensity": [], "peaks": [],
                "metadata": {"format": "Bruker", "error": "nmrglue not installed"}}


def _pick_peaks_simple(ppm: list[float], intensity: list[float],
                        threshold_pct: float = 5.0) -> list[dict]:
    """Simple peak picking by threshold."""
    if not intensity:
        return []

    max_int = max(abs(v) for v in intensity) if intensity else 1
    threshold = max_int * (threshold_pct / 100.0)
    peaks = []

    for i in range(1, len(intensity) - 1):
        if (abs(intensity[i]) > threshold and
                abs(intensity[i]) > abs(intensity[i-1]) and
                abs(intensity[i]) > abs(intensity[i+1])):
            peaks.append({
                "ppm": round(ppm[i], 2) if i < len(ppm) else 0,
                "intensity": round(intensity[i], 1),
            })

    # Sort by chemical shift
    peaks.sort(key=lambda p: p["ppm"], reverse=True)
    return peaks[:50]


# ---------------------------------------------------------------------------
# HPLC parser
# ---------------------------------------------------------------------------

def parse_hplc(filepath: str) -> dict:
    """Parse HPLC data from CSV/TXT."""
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return {"retention_time": [], "absorbance": [], "peaks": [], "metadata": {}}

    rt_col = next((h for h in rows[0].keys() if "rt" in h.lower() or "time" in h.lower()), None)
    abs_col = next((h for h in rows[0].keys()
                     if "abs" in h.lower() or "au" in h.lower() or "int" in h.lower()), None)

    rt_list, abs_list = [], []
    for row in rows[:5000]:
        try:
            rt = float(row[rt_col]) if rt_col and row.get(rt_col) else 0
            ab = float(row[abs_col]) if abs_col and row.get(abs_col) else 0
            rt_list.append(round(rt, 3))
            abs_list.append(round(ab, 4))
        except (ValueError, TypeError):
            continue

    peaks = _detect_hplc_peaks(rt_list, abs_list)

    return {
        "retention_time": rt_list,
        "absorbance": abs_list,
        "peaks": peaks,
        "metadata": {"num_points": len(rt_list), "format": "CSV"},
    }


def _detect_hplc_peaks(rt: list[float], absorbance: list[float],
                         min_height_pct: float = 2.0) -> list[dict]:
    """Detect peaks in HPLC chromatogram."""
    if not absorbance:
        return []

    max_abs = max(absorbance)
    threshold = max_abs * (min_height_pct / 100.0)
    peaks = []

    for i in range(2, len(absorbance) - 2):
        if (absorbance[i] > threshold and
                absorbance[i] > absorbance[i-1] and
                absorbance[i] > absorbance[i+1] and
                absorbance[i] > absorbance[i-2] and
                absorbance[i] > absorbance[i+2]):
            area_pct = round((absorbance[i] / max_abs) * 100, 1) if max_abs > 0 else 0
            peaks.append({
                "retention_time": rt[i] if i < len(rt) else 0,
                "height": round(absorbance[i], 4),
                "area_percent": area_pct,
            })

    peaks.sort(key=lambda p: p["height"], reverse=True)
    return peaks[:20]


# ---------------------------------------------------------------------------
# Automated interpretation (rule-based)
# ---------------------------------------------------------------------------

def interpret_lcms(
    parsed_data: dict,
    expected_mw: float | None = None,
    expected_product_smiles: str | None = None,
) -> dict:
    """Interpret LCMS data using rule-based analysis.

    Checks for [M+H]+ ion, estimates conversion, identifies impurities.
    """
    spectra = parsed_data.get("spectra", [])
    product_found = False
    product_rt = None
    impurities = []
    anomalies = []

    if expected_mw:
        # Check for [M+H]+ peak
        mh_target = expected_mw + 1.008
        for spec in spectra:
            for mz in spec.get("mz_list", []):
                if abs(mz - mh_target) < 0.5:
                    product_found = True
                    product_rt = spec.get("rt")
                    break

        if not product_found:
            # Check [M+Na]+ adduct
            mna_target = expected_mw + 22.99
            for spec in spectra:
                for mz in spec.get("mz_list", []):
                    if abs(mz - mna_target) < 0.5:
                        product_found = True
                        product_rt = spec.get("rt")
                        break

    # Estimate conversion from TIC
    chromo = parsed_data.get("chromatogram", {})
    tic = chromo.get("tic", [])
    conversion = 0.0
    if tic:
        max_tic = max(tic) if tic else 1
        # Cannot determine conversion from MS alone — requires quantitative calibration
        conversion = None

    # Count major peaks as potential impurities
    major_peaks = 0
    for spec in spectra[:10]:
        mz_list = spec.get("mz_list", [])
        int_list = spec.get("intensity_list", [])
        if int_list:
            max_int = max(int_list)
            for j, intensity in enumerate(int_list):
                if intensity > max_int * 0.1 and j < len(mz_list):
                    mz = mz_list[j]
                    if expected_mw and abs(mz - (expected_mw + 1.008)) > 1.0:
                        major_peaks += 1

    if major_peaks > 3:
        impurities.append({"identity": "Multiple unidentified peaks", "percentage": 5.0})

    # Anomaly detection
    if not product_found and expected_mw:
        anomalies.append("Expected product mass not detected — check reaction or vial identity")
    if major_peaks > 5:
        anomalies.append("Multiple major peaks detected — possible side reactions")

    purity = max(0, 100.0 - sum(imp["percentage"] for imp in impurities))

    return {
        "conversion": round(conversion, 1),
        "purity": round(purity, 1),
        "majorProductConfirmed": product_found,
        "impurities": impurities,
        "anomalies": anomalies,
        "product_rt": product_rt,
        "summary": _generate_summary(conversion, purity, product_found, impurities, anomalies),
    }


def interpret_hplc(
    parsed_data: dict,
    expected_rt: float | None = None,
) -> dict:
    """Interpret HPLC data."""
    peaks = parsed_data.get("peaks", [])

    if not peaks:
        return {
            "conversion": 0, "purity": 0, "majorProductConfirmed": False,
            "impurities": [], "anomalies": ["No peaks detected in HPLC trace"],
            "summary": "No peaks detected — check injection or detector settings.",
        }

    # Largest peak = presumed product
    main_peak = peaks[0]
    total_area = sum(p.get("area_percent", 0) for p in peaks)
    product_area = main_peak.get("area_percent", 0)

    purity = round(product_area, 1) if total_area > 0 else 0.0
    # HPLC purity is the area% of the main peak — NOT conversion
    # Conversion requires a reference standard; we only have area%
    conversion = None  # Cannot determine without calibration

    impurities = []
    for p in peaks[1:5]:  # Top impurities
        if p.get("area_percent", 0) > 1.0:
            impurities.append({
                "identity": f"Peak at RT={p['retention_time']:.2f} min",
                "percentage": round(p["area_percent"], 1),
            })

    anomalies = []
    if purity < 80:
        anomalies.append("Low purity detected — consider re-purification")
    if len(peaks) > 10:
        anomalies.append("Complex mixture — multiple significant peaks")

    return {
        "conversion": conversion,
        "purity": purity,
        "majorProductConfirmed": purity > 50,
        "impurities": impurities,
        "anomalies": anomalies,
        "summary": _generate_summary(conversion, purity, purity > 50, impurities, anomalies),
    }


def interpret_nmr(
    parsed_data: dict,
    expected_peaks: list[float] | None = None,
) -> dict:
    """Interpret NMR data."""
    peaks = parsed_data.get("peaks", [])

    product_confirmed = False
    if expected_peaks and peaks:
        # Check if expected peaks are present (within ±0.1 ppm)
        matched = 0
        for exp_ppm in expected_peaks:
            for obs in peaks:
                if abs(obs["ppm"] - exp_ppm) < 0.1:
                    matched += 1
                    break
        product_confirmed = matched >= len(expected_peaks) * 0.6

    anomalies = []
    if not peaks:
        anomalies.append("No peaks detected — check sample concentration or acquisition")

    # Look for common issues
    solvent_peaks = {7.26: "CDCl3", 2.50: "DMSO-d6", 3.31: "MeOD", 4.79: "D2O"}
    detected_solvents = []
    for obs in peaks:
        for sol_ppm, sol_name in solvent_peaks.items():
            if abs(obs["ppm"] - sol_ppm) < 0.05:
                detected_solvents.append(sol_name)

    summary_parts = []
    if peaks:
        summary_parts.append(f"{len(peaks)} peaks detected")
    if detected_solvents:
        summary_parts.append(f"Solvent: {', '.join(set(detected_solvents))}")
    if product_confirmed:
        summary_parts.append("Expected peaks matched — product confirmed")

    return {
        "conversion": 0,  # NMR doesn't directly measure conversion
        "purity": 0,  # Would need integration
        "majorProductConfirmed": product_confirmed,
        "impurities": [],
        "anomalies": anomalies,
        "num_peaks": len(peaks),
        "detected_solvents": list(set(detected_solvents)),
        "summary": ". ".join(summary_parts) if summary_parts else "NMR data parsed — manual review recommended.",
    }


def _generate_summary(
    conversion: float, purity: float, product_found: bool,
    impurities: list[dict], anomalies: list[str],
) -> str:
    """Generate a human-readable summary."""
    parts = []

    if product_found:
        parts.append(f"Product confirmed. {conversion:.1f}% conversion, {purity:.1f}% purity.")
    else:
        parts.append("Product NOT confirmed in analytical data.")

    if impurities:
        imp_str = "; ".join(f"{imp['identity']} ({imp['percentage']:.1f}%)" for imp in impurities[:3])
        parts.append(f"Impurities: {imp_str}.")

    if anomalies:
        parts.append(f"Anomalies: {'; '.join(anomalies[:2])}.")

    if not anomalies and product_found and purity > 90:
        parts.append("Reaction successful — proceed to next step.")
    elif not product_found:
        parts.append("Recommend repeating reaction or troubleshooting conditions.")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# LLM interpretation (optional enrichment)
# ---------------------------------------------------------------------------

async def interpret_with_llm(
    parsed_data: dict,
    rule_interpretation: dict,
    instrument_type: str = "LCMS",
    expected_product: str = "",
) -> str:
    """Enrich rule-based interpretation with LLM summary."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return rule_interpretation.get("summary", "")

    try:
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=api_key)

        prompt = f"""You are an expert analytical chemist interpreting {instrument_type} data.

Expected product: {expected_product}
Rule-based interpretation:
- Conversion: {rule_interpretation.get('conversion', 'N/A')}%
- Purity: {rule_interpretation.get('purity', 'N/A')}%
- Product confirmed: {rule_interpretation.get('majorProductConfirmed', False)}
- Impurities: {json.dumps(rule_interpretation.get('impurities', []))}
- Anomalies: {json.dumps(rule_interpretation.get('anomalies', []))}

Provide a concise 2-3 sentence interpretation for a synthetic chemist.
Focus on: is the reaction successful? What are the main concerns? What should the chemist do next?"""

        message = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    except Exception as e:
        logger.warning(f"LLM interpretation failed: {e}")
        return rule_interpretation.get("summary", "")


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------

def analyze_file(
    filepath: str,
    instrument_type: str | None = None,
    expected_mw: float | None = None,
    expected_product_smiles: str | None = None,
    sample_id: str | None = None,
) -> dict:
    """Full analysis pipeline for an instrument file.

    1. Detect instrument type
    2. Parse raw data
    3. Interpret with rules
    4. Return structured result

    Returns an AnalysisResult-compatible dict.
    """
    filename = os.path.basename(filepath)
    if instrument_type is None:
        instrument_type = detect_instrument(filename)

    # Calculate expected MW if SMILES provided
    if expected_product_smiles and expected_mw is None:
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors
            mol = Chem.MolFromSmiles(expected_product_smiles)
            if mol:
                expected_mw = Descriptors.ExactMolWt(mol)
        except ImportError:
            pass

    # Parse
    if instrument_type == "LCMS":
        parsed = parse_lcms(filepath)
        interp = interpret_lcms(parsed, expected_mw=expected_mw)
    elif instrument_type == "HPLC":
        parsed = parse_hplc(filepath)
        interp = interpret_hplc(parsed)
    elif instrument_type == "NMR":
        parsed = parse_nmr(filepath)
        interp = interpret_nmr(parsed)
    else:
        parsed = {}
        interp = {
            "conversion": 0, "purity": 0, "majorProductConfirmed": False,
            "impurities": [], "anomalies": [f"Unsupported instrument type: {instrument_type}"],
            "summary": f"Cannot interpret {instrument_type} data automatically.",
        }

    # Determine status
    has_anomalies = len(interp.get("anomalies", [])) > 0
    status = "anomaly" if has_anomalies else "interpreted"

    file_size = os.path.getsize(filepath) if os.path.exists(filepath) else 0

    return {
        "id": f"FILE-{uuid.uuid4().hex[:6].upper()}",
        "filename": filename,
        "instrument": instrument_type,
        "sampleId": sample_id or "",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "fileSize": _format_size(file_size),
        "status": status,
        "interpretation": {
            "conversion": interp.get("conversion", 0),
            "purity": interp.get("purity", 0),
            "majorProductConfirmed": interp.get("majorProductConfirmed", False),
            "impurities": interp.get("impurities", []),
            "anomalies": interp.get("anomalies", []),
            "summary": interp.get("summary", ""),
        },
        "raw_data_summary": {
            "num_spectra": parsed.get("metadata", {}).get("num_spectra", 0),
            "num_peaks": len(parsed.get("peaks", [])),
        },
    }


def _format_size(size_bytes: int) -> str:
    """Format file size for display."""
    if size_bytes >= 1_000_000:
        return f"{size_bytes / 1_000_000:.1f} MB"
    elif size_bytes >= 1_000:
        return f"{size_bytes / 1_000:.1f} KB"
    return f"{size_bytes} B"


# ---------------------------------------------------------------------------
# Batch analysis
# ---------------------------------------------------------------------------

def analyze_batch(
    filepaths: list[str],
    expected_product_smiles: str | None = None,
    expected_mw: float | None = None,
) -> dict:
    """Analyze multiple instrument files.

    Returns:
        {files: [AnalysisResult], summary: {total, interpreted, anomalies, pending}}
    """
    results = []
    for fp in filepaths:
        try:
            result = analyze_file(
                fp,
                expected_product_smiles=expected_product_smiles,
                expected_mw=expected_mw,
            )
            results.append(result)
        except Exception as e:
            logger.warning(f"Failed to analyze {fp}: {e}")
            results.append({
                "id": f"FILE-{uuid.uuid4().hex[:6].upper()}",
                "filename": os.path.basename(fp),
                "instrument": detect_instrument(fp),
                "status": "pending",
                "interpretation": None,
                "error": str(e),
            })

    interpreted = sum(1 for r in results if r.get("status") == "interpreted")
    anomalies = sum(1 for r in results if r.get("status") == "anomaly")
    pending = sum(1 for r in results if r.get("status") == "pending")

    return {
        "files": results,
        "summary": {
            "total": len(results),
            "interpreted": interpreted,
            "anomalies": anomalies,
            "pending": pending,
        },
    }
