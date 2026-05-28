"""
RecovAI — FastAPI Backend
=========================
Single-file REST API that wires all 5 engines together.
The frontend (React/Streamlit) POSTs shift data here and
receives structured predictions, anomaly scores, SHAP values,
reagent recommendations, drift status, and NLP reports.

Run:
    uvicorn main:app --reload --port 8000

Endpoints:
    POST /predict          — full pipeline for one shift (engines 1–4)
    POST /report           — NLP report (engine 5, requires ANTHROPIC_API_KEY)
    POST /drift/batch      — batch PSI check on a CSV upload
    GET  /health           — liveness check
    GET  /model/info       — model metadata
"""

import os
import io
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import joblib

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

# ── Database ────────────────────────────────────────────────────────────────
from database import init_db, get_db, save_prediction, save_report, get_recent_predictions, get_anomaly_predictions, get_prediction_by_id

# ── Engine imports ─────────────────────────────────────────────────────────
from engines.engine1_reagent import ReagentOptimizer, REAGENT_FEATURES
from engines.engine2_anomaly import AnomalyDetector
from engines.engine3_shap    import ShapExplainer
from engines.engine4_psi     import PSIMonitor
from engines.engine5_nlp     import ShiftReportGenerator

# ── Config ─────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
log = logging.getLogger("recovai")

MODEL_DIR   = Path(os.environ.get("RECOVAI_MODEL_DIR", "recovai_output"))
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# ── Pydantic schemas ────────────────────────────────────────────────────────

class ShiftInput(BaseModel):
    """Single shift — all process variables."""
    head_grade:    float = Field(..., ge=0.1, le=5.0,   description="Head Grade (%Cu)")
    feed_rate:     float = Field(..., ge=20,  le=300,   description="Feed Rate (MT/h)")
    ph:            float = Field(..., ge=7.0, le=13.0,  description="Flotation pH")
    pulp_density:  float = Field(..., ge=10,  le=60,    description="Pulp Density (%)")
    air_flow:      float = Field(..., ge=2,   le=40,    description="Air Flow Rate (m³/min)")
    sipx:          float = Field(..., ge=0,   le=150,   description="SIPX Dose (g/t)")
    frother:       float = Field(..., ge=0,   le=80,    description="Frother Dose (g/t)")
    lime:          float = Field(..., ge=0,   le=10,    description="Lime Dose (kg/t)")
    depressant:    float = Field(..., ge=0,   le=100,   description="Depressant Dose (g/t)")
    particle_size: float = Field(150.0, ge=50, le=400,  description="Feed Particle Size P80 (µm)")
    water_recovery:float = Field(72.0,  ge=30, le=95,   description="Water Recovery (%)")
    rougher_grade: float = Field(18.0,  ge=2,  le=50,   description="Rougher Conc Grade (%Cu)")

    def to_feature_dict(self) -> Dict[str, float]:
        return {
            "Head Grade (%Cu)":                self.head_grade,
            "Feed Rate (MT/h)":                self.feed_rate,
            "Flotation pH":                    self.ph,
            "Pulp Density (%)":                self.pulp_density,
            "Air Flow Rate (m3/min)":          self.air_flow,
            "SIPX Dose (g/t)":                 self.sipx,
            "Frother Dose (g/t)":              self.frother,
            "Lime Dose (kg/t)":                self.lime,
            "Depressant Dose (g/t)":           self.depressant,
            "Feed Particle Size (P80 microns)":self.particle_size,
            "Water Recovery (%)":              self.water_recovery,
            "Rougher Conc Grade (%Cu)":        self.rougher_grade,
        }


class ReportRequest(BaseModel):
    shift:          ShiftInput
    predicted_rec:  float
    shap_result:    Dict
    anomaly_result: Dict
    reagent_result: Dict
    drift_result:   Dict
    shift_id:       Optional[str] = None


# ── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="RecovAI API",
    description="Copper flotation recovery intelligence — 5-engine backend",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model registry (lazy-loaded at startup) ──────────────────────────────────

_registry: Dict = {}


def _load_engines() -> None:
    """Load all trained artefacts from disk into memory once at startup."""
    log.info("Loading engines...")

    # XGBoost / RF model (primary predictor)
    xgb_path = MODEL_DIR / "model_recovery_xgb.pkl"
    rf_path  = MODEL_DIR / "model_recovery_rf_clean.pkl"
    if xgb_path.exists():
        _registry["model"] = joblib.load(xgb_path)
        _registry["model_name"] = "XGBoost"
        log.info("  ✓ XGBoost model loaded")
    elif rf_path.exists():
        _registry["model"] = joblib.load(rf_path)
        _registry["model_name"] = "RandomForest"
        log.info("  ✓ RandomForest model loaded")
    else:
        log.warning("  ⚠ No prediction model found — /predict will fail")

    if "model" in _registry:
        _registry["explainer"] = ShapExplainer(_registry["model"])
        _registry["reagent"]   = ReagentOptimizer.__new__(ReagentOptimizer)
        _registry["reagent"].model = _registry["model"]
        _registry["reagent"].feature_names = _registry["explainer"].feature_names
        _registry["reagent"].reagent_features = REAGENT_FEATURES
        from engines.engine1_reagent import REAGENT_BOUNDS
        _registry["reagent"].bounds = REAGENT_BOUNDS

    # Anomaly detector
    iso_path = MODEL_DIR / "isolation_forest.pkl"
    if iso_path.exists():
        _registry["anomaly"] = AnomalyDetector.load(str(iso_path))
        log.info("  ✓ Isolation Forest loaded")
    else:
        log.warning("  ⚠ No isolation_forest.pkl — anomaly scoring unavailable")

    # PSI monitor
    psi_path = MODEL_DIR / "psi_monitor.pkl"
    if psi_path.exists():
        _registry["psi"] = PSIMonitor.load(str(psi_path))
        log.info("  ✓ PSI monitor loaded")
    else:
        log.warning("  ⚠ No psi_monitor.pkl — drift checking unavailable")

    # NLP generator (optional — needs API key)
    if ANTHROPIC_KEY:
        _registry["nlp"] = ShiftReportGenerator(api_key=ANTHROPIC_KEY)
        log.info("  ✓ NLP report generator ready")
    else:
        log.warning("  ⚠ ANTHROPIC_API_KEY not set — /report will be disabled")

    log.info("Engine loading complete.")


@app.on_event("startup")
async def startup() -> None:
    init_db()          # create DB tables if they don't exist yet
    _load_engines()


# ── Health ───────────────────────────────────────────────────────────────────

@app.get("/health", tags=["system"])
def health():
    return {
        "status":  "ok",
        "engines": {
            "predictor": _registry.get("model_name", "not loaded"),
            "anomaly":   "loaded" if "anomaly"   in _registry else "not loaded",
            "psi":       "loaded" if "psi"        in _registry else "not loaded",
            "nlp":       "ready"  if "nlp"        in _registry else "no API key",
        },
    }


@app.get("/model/info", tags=["system"])
def model_info():
    if "model" not in _registry:
        raise HTTPException(503, "No prediction model loaded")
    m = _registry["model"]
    return {
        "model_type":  type(m).__name__,
        "model_name":  _registry.get("model_name"),
        "n_features":  getattr(m, "n_features_in_", "unknown"),
        "model_dir":   str(MODEL_DIR),
    }


# ── Core prediction endpoint ──────────────────────────────────────────────────

@app.post("/predict", tags=["prediction"])
def predict(shift_in: ShiftInput, db: Session = Depends(get_db)):
    """
    Run all 4 analytical engines for a single shift.

    Returns:
        predicted_recovery  — XGBoost/RF prediction
        shap                — SHAP values + top 3 drivers
        anomaly             — IsolationForest score + label
        reagent             — optimizer result + gap table
        drift               — PSI status for this shift's values
    """
    if "model" not in _registry:
        raise HTTPException(503, "Prediction model not loaded. Train first.")

    shift = shift_in.to_feature_dict()

    # ── 1. Prediction ─────────────────────────────────────────────────────
    feat_names = _registry["explainer"].feature_names
    X = np.array([[shift[f] for f in feat_names]])
    predicted  = float(_registry["model"].predict(X)[0])

    # ── 2. SHAP ───────────────────────────────────────────────────────────
    shap_result = _registry["explainer"].explain_shift(shift)
    shap_payload = _registry["explainer"].get_dashboard_payload(shap_result)

    # ── 3. Anomaly ────────────────────────────────────────────────────────
    if "anomaly" in _registry:
        anomaly_result = _registry["anomaly"].score_shift(shift)
    else:
        anomaly_result = {
            "score": None, "label": "UNAVAILABLE", "colour": "GREY",
            "is_anomaly": None, "feature_z": {}, "top_contributors": []
        }

    # ── 4. Reagent optimizer ──────────────────────────────────────────────
    current_doses = {k: shift[k] for k in REAGENT_FEATURES}
    fixed_conds   = {k: v for k, v in shift.items() if k not in REAGENT_FEATURES}
    reagent_result = _registry["reagent"].optimize(fixed_conds, current_doses)

    # ── 5. Drift (single-shift check) ────────────────────────────────────
    if "psi" in _registry:
        drift_result = _registry["psi"].check_single_shift(shift)
    else:
        drift_result = {
            "overall_status": "UNAVAILABLE", "overall_colour": "GREY",
            "worst_psi": None, "flagged": [], "features": {}
        }

    response = {
        "predicted_recovery": round(predicted, 3),
        "shap":               shap_payload,
        "anomaly":            anomaly_result,
        "reagent":            reagent_result,
        "drift":              drift_result,
    }

    # ── Persist to database ───────────────────────────────────────────────
    try:
        db_record = save_prediction(db, shift_in.model_dump(), response)
        response["db_id"] = db_record.id   # expose the row ID to the caller
    except Exception as exc:
        log.error(f"DB write failed (prediction still returned): {exc}")

    return response


# ── NLP report endpoint ────────────────────────────────────────────────────

@app.post("/report", tags=["report"])
def generate_report(req: ReportRequest, db: Session = Depends(get_db)):
    """Generate a plain-English shift report using Claude."""
    if "nlp" not in _registry:
        raise HTTPException(
            503,
            "NLP generator not available. Set ANTHROPIC_API_KEY environment variable."
        )
    shift = req.shift.to_feature_dict()
    result = _registry["nlp"].generate(
        shift_data     = shift,
        predicted_rec  = req.predicted_rec,
        shap_result    = req.shap_result,
        anomaly_result = req.anomaly_result,
        reagent_result = req.reagent_result,
        drift_result   = req.drift_result,
        shift_id       = req.shift_id,
    )

    # ── Persist to database ───────────────────────────────────────────────
    try:
        save_report(db, result, shift_id=req.shift_id)
    except Exception as exc:
        log.error(f"DB write failed (report still returned): {exc}")

    return result


# ── Batch drift endpoint ───────────────────────────────────────────────────

@app.post("/drift/batch", tags=["drift"])
async def batch_drift(file: UploadFile = File(...)):
    """
    Upload a CSV of recent shifts; receive PSI drift report for all features.
    CSV must contain the same feature columns as training data.
    """
    if "psi" not in _registry:
        raise HTTPException(503, "PSI monitor not loaded")

    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")

    missing = [f for f in _registry["psi"].feature_names if f not in df.columns]
    if missing:
        raise HTTPException(400, f"CSV missing columns: {missing}")

    result = _registry["psi"].check(df)
    return result


# ── Sensitivity surface endpoint (for contour plots) ──────────────────────

@app.post("/reagent/surface", tags=["reagent"])
def reagent_surface(
    shift_in:  ShiftInput,
    reagent_x: str = "SIPX Dose (g/t)",
    reagent_y: str = "Frother Dose (g/t)",
    n_points:  int = 15,
):
    """Return a 2-D recovery surface for two reagents — used by contour plots."""
    if "reagent" not in _registry:
        raise HTTPException(503, "Reagent engine not loaded")
    shift = shift_in.to_feature_dict()
    surface = _registry["reagent"].sensitivity_surface(
        shift, reagent_x, reagent_y, n_points
    )
    return surface


# ── Reload models without restart ─────────────────────────────────────────

@app.post("/admin/reload", tags=["system"])
def reload_models(background_tasks: BackgroundTasks):
    """Hot-reload all models from disk (e.g. after retraining)."""
    background_tasks.add_task(_load_engines)
    return {"message": "Model reload scheduled in background."}


# ── Database query endpoints ───────────────────────────────────────────────

@app.get("/history/predictions", tags=["history"])
def list_predictions(limit: int = 50, db: Session = Depends(get_db)):
    """
    Return the N most recent shift predictions from the database.
    Useful for building a history table in the dashboard.
    """
    records = get_recent_predictions(db, limit=limit)
    return [
        {
            "id":                 r.id,
            "created_at":         r.created_at.isoformat() if r.created_at else None,
            "predicted_recovery": r.predicted_recovery,
            "anomaly_label":      r.anomaly_label,
            "anomaly_score":      r.anomaly_score,
            "is_anomaly":         r.is_anomaly,
            "drift_status":       r.drift_status,
            "reagent_gain":       r.reagent_gain,
            # Input summary
            "head_grade":         r.head_grade,
            "feed_rate":          r.feed_rate,
            "ph":                 r.ph,
            "sipx":               r.sipx,
            "frother":            r.frother,
        }
        for r in records
    ]


@app.get("/history/predictions/{prediction_id}", tags=["history"])
def get_prediction(prediction_id: int, db: Session = Depends(get_db)):
    """Return full detail (including JSON engine payloads) for a single prediction."""
    import json as _json
    r = get_prediction_by_id(db, prediction_id)
    if not r:
        raise HTTPException(404, f"Prediction {prediction_id} not found")
    return {
        "id":                 r.id,
        "created_at":         r.created_at.isoformat() if r.created_at else None,
        "inputs": {
            "head_grade":     r.head_grade,
            "feed_rate":      r.feed_rate,
            "ph":             r.ph,
            "pulp_density":   r.pulp_density,
            "air_flow":       r.air_flow,
            "sipx":           r.sipx,
            "frother":        r.frother,
            "lime":           r.lime,
            "depressant":     r.depressant,
            "particle_size":  r.particle_size,
            "water_recovery": r.water_recovery,
            "rougher_grade":  r.rougher_grade,
        },
        "predicted_recovery": r.predicted_recovery,
        "anomaly_label":      r.anomaly_label,
        "anomaly_score":      r.anomaly_score,
        "is_anomaly":         r.is_anomaly,
        "drift_status":       r.drift_status,
        "reagent_gain":       r.reagent_gain,
        "shap":               _json.loads(r.shap_json)    if r.shap_json    else None,
        "anomaly":            _json.loads(r.anomaly_json) if r.anomaly_json else None,
        "reagent":            _json.loads(r.reagent_json) if r.reagent_json else None,
        "drift":              _json.loads(r.drift_json)   if r.drift_json   else None,
    }


@app.get("/history/anomalies", tags=["history"])
def list_anomalies(limit: int = 50, db: Session = Depends(get_db)):
    """Return the N most recent shifts that were flagged as anomalies."""
    records = get_anomaly_predictions(db, limit=limit)
    return [
        {
            "id":                 r.id,
            "created_at":         r.created_at.isoformat() if r.created_at else None,
            "predicted_recovery": r.predicted_recovery,
            "anomaly_label":      r.anomaly_label,
            "anomaly_score":      r.anomaly_score,
            "drift_status":       r.drift_status,
        }
        for r in records
    ]