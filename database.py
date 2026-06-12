"""
RecovAI — Database Layer
========================
SQLite-backed persistence using SQLAlchemy ORM.

Tables
------
  shift_predictions  — one row per /predict call
  shift_reports      — one row per /report call (linked to a prediction)

Usage (inside FastAPI endpoints)
---------------------------------
    from database import SessionLocal, save_prediction, save_report

    db = SessionLocal()
    try:
        save_prediction(db, shift_input_dict, prediction_response_dict)
    finally:
        db.close()

Switch to PostgreSQL later
---------------------------
Just change DATABASE_URL to:
    "postgresql://user:password@host:5432/recovai"
and install psycopg2-binary. Everything else stays identical.
"""

import os
import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from sqlalchemy import (
    create_engine, Column, Integer, Float, String,
    Text, DateTime, ForeignKey, Boolean, inspect, text
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

# ── Database URL ─────────────────────────────────────────────────────────────
# Default: SQLite file in the project root.
# Override by setting the DATABASE_URL environment variable.
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./recovai.db")

engine = create_engine(
    DATABASE_URL,
    # connect_args only needed for SQLite (allows multi-thread access in FastAPI)
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    echo=False,   # set True to log all SQL — useful for debugging
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ── ORM Models ───────────────────────────────────────────────────────────────

class ShiftPrediction(Base):
    """
    Stores every /predict call.

    - Input columns mirror ShiftInput fields (all floats).
    - Output columns store the top-level numeric results.
    - Full engine payloads (SHAP, anomaly, reagent, drift) are stored as JSON
      text so we don't need a schema change every time the engine output evolves.
    """
    __tablename__ = "shift_predictions"

    id              = Column(Integer, primary_key=True, index=True)
    created_at      = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # ── Shift metadata (from submit form) ─────────────────────────────────
    shift_date      = Column(String(20), nullable=True, index=True)   # YYYY-MM-DD
    shift           = Column(String(20), nullable=True)                # Morning / Afternoon / Night
    operator_name   = Column(String(100), nullable=True)
    notes           = Column(Text, nullable=True)
    actual_recovery = Column(Float, nullable=True)

    # ── Shift inputs ──────────────────────────────────────────────────────
    head_grade      = Column(Float, nullable=False)
    feed_rate       = Column(Float, nullable=False)
    ph              = Column(Float, nullable=False)
    pulp_density    = Column(Float, nullable=False)
    air_flow        = Column(Float, nullable=False)
    sipx            = Column(Float, nullable=False)
    frother         = Column(Float, nullable=False)
    lime            = Column(Float, nullable=False)
    depressant      = Column(Float, nullable=False)
    particle_size   = Column(Float, nullable=False)
    water_recovery  = Column(Float, nullable=False)
    rougher_grade   = Column(Float, nullable=False)

    # ── Top-level prediction output ───────────────────────────────────────
    predicted_recovery  = Column(Float,  nullable=True)

    # Anomaly summary (easy filtering without parsing JSON)
    anomaly_label   = Column(String(20), nullable=True)   # e.g. "NORMAL", "ANOMALY"
    anomaly_score   = Column(Float,      nullable=True)
    is_anomaly      = Column(Boolean,    nullable=True)

    # Drift summary
    drift_status    = Column(String(20), nullable=True)   # e.g. "OK", "WARNING", "CRITICAL"

    # Reagent optimizer — optimal recovery gain
    reagent_gain    = Column(Float, nullable=True)        # optimised_recovery - predicted

    # ── Full engine payloads as JSON ──────────────────────────────────────
    shap_json       = Column(Text, nullable=True)
    anomaly_json    = Column(Text, nullable=True)
    reagent_json    = Column(Text, nullable=True)
    drift_json      = Column(Text, nullable=True)

    # Back-reference to reports generated for this prediction
    reports         = relationship("ShiftReport", back_populates="prediction")

    def __repr__(self):
        return (
            f"<ShiftPrediction id={self.id} "
            f"recovery={self.predicted_recovery} "
            f"anomaly={self.anomaly_label} "
            f"at={self.created_at}>"
        )


class ShiftReport(Base):
    """
    Stores every /report call (NLP-generated text reports).
    Each report is linked back to the ShiftPrediction it was generated from.
    """
    __tablename__ = "shift_reports"

    id              = Column(Integer, primary_key=True, index=True)
    created_at      = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Optional shift_id label supplied by the caller (e.g. "2024-06-01-D")
    shift_id        = Column(String(100), nullable=True, index=True)

    # FK back to ShiftPrediction (nullable — report can be standalone)
    prediction_id   = Column(Integer, ForeignKey("shift_predictions.id"), nullable=True)
    prediction      = relationship("ShiftPrediction", back_populates="reports")

    # Report content
    report_text     = Column(Text, nullable=True)   # plain-English narrative
    full_response   = Column(Text, nullable=True)   # full JSON response from engine 5

    def __repr__(self):
        return (
            f"<ShiftReport id={self.id} "
            f"shift_id={self.shift_id} "
            f"prediction_id={self.prediction_id} "
            f"at={self.created_at}>"
        )


# ── Schema creation ───────────────────────────────────────────────────────────

def _migrate_shift_metadata_columns() -> None:
    """Add shift metadata columns to existing SQLite databases."""
    if not DATABASE_URL.startswith("sqlite"):
        return
    insp = inspect(engine)
    if "shift_predictions" not in insp.get_table_names():
        return
    existing = {c["name"] for c in insp.get_columns("shift_predictions")}
    additions = [
        ("shift_date", "VARCHAR(20)"),
        ("shift", "VARCHAR(20)"),
        ("operator_name", "VARCHAR(100)"),
        ("notes", "TEXT"),
        ("actual_recovery", "FLOAT"),
    ]
    with engine.begin() as conn:
        for col, col_type in additions:
            if col not in existing:
                conn.execute(
                    text(f"ALTER TABLE shift_predictions ADD COLUMN {col} {col_type}")
                )


def init_db() -> None:
    """Create all tables if they don't exist. Safe to call on every startup."""
    Base.metadata.create_all(bind=engine)
    _migrate_shift_metadata_columns()


# ── Dependency for FastAPI ────────────────────────────────────────────────────

def get_db():
    """
    FastAPI dependency that yields a DB session and ensures it's closed.

    Usage in endpoint:
        from fastapi import Depends
        from database import get_db
        from sqlalchemy.orm import Session

        @app.post("/predict")
        def predict(shift_in: ShiftInput, db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── CRUD helpers ──────────────────────────────────────────────────────────────

def save_prediction(
    db,
    shift_input: Dict[str, Any],
    response: Dict[str, Any],
) -> ShiftPrediction:
    """
    Persist a /predict call to the database.

    Parameters
    ----------
    db          : SQLAlchemy Session
    shift_input : dict with the same keys as ShiftInput (Pydantic model fields)
    response    : the full dict returned by the /predict endpoint

    Returns
    -------
    ShiftPrediction ORM object (already committed, has an .id)
    """
    # Extract anomaly summary
    anomaly = response.get("anomaly", {})
    anomaly_label = anomaly.get("label")
    anomaly_score = anomaly.get("score")
    is_anomaly    = anomaly.get("is_anomaly")

    # Extract drift summary
    drift = response.get("drift", {})
    drift_status = drift.get("overall_status")

    # Reagent gain = optimised_recovery - predicted_recovery
    reagent = response.get("reagent", {})
    opt_recovery = reagent.get("optimised_recovery")
    pred_recovery = response.get("predicted_recovery")
    reagent_gain = None
    if opt_recovery is not None and pred_recovery is not None:
        reagent_gain = round(float(opt_recovery) - float(pred_recovery), 4)

    actual = shift_input.get("actual_recovery")
    if actual is not None and actual != "":
        try:
            actual = float(actual)
        except (TypeError, ValueError):
            actual = None
    else:
        actual = None

    record = ShiftPrediction(
        shift_date      = shift_input.get("shift_date") or None,
        shift           = shift_input.get("shift") or None,
        operator_name   = shift_input.get("operator_name") or None,
        notes           = shift_input.get("notes") or None,
        actual_recovery = actual,

        # Inputs
        head_grade     = shift_input.get("head_grade"),
        feed_rate      = shift_input.get("feed_rate"),
        ph             = shift_input.get("ph"),
        pulp_density   = shift_input.get("pulp_density"),
        air_flow       = shift_input.get("air_flow"),
        sipx           = shift_input.get("sipx"),
        frother        = shift_input.get("frother"),
        lime           = shift_input.get("lime"),
        depressant     = shift_input.get("depressant"),
        particle_size  = shift_input.get("particle_size"),
        water_recovery = shift_input.get("water_recovery"),
        rougher_grade  = shift_input.get("rougher_grade"),

        # Top-level outputs
        predicted_recovery = pred_recovery,
        anomaly_label      = anomaly_label,
        anomaly_score      = anomaly_score,
        is_anomaly         = is_anomaly,
        drift_status       = drift_status,
        reagent_gain       = reagent_gain,

        # Full JSON payloads
        shap_json    = json.dumps(response.get("shap")),
        anomaly_json = json.dumps(anomaly),
        reagent_json = json.dumps(reagent),
        drift_json   = json.dumps(drift),
    )

    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def save_report(
    db,
    response: Dict[str, Any],
    shift_id: Optional[str] = None,
    prediction_id: Optional[int] = None,
) -> ShiftReport:
    """
    Persist a /report call to the database.

    Parameters
    ----------
    db            : SQLAlchemy Session
    response      : full dict returned by the /report endpoint
    shift_id      : optional human-readable shift label
    prediction_id : optional FK to link back to a ShiftPrediction row
    """
    # engine5 may return {"report": "...", ...} or {"text": "..."} — handle both
    report_text = (
        response.get("report")
        or response.get("text")
        or response.get("summary")
        or ""
    )

    record = ShiftReport(
        shift_id      = shift_id,
        prediction_id = prediction_id,
        report_text   = report_text,
        full_response = json.dumps(response),
    )

    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_recent_predictions(db, limit: int = 50):
    """Return the N most recent predictions, newest first."""
    return (
        db.query(ShiftPrediction)
        .order_by(ShiftPrediction.created_at.desc())
        .limit(limit)
        .all()
    )


def get_anomaly_predictions(db, limit: int = 50):
    """Return the N most recent shifts flagged as anomalies."""
    return (
        db.query(ShiftPrediction)
        .filter(ShiftPrediction.is_anomaly == True)
        .order_by(ShiftPrediction.created_at.desc())
        .limit(limit)
        .all()
    )


def get_prediction_by_id(db, prediction_id: int) -> Optional[ShiftPrediction]:
    """Fetch a single prediction by primary key."""
    return db.query(ShiftPrediction).filter(ShiftPrediction.id == prediction_id).first()