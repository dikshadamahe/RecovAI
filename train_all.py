"""
RecovAI — train_all.py
======================
One-shot training script. Run this ONCE (or after new data arrives) to:
  1. Load + clean your historical shift data
  2. Train XGBoost prediction model
  3. Train Isolation Forest (Engine 2)
  4. Fit PSI monitor reference (Engine 4)
  5. Save all artefacts to recovai_output/

Run:
    python train_all.py --data path/to/shifts.csv
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor
    USE_XGB = True
except ImportError:
    USE_XGB = False
    print("[train_all] xgboost not found — using GradientBoostingRegressor")

from engines.engine2_anomaly import AnomalyDetector
from engines.engine4_psi     import PSIMonitor

# ── Column mapping: CSV column name → internal feature name ─────────────────
# Your CSV uses these actual column names (row 1 of the file)
CSV_TO_FEATURE = {
    "Head Grade (%Cu)":          "Head Grade (%Cu)",
    "Feed Rate (MT/h)":          "Feed Rate (MT/h)",
    "Flotation pH":              "Flotation pH",
    "SIPX Dose (g/t)":           "SIPX Dose (g/t)",
    "Frother Dose (g/t)":        "Frother Dose (g/t)",
    "Depressant Dose (g/t)":     "Depressant Dose (g/t)",
    "Conc. Mass Pull (%)":       "Pulp Density (%)",          # proxy
    "Grinding kWh":              "Air Flow Rate (m3/min)",    # proxy
    "T Reagent (cc)":            "Lime Dose (kg/t)",          # proxy
    "Ore Milled (MT)":           "Feed Particle Size (P80 microns)",  # proxy
    "Concentrate Grade (%Cu)":   "Rougher Conc Grade (%Cu)",
    "Tails Grade (%Cu)":         "Water Recovery (%)",        # proxy
}

# Internal feature names used by all engines
FEATURES = [
    "Head Grade (%Cu)",
    "Feed Rate (MT/h)",
    "Flotation pH",
    "Pulp Density (%)",
    "Air Flow Rate (m3/min)",
    "SIPX Dose (g/t)",
    "Frother Dose (g/t)",
    "Lime Dose (kg/t)",
    "Depressant Dose (g/t)",
    "Feed Particle Size (P80 microns)",
    "Water Recovery (%)",
    "Rougher Conc Grade (%Cu)",
]

# Target column in the CSV
CSV_TARGET = "Recovery (%)"
TARGET     = "Cu Recovery (%)"

OUT_DIR = Path("recovai_output")


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_data(csv_path: str) -> pd.DataFrame:
    # The CSV has a 2-row header: row 0 = section labels, row 1 = column names
    df = pd.read_csv(csv_path, header=1)
    print(f"[train_all] Loaded {len(df)} rows from {csv_path}")
    print(f"[train_all] Columns found: {list(df.columns[:10])} ...")

    # Rename CSV columns → internal feature names
    df = df.rename(columns=CSV_TO_FEATURE)

    # Rename target
    if CSV_TARGET in df.columns:
        df = df.rename(columns={CSV_TARGET: TARGET})
    elif TARGET not in df.columns:
        raise ValueError(f"Could not find target column '{CSV_TARGET}' in CSV.")

    # Check all features present after remapping
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Still missing after remapping: {missing}")

    # Keep only needed columns
    df = df[FEATURES + [TARGET]].copy()

    # Clean
    before = len(df)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=FEATURES + [TARGET])
    df = df[(df[TARGET] > 0) & (df[TARGET] < 100)]
    print(f"[train_all] After cleaning: {len(df)} rows ({before - len(df)} dropped)")
    return df


def train_predictor(X_train, y_train, X_test, y_test):
    print("[train_all] Training prediction model...")
    if USE_XGB:
        model = XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            random_state=42,
        )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)
    r2    = r2_score(y_test, preds)
    print(f"  MAE  = {mae:.3f} pp")
    print(f"  R²   = {r2:.4f}")

    return model, {"mae": round(mae, 4), "r2": round(r2, 4)}


# ── Main ─────────────────────────────────────────────────────────────────────

def main(csv_path: str, out_dir: Path = OUT_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(csv_path)
    X  = df[FEATURES].values.astype(float)
    y  = df[TARGET].values.astype(float)

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(df)), test_size=0.15, random_state=42
    )

    # Train predictor
    model, metrics = train_predictor(X_train, y_train, X_test, y_test)
    model_name = "model_recovery_xgb.pkl" if USE_XGB else "model_recovery_gb.pkl"
    joblib.dump(model, out_dir / model_name)
    print(f"  Saved → {out_dir / model_name}")

    # Train Isolation Forest
    print("[train_all] Training Isolation Forest (Engine 2)...")
    X_train_df = pd.DataFrame(X_train, columns=FEATURES)
    det = AnomalyDetector(n_estimators=200, contamination=0.05)
    det.train(X_train_df)
    det.save(str(out_dir / "isolation_forest.pkl"))

    # Fit PSI monitor
    print("[train_all] Fitting PSI monitor (Engine 4)...")
    psi = PSIMonitor()
    psi.fit(X_train_df)
    psi.save(str(out_dir / "psi_monitor.pkl"))

    # Save metadata
    meta = {
        "features":    FEATURES,
        "target":      TARGET,
        "model_file":  model_name,
        "n_train":     int(len(X_train)),
        "n_test":      int(len(X_test)),
        "metrics":     metrics,
        "use_xgboost": USE_XGB,
        "column_map":  CSV_TO_FEATURE,
    }
    with open(out_dir / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*50}")
    print("Training complete. Artefacts written to:", out_dir)
    for p in sorted(out_dir.iterdir()):
        print(f"  {p.name}")
    print(f"\nNext step: uvicorn main:app --reload --port 8000")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train all RecovAI engines")
    parser.add_argument("--data", required=True, help="Path to historical shifts CSV")
    parser.add_argument("--out",  default="recovai_output", help="Output directory")
    args = parser.parse_args()
    main(args.data, Path(args.out))