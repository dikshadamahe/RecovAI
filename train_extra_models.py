import os
import joblib
import pandas as pd
import numpy as np

from recovai_train import load_dataset, DATASET_PATH, FEATURES
from engines.engine2_anomaly import AnomalyDetector
from engines.engine4_psi import PSIMonitor

def main():
    print("=" * 60)
    print("Fitting Anomaly Detector & PSI Monitor on 24-feature schema")
    print("=" * 60)

    # 1. Load data
    df = load_dataset(DATASET_PATH)
    print(f"Loaded {len(df)} shifts.")

    # 2. Extract and dropna
    X = df[FEATURES].dropna()
    print(f"Shifts after dropping missing: {len(X)}")

    # 3. Fit Anomaly Detector
    print("Training Isolation Forest Anomaly Detector...")
    det = AnomalyDetector(n_estimators=200, contamination=0.05)
    det.train(X)
    det.save("recovai_output/isolation_forest.pkl")

    # 4. Fit PSI Monitor
    print("Fitting PSI Drift Monitor...")
    psi = PSIMonitor()
    psi.fit(X)
    psi.save("recovai_output/psi_monitor.pkl")

    print("\nTraining of extras complete!")

if __name__ == "__main__":
    main()
