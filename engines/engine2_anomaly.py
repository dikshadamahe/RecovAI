"""
RecovAI — Engine 2: Isolation Forest Anomaly Detection
=======================================================
Trains an Isolation Forest on clean historical shift data and scores
every new shift for anomalousness — no labelled data required.

Usage:
    from engines.engine2_anomaly import AnomalyDetector
    det = AnomalyDetector()
    det.train(X_train_df)
    det.save("recovai_output/isolation_forest.pkl")

    # At inference time:
    det = AnomalyDetector.load("recovai_output/isolation_forest.pkl")
    result = det.score_shift(shift_dict)
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

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

# Thresholds on the IsolationForest decision_function score
# More negative = more anomalous
THRESHOLD_ALERT   = -0.20   # flag as anomaly
THRESHOLD_WARNING = -0.10   # flag as suspicious


class AnomalyDetector:
    """
    Wraps sklearn IsolationForest with a scaler, provides per-feature
    contribution attribution, and returns structured results for the dashboard.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        contamination: float = 0.05,
        random_state: int = 42,
    ):
        self.n_estimators  = n_estimators
        self.contamination = contamination
        self.random_state  = random_state
        self.model: Optional[IsolationForest] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = FEATURES
        self.training_stats: Dict = {}

    # ── Training ──────────────────────────────────────────────────────────

    def train(self, X: pd.DataFrame) -> "AnomalyDetector":
        """
        Fit the Isolation Forest on clean historical data.

        Parameters
        ----------
        X : pd.DataFrame
            Historical shift data. Columns must include all FEATURES.
        """
        X_arr = X[self.feature_names].values.astype(float)

        # Scale — helps IF deal with features of very different magnitudes
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_arr)

        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_samples="auto",
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(X_scaled)

        # Store training statistics for per-feature deviation analysis
        self.training_stats = {
            "mean": X[self.feature_names].mean().to_dict(),
            "std":  X[self.feature_names].std().to_dict(),
            "min":  X[self.feature_names].min().to_dict(),
            "max":  X[self.feature_names].max().to_dict(),
            "n_samples": len(X),
        }

        # Compute training score distribution
        scores = self.model.decision_function(X_scaled)
        self.training_stats["score_mean"] = float(np.mean(scores))
        self.training_stats["score_std"]  = float(np.std(scores))
        self.training_stats["score_p5"]   = float(np.percentile(scores, 5))

        print(f"[Engine 2] Trained on {len(X)} shifts. "
              f"Estimated anomaly fraction: {self.contamination:.0%}")
        return self

    # ── Inference ─────────────────────────────────────────────────────────

    def score_shift(self, shift: Dict[str, float]) -> Dict:
        """
        Score a single shift dictionary.

        Returns
        -------
        dict with:
            score          — float, IF decision function (-∞ → 0+ = normal)
            label          — 'NORMAL' | 'SUSPICIOUS' | 'ANOMALY'
            colour         — 'GREEN' | 'AMBER' | 'RED'
            is_anomaly     — bool (hard IF prediction: -1 = anomaly)
            feature_z      — dict, z-score of each feature vs training mean
            top_contributors — list of (feature, z_score) for 3 most deviant
        """
        self._check_fitted()
        X_arr  = np.array([[shift[f] for f in self.feature_names]])
        X_sc   = self.scaler.transform(X_arr)

        score    = float(self.model.decision_function(X_sc)[0])
        raw_pred = int(self.model.predict(X_sc)[0])   # -1 or +1

        label, colour = self._classify(score)

        # Per-feature z-scores vs training distribution
        feature_z = {}
        for f in self.feature_names:
            mu  = self.training_stats["mean"].get(f, 0)
            sig = self.training_stats["std"].get(f, 1) or 1
            feature_z[f] = round((shift[f] - mu) / sig, 3)

        top_contributors = sorted(
            feature_z.items(), key=lambda x: abs(x[1]), reverse=True
        )[:3]

        return {
            "score":             round(score, 4),
            "label":             label,
            "colour":            colour,
            "is_anomaly":        raw_pred == -1,
            "feature_z":         feature_z,
            "top_contributors":  top_contributors,
            "threshold_alert":   THRESHOLD_ALERT,
            "threshold_warning": THRESHOLD_WARNING,
        }

    def score_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score a DataFrame of shifts. Returns df with added columns:
        anomaly_score, anomaly_label, anomaly_colour, is_anomaly.
        """
        self._check_fitted()
        X_arr = df[self.feature_names].values.astype(float)
        X_sc  = self.scaler.transform(X_arr)
        scores = self.model.decision_function(X_sc)
        preds  = self.model.predict(X_sc)

        out = df.copy()
        out["anomaly_score"]  = np.round(scores, 4)
        out["is_anomaly"]     = preds == -1
        out["anomaly_label"]  = [self._classify(s)[0] for s in scores]
        out["anomaly_colour"] = [self._classify(s)[1] for s in scores]
        return out

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model":          self.model,
            "scaler":         self.scaler,
            "feature_names":  self.feature_names,
            "training_stats": self.training_stats,
            "config": {
                "n_estimators":  self.n_estimators,
                "contamination": self.contamination,
            },
        }
        joblib.dump(payload, path)
        print(f"[Engine 2] Model saved → {path}")

    @classmethod
    def load(cls, path: str) -> "AnomalyDetector":
        payload    = joblib.load(path)
        cfg        = payload["config"]
        det        = cls(n_estimators=cfg["n_estimators"],
                         contamination=cfg["contamination"])
        det.model          = payload["model"]
        det.scaler         = payload["scaler"]
        det.feature_names  = payload["feature_names"]
        det.training_stats = payload["training_stats"]
        return det

    # ── Helpers ───────────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model not fitted. Call .train() or .load() first.")

    @staticmethod
    def _classify(score: float) -> Tuple[str, str]:
        if score < THRESHOLD_ALERT:
            return "ANOMALY", "RED"
        elif score < THRESHOLD_WARNING:
            return "SUSPICIOUS", "AMBER"
        else:
            return "NORMAL", "GREEN"


# ── Standalone test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile, os
    print("Engine 2 — Isolation Forest Anomaly Detection\n" + "="*48)

    rng = np.random.default_rng(42)
    n   = 1000

    # Synthetic training data — realistic-ish ranges
    train_data = {
        "Head Grade (%Cu)":                rng.normal(1.2,  0.15, n),
        "Feed Rate (MT/h)":                rng.normal(120,  10,   n),
        "Flotation pH":                    rng.normal(10.5, 0.3,  n),
        "Pulp Density (%)":                rng.normal(32,   2,    n),
        "Air Flow Rate (m3/min)":          rng.normal(14,   1.5,  n),
        "SIPX Dose (g/t)":                 rng.normal(40,   5,    n),
        "Frother Dose (g/t)":              rng.normal(20,   3,    n),
        "Lime Dose (kg/t)":                rng.normal(2.5,  0.4,  n),
        "Depressant Dose (g/t)":           rng.normal(25,   4,    n),
        "Feed Particle Size (P80 microns)":rng.normal(150,  15,   n),
        "Water Recovery (%)":              rng.normal(72,   4,    n),
        "Rougher Conc Grade (%Cu)":        rng.normal(18,   2,    n),
    }
    X_train = pd.DataFrame(train_data)

    det = AnomalyDetector(n_estimators=100, contamination=0.05)
    det.train(X_train)

    # Normal shift
    normal_shift = {f: float(X_train[f].mean()) for f in FEATURES}
    res_n = det.score_shift(normal_shift)
    print(f"\nNormal shift  → score: {res_n['score']:+.4f}  [{res_n['label']}]")

    # Anomalous shift (extreme values)
    anomaly_shift = normal_shift.copy()
    anomaly_shift["Flotation pH"]       = 8.0    # way off
    anomaly_shift["SIPX Dose (g/t)"]    = 80.0   # maxed out
    anomaly_shift["Feed Rate (MT/h)"]   = 190.0  # extreme
    res_a = det.score_shift(anomaly_shift)
    print(f"Anomaly shift → score: {res_a['score']:+.4f}  [{res_a['label']}]")
    print(f"Top contributors: {res_a['top_contributors']}")

    # Save / reload
    tmp = tempfile.mktemp(suffix=".pkl")
    det.save(tmp)
    det2 = AnomalyDetector.load(tmp)
    res2 = det2.score_shift(normal_shift)
    assert abs(res2["score"] - res_n["score"]) < 1e-6, "Reload mismatch!"
    os.unlink(tmp)

    print("\nEngine 2 OK ✓")
