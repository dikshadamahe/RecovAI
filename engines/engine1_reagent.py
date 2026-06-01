"""
RecovAI — Engine 1: Reagent Dose Intelligence Engine
=====================================================
Uses SciPy response-surface optimization to find the mathematically
optimal reagent doses (SIPX, Frother, Lime, Depressant) for a given
set of fixed feed conditions, then classifies the gap vs actual plant usage.

Usage:
    from engines.engine1_reagent import ReagentOptimizer
    opt = ReagentOptimizer(model_path="recovai_output/model_recovery_xgb.pkl")
    result = opt.optimize(current_conditions, current_doses)
"""

import numpy as np
import pandas as pd
import joblib
from scipy.optimize import minimize, differential_evolution
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings("ignore")

FEATURES = [
    "Ore Milled (MT)",
    "Head Grade (%Cu)",
    "COPPER IN HEAD (MT)",
    "Feed Rate (MT/h)",
    "Grinding kWh",
    "Lime Bags",
    "T Reagent (cc)",
    "Pine Oil (cc)",
    "Flotation pH",
    "Milling Running Hours",
    "SIPX Dose (g/t)",
    "Frother Dose (g/t)",
    "Depressant Dose (g/t)",
    "Prev_Recovery (%)",
    "Prev_Feed Rate (MT/h)",
    "Prev_Head Grade (%Cu)",
    "Prev_Flotation pH",
    "Roll7_Recovery (%)",
    "Roll7_Head Grade (%Cu)",
    "Roll7_Feed Rate (MT/h)",
    "Feed_Condition_Num",
    "Shift_Num",
    "Month",
    "Day_of_Week",
]

# Reagent features that the optimizer is free to vary
REAGENT_FEATURES = [
    "SIPX Dose (g/t)",
    "Frother Dose (g/t)",
    "Depressant Dose (g/t)",
]

# Physical bounds for each reagent — (min, max)
REAGENT_BOUNDS: Dict[str, Tuple[float, float]] = {
    "SIPX Dose (g/t)":       (5.0,  35.0),
    "Frother Dose (g/t)":    (3.0,   20.0),
    "Depressant Dose (g/t)": (1.0,   12.0),
}

# Gap thresholds for traffic-light classification
GAP_GREEN  = 10.0   # < 10 %  → Optimal
GAP_AMBER  = 20.0   # 10–20 % → Review needed

# > 20 % → Mismatch / recovery risk


class ReagentOptimizer:
    """
    Wraps a trained XGBoost (or any sklearn-API) regressor and exposes a
    response-surface optimizer for reagent doses.
    """

    def __init__(self, model_path: str = "recovai_output/model_recovery_xgb.pkl"):
        self.model = joblib.load(model_path)
        self.feature_names = FEATURES
        self.reagent_features = REAGENT_FEATURES
        self.bounds = REAGENT_BOUNDS

    # ── Internal helpers ──────────────────────────────────────────────────

    def _build_input(
        self,
        fixed_conditions: Dict[str, float],
        reagent_values: np.ndarray,
    ) -> np.ndarray:
        """
        Combine fixed feed conditions with candidate reagent values into
        a single feature vector ordered to match training.
        """
        row = {}
        row.update(fixed_conditions)
        for name, val in zip(self.reagent_features, reagent_values):
            row[name] = val
        return np.array([row[f] for f in self.feature_names]).reshape(1, -1)

    def _objective(
        self,
        reagent_values: np.ndarray,
        fixed_conditions: Dict[str, float],
    ) -> float:
        """Negative predicted recovery (we minimise, so negate)."""
        X = self._build_input(fixed_conditions, reagent_values)
        return -float(self.model.predict(X)[0])

    # ── Public API ────────────────────────────────────────────────────────

    def optimize(
        self,
        current_conditions: Dict[str, float],
        current_doses: Dict[str, float],
        method: str = "L-BFGS-B",
        use_global: bool = False,
    ) -> Dict:
        """
        Find optimal reagent doses for the given feed conditions.

        Parameters
        ----------
        current_conditions : dict
            Fixed process variables (everything except reagents).
        current_doses : dict
            What the plant is actually dosing right now.
        method : str
            SciPy local optimizer — 'L-BFGS-B' (fast) or 'SLSQP'.
        use_global : bool
            If True, run differential_evolution first (slower, more thorough).

        Returns
        -------
        dict with keys:
            optimal_doses     — dict reagent→optimal value
            current_doses     — dict reagent→actual value
            gaps              — dict reagent→gap analysis
            predicted_recovery_optimal  — float
            predicted_recovery_current  — float
            recovery_gain     — float (pp improvement from optimising)
        """
        # Strip reagent keys from conditions so the dict is purely "fixed"
        fixed = {k: v for k, v in current_conditions.items()
                 if k not in self.reagent_features}

        x0 = np.array([current_doses.get(r, np.mean(self.bounds[r]))
                       for r in self.reagent_features])
        bounds = [self.bounds[r] for r in self.reagent_features]

        # ── Global search (optional) ──────────────────────────────────────
        if use_global:
            de_result = differential_evolution(
                self._objective,
                bounds=bounds,
                args=(fixed,),
                seed=42,
                maxiter=200,
                tol=1e-6,
                workers=1,
            )
            x0 = de_result.x   # warm-start local search from global best

        # ── Local refinement ──────────────────────────────────────────────
        result = minimize(
            self._objective,
            x0=x0,
            args=(fixed,),
            method=method,
            bounds=bounds,
            options={"ftol": 1e-8, "maxiter": 1000},
        )

        optimal_doses = {
            r: round(float(v), 2)
            for r, v in zip(self.reagent_features, result.x)
        }

        # ── Predicted recoveries ──────────────────────────────────────────
        X_opt = self._build_input(fixed, result.x)
        rec_optimal = float(self.model.predict(X_opt)[0])

        x_current = np.array([current_doses.get(r, np.mean(self.bounds[r]))
                               for r in self.reagent_features])
        X_cur = self._build_input(fixed, x_current)
        rec_current = float(self.model.predict(X_cur)[0])

        # ── Gap analysis ──────────────────────────────────────────────────
        gaps = {}
        for r in self.reagent_features:
            actual  = current_doses.get(r, np.nan)
            optimal = optimal_doses[r]
            gap_pct = abs(actual - optimal) / max(abs(actual), 1e-6) * 100
            colour, label = self._classify_gap(gap_pct)
            direction = "reduce" if actual > optimal else "increase"
            delta = abs(actual - optimal)
            gaps[r] = {
                "actual":    round(actual, 2),
                "optimal":   optimal,
                "gap_pct":   round(gap_pct, 2),
                "colour":    colour,
                "label":     label,
                "direction": direction,
                "delta":     round(delta, 2),
                "action":    f"{direction.capitalize()} by {delta:.1f} units",
            }

        return {
            "optimal_doses":               optimal_doses,
            "current_doses":               {r: current_doses.get(r) for r in self.reagent_features},
            "gaps":                        gaps,
            "predicted_recovery_optimal":  round(rec_optimal, 3),
            "predicted_recovery_current":  round(rec_current, 3),
            "recovery_gain":               round(rec_optimal - rec_current, 3),
            "optimizer_success":           result.success,
            "optimizer_message":           result.message,
        }

    @staticmethod
    def _classify_gap(gap_pct: float) -> Tuple[str, str]:
        if gap_pct < GAP_GREEN:
            return "GREEN", "Optimal"
        elif gap_pct < GAP_AMBER:
            return "AMBER", "Review needed"
        else:
            return "RED", "Mismatch — recovery risk"

    def sensitivity_surface(
        self,
        fixed_conditions: Dict[str, float],
        reagent_x: str,
        reagent_y: str,
        n_points: int = 20,
    ) -> Dict:
        """
        Generate a 2-D recovery surface for two reagents (for contour plots).
        All other reagents are held at their midpoint.
        """
        x_vals = np.linspace(*self.bounds[reagent_x], n_points)
        y_vals = np.linspace(*self.bounds[reagent_y], n_points)
        fixed = {k: v for k, v in fixed_conditions.items()
                 if k not in self.reagent_features}

        mid_doses = {r: np.mean(self.bounds[r]) for r in self.reagent_features}
        surface = np.zeros((n_points, n_points))

        for i, xv in enumerate(x_vals):
            for j, yv in enumerate(y_vals):
                doses = mid_doses.copy()
                doses[reagent_x] = xv
                doses[reagent_y] = yv
                vec = np.array([doses[r] for r in self.reagent_features])
                X   = self._build_input(fixed, vec)
                surface[j, i] = float(self.model.predict(X)[0])

        return {
            "x_label":  reagent_x,
            "y_label":  reagent_y,
            "x_values": x_vals.tolist(),
            "y_values": y_vals.tolist(),
            "recovery": surface.tolist(),
        }


# ── Standalone test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    print("Engine 1 — Reagent Dose Intelligence\n" + "="*45)

    # Minimal mock model for smoke-test (remove when real model exists)
    from sklearn.linear_model import LinearRegression
    import tempfile

    rng   = np.random.default_rng(0)
    X_tr  = rng.uniform(0, 1, (500, len(FEATURES)))
    y_tr  = 80 + X_tr[:, 0]*10 - X_tr[:, 3]*2 + rng.normal(0, 0.5, 500)
    mock  = LinearRegression().fit(X_tr, y_tr)

    tmp = tempfile.mktemp(suffix=".pkl")
    joblib.dump(mock, tmp)

    opt = ReagentOptimizer(model_path=tmp)

    conditions = {
        "Head Grade (%Cu)":            1.2,
        "Feed Rate (MT/h)":            120.0,
        "Flotation pH":                10.5,
        "Pulp Density (%)":            32.0,
        "Air Flow Rate (m3/min)":      14.0,
        "Feed Particle Size (P80 microns)": 150.0,
        "Water Recovery (%)":          72.0,
        "Rougher Conc Grade (%Cu)":    18.0,
    }
    actual_doses = {
        "SIPX Dose (g/t)":       40.0,
        "Frother Dose (g/t)":    20.0,
        "Lime Dose (kg/t)":       2.5,
        "Depressant Dose (g/t)": 25.0,
    }

    res = opt.optimize(conditions, actual_doses)

    print(f"Current recovery:  {res['predicted_recovery_current']:.2f}%")
    print(f"Optimal recovery:  {res['predicted_recovery_optimal']:.2f}%")
    print(f"Recovery gain:     +{res['recovery_gain']:.3f} pp\n")
    print(f"{'Reagent':<28} {'Actual':>8} {'Optimal':>8} {'Gap %':>7} {'Status'}")
    print("-" * 65)
    for r, g in res["gaps"].items():
        print(f"{r:<28} {g['actual']:>8.1f} {g['optimal']:>8.1f} "
              f"{g['gap_pct']:>6.1f}%  [{g['colour']}] {g['label']}")

    os.unlink(tmp)
    print("\nEngine 1 OK ✓")
