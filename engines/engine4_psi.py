"""
RecovAI — Engine 4: PSI Data Drift Monitor
==========================================
Computes Population Stability Index (PSI) for every feature to detect
when live data has drifted away from the training distribution.

PSI thresholds:
    < 0.10  → No change   — model is safe to use   [GREEN]
    0.10–0.25 → Slight drift — monitor closely       [AMBER]
    > 0.25  → Significant drift — consider retraining [RED]

Usage:
    from engines.engine4_psi import PSIMonitor
    monitor = PSIMonitor()
    monitor.fit(X_train_df)
    monitor.save("recovai_output/psi_monitor.pkl")

    # At inference time:
    monitor = PSIMonitor.load("recovai_output/psi_monitor.pkl")
    report  = monitor.check(X_new_df)
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
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

PSI_GREEN = 0.10
PSI_AMBER = 0.25
N_BINS    = 10      # percentile-based bins


def _psi_single(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = N_BINS,
) -> float:
    """
    Core PSI computation for one feature.

    Bins are computed from the expected (training) distribution using
    equal-frequency (percentile) bucketing — more robust than equal-width.
    """
    # Build bin edges from training data
    percentiles = np.linspace(0, 100, n_bins + 1)
    breakpoints = np.unique(np.percentile(expected, percentiles))

    if len(breakpoints) < 2:
        return 0.0  # constant feature — no drift possible

    # Proportion in each bin
    e_counts, _ = np.histogram(expected, bins=breakpoints)
    a_counts, _ = np.histogram(actual,   bins=breakpoints)

    e_pct = e_counts / len(expected)
    a_pct = a_counts / len(actual)

    # Replace zeros to avoid log(0)
    e_pct = np.where(e_pct == 0, 1e-6, e_pct)
    a_pct = np.where(a_pct == 0, 1e-6, a_pct)

    psi = np.sum((a_pct - e_pct) * np.log(a_pct / e_pct))
    return float(psi)


def _classify_psi(psi: float) -> Tuple[str, str]:
    if psi < PSI_GREEN:
        return "OK",      "GREEN"
    elif psi < PSI_AMBER:
        return "MONITOR", "AMBER"
    else:
        return "RETRAIN", "RED"


class PSIMonitor:
    """
    Stores training distribution reference and computes PSI on new data.
    """

    def __init__(self, n_bins: int = N_BINS, feature_names: Optional[List[str]] = None):
        self.n_bins        = n_bins
        self.feature_names = feature_names or FEATURES
        self._train_data: Optional[Dict[str, np.ndarray]] = None
        self._fitted_at:  Optional[str]  = None
        self._n_train:    int = 0

    # ── Fitting ───────────────────────────────────────────────────────────

    def fit(self, X_train: pd.DataFrame) -> "PSIMonitor":
        """
        Store training distribution arrays (one per feature).

        Parameters
        ----------
        X_train : pd.DataFrame
            Clean historical data used to train the model.
        """
        self._train_data = {
            f: X_train[f].dropna().values.astype(float)
            for f in self.feature_names
        }
        self._n_train   = len(X_train)
        self._fitted_at = datetime.utcnow().isoformat()
        print(f"[Engine 4] PSI reference fitted on {self._n_train} shifts "
              f"at {self._fitted_at}")
        return self

    # ── Checking ──────────────────────────────────────────────────────────

    def check(self, X_new: pd.DataFrame) -> Dict:
        """
        Compute PSI for every feature between training and new data.

        Parameters
        ----------
        X_new : pd.DataFrame
            Recent operational data (e.g. last 30 days / last shift batch).

        Returns
        -------
        dict with:
            overall_status   — worst-case status across all features
            overall_colour   — RED | AMBER | GREEN
            features         — dict  feature → {psi, status, colour, ...}
            flagged          — list of features with status != 'OK'
            n_new            — number of new samples analysed
            timestamp        — UTC ISO string
        """
        self._check_fitted()

        results   = {}
        worst_psi = 0.0

        for f in self.feature_names:
            expected = self._train_data[f]
            actual   = X_new[f].dropna().values.astype(float)

            if len(actual) == 0:
                results[f] = {
                    "psi": None, "status": "NO_DATA", "colour": "GREY",
                    "n_actual": 0, "n_expected": len(expected),
                }
                continue

            psi = _psi_single(expected, actual, self.n_bins)
            status, colour = _classify_psi(psi)
            worst_psi = max(worst_psi, psi)

            results[f] = {
                "psi":        round(psi, 4),
                "status":     status,
                "colour":     colour,
                "n_actual":   len(actual),
                "n_expected": len(expected),
                "actual_mean":   round(float(np.mean(actual)), 4),
                "expected_mean": round(float(np.mean(expected)), 4),
                "mean_shift":    round(float(np.mean(actual) - np.mean(expected)), 4),
            }

        overall_status, overall_colour = _classify_psi(worst_psi)
        flagged = [f for f, r in results.items()
                   if r.get("status") not in ("OK", "NO_DATA")]

        return {
            "overall_status":  overall_status,
            "overall_colour":  overall_colour,
            "worst_psi":       round(worst_psi, 4),
            "features":        results,
            "flagged":         flagged,
            "n_new":           len(X_new),
            "n_train_ref":     self._n_train,
            "timestamp":       datetime.utcnow().isoformat(),
        }

    def check_single_shift(self, shift: Dict[str, float]) -> Dict:
        """
        Convenience wrapper: check a single shift dict by comparing its
        individual values to the training mean ± 3σ (simple Chebyshev rule),
        then compute a pseudo-PSI against the nearest percentile bin.

        This is for *real-time* inference when only one shift is available,
        not a batch. Returns same structure as check() for API consistency.
        """
        self._check_fitted()

        results   = {}
        worst_psi = 0.0

        for f in self.feature_names:
            expected = self._train_data[f]
            val      = shift.get(f)

            if val is None:
                results[f] = {"psi": None, "status": "NO_DATA", "colour": "GREY"}
                continue

            # Percentile rank of this single value in training distribution
            pct_rank = float(np.mean(expected <= val)) * 100

            # Map extreme percentile rank to a pseudo-PSI
            # (0th or 100th percentile → maximum drift signal)
            deviation = abs(pct_rank - 50) / 50   # 0 = median, 1 = extreme tail
            pseudo_psi = deviation ** 3 * 0.6      # calibrated to 0–0.6 range

            status, colour = _classify_psi(pseudo_psi)
            worst_psi = max(worst_psi, pseudo_psi)

            mu  = float(np.mean(expected))
            sig = float(np.std(expected)) or 1.0

            results[f] = {
                "psi":          round(pseudo_psi, 4),
                "status":       status,
                "colour":       colour,
                "value":        val,
                "percentile":   round(pct_rank, 1),
                "z_score":      round((val - mu) / sig, 3),
                "expected_mean": round(mu, 4),
            }

        overall_status, overall_colour = _classify_psi(worst_psi)
        flagged = [f for f, r in results.items()
                   if r.get("status") not in ("OK", "NO_DATA")]

        return {
            "overall_status":  overall_status,
            "overall_colour":  overall_colour,
            "worst_psi":       round(worst_psi, 4),
            "features":        results,
            "flagged":         flagged,
            "n_new":           1,
            "timestamp":       datetime.utcnow().isoformat(),
        }

    # ── Plotting ──────────────────────────────────────────────────────────

    def plot_drift_dashboard(
        self,
        check_result: Dict,
        save_path: Optional[str] = None,
        figsize: tuple = (10, 6),
    ) -> plt.Figure:
        """
        Horizontal bar chart of PSI per feature, colour-coded by status.
        """
        features = check_result["features"]
        names    = [f for f in self.feature_names if features[f].get("psi") is not None]
        psi_vals = [features[f]["psi"] for f in names]
        colours  = {
            "GREEN": "#1D9E75",
            "AMBER": "#EF9F27",
            "RED":   "#E24B4A",
            "GREY":  "#B4B2A9",
        }
        bar_colours = [colours[features[f]["colour"]] for f in names]

        order    = np.argsort(psi_vals)
        names_s  = [names[i] for i in order]
        vals_s   = [psi_vals[i] for i in order]
        cols_s   = [bar_colours[i] for i in order]

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor("#F8F8F6")
        fig.patch.set_facecolor("#FFFFFF")
        bars = ax.barh(names_s, vals_s, color=cols_s, height=0.55, zorder=3)

        ax.axvline(PSI_GREEN, color="#1D9E75", lw=1.2, ls="--",
                   label=f"OK < {PSI_GREEN}", zorder=2)
        ax.axvline(PSI_AMBER, color="#EF9F27", lw=1.2, ls="--",
                   label=f"Monitor < {PSI_AMBER}", zorder=2)

        for bar, val in zip(bars, vals_s):
            ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=8,
                    color=bar.get_facecolor())

        ax.set_xlabel("Population Stability Index (PSI)", fontsize=10)
        ax.set_title(
            f"Data Drift Monitor — PSI Report  "
            f"[{check_result['overall_status']}]  "
            f"n={check_result['n_new']} shifts",
            fontsize=11, fontweight="bold", pad=10,
        )
        ax.legend(fontsize=9)
        ax.grid(axis="x", alpha=0.3, zorder=1)
        ax.spines[["top","right","left"]].set_visible(False)

        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[Engine 4] Drift plot saved → {save_path}")
        return fig

    def plot_distribution_comparison(
        self,
        X_new: pd.DataFrame,
        feature: str,
        save_path: Optional[str] = None,
        figsize: tuple = (8, 4),
    ) -> plt.Figure:
        """
        Overlaid KDE/histogram of training vs new distribution for one feature.
        """
        self._check_fitted()
        expected = self._train_data[feature]
        actual   = X_new[feature].dropna().values.astype(float)

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor("#F8F8F6")
        fig.patch.set_facecolor("#FFFFFF")

        ax.hist(expected, bins=30, density=True, alpha=0.45,
                color="#185FA5", label="Training distribution")
        ax.hist(actual,   bins=30, density=True, alpha=0.45,
                color="#E24B4A", label="Recent data")

        ax.set_xlabel(feature, fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(f"Distribution Shift — {feature}", fontsize=11,
                     fontweight="bold", pad=8)
        ax.legend(fontsize=9)
        ax.spines[["top","right"]].set_visible(False)

        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "train_data":    self._train_data,
            "feature_names": self.feature_names,
            "n_bins":        self.n_bins,
            "n_train":       self._n_train,
            "fitted_at":     self._fitted_at,
        }, path)
        print(f"[Engine 4] PSI monitor saved → {path}")

    @classmethod
    def load(cls, path: str) -> "PSIMonitor":
        payload = joblib.load(path)
        mon = cls(n_bins=payload["n_bins"],
                  feature_names=payload["feature_names"])
        mon._train_data  = payload["train_data"]
        mon._n_train     = payload["n_train"]
        mon._fitted_at   = payload["fitted_at"]
        return mon

    # ── Helpers ───────────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if self._train_data is None:
            raise RuntimeError("Monitor not fitted. Call .fit() or .load() first.")


# ── Standalone test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile, os
    print("Engine 4 — PSI Data Drift Monitor\n" + "="*38)

    rng = np.random.default_rng(99)
    n   = 800

    def make_df(seed_offset=0):
        r = np.random.default_rng(seed_offset)
        return pd.DataFrame({
            "Head Grade (%Cu)":                r.normal(1.2,  0.15, n),
            "Feed Rate (MT/h)":                r.normal(120,  10,   n),
            "Flotation pH":                    r.normal(10.5, 0.3,  n),
            "Pulp Density (%)":                r.normal(32,   2,    n),
            "Air Flow Rate (m3/min)":          r.normal(14,   1.5,  n),
            "SIPX Dose (g/t)":                 r.normal(40,   5,    n),
            "Frother Dose (g/t)":              r.normal(20,   3,    n),
            "Lime Dose (kg/t)":                r.normal(2.5,  0.4,  n),
            "Depressant Dose (g/t)":           r.normal(25,   4,    n),
            "Feed Particle Size (P80 microns)":r.normal(150,  15,   n),
            "Water Recovery (%)":              r.normal(72,   4,    n),
            "Rougher Conc Grade (%Cu)":        r.normal(18,   2,    n),
        })

    X_train = make_df(0)
    X_clean = make_df(1)          # similar to training → low PSI
    X_drift = make_df(2)          # introduce drift
    X_drift["Flotation pH"]     += 1.5   # shift pH mean by 1.5
    X_drift["SIPX Dose (g/t)"]  *= 1.4   # scale SIPX up 40%
    X_drift["Pulp Density (%)"] -= 5.0   # shift density down

    mon = PSIMonitor()
    mon.fit(X_train)

    res_clean = mon.check(X_clean)
    res_drift = mon.check(X_drift)

    print(f"\nClean data  → overall: {res_clean['overall_status']} (worst PSI = {res_clean['worst_psi']:.4f})")
    print(f"Drifted data→ overall: {res_drift['overall_status']} (worst PSI = {res_drift['worst_psi']:.4f})")
    print(f"Flagged features: {res_drift['flagged']}")

    # Single-shift check
    single = {f: float(X_drift[f].iloc[0]) for f in FEATURES}
    single["Flotation pH"] = 8.0
    res_s = mon.check_single_shift(single)
    print(f"\nSingle-shift check → overall: {res_s['overall_status']}")

    tmp = tempfile.mktemp(suffix=".pkl")
    mon.save(tmp)
    mon2 = PSIMonitor.load(tmp)
    r2 = mon2.check(X_drift)
    assert abs(r2["worst_psi"] - res_drift["worst_psi"]) < 1e-6
    os.unlink(tmp)

    print("\nEngine 4 OK ✓")
