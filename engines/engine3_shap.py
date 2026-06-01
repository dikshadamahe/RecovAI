"""
RecovAI — Engine 3: SHAP Explainability Module
===============================================
Provides per-prediction SHAP decomposition using TreeExplainer.
Generates waterfall plots (single prediction) and beeswarm plots
(global model behaviour). Returns structured data for the dashboard
and top drivers for the NLP report.

Usage:
    from engines.engine3_shap import ShapExplainer
    exp = ShapExplainer(model)
    result = exp.explain_shift(shift_dict)
    exp.plot_waterfall(result, save_path="output/waterfall.png")
    exp.plot_beeswarm(X_df, save_path="output/beeswarm.png")
"""

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")       # non-interactive backend — safe for servers
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Optional, Union
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

# Colours that match the dashboard palette
CLR_POS  = "#1D9E75"   # teal-green — feature pushed recovery up
CLR_NEG  = "#E24B4A"   # red        — feature pushed recovery down
CLR_BASE = "#888780"   # grey


class ShapExplainer:
    """
    SHAP explainer wrapper. Works with XGBoost, RandomForest, or any
    tree-based model supported by shap.TreeExplainer.
    """

    def __init__(self, model, feature_names: Optional[List[str]] = None):
        self.model         = model
        self.feature_names = feature_names or FEATURES
        self.explainer     = shap.TreeExplainer(model)
        self._beeswarm_shap_values = None   # cache for repeated beeswarm calls

    # ── Core explain ─────────────────────────────────────────────────────

    def explain_shift(self, shift: Dict[str, float]) -> Dict:
        """
        Explain a single shift prediction.

        Parameters
        ----------
        shift : dict  {feature_name: value}

        Returns
        -------
        dict with:
            shap_values      — list of float, one per feature
            base_value       — model expected value (baseline recovery)
            prediction       — predicted recovery for this shift
            feature_values   — dict feature→value
            top_positive     — list of (feature, shap_val) top 3 positive drivers
            top_negative     — list of (feature, shap_val) top 3 negative drivers
            top_3_drivers    — same list, signed, top 3 by |shap|
            explanation_obj  — shap.Explanation (for shap plot functions)
        """
        x_arr = np.array([[shift[f] for f in self.feature_names]])

        shap_vals  = self.explainer.shap_values(x_arr)

        # Handle multi-output models (some RF wrappers return list)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]

        sv       = shap_vals[0]            # 1-D array, one value per feature
        base_val = float(self.explainer.expected_value
                         if not isinstance(self.explainer.expected_value, (list, np.ndarray))
                         else self.explainer.expected_value[0])
        pred     = float(base_val + sv.sum())

        # Build shap.Explanation for native plot functions
        explanation = shap.Explanation(
            values=sv,
            base_values=base_val,
            data=x_arr[0],
            feature_names=self.feature_names,
        )

        # Rank features
        pairs = list(zip(self.feature_names, sv.tolist()))
        pairs_abs = sorted(pairs, key=lambda p: abs(p[1]), reverse=True)

        top_positive = [(f, round(v, 4)) for f, v in pairs_abs if v > 0][:3]
        top_negative = [(f, round(v, 4)) for f, v in pairs_abs if v < 0][:3]
        top_3        = [(f, round(v, 4)) for f, v in pairs_abs[:3]]

        return {
            "shap_values":    [round(float(v), 4) for v in sv],
            "base_value":     round(base_val, 4),
            "prediction":     round(pred, 4),
            "feature_values": {f: shift[f] for f in self.feature_names},
            "top_positive":   top_positive,
            "top_negative":   top_negative,
            "top_3_drivers":  top_3,
            "explanation_obj": explanation,
        }

    # ── Plots ─────────────────────────────────────────────────────────────

    def plot_waterfall(
        self,
        result: Dict,
        save_path: Optional[str] = None,
        max_display: int = 10,
        figsize: tuple = (9, 5),
    ) -> plt.Figure:
        """
        Waterfall plot — shows how each feature moves the prediction
        away from the base value for ONE shift.
        """
        exp = result["explanation_obj"]

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor("#F8F8F6")
        fig.patch.set_facecolor("#FFFFFF")

        sv   = np.array(result["shap_values"])
        fv   = [result["feature_values"][f] for f in self.feature_names]
        fn   = self.feature_names
        base = result["base_value"]
        pred = result["prediction"]

        # Sort by |shap| descending, keep top max_display
        order   = np.argsort(np.abs(sv))[::-1][:max_display]
        sv_top  = sv[order]
        fn_top  = [fn[i] for i in order]
        fv_top  = [fv[i] for i in order]

        # Cumulative running total (waterfall logic)
        running = base
        lefts, widths, colours, labels = [], [], [], []
        for s in sv_top[::-1]:
            widths.append(s)
            lefts.append(running if s >= 0 else running + s)
            colours.append(CLR_POS if s >= 0 else CLR_NEG)
            running += s

        y_pos = range(len(sv_top))
        bars  = ax.barh(
            list(y_pos), widths[::-1], left=lefts[::-1],
            color=colours[::-1], height=0.55, zorder=3
        )

        # Feature labels
        ytick_labels = [
            f"{fn_top[::-1][i]}  = {fv_top[::-1][i]:.2f}"
            for i in range(len(sv_top))
        ]
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(ytick_labels, fontsize=9)

        # SHAP value annotations on bars
        for bar, val in zip(bars, widths[::-1]):
            xc = bar.get_x() + bar.get_width() / 2
            ax.text(xc, bar.get_y() + bar.get_height() / 2,
                    f"{val:+.2f}", ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold", zorder=4)

        # Base and prediction lines
        ax.axvline(base, color=CLR_BASE, lw=1.2, ls="--", zorder=2, label=f"Base = {base:.1f}%")
        ax.axvline(pred, color="#185FA5", lw=1.5, ls="-",  zorder=2, label=f"Prediction = {pred:.2f}%")

        ax.set_xlabel("Predicted Cu Recovery (%)", fontsize=10)
        ax.set_title("SHAP Waterfall — Prediction Drivers (Single Shift)",
                     fontsize=11, fontweight="bold", pad=10)
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(axis="x", alpha=0.3, zorder=1)
        ax.spines[["top","right","left"]].set_visible(False)

        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[Engine 3] Waterfall saved → {save_path}")
        return fig

    def plot_beeswarm(
        self,
        X: pd.DataFrame,
        save_path: Optional[str] = None,
        max_display: int = 10,
        figsize: tuple = (9, 6),
    ) -> plt.Figure:
        """
        Beeswarm plot — global model behaviour across ALL shifts.
        Each dot = one shift; colour = feature value; x-axis = SHAP impact.
        """
        X_arr = X[self.feature_names].values.astype(float)
        shap_vals = self.explainer.shap_values(X_arr)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]

        # Cache for potential re-use
        self._beeswarm_shap_values = shap_vals

        # Rank features by mean |SHAP|
        mean_abs = np.mean(np.abs(shap_vals), axis=0)
        order    = np.argsort(mean_abs)[::-1][:max_display]

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor("#F8F8F6")
        fig.patch.set_facecolor("#FFFFFF")

        cmap = plt.cm.RdYlGn
        y_positions = list(range(len(order)))

        for row_idx, feat_idx in enumerate(order[::-1]):
            sv_col   = shap_vals[:, feat_idx]
            feat_col = X_arr[:, feat_idx]

            # Normalise feature values for colour mapping
            vmin, vmax = feat_col.min(), feat_col.max()
            norm_vals  = (feat_col - vmin) / max(vmax - vmin, 1e-9)

            # Jitter y positions for beeswarm effect
            jitter = np.random.default_rng(feat_idx).uniform(-0.25, 0.25, len(sv_col))
            y_jit  = row_idx + jitter

            ax.scatter(
                sv_col, y_jit,
                c=cmap(norm_vals), alpha=0.55, s=14, zorder=3,
                linewidths=0, rasterized=True
            )

        feat_labels = [self.feature_names[i] for i in order[::-1]]
        ax.set_yticks(y_positions)
        ax.set_yticklabels(feat_labels, fontsize=9)
        ax.axvline(0, color=CLR_BASE, lw=1, ls="--", zorder=2)
        ax.set_xlabel("SHAP value  (impact on predicted Cu Recovery)", fontsize=10)
        ax.set_title("SHAP Beeswarm — Global Feature Importance", fontsize=11,
                     fontweight="bold", pad=10)
        ax.grid(axis="x", alpha=0.3, zorder=1)
        ax.spines[["top","right","left"]].set_visible(False)

        # Colour bar legend
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.01, shrink=0.6)
        cbar.set_label("Feature value\n(low → high)", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[Engine 3] Beeswarm saved → {save_path}")
        return fig

    def plot_summary_bar(
        self,
        X: pd.DataFrame,
        save_path: Optional[str] = None,
        figsize: tuple = (8, 5),
    ) -> plt.Figure:
        """
        Horizontal bar chart of mean |SHAP| — clean alternative to beeswarm
        for executive summaries.
        """
        X_arr = X[self.feature_names].values.astype(float)
        shap_vals = self.explainer.shap_values(X_arr)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]

        mean_abs = np.mean(np.abs(shap_vals), axis=0)
        order    = np.argsort(mean_abs)
        labels   = [self.feature_names[i] for i in order]
        values   = mean_abs[order]
        colours  = [CLR_POS if v > np.median(values) else CLR_BASE for v in values]

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor("#F8F8F6")
        fig.patch.set_facecolor("#FFFFFF")
        ax.barh(labels, values, color=colours, height=0.6, zorder=3)
        ax.set_xlabel("Mean |SHAP value|", fontsize=10)
        ax.set_title("Feature Importance — Mean Absolute SHAP", fontsize=11,
                     fontweight="bold", pad=10)
        ax.grid(axis="x", alpha=0.3, zorder=1)
        ax.spines[["top","right","left"]].set_visible(False)
        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def get_dashboard_payload(self, result: Dict) -> Dict:
        """
        Returns a JSON-serialisable dict ready for the FastAPI → frontend.
        Strips the shap.Explanation object.
        """
        return {
            k: v for k, v in result.items()
            if k != "explanation_obj"
        }


# ── Standalone test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from sklearn.ensemble import GradientBoostingRegressor

    print("Engine 3 — SHAP Explainability\n" + "="*38)

    rng  = np.random.default_rng(0)
    n    = 400
    X_tr = pd.DataFrame({
        "Head Grade (%Cu)":                rng.normal(1.2, 0.15, n),
        "Feed Rate (MT/h)":                rng.normal(120, 10, n),
        "Flotation pH":                    rng.normal(10.5, 0.3, n),
        "Pulp Density (%)":                rng.normal(32, 2, n),
        "Air Flow Rate (m3/min)":          rng.normal(14, 1.5, n),
        "SIPX Dose (g/t)":                 rng.normal(40, 5, n),
        "Frother Dose (g/t)":              rng.normal(20, 3, n),
        "Lime Dose (kg/t)":                rng.normal(2.5, 0.4, n),
        "Depressant Dose (g/t)":           rng.normal(25, 4, n),
        "Feed Particle Size (P80 microns)":rng.normal(150, 15, n),
        "Water Recovery (%)":              rng.normal(72, 4, n),
        "Rougher Conc Grade (%Cu)":        rng.normal(18, 2, n),
    })
    y_tr = (80 + X_tr["Head Grade (%Cu)"] * 8
               - (X_tr["Flotation pH"] - 10.5).abs() * 2
               + rng.normal(0, 0.5, n))

    model = GradientBoostingRegressor(n_estimators=80, random_state=0)
    model.fit(X_tr, y_tr)

    explainer = ShapExplainer(model)

    shift = X_tr.iloc[0].to_dict()
    result = explainer.explain_shift(shift)

    print(f"Base value:  {result['base_value']:.2f}%")
    print(f"Prediction:  {result['prediction']:.2f}%")
    print("Top 3 drivers:")
    for f, v in result["top_3_drivers"]:
        sign = "▲" if v > 0 else "▼"
        print(f"  {sign} {f}: {v:+.3f} pp")

    import tempfile
    tmp_w = tempfile.mktemp(suffix="_waterfall.png")
    tmp_b = tempfile.mktemp(suffix="_beeswarm.png")
    explainer.plot_waterfall(result, save_path=tmp_w)
    explainer.plot_beeswarm(X_tr.head(100), save_path=tmp_b)
    os.unlink(tmp_w); os.unlink(tmp_b)

    print("\nEngine 3 OK ✓")
