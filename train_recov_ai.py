"""
╔══════════════════════════════════════════════════════════════════════╗
║           RecovAI — Copper Recovery Prediction Model                ║
║           XGBoost + Random Forest Training Script                   ║
║           Target: R² > 0.85 | MAE < 0.5% recovery                  ║
╚══════════════════════════════════════════════════════════════════════╝

Usage:
    python train_recov_ai.py

Outputs (saved to ./outputs/):
    - xgboost_model.pkl
    - random_forest_model.pkl
    - feature_importance.png
    - model_comparison.png
    - predictions_vs_actual.png
    - training_report.txt
"""

import os
import warnings
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────
# 0. CONFIG
# ─────────────────────────────────────────────────────────────────────
DATA_PATH   = "data/processed/ML_Dataset_Copper_TARGET85.csv"
OUTPUT_DIR  = "outputs"
TARGET_COL  = "Recovery (%)"
TRAIN_CUTOFF = "2026-01-01"   # Time-series split — everything before = train

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 65)
print("   RecovAI — Copper Recovery Prediction Training Pipeline")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────
print("\n[1/7] Loading dataset...")
df = pd.read_csv(DATA_PATH, header=1)
df = df.dropna(how="all").reset_index(drop=True)
df["Date_parsed"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
df = df.sort_values("Date_parsed").reset_index(drop=True)

print(f"      Rows: {len(df):,} | Columns: {len(df.columns)}")
print(f"      Date range: {df['Date_parsed'].min().date()} → {df['Date_parsed'].max().date()}")
print(f"      Recovery % — Mean: {df[TARGET_COL].mean():.2f}  "
      f"Std: {df[TARGET_COL].std():.2f}  "
      f"Min: {df[TARGET_COL].min():.2f}  "
      f"Max: {df[TARGET_COL].max():.2f}")

# ─────────────────────────────────────────────────────────────────────
# 2. FEATURE SELECTION
#    Remove leakage columns (derived from Recovery itself at shift-end)
#    Keep only inputs an operator would KNOW at shift START
# ─────────────────────────────────────────────────────────────────────
print("\n[2/7] Selecting features (removing leakage columns)...")

# These columns are calculated FROM Recovery — using them would be cheating
LEAKAGE_COLS = [
    "COPPER IN CONCENTRATE (MT)",   # = ore × grade × recovery
    "COPPER IN TAILINGS (MT)",      # = copper_in - copper_conc
    "Concentrate Production (MT)",  # derived from recovery
    "COPPER IN HEAD (MT)",          # = ore × grade (borderline but keep out)
    "TAILINGS (MT)",                # = ore - concentrate
]

# Drop non-feature columns
DROP_COLS = LEAKAGE_COLS + [
    "Date", "Date_parsed", "Shift", "Source",
    "Estimated Feed Condition",     # string version — use Feed_Condition_Num
    "T Reagent (cc)",               # constant 1200 across all rows (zero variance)
]

FEATURE_COLS = [c for c in df.columns if c not in DROP_COLS + [TARGET_COL]]

print(f"      Features used: {len(FEATURE_COLS)}")
for f in FEATURE_COLS:
    print(f"        • {f}")

# ─────────────────────────────────────────────────────────────────────
# 3. TRAIN / TEST SPLIT  (time-based — never random for time-series)
# ─────────────────────────────────────────────────────────────────────
print(f"\n[3/7] Time-series train/test split (cutoff: {TRAIN_CUTOFF})...")

df["Date_parsed"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
train_mask = df["Date_parsed"] < TRAIN_CUTOFF
test_mask  = df["Date_parsed"] >= TRAIN_CUTOFF

X_train = df.loc[train_mask, FEATURE_COLS]
y_train = df.loc[train_mask, TARGET_COL]
X_test  = df.loc[test_mask,  FEATURE_COLS]
y_test  = df.loc[test_mask,  TARGET_COL]

print(f"      Train: {len(X_train):,} rows  "
      f"({df.loc[train_mask,'Date_parsed'].min().date()} → "
      f"{df.loc[train_mask,'Date_parsed'].max().date()})")
print(f"      Test:  {len(X_test):,} rows   "
      f"({df.loc[test_mask,'Date_parsed'].min().date()} → "
      f"{df.loc[test_mask,'Date_parsed'].max().date()})")

# Handle any NaNs (impute with column median)
X_train = X_train.fillna(X_train.median())
X_test  = X_test.fillna(X_train.median())

# ─────────────────────────────────────────────────────────────────────
# 4. TRAIN XGBOOST
# ─────────────────────────────────────────────────────────────────────
print("\n[4/7] Training XGBoost model...")

xgb_model = XGBRegressor(
    n_estimators    = 400,
    max_depth       = 5,
    learning_rate   = 0.05,
    subsample       = 0.8,
    colsample_bytree= 0.8,
    min_child_weight= 3,
    reg_alpha       = 0.1,
    reg_lambda      = 1.0,
    random_state    = 42,
    verbosity       = 0,
    eval_metric     = "rmse",
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

y_pred_xgb = xgb_model.predict(X_test)
r2_xgb   = r2_score(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
mae_xgb  = mean_absolute_error(y_test, y_pred_xgb)

print(f"      XGBoost  →  R²: {r2_xgb:.4f}  |  RMSE: {rmse_xgb:.4f}%  |  MAE: {mae_xgb:.4f}%")

# ─────────────────────────────────────────────────────────────────────
# 5. TRAIN RANDOM FOREST
# ─────────────────────────────────────────────────────────────────────
print("\n[5/7] Training Random Forest model...")

rf_model = RandomForestRegressor(
    n_estimators = 300,
    max_depth    = 12,
    min_samples_split = 5,
    min_samples_leaf  = 3,
    max_features = "sqrt",
    random_state = 42,
    n_jobs       = -1,
)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
r2_rf   = r2_score(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mae_rf  = mean_absolute_error(y_test, y_pred_rf)

print(f"      Random Forest →  R²: {r2_rf:.4f}  |  RMSE: {rmse_rf:.4f}%  |  MAE: {mae_rf:.4f}%")

# ─────────────────────────────────────────────────────────────────────
# 6. SAVE MODELS
# ─────────────────────────────────────────────────────────────────────
print("\n[6/7] Saving models and plots...")

xgb_path = os.path.join(OUTPUT_DIR, "xgboost_model.pkl")
rf_path  = os.path.join(OUTPUT_DIR, "random_forest_model.pkl")

with open(xgb_path, "wb") as f:
    pickle.dump({"model": xgb_model, "features": FEATURE_COLS}, f)

with open(rf_path, "wb") as f:
    pickle.dump({"model": rf_model, "features": FEATURE_COLS}, f)

print(f"      Saved: {xgb_path}")
print(f"      Saved: {rf_path}")

# ─────────────────────────────────────────────────────────────────────
# 7. PLOTS
# ─────────────────────────────────────────────────────────────────────

COLORS = {
    "xgb"    : "#1F4E79",
    "rf"     : "#2E75B6",
    "actual" : "#375623",
    "grid"   : "#E8E8E8",
    "accent" : "#C55A11",
}

# ── Plot 1: Feature Importance (XGBoost) ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.patch.set_facecolor("#F8F9FA")

for ax, (model, model_name, color) in zip(
    axes,
    [(xgb_model, "XGBoost", COLORS["xgb"]),
     (rf_model,  "Random Forest", COLORS["rf"])]
):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        importances = model.feature_importances_

    fi = pd.Series(importances, index=FEATURE_COLS).sort_values(ascending=True)
    top_n = fi.tail(15)

    bars = ax.barh(top_n.index, top_n.values, color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Feature Importance Score", fontsize=11, color="#333333")
    ax.set_title(f"{model_name}\nFeature Importance (Top 15)", fontsize=13, fontweight="bold", color="#1F4E79", pad=12)
    ax.set_facecolor("#F8F9FA")
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.8)
    ax.spines[["top","right","left"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=9)

    # Value labels
    for bar, val in zip(bars, top_n.values):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", ha="left", fontsize=8, color="#555555")

plt.suptitle("RecovAI — Feature Importance Analysis\nCopper Recovery Prediction",
             fontsize=14, fontweight="bold", color="#1F4E79", y=1.01)
plt.tight_layout()
fi_path = os.path.join(OUTPUT_DIR, "feature_importance.png")
plt.savefig(fi_path, dpi=150, bbox_inches="tight", facecolor="#F8F9FA")
plt.close()
print(f"      Saved: {fi_path}")

# ── Plot 2: Predictions vs Actual ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor("#F8F9FA")

test_dates = df.loc[test_mask, "Date_parsed"].values

for ax, (y_pred, model_name, color, r2, rmse, mae) in zip(
    axes,
    [(y_pred_xgb, "XGBoost",      COLORS["xgb"], r2_xgb, rmse_xgb, mae_xgb),
     (y_pred_rf,  "Random Forest", COLORS["rf"],  r2_rf,  rmse_rf,  mae_rf)]
):
    ax.scatter(y_test, y_pred, alpha=0.45, s=18, color=color, edgecolors="none", label="Predictions")
    mn, mx = min(y_test.min(), y_pred.min()) - 0.3, max(y_test.max(), y_pred.max()) + 0.3
    ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Perfect fit", alpha=0.7)

    ax.set_xlabel("Actual Recovery (%)", fontsize=11)
    ax.set_ylabel("Predicted Recovery (%)", fontsize=11)
    ax.set_title(f"{model_name}", fontsize=13, fontweight="bold", color="#1F4E79")
    ax.set_facecolor("#F8F9FA")
    ax.grid(color=COLORS["grid"], linewidth=0.8)
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(fontsize=9)

    # Metrics box
    metrics_text = f"R²   = {r2:.4f}\nRMSE = {rmse:.4f}%\nMAE  = {mae:.4f}%"
    ax.text(0.04, 0.96, metrics_text, transform=ax.transAxes,
            fontsize=10, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor=color, alpha=0.9, linewidth=1.5))

    # Target line annotation
    target_met = "✓ TARGET MET" if r2 >= 0.85 else "✗ Below target"
    target_color = COLORS["accent"] if r2 >= 0.85 else "red"
    ax.text(0.96, 0.04, target_met, transform=ax.transAxes,
            fontsize=10, fontweight="bold", color=target_color,
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=target_color, alpha=0.9))

plt.suptitle("RecovAI — Predicted vs Actual Recovery (%)\nTest Set Evaluation",
             fontsize=14, fontweight="bold", color="#1F4E79")
plt.tight_layout()
pva_path = os.path.join(OUTPUT_DIR, "predictions_vs_actual.png")
plt.savefig(pva_path, dpi=150, bbox_inches="tight", facecolor="#F8F9FA")
plt.close()
print(f"      Saved: {pva_path}")

# ── Plot 3: Model Comparison Bar Chart ───────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.patch.set_facecolor("#F8F9FA")

metrics = {
    "R²"  : ([r2_xgb,   r2_rf],   0.85,  True),
    "RMSE": ([rmse_xgb, rmse_rf],  0.5,   False),
    "MAE" : ([mae_xgb,  mae_rf],   0.5,   False),
}
models = ["XGBoost", "Random Forest"]

for ax, (metric_name, (vals, threshold, higher_better)) in zip(axes, metrics.items()):
    bar_colors = []
    for v in vals:
        if higher_better:
            bar_colors.append(COLORS["xgb"] if v >= threshold else "#C00000")
        else:
            bar_colors.append(COLORS["xgb"] if v <= threshold else "#C00000")

    bars = ax.bar(models, vals, color=bar_colors, alpha=0.85, edgecolor="white", linewidth=0.5, width=0.5)
    ax.axhline(threshold, color=COLORS["accent"], linewidth=1.5, linestyle="--",
               label=f"Target {'≥' if higher_better else '≤'} {threshold}")
    ax.set_title(metric_name, fontsize=13, fontweight="bold", color="#1F4E79")
    ax.set_facecolor("#F8F9FA")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax.spines[["top","right","left"]].set_visible(False)
    ax.legend(fontsize=9)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

plt.suptitle("RecovAI — Model Performance Comparison\nXGBoost vs Random Forest",
             fontsize=14, fontweight="bold", color="#1F4E79")
plt.tight_layout()
cmp_path = os.path.join(OUTPUT_DIR, "model_comparison.png")
plt.savefig(cmp_path, dpi=150, bbox_inches="tight", facecolor="#F8F9FA")
plt.close()
print(f"      Saved: {cmp_path}")

# ── Plot 4: Time-series prediction on test set ───────────────────────
fig, ax = plt.subplots(figsize=(16, 5))
fig.patch.set_facecolor("#F8F9FA")
ax.set_facecolor("#F8F9FA")

idx = range(len(y_test))
ax.plot(idx, y_test.values,   color=COLORS["actual"], linewidth=1.2, label="Actual Recovery",    alpha=0.9)
ax.plot(idx, y_pred_xgb,      color=COLORS["xgb"],    linewidth=1.0, label="XGBoost Predicted",  alpha=0.85, linestyle="--")
ax.plot(idx, y_pred_rf,       color=COLORS["rf"],     linewidth=1.0, label="RF Predicted",        alpha=0.85, linestyle=":")
ax.axhline(85, color="red", linewidth=1.0, linestyle="-.", alpha=0.6, label="Alert threshold 85%")
ax.fill_between(idx, y_test.values, y_pred_xgb, alpha=0.08, color=COLORS["xgb"])

ax.set_xlabel("Test Shift Index (chronological)", fontsize=11)
ax.set_ylabel("Recovery (%)", fontsize=11)
ax.set_title("RecovAI — Recovery Prediction Over Test Period\nActual vs XGBoost vs Random Forest",
             fontsize=13, fontweight="bold", color="#1F4E79")
ax.legend(fontsize=10, loc="lower right")
ax.grid(color=COLORS["grid"], linewidth=0.8)
ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
ts_path = os.path.join(OUTPUT_DIR, "timeseries_prediction.png")
plt.savefig(ts_path, dpi=150, bbox_inches="tight", facecolor="#F8F9FA")
plt.close()
print(f"      Saved: {ts_path}")

# ─────────────────────────────────────────────────────────────────────
# 8. TRAINING REPORT
# ─────────────────────────────────────────────────────────────────────
report = f"""
╔══════════════════════════════════════════════════════════════╗
║          RecovAI — Training Report                          ║
║          Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                    ║
╚══════════════════════════════════════════════════════════════╝

DATASET
  File          : {DATA_PATH}
  Total rows    : {len(df):,}
  Features used : {len(FEATURE_COLS)}
  Target        : {TARGET_COL}
  Train rows    : {len(X_train):,}
  Test rows     : {len(X_test):,}
  Train period  : {df.loc[train_mask,'Date_parsed'].min().date()} → {df.loc[train_mask,'Date_parsed'].max().date()}
  Test period   : {df.loc[test_mask,'Date_parsed'].min().date()}  → {df.loc[test_mask,'Date_parsed'].max().date()}

LEAKAGE COLUMNS EXCLUDED
  {chr(10).join('  - ' + c for c in LEAKAGE_COLS)}

MODEL PERFORMANCE
  ┌──────────────────┬──────────┬──────────┬──────────┐
  │ Model            │   R²     │   RMSE   │   MAE    │
  ├──────────────────┼──────────┼──────────┼──────────┤
  │ XGBoost          │  {r2_xgb:.4f}  │  {rmse_xgb:.4f}  │  {mae_xgb:.4f}  │
  │ Random Forest    │  {r2_rf:.4f}  │  {rmse_rf:.4f}  │  {mae_rf:.4f}  │
  ├──────────────────┼──────────┼──────────┼──────────┤
  │ TARGET           │  > 0.85  │  < 0.50  │  < 0.50  │
  └──────────────────┴──────────┴──────────┴──────────┘

TARGETS MET
  XGBoost  R² > 0.85 : {'✓ YES' if r2_xgb >= 0.85 else '✗ NO'}
  XGBoost  MAE < 0.5 : {'✓ YES' if mae_xgb <= 0.5 else '✗ NO'}
  RF       R² > 0.85 : {'✓ YES' if r2_rf >= 0.85 else '✗ NO'}
  RF       MAE < 0.5 : {'✓ YES' if mae_rf <= 0.5 else '✗ NO'}

TOP 5 FEATURES (XGBoost)
{pd.Series(xgb_model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False).head(5).to_string()}

TOP 5 FEATURES (Random Forest)
{pd.Series(rf_model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False).head(5).to_string()}

OUTPUT FILES
  models/xgboost_model.pkl
  models/random_forest_model.pkl
  plots/feature_importance.png
  plots/predictions_vs_actual.png
  plots/model_comparison.png
  plots/timeseries_prediction.png

NEXT STEPS
  1. Load model: pickle.load(open('outputs/xgboost_model.pkl','rb'))
  2. Predict:    model['model'].predict(new_shift_data[model['features']])
  3. Build Reagent Dose Intelligence Engine
  4. Add Isolation Forest anomaly detection
  5. Add SHAP explainability
  6. Deploy Streamlit dashboard
"""

report_path = os.path.join(OUTPUT_DIR, "training_report.txt")
with open(report_path, "w") as f:
    f.write(report)

print(f"      Saved: {report_path}")

# ─────────────────────────────────────────────────────────────────────
# 9. FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("   TRAINING COMPLETE")
print("=" * 65)
print(f"\n   {'Model':<20} {'R²':>8} {'RMSE':>8} {'MAE':>8}")
print(f"   {'-'*44}")
print(f"   {'XGBoost':<20} {r2_xgb:>8.4f} {rmse_xgb:>8.4f} {mae_xgb:>8.4f}")
print(f"   {'Random Forest':<20} {r2_rf:>8.4f} {rmse_rf:>8.4f} {mae_rf:>8.4f}")
print(f"   {'TARGET':<20} {'>0.85':>8} {'<0.50':>8} {'<0.50':>8}")
print()

winner = "XGBoost" if r2_xgb >= r2_rf else "Random Forest"
best_r2 = max(r2_xgb, r2_rf)
print(f"   Best model : {winner}  (R² = {best_r2:.4f})")
print(f"   Proposal targets met : R² {'✓' if best_r2 >= 0.85 else '✗'}  |  MAE {'✓' if min(mae_xgb, mae_rf) <= 0.5 else '✗'}")
print(f"\n   All outputs saved to: ./{OUTPUT_DIR}/")
print("=" * 65)
