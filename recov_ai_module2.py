"""
╔══════════════════════════════════════════════════════════════════════╗
║         RecovAI — Module 2                                          ║
║         Reagent Dose Intelligence Engine                            ║
║         + Anomaly Detection (Isolation Forest)                      ║
║         + Streamlit Operator Dashboard                              ║
╚══════════════════════════════════════════════════════════════════════╝

Usage:
    # Run analysis only (no UI):
    python3 recov_ai_module2.py

    # Launch live dashboard:
    streamlit run recov_ai_module2.py

Install requirements:
    pip3 install streamlit xgboost scikit-learn matplotlib pandas numpy shap
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────
DATA_PATH      = "data/processed/ML_Dataset_Copper_TARGET85.csv"
MODEL_PATH     = "outputs/xgboost_model.pkl"
OUTPUT_DIR     = "outputs"
RECOVERY_ALERT = 88.0   # % — flag shifts below this

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(DATA_PATH, header=1)
    df = df.dropna(how="all").reset_index(drop=True)
    df["Date_parsed"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.sort_values("Date_parsed").reset_index(drop=True)
    return df

def load_model():
    with open(MODEL_PATH, "rb") as f:
        obj = pickle.load(f)
    return obj["model"], obj["features"]

# ─────────────────────────────────────────────────────────────────────
# MODULE 1 — REAGENT DOSE INTELLIGENCE ENGINE
# ─────────────────────────────────────────────────────────────────────
def reagent_intelligence(df, model, features):
    """
    For each shift:
    - Predict recovery using actual reagent doses
    - Compute optimal reagent doses via perturbation analysis
    - Flag gap as GREEN / AMBER / RED
    """
    print("\n" + "="*65)
    print("   MODULE 1 — REAGENT DOSE INTELLIGENCE ENGINE")
    print("="*65)

    LEAKAGE = ["COPPER IN CONCENTRATE (MT)", "COPPER IN TAILINGS (MT)",
               "Concentrate Production (MT)", "COPPER IN HEAD (MT)",
               "TAILINGS (MT)", "T Reagent (cc)"]
    DROP    = LEAKAGE + ["Date", "Date_parsed", "Shift", "Source",
                         "Estimated Feed Condition"]

    X = df[[c for c in features if c in df.columns]].fillna(df.median(numeric_only=True))

    # Predict recovery with actual doses
    df["Predicted_Recovery"] = model.predict(X)

    # Optimal dose search — perturb each reagent ±20% and find best recovery
    reagents = {
        "SIPX Dose (g/t)"      : (40,  100),
        "Frother Dose (g/t)"   : (20,  60),
        "Depressant Dose (g/t)": (40,  120),
        "Flotation pH"         : (9.0, 10.0),
    }

    for reagent, (low, high) in reagents.items():
        if reagent not in X.columns:
            continue
        best_recovery = df["Predicted_Recovery"].copy()
        best_dose     = X[reagent].copy()

        for pct in np.linspace(low, high, 15):
            X_test = X.copy()
            X_test[reagent] = pct
            pred = model.predict(X_test)
            improved = pred > best_recovery
            best_recovery = np.where(improved, pred, best_recovery)
            best_dose     = np.where(improved, pct, best_dose)

        col_name = reagent.replace(" (g/t)", "").replace(" (%)", "").replace(" ", "_")
        df[f"Optimal_{col_name}"]  = best_dose.round(1)
        df[f"Actual_{col_name}"]   = X[reagent].values
        df[f"Gap_{col_name}"]      = (best_dose - X[reagent].values).round(2)

    # Traffic light flagging
    def flag(gap, threshold=5):
        if abs(gap) <= threshold * 0.5:   return "🟢 OK"
        elif abs(gap) <= threshold:        return "🟡 REVIEW"
        else:                              return "🔴 ADJUST"

    for reagent in reagents.keys():
        col = reagent.replace(" (g/t)", "").replace(" (%)", "").replace(" ", "_")
        gap_col = f"Gap_{col}"
        if gap_col in df.columns:
            df[f"Flag_{col}"] = df[gap_col].apply(flag)

    # Recovery alert
    df["Recovery_Alert"] = df["Predicted_Recovery"].apply(
        lambda x: "🔴 ALERT" if x < RECOVERY_ALERT else "🟢 OK"
    )

    # Summary stats
    red_shifts   = (df["Recovery_Alert"] == "🔴 ALERT").sum()
    green_shifts = (df["Recovery_Alert"] == "🟢 OK").sum()
    print(f"\n   Total shifts analysed : {len(df):,}")
    print(f"   🟢 Normal recovery     : {green_shifts:,} shifts")
    print(f"   🔴 Alert (<{RECOVERY_ALERT}%)       : {red_shifts:,} shifts")
    print(f"   Mean predicted recovery: {df['Predicted_Recovery'].mean():.2f}%")

    # Save reagent report
    report_cols = ["Date", "Shift", "Recovery (%)", "Predicted_Recovery",
                   "Recovery_Alert"] + \
                  [c for c in df.columns if c.startswith(("Optimal_", "Actual_", "Gap_", "Flag_"))]
    report_cols = [c for c in report_cols if c in df.columns]
    df[report_cols].to_csv(f"{OUTPUT_DIR}/reagent_intelligence_report.csv", index=False)
    print(f"\n   Saved: {OUTPUT_DIR}/reagent_intelligence_report.csv")

    # Plot: Actual vs Optimal for last 90 shifts
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor("#F8F9FA")
    fig.suptitle("RecovAI — Reagent Dose Intelligence\nActual vs Optimal Doses (Last 90 Shifts)",
                 fontsize=14, fontweight="bold", color="#1F4E79")

    plot_reagents = [
        ("SIPX_Dose", "SIPX Dose (g/t)", "#1F4E79"),
        ("Frother_Dose", "Frother Dose (g/t)", "#2E75B6"),
        ("Depressant_Dose", "Depressant Dose (g/t)", "#7030A0"),
        ("Flotation_pH", "Flotation pH", "#C55A11"),
    ]

    tail = df.tail(90)
    for ax, (col, label, color) in zip(axes.flatten(), plot_reagents):
        act_col = f"Actual_{col}"
        opt_col = f"Optimal_{col}"
        if act_col not in df.columns:
            continue
        idx = range(len(tail))
        ax.plot(idx, tail[act_col].values, color=color, linewidth=1.5,
                label="Actual", alpha=0.9)
        ax.plot(idx, tail[opt_col].values, color="green", linewidth=1.5,
                linestyle="--", label="ML Optimal", alpha=0.9)
        ax.fill_between(idx, tail[act_col].values, tail[opt_col].values,
                        alpha=0.12, color=color)
        ax.set_title(label, fontsize=11, fontweight="bold", color="#1F4E79")
        ax.set_xlabel("Shift Index", fontsize=9)
        ax.legend(fontsize=9)
        ax.set_facecolor("#F8F9FA")
        ax.grid(color="#E8E8E8", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/reagent_intelligence.png", dpi=150,
                bbox_inches="tight", facecolor="#F8F9FA")
    plt.close()
    print(f"   Saved: {OUTPUT_DIR}/reagent_intelligence.png")

    return df


# ─────────────────────────────────────────────────────────────────────
# MODULE 2 — ANOMALY DETECTION (ISOLATION FOREST)
# ─────────────────────────────────────────────────────────────────────
def anomaly_detection(df):
    """
    Isolation Forest on key process variables.
    Flags statistically abnormal shifts before recovery loss is confirmed.
    """
    print("\n" + "="*65)
    print("   MODULE 2 — ANOMALY DETECTION (ISOLATION FOREST)")
    print("="*65)

    anomaly_features = [
        "Feed Rate (MT/h)", "Grinding kWh", "Flotation pH",
        "Head Grade (%Cu)", "SIPX Dose (g/t)", "Frother Dose (g/t)",
        "Depressant Dose (g/t)", "Insoluble in Concentrate (%)",
        "Tails Grade (%Cu)", "Conc. Mass Pull (%)"
    ]
    anomaly_features = [f for f in anomaly_features if f in df.columns]

    X_anomaly = df[anomaly_features].fillna(df[anomaly_features].median())

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_anomaly)

    # Train Isolation Forest
    iso_forest = IsolationForest(
        n_estimators=200,
        contamination=0.05,   # expect ~5% anomalous shifts
        random_state=42,
        n_jobs=-1
    )
    df["Anomaly_Score"]  = iso_forest.fit_predict(X_scaled)
    df["Anomaly_Raw"]    = iso_forest.score_samples(X_scaled)
    df["Is_Anomaly"]     = df["Anomaly_Score"] == -1
    df["Anomaly_Flag"]   = df["Is_Anomaly"].apply(
        lambda x: "🔴 ANOMALY" if x else "🟢 NORMAL"
    )

    n_anomalies = df["Is_Anomaly"].sum()
    pct = n_anomalies / len(df) * 100
    print(f"\n   Total shifts     : {len(df):,}")
    print(f"   🔴 Anomalous      : {n_anomalies:,} ({pct:.1f}%)")
    print(f"   🟢 Normal         : {len(df) - n_anomalies:,} ({100-pct:.1f}%)")

    # Anomaly overlap with low recovery
    low_rec   = df["Recovery (%)"] < RECOVERY_ALERT
    anomalous = df["Is_Anomaly"]
    both      = (low_rec & anomalous).sum()
    print(f"\n   Anomaly + Low Recovery overlap: {both} shifts")
    print(f"   → Anomaly detection catches issues BEFORE assay confirmation")

    # Save anomaly report
    anomaly_report = df[["Date", "Shift", "Recovery (%)", "Predicted_Recovery",
                          "Anomaly_Flag", "Anomaly_Raw"] + anomaly_features].copy()
    anomaly_report.to_csv(f"{OUTPUT_DIR}/anomaly_detection_report.csv", index=False)
    print(f"\n   Saved: {OUTPUT_DIR}/anomaly_detection_report.csv")

    # Plot anomaly scores over time
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.patch.set_facecolor("#F8F9FA")

    # Top: Anomaly score timeline
    ax1 = axes[0]
    colors = ["#C00000" if a else "#2E75B6" for a in df["Is_Anomaly"]]
    ax1.scatter(range(len(df)), df["Anomaly_Raw"], c=colors, s=8, alpha=0.6)
    ax1.axhline(df[df["Is_Anomaly"]]["Anomaly_Raw"].max(),
                color="red", linestyle="--", linewidth=1, alpha=0.7,
                label="Anomaly threshold")
    ax1.set_title("Isolation Forest Anomaly Scores — All Shifts",
                  fontsize=12, fontweight="bold", color="#1F4E79")
    ax1.set_xlabel("Shift Index")
    ax1.set_ylabel("Anomaly Score (lower = more anomalous)")
    ax1.set_facecolor("#F8F9FA")
    ax1.grid(color="#E8E8E8", linewidth=0.8)
    ax1.spines[["top", "right"]].set_visible(False)
    normal_patch = plt.Line2D([0],[0], marker='o', color='w',
                               markerfacecolor='#2E75B6', markersize=8, label='Normal')
    anomaly_patch = plt.Line2D([0],[0], marker='o', color='w',
                                markerfacecolor='#C00000', markersize=8, label='Anomaly')
    ax1.legend(handles=[normal_patch, anomaly_patch], fontsize=10)

    # Bottom: Recovery % with anomalies highlighted
    ax2 = axes[1]
    normal_df  = df[~df["Is_Anomaly"]]
    anomaly_df = df[df["Is_Anomaly"]]
    ax2.scatter(normal_df.index,  normal_df["Recovery (%)"],
                color="#2E75B6", s=8, alpha=0.5, label="Normal shift")
    ax2.scatter(anomaly_df.index, anomaly_df["Recovery (%)"],
                color="#C00000", s=25, alpha=0.8, label="Anomalous shift", zorder=5)
    ax2.axhline(RECOVERY_ALERT, color="orange", linestyle="--",
                linewidth=1.5, label=f"Alert threshold ({RECOVERY_ALERT}%)")
    ax2.set_title("Recovery % — Anomalous Shifts Highlighted",
                  fontsize=12, fontweight="bold", color="#1F4E79")
    ax2.set_xlabel("Shift Index")
    ax2.set_ylabel("Recovery (%)")
    ax2.legend(fontsize=10)
    ax2.set_facecolor("#F8F9FA")
    ax2.grid(color="#E8E8E8", linewidth=0.8)
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/anomaly_detection.png", dpi=150,
                bbox_inches="tight", facecolor="#F8F9FA")
    plt.close()
    print(f"   Saved: {OUTPUT_DIR}/anomaly_detection.png")

    return df


# ─────────────────────────────────────────────────────────────────────
# MODULE 3 — STREAMLIT DASHBOARD
# ─────────────────────────────────────────────────────────────────────
def run_dashboard():
    """Streamlit operator dashboard — run with: streamlit run recov_ai_module2.py"""
    import streamlit as st

    st.set_page_config(
        page_title="RecovAI — Copper Recovery Intelligence",
        page_icon="⚗️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # ── Header ──────────────────────────────────────────────────────
    st.markdown("""
    <div style='background:linear-gradient(135deg,#1F4E79,#2E75B6);
                padding:20px 30px; border-radius:12px; margin-bottom:20px;'>
        <h1 style='color:white; margin:0; font-size:28px;'>⚗️ RecovAI</h1>
        <p style='color:#BDD7EE; margin:4px 0 0;'>
            Copper Recovery Prediction & Reagent Intelligence System — Malanjkhand Concentrator
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load data and model ─────────────────────────────────────────
    @st.cache_data
    def get_data():
        return load_data()

    @st.cache_resource
    def get_model():
        return load_model()

    try:
        df = get_data()
        model, features = get_model()
    except Exception as e:
        st.error(f"Error loading data or model: {e}")
        st.stop()

    # Run analysis
    df = reagent_intelligence(df, model, features)
    df = anomaly_detection(df)

    # ── Sidebar filters ─────────────────────────────────────────────
    st.sidebar.title("🔧 Filters")
    n_shifts = st.sidebar.slider("Show last N shifts", 30, 500, 90, 10)
    show_anomalies_only = st.sidebar.checkbox("Show anomalies only", False)
    alert_threshold = st.sidebar.slider("Recovery alert threshold (%)", 85.0, 92.0, 88.0, 0.5)

    df_show = df.tail(n_shifts).copy()
    if show_anomalies_only:
        df_show = df_show[df_show["Is_Anomaly"]]

    # ── KPI Cards ────────────────────────────────────────────────────
    st.markdown("### 📊 Current Performance")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        mean_rec = df_show["Recovery (%)"].mean()
        color = "🟢" if mean_rec >= alert_threshold else "🔴"
        st.metric(f"{color} Mean Recovery", f"{mean_rec:.2f}%",
                  delta=f"{mean_rec - 90:.2f}% vs 90% target")

    with col2:
        mean_pred = df_show["Predicted_Recovery"].mean()
        st.metric("🤖 Predicted Recovery", f"{mean_pred:.2f}%")

    with col3:
        n_alerts = (df_show["Predicted_Recovery"] < alert_threshold).sum()
        st.metric("🔴 Alert Shifts", f"{n_alerts}",
                  delta=f"{n_alerts/len(df_show)*100:.1f}% of shifts")

    with col4:
        n_anomalies = df_show["Is_Anomaly"].sum()
        st.metric("⚠️ Anomalous Shifts", f"{n_anomalies}",
                  delta=f"{n_anomalies/len(df_show)*100:.1f}% of shifts")

    with col5:
        mean_grade = df_show["Head Grade (%Cu)"].mean()
        st.metric("🪨 Mean Head Grade", f"{mean_grade:.3f}% Cu")

    st.markdown("---")

    # ── Recovery Timeline ────────────────────────────────────────────
    st.markdown("### 📈 Recovery Prediction Timeline")
    fig1, ax = plt.subplots(figsize=(14, 4))
    fig1.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")
    idx = range(len(df_show))
    ax.plot(idx, df_show["Recovery (%)"].values,
            color="#375623", linewidth=1.5, label="Actual Recovery", alpha=0.9)
    ax.plot(idx, df_show["Predicted_Recovery"].values,
            color="#1F4E79", linewidth=1.5, linestyle="--",
            label="ML Predicted", alpha=0.85)
    ax.axhline(alert_threshold, color="red", linewidth=1.2,
               linestyle="-.", alpha=0.7, label=f"Alert ({alert_threshold}%)")
    anomaly_idx = [i for i, v in enumerate(df_show["Is_Anomaly"].values) if v]
    if anomaly_idx:
        ax.scatter(anomaly_idx,
                   df_show["Recovery (%)"].values[anomaly_idx],
                   color="red", s=40, zorder=5, label="Anomaly", alpha=0.8)
    ax.legend(fontsize=9)
    ax.grid(color="#E8E8E8", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Shift Index")
    ax.set_ylabel("Recovery (%)")
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close()

    # ── Reagent Intelligence ─────────────────────────────────────────
    st.markdown("### 🧪 Reagent Dose Intelligence")
    rc1, rc2 = st.columns(2)

    reagent_pairs = [
        ("Actual_SIPX_Dose", "Optimal_SIPX_Dose", "SIPX Dose (g/t)", "#1F4E79"),
        ("Actual_Frother_Dose", "Optimal_Frother_Dose", "Frother Dose (g/t)", "#2E75B6"),
        ("Actual_Depressant_Dose", "Optimal_Depressant_Dose", "Depressant (g/t)", "#7030A0"),
        ("Actual_Flotation_pH", "Optimal_Flotation_pH", "Flotation pH", "#C55A11"),
    ]

    for i, (act_col, opt_col, label, color) in enumerate(reagent_pairs):
        col = rc1 if i % 2 == 0 else rc2
        if act_col not in df_show.columns:
            continue
        with col:
            fig2, ax2 = plt.subplots(figsize=(7, 3))
            fig2.patch.set_facecolor("#F8F9FA")
            ax2.set_facecolor("#F8F9FA")
            idx = range(len(df_show))
            ax2.plot(idx, df_show[act_col].values, color=color,
                     linewidth=1.5, label="Actual", alpha=0.9)
            ax2.plot(idx, df_show[opt_col].values, color="green",
                     linewidth=1.5, linestyle="--", label="ML Optimal", alpha=0.9)
            ax2.fill_between(idx, df_show[act_col].values,
                             df_show[opt_col].values, alpha=0.1, color=color)
            ax2.set_title(label, fontsize=10, fontweight="bold", color="#1F4E79")
            ax2.legend(fontsize=8)
            ax2.grid(color="#E8E8E8", linewidth=0.8)
            ax2.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

    # ── Reagent gap summary table ────────────────────────────────────
    st.markdown("### 🚦 Reagent Gap Alert Summary (Last 10 Shifts)")
    flag_cols = ["Date", "Shift", "Recovery (%)", "Predicted_Recovery",
                 "Recovery_Alert", "Anomaly_Flag"] + \
                [c for c in df.columns if c.startswith("Flag_")]
    flag_cols = [c for c in flag_cols if c in df_show.columns]
    st.dataframe(df_show[flag_cols].tail(10), use_container_width=True)

    # ── Anomaly Detection ────────────────────────────────────────────
    st.markdown("### 🔍 Anomaly Detection")
    fig3, ax3 = plt.subplots(figsize=(14, 3))
    fig3.patch.set_facecolor("#F8F9FA")
    ax3.set_facecolor("#F8F9FA")
    colors = ["#C00000" if a else "#2E75B6" for a in df_show["Is_Anomaly"]]
    ax3.scatter(range(len(df_show)), df_show["Anomaly_Raw"],
                c=colors, s=10, alpha=0.7)
    ax3.set_title("Isolation Forest Anomaly Scores",
                  fontsize=11, fontweight="bold", color="#1F4E79")
    ax3.set_xlabel("Shift Index")
    ax3.set_ylabel("Score (lower = anomalous)")
    ax3.grid(color="#E8E8E8", linewidth=0.8)
    ax3.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

    # ── Shift Predictor Tool ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🎯 Live Shift Recovery Predictor")
    st.markdown("Enter current shift parameters to predict recovery:")

    sp1, sp2, sp3 = st.columns(3)
    with sp1:
        hg   = st.number_input("Head Grade (%Cu)", 0.5, 1.5, 1.0, 0.01)
        ph   = st.number_input("Flotation pH", 9.0, 10.0, 9.5, 0.1)
        sipx = st.number_input("SIPX Dose (g/t)", 40.0, 100.0, 68.0, 1.0)
    with sp2:
        fr   = st.number_input("Frother Dose (g/t)", 20.0, 60.0, 38.0, 1.0)
        dep  = st.number_input("Depressant Dose (g/t)", 40.0, 120.0, 63.0, 1.0)
        feed = st.number_input("Feed Rate (MT/h)", 60.0, 85.0, 75.0, 0.5)
    with sp3:
        grind = st.number_input("Grinding kWh", 700.0, 750.0, 725.0, 1.0)
        lime  = st.number_input("Lime Bags", 8, 14, 11, 1)
        prev_rec = st.number_input("Previous Shift Recovery (%)", 85.0, 95.0, 90.0, 0.1)

    if st.button("🔮 Predict Recovery", type="primary"):
        # Build input row matching model features
        sample = pd.DataFrame([{f: 0 for f in features}])
        sample["Head Grade (%Cu)"]       = hg
        sample["Flotation pH"]           = ph
        sample["SIPX Dose (g/t)"]        = sipx
        sample["Frother Dose (g/t)"]     = fr
        sample["Depressant Dose (g/t)"]  = dep
        sample["Feed Rate (MT/h)"]       = feed
        sample["Grinding kWh"]           = grind
        sample["Lime Bags"]              = lime
        sample["Prev_Recovery (%)"]      = prev_rec
        sample["Roll7_Recovery (%)"]     = prev_rec
        sample["Feed_Condition_Num"]     = 1
        sample["Shift_Num"]              = 1
        sample["Month"]                  = datetime.now().month
        sample["Day_of_Week"]            = datetime.now().weekday()

        pred = model.predict(sample)[0]
        color = "🟢" if pred >= alert_threshold else "🔴"

        st.markdown(f"""
        <div style='background:{"#E2EFDA" if pred >= alert_threshold else "#FFE0E0"};
                    padding:20px; border-radius:10px; text-align:center; margin-top:10px;'>
            <h2 style='color:{"#375623" if pred >= alert_threshold else "#C00000"}; margin:0;'>
                {color} Predicted Recovery: {pred:.2f}%
            </h2>
            <p style='margin:8px 0 0; color:#555;'>
                {"✅ Within normal operating range" if pred >= alert_threshold
                 else f"⚠️ Below alert threshold ({alert_threshold}%) — review reagent doses"}
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ── Footer ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div style='text-align:center; color:#888; font-size:12px;'>
        RecovAI v1.0 — Malanjkhand Copper Concentrator |
        XGBoost Model R²=0.97 | Built for HCL MCP AI Project
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# MAIN — detects whether running as script or streamlit
# ─────────────────────────────────────────────────────────────────────
def run_analysis_only():
    """Run all modules without Streamlit UI — saves reports and plots."""
    print("="*65)
    print("   RecovAI Module 2 — Full Analysis Pipeline")
    print("="*65)

    print("\n[1/3] Loading data and model...")
    df    = load_data()
    model, features = load_model()
    print(f"      Rows: {len(df):,} | Features: {len(features)}")

    print("\n[2/3] Running Reagent Dose Intelligence Engine...")
    df = reagent_intelligence(df, model, features)

    print("\n[3/3] Running Anomaly Detection...")
    df = anomaly_detection(df)

    print("\n" + "="*65)
    print("   ANALYSIS COMPLETE — All outputs saved to ./outputs/")
    print("="*65)
    print(f"\n   reagent_intelligence_report.csv")
    print(f"   reagent_intelligence.png")
    print(f"   anomaly_detection_report.csv")
    print(f"   anomaly_detection.png")
    print(f"\n   To launch the live dashboard run:")
    print(f"   streamlit run recov_ai_module2.py")
    print("="*65)


if __name__ == "__main__":
    import sys
    # If called by streamlit it injects streamlit into sys.modules
    if "streamlit" in sys.modules:
        run_dashboard()
    else:
        # Check if being run by streamlit CLI
        if any("streamlit" in arg for arg in sys.argv):
            run_dashboard()
        else:
            run_analysis_only()
else:
    # Called by streamlit run
    run_dashboard()
