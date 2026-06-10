# 🔵 RecovAI — AI-Powered Decision Support System
### Hindustan Copper Limited · Malanjkhand Copper Project

> An intelligent, full-stack decision support system for copper flotation plant optimization — combining XGBoost predictive modeling, real-time anomaly detection, SHAP explainability, PSI drift monitoring, and AI-generated shift reports.

---

## 📸 Project Screenshots

> **Add your project screenshots here.** Replace the placeholders below with actual images.

### Dashboard Overview



### Prediction Interface


### SHAP Explainability Panel



### Anomaly Detection View



### AI Shift Report



### PSI Drift Monitor

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Architecture](#-architecture)
- [AI Engines](#-ai-engines)
- [Model Performance](#-model-performance)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [Running the Application](#-running-the-application)
- [API Endpoints](#-api-endpoints)
- [Dataset](#-dataset)
- [Team](#-team)

---

## 🧠 Project Overview

**RecovAI** is an AI-powered decision support system built for **Hindustan Copper Limited (HCL)** targeting the **Malanjkhand Copper Project** flotation plant. The system predicts copper recovery rates, recommends optimal reagent dosages, detects operational anomalies, and generates plain-English shift performance reports — all in real time.

### Problem Statement
Copper flotation recovery is highly sensitive to process variables such as reagent dosage, feed grade, pulp density, and pH. Manual monitoring is error-prone and reactive. RecovAI addresses this by providing:
- **Predictive insights** before a shift ends
- **Automated root-cause analysis** via SHAP
- **Early anomaly warnings** via Isolation Forest
- **Data drift alerts** via PSI monitoring
- **AI-generated shift reports** via Groq LLM (Llama 3)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Frontend (HTML/JS)                        │
│              hcl_recovai_v3.html + shift_data.js            │
└────────────────────────┬────────────────────────────────────┘
                         │  HTTP / REST
┌────────────────────────▼────────────────────────────────────┐
│               FastAPI Backend  (main.py v2.0)               │
│                    Port: 8000                                │
├─────────────┬──────────┬──────────┬──────────┬─────────────┤
│  Engine 1   │ Engine 2 │ Engine 3 │ Engine 4 │  Engine 5   │
│  Reagent    │ Anomaly  │  SHAP    │   PSI    │   NLP       │
│  Optimizer  │ Detector │ Explainer│ Monitor  │  Reporter   │
├─────────────┴──────────┴──────────┴──────────┴─────────────┤
│              ML Models  (/models/)                          │
│   xgb_model.pkl  |  rf_model.pkl  |  iso_forest.pkl        │
│   linear_model.pkl  |  scaler.pkl                          │
├─────────────────────────────────────────────────────────────┤
│              SQLite Database  (recovai.db)                  │
│              SQLAlchemy ORM  (database.py)                  │
└─────────────────────────────────────────────────────────────┘
```

---

## ⚙️ AI Engines

The system is powered by five independent, modular intelligence engines:

### Engine 1 — Reagent Dose Intelligence (`engine1_reagent.py`)
Uses **SciPy response-surface optimization** (`scipy.optimize.differential_evolution`) to find mathematically optimal reagent doses (SIPX, Frother, Lime, Depressant) for given feed conditions. Classifies the gap between actual and optimal plant usage.

### Engine 2 — Anomaly Detection (`engine2_anomaly.py`)
Trains an **Isolation Forest** on clean historical shift data to score every incoming shift for anomalousness — no labelled data required. Flags unusual operational conditions in real time.

### Engine 3 — SHAP Explainability (`engine3_shap.py`)
Provides per-prediction **SHAP decomposition** using `TreeExplainer`. Generates waterfall plots (single prediction) and beeswarm plots (global model behaviour). Returns structured driver data for the dashboard and NLP report.

### Engine 4 — PSI Data Drift Monitor (`engine4_psi.py`)
Computes **Population Stability Index (PSI)** for every feature to detect when live production data has drifted away from the training distribution.

| PSI Value | Status | Action |
|-----------|--------|--------|
| < 0.10 | 🟢 No Change | Model safe to use |
| 0.10 – 0.25 | 🟡 Slight Drift | Monitor closely |
| > 0.25 | 🔴 Significant Drift | Consider retraining |

### Engine 5 — NLP Shift Report Generator (`engine5_nlp.py`)
Calls the **Groq API (Llama 3)** to generate concise, plain-English shift performance reports from structured engine outputs. Falls back to a rule-based expert system if the API is unavailable.

---

## 📊 Model Performance

Trained on **1,778 shifts** of real copper plant data (Oct 2024 – May 2026), with an 80/20 temporal train-test split.

| Model | R² Score | RMSE | MAE | Target Met |
|-------|----------|------|-----|------------|
| **XGBoost** | **0.9720** | 0.1388 | 0.1042 | ✅ Yes |
| Random Forest | 0.9103 | 0.2483 | 0.1970 | ✅ Yes |
| Target Threshold | > 0.85 | < 0.50 | < 0.50 | — |

### Top 5 Features (XGBoost)
| Feature | Importance |
|---------|------------|
| Feed_Condition_Num | 0.598 |
| Prev_Recovery (%) | 0.105 |
| Conc. Mass Pull (%) | 0.057 |
| Tails Grade (%Cu) | 0.050 |
| Roll7_Recovery (%) | 0.046 |

> Leakage columns excluded from training: `COPPER IN CONCENTRATE (MT)`, `COPPER IN TAILINGS (MT)`, `Concentrate Production (MT)`, `COPPER IN HEAD (MT)`, `TAILINGS (MT)`.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend Framework** | FastAPI 0.100+ |
| **ASGI Server** | Uvicorn |
| **Primary ML Model** | XGBoost 2.0+ |
| **Ensemble Model** | Scikit-learn Random Forest |
| **Anomaly Detection** | Scikit-learn Isolation Forest |
| **Explainability** | SHAP (TreeExplainer) |
| **Optimization** | SciPy (differential evolution) |
| **LLM / NLP Reports** | Groq API (Llama 3) |
| **Data Processing** | Pandas, NumPy |
| **Database** | SQLite + SQLAlchemy ORM |
| **Serialization** | Joblib, Pickle |
| **Frontend** | Vanilla HTML/CSS/JS |
| **Config** | Python-dotenv |

---

## 📁 Project Structure

```
HCL-MCP-AI-project-/
│
├── main.py                        # FastAPI app — all endpoints (v2.0)
├── database.py                    # SQLAlchemy ORM, DB init, CRUD helpers
│
├── engines/                       # Modular AI engine package
│   ├── __init__.py
│   ├── engine1_reagent.py         # Reagent dose optimization
│   ├── engine2_anomaly.py         # Isolation Forest anomaly detection
│   ├── engine3_shap.py            # SHAP explainability
│   ├── engine4_psi.py             # PSI data drift monitoring
│   └── engine5_nlp.py             # Groq LLM shift report generation
│
├── models/                        # Trained model artifacts
│   ├── xgb_model.pkl              # Primary XGBoost model
│   ├── xgb_model.json             # XGBoost model (JSON format)
│   ├── rf_model.pkl               # Random Forest model
│   ├── linear_model.pkl           # Linear baseline model
│   ├── scaler.pkl                 # Feature scaler
│   └── iso_forest.pkl             # Isolation Forest for anomaly detection
│
├── recovai_output/                # Training artifacts & plots
│   ├── model_recovery_xgb_clean.json
│   ├── model_recovery_rf_clean.pkl
│   ├── isolation_forest.pkl
│   ├── psi_monitor.pkl
│   ├── plot_shap_recovery.png
│   ├── xgb_clean_actual_vs_predicted.png
│   └── ...
│
├── data/
│   └── processed/
│       ├── ML_Dataset_Copper_TARGET85.csv    # Primary training dataset
│       └── ML_Implementation_Dataset_Copper.csv
│
├── outputs/                       # Generated charts & reports
│   ├── feature_importance.png
│   ├── model_comparison.png
│   ├── predictions_vs_actual.png
│   ├── timeseries_prediction.png
│   └── training_report.txt
│
├── assets/                        # Frontend static assets
│   ├── logo.png
│   └── shift_data.js              # Shift data helper
│
├── hcl_recovai_v3.html            # Main frontend UI
├── shifts_dataset.csv             # Live shifts dataset
├── recovai.db                     # SQLite database
│
├── train_all.py                   # Master training script
├── train_recov_ai.py              # RecovAI-specific training
├── recov_train_rf.py              # Random Forest training
├── recov_train_linear.py          # Linear model training
├── train_extra_models.py          # Additional model experiments
│
├── requirements.txt               # Backend Python dependencies
├── requirements_frontend.txt      # Frontend dependencies
└── README.md                      # This file
```

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.10+
- pip
- A **Groq API key** (for NLP shift reports) — get one at [console.groq.com](https://console.groq.com)

### 1. Clone the Repository
```bash
git clone https://github.com/dikshadamahe/HCL-MCP-AI-project-.git
cd HCL-MCP-AI-project-
```

### 2. Create & Activate a Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

Core packages installed:
```
fastapi>=0.100.0       uvicorn[standard]>=0.20.0    pydantic>=2.0
numpy>=1.26            pandas>=2.2                   scikit-learn>=1.3.0
xgboost>=2.0.0         scipy>=1.11.0                 matplotlib>=3.7.0
shap>=0.44.0           joblib>=1.3.0                 openpyxl>=3.1.0
groq>=0.5.0            python-dotenv>=1.0.0
```

### 4. Set Environment Variables
```bash
# Windows
set GROQ_API_KEY=gsk-your-key-here

# Mac / Linux
export GROQ_API_KEY=gsk-your-key-here
```

Or create a `.env` file in the project root:
```
GROQ_API_KEY=gsk-your-key-here
```

### 5. Train the Models (if not already trained)
```bash
python train_all.py
```

Trained model files will appear in `models/` and `recovai_output/`.

---

## ▶️ Running the Application

### Start the Backend Server
```bash
uvicorn main:app --reload --port 8000
```

### Verify the Server is Running
Open your browser and navigate to:
```
http://localhost:8000/api/test
```
All fields should show `true` / loaded counts.

### Open the Frontend
Open `hcl_recovai_v3.html` directly in your browser, or serve it via any static file server.

### API Documentation (Swagger UI)
```
http://localhost:8000/docs
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/api/test` | System readiness check (models, DB, engines) |
| `POST` | `/predict` | Predict copper recovery for a shift |
| `POST` | `/optimize` | Get optimal reagent dose recommendations |
| `GET` | `/api/anomalies` | Fetch anomaly scores for recent shifts |
| `GET` | `/api/importance` | Feature importance data (SHAP + model) |
| `GET` | `/api/heatmap` | Correlation heatmap data |
| `GET` | `/api/pdp` | Partial dependence plot data |
| `GET` | `/api/dashboard` | Aggregated dashboard metrics |
| `POST` | `/report` | Generate AI shift report or chat response |
| `GET` | `/api/report/download` | Download latest shift report |
| `POST` | `/api/forecast` | Multi-shift forecast |
| `POST` | `/api/contact` | Submit contact/feedback form |
| `POST` | `/api/feedback` | Submit prediction feedback |

---

## 📂 Dataset

The system is trained on real copper flotation plant data from the **Malanjkhand Copper Project**.

| Property | Value |
|----------|-------|
| Total Shifts | 1,778 |
| Training Period | Oct 2024 – Dec 2025 |
| Test Period | Jan 2026 – May 2026 |
| Features | 26 process variables |
| Target Variable | `Recovery (%)` |

Key input features include: Ore Milled (MT), Head Grade (%Cu), Feed Rate (MT/h), Flotation pH, Grinding kWh, SIPX Dose, Frother Dose, Lime Bags, Depressant Dose, Pulp Density, Air Flow, Tails Grade, Conc. Mass Pull, and lagged/rolling statistics.

---

## 👥 Team



- Bhavya Jaiprakash Khatri
- Diksha Damahe
- Hiya Porwal
- Ritica Awasthi

**Institution:** VIT Bhopal University  
**Client:** Hindustan Copper Limited (HCL)  
**Project:** Malanjkhand Copper Project — Flotation Plant Optimization  

---

## 📄 License

This project was developed as part of an academic industry collaboration. All rights reserved by the respective institution and client organization.

---


