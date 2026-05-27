# ⚙️ HCL Copper Recovery AI — RecovAI

Intelligent decision-support platform for copper flotation plant operations.
Predicts shift recovery %, optimises reagent doses, flags anomalies, and generates AI-enhanced shift reports — all from a Streamlit dashboard powered by Groq (Llama 3).

---

## 📁 Project Structure

```
HCL-MCP-AI-project--main/
│
├── app.py                          ← Main Streamlit dashboard (run this)
├── recovai_train.py                ← XGBoost training script
├── recov_train_rf.py               ← Random Forest training script
├── recov_train_linear.py           ← Linear model training script
├── eda_analyisis.ipynb             ← Exploratory data analysis notebook
│
├── ML_Dataset_Copper_TARGET85.csv  ← Main dataset (required)
│
├── recovai_output/                 ← Pre-trained models + plots (already included)
│   ├── model_recovery_xgb_clean.json
│   ├── model_recovery_rf_clean.pkl
│   ├── model_recovery_linear_clean.pkl
│   └── *.png                       ← Feature importance / SHAP plots
│
├── data/processed/                 ← Processed data files
├── requirements.txt                ← Python dependencies
└── .env                            ← Your API keys (you create this — see below)
```

---

## 🖥️ Setup on a New PC — Step by Step

### Step 1 — Install Python

Make sure Python 3.10 or higher is installed.

```bash
python --version
```

If not installed, download from [python.org](https://python.org) and install it.
During installation on Windows, **tick "Add Python to PATH"**.

---

### Step 2 — Extract the ZIP

Extract the downloaded ZIP file anywhere you like, for example your Desktop.

```bash
# On Mac/Linux you can also run:
unzip HCL-MCP-AI-project--main.zip
```

Then open a terminal and navigate into the folder:

```bash
cd HCL-MCP-AI-project--main
```

> **Windows users:** Open the folder in File Explorer, then right-click inside it → "Open in Terminal" or "Git Bash here".

---

### Step 3 — Create a Virtual Environment

A virtual environment keeps this project's packages separate from everything else on your PC.

```bash
python -m venv .venv
```

Now activate it:

```bash
# Mac / Linux
source .venv/bin/activate

# Windows (Command Prompt)
.venv\Scripts\activate.bat

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

You should see `(.venv)` appear at the start of your terminal line. That means it's active.

---

### Step 4 — Install Dependencies

```bash
pip install -r requirements.txt
pip install streamlit groq python-dotenv seaborn scipy
```

This installs everything — XGBoost, scikit-learn, pandas, Streamlit, Groq, and more.
It may take 2–3 minutes on first install.

---

### Step 5 — Add Your Groq API Key

The app uses Groq (free) for AI-enhanced shift reports and the "Ask the Plant" chat.

1. Go to [console.groq.com](https://console.groq.com) and sign up (free)
2. Click **API Keys** → **Create API Key** → copy it
3. In the project folder, create a file called `.env`:

```bash
# Mac/Linux
touch .env

# Windows — just create a new text file named .env (no .txt extension)
```

Open `.env` in any text editor and add this line:

```
GROQ_API_KEY=your_actual_key_here
```

Save and close. The app reads this automatically on startup.

> The AI features (shift report enhancement, Ask the Plant) need this key.
> Everything else (predictor, optimizer, anomaly engine) works without it.

---

### Step 6 — Run the App

```bash
streamlit run app.py
```

Your browser will open automatically at `http://localhost:8501`.

If it doesn't open, manually go to that address in any browser.

---

## 🔄 Retrain the Models (Optional)

The `recovai_output/` folder already contains pre-trained models so you can skip this.
If you have new data and want to retrain:

```bash
# Train all three models
python recovai_train.py       # XGBoost
python recov_train_rf.py      # Random Forest
python recov_train_linear.py  # Linear
```

Trained models and plots are saved automatically into `recovai_output/`.

---

## 📊 App Modules

| Module | What it does |
|---|---|
| 🏠 Home | Overview of all modules and loaded model status |
| 🎯 Recovery Predictor | Predict next-shift recovery % with confidence interval |
| 💊 Reagent Dose Optimizer | Find minimum-cost SIPX/Frother/Depressant doses for a target |
| 🚨 Anomaly & Alert Engine | Flag unusual shifts using Isolation Forest |
| 📊 Shift Performance Dashboard | Actual vs predicted recovery trends and monthly box plots |
| 🔬 Feature Impact Explorer | Feature importance, correlation heatmap, partial dependence |
| 📝 Shift Report Generator | Auto-generate formatted `.xlsx` shift report (+ AI summary via Groq) |
| 💬 Ask the Plant | Natural language Q&A over shift history via Groq (Llama 3) |
| 📈 Recovery Trend Forecaster | 7-day rolling recovery forecast from planned shift inputs |

---

## ⚠️ Common Issues

**`streamlit: command not found`**
→ Your virtual environment is not activated. Run `source .venv/bin/activate` (Mac/Linux) or `.venv\Scripts\activate.bat` (Windows) first.

**`ModuleNotFoundError: No module named 'groq'`**
→ Run `pip install groq` inside your activated virtual environment.

**`Dataset not found` error in the app**
→ Make sure `ML_Dataset_Copper_TARGET85.csv` is in the root project folder (same level as `app.py`).

**Models showing ❌ in sidebar**
→ Check that the `recovai_output/` folder is present and contains the `.json` / `.pkl` model files.

**Groq API call failed**
→ Check your `.env` file has `GROQ_API_KEY=...` with no spaces around the `=`. Make sure the key is valid at [console.groq.com](https://console.groq.com).

**PowerShell says "execution of scripts is disabled"**
→ Run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` then try activating again.

---

## 🧰 Requirements Summary

| Requirement | Version |
|---|---|
| Python | 3.10+ |
| Streamlit | latest |
| XGBoost | ≥ 2.0.0 |
| scikit-learn | ≥ 1.3.0 |
| pandas | ≥ 2.0.0 |
| Groq Python SDK | latest |
| openpyxl | ≥ 3.1.0 |

---

## 👤 Project

HCL RecovAI — Copper Flotation Intelligence Platform
Built for plant operators and shift managers.
AI powered by Groq (Llama 3.3-70b).
