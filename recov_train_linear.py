import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATASET_PATH = "ML_Dataset_Copper_TARGET85.csv"
OUTPUT_DIR = "recovai_output"
TARGET = "Recovery (%)"
RANDOM_STATE = 42


LEAKAGE_OR_OUTPUT_COLUMNS = {
    "Recovery (%)",
    "Tails Grade (%Cu)",
    "Concentrate Grade (%Cu)",
    "Concentrate Production (MT)",
    "COPPER IN CONCENTRATE (MT)",
    "COPPER IN TAILINGS (MT)",
    "TAILINGS (MT)",
    "Conc. Mass Pull (%)",
}


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


def load_dataset(path):
    df = pd.read_csv(path, header=1)
    df.columns = df.columns.str.strip()

    non_numeric_cols = {"Date", "Shift", "Estimated Feed Condition", "Source"}
    for col in df.columns:
        if col not in non_numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.sort_values(["Date", "Shift_Num"], na_position="last").reset_index(drop=True)
    return df


def validate_columns(df):
    missing = [col for col in FEATURES + [TARGET] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    leaked = sorted(set(FEATURES) & LEAKAGE_OR_OUTPUT_COLUMNS)
    if leaked:
        raise ValueError(f"Remove leakage/output columns from FEATURES: {leaked}")


def evaluate(model, X_train, X_test, y_train, y_test):
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    return {
        "train_r2": r2_score(y_train, train_pred),
        "test_r2": r2_score(y_test, test_pred),
        "test_mae": mean_absolute_error(y_test, test_pred),
        "test_rmse": np.sqrt(mean_squared_error(y_test, test_pred)),
        "test_pred": test_pred,
    }


def save_plot(y_test, test_pred, model_name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_test, y=test_pred, s=35)
    min_val = min(y_test.min(), test_pred.min())
    max_val = max(y_test.max(), test_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect prediction")
    plt.xlabel("Actual Recovery (%)")
    plt.ylabel("Predicted Recovery (%)")
    plt.title(f"{model_name}: Actual vs Predicted Recovery")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "linear_actual_vs_predicted.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    return plot_path


def main():
    print("=" * 60)
    print("Linear Regression Copper Recovery Training - Leakage-Free")
    print("=" * 60)

    df = load_dataset(DATASET_PATH)
    validate_columns(df)

    print(f"Dataset shape : {df.shape}")
    print(f"Date range    : {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"Target        : {TARGET}")
    print(f"Features      : {len(FEATURES)}")

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    data = pd.concat([X, y], axis=1).dropna()
    X = data[FEATURES]
    y = data[TARGET]

    split_index = int(len(X) * 0.8)
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    print(f"Rows after cleaning : {len(X)}")
    print(f"Train rows          : {len(X_train)}")
    print(f"Test rows           : {len(X_test)}")

    models = {
        "Linear Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "Ridge Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
            ]
        ),
        "Lasso Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Lasso(alpha=0.001, random_state=RANDOM_STATE, max_iter=10000)),
            ]
        ),
    }

    results_by_model = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        results = evaluate(model, X_train, X_test, y_train, y_test)
        results_by_model[name] = (model, results)

        print(f"\n{name} Results")
        print("-" * 40)
        print(f"Train R2 : {results['train_r2']:.4f}")
        print(f"Test R2  : {results['test_r2']:.4f}")
        print(f"Test MAE : {results['test_mae']:.4f}%")
        print(f"Test RMSE: {results['test_rmse']:.4f}%")

    best_name = max(results_by_model, key=lambda key: results_by_model[key][1]["test_r2"])
    best_model, best_results = results_by_model[best_name]

    print("\nBest Linear Model")
    print("-" * 40)
    print(best_name)

    plot_path = save_plot(y_test, best_results["test_pred"], best_name)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_path = os.path.join(OUTPUT_DIR, "model_recovery_linear_clean.pkl")
    feature_path = os.path.join(OUTPUT_DIR, "features_linear_clean.pkl")

    joblib.dump(best_model, model_path)
    joblib.dump(FEATURES, feature_path)

    print("\nSaved files:")
    print(model_path)
    print(feature_path)
    print(plot_path)


if __name__ == "__main__":
    main()
