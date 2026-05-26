import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


DATASET_PATH = "ML_Dataset_Copper_TARGET85.csv"
OUTPUT_DIR = "recovai_output"
TARGET = "Recovery (%)"
RANDOM_STATE = 42


# These columns can stay in the CSV, but they must not be used as model inputs.
# They are output/result columns or directly related to recovery calculation.
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


# Only use values known before/during the shift.
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
    # header=1 is required because row 1 has group labels and row 2 has real names.
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


def train_model(X_train, y_train):
    model = XGBRegressor(
        n_estimators=260,
        max_depth=2,
        learning_rate=0.035,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=4,
        reg_alpha=0.25,
        reg_lambda=2.0,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_train, X_test, y_train, y_test):
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    results = {
        "train_r2": r2_score(y_train, train_pred),
        "test_r2": r2_score(y_test, test_pred),
        "test_mae": mean_absolute_error(y_test, test_pred),
        "test_rmse": np.sqrt(mean_squared_error(y_test, test_pred)),
        "test_pred": test_pred,
    }
    return results


def save_plots(model, y_test, test_pred):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_test, y=test_pred, s=35)
    min_val = min(y_test.min(), test_pred.min())
    max_val = max(y_test.max(), test_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect prediction")
    plt.xlabel("Actual Recovery (%)")
    plt.ylabel("Predicted Recovery (%)")
    plt.title("XGBoost: Actual vs Predicted Recovery")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "xgb_clean_actual_vs_predicted.png"), dpi=150)
    plt.close()

    importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    plt.figure(figsize=(10, 7))
    importance.head(15).sort_values().plot(kind="barh")
    plt.title("Top 15 XGBoost Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "xgb_clean_feature_importance.png"), dpi=150)
    plt.close()

    return importance


def main():
    print("=" * 60)
    print("XGBoost Copper Recovery Training - Leakage-Free")
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

    model = train_model(X_train, y_train)
    results = evaluate(model, X_train, X_test, y_train, y_test)

    print("\nXGBoost Recovery Prediction Results")
    print("-" * 40)
    print(f"Train R2 : {results['train_r2']:.4f}")
    print(f"Test R2  : {results['test_r2']:.4f}")
    print(f"Test MAE : {results['test_mae']:.4f}%")
    print(f"Test RMSE: {results['test_rmse']:.4f}%")

    gap = results["train_r2"] - results["test_r2"]
    if gap > 0.25:
        print("\nNote: Train R2 is much higher than Test R2.")
        print("This usually means overfitting or weak future-prediction signal in the dataset.")

    importance = save_plots(model, y_test, results["test_pred"])

    print("\nTop Feature Importances:")
    print(importance.head(15))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_model(os.path.join(OUTPUT_DIR, "model_recovery_xgb_clean.json"))
    joblib.dump(FEATURES, os.path.join(OUTPUT_DIR, "features_xgb_clean.pkl"))

    print("\nSaved files:")
    print(os.path.join(OUTPUT_DIR, "model_recovery_xgb_clean.json"))
    print(os.path.join(OUTPUT_DIR, "features_xgb_clean.pkl"))
    print(os.path.join(OUTPUT_DIR, "xgb_clean_actual_vs_predicted.png"))
    print(os.path.join(OUTPUT_DIR, "xgb_clean_feature_importance.png"))


if __name__ == "__main__":
    main()
