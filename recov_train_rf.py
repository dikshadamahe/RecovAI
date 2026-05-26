import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance


DATASET_PATH = "ML_Dataset_Copper_TARGET85.csv"
OUTPUT_DIR = "recovai_output"
TARGET = "Recovery (%)"
RANDOM_STATE = 42
N_JOBS = 1


# Keep these columns in the CSV for EDA/reporting, but never use them as inputs.
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
    # header=1 because row 1 has group labels and row 2 has real column names.
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
    model = RandomForestRegressor(
        n_estimators=700,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features=0.5,
        bootstrap=True,
        oob_score=True,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_train, X_test, y_train, y_test):
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    return {
        "train_r2": r2_score(y_train, train_pred),
        "test_r2": r2_score(y_test, test_pred),
        "test_mae": mean_absolute_error(y_test, test_pred),
        "test_rmse": np.sqrt(mean_squared_error(y_test, test_pred)),
        "oob_r2": model.oob_score_,
        "test_pred": test_pred,
    }


def save_plots(model, X_test, y_test, test_pred):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_test, y=test_pred, s=35)
    min_val = min(y_test.min(), test_pred.min())
    max_val = max(y_test.max(), test_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect prediction")
    plt.xlabel("Actual Recovery (%)")
    plt.ylabel("Predicted Recovery (%)")
    plt.title("Random Forest: Actual vs Predicted Recovery")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "rf_clean_actual_vs_predicted.png"), dpi=150)
    plt.close()

    importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    plt.figure(figsize=(10, 7))
    importance.head(15).sort_values().plot(kind="barh")
    plt.title("Top 15 Random Forest Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "rf_clean_feature_importance.png"), dpi=150)
    plt.close()

    perm = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )
    perm_importance = pd.Series(perm.importances_mean, index=FEATURES).sort_values(ascending=False)
    plt.figure(figsize=(10, 7))
    perm_importance.head(15).sort_values().plot(kind="barh")
    plt.title("Top 15 Random Forest Permutation Importances")
    plt.xlabel("Mean decrease in R2 after shuffle")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "rf_clean_permutation_importance.png"), dpi=150)
    plt.close()

    return importance, perm_importance


def main():
    print("=" * 60)
    print("Random Forest Copper Recovery Training - Leakage-Free")
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

    print("\nRandom Forest Recovery Prediction Results")
    print("-" * 40)
    print(f"Train R2 : {results['train_r2']:.4f}")
    print(f"Test R2  : {results['test_r2']:.4f}")
    print(f"OOB R2   : {results['oob_r2']:.4f}")
    print(f"Test MAE : {results['test_mae']:.4f}%")
    print(f"Test RMSE: {results['test_rmse']:.4f}%")

    gap = results["train_r2"] - results["test_r2"]
    if gap > 0.25:
        print("\nNote: Train R2 is much higher than Test R2.")
        print("This usually means overfitting or weak future-prediction signal in the dataset.")

    importance, perm_importance = save_plots(model, X_test, y_test, results["test_pred"])

    print("\nTop Feature Importances:")
    print(importance.head(15))

    print("\nTop Permutation Importances:")
    print(perm_importance.head(15))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(OUTPUT_DIR, "model_recovery_rf_clean.pkl"))
    joblib.dump(FEATURES, os.path.join(OUTPUT_DIR, "features_rf_clean.pkl"))

    print("\nSaved files:")
    print(os.path.join(OUTPUT_DIR, "model_recovery_rf_clean.pkl"))
    print(os.path.join(OUTPUT_DIR, "features_rf_clean.pkl"))
    print(os.path.join(OUTPUT_DIR, "rf_clean_actual_vs_predicted.png"))
    print(os.path.join(OUTPUT_DIR, "rf_clean_feature_importance.png"))
    print(os.path.join(OUTPUT_DIR, "rf_clean_permutation_importance.png"))


if __name__ == "__main__":
    main()
