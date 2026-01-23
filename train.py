# train.py
import argparse
import json
from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor

import joblib


# -----------------------------
# Default paths (edit as needed)
# -----------------------------
DEFAULT_INPUT_CSV = "data/processed/listings_combined_clean.csv"
DEFAULT_MODEL_PATH = "models/model.joblib"
DEFAULT_METRICS_PATH = "results/metrics.json"


def _ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def train_model(
    df: pd.DataFrame,
    target_col: str,
    task: str,
    test_size: float,
    random_state: int,
):
    """
    Trains a model with a simple, standard sklearn preprocessing pipeline:
    - numeric: impute missing (median)
    - categorical: impute missing (most_frequent) + one-hot encode
    Returns (pipeline, metrics_dict).
    """
    if target_col not in df.columns:
        raise ValueError(f"target_col='{target_col}' not found in columns: {list(df.columns)[:15]}...")

    if target_col == "price":
        df[target_col] = (
            df[target_col]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    df = df.dropna(subset=[target_col])

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Identify column types
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )

    # Choose a basic model (kept simple on purpose)
    if task == "regression":
        model = RandomForestRegressor(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
        )
    elif task == "classification":
        model = LogisticRegression(
            max_iter=2000,
            n_jobs=-1,
        )
    else:
        raise ValueError("task must be 'regression' or 'classification'")

    clf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    # Train
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)

    metrics = {
        "task": task,
        "target_col": target_col,
        "n_rows": int(len(df)),
        "n_features_raw": int(X.shape[1]),
        "test_size": float(test_size),
        "random_state": int(random_state),
    }

    if task == "regression":
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        metrics.update(
            {
                "rmse": float(rmse),
                "mae": float(mae),
                "r2": float(r2),
            }
        )

    else:  # classification
        # For classification metrics, y_pred is class labels
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        metrics.update(
            {
                "accuracy": float(acc),
                "f1_weighted": float(f1),
            }
        )

        # AUC only if binary and model supports predict_proba
        try:
            if y.nunique() == 2 and hasattr(clf, "predict_proba"):
                y_proba = clf.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_proba)
                metrics["roc_auc"] = float(auc)
        except Exception:
            # Keep it minimal: don't fail training just because AUC couldn't be computed
            pass

    return clf, metrics


def main():
    parser = argparse.ArgumentParser(description="Train model on processed AirBnB data.")
    parser.add_argument("--in-csv", default=DEFAULT_INPUT_CSV, help="Input CSV (processed)")
    parser.add_argument("--target-col", required=True, help="Target column name (required)")
    parser.add_argument(
        "--task",
        choices=["regression", "classification"],
        default="regression",
        help="Task type (default: regression)",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test size fraction (default: 0.2)")
    parser.add_argument("--random-state", type=int, default=42, help="Random state (default: 42)")
    parser.add_argument("--out-model", default=DEFAULT_MODEL_PATH, help="Path to save trained model artifact")
    parser.add_argument("--out-metrics", default=DEFAULT_METRICS_PATH, help="Path to save metrics JSON")
    args = parser.parse_args()

    df = pd.read_csv(args.in_csv)

    model, metrics = train_model(
        df=df,
        target_col=args.target_col,
        task=args.task,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    _ensure_parent_dir(args.out_model)
    joblib.dump(model, args.out_model)

    _ensure_parent_dir(args.out_metrics)
    with open(args.out_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("✅ Training complete")
    print(f"Saved model to:   {args.out_model}")
    print(f"Saved metrics to: {args.out_metrics}")
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()
