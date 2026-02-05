# train_classifier.py
"""
Classification approach: Predict "excellent" (rating >= 4.8) vs "not excellent" ratings.
This is often more practical than predicting exact scores.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import joblib

from utils import _ensure_parent_dir

# -----------------------------
# Default paths
# -----------------------------
DEFAULT_INPUT_CSV = "data/processed/listings_combined_clean.csv"
DEFAULT_MODEL_PATH = "models/classifier.joblib"
DEFAULT_METRICS_PATH = "results/classifier_metrics.json"

TARGET_COL = "review_scores_rating"
EXCELLENT_THRESHOLD = 4.8  # Ratings >= this are "excellent"

# Columns to exclude from features
COLS_TO_EXCLUDE = ["city"]



def build_preprocessor(numeric_cols: list, categorical_cols: list) -> ColumnTransformer:
    """Build sklearn preprocessing pipeline."""
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor


def get_model_configs(random_state: int) -> dict:
    """Return classifier configurations with hyperparameter search spaces."""
    return {
        "logistic": {
            "model": LogisticRegression(random_state=random_state, max_iter=1000),
            "params": {
                "model__C": [0.01, 0.1, 1.0, 10.0],
                "model__penalty": ["l2"],
            },
        },
        "random_forest": {
            "model": RandomForestClassifier(random_state=random_state, n_jobs=2),
            "params": {
                "model__n_estimators": [100, 200, 300],
                "model__max_depth": [10, 20, 30, None],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
            },
        },
        "gradient_boosting": {
            "model": GradientBoostingClassifier(random_state=random_state),
            "params": {
                "model__n_estimators": [100, 200],
                "model__max_depth": [3, 5, 7],
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__min_samples_split": [2, 5, 10],
            },
        },
    }


def train_classifier(
    df: pd.DataFrame,
    model_type: str = "random_forest",
    threshold: float = EXCELLENT_THRESHOLD,
    test_size: float = 0.2,
    random_state: int = 42,
    cv_folds: int = 5,
    n_iter_search: int = 20,
    tune_hyperparams: bool = True,
):
    """
    Train a classification model to predict excellent vs not-excellent ratings.

    Args:
        df: Input dataframe with features and target
        model_type: One of 'logistic', 'random_forest', 'gradient_boosting'
        threshold: Rating threshold for "excellent" (default 4.8)
        test_size: Fraction of data for test set
        random_state: Random seed for reproducibility
        cv_folds: Number of cross-validation folds
        n_iter_search: Number of iterations for RandomizedSearchCV
        tune_hyperparams: Whether to perform hyperparameter tuning

    Returns:
        (pipeline, metrics_dict, feature_importance_dict)
    """
    # Create binary target: 1 = excellent (>= threshold), 0 = not excellent
    y = (df[TARGET_COL] >= threshold).astype(int)
    X = df.drop(columns=[TARGET_COL] + [c for c in COLS_TO_EXCLUDE if c in df.columns])

    # Show class distribution
    n_excellent = y.sum()
    n_total = len(y)
    print(f"Target: rating >= {threshold} is 'excellent'")
    print(f"Class distribution: {n_excellent:,} excellent ({n_excellent/n_total*100:.1f}%), "
          f"{n_total - n_excellent:,} not excellent ({(n_total-n_excellent)/n_total*100:.1f}%)")

    # Identify column types
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    print(f"Features: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical")

    # Build preprocessor
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    # Get model config
    model_configs = get_model_configs(random_state)
    if model_type not in model_configs:
        raise ValueError(f"Unknown model_type: {model_type}. Choose from {list(model_configs.keys())}")

    model_config = model_configs[model_type]

    # Build pipeline
    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model_config["model"]),
    ])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,  # Keep class proportions in train/test
        shuffle=True,
    )

    print(f"Train size: {len(X_train):,}, Test size: {len(X_test):,}")

    # Hyperparameter tuning (optional)
    if tune_hyperparams and model_config["params"]:
        print(f"\nTuning hyperparameters ({n_iter_search} iterations, {cv_folds}-fold CV)...")
        search = RandomizedSearchCV(
            pipeline,
            model_config["params"],
            n_iter=n_iter_search,
            cv=cv_folds,
            scoring="roc_auc",  # Use AUC for classification
            random_state=random_state,
            n_jobs=2,
            verbose=1,
        )
        search.fit(X_train, y_train)
        pipeline = search.best_estimator_
        best_params = search.best_params_
        print(f"Best params: {best_params}")
    else:
        pipeline.fit(X_train, y_train)
        best_params = {}

    # Cross-validation on training set
    print(f"\nRunning {cv_folds}-fold cross-validation...")
    cv_scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=cv_folds,
        scoring="roc_auc",
        n_jobs=2,
    )
    cv_auc = cv_scores.mean()
    cv_auc_std = cv_scores.std()

    # Evaluate on test set
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    test_accuracy = accuracy_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred)
    test_recall = recall_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)
    test_auc = roc_auc_score(y_test, y_pred_proba)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Compute baseline (always predict majority class)
    majority_class = y_train.mode()[0]
    baseline_accuracy = (y_test == majority_class).mean()

    # Feature importance (for tree-based models)
    feature_importance = {}
    if hasattr(pipeline.named_steps["model"], "feature_importances_"):
        try:
            feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
        except AttributeError:
            feature_names = numeric_cols + [f"cat_{i}" for i in range(100)]

        importances = pipeline.named_steps["model"].feature_importances_
        indices = np.argsort(importances)[::-1][:20]
        for i in indices:
            if i < len(feature_names):
                feature_importance[str(feature_names[i])] = float(importances[i])

    # Build metrics dict
    metrics = {
        "model_type": model_type,
        "threshold": float(threshold),
        "n_rows": int(len(df)),
        "n_excellent": int(n_excellent),
        "n_not_excellent": int(n_total - n_excellent),
        "class_balance": f"{n_excellent/n_total*100:.1f}% excellent",
        "n_features_numeric": len(numeric_cols),
        "n_features_categorical": len(categorical_cols),
        "test_size": float(test_size),
        "random_state": int(random_state),
        "cv_folds": int(cv_folds),
        "cv_auc_mean": float(cv_auc),
        "cv_auc_std": float(cv_auc_std),
        "test_accuracy": float(test_accuracy),
        "test_precision": float(test_precision),
        "test_recall": float(test_recall),
        "test_f1": float(test_f1),
        "test_auc": float(test_auc),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "baseline_accuracy": float(baseline_accuracy),
        "accuracy_improvement_vs_baseline": float((test_accuracy - baseline_accuracy) / baseline_accuracy * 100),
        "best_params": best_params,
    }

    return pipeline, metrics, feature_importance


def main():
    parser = argparse.ArgumentParser(description="Train classification model on AirBnB data.")
    parser.add_argument("--in-csv", default=DEFAULT_INPUT_CSV, help="Input CSV (processed)")
    parser.add_argument(
        "--model-type",
        choices=["logistic", "random_forest", "gradient_boosting"],
        default="random_forest",
        help="Model type (default: random_forest)",
    )
    parser.add_argument("--threshold", type=float, default=EXCELLENT_THRESHOLD,
                        help=f"Rating threshold for 'excellent' (default: {EXCELLENT_THRESHOLD})")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test size fraction (default: 0.2)")
    parser.add_argument("--random-state", type=int, default=42, help="Random state (default: 42)")
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of CV folds (default: 5)")
    parser.add_argument("--n-iter", type=int, default=20, help="Number of RandomizedSearchCV iterations (default: 20)")
    parser.add_argument("--no-tune", action="store_true", help="Skip hyperparameter tuning")
    parser.add_argument("--out-model", default=DEFAULT_MODEL_PATH, help="Path to save trained model")
    parser.add_argument("--out-metrics", default=DEFAULT_METRICS_PATH, help="Path to save metrics JSON")
    args = parser.parse_args()

    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    df = pd.read_csv(args.in_csv)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

    print()
    print("=" * 60)
    print(f"TRAINING CLASSIFIER: {args.model_type.upper()}")
    print("=" * 60)

    pipeline, metrics, feature_importance = train_classifier(
        df=df,
        model_type=args.model_type,
        threshold=args.threshold,
        test_size=args.test_size,
        random_state=args.random_state,
        cv_folds=args.cv_folds,
        n_iter_search=args.n_iter,
        tune_hyperparams=not args.no_tune,
    )

    # Save model
    _ensure_parent_dir(args.out_model)
    joblib.dump(pipeline, args.out_model)

    # Save metrics
    _ensure_parent_dir(args.out_metrics)
    output = {
        "metrics": metrics,
        "feature_importance": feature_importance,
    }
    with open(args.out_metrics, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    # Print results
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Cross-Validation AUC: {metrics['cv_auc_mean']:.4f} (+/- {metrics['cv_auc_std']:.4f})")
    print()
    print(f"Test Accuracy:        {metrics['test_accuracy']:.4f} ({metrics['test_accuracy']*100:.1f}%)")
    print(f"Test Precision:       {metrics['test_precision']:.4f}")
    print(f"Test Recall:          {metrics['test_recall']:.4f}")
    print(f"Test F1 Score:        {metrics['test_f1']:.4f}")
    print(f"Test AUC:             {metrics['test_auc']:.4f}")
    print()
    print(f"Baseline Accuracy:    {metrics['baseline_accuracy']:.4f} (always predict majority)")
    print(f"Improvement:          {metrics['accuracy_improvement_vs_baseline']:.1f}%")
    print()
    print("Confusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"                 Predicted")
    print(f"              Not Exc  Excellent")
    print(f"Actual Not Exc  {cm['true_negatives']:5d}    {cm['false_positives']:5d}")
    print(f"Actual Excellent{cm['false_negatives']:5d}    {cm['true_positives']:5d}")

    if feature_importance:
        print()
        print("Top 10 Feature Importances:")
        for i, (feat, imp) in enumerate(list(feature_importance.items())[:10], 1):
            print(f"  {i:2d}. {feat}: {imp:.4f}")

    print()
    print(f"Model saved to:   {args.out_model}")
    print(f"Metrics saved to: {args.out_metrics}")


if __name__ == "__main__":
    main()
