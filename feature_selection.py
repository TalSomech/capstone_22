# feature_selection.py
"""
Feature Selection Script for AirBnB Rating Prediction

This script:
1. Ranks all features by importance
2. Tests different feature subsets (top N features)
3. Finds the optimal number of features for lowest RMSE
4. Saves detailed results including all feature importances
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

import joblib

# -----------------------------
# Default paths
# -----------------------------
DEFAULT_INPUT_CSV = "data/processed/listings_combined_clean.csv"
DEFAULT_RESULTS_PATH = "results/feature_selection_results.json"

TARGET_COL = "review_scores_rating"
COLS_TO_EXCLUDE = ["city"]


def _ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


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


def get_feature_importances(
    df: pd.DataFrame,
    random_state: int = 42,
    n_estimators: int = 100,
):
    """
    Train a quick Random Forest and extract feature importances.
    Returns a dict of {feature_name: importance} sorted by importance.
    """
    # Prepare data
    X = df.drop(columns=[TARGET_COL] + [c for c in COLS_TO_EXCLUDE if c in df.columns])
    y = df[TARGET_COL]

    # Identify column types
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    # Build preprocessor and pipeline
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=2,
            max_depth=20,  # Limit depth for speed
        )),
    ])

    # Fit on all data to get importances
    pipeline.fit(X, y)

    # Get feature names after preprocessing
    try:
        feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    except AttributeError:
        feature_names = numeric_cols + [f"cat_{i}" for i in range(100)]

    # Get importances
    importances = pipeline.named_steps["model"].feature_importances_

    # Create dict and sort by importance
    importance_dict = {}
    for i, name in enumerate(feature_names):
        if i < len(importances):
            # Clean up the feature name (remove num__ or cat__ prefix)
            clean_name = str(name)
            if clean_name.startswith("num__"):
                clean_name = clean_name[5:]
            elif clean_name.startswith("cat__"):
                clean_name = clean_name[5:]
            importance_dict[clean_name] = float(importances[i])

    # Sort by importance (descending)
    sorted_importance = dict(sorted(
        importance_dict.items(),
        key=lambda x: x[1],
        reverse=True
    ))

    return sorted_importance, numeric_cols, categorical_cols


def evaluate_feature_subset(
    df: pd.DataFrame,
    features_to_use: list,
    random_state: int = 42,
    test_size: float = 0.2,
    cv_folds: int = 5,
):
    """
    Evaluate model performance using only the specified features.
    Returns RMSE, MAE, R2.
    """
    # Filter to only use specified features + target
    available_features = [f for f in features_to_use if f in df.columns]

    X = df[available_features]
    y = df[TARGET_COL]

    # Identify column types
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    # Build preprocessor and pipeline
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=200,
            random_state=random_state,
            n_jobs=2,
            max_depth=30,
            min_samples_split=10,
            min_samples_leaf=4,
        )),
    ])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    # Fit and evaluate
    pipeline.fit(X_train, y_train)

    # Cross-validation
    cv_scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=cv_folds,
        scoring="neg_root_mean_squared_error",
        n_jobs=2,
    )
    cv_rmse = -cv_scores.mean()

    # Test set evaluation
    y_pred = pipeline.predict(X_test)
    test_rmse = mean_squared_error(y_test, y_pred) ** 0.5
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)

    return {
        "cv_rmse": float(cv_rmse),
        "test_rmse": float(test_rmse),
        "test_mae": float(test_mae),
        "test_r2": float(test_r2),
        "n_features": len(available_features),
        "features_used": available_features,
    }


def main():
    parser = argparse.ArgumentParser(description="Feature selection for AirBnB rating prediction.")
    parser.add_argument("--in-csv", default=DEFAULT_INPUT_CSV, help="Input CSV (processed)")
    parser.add_argument("--out-results", default=DEFAULT_RESULTS_PATH, help="Output path for results JSON")
    parser.add_argument("--random-state", type=int, default=42, help="Random state (default: 42)")
    args = parser.parse_args()

    print("=" * 60)
    print("FEATURE SELECTION ANALYSIS")
    print("=" * 60)

    # Load data
    df = pd.read_csv(args.in_csv)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Step 1: Get feature importances
    print()
    print("-" * 60)
    print("Step 1: Computing Feature Importances")
    print("-" * 60)

    importances, numeric_cols, categorical_cols = get_feature_importances(
        df, random_state=args.random_state
    )

    print(f"\nFound {len(importances)} features after preprocessing")
    print("\nAll Features Ranked by Importance:")
    print("-" * 40)
    for i, (feat, imp) in enumerate(importances.items(), 1):
        print(f"{i:3d}. {feat:40s} {imp:.6f}")

    # Step 2: Map back to original column names
    # Get the original column names (before one-hot encoding)
    original_features = [c for c in df.columns if c not in [TARGET_COL] + COLS_TO_EXCLUDE]

    # Aggregate importances for categorical features (sum one-hot encoded importances)
    original_importances = {}
    for col in original_features:
        if col in numeric_cols:
            # Numeric feature - direct lookup
            if col in importances:
                original_importances[col] = importances[col]
            else:
                original_importances[col] = 0.0
        else:
            # Categorical feature - sum all one-hot encoded values
            total_imp = 0.0
            for feat_name, imp in importances.items():
                if feat_name.startswith(col + "_") or feat_name == col:
                    total_imp += imp
            original_importances[col] = total_imp

    # Sort original importances
    original_importances = dict(sorted(
        original_importances.items(),
        key=lambda x: x[1],
        reverse=True
    ))

    print()
    print("\nOriginal Features Ranked by Importance:")
    print("-" * 40)
    for i, (feat, imp) in enumerate(original_importances.items(), 1):
        print(f"{i:3d}. {feat:40s} {imp:.6f}")

    # Step 3: Test different feature subsets
    print()
    print("-" * 60)
    print("Step 2: Testing Feature Subsets")
    print("-" * 60)

    # Get ranked feature names
    ranked_features = list(original_importances.keys())

    # Test different numbers of features
    subset_sizes = [5, 10, 15, 20, 25, 30, 35, 40, len(ranked_features)]
    subset_sizes = [s for s in subset_sizes if s <= len(ranked_features)]
    subset_sizes = sorted(set(subset_sizes))  # Remove duplicates and sort

    subset_results = []
    best_rmse = float('inf')
    best_n_features = 0
    best_features = []

    for n in subset_sizes:
        features_subset = ranked_features[:n]
        print(f"\nTesting top {n} features...")

        result = evaluate_feature_subset(
            df,
            features_to_use=features_subset,
            random_state=args.random_state,
        )

        subset_results.append({
            "n_features": n,
            "test_rmse": result["test_rmse"],
            "test_mae": result["test_mae"],
            "test_r2": result["test_r2"],
            "cv_rmse": result["cv_rmse"],
        })

        print(f"  Test RMSE: {result['test_rmse']:.4f}, R²: {result['test_r2']:.4f}")

        if result["test_rmse"] < best_rmse:
            best_rmse = result["test_rmse"]
            best_n_features = n
            best_features = features_subset.copy()

    # Step 4: Print summary
    print()
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    print("\nRMSE by Number of Features:")
    print("-" * 40)
    for r in subset_results:
        marker = " <-- BEST" if r["n_features"] == best_n_features else ""
        print(f"  {r['n_features']:3d} features: RMSE = {r['test_rmse']:.4f}, R² = {r['test_r2']:.4f}{marker}")

    print()
    print(f"BEST: {best_n_features} features with RMSE = {best_rmse:.4f}")
    print()
    print("Best Feature Set:")
    for i, feat in enumerate(best_features, 1):
        imp = original_importances.get(feat, 0)
        print(f"  {i:2d}. {feat} (importance: {imp:.6f})")

    # Step 5: Save results
    _ensure_parent_dir(args.out_results)

    results = {
        "summary": {
            "total_features_available": len(original_features),
            "best_n_features": best_n_features,
            "best_test_rmse": best_rmse,
            "best_features": best_features,
        },
        "all_features_importance": original_importances,
        "detailed_feature_importance": importances,
        "subset_comparison": subset_results,
        "feature_categories": {
            "numeric_features": numeric_cols,
            "categorical_features": categorical_cols,
        },
    }

    with open(args.out_results, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print()
    print(f"Results saved to: {args.out_results}")


if __name__ == "__main__":
    main()
