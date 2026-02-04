# feature_diagnostics.py
"""
Feature Diagnostics Script for AirBnB Rating Prediction

This script performs:
1. Permutation Importance - More reliable than impurity-based importance
2. Correlation Matrix - Identify features with |r| > 0.9
3. VIF Analysis - Check Variance Inflation Factor for multicollinearity
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor

import joblib

from utils import _ensure_parent_dir

# -----------------------------
# Default paths
# -----------------------------
DEFAULT_INPUT_CSV = "data/processed/listings_combined_clean.csv"
DEFAULT_OUTPUT_DIR = "results/diagnostics"

TARGET_COL = "review_scores_rating"

# Same exclusions as train.py
COLS_TO_EXCLUDE = [
    "city", "host_id", "room_type",
    "minimum_minimum_nights", "maximum_minimum_nights",
    "minimum_maximum_nights", "maximum_maximum_nights",
    "minimum_nights_avg_ntm", "maximum_nights_avg_ntm",
    "availability_30", "availability_60", "availability_90",
    "number_of_reviews_ltm", "number_of_reviews_l30d",
    "calculated_host_listings_count_entire_homes",
    "calculated_host_listings_count_private_rooms",
    "calculated_host_listings_count_shared_rooms",
    "host_total_listings_count",
    "host_has_profile_pic",
]




def load_and_prepare_data(csv_path: str):
    """Load data and prepare features."""
    df = pd.read_csv(csv_path)

    # Prepare features
    X = df.drop(columns=[TARGET_COL] + [c for c in COLS_TO_EXCLUDE if c in df.columns])
    y = df[TARGET_COL]

    # Identify column types
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    return X, y, numeric_cols, categorical_cols


def analyze_correlations(X: pd.DataFrame, numeric_cols: list, output_dir: str):
    """
    Analyze correlations between numeric features.
    Identify pairs with |r| > 0.9 (high multicollinearity).
    """
    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS")
    print("=" * 60)

    # Get only numeric columns that exist
    numeric_df = X[numeric_cols].copy()

    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()

    # Find highly correlated pairs (|r| > 0.9)
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.9:
                high_corr_pairs.append({
                    "feature_1": corr_matrix.columns[i],
                    "feature_2": corr_matrix.columns[j],
                    "correlation": round(corr_val, 4)
                })

    # Find moderately correlated pairs (|r| > 0.7)
    moderate_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if 0.7 < abs(corr_val) <= 0.9:
                moderate_corr_pairs.append({
                    "feature_1": corr_matrix.columns[i],
                    "feature_2": corr_matrix.columns[j],
                    "correlation": round(corr_val, 4)
                })

    # Print results
    if high_corr_pairs:
        print(f"\nHIGH CORRELATION (|r| > 0.9) - {len(high_corr_pairs)} pairs found:")
        print("-" * 60)
        for pair in sorted(high_corr_pairs, key=lambda x: abs(x["correlation"]), reverse=True):
            print(f"  {pair['feature_1']:30s} <-> {pair['feature_2']:30s} r={pair['correlation']:+.4f}")
    else:
        print("\nNo feature pairs with |r| > 0.9 found.")

    if moderate_corr_pairs:
        print(f"\nMODERATE CORRELATION (0.7 < |r| <= 0.9) - {len(moderate_corr_pairs)} pairs:")
        print("-" * 60)
        for pair in sorted(moderate_corr_pairs, key=lambda x: abs(x["correlation"]), reverse=True)[:10]:
            print(f"  {pair['feature_1']:30s} <-> {pair['feature_2']:30s} r={pair['correlation']:+.4f}")
        if len(moderate_corr_pairs) > 10:
            print(f"  ... and {len(moderate_corr_pairs) - 10} more pairs")

    # Save correlation matrix plot
    plt.figure(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, square=True, linewidths=0.5)
    plt.title("Feature Correlation Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_matrix.png", dpi=150)
    plt.close()
    print(f"\nCorrelation matrix saved to: {output_dir}/correlation_matrix.png")

    return {
        "high_correlation_pairs": high_corr_pairs,
        "moderate_correlation_pairs": moderate_corr_pairs[:20],  # Top 20
    }


def analyze_vif(X: pd.DataFrame, numeric_cols: list, output_dir: str):
    """
    Calculate Variance Inflation Factor (VIF) for numeric features.
    VIF > 10 indicates serious multicollinearity.
    VIF > 5 is concerning.
    """
    print("\n" + "=" * 60)
    print("VIF ANALYSIS (Variance Inflation Factor)")
    print("=" * 60)

    # Get numeric data and handle missing values
    numeric_df = X[numeric_cols].copy()
    numeric_df = numeric_df.fillna(numeric_df.median())

    # Remove columns with zero variance
    zero_var_cols = numeric_df.columns[numeric_df.std() == 0].tolist()
    if zero_var_cols:
        print(f"\nRemoving {len(zero_var_cols)} zero-variance columns: {zero_var_cols}")
        numeric_df = numeric_df.drop(columns=zero_var_cols)

    # Calculate VIF for each feature
    vif_data = []
    for i, col in enumerate(numeric_df.columns):
        try:
            vif = variance_inflation_factor(numeric_df.values, i)
            vif_data.append({"feature": col, "VIF": round(vif, 2)})
        except Exception as e:
            vif_data.append({"feature": col, "VIF": float('inf')})

    # Sort by VIF descending
    vif_df = pd.DataFrame(vif_data).sort_values("VIF", ascending=False)

    # Categorize
    severe = vif_df[vif_df["VIF"] > 10]
    concerning = vif_df[(vif_df["VIF"] > 5) & (vif_df["VIF"] <= 10)]
    acceptable = vif_df[vif_df["VIF"] <= 5]

    print(f"\nVIF Categories:")
    print(f"  Severe (VIF > 10):    {len(severe)} features")
    print(f"  Concerning (5-10):    {len(concerning)} features")
    print(f"  Acceptable (VIF <= 5): {len(acceptable)} features")

    if len(severe) > 0:
        print(f"\nSEVERE MULTICOLLINEARITY (VIF > 10):")
        print("-" * 40)
        for _, row in severe.head(15).iterrows():
            print(f"  {row['feature']:35s} VIF = {row['VIF']:>8.2f}")

    if len(concerning) > 0:
        print(f"\nCONCERNING (5 < VIF <= 10):")
        print("-" * 40)
        for _, row in concerning.head(10).iterrows():
            print(f"  {row['feature']:35s} VIF = {row['VIF']:>8.2f}")

    # Save VIF results
    vif_df.to_csv(f"{output_dir}/vif_analysis.csv", index=False)
    print(f"\nVIF analysis saved to: {output_dir}/vif_analysis.csv")

    return {
        "severe_multicollinearity": severe.to_dict("records"),
        "concerning_multicollinearity": concerning.to_dict("records"),
        "total_features": len(vif_df),
    }


def analyze_permutation_importance(
    X: pd.DataFrame, y: pd.Series,
    numeric_cols: list, categorical_cols: list,
    output_dir: str, random_state: int = 42
):
    """
    Calculate permutation importance.
    More reliable than impurity-based importance for Random Forest.
    """
    print("\n" + "=" * 60)
    print("PERMUTATION IMPORTANCE ANALYSIS")
    print("=" * 60)

    # Build preprocessor
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

    # Build and train model
    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            random_state=random_state,
            n_jobs=2,
        )),
    ])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    print("Training model for permutation importance...")
    pipeline.fit(X_train, y_train)

    # Calculate permutation importance on test set
    print("Calculating permutation importance (this may take a few minutes)...")
    perm_importance = permutation_importance(
        pipeline, X_test, y_test,
        n_repeats=10,
        random_state=random_state,
        n_jobs=2,
    )

    # Get feature names
    feature_names = X.columns.tolist()

    # Create importance dataframe
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": perm_importance.importances_mean,
        "importance_std": perm_importance.importances_std,
    }).sort_values("importance_mean", ascending=False)

    # Print top features
    print("\nTOP 20 FEATURES BY PERMUTATION IMPORTANCE:")
    print("-" * 60)
    for i, row in importance_df.head(20).iterrows():
        print(f"  {row['feature']:35s} {row['importance_mean']:+.6f} (+/- {row['importance_std']:.6f})")

    # Identify features with negative or zero importance
    negative_importance = importance_df[importance_df["importance_mean"] <= 0]
    if len(negative_importance) > 0:
        print(f"\nFEATURES WITH ZERO OR NEGATIVE IMPORTANCE ({len(negative_importance)}):")
        print("(These features may be candidates for removal)")
        print("-" * 60)
        for _, row in negative_importance.iterrows():
            print(f"  {row['feature']:35s} {row['importance_mean']:+.6f}")

    # Save results
    importance_df.to_csv(f"{output_dir}/permutation_importance.csv", index=False)

    # Create bar plot
    plt.figure(figsize=(12, 10))
    top_n = min(30, len(importance_df))
    top_features = importance_df.head(top_n)

    colors = ['green' if x > 0 else 'red' for x in top_features["importance_mean"]]
    plt.barh(range(top_n), top_features["importance_mean"], xerr=top_features["importance_std"],
             color=colors, alpha=0.7)
    plt.yticks(range(top_n), top_features["feature"])
    plt.xlabel("Permutation Importance (decrease in R²)")
    plt.title("Feature Permutation Importance (Top 30)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/permutation_importance.png", dpi=150)
    plt.close()

    print(f"\nPermutation importance saved to: {output_dir}/permutation_importance.csv")
    print(f"Plot saved to: {output_dir}/permutation_importance.png")

    return {
        "top_20_features": importance_df.head(20).to_dict("records"),
        "features_to_consider_removing": negative_importance["feature"].tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Feature diagnostics for AirBnB model.")
    parser.add_argument("--in-csv", default=DEFAULT_INPUT_CSV, help="Input CSV (processed)")
    parser.add_argument("--out-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory for results")
    parser.add_argument("--random-state", type=int, default=42, help="Random state")
    args = parser.parse_args()

    _ensure_parent_dir(args.out_dir)


    print("=" * 60)
    print("FEATURE DIAGNOSTICS")
    print("=" * 60)

    # Load data
    X, y, numeric_cols, categorical_cols = load_and_prepare_data(args.in_csv)
    print(f"Loaded {len(X):,} rows, {len(X.columns)} features")
    print(f"  Numeric: {len(numeric_cols)}, Categorical: {len(categorical_cols)}")

    # Run analyses
    results = {}

    # 1. Correlation analysis
    results["correlation"] = analyze_correlations(X, numeric_cols, args.out_dir)

    # 2. VIF analysis
    results["vif"] = analyze_vif(X, numeric_cols, args.out_dir)

    # 3. Permutation importance
    results["permutation_importance"] = analyze_permutation_importance(
        X, y, numeric_cols, categorical_cols, args.out_dir, args.random_state
    )

    # Save combined results
    with open(f"{args.out_dir}/diagnostics_summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 60)
    print(f"All results saved to: {args.out_dir}/")

    # Summary recommendations
    print("\nRECOMMENDATIONS:")
    print("-" * 60)

    n_high_corr = len(results["correlation"]["high_correlation_pairs"])
    n_severe_vif = len(results["vif"]["severe_multicollinearity"])
    n_negative_imp = len(results["permutation_importance"]["features_to_consider_removing"])

    if n_high_corr > 0:
        print(f"  - {n_high_corr} feature pairs have |r| > 0.9. Consider dropping one from each pair.")
    if n_severe_vif > 0:
        print(f"  - {n_severe_vif} features have VIF > 10. High multicollinearity detected.")
    if n_negative_imp > 0:
        print(f"  - {n_negative_imp} features have zero/negative permutation importance. Consider removal.")


if __name__ == "__main__":
    main()
