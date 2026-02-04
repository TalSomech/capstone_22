# train.py
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import joblib
import wandb

from utils import build_preprocessor, _ensure_parent_dir

# -----------------------------
# Default paths
# -----------------------------
DEFAULT_INPUT_CSV = "data/processed/listings_combined_clean.csv"
DEFAULT_MODEL_PATH = "models/model.joblib"
DEFAULT_METRICS_PATH = "results/metrics.json"

TARGET_COL = "review_scores_rating"

def get_model_configs(random_state: int) -> dict:
    """Return model configurations with hyperparameter search spaces."""
    return {
        "ridge": {
            "model": Ridge(random_state=random_state),
            "params": {
                "model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
            },
        },
        "random_forest": {
            "model": RandomForestRegressor(random_state=random_state, n_jobs=-1),
            "params": {
                "model__n_estimators": [100, 200, 300],
                "model__max_depth": [10, 20, 30, None],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
            },
        },
        "gradient_boosting": {
            "model": GradientBoostingRegressor(random_state=random_state),
            "params": {
                "model__n_estimators": [100, 200, 300],
                "model__max_depth": [3, 5, 7],
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__min_samples_split": [2, 5, 10],
            },
        },
    }


def train_model(
    df: pd.DataFrame,
    model_type: str = "random_forest",
    test_size: float = 0.2,
    random_state: int = 42,
    cv_folds: int = 5,
    n_iter_search: int = 20,
    tune_hyperparams: bool = True,
):
    """
    Train a regression model with cross-validation and optional hyperparameter tuning.

    Args:
        df: Input dataframe with features and target
        model_type: One of 'ridge', 'random_forest', 'gradient_boosting'
        test_size: Fraction of data for test set
        random_state: Random seed for reproducibility
        cv_folds: Number of cross-validation folds
        n_iter_search: Number of iterations for RandomizedSearchCV
        tune_hyperparams: Whether to perform hyperparameter tuning

    Returns:
        (pipeline, metrics_dict, feature_importance_dict)
    """
    # Prepare features and target
    # exclude_cols = [c for c in COLS_TO_EXCLUDE if c in df.columns]
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # Split data FIRST (before target encoding to avoid leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    print(f"Train size: {len(X_train):,}, Test size: {len(X_test):,}")

    # Identify column types
    numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X_train.columns if c not in numeric_cols]

    # Identify which categorical columns should use target encoding (high cardinality)
    from preprocess import TARGET_ENCODE_COLS
    target_encode_cols = [c for c in TARGET_ENCODE_COLS if c in categorical_cols]
    onehot_cols = [c for c in categorical_cols if c not in target_encode_cols]

    if target_encode_cols:
        print(f"Target encoding (high cardinality): {target_encode_cols}")
    if onehot_cols:
        print(f"One-hot encoding (low cardinality): {onehot_cols}")
    print(f"Features: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical")
    print(f"Target: {TARGET_COL} (mean={y.mean():.3f}, std={y.std():.3f})")

    # Build preprocessor with target encoding integrated
    preprocessor = build_preprocessor(numeric_cols, categorical_cols, target_encode_cols)

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

    # Hyperparameter tuning (optional)
    if tune_hyperparams and model_config["params"]:
        print(f"\nTuning hyperparameters ({n_iter_search} iterations, {cv_folds}-fold CV)...")
        search = RandomizedSearchCV(
            pipeline,
            model_config["params"],
            n_iter=n_iter_search,
            cv=cv_folds,
            scoring="neg_root_mean_squared_error",
            random_state=random_state,
            n_jobs=-1,
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
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    cv_rmse = -cv_scores.mean()
    cv_rmse_std = cv_scores.std()

    # Evaluate on test set
    y_pred = pipeline.predict(X_test)
    test_rmse = mean_squared_error(y_test, y_pred) ** 0.5
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)

    # Compute dummy baseline (predict mean)
    dummy_pred = np.full_like(y_test, y_train.mean())
    dummy_rmse = mean_squared_error(y_test, dummy_pred) ** 0.5

    # Feature importance (for tree-based models)
    feature_importance = {}
    if hasattr(pipeline.named_steps["model"], "feature_importances_"):
        # Get feature names after preprocessing
        try:
            feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
        except AttributeError:
            feature_names = numeric_cols + [f"cat_{i}" for i in range(100)]

        importances = pipeline.named_steps["model"].feature_importances_
        # Get top 20 features
        indices = np.argsort(importances)[::-1][:20]
        for i in indices:
            if i < len(feature_names):
                feature_importance[str(feature_names[i])] = float(importances[i])

    # Build metrics dict
    metrics = {
        "model_type": model_type,
        "n_rows": int(len(df)),
        "n_features_numeric": len(numeric_cols),
        "n_features_categorical": len(categorical_cols),
        "test_size": float(test_size),
        "random_state": int(random_state),
        "cv_folds": int(cv_folds),
        "cv_rmse_mean": float(cv_rmse),
        "cv_rmse_std": float(cv_rmse_std),
        "test_rmse": float(test_rmse),
        "test_mae": float(test_mae),
        "test_r2": float(test_r2),
        "dummy_rmse": float(dummy_rmse),
        "rmse_improvement_vs_dummy": float((dummy_rmse - test_rmse) / dummy_rmse * 100),
        "best_params": best_params,
    }

    return pipeline, metrics, feature_importance


def main():
    parser = argparse.ArgumentParser(description="Train regression model on AirBnB data.")
    parser.add_argument("--in-csv", default=DEFAULT_INPUT_CSV, help="Input CSV (processed)")
    parser.add_argument(
        "--model-type",
        choices=["ridge", "random_forest", "gradient_boosting"],
        default="random_forest",
        help="Model type (default: random_forest)",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test size fraction (default: 0.2)")
    parser.add_argument("--random-state", type=int, default=42, help="Random state (default: 42)")
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of CV folds (default: 5)")
    parser.add_argument("--n-iter", type=int, default=20, help="Number of RandomizedSearchCV iterations (default: 20)")
    parser.add_argument("--no-tune", action="store_true", help="Skip hyperparameter tuning")
    parser.add_argument("--out-model", default=DEFAULT_MODEL_PATH, help="Path to save trained model")
    parser.add_argument("--out-metrics", default=DEFAULT_METRICS_PATH, help="Path to save metrics JSON")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases experiment tracking")
    args = parser.parse_args()

    # Initialize wandb if enabled
    use_wandb = args.wandb
    if use_wandb:
        wandb.init(
            project="airbnb-capstone",
            config={
                "model_type": args.model_type,
                "test_size": args.test_size,
                "random_state": args.random_state,
                "cv_folds": args.cv_folds,
                "n_iter": args.n_iter,
                "tune_hyperparams": not args.no_tune,
            },
        )

    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    df = pd.read_csv(args.in_csv)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

    print()
    print("=" * 60)
    print(f"TRAINING MODEL: {args.model_type.upper()}")
    print("=" * 60)

    pipeline, metrics, feature_importance = train_model(
        df=df,
        model_type=args.model_type,
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
    print(f"Cross-Validation RMSE: {metrics['cv_rmse_mean']:.4f} (+/- {metrics['cv_rmse_std']:.4f})")
    print(f"Test RMSE:             {metrics['test_rmse']:.4f}")
    print(f"Test MAE:              {metrics['test_mae']:.4f}")
    print(f"Test R2:               {metrics['test_r2']:.4f}")
    print(f"Dummy RMSE (baseline): {metrics['dummy_rmse']:.4f}")
    print(f"Improvement vs Dummy:  {metrics['rmse_improvement_vs_dummy']:.1f}%")

    if feature_importance:
        print()
        print("Top 10 Feature Importances:")
        for i, (feat, imp) in enumerate(list(feature_importance.items())[:10], 1):
            print(f"  {i:2d}. {feat}: {imp:.4f}")

    print()
    print(f"Model saved to:   {args.out_model}")
    print(f"Metrics saved to: {args.out_metrics}")

    # Log to wandb
    if use_wandb:
        # Log best hyperparameters
        if metrics.get("best_params"):
            wandb.config.update(metrics["best_params"])

        # Log metrics
        wandb.log({
            "cv_rmse": metrics["cv_rmse_mean"],
            "cv_rmse_std": metrics["cv_rmse_std"],
            "test_rmse": metrics["test_rmse"],
            "test_mae": metrics["test_mae"],
            "test_r2": metrics["test_r2"],
            "dummy_rmse": metrics["dummy_rmse"],
            "improvement_vs_dummy": metrics["rmse_improvement_vs_dummy"],
        })

        # Log feature importances as a table
        if feature_importance:
            table = wandb.Table(columns=["feature", "importance"])
            for feat, imp in feature_importance.items():
                table.add_data(feat, imp)
            wandb.log({
                "feature_importance": wandb.plot.bar(
                    table, "feature", "importance", title="Feature Importances"
                ),
            })

        # Log model as artifact
        artifact = wandb.Artifact(
            f"model-{args.model_type}", type="model",
            description=f"{args.model_type} trained on AirBnB data",
        )
        artifact.add_file(args.out_model)
        wandb.log_artifact(artifact)

        wandb.finish()
        print("Wandb run finished.")


if __name__ == "__main__":
    main()