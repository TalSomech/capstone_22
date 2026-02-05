# predict.py
"""
Prediction module for Airbnb rating model.

Provides functions for:
- Loading trained model
- Running predictions on preprocessed data
- Running predictions on raw user data (with preprocessing)
"""
import argparse
from pathlib import Path

import pandas as pd
import joblib

from preprocess import preprocess_for_inference, load_feature_template
from utils import _ensure_parent_dir

DEFAULT_MODEL_PATH = "models/model.joblib"
DEFAULT_TEMPLATE_PATH = "models/feature_template.json"
DEFAULT_INPUT_CSV = "data/processed/listings_combined_clean.csv"
DEFAULT_OUTPUT_CSV = "results/predictions.csv"


def load_model(model_path: str = DEFAULT_MODEL_PATH):
    """
    Load trained model from disk.

    Args:
        model_path: Path to the joblib model file

    Returns:
        Loaded sklearn pipeline/model
    """
    return joblib.load(model_path)


def predict_preprocessed(model, df: pd.DataFrame) -> pd.Series:
    """
    Run predictions on already-preprocessed data.

    Args:
        model: Trained sklearn model/pipeline
        df: DataFrame that's already in the correct format for the model

    Returns:
        Series of predictions
    """
    return pd.Series(model.predict(df), index=df.index)


def predict_raw(
    model,
    raw_df: pd.DataFrame,
    feature_template: dict,
) -> pd.Series:
    """
    Run predictions on raw user data.

    This function handles the full pipeline:
    1. Preprocess raw data to match model's expected format
    2. Run model predictions

    Args:
        model: Trained sklearn model/pipeline
        raw_df: Raw DataFrame from user (CSV upload, form input, etc.)
        feature_template: Dict with 'columns', 'medians', 'modes'

    Returns:
        Series of predictions
    """
    # Preprocess to match model format
    processed_df = preprocess_for_inference(raw_df, feature_template)

    # Run predictions
    return predict_preprocessed(model, processed_df)


def predict_to_file(
    model_path: str,
    input_csv: str,
    output_csv: str,
):
    """
    Load model, run predictions on CSV, save results.

    Legacy function for CLI usage with preprocessed data.
    """
    model = load_model(model_path)
    df = pd.read_csv(input_csv)

    predictions = model.predict(df)

    output_df = df.copy()
    output_df["prediction"] = predictions

    _ensure_parent_dir(output_csv)
    output_df.to_csv(output_csv, index=False)

    return output_df


def main():
    parser = argparse.ArgumentParser(description="Run inference using trained model.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Path to trained model artifact")
    parser.add_argument("--in-csv", default=DEFAULT_INPUT_CSV, help="Input CSV for prediction")
    parser.add_argument("--out-csv", default=DEFAULT_OUTPUT_CSV, help="Output CSV with predictions")
    parser.add_argument("--raw", action="store_true", help="Input is raw data (needs preprocessing)")
    parser.add_argument("--template-path", default=DEFAULT_TEMPLATE_PATH, help="Path to feature template JSON")
    args = parser.parse_args()

    model = load_model(args.model_path)
    df = pd.read_csv(args.in_csv)

    if args.raw:
        # Raw data needs preprocessing
        template = load_feature_template(args.template_path)
        predictions = predict_raw(model, df, template)
    else:
        # Already preprocessed
        predictions = predict_preprocessed(model, df)

    output_df = df.copy()
    output_df["prediction"] = predictions

    _ensure_parent_dir(args.out_csv)
    output_df.to_csv(args.out_csv, index=False)

    print("Prediction complete")
    print(f"Model used:      {args.model_path}")
    print(f"Input data:      {args.in_csv}")
    print(f"Predictions to:  {args.out_csv}")


if __name__ == "__main__":
    main()
