# predict.py
# predict.py
import argparse
from pathlib import Path

import pandas as pd
import joblib

from utils import _ensure_parent_dir

DEFAULT_MODEL_PATH = "models/model.joblib"
DEFAULT_INPUT_CSV = "data/processed/listings_combined_clean.csv"
DEFAULT_OUTPUT_CSV = "results/predictions.csv"


def predict(
    model_path: str,
    input_csv: str,
    output_csv: str,
):
    # Load model pipeline
    model = joblib.load(model_path)

    # Load input data
    df = pd.read_csv(input_csv)

    # Predict
    predictions = model.predict(df)

    # Save results
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
    args = parser.parse_args()

    predict(
        model_path=args.model_path,
        input_csv=args.in_csv,
        output_csv=args.out_csv,
    )

    print("✅ Prediction complete")
    print(f"Model used:      {args.model_path}")
    print(f"Input data:      {args.in_csv}")
    print(f"Predictions to:  {args.out_csv}")


if __name__ == "__main__":
    main()