# # preprocess.py
import argparse
import json
from pathlib import Path

import pandas as pd

# -----------------------------
# Default paths (edit as needed)
# -----------------------------
DEFAULT_LA_PATH = "data/raw/listingsLA.csv"
DEFAULT_NYC_PATH = "data/raw/listingsNYC.csv"

DEFAULT_OUTPUT_CSV = "data/processed/listings_combined_clean.csv"
DEFAULT_SUMMARY_JSON = "data/processed/cleaning_summary.json"

COLUMNS_TO_EXCLUDE = [
    "host_name", "last_scraped", "first_review", "last_review", "host_since",
    "name", "description", "neighborhood_overview", "host_about",
    "host_verifications", "amenities", "bathrooms_text",
]


def preprocess(la_path: str, nyc_path: str):
    """
    Loads LA + NYC listings, adds a 'city' column, concatenates them,
    drops selected columns (if exist), drops duplicates,
    and returns (df_clean, cleaning_summary).
    """
    df_la = pd.read_csv(la_path)
    df_la["city"] = "LA"

    df_nyc = pd.read_csv(nyc_path)
    df_nyc["city"] = "NYC"

    df_combined = pd.concat([df_la, df_nyc], ignore_index=True)

    # Exclude irrelevant columns (only those that actually exist)
    cols_present = [c for c in COLUMNS_TO_EXCLUDE if c in df_combined.columns]
    df_clean = df_combined.drop(columns=cols_present)

    # Remove duplicates
    before = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    after = len(df_clean)

    cleaning_summary = {
        "original_rows": int(len(df_combined)),
        "original_cols": int(len(df_combined.columns)),
        "excluded_cols_requested": int(len(COLUMNS_TO_EXCLUDE)),
        "excluded_cols_found": int(len(cols_present)),
        "duplicates_removed": int(before - after),
        "final_rows": int(len(df_clean)),
        "final_cols": int(len(df_clean.columns)),
        "excluded_columns_list": cols_present,
    }

    return df_clean, cleaning_summary


def _ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="AirBnB preprocessing: LA + NYC merge, drop columns, drop duplicates.")
    parser.add_argument("--la-path", default=DEFAULT_LA_PATH, help="Path to listingsLA.csv")
    parser.add_argument("--nyc-path", default=DEFAULT_NYC_PATH, help="Path to listingsNYC.csv")
    parser.add_argument("--out-csv", default=DEFAULT_OUTPUT_CSV, help="Output path for cleaned combined CSV")
    parser.add_argument("--out-summary", default=DEFAULT_SUMMARY_JSON, help="Output path for cleaning summary JSON")
    args = parser.parse_args()

    df_clean, summary = preprocess(args.la_path, args.nyc_path)

    _ensure_parent_dir(args.out_csv)
    df_clean.to_csv(args.out_csv, index=False)

    _ensure_parent_dir(args.out_summary)
    with open(args.out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("✅ Preprocessing complete")
    print(f"Saved cleaned data to: {args.out_csv}")
    print(f"Saved summary to:      {args.out_summary}")
    print(f"Rows: {summary['original_rows']} → {summary['final_rows']} | Duplicates removed: {summary['duplicates_removed']}")


if __name__ == "__main__":
    main()

