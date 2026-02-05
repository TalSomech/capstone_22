from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from category_encoders import TargetEncoder


def build_preprocessor(
        numeric_cols: list,
        categorical_cols: list,
        target_encode_cols: list = None,
) -> ColumnTransformer:
    """
    Build sklearn preprocessing pipeline with Target Encoding for high-cardinality features.

    Args:
        numeric_cols: List of numeric feature columns
        categorical_cols: List of categorical columns for one-hot encoding
        target_encode_cols: List of columns for target encoding (e.g., neighbourhood)
    """
    target_encode_cols = target_encode_cols or []

    # Separate categorical columns: target-encoded vs one-hot encoded
    onehot_cols = [c for c in categorical_cols if c not in target_encode_cols]

    # --- Transformers ---

    # Numeric: Fill missing values, then scale
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Target Encoding: Best for high-cardinality features like Neighborhoods
    # smoothing=10.0 prevents overfitting on small neighborhoods
    target_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", TargetEncoder(smoothing=10.0)),
    ])

    # One-Hot Encoding: Best for low-cardinality features
    onehot_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    # --- Assemble ---
    transformers = [
        ("num", numeric_transformer, numeric_cols),
    ]

    # Only add target transformer if there are columns to encode
    if target_encode_cols:
        transformers.append(("target", target_transformer, target_encode_cols))

    # Only add onehot transformer if there are columns to encode
    if onehot_cols:
        transformers.append(("cat", onehot_transformer, onehot_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    return preprocessor


def _ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)