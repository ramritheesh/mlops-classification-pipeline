"""Data validation module ensuring dataset integrity before training and inference."""

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.data import get_feature_names


class DataValidationError(Exception):
    """Custom exception raised when data validation fails."""

    pass


def validate_raw_data(df: pd.DataFrame, target_col: str = "target") -> Dict[str, Any]:
    """Validate raw DataFrame against schema, type, range, and target constraints.

    Args:
        df: Raw pandas DataFrame.
        target_col: Name of the target column expected in the dataset.

    Returns:
        Dictionary with validation summary metadata if validation succeeds.

    Raises:
        DataValidationError: If any validation rule is violated.
    """
    if df.empty:
        raise DataValidationError("Input DataFrame is empty.")

    # 1. Target column existence
    if target_col not in df.columns:
        raise DataValidationError(f"Missing required target column '{target_col}'.")

    # 2. Check feature columns existence
    expected_features = get_feature_names()
    missing_features = [f for f in expected_features if f not in df.columns]
    if missing_features:
        raise DataValidationError(
            f"Missing {len(missing_features)} expected feature(s): {missing_features[:5]}"
        )

    # 3. Missing values check
    null_counts = df[expected_features + [target_col]].isnull().sum()
    total_nulls = int(null_counts.sum())
    if total_nulls > 0:
        bad_cols = null_counts[null_counts > 0].to_dict()
        raise DataValidationError(f"Dataset contains {total_nulls} null values: {bad_cols}")

    # 4. Data types check (all features must be numeric)
    for col in expected_features:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise DataValidationError(
                f"Feature column '{col}' is not numeric (type: {df[col].dtype})."
            )

    # 5. Physical range boundary check (cell measurements cannot be negative)
    for col in expected_features:
        if (df[col] < 0).any():
            num_negative = int((df[col] < 0).sum())
            raise DataValidationError(
                f"Feature '{col}' contains {num_negative} negative values. "
                "Measurements must be >= 0."
            )

    # 6. Target validity check
    unique_targets = set(df[target_col].unique())
    if not unique_targets.issubset({0, 1}):
        raise DataValidationError(
            f"Target column contains invalid classes: {unique_targets}. "
            "Expected subset of {0, 1}."
        )

    class_distribution = df[target_col].value_counts().to_dict()

    return {
        "status": "VALID",
        "sample_count": len(df),
        "feature_count": len(expected_features),
        "class_distribution": class_distribution,
    }


def validate_split(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    tolerance: float = 0.05,
) -> Dict[str, Any]:
    """Validate that train and test splits are disjoint and properly stratified.

    Args:
        X_train: Training features DataFrame.
        X_test: Test features DataFrame.
        y_train: Training targets Series.
        y_test: Test targets Series.
        tolerance: Allowed difference between train and test class 1 proportions.

    Returns:
        Validation summary dictionary.

    Raises:
        DataValidationError: If splits leak data, overlap, or violate stratification.
    """
    # 1. Check length alignment
    if len(X_train) != len(y_train):
        raise DataValidationError(
            f"X_train length ({len(X_train)}) does not match y_train length ({len(y_train)})."
        )
    if len(X_test) != len(y_test):
        raise DataValidationError(
            f"X_test length ({len(X_test)}) does not match y_test length ({len(y_test)})."
        )

    # 2. Check index disjointness (no overlap between train and test samples)
    overlap = set(X_train.index).intersection(set(X_test.index))
    if overlap:
        raise DataValidationError(
            f"Data leakage detected! {len(overlap)} sample indices overlap "
            "between train and test sets."
        )

    # 3. Stratification check
    train_ratio = float(y_train.mean())
    test_ratio = float(y_test.mean())
    diff = abs(train_ratio - test_ratio)
    if diff > tolerance:
        raise DataValidationError(
            f"Stratification violation: train positive ratio ({train_ratio:.4f}) differs "
            f"from test positive ratio ({test_ratio:.4f}) by {diff:.4f} > tolerance ({tolerance})."
        )

    return {
        "status": "VALID",
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "train_positive_ratio": round(train_ratio, 4),
        "test_positive_ratio": round(test_ratio, 4),
        "index_overlap_count": len(overlap),
    }


def validate_inference_features(
    features: Dict[str, float], expected_features: List[str] = None
) -> None:
    """Validate a single inference feature dictionary against expected schema and bounds."""
    if expected_features is None:
        expected_features = get_feature_names()

    missing = [f for f in expected_features if f not in features]
    if missing:
        raise DataValidationError(f"Missing inference features: {missing[:5]}")

    for k, v in features.items():
        if k in expected_features:
            if not isinstance(v, (int, float, np.floating, np.integer)):
                raise DataValidationError(f"Feature '{k}' value '{v}' is not numeric.")
            if np.isnan(v) or np.isinf(v):
                raise DataValidationError(f"Feature '{k}' contains NaN or Inf.")
            if v < 0:
                raise DataValidationError(f"Feature '{k}' must be non-negative, got {v}.")
