"""Data loading and train/test splitting module."""

from typing import List, Tuple

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

TARGET_NAMES = ["malignant", "benign"]


def standardize_column_name(col: str) -> str:
    """Convert column names to clean lowercase snake_case format."""
    return col.strip().lower().replace(" ", "_").replace("(", "").replace(")", "")


def load_raw_data() -> pd.DataFrame:
    """Load the Breast Cancer Wisconsin dataset as a pandas DataFrame.

    Standardizes all feature column names to snake_case.
    Returns:
        pd.DataFrame containing 30 numeric feature columns and 1 'target' column.
    """
    dataset = load_breast_cancer(as_frame=True)
    df = dataset.frame.copy()
    df.columns = [standardize_column_name(c) for c in df.columns]
    return df


def get_feature_names() -> List[str]:
    """Return the ordered list of 30 standardized feature names."""
    dataset = load_breast_cancer(as_frame=True)
    return [standardize_column_name(f) for f in dataset.feature_names]


def get_target_names() -> List[str]:
    """Return target class names where 0=malignant, 1=benign."""
    return list(TARGET_NAMES)


def split_data(
    df: pd.DataFrame,
    target_col: str = "target",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Perform a reproducible, stratified train/test split.

    Args:
        df: Input DataFrame containing features and target column.
        target_col: Name of the target column.
        test_size: Proportion of dataset allocated to held-out test set (default 0.2).
        random_state: Random seed for reproducible splitting.

    Returns:
        (X_train, X_test, y_train, y_test)
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test
