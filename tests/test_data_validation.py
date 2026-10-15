"""Tests for data loading, schema integrity, and validation constraints."""

import numpy as np
import pandas as pd
import pytest

from src.data import get_feature_names, get_target_names, split_data
from src.validation import (
    DataValidationError,
    validate_inference_features,
    validate_raw_data,
    validate_split,
)


def test_load_raw_data_shape_and_columns(raw_dataset: pd.DataFrame) -> None:
    """Test raw dataset dimensions and column naming conventions."""
    assert raw_dataset.shape == (569, 31)
    assert "target" in raw_dataset.columns
    # All columns must be lowercase snake_case
    for col in raw_dataset.columns:
        assert " " not in col
        assert col.islower()


def test_get_feature_names_count_and_format() -> None:
    """Test feature names list properties."""
    features = get_feature_names()
    assert len(features) == 30
    assert len(set(features)) == 30
    assert "mean_radius" in features
    assert "worst_fractal_dimension" in features


def test_get_target_names() -> None:
    """Test target class ordering: 0=malignant, 1=benign."""
    targets = get_target_names()
    assert targets == ["malignant", "benign"]


def test_validate_raw_data_success(raw_dataset: pd.DataFrame) -> None:
    """Test validation succeeds on pristine dataset."""
    result = validate_raw_data(raw_dataset)
    assert result["status"] == "VALID"
    assert result["sample_count"] == 569
    assert result["feature_count"] == 30
    assert result["class_distribution"] == {1: 357, 0: 212}


def test_validate_raw_data_empty() -> None:
    """Test validation raises on empty DataFrame."""
    empty_df = pd.DataFrame()
    with pytest.raises(DataValidationError, match="Input DataFrame is empty"):
        validate_raw_data(empty_df)


def test_validate_raw_data_missing_target(raw_dataset: pd.DataFrame) -> None:
    """Test validation raises when target column is missing."""
    corrupted = raw_dataset.drop(columns=["target"])
    with pytest.raises(DataValidationError, match="Missing required target column"):
        validate_raw_data(corrupted)


def test_validate_raw_data_missing_feature(raw_dataset: pd.DataFrame) -> None:
    """Test validation raises when an expected feature column is missing."""
    corrupted = raw_dataset.drop(columns=["mean_radius"])
    with pytest.raises(DataValidationError, match="Missing 1 expected feature"):
        validate_raw_data(corrupted)


def test_validate_raw_data_null_values(raw_dataset: pd.DataFrame) -> None:
    """Test validation raises when NaN values are injected."""
    corrupted = raw_dataset.copy()
    corrupted.loc[0, "mean_texture"] = np.nan
    with pytest.raises(DataValidationError, match="contains 1 null values"):
        validate_raw_data(corrupted)


def test_validate_raw_data_non_numeric(raw_dataset: pd.DataFrame) -> None:
    """Test validation raises when string data is injected into numeric features."""
    corrupted = raw_dataset.copy()
    corrupted["mean_radius"] = corrupted["mean_radius"].astype(str)
    with pytest.raises(DataValidationError, match="is not numeric"):
        validate_raw_data(corrupted)


def test_validate_raw_data_negative_measurements(raw_dataset: pd.DataFrame) -> None:
    """Test validation raises on negative physical measurements."""
    corrupted = raw_dataset.copy()
    corrupted.loc[5, "mean_area"] = -10.0
    with pytest.raises(DataValidationError, match="contains 1 negative values"):
        validate_raw_data(corrupted)


def test_validate_raw_data_invalid_target_classes(raw_dataset: pd.DataFrame) -> None:
    """Test validation raises when target column has invalid classes."""
    corrupted = raw_dataset.copy()
    corrupted.loc[0, "target"] = 2
    with pytest.raises(DataValidationError, match="Target column contains invalid classes"):
        validate_raw_data(corrupted)


def test_split_data_reproducibility(raw_dataset: pd.DataFrame) -> None:
    """Test split is 100% deterministic with fixed random_state."""
    X1, X2, y1, y2 = split_data(raw_dataset, random_state=42)
    X3, X4, y3, y4 = split_data(raw_dataset, random_state=42)

    pd.testing.assert_frame_equal(X1, X3)
    pd.testing.assert_frame_equal(X2, X4)
    pd.testing.assert_series_equal(y1, y3)
    pd.testing.assert_series_equal(y2, y4)


def test_split_data_no_index_overlap(train_test_splits) -> None:
    """Test zero sample overlap between train and test sets (prevent data leakage)."""
    X_train, X_test, y_train, y_test = train_test_splits
    overlap = set(X_train.index).intersection(set(X_test.index))
    assert len(overlap) == 0


def test_split_data_stratification_preservation(train_test_splits) -> None:
    """Test that train and test partitions preserve target class proportions."""
    _, _, y_train, y_test = train_test_splits
    train_ratio = y_train.mean()
    test_ratio = y_test.mean()
    # Benign positive ratio must be within 1% between splits
    assert abs(train_ratio - test_ratio) < 0.01


def test_validate_split_leakage_detection(train_test_splits) -> None:
    """Test that validate_split catches intentional index overlap between splits."""
    X_train, X_test, y_train, y_test = train_test_splits
    # Intentionally leak one index from train to test
    corrupted_X_test = pd.concat([X_test, X_train.iloc[:1]])
    corrupted_y_test = pd.concat([y_test, y_train.iloc[:1]])

    with pytest.raises(DataValidationError, match="Data leakage detected"):
        validate_split(X_train, corrupted_X_test, y_train, corrupted_y_test)


def test_validate_inference_features_valid(sample_features: dict) -> None:
    """Test validate_inference_features passes on a valid feature dict."""
    # Should not raise
    validate_inference_features(sample_features)


def test_validate_inference_features_negative(sample_features: dict) -> None:
    """Test validate_inference_features catches negative values."""
    corrupted = sample_features.copy()
    corrupted["worst_area"] = -1.0
    with pytest.raises(DataValidationError, match="must be non-negative"):
        validate_inference_features(corrupted)
