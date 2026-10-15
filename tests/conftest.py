"""Shared pytest fixtures for the MLOps pipeline test suite."""

import pandas as pd
import pytest

from src.data import get_feature_names, load_raw_data, split_data


@pytest.fixture(scope="session")
def raw_dataset() -> pd.DataFrame:
    """Load the Breast Cancer dataset once for the test session."""
    return load_raw_data()


@pytest.fixture(scope="session")
def train_test_splits(raw_dataset: pd.DataFrame):
    """Return reproducible 80/20 train/test split."""
    return split_data(raw_dataset, test_size=0.2, random_state=42)


@pytest.fixture(scope="session")
def sample_features(raw_dataset: pd.DataFrame) -> dict:
    """Return a single valid feature dictionary."""
    features = get_feature_names()
    return raw_dataset[features].iloc[0].to_dict()


@pytest.fixture(scope="session")
def sample_malignant_features(train_test_splits) -> dict:
    """Return a known malignant (class 0) test sample."""
    _, X_test, _, y_test = train_test_splits
    mal_idx = y_test[y_test == 0].index[0]
    return X_test.loc[mal_idx].to_dict()


@pytest.fixture(scope="session")
def sample_benign_features(train_test_splits) -> dict:
    """Return a known benign (class 1) test sample."""
    _, X_test, _, y_test = train_test_splits
    ben_idx = y_test[y_test == 1].index[0]
    return X_test.loc[ben_idx].to_dict()


@pytest.fixture
def isolated_log_dir(tmp_path, monkeypatch):
    """Redirect prediction logging to an isolated temporary directory during testing."""
    test_log_dir = tmp_path / "test_logs"
    test_log_dir.mkdir()
    test_log_file = test_log_dir / "test_prediction_logs.jsonl"

    import app.monitoring as mon

    monkeypatch.setattr(mon, "LOGS_DIR", str(test_log_dir))
    monkeypatch.setattr(mon, "LOG_FILE", str(test_log_file))

    yield test_log_file
