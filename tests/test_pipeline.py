"""Tests for ML pipelines, cross-validation, champion selection, and diagnostic metrics."""

import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.evaluate import compute_metrics, plot_and_save_confusion_matrix
from src.predict import load_model, predict_batch, predict_sample
from src.preprocessing import build_pipeline
from src.train import (
    run_cross_validation,
    select_champion_model,
)


def test_build_pipeline_structure() -> None:
    """Test pipeline encapsulation of StandardScaler and classifier."""
    clf = LogisticRegression()
    pipeline = build_pipeline(clf)
    assert len(pipeline.steps) == 2
    assert pipeline.steps[0][0] == "scaler"
    assert isinstance(pipeline.steps[0][1], StandardScaler)
    assert pipeline.steps[1][0] == "classifier"


def test_pipeline_prevents_data_leakage(train_test_splits) -> None:
    """Test that scaler inside pipeline learns statistics strictly from X_train."""
    X_train, X_test, y_train, _ = train_test_splits
    pipeline = build_pipeline(LogisticRegression(max_iter=500))
    pipeline.fit(X_train, y_train)

    scaler = pipeline.named_steps["scaler"]
    # Scaler mean must match X_train mean, completely unaffected by X_test
    np.testing.assert_allclose(scaler.mean_, X_train.mean(axis=0).values, rtol=1e-5)
    np.testing.assert_allclose(scaler.var_, X_train.var(axis=0, ddof=0).values, rtol=1e-5)


def test_cross_validation_metrics_bounds(train_test_splits) -> None:
    """Test cross-validation calculates bounded metrics for Malignant positive class."""
    X_train, _, y_train, _ = train_test_splits
    # Run on subset for fast unit testing
    sub_X = X_train.iloc[:100]
    sub_y = y_train.iloc[:100]

    pipeline = build_pipeline(LogisticRegression(max_iter=500, random_state=42))
    cv_metrics = run_cross_validation(pipeline, sub_X, sub_y, n_splits=3, random_state=42)

    required_keys = [
        "cv_mean_accuracy",
        "cv_mean_f1_malignant",
        "cv_mean_recall_malignant",
        "cv_mean_precision_malignant",
    ]
    for key in required_keys:
        assert key in cv_metrics
        assert 0.0 <= cv_metrics[key] <= 1.0


def test_champion_selection_rule() -> None:
    """Test champion selection prioritizes Malignant F1-score on training CV."""
    candidates = {
        "model_a": {
            "cv_metrics": {
                "cv_mean_f1_malignant": 0.95,
                "cv_mean_recall_malignant": 0.94,
            }
        },
        "model_b": {
            "cv_metrics": {
                "cv_mean_f1_malignant": 0.98,
                "cv_mean_recall_malignant": 0.97,
            }
        },
    }
    champ_name, champ_cand = select_champion_model(candidates)
    assert champ_name == "model_b"
    assert champ_cand["cv_metrics"]["cv_mean_f1_malignant"] == 0.98


def test_confusion_matrix_diagnostic_terminology_regression() -> None:
    """REGRESSION TEST: Verify diagnostic mapping where Malignant is the Positive class (0).

    Target scenario:
    - 42 True Malignants: 41 predicted Malignant, 1 predicted Benign
    - 72 True Benigns: 1 predicted Malignant, 71 predicted Benign
    Must yield:
    - TP = 41 (True Malignant predicted Malignant)
    - FN = 1  (True Malignant predicted Benign)
    - FP = 1  (True Benign predicted Malignant)
    - TN = 71 (True Benign predicted Benign)
    - Malignant Recall = TP / (TP + FN) = 41 / 42 = 0.9762
    - Malignant Precision = TP / (TP + FP) = 41 / 42 = 0.9762
    - Malignant F1 = 0.9762
    """
    # Construct exact scenario
    y_true = np.array([0] * 42 + [1] * 72)
    # Predictions: 41 correct 0s, 1 flipped to 1; 71 correct 1s, 1 flipped to 0
    y_pred = np.array([0] * 41 + [1] * 1 + [0] * 1 + [1] * 71)

    metrics = compute_metrics(y_true, y_pred)

    assert metrics["true_positives_malignant"] == 41
    assert metrics["false_negatives_malignant"] == 1
    assert metrics["false_positives_benign"] == 1
    assert metrics["true_negatives_benign"] == 71

    assert metrics["recall_malignant"] == 0.9762
    assert metrics["precision_malignant"] == 0.9762
    assert metrics["f1_malignant"] == 0.9762
    assert metrics["accuracy"] == round(112.0 / 114.0, 4)


def test_plot_confusion_matrix_artifact(tmp_path) -> None:
    """Test confusion matrix visualization generates a readable image file."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    out_file = tmp_path / "test_cm.png"

    saved_path = plot_and_save_confusion_matrix(y_true, y_pred, str(out_file), "Test CM")
    assert os.path.exists(saved_path)
    assert os.path.getsize(saved_path) > 1000  # Non-trivial image file


def test_model_export_and_reload_determinism(train_test_splits, tmp_path) -> None:
    """Test model serialization to joblib reloads and produces identical predictions."""
    X_train, X_test, y_train, _ = train_test_splits
    pipeline = build_pipeline(LogisticRegression(max_iter=500, random_state=42))
    pipeline.fit(X_train, y_train)

    export_path = tmp_path / "model.joblib"
    import joblib

    joblib.dump(pipeline, export_path)

    reloaded = load_model(str(export_path))
    orig_preds = pipeline.predict(X_test)
    reloaded_preds = reloaded.predict(X_test)
    np.testing.assert_array_equal(orig_preds, reloaded_preds)

    orig_probs = pipeline.predict_proba(X_test)
    reloaded_probs = reloaded.predict_proba(X_test)
    np.testing.assert_allclose(orig_probs, reloaded_probs, rtol=1e-5)


def test_predict_sample_and_batch_utilities(train_test_splits, sample_features) -> None:
    """Test predict_sample and predict_batch functions from src.predict."""
    X_train, _, y_train, _ = train_test_splits
    pipeline = build_pipeline(LogisticRegression(max_iter=500, random_state=42))
    pipeline.fit(X_train, y_train)

    single_res = predict_sample(pipeline, sample_features)
    assert single_res["prediction"] in (0, 1)
    assert single_res["class_name"] in ("malignant", "benign")
    assert 0.0 <= single_res["probability"] <= 1.0
    probs = single_res["probabilities"]
    assert abs(probs["malignant"] + probs["benign"] - 1.0) < 1e-4

    batch_res = predict_batch(pipeline, [sample_features, sample_features])
    assert len(batch_res) == 2
    assert batch_res[0]["prediction"] == single_res["prediction"]
