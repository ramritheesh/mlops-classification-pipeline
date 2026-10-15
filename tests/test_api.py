"""Integration tests for FastAPI inference endpoints, monitoring, and error handling."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.model_loader import ModelLoader
from app.monitoring import compute_monitoring_summary, read_prediction_logs


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Fixture providing TestClient with startup lifespan initialized."""
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client: TestClient) -> None:
    """Test GET / returns service information and docs URL."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service_name"] == "Breast Cancer Classification MLOps API"
    assert data["status"] == "running"
    assert "/docs" in data["docs_url"]
    assert "DEMONSTRATION ONLY" in data["disclaimer"]


def test_health_endpoint(client: TestClient) -> None:
    """Test GET /health returns healthy status and active model metadata."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["model_name"] == "breast-cancer-champion"
    assert "timestamp" in data


def test_model_info_endpoint(client: TestClient) -> None:
    """Test GET /model-info returns governance metadata, metrics, and features."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "breast-cancer-champion"
    assert len(data["features"]) == 30
    assert "test_metrics" in data
    assert data["test_metrics"]["accuracy"] > 0.90
    assert "DEMONSTRATION ONLY" in data["disclaimer"]


def test_predict_single_malignant(client: TestClient, sample_malignant_features: dict) -> None:
    """Test POST /predict with a known malignant sample."""
    response = client.post("/predict", json=sample_malignant_features)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == 0
    assert data["class_name"] == "malignant"
    assert 0.0 <= data["probability"] <= 1.0
    probs = data["probabilities"]
    assert abs(probs["malignant"] + probs["benign"] - 1.0) < 1e-4
    assert "DEMONSTRATION ONLY" in data["disclaimer"]


def test_predict_single_benign(client: TestClient, sample_benign_features: dict) -> None:
    """Test POST /predict with a known benign sample."""
    response = client.post("/predict", json=sample_benign_features)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == 1
    assert data["class_name"] == "benign"
    assert 0.0 <= data["probability"] <= 1.0
    probs = data["probabilities"]
    assert abs(probs["malignant"] + probs["benign"] - 1.0) < 1e-4


def test_predict_missing_field_fails(client: TestClient, sample_features: dict) -> None:
    """Test POST /predict returns 422 Unprocessable Entity when required field is missing."""
    corrupted = sample_features.copy()
    del corrupted["mean_radius"]
    response = client.post("/predict", json=corrupted)
    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "Validation Error"
    assert any("mean_radius" in d for d in data["details"])


def test_predict_negative_value_fails(client: TestClient, sample_features: dict) -> None:
    """Test POST /predict returns 422 when physical measurement is negative."""
    corrupted = sample_features.copy()
    corrupted["mean_radius"] = -10.0
    response = client.post("/predict", json=corrupted)
    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "Validation Error"
    assert any("greater than or equal to 0" in d for d in data["details"])


def test_predict_unknown_extra_field_fails(client: TestClient, sample_features: dict) -> None:
    """Test POST /predict forbids unexpected extra fields (e.g., patient identifiers)."""
    corrupted = sample_features.copy()
    corrupted["patient_ssn"] = "000-11-2222"
    response = client.post("/predict", json=corrupted)
    assert response.status_code == 422
    data = response.json()
    assert any("Extra inputs are not permitted" in d for d in data["details"])


def test_predict_batch_both_classes(
    client: TestClient, sample_malignant_features: dict, sample_benign_features: dict
) -> None:
    """Test POST /predict/batch containing both classes."""
    payload = {"samples": [sample_malignant_features, sample_benign_features]}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["batch_size"] == 2
    preds = data["predictions"]
    assert preds[0]["prediction"] == 0
    assert preds[0]["class_name"] == "malignant"
    assert preds[1]["prediction"] == 1
    assert preds[1]["class_name"] == "benign"


def test_predict_batch_empty_fails(client: TestClient) -> None:
    """Test POST /predict/batch returns 422 if sample array is empty."""
    response = client.post("/predict/batch", json={"samples": []})
    assert response.status_code == 422


def test_predict_batch_exceeds_max_limit_fails(client: TestClient, sample_features: dict) -> None:
    """Test POST /predict/batch returns 422 if sample array exceeds 100 items."""
    oversized = {"samples": [sample_features] * 101}
    response = client.post("/predict/batch", json=oversized)
    assert response.status_code == 422
    data = response.json()
    assert any("at most 100 items" in d for d in data["details"])


def test_monitoring_records_both_classes_accurately(
    client: TestClient,
    sample_malignant_features: dict,
    sample_benign_features: dict,
    isolated_log_dir,
) -> None:
    """CORRECTION B TEST: Verify monitoring logs record both classes accurately."""
    # Send 2 malignant and 1 benign requests
    client.post("/predict", json=sample_malignant_features)
    client.post("/predict", json=sample_malignant_features)
    client.post("/predict", json=sample_benign_features)

    summary = compute_monitoring_summary(str(isolated_log_dir))
    dist = summary["prediction_distribution"]

    # Verify both classes exist in observed distribution without hardcoding
    assert "malignant" in dist
    assert "benign" in dist
    assert dist["malignant"] >= 2
    assert dist["benign"] >= 1
    assert "latency_note" in summary
    assert "local single-process test observations" in summary["latency_note"]

    # Privacy verification: assert no raw PII in log records
    records = read_prediction_logs(str(isolated_log_dir))
    for r in records:
        assert "patient_name" not in r
        assert "patient_ssn" not in r
        # Feature vector is summarized statistically
        assert "feature_summary" in r
        assert "mean" in r["feature_summary"]


def test_unloaded_model_error_handling() -> None:
    """Test 503 Service Unavailable when model is not loaded."""
    with patch.object(ModelLoader, "is_loaded", property(lambda self: False)):
        with TestClient(app) as test_client:
            res_health = test_client.get("/health")
            assert res_health.status_code == 503
            assert "Production model is not loaded" in res_health.json()["detail"]

            res_pred = test_client.post(
                "/predict",
                json={
                    "mean_radius": 10.0,
                    "mean_texture": 10.0,
                    "mean_perimeter": 10.0,
                    "mean_area": 10.0,
                    "mean_smoothness": 0.1,
                    "mean_compactness": 0.1,
                    "mean_concavity": 0.1,
                    "mean_concave_points": 0.1,
                    "mean_symmetry": 0.1,
                    "mean_fractal_dimension": 0.1,
                    "radius_error": 0.1,
                    "texture_error": 0.1,
                    "perimeter_error": 0.1,
                    "area_error": 10.0,
                    "smoothness_error": 0.01,
                    "compactness_error": 0.01,
                    "concavity_error": 0.01,
                    "concave_points_error": 0.01,
                    "symmetry_error": 0.01,
                    "fractal_dimension_error": 0.01,
                    "worst_radius": 10.0,
                    "worst_texture": 10.0,
                    "worst_perimeter": 10.0,
                    "worst_area": 10.0,
                    "worst_smoothness": 0.1,
                    "worst_compactness": 0.1,
                    "worst_concavity": 0.1,
                    "worst_concave_points": 0.1,
                    "worst_symmetry": 0.1,
                    "worst_fractal_dimension": 0.1,
                },
            )
            assert res_pred.status_code == 503
            assert "Model is not loaded" in res_pred.json()["detail"]
