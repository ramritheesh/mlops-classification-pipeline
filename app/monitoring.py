"""Privacy-conscious prediction logging and local monitoring module."""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger("uvicorn.error")
LOGS_DIR = "logs"
LOG_FILE = os.path.join(LOGS_DIR, "prediction_logs.jsonl")


def init_monitoring(log_dir: str = LOGS_DIR) -> None:
    """Ensure the monitoring log directory exists."""
    os.makedirs(log_dir, exist_ok=True)


def log_prediction(
    request_id: str,
    model_version: str,
    feature_dict: Dict[str, float],
    prediction: int,
    class_name: str,
    probability: float,
    latency_ms: float,
) -> None:
    """Append a structured, privacy-preserving prediction record to JSONL logs.

    Privacy Design:
    - Does NOT store raw customer identifiers or unbounded patient metadata.
    - Captures high-level statistical summary of features (mean, std, min, max)
      rather than identifying clinical data, enabling drift detection while protecting privacy.
    """
    init_monitoring()

    feature_values = list(feature_dict.values())
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "model_version": str(model_version),
        "status": "success",
        "feature_summary": {
            "mean": round(float(np.mean(feature_values)), 4),
            "std": round(float(np.std(feature_values)), 4),
            "min": round(float(np.min(feature_values)), 4),
            "max": round(float(np.max(feature_values)), 4),
            "feature_count": len(feature_values),
        },
        "prediction": prediction,
        "class_name": class_name,
        "probability": round(probability, 4),
        "latency_ms": round(latency_ms, 2),
    }

    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        logger.warning(f"Failed to write prediction log: {e}")


def log_error(
    request_id: str,
    error_type: str,
    error_detail: str,
    latency_ms: float = 0.0,
) -> None:
    """Log validation or processing errors for service monitoring."""
    init_monitoring()
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "status": "error",
        "error_type": error_type,
        "error_detail": error_detail,
        "latency_ms": round(latency_ms, 2),
    }
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        logger.warning(f"Failed to write error log: {e}")


def read_prediction_logs(log_path: str = LOG_FILE) -> List[Dict[str, Any]]:
    """Read all entries from the local prediction logs."""
    if not os.path.exists(log_path):
        return []
    records = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def compute_monitoring_summary(log_path: str = LOG_FILE) -> Dict[str, Any]:
    """Compute aggregate monitoring metrics from logged requests.

    Note on MLOps Concept vs. Service Monitoring:
    - Service Monitoring: Latency, throughput, error rates, and request count.
    - ML Model Monitoring: Prediction distribution drift and feature drift.
    - Ground-Truth Limitation: True model accuracy and concept drift CANNOT be
      computed in real time without delayed or verified ground-truth labels.
    """
    records = read_prediction_logs(log_path)
    if not records:
        return {
            "total_logged_events": 0,
            "successful_predictions": 0,
            "error_events": 0,
            "prediction_distribution": {},
            "average_latency_ms": 0.0,
            "monitoring_note": "No logged prediction events found yet.",
        }

    success_records = [r for r in records if r.get("status") == "success"]
    error_records = [r for r in records if r.get("status") == "error"]

    distribution: Dict[str, int] = {}
    latencies: List[float] = []

    for r in success_records:
        c_name = r.get("class_name", "unknown")
        distribution[c_name] = distribution.get(c_name, 0) + 1
        latencies.append(r.get("latency_ms", 0.0))

    avg_latency = float(np.mean(latencies)) if latencies else 0.0

    return {
        "total_logged_events": len(records),
        "successful_predictions": len(success_records),
        "error_events": len(error_records),
        "prediction_distribution": distribution,
        "average_latency_ms": round(avg_latency, 2),
        "latency_note": (
            "Latency measurements are local single-process test observations, "
            "not production SLAs or production benchmarks."
        ),
        "monitoring_note": (
            "Service metrics and prediction distribution computed from local logs. "
            "True accuracy and concept drift require ground-truth outcomes."
        ),
    }
