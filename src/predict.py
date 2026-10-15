"""Inference utility module for loading models and executing predictions."""

import json
import os
from typing import Any, Dict, List

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.data import TARGET_NAMES, get_feature_names


def load_model(model_path: str = "models/model.joblib") -> Pipeline:
    """Load the trained production Pipeline from disk.

    Args:
        model_path: Path to serialized joblib pipeline.

    Returns:
        Fitted scikit-learn Pipeline.

    Raises:
        FileNotFoundError: If model artifact does not exist.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Production model artifact not found at '{model_path}'. "
            "Run training workflow first via 'python scripts/run_training.py'."
        )
    return joblib.load(model_path)


def load_metadata(metadata_path: str = "models/metadata.json") -> Dict[str, Any]:
    """Load production model metadata JSON.

    Args:
        metadata_path: Path to metadata.json.

    Returns:
        Metadata dictionary.
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Model metadata not found at '{metadata_path}'. "
            "Run training workflow first via 'python scripts/run_training.py'."
        )
    with open(metadata_path, "r") as f:
        return json.load(f)


def predict_sample(model: Pipeline, features: Dict[str, float]) -> Dict[str, Any]:
    """Execute prediction on a single feature dictionary.

    Args:
        model: Fitted scikit-learn Pipeline.
        features: Dictionary mapping 30 standardized feature names to float values.

    Returns:
        Dictionary with prediction integer, class_name, and class probabilities.
    """
    expected_order = get_feature_names()
    ordered_values = {f: [features[f]] for f in expected_order}
    df = pd.DataFrame(ordered_values)

    pred = int(model.predict(df)[0])
    class_name = TARGET_NAMES[pred]

    result = {
        "prediction": pred,
        "class_name": class_name,
    }

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(df)[0]
        result["probability"] = round(float(probs[pred]), 4)
        result["probabilities"] = {
            TARGET_NAMES[0]: round(float(probs[0]), 4),
            TARGET_NAMES[1]: round(float(probs[1]), 4),
        }

    return result


def predict_batch(model: Pipeline, samples: List[Dict[str, float]]) -> List[Dict[str, Any]]:
    """Execute prediction on a batch of feature dictionaries.

    Args:
        model: Fitted scikit-learn Pipeline.
        samples: List of dictionaries each containing the 30 feature values.

    Returns:
        List of prediction result dictionaries.
    """
    if not samples:
        return []

    expected_order = get_feature_names()
    rows = []
    for s in samples:
        rows.append({f: s[f] for f in expected_order})
    df = pd.DataFrame(rows)

    preds = model.predict(df)
    probs = model.predict_proba(df) if hasattr(model, "predict_proba") else None

    results = []
    for i, pred_int in enumerate(preds):
        p = int(pred_int)
        item = {
            "prediction": p,
            "class_name": TARGET_NAMES[p],
        }
        if probs is not None:
            item["probability"] = round(float(probs[i][p]), 4)
            item["probabilities"] = {
                TARGET_NAMES[0]: round(float(probs[i][0]), 4),
                TARGET_NAMES[1]: round(float(probs[i][1]), 4),
            }
        results.append(item)

    return results
