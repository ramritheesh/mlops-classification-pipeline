#!/usr/bin/env python3
"""Verification script testing that the saved production model reloads and predicts accurately."""

import sys
from pathlib import Path

# Ensure project root is in sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import get_feature_names, load_raw_data
from src.predict import load_metadata, load_model, predict_batch, predict_sample


def main() -> None:
    print("=" * 70)
    print("VERIFYING EXPORTED PRODUCTION MODEL ARTIFACT")
    print("=" * 70)

    try:
        # 1. Load model artifact
        print("1. Loading 'models/model.joblib'...")
        model = load_model("models/model.joblib")
        print(f"   Model successfully loaded! Steps in pipeline: {[s[0] for s in model.steps]}")

        # 2. Load and inspect metadata
        print("2. Loading 'models/metadata.json'...")
        metadata = load_metadata("models/metadata.json")
        print(f"   Registered Name:    {metadata.get('model_name')}")
        print(f"   Model Type:         {metadata.get('model_type')}")
        print(f"   Registered Version: {metadata.get('registered_version')}")
        print(f"   MLflow Run ID:      {metadata.get('mlflow_run_id')}")
        print(f"   Test Accuracy:      {metadata.get('test_metrics', {}).get('accuracy')}")
        print(f"   Test F1 Macro:      {metadata.get('test_metrics', {}).get('f1_macro')}")

        # 3. Test single sample inference
        print("3. Testing single sample inference...")
        df = load_raw_data()
        feature_cols = get_feature_names()
        sample_dict = df[feature_cols].iloc[0].to_dict()
        true_label = int(df["target"].iloc[0])

        single_result = predict_sample(model, sample_dict)
        print(f"   True Label:      {true_label} ({metadata['target_classes'][true_label]})")
        print(f"   Prediction:      {single_result['prediction']} ({single_result['class_name']})")
        print(f"   Confidence:      {single_result.get('probability')}")
        print(f"   Probabilities:   {single_result.get('probabilities')}")

        assert "prediction" in single_result
        assert single_result["prediction"] in (0, 1)
        assert 0.0 <= single_result["probability"] <= 1.0

        # 4. Test batch inference
        print("4. Testing batch inference (5 samples)...")
        batch_samples = [df[feature_cols].iloc[i].to_dict() for i in range(5)]
        batch_results = predict_batch(model, batch_samples)
        print(f"   Batch returned {len(batch_results)} predictions.")
        for idx, res in enumerate(batch_results):
            pred_val = res["prediction"]
            c_name = res["class_name"]
            prob_val = res["probability"]
            print(f"     Sample {idx + 1}: Pred={pred_val} ({c_name}) | Prob={prob_val}")

        assert len(batch_results) == 5
        print("\n" + "=" * 70)
        print("ALL MODEL ARTIFACT VERIFICATION CHECKS PASSED!")
        print("=" * 70)

    except Exception as e:
        print(f"\nModel verification FAILED: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
