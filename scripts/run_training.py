#!/usr/bin/env python3
"""CLI runner script for training, cross-validation, MLflow logging, and champion export."""

import sys
from pathlib import Path

# Ensure project root is in sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train import train_and_evaluate_pipeline


def main() -> None:
    try:
        summary = train_and_evaluate_pipeline()
        print("\nPipeline execution succeeded!")
        print(f"Champion Model: {summary.get('model_name')}")
        print(f"Algorithm:      {summary.get('model_type')}")
        print(f"Registered Ver: {summary.get('registered_version')}")
        print(f"Test Accuracy:  {summary.get('test_metrics', {}).get('accuracy')}")
        print(f"Test F1:        {summary.get('test_metrics', {}).get('f1_macro')}")
        print(f"Test Recall:    {summary.get('test_metrics', {}).get('recall_malignant')}")
    except Exception as e:
        print(f"\nTraining pipeline failed with error: {e}", file=sys.stderr)
        raise e


if __name__ == "__main__":
    main()
