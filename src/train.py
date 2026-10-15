"""Model training, cross-validation, selection, evaluation, and MLflow logging pipeline."""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Tuple

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.data import get_feature_names, get_target_names, load_raw_data, split_data
from src.evaluate import compute_metrics, plot_and_save_confusion_matrix
from src.preprocessing import build_pipeline
from src.validation import validate_raw_data, validate_split

EXPERIMENT_NAME = "breast-cancer-classification"
CHAMPION_REGISTRY_NAME = "breast-cancer-champion"
TRACKING_URI = "sqlite:///mlruns.db"
RANDOM_STATE = 42


def configure_mlflow(tracking_uri: str = TRACKING_URI) -> None:
    """Set the MLflow tracking URI and create/set active experiment."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)


def run_cross_validation(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
) -> Dict[str, float]:
    """Execute stratified k-fold cross-validation on the training set.

    Treats Malignant (class 0) as the primary positive class.

    Args:
        pipeline: Preprocessing + Estimator scikit-learn Pipeline.
        X_train: Training features.
        y_train: Training targets.
        n_splits: Number of cross-validation folds.
        random_state: Random state for KFold shuffle.

    Returns:
        Dictionary of mean and std for each CV metric.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "f1_malignant": make_scorer(f1_score, pos_label=0),
        "recall_malignant": make_scorer(recall_score, pos_label=0),
        "precision_malignant": make_scorer(precision_score, pos_label=0, zero_division=0),
        "roc_auc": "roc_auc",
    }

    cv_results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=skf,
        scoring=scoring,
        return_train_score=False,
    )

    metrics = {}
    for metric_name in scoring:
        scores = cv_results[f"test_{metric_name}"]
        metrics[f"cv_mean_{metric_name}"] = round(float(np.mean(scores)), 4)
        metrics[f"cv_std_{metric_name}"] = round(float(np.std(scores)), 4)

    return metrics


def select_champion_model(
    model_candidates: Dict[str, Dict[str, Any]],
) -> Tuple[str, Dict[str, Any]]:
    """Select champion model using Stratified CV F1-score on the training set.

    Documented Selection Rule:
    1. Primary Metric: Highest mean CV F1-score on Malignant class (cv_mean_f1_malignant).
    2. Secondary Metric (Tie-breaker if |F1_diff| <= 0.005): Highest mean CV Malignant Recall.
    3. Simplicity Tie-breaker: Prefer Logistic Regression for lower latency.

    Args:
        model_candidates: Dict mapping model_name -> candidate dict with pipeline, params.

    Returns:
        (champion_name, champion_info_dict)
    """
    sorted_candidates = sorted(
        model_candidates.items(),
        key=lambda item: (
            item[1]["cv_metrics"]["cv_mean_f1_malignant"],
            item[1]["cv_metrics"]["cv_mean_recall_malignant"],
        ),
        reverse=True,
    )

    champion_name, champion_info = sorted_candidates[0]
    return champion_name, champion_info


def train_and_evaluate_pipeline(
    tracking_uri: str = TRACKING_URI,
    models_dir: str = "models",
    reports_dir: str = "reports",
) -> Dict[str, Any]:
    """Execute complete end-to-end ML workflow."""
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    print("=" * 70)
    print("STEP 1: Loading & Validating Data")
    print("=" * 70)
    df = load_raw_data()
    raw_val = validate_raw_data(df)
    print(
        f"Dataset loaded: {raw_val['sample_count']} samples, "
        f"{raw_val['feature_count']} features."
    )
    print(f"Target distribution: {raw_val['class_distribution']}")

    print("\n" + "=" * 70)
    print("STEP 2: Stratified Train/Test Split (80/20, random_state=42)")
    print("=" * 70)
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, random_state=RANDOM_STATE)
    split_val = validate_split(X_train, X_test, y_train, y_test)
    print(
        f"Train samples: {split_val['train_samples']} "
        f"(Positive/Benign ratio: {split_val['train_positive_ratio']})"
    )
    print(
        f"Test samples:  {split_val['test_samples']} "
        f"(Positive/Benign ratio: {split_val['test_positive_ratio']})"
    )
    print(f"Index overlap: {split_val['index_overlap_count']}")

    print("\n" + "=" * 70)
    print("STEP 3: Defining Candidate Models & Leakage-Safe Pipelines")
    print("=" * 70)
    candidates = {
        "logistic_regression": {
            "model_type": "LogisticRegression",
            "pipeline": build_pipeline(
                LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, C=1.0)
            ),
            "params": {
                "algorithm": "LogisticRegression",
                "scaler": "StandardScaler",
                "C": 1.0,
                "penalty": "l2",
                "max_iter": 1000,
                "cv_folds": 5,
                "random_state": RANDOM_STATE,
            },
        },
        "random_forest": {
            "model_type": "RandomForestClassifier",
            "pipeline": build_pipeline(
                RandomForestClassifier(n_estimators=100, max_depth=5, random_state=RANDOM_STATE)
            ),
            "params": {
                "algorithm": "RandomForestClassifier",
                "scaler": "StandardScaler",
                "n_estimators": 100,
                "max_depth": 5,
                "cv_folds": 5,
                "random_state": RANDOM_STATE,
            },
        },
    }

    print("\n" + "=" * 70)
    print("STEP 4: Cross-Validation on Training Data (Positive = Malignant)")
    print("=" * 70)
    for name, cand in candidates.items():
        print(f"Evaluating {name} with 5-fold Stratified CV...")
        cv_metrics = run_cross_validation(
            cand["pipeline"], X_train, y_train, n_splits=5, random_state=RANDOM_STATE
        )
        cand["cv_metrics"] = cv_metrics
        print(
            f"  -> Mean CV F1 (Malignant): {cv_metrics['cv_mean_f1_malignant']:.4f} "
            f"(±{cv_metrics['cv_std_f1_malignant']:.4f}), "
            f"Recall: {cv_metrics['cv_mean_recall_malignant']:.4f}, "
            f"Accuracy: {cv_metrics['cv_mean_accuracy']:.4f}"
        )

    print("\n" + "=" * 70)
    print("STEP 5: Model Selection (Champion Decision)")
    print("=" * 70)
    champion_name, champion_cand = select_champion_model(candidates)
    print(f"SELECTED CHAMPION: {champion_name.upper()}")
    print(
        f"Selection metric: cv_mean_f1_malignant = "
        f"{champion_cand['cv_metrics']['cv_mean_f1_malignant']:.4f}"
    )
    print(
        "Selection Rule: Highest 5-fold Stratified CV F1-score on Malignant class on training data."
    )

    print("\n" + "=" * 70)
    print("STEP 6: MLflow Experiment Logging & Final Evaluation on Untouched Test Set")
    print("=" * 70)
    configure_mlflow(tracking_uri)

    champion_summary = {}
    registered_version = None

    for name, cand in candidates.items():
        is_champ = name == champion_name
        with mlflow.start_run(run_name=f"{name}_run") as run:
            run_id = run.info.run_id
            cand["run_id"] = run_id

            # Fit pipeline on full training set
            pipeline = cand["pipeline"]
            pipeline.fit(X_train, y_train)

            # Evaluate on held-out test set
            y_pred = pipeline.predict(X_test)
            y_prob_malignant = (
                pipeline.predict_proba(X_test)[:, 0] if hasattr(pipeline, "predict_proba") else None
            )
            test_metrics = compute_metrics(y_test.values, y_pred, y_prob_malignant)
            cand["test_metrics"] = test_metrics

            # Log tags
            mlflow.set_tag("model_type", cand["model_type"])
            mlflow.set_tag("is_champion", str(is_champ))
            mlflow.set_tag("stage", "champion" if is_champ else "candidate")

            # Log parameters
            mlflow.log_params(cand["params"])

            # Log CV metrics
            for k, v in cand["cv_metrics"].items():
                mlflow.log_metric(k, v)

            # Log Test metrics
            for k, v in test_metrics.items():
                mlflow.log_metric(f"test_{k}", v)

            # Generate and log confusion matrix plot
            cm_filename = f"confusion_matrix_{name}.png" if not is_champ else "confusion_matrix.png"
            cm_path = os.path.join(reports_dir, cm_filename)
            plot_title = (
                f"Confusion Matrix: {cand['model_type']} "
                f"({'Champion' if is_champ else 'Baseline'})"
            )
            plot_and_save_confusion_matrix(
                y_test.values,
                y_pred,
                output_path=cm_path,
                title=plot_title,
            )
            mlflow.log_artifact(cm_path, artifact_path="evaluation_plots")

            # Infer signature and log model
            input_example = X_train.iloc[:2].copy()
            signature = infer_signature(X_train, pipeline.predict(X_train))
            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model",
                signature=signature,
                input_example=input_example,
            )

            print(f"\nLogged {name} (Run ID: {run_id})")
            print(f"  Test Accuracy:          {test_metrics['accuracy']:.4f}")
            print(f"  Test F1 Malignant:      {test_metrics['f1_malignant']:.4f}")
            print(f"  Test Recall Malignant:  {test_metrics['recall_malignant']:.4f}")
            print(f"  Test ROC-AUC:           {test_metrics['roc_auc']:.4f}")
            print("  Confusion Matrix Breakdown (Positive = Malignant):")
            print(
                f"    TP (True Malignant):  {test_metrics['true_positives_malignant']} | "
                f"FN (Missed Malignant): {test_metrics['false_negatives_malignant']}"
            )
            print(
                f"    FP (False Alarm):     {test_metrics['false_positives_benign']} | "
                f"TN (True Benign):     {test_metrics['true_negatives_benign']}"
            )

            # If champion, register in Model Registry
            if is_champ:
                model_uri = f"runs:/{run_id}/model"
                try:
                    reg_info = mlflow.register_model(model_uri, CHAMPION_REGISTRY_NAME)
                    registered_version = reg_info.version
                    print(
                        f"  Successfully registered in MLflow Model Registry: "
                        f"'{CHAMPION_REGISTRY_NAME}' version {registered_version}"
                    )
                except Exception as reg_err:
                    print(
                        f"  Warning: Model registration failed ({reg_err}). "
                        "Proceeding with artifact export fallback."
                    )
                    registered_version = "1-fallback"

                # Export production model artifact
                model_export_path = os.path.join(models_dir, "model.joblib")
                joblib.dump(pipeline, model_export_path)
                print(f"  Exported champion model to: {model_export_path}")

                # Export metadata.json
                metadata = {
                    "model_name": CHAMPION_REGISTRY_NAME,
                    "model_type": cand["model_type"],
                    "mlflow_run_id": run_id,
                    "mlflow_experiment_name": EXPERIMENT_NAME,
                    "registered_version": registered_version,
                    "selection_metric": "cv_mean_f1_malignant",
                    "selection_rule": (
                        "Highest 5-fold Stratified CV F1-score on Malignant class on training set; "
                        "tie-breaker on CV Malignant Recall"
                    ),
                    "training_timestamp": datetime.now(timezone.utc).isoformat(),
                    "random_state": RANDOM_STATE,
                    "test_size": 0.2,
                    "train_sample_count": len(X_train),
                    "test_sample_count": len(X_test),
                    "features": get_feature_names(),
                    "target_classes": get_target_names(),
                    "positive_class": "malignant (0)",
                    "cv_metrics": cand["cv_metrics"],
                    "test_metrics": test_metrics,
                    "hyperparameters": cand["params"],
                }
                metadata_path = os.path.join(models_dir, "metadata.json")
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)
                print(f"  Exported metadata to: {metadata_path}")

                champion_summary = metadata

    print("\n" + "=" * 70)
    print("TRAINING WORKFLOW COMPLETED")
    print("=" * 70)
    return champion_summary
