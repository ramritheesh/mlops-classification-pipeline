"""Model evaluation module calculating classification metrics and visual artifacts.

Clinical / Diagnostic Terminology:
- Positive Class: Malignant (class 0) - Condition of primary interest
- Negative Class: Benign (class 1) - Non-cancerous finding
- True Positive (TP): True Malignant correctly predicted as Malignant (cm[0, 0])
- False Negative (FN): True Malignant incorrectly predicted as Benign (cm[0, 1]) [CRITICAL ERROR]
- False Positive (FP): True Benign incorrectly predicted as Malignant (cm[1, 0]) [FALSE ALARM]
- True Negative (TN): True Benign correctly predicted as Benign (cm[1, 1])
"""

import os
from typing import Any, Dict, Optional

# Configure Matplotlib temp cache directory to avoid permissions warnings
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for headless/CI execution
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

DISPLAY_LABELS = ["Malignant (Pos)", "Benign (Neg)"]


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob_malignant: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute comprehensive classification metrics with Malignant (0) as positive class.

    Args:
        y_true: True binary target labels (0 = malignant, 1 = benign).
        y_pred: Predicted binary target labels (0 = malignant, 1 = benign).
        y_prob_malignant: Predicted probability of the positive class (class 0, malignant).

    Returns:
        Dictionary of rounded metrics and confusion matrix counts.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # Clinical confusion matrix mapping:
    tp = int(cm[0, 0])  # True Malignant predicted Malignant
    fn = int(cm[0, 1])  # True Malignant predicted Benign
    fp = int(cm[1, 0])  # True Benign predicted Malignant
    tn = int(cm[1, 1])  # True Benign predicted Benign

    metrics: Dict[str, Any] = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        # Macro metrics (unweighted average across both classes)
        "precision_macro": round(
            float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4
        ),
        "recall_macro": round(
            float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4
        ),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        # Primary Positive Class: Malignant (pos_label=0)
        "precision_malignant": round(
            float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)), 4
        ),
        "recall_malignant": round(
            float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)), 4
        ),
        "f1_malignant": round(float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)), 4),
        # Negative Class: Benign (pos_label=1)
        "precision_benign": round(
            float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)), 4
        ),
        "recall_benign": round(
            float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)), 4
        ),
        "f1_benign": round(float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        # Diagnostic counts
        "true_positives_malignant": tp,
        "false_negatives_malignant": fn,
        "false_positives_benign": fp,
        "true_negatives_benign": tn,
    }

    if y_prob_malignant is not None:
        try:
            # Positive class is malignant (y_true == 0)
            metrics["roc_auc"] = round(float(roc_auc_score(y_true == 0, y_prob_malignant)), 4)
        except Exception:
            metrics["roc_auc"] = 0.0

    return metrics


def plot_and_save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str = "reports/confusion_matrix.png",
    title: str = "Confusion Matrix",
) -> str:
    """Generate and save an annotated confusion matrix plot.

    Labels:
    - [0, 0]: True Positive (TP = Malignant correctly detected)
    - [0, 1]: False Negative (FN = Malignant missed as Benign)
    - [1, 0]: False Positive (FP = Benign misdiagnosed as Malignant)
    - [1, 1]: True Negative (TN = Benign correctly identified)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=DISPLAY_LABELS)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format="d")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.text(
        0.5,
        -0.15,
        "TP: Malignant -> Malignant | FN: Malignant -> Benign\n"
        "FP: Benign -> Malignant | TN: Benign -> Benign",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=9,
        color="dimgray",
    )
    plt.tight_layout()

    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return os.path.abspath(output_path)
