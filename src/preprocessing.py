"""Preprocessing pipeline builder.

Encapsulates feature scaling and estimators inside a scikit-learn Pipeline
to eliminate data leakage during cross-validation and production inference.
"""

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_pipeline(classifier: BaseEstimator) -> Pipeline:
    """Construct an end-to-end scikit-learn Pipeline with scaling and classifier.

    Why this matters for MLOps:
    1. Zero Data Leakage: StandardScaler is fitted strictly on the training partition
       (or each cross-validation train fold). The evaluation partition is only transformed.
    2. Atomic Deployment: The fitted pipeline stores both the scaler parameters
       (mean, scale) and model weights in a single serializable object.
    3. Production Simplicity: Downstream inference services feed raw, unscaled feature
       dictionaries directly into pipeline.predict().

    Args:
        classifier: A scikit-learn compatible classifier (e.g., LogisticRegression).

    Returns:
        Configured scikit-learn Pipeline instance.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )
