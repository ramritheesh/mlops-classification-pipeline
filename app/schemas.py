"""Pydantic v2 request and response schemas for FastAPI inference endpoints."""

from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict, Field

DEMONSTRATION_DISCLAIMER = (
    "DEMONSTRATION ONLY: This API is an educational MLOps engineering benchmark. "
    "It is strictly NOT a certified clinical diagnostic tool and must NEVER be used "
    "for real-world medical diagnosis or healthcare decision-making."
)


class CancerFeatures(BaseModel):
    """Input features schema representing cell nucleus measurements.

    All values must be non-negative real numbers.
    Extra unknown fields are strictly forbidden to ensure schema integrity.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "mean_radius": 17.99,
                "mean_texture": 10.38,
                "mean_perimeter": 122.8,
                "mean_area": 1001.0,
                "mean_smoothness": 0.1184,
                "mean_compactness": 0.2776,
                "mean_concavity": 0.3001,
                "mean_concave_points": 0.1471,
                "mean_symmetry": 0.2419,
                "mean_fractal_dimension": 0.07871,
                "radius_error": 1.095,
                "texture_error": 0.9053,
                "perimeter_error": 8.589,
                "area_error": 153.4,
                "smoothness_error": 0.006399,
                "compactness_error": 0.04904,
                "concavity_error": 0.05373,
                "concave_points_error": 0.01587,
                "symmetry_error": 0.03003,
                "fractal_dimension_error": 0.006193,
                "worst_radius": 25.38,
                "worst_texture": 17.33,
                "worst_perimeter": 184.6,
                "worst_area": 2019.0,
                "worst_smoothness": 0.1622,
                "worst_compactness": 0.6656,
                "worst_concavity": 0.7119,
                "worst_concave_points": 0.2654,
                "worst_symmetry": 0.4601,
                "worst_fractal_dimension": 0.1189,
            }
        },
    )

    mean_radius: float = Field(..., ge=0.0, le=100.0, description="Mean radius of cell nuclei")
    mean_texture: float = Field(
        ..., ge=0.0, le=100.0, description="Mean texture (std of gray-scale values)"
    )
    mean_perimeter: float = Field(
        ..., ge=0.0, le=500.0, description="Mean perimeter of cell nuclei"
    )
    mean_area: float = Field(..., ge=0.0, le=5000.0, description="Mean area of cell nuclei")
    mean_smoothness: float = Field(
        ..., ge=0.0, le=1.0, description="Mean smoothness (local variation in radius)"
    )
    mean_compactness: float = Field(
        ..., ge=0.0, le=2.0, description="Mean compactness (perimeter^2 / area - 1.0)"
    )
    mean_concavity: float = Field(
        ..., ge=0.0, le=2.0, description="Mean concavity (severity of concave portions)"
    )
    mean_concave_points: float = Field(
        ..., ge=0.0, le=1.0, description="Mean concave points (number of concave portions)"
    )
    mean_symmetry: float = Field(..., ge=0.0, le=1.0, description="Mean symmetry of cell nuclei")
    mean_fractal_dimension: float = Field(
        ..., ge=0.0, le=1.0, description="Mean fractal dimension ('coastline approx' - 1)"
    )

    radius_error: float = Field(..., ge=0.0, le=50.0, description="Standard error for radius")
    texture_error: float = Field(..., ge=0.0, le=50.0, description="Standard error for texture")
    perimeter_error: float = Field(
        ..., ge=0.0, le=100.0, description="Standard error for perimeter"
    )
    area_error: float = Field(..., ge=0.0, le=1000.0, description="Standard error for area")
    smoothness_error: float = Field(
        ..., ge=0.0, le=1.0, description="Standard error for smoothness"
    )
    compactness_error: float = Field(
        ..., ge=0.0, le=1.0, description="Standard error for compactness"
    )
    concavity_error: float = Field(..., ge=0.0, le=1.0, description="Standard error for concavity")
    concave_points_error: float = Field(
        ..., ge=0.0, le=1.0, description="Standard error for concave points"
    )
    symmetry_error: float = Field(..., ge=0.0, le=1.0, description="Standard error for symmetry")
    fractal_dimension_error: float = Field(
        ..., ge=0.0, le=1.0, description="Standard error for fractal dimension"
    )

    worst_radius: float = Field(..., ge=0.0, le=100.0, description="Largest ('worst') radius")
    worst_texture: float = Field(..., ge=0.0, le=100.0, description="Largest ('worst') texture")
    worst_perimeter: float = Field(..., ge=0.0, le=500.0, description="Largest ('worst') perimeter")
    worst_area: float = Field(..., ge=0.0, le=5000.0, description="Largest ('worst') area")
    worst_smoothness: float = Field(..., ge=0.0, le=1.0, description="Largest ('worst') smoothness")
    worst_compactness: float = Field(
        ..., ge=0.0, le=3.0, description="Largest ('worst') compactness"
    )
    worst_concavity: float = Field(..., ge=0.0, le=3.0, description="Largest ('worst') concavity")
    worst_concave_points: float = Field(
        ..., ge=0.0, le=1.0, description="Largest ('worst') concave points"
    )
    worst_symmetry: float = Field(..., ge=0.0, le=1.0, description="Largest ('worst') symmetry")
    worst_fractal_dimension: float = Field(
        ..., ge=0.0, le=1.0, description="Largest ('worst') fractal dimension"
    )


class PredictionResponse(BaseModel):
    """Response schema for single prediction."""

    prediction: int = Field(..., description="Predicted numeric class (0 = malignant, 1 = benign)")
    class_name: str = Field(..., description="Target diagnosis label ('malignant' or 'benign')")
    probability: float = Field(..., description="Confidence probability for the predicted class")
    probabilities: Dict[str, float] = Field(..., description="Full class probability distribution")
    model_version: str = Field(..., description="Active production model version")
    request_id: str = Field(..., description="Unique request identifier for tracing")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")
    disclaimer: str = Field(
        default=DEMONSTRATION_DISCLAIMER, description="Educational medical disclaimer"
    )


class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions with enforced batch size limits."""

    samples: List[CancerFeatures] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Array of 1 to 100 cancer feature samples",
    )


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""

    predictions: List[PredictionResponse] = Field(..., description="List of prediction results")
    batch_size: int = Field(..., description="Number of samples evaluated")
    total_latency_ms: float = Field(
        ..., description="Total batch processing latency in milliseconds"
    )
    disclaimer: str = Field(
        default=DEMONSTRATION_DISCLAIMER, description="Educational medical disclaimer"
    )


class HealthResponse(BaseModel):
    """Service health and model readiness response schema."""

    status: str = Field(..., description="Overall service status ('healthy' or 'degraded')")
    model_loaded: bool = Field(
        ..., description="Whether the production ML model is loaded in memory"
    )
    model_name: str = Field(..., description="Name of the loaded production model")
    model_version: str = Field(..., description="Registered version of the model")
    timestamp: str = Field(..., description="Current ISO 8601 UTC timestamp")


class ModelInfoResponse(BaseModel):
    """Model governance and evaluation metadata response schema."""

    model_name: str = Field(..., description="Model registry name")
    model_type: str = Field(..., description="Algorithm type")
    registered_version: Any = Field(..., description="Registered version identifier")
    mlflow_run_id: str = Field(..., description="MLflow tracking run ID")
    selection_metric: str = Field(..., description="Metric used for champion selection")
    selection_rule: str = Field(..., description="Documented champion decision rule")
    cv_metrics: Dict[str, float] = Field(
        ..., description="Cross-validation performance on training set"
    )
    test_metrics: Dict[str, float] = Field(..., description="Final metrics on held-out test set")
    features: List[str] = Field(..., description="Ordered list of required feature names")
    target_classes: List[str] = Field(..., description="Target class names (0=malignant, 1=benign)")
    training_timestamp: str = Field(..., description="Timestamp when the model was trained")
    disclaimer: str = Field(
        default=DEMONSTRATION_DISCLAIMER, description="Educational medical disclaimer"
    )


class RootResponse(BaseModel):
    """Root service information response schema."""

    service_name: str
    version: str
    status: str
    docs_url: str
    disclaimer: str = DEMONSTRATION_DISCLAIMER
