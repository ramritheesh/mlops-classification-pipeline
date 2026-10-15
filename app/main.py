"""FastAPI Production Inference Application for Breast Cancer Classification.

This service exposes REST endpoints for model health, governance metadata,
single prediction, and batch prediction with Pydantic v2 validation.

IMPORTANT DISCLAIMER:
This API is strictly an educational MLOps engineering demonstration.
It is NOT a medical diagnostic tool and must NEVER be used for clinical decision-making.
"""

import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncGenerator

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.model_loader import model_loader
from app.monitoring import compute_monitoring_summary, log_error, log_prediction
from app.schemas import (
    DEMONSTRATION_DISCLAIMER,
    BatchPredictionRequest,
    BatchPredictionResponse,
    CancerFeatures,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    RootResponse,
)
from src.data import TARGET_NAMES


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifespan: load model once at startup."""
    try:
        model_loader.load("models/model.joblib", "models/metadata.json")
    except Exception as e:
        # Allow server to start so /health can report degraded status
        print(f"CRITICAL: Failed to load model at startup: {e}")
    yield


app = FastAPI(
    title="Breast Cancer Classification MLOps API",
    description=(
        "Production-style machine learning inference API with experiment tracking, "
        "model governance, input validation, and prediction monitoring.\n\n"
        f"**IMPORTANT NOTICE:** {DEMONSTRATION_DISCLAIMER}"
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle Pydantic validation errors cleanly without leaking internal traces."""
    req_id = str(uuid.uuid4())
    error_messages = []
    for err in exc.errors():
        loc = " -> ".join(str(loc_part) for loc_part in err.get("loc", []))
        msg = err.get("msg", "Invalid value")
        error_messages.append(f"{loc}: {msg}")

    detail_str = "; ".join(error_messages)
    log_error(
        request_id=req_id,
        error_type="ValidationError",
        error_detail=detail_str,
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation Error",
            "message": "Input data failed schema validation rules.",
            "details": error_messages,
            "request_id": req_id,
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unhandled server exceptions cleanly without exposing stack traces."""
    req_id = str(uuid.uuid4())
    log_error(
        request_id=req_id,
        error_type="InternalServerError",
        error_detail=str(exc),
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred while processing your request.",
            "request_id": req_id,
        },
    )


@app.get("/", response_model=RootResponse, tags=["General"])
def root() -> RootResponse:
    """Return service identification, status, and interactive documentation link."""
    return RootResponse(
        service_name="Breast Cancer Classification MLOps API",
        version="1.0.0",
        status="running",
        docs_url="/docs",
        disclaimer=DEMONSTRATION_DISCLAIMER,
    )


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check() -> HealthResponse:
    """Return service health status and model availability."""
    is_loaded = model_loader.is_loaded
    if not is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Production model is not loaded.",
        )
    meta = model_loader.metadata
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_name=meta.get("model_name", "unknown"),
        model_version=str(meta.get("registered_version", "1")),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Governance"])
def model_info() -> ModelInfoResponse:
    """Return model metadata, evaluation metrics, and governance parameters."""
    if not model_loader.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model metadata is unavailable.",
        )
    meta = model_loader.metadata
    return ModelInfoResponse(
        model_name=meta.get("model_name", "unknown"),
        model_type=meta.get("model_type", "unknown"),
        registered_version=meta.get("registered_version", 1),
        mlflow_run_id=meta.get("mlflow_run_id", "unknown"),
        selection_metric=meta.get("selection_metric", "cv_mean_f1"),
        selection_rule=meta.get("selection_rule", "unknown"),
        cv_metrics=meta.get("cv_metrics", {}),
        test_metrics=meta.get("test_metrics", {}),
        features=meta.get("features", []),
        target_classes=meta.get("target_classes", TARGET_NAMES),
        training_timestamp=meta.get("training_timestamp", "unknown"),
        disclaimer=DEMONSTRATION_DISCLAIMER,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_single(features: CancerFeatures) -> PredictionResponse:
    """Predict tumor class for a single set of cell nucleus measurements."""
    if not model_loader.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded.",
        )

    start_time = time.perf_counter()
    req_id = str(uuid.uuid4())
    feature_dict = features.model_dump()
    model = model_loader.model
    meta = model_loader.metadata

    # Convert to single-row DataFrame with expected feature ordering
    expected_order = meta.get("features", list(feature_dict.keys()))
    ordered_dict = {col: [feature_dict[col]] for col in expected_order}
    df = pd.DataFrame(ordered_dict)

    pred = int(model.predict(df)[0])
    class_name = TARGET_NAMES[pred]

    probabilities = {}
    confidence = 1.0
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(df)[0]
        confidence = round(float(probs[pred]), 4)
        probabilities = {
            TARGET_NAMES[0]: round(float(probs[0]), 4),
            TARGET_NAMES[1]: round(float(probs[1]), 4),
        }

    latency_ms = (time.perf_counter() - start_time) * 1000.0

    # Privacy-conscious logging
    log_prediction(
        request_id=req_id,
        model_version=str(meta.get("registered_version", 1)),
        feature_dict=feature_dict,
        prediction=pred,
        class_name=class_name,
        probability=confidence,
        latency_ms=latency_ms,
    )

    return PredictionResponse(
        prediction=pred,
        class_name=class_name,
        probability=confidence,
        probabilities=probabilities,
        model_version=str(meta.get("registered_version", 1)),
        request_id=req_id,
        latency_ms=round(latency_ms, 2),
        disclaimer=DEMONSTRATION_DISCLAIMER,
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch(batch_request: BatchPredictionRequest) -> BatchPredictionResponse:
    """Predict tumor classes for a batch of 1 to 100 samples."""
    if not model_loader.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded.",
        )

    start_time = time.perf_counter()
    model = model_loader.model
    meta = model_loader.metadata
    samples = batch_request.samples
    expected_order = meta.get("features", [])

    rows = [{col: s.model_dump()[col] for col in expected_order} for s in samples]
    df = pd.DataFrame(rows)

    preds = model.predict(df)
    probs = model.predict_proba(df) if hasattr(model, "predict_proba") else None

    responses = []
    for i, pred_int in enumerate(preds):
        p = int(pred_int)
        c_name = TARGET_NAMES[p]
        conf = round(float(probs[i][p]), 4) if probs is not None else 1.0
        prob_dist = (
            {
                TARGET_NAMES[0]: round(float(probs[i][0]), 4),
                TARGET_NAMES[1]: round(float(probs[i][1]), 4),
            }
            if probs is not None
            else {}
        )
        sample_id = str(uuid.uuid4())

        # Log individual prediction
        log_prediction(
            request_id=sample_id,
            model_version=str(meta.get("registered_version", 1)),
            feature_dict=rows[i],
            prediction=p,
            class_name=c_name,
            probability=conf,
            latency_ms=0.0,
        )

        responses.append(
            PredictionResponse(
                prediction=p,
                class_name=c_name,
                probability=conf,
                probabilities=prob_dist,
                model_version=str(meta.get("registered_version", 1)),
                request_id=sample_id,
                latency_ms=0.0,
                disclaimer=DEMONSTRATION_DISCLAIMER,
            )
        )

    total_latency_ms = (time.perf_counter() - start_time) * 1000.0

    return BatchPredictionResponse(
        predictions=responses,
        batch_size=len(responses),
        total_latency_ms=round(total_latency_ms, 2),
        disclaimer=DEMONSTRATION_DISCLAIMER,
    )


@app.get("/monitoring/summary", tags=["Monitoring"])
def monitoring_summary() -> dict:
    """Return summary statistics of logged predictions and service requests."""
    return compute_monitoring_summary()
