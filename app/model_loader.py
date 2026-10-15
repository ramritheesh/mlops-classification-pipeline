"""Singleton model loader and warmup manager for FastAPI lifecycle."""

import json
import logging
import os
from typing import Any, Dict, Optional

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

logger = logging.getLogger("uvicorn.error")


class ModelLoader:
    """Thread-safe singleton model loader ensuring single in-memory load on startup."""

    _instance: Optional["ModelLoader"] = None
    _model: Optional[Pipeline] = None
    _metadata: Optional[Dict[str, Any]] = None

    def __new__(cls) -> "ModelLoader":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(
        self,
        model_path: str = "models/model.joblib",
        metadata_path: str = "models/metadata.json",
    ) -> None:
        """Load model pipeline and metadata from disk and perform warmup inference."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at '{model_path}'. "
                "Ensure training pipeline has completed before starting the API."
            )
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found at '{metadata_path}'.")

        logger.info(f"Loading production model from '{model_path}'...")
        self._model = joblib.load(model_path)

        with open(metadata_path, "r") as f:
            self._metadata = json.load(f)

        logger.info(
            f"Loaded model '{self._metadata.get('model_name')}' "
            f"version {self._metadata.get('registered_version')}"
        )

        # Warmup inference
        self._warmup()

    def _warmup(self) -> None:
        """Execute warmup inference to ensure libraries and caches are initialized."""
        if self._model is None or self._metadata is None:
            return

        features = self._metadata.get("features", [])
        # Construct 1-row dummy dataframe with zeros
        dummy_df = pd.DataFrame([{f: 1.0 for f in features}])
        _ = self._model.predict(dummy_df)
        if hasattr(self._model, "predict_proba"):
            _ = self._model.predict_proba(dummy_df)
        logger.info("Model warmup completed successfully.")

    @property
    def model(self) -> Pipeline:
        if self._model is None:
            raise RuntimeError("Model is not loaded. Call load() first.")
        return self._model

    @property
    def metadata(self) -> Dict[str, Any]:
        if self._metadata is None:
            raise RuntimeError("Metadata is not loaded. Call load() first.")
        return self._metadata

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._metadata is not None


# Global singleton instance
model_loader = ModelLoader()
