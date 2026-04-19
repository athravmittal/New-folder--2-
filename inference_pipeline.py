"""
inference_pipeline.py
---------------------
Real-time inference pipeline for fatigue/cognitive-load prediction.

This module is the backend-facing API. It:
  - Accepts pre-aggregated feature vectors (never raw events)
  - Optionally incorporates user-side baseline deviations (z-scores)
  - Returns a prediction with confidence and per-class probabilities
  - Does NOT store any user-identifying or raw behavioral data

Privacy contract:
  - The caller (frontend/edge) computes and sends ONLY aggregated features
  - The caller MAY send baseline_mean + baseline_std for deviation features
  - This backend never stores, logs, or persists baseline information
"""

import logging
import numpy as np
from typing import Optional
from datetime import datetime, timezone

from feature_engineering import compute_deviation_features, normalize_label
from training_pipeline import load_model, update_model, schedule_batch_retrain

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Inference Engine (stateless singleton pattern)
# ---------------------------------------------------------------------------

class InferenceEngine:
    """
    Singleton inference engine that wraps the global model.

    Maintains:
      - The loaded model artifact (pipeline + metadata)
      - A bounded update buffer for batch retraining (RandomForest path)
      - Prediction history (in-memory, session-scoped, not persisted)
    """

    def __init__(self, max_buffer_size: int = 200):
        self._model_artifact: Optional[dict] = None
        self._update_buffer: list = []           # list of (features, label)
        self._max_buffer = max_buffer_size
        self._prediction_count = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the global model from disk."""
        self._model_artifact = load_model()
        log.info("InferenceEngine: model loaded.")

    def reload(self) -> None:
        """Hot-reload model without restarting (e.g. after retrain)."""
        self._model_artifact = load_model()
        log.info("InferenceEngine: model hot-reloaded.")

    def is_ready(self) -> bool:
        return self._model_artifact is not None

    # ------------------------------------------------------------------
    # Core Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        raw_features: dict,
        baseline_mean: Optional[dict] = None,
        baseline_std:  Optional[dict] = None,
    ) -> dict:
        """
        Predict cognitive load / fatigue level from aggregated features.

        Args:
            raw_features   : feature dict for the current 5-min window
                             (produced by feature_engineering.build_full_feature_vector)
            baseline_mean  : user's personal feature means (client-computed, optional)
            baseline_std   : user's personal feature stds  (client-computed, optional)

        Returns a prediction dict:
            {
                "label":        "HIGH",
                "confidence":   0.83,
                "probabilities": {"LOW": 0.05, "AVERAGE": 0.12, "HIGH": 0.83},
                "timestamp":    "2024-01-15T10:30:00Z",
                "feature_count": 22,
            }
        """
        if not self.is_ready():
            raise RuntimeError("InferenceEngine not loaded. Call .load() first.")

        # 1. Augment with deviation features if baseline is provided
        features = dict(raw_features)
        if baseline_mean and baseline_std:
            deviations = compute_deviation_features(
                current=raw_features,
                baseline_mean=baseline_mean,
                baseline_std=baseline_std,
            )
            features.update(deviations)

        # 2. Align to training feature schema (fill missing = 0)
        pipeline      = self._model_artifact["pipeline"]
        feature_names = self._model_artifact["feature_names"]
        le            = self._model_artifact["label_encoder"]

        x = np.array([[features.get(f, 0.0) for f in feature_names]])

        # 3. Predict
        pred_enc  = pipeline.predict(x)[0]
        pred_label = le.inverse_transform([pred_enc])[0]

        # 4. Probabilities (if model supports it)
        probabilities = {}
        if hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba(x)[0]
            probabilities = {cls: round(float(p), 4)
                             for cls, p in zip(le.classes_, proba)}
            confidence = float(proba.max())
        else:
            confidence = 1.0  # SGD with hinge doesn't produce probabilities
            probabilities = {cls: (1.0 if cls == pred_label else 0.0)
                             for cls in le.classes_}

        self._prediction_count += 1

        return {
            "label":         pred_label,
            "confidence":    round(confidence, 4),
            "probabilities": probabilities,
            "timestamp":     datetime.now(timezone.utc).isoformat(),
            "feature_count": len(feature_names),
        }

    # ------------------------------------------------------------------
    # Continuous Learning
    # ------------------------------------------------------------------

    def submit_feedback(
        self,
        features: dict,
        true_label: str,
        baseline_mean: Optional[dict] = None,
        baseline_std:  Optional[dict] = None,
    ) -> dict:
        """
        Accept a labeled observation for model updating.

        Two paths:
          - SGD model: immediate partial_fit
          - RandomForest: buffer the sample, trigger batch retrain at threshold

        Args:
            features      : same feature dict used for prediction
            true_label    : ground-truth label (user-confirmed or auto-collected)
            baseline_mean : optional, for augmenting with deviation features
            baseline_std  : optional

        Returns a status dict.
        """
        if not self.is_ready():
            raise RuntimeError("InferenceEngine not loaded.")

        # Augment with deviations if available
        augmented = dict(features)
        if baseline_mean and baseline_std:
            deviations = compute_deviation_features(features, baseline_mean, baseline_std)
            augmented.update(deviations)

        canonical_label = normalize_label(true_label)
        model_type      = self._model_artifact.get("model_type", "random_forest")

        if model_type == "sgd":
            # Immediate incremental update
            self._model_artifact = update_model(
                self._model_artifact, augmented, canonical_label
            )
            return {"status": "updated", "method": "incremental", "label": canonical_label}

        else:
            # Buffer for batch retrain
            self._update_buffer.append((augmented, canonical_label))

            # Trim buffer to max size (sliding window — keep recent data)
            if len(self._update_buffer) > self._max_buffer:
                self._update_buffer = self._update_buffer[-self._max_buffer:]

            # Attempt retrain
            new_artifact = schedule_batch_retrain(
                self._model_artifact,
                self._update_buffer,
                min_samples=50,
            )
            if new_artifact is not None:
                self._model_artifact = new_artifact
                self._update_buffer.clear()
                return {
                    "status":  "retrained",
                    "method":  "batch",
                    "label":   canonical_label,
                    "buffer":  0,
                }

            return {
                "status": "buffered",
                "method": "batch",
                "label":  canonical_label,
                "buffer": len(self._update_buffer),
            }

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> dict:
        """Return engine health and metadata."""
        if not self.is_ready():
            return {"ready": False}
        return {
            "ready":            True,
            "model_type":       self._model_artifact.get("model_type"),
            "trained_at":       self._model_artifact.get("trained_at"),
            "last_updated":     self._model_artifact.get("last_updated"),
            "feature_count":    len(self._model_artifact.get("feature_names", [])),
            "predictions_made": self._prediction_count,
            "buffer_size":      len(self._update_buffer),
        }


# ---------------------------------------------------------------------------
# Module-level singleton (import and use directly)
# ---------------------------------------------------------------------------

engine = InferenceEngine()


# ---------------------------------------------------------------------------
# Public API functions (stateless wrappers around the singleton)
# ---------------------------------------------------------------------------

def predict_realtime(
    features: dict,
    baseline_mean: Optional[dict] = None,
    baseline_std:  Optional[dict] = None,
) -> dict:
    """
    Top-level inference entry point.

    Call this from your REST API handler (Flask/FastAPI/etc.).

    Example:
        result = predict_realtime(
            features={"typing_speed": 45.2, "ikd_mean": 120.3, ...},
            baseline_mean={"typing_speed": 55.0, "ikd_mean": 100.0, ...},
            baseline_std={"typing_speed": 8.0, "ikd_mean": 15.0, ...},
        )
        # → {"label": "HIGH", "confidence": 0.81, "probabilities": {...}, ...}
    """
    if not engine.is_ready():
        engine.load()
    return engine.predict(features, baseline_mean, baseline_std)


def submit_label_feedback(
    features: dict,
    true_label: str,
    baseline_mean: Optional[dict] = None,
    baseline_std:  Optional[dict] = None,
) -> dict:
    """
    Accept user-confirmed label feedback for continuous learning.

    Call this when the user confirms or corrects a prediction.
    """
    if not engine.is_ready():
        engine.load()
    return engine.submit_feedback(features, true_label, baseline_mean, baseline_std)


def get_engine_status() -> dict:
    """Return current engine status and metadata."""
    return engine.status()
