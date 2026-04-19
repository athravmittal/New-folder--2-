"""
api_server.py
-------------
Lightweight REST API exposing the fatigue prediction backend.

Framework: Flask (swap for FastAPI with minimal changes)

Endpoints:
  POST /predict          → real-time fatigue prediction
  POST /feedback         → submit confirmed label for continuous learning
  POST /train            → trigger full training pipeline (admin)
  GET  /status           → engine health check
  GET  /health           → liveness probe

Privacy:
  - No request body is logged beyond feature keys (no values)
  - No IP or user identifiers are persisted
  - Baseline data in request is used transiently and discarded
"""

import logging
from flask import Flask, request, jsonify, abort

from inference_pipeline import (
    predict_realtime,
    submit_label_feedback,
    get_engine_status,
    engine,
)
from training_pipeline import build_training_dataset, train_global_model

log = logging.getLogger(__name__)
app = Flask(__name__)


# ---------------------------------------------------------------------------
# Startup: pre-load model
# ---------------------------------------------------------------------------

@app.before_request
def ensure_model_loaded():
    """Load model on first request if not already loaded."""
    if not engine.is_ready():
        try:
            engine.load()
        except FileNotFoundError:
            pass  # /health and /train will still work; /predict returns 503


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    """Kubernetes/Docker liveness probe."""
    return jsonify({"status": "ok"}), 200


@app.route("/status", methods=["GET"])
def status():
    """
    Return engine status and model metadata.

    Response:
        {
            "ready": true,
            "model_type": "random_forest",
            "trained_at": "2024-01-15T09:00:00",
            "feature_count": 22,
            "predictions_made": 418,
            "buffer_size": 12
        }
    """
    return jsonify(get_engine_status()), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Real-time fatigue prediction.

    Request JSON:
        {
            "features": {
                "typing_speed": 42.1,
                "ikd_mean": 130.5,
                "hold_time_mean": 82.0,
                ... (all feature keys)
            },
            "baseline_mean": {          ← optional, client-computed
                "typing_speed": 55.0,
                "ikd_mean": 110.0,
                ...
            },
            "baseline_std": {           ← optional, client-computed
                "typing_speed": 9.0,
                "ikd_mean": 18.0,
                ...
            }
        }

    Response:
        {
            "label": "HIGH",
            "confidence": 0.82,
            "probabilities": {"LOW": 0.05, "AVERAGE": 0.13, "HIGH": 0.82},
            "timestamp": "2024-01-15T10:30:00+00:00",
            "feature_count": 22
        }
    """
    if not engine.is_ready():
        return jsonify({"error": "Model not loaded. Train the model first via POST /train"}), 503

    body = request.get_json(silent=True)
    if not body or "features" not in body:
        abort(400, description="Request must include a 'features' dict.")

    features      = body["features"]
    baseline_mean = body.get("baseline_mean")
    baseline_std  = body.get("baseline_std")

    # Validate: features must be a dict of numbers
    if not isinstance(features, dict):
        abort(400, description="'features' must be a JSON object.")

    log.debug(f"/predict called with {len(features)} feature keys")

    try:
        result = predict_realtime(features, baseline_mean, baseline_std)
        return jsonify(result), 200
    except Exception as e:
        log.exception("Prediction failed")
        return jsonify({"error": str(e)}), 500


@app.route("/feedback", methods=["POST"])
def feedback():
    """
    Submit a labeled observation for continuous learning.

    Call this after the user confirms or corrects a prediction.

    Request JSON:
        {
            "features": { ... same as /predict ... },
            "label": "HIGH",            ← confirmed ground truth
            "baseline_mean": { ... },   ← optional
            "baseline_std":  { ... }    ← optional
        }

    Response:
        {
            "status": "buffered" | "updated" | "retrained",
            "method": "batch" | "incremental",
            "label": "HIGH",
            "buffer": 23                ← only for batch path
        }
    """
    if not engine.is_ready():
        return jsonify({"error": "Model not loaded."}), 503

    body = request.get_json(silent=True)
    if not body or "features" not in body or "label" not in body:
        abort(400, description="Request must include 'features' and 'label'.")

    features      = body["features"]
    true_label    = body["label"]
    baseline_mean = body.get("baseline_mean")
    baseline_std  = body.get("baseline_std")

    try:
        result = submit_label_feedback(features, true_label, baseline_mean, baseline_std)
        return jsonify(result), 200
    except Exception as e:
        log.exception("Feedback submission failed")
        return jsonify({"error": str(e)}), 500


@app.route("/train", methods=["POST"])
def train():
    """
    Trigger a full training pipeline run. (Admin / offline use)

    Request JSON:
        {
            "keystroke_path":  "data/keystrokes.csv",
            "mouse_path":      "data/mouse.csv",
            "inactivity_path": "data/inactivity.csv",
            "window_path":     "data/windows.csv",
            "label_path":      "data/labels.csv",
            "model_type":      "random_forest"    ← or "sgd"
        }

    Response:
        {
            "status": "trained",
            "model_type": "random_forest",
            "trained_at": "2024-01-15T09:00:00",
            "test_report": "..."
        }
    """
    body = request.get_json(silent=True) or {}

    required = ["keystroke_path", "mouse_path", "inactivity_path",
                 "window_path", "label_path"]
    missing = [k for k in required if k not in body]
    if missing:
        abort(400, description=f"Missing required fields: {missing}")

    model_type = body.get("model_type", "random_forest")

    try:
        df = build_training_dataset(
            keystroke_path  = body["keystroke_path"],
            mouse_path      = body["mouse_path"],
            inactivity_path = body["inactivity_path"],
            window_path     = body["window_path"],
            label_path      = body["label_path"],
        )
        result = train_global_model(df, model_type=model_type, save=True)
        engine.reload()  # hot-reload the new model

        return jsonify({
            "status":      "trained",
            "model_type":  result["model_type"],
            "trained_at":  result["trained_at"],
            "test_report": result["test_report"],
        }), 200

    except Exception as e:
        log.exception("Training failed")
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    app.run(host="0.0.0.0", port=8080, debug=False)
