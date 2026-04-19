"""
training_pipeline.py
--------------------
Global model training pipeline.

Responsibilities:
  - Load and merge historical behavioral datasets
  - Apply time-aware train/test split (no data leakage)
  - Build feature matrix using feature_engineering functions
  - Train either a batch (RandomForest) or incremental (SGDClassifier) model
  - Persist model artifact and feature metadata

Privacy note:
  - Only aggregated per-window features are used
  - No raw keystrokes, mouse coordinates, or personal identifiers are stored
  - User-specific baselines are NOT computed here (they live client-side)
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from feature_engineering import (
    extract_keystroke_features,
    extract_mouse_features,
    extract_inactivity_features,
    extract_window_features,
    compute_deviation_features,
    normalize_label,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WINDOW_SIZE      = "5min"   # aggregation granularity
LABEL_CLASSES    = ["LOW", "AVERAGE", "HIGH"]
MODEL_PATH       = "artifacts/global_model.pkl"
METADATA_PATH    = "artifacts/feature_metadata.pkl"

INCREMENTAL_MODEL_TYPES = (SGDClassifier,)


# ---------------------------------------------------------------------------
# Dataset Builders
# ---------------------------------------------------------------------------

def load_and_window_keystrokes(path: str) -> pd.DataFrame:
    """
    Load raw keystroke CSV and aggregate into time windows.

    Expected raw columns: timestamp, hold_time, ikd
    Returns: DataFrame indexed by window_start with keystroke feature columns.
    """
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp")

    records = []
    for window_start, group in df.groupby(pd.Grouper(key="timestamp", freq=WINDOW_SIZE)):
        feat = extract_keystroke_features(group)
        feat["window_start"] = window_start
        records.append(feat)

    return pd.DataFrame(records).set_index("window_start")


def load_and_window_mouse(path: str) -> pd.DataFrame:
    """Load raw mouse CSV and aggregate into time windows."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp")

    records = []
    for window_start, group in df.groupby(pd.Grouper(key="timestamp", freq=WINDOW_SIZE)):
        feat = extract_mouse_features(group)
        feat["window_start"] = window_start
        records.append(feat)

    return pd.DataFrame(records).set_index("window_start")


def load_and_window_inactivity(path: str) -> pd.DataFrame:
    """Load inactivity CSV and aggregate into time windows."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp")

    records = []
    for window_start, group in df.groupby(pd.Grouper(key="timestamp", freq=WINDOW_SIZE)):
        feat = extract_inactivity_features(group)
        feat["window_start"] = window_start
        records.append(feat)

    return pd.DataFrame(records).set_index("window_start")


def load_and_window_windows(path: str) -> pd.DataFrame:
    """Load active-window CSV and aggregate into time windows."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp")

    records = []
    for window_start, group in df.groupby(pd.Grouper(key="timestamp", freq=WINDOW_SIZE)):
        feat = extract_window_features(group)
        feat["window_start"] = window_start
        records.append(feat)

    return pd.DataFrame(records).set_index("window_start")


def load_labels(path: str) -> pd.Series:
    """
    Load ground-truth condition labels.

    Expected CSV columns: timestamp, label
    Returns: Series indexed by window_start (5-min bins), values = canonical label.

    IMPORTANT: Labels are NOT forward/backward filled.
    Only windows with an explicit label within the window are used for training.
    This eliminates label leakage into unlabeled periods.
    """
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["label"] = df["label"].apply(normalize_label)
    df = df.set_index("timestamp").sort_index()

    # Resample: take the MODE label per window (most common annotation)
    # Windows with no label annotation are dropped (NaN → excluded later)
    resampled = (
        df["label"]
        .resample(WINDOW_SIZE)
        .apply(lambda x: x.mode()[0] if not x.empty else np.nan)
    )
    resampled.index.name = "window_start"
    return resampled


# ---------------------------------------------------------------------------
# Dataset Assembly
# ---------------------------------------------------------------------------

def build_training_dataset(
    keystroke_path:  str,
    mouse_path:      str,
    inactivity_path: str,
    window_path:     str,
    label_path:      str,
) -> pd.DataFrame:
    """
    Merge all modality feature DataFrames with ground-truth labels.

    Returns a clean DataFrame where:
      - Each row = one 5-minute window
      - Columns = all behavioral features
      - 'label' column = ground-truth condition
      - Windows without explicit labels are DROPPED
    """
    log.info("Loading and windowing all modalities...")
    ks   = load_and_window_keystrokes(keystroke_path)
    ms   = load_and_window_mouse(mouse_path)
    inac = load_and_window_inactivity(inactivity_path)
    win  = load_and_window_windows(window_path)
    lbl  = load_labels(label_path)

    log.info("Merging modalities...")
    merged = ks.join(ms, how="outer")\
               .join(inac, how="outer")\
               .join(win, how="outer")

    merged["label"] = lbl

    # Drop windows with no label — critical to avoid leakage
    before = len(merged)
    merged = merged.dropna(subset=["label"])
    after  = len(merged)
    log.info(f"Dropped {before - after} unlabeled windows. Retained {after} labeled windows.")

    # Fill missing feature values with 0 (window had no events of that type)
    feature_cols = [c for c in merged.columns if c != "label"]
    merged[feature_cols] = merged[feature_cols].fillna(0.0)

    log.info(f"Label distribution:\n{merged['label'].value_counts()}")
    return merged


# ---------------------------------------------------------------------------
# Time-Aware Train/Test Split
# ---------------------------------------------------------------------------

def time_aware_split(
    df: pd.DataFrame,
    test_fraction: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically — never randomly.

    The last `test_fraction` of time is held out for evaluation.
    This simulates the real-world scenario where the model is tested
    on future, unseen sessions.
    """
    df = df.sort_index()
    cutoff = int(len(df) * (1 - test_fraction))
    train = df.iloc[:cutoff]
    test  = df.iloc[cutoff:]
    log.info(f"Train size: {len(train)} | Test size: {len(test)}")
    return train, test


# ---------------------------------------------------------------------------
# Model Training
# ---------------------------------------------------------------------------

def train_global_model(
    df: pd.DataFrame,
    model_type: str = "random_forest",   # "random_forest" | "sgd"
    save: bool = True,
) -> dict:
    """
    Train (or retrain) the global fatigue prediction model.

    Args:
        df         : labeled DataFrame from build_training_dataset()
        model_type : "random_forest" for batch | "sgd" for incremental
        save       : whether to persist the model artifact

    Returns a result dict with keys: model, feature_names, label_encoder,
    train_report, test_report.
    """
    log.info(f"Training global model (type={model_type})...")

    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values
    y = df["label"].values

    # Label encoding
    le = LabelEncoder()
    le.fit(LABEL_CLASSES)
    y_enc = le.transform(y)

    # Time-aware split
    train_df, test_df = time_aware_split(df)
    X_train = train_df[feature_cols].values
    y_train = le.transform(train_df["label"].values)
    X_test  = test_df[feature_cols].values
    y_test  = le.transform(test_df["label"].values)

    # Build sklearn Pipeline (scaler + classifier)
    if model_type == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            class_weight="balanced",   # handles label imbalance
            n_jobs=-1,
            random_state=42,
        )
        pipeline = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        pipeline.fit(X_train, y_train)

    elif model_type == "sgd":
        # SGDClassifier supports partial_fit → incremental learning
        # Note: class_weight must be None for partial_fit compatibility;
        # handle imbalance at data level or via sample weighting instead
        clf = SGDClassifier(
            loss="modified_huber",     # supports predict_proba
            class_weight=None,
            max_iter=1000,
            random_state=42,
        )
        pipeline = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        # Note: for incremental, we fit on all data first; updates come later
        pipeline.fit(X_train, y_train)

    else:
        raise ValueError(f"Unknown model_type: {model_type!r}. Use 'random_forest' or 'sgd'.")

    # Evaluation
    y_pred_train = pipeline.predict(X_train)
    y_pred_test  = pipeline.predict(X_test)
    all_classes  = list(range(len(le.classes_)))
    train_report = classification_report(y_train, y_pred_train, labels=all_classes,
                                         target_names=le.classes_, zero_division=0)
    test_report  = classification_report(y_test, y_pred_test, labels=all_classes,
                                         target_names=le.classes_, zero_division=0)

    log.info(f"Train Report:\n{train_report}")
    log.info(f"Test  Report:\n{test_report}")

    result = {
        "pipeline":      pipeline,
        "feature_names": feature_cols,
        "label_encoder": le,
        "model_type":    model_type,
        "trained_at":    datetime.utcnow().isoformat(),
        "train_report":  train_report,
        "test_report":   test_report,
    }

    if save:
        _save_model(result)

    return result


def _save_model(result: dict) -> None:
    """Persist model artifact and metadata to disk."""
    os.makedirs("artifacts", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(result, f)
    log.info(f"Model saved to {MODEL_PATH}")


def load_model() -> dict:
    """Load persisted model artifact."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No model found at {MODEL_PATH}. Run train_global_model() first."
        )
    with open(MODEL_PATH, "rb") as f:
        result = pickle.load(f)
    log.info(f"Model loaded (trained at {result.get('trained_at', 'unknown')})")
    return result


# ---------------------------------------------------------------------------
# Continuous / Incremental Model Update
# ---------------------------------------------------------------------------

def update_model(
    model_artifact: dict,
    new_features: dict,
    true_label: str,
) -> dict:
    """
    Incrementally update the model with a single new labeled observation.

    Only works when model_type == "sgd" (supports partial_fit).
    For RandomForest, logs a warning and schedules a batch retrain instead.

    Args:
        model_artifact : loaded model dict (from load_model or train_global_model)
        new_features   : flat feature dict for the new window
        true_label     : ground-truth label for this window

    Returns the updated model_artifact.
    """
    pipeline    = model_artifact["pipeline"]
    feature_names = model_artifact["feature_names"]
    le          = model_artifact["label_encoder"]
    model_type  = model_artifact.get("model_type", "random_forest")

    if model_type != "sgd":
        log.warning(
            "Incremental update requested but model is RandomForest. "
            "Queue this sample for batch retraining instead. "
            "Switch to model_type='sgd' for online learning."
        )
        return model_artifact   # no-op; caller should queue for batch retrain

    # Align features to training schema
    x = np.array([[new_features.get(f, 0.0) for f in feature_names]])
    y = le.transform([normalize_label(true_label)])

    # Partial fit on the SGD classifier (inside the pipeline)
    # We must transform through the scaler first
    scaler = pipeline.named_steps["scaler"]
    clf    = pipeline.named_steps["clf"]
    x_scaled = scaler.transform(x)
    all_classes = np.arange(len(le.classes_))
    clf.partial_fit(x_scaled, y, classes=all_classes)

    model_artifact["last_updated"] = datetime.utcnow().isoformat()
    log.info(f"Model incrementally updated with label={true_label}")

    return model_artifact


def schedule_batch_retrain(
    model_artifact: dict,
    update_buffer: list,   # list of (feature_dict, label) tuples
    min_samples: int = 50,
) -> Optional[dict]:
    """
    Perform a batch retrain when the update buffer reaches a threshold.

    Used for RandomForest, which cannot learn incrementally.
    New samples are appended to a rolling training set and the model
    is retrained from scratch.

    Args:
        model_artifact : current loaded model
        update_buffer  : list of (features_dict, label) accumulated since last train
        min_samples    : minimum buffer size before triggering retrain

    Returns updated model_artifact, or None if buffer is too small.
    """
    if len(update_buffer) < min_samples:
        log.info(f"Buffer has {len(update_buffer)}/{min_samples} samples. Retrain not triggered.")
        return None

    log.info(f"Buffer full ({len(update_buffer)} samples). Triggering batch retrain...")

    feature_names = model_artifact["feature_names"]
    le            = model_artifact["label_encoder"]

    rows = []
    for feat_dict, label in update_buffer:
        row = {f: feat_dict.get(f, 0.0) for f in feature_names}
        row["label"] = normalize_label(label)
        rows.append(row)

    new_df = pd.DataFrame(rows)
    return train_global_model(new_df, model_type=model_artifact["model_type"], save=True)
