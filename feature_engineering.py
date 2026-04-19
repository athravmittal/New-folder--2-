"""
feature_engineering.py
-----------------------
Reusable feature extraction functions for behavioral signals.
Designed to be called both during training (historical data)
and at inference time (real-time aggregated windows).

All functions operate on a time-windowed DataFrame and return
a flat dict of numerical features — no raw data is retained.
"""

import numpy as np
import pandas as pd
from typing import Optional


# ---------------------------------------------------------------------------
# Keystroke Features
# ---------------------------------------------------------------------------

def extract_keystroke_features(df: pd.DataFrame) -> dict:
    """
    Extract typing-behavior features from a keystroke event DataFrame.

    Expected columns:
        - timestamp  : event time (datetime or float seconds)
        - hold_time  : key-press duration in milliseconds
        - ikd        : inter-key delay in milliseconds (time between keydown events)

    Returns a flat dict of aggregated numerical features.
    """
    if df is None or df.empty:
        return _empty_keystroke_features()

    features = {}

    # --- Typing Speed (keystrokes per minute) ---
    if "timestamp" in df.columns and len(df) > 1:
        duration_minutes = _duration_minutes(df["timestamp"])
        features["typing_speed"] = len(df) / duration_minutes if duration_minutes > 0 else 0.0
    else:
        features["typing_speed"] = 0.0

    # --- Inter-Key Delay (IKD) ---
    if "ikd" in df.columns:
        ikd = df["ikd"].dropna()
        features["ikd_mean"]   = float(ikd.mean())   if len(ikd) > 0 else 0.0
        features["ikd_std"]    = float(ikd.std())    if len(ikd) > 1 else 0.0
        features["ikd_median"] = float(ikd.median()) if len(ikd) > 0 else 0.0
        features["ikd_p90"]    = float(ikd.quantile(0.90)) if len(ikd) > 0 else 0.0
    else:
        features.update({"ikd_mean": 0.0, "ikd_std": 0.0,
                          "ikd_median": 0.0, "ikd_p90": 0.0})

    # --- Key Hold Time ---
    if "hold_time" in df.columns:
        ht = df["hold_time"].dropna()
        features["hold_time_mean"]   = float(ht.mean())   if len(ht) > 0 else 0.0
        features["hold_time_std"]    = float(ht.std())    if len(ht) > 1 else 0.0
        features["hold_time_median"] = float(ht.median()) if len(ht) > 0 else 0.0
    else:
        features.update({"hold_time_mean": 0.0, "hold_time_std": 0.0,
                          "hold_time_median": 0.0})

    # --- Error Proxy: very short hold times suggest accidental keys ---
    if "hold_time" in df.columns:
        ht = df["hold_time"].dropna()
        features["short_hold_ratio"] = float((ht < 50).mean()) if len(ht) > 0 else 0.0
    else:
        features["short_hold_ratio"] = 0.0

    return features


def _empty_keystroke_features() -> dict:
    return {k: 0.0 for k in [
        "typing_speed", "ikd_mean", "ikd_std", "ikd_median", "ikd_p90",
        "hold_time_mean", "hold_time_std", "hold_time_median", "short_hold_ratio"
    ]}


# ---------------------------------------------------------------------------
# Mouse Features
# ---------------------------------------------------------------------------

def extract_mouse_features(df: pd.DataFrame) -> dict:
    """
    Extract mouse-movement features from a movement event DataFrame.

    Expected columns:
        - timestamp : event time
        - x, y      : screen coordinates
        - speed     : pixels/second (pre-computed or derived here)
    """
    if df is None or df.empty:
        return _empty_mouse_features()

    features = {}

    # --- Compute speed if not present ---
    if "speed" not in df.columns and all(c in df.columns for c in ["x", "y", "timestamp"]):
        df = df.sort_values("timestamp").copy()
        dx = df["x"].diff()
        dy = df["y"].diff()
        dt = pd.to_datetime(df["timestamp"]).diff().dt.total_seconds().replace(0, np.nan)
        df["speed"] = np.sqrt(dx**2 + dy**2) / dt

    if "speed" in df.columns:
        spd = df["speed"].dropna()
        features["mouse_speed_mean"]   = float(spd.mean())   if len(spd) > 0 else 0.0
        features["mouse_speed_std"]    = float(spd.std())    if len(spd) > 1 else 0.0
        features["mouse_speed_median"] = float(spd.median()) if len(spd) > 0 else 0.0
        features["mouse_speed_p90"]    = float(spd.quantile(0.90)) if len(spd) > 0 else 0.0
        # Tremor proxy: high std relative to mean suggests shaky movement
        features["mouse_tremor_ratio"] = (
            features["mouse_speed_std"] / (features["mouse_speed_mean"] + 1e-6)
        )
    else:
        features.update(_empty_mouse_features())

    return features


def _empty_mouse_features() -> dict:
    return {k: 0.0 for k in [
        "mouse_speed_mean", "mouse_speed_std",
        "mouse_speed_median", "mouse_speed_p90", "mouse_tremor_ratio"
    ]}


# ---------------------------------------------------------------------------
# Inactivity Features
# ---------------------------------------------------------------------------

def extract_inactivity_features(df: pd.DataFrame) -> dict:
    """
    Extract inactivity features from an inactivity-event DataFrame.

    Expected columns:
        - duration  : inactivity gap in seconds
        - timestamp : start of inactivity period
    """
    if df is None or df.empty:
        return _empty_inactivity_features()

    features = {}

    if "duration" in df.columns:
        dur = df["duration"].dropna()
        features["inactivity_count"]        = float(len(dur))
        features["inactivity_duration_mean"] = float(dur.mean())   if len(dur) > 0 else 0.0
        features["inactivity_duration_max"]  = float(dur.max())    if len(dur) > 0 else 0.0
        features["inactivity_duration_sum"]  = float(dur.sum())    if len(dur) > 0 else 0.0
        # Long pause ratio: pauses > 30 s indicate zoning out
        features["long_pause_ratio"] = float((dur > 30).mean()) if len(dur) > 0 else 0.0
    else:
        features.update(_empty_inactivity_features())

    return features


def _empty_inactivity_features() -> dict:
    return {k: 0.0 for k in [
        "inactivity_count", "inactivity_duration_mean",
        "inactivity_duration_max", "inactivity_duration_sum", "long_pause_ratio"
    ]}


# ---------------------------------------------------------------------------
# Active Window Features
# ---------------------------------------------------------------------------

def extract_window_features(df: pd.DataFrame) -> dict:
    """
    Extract focus/attention features from an active-window event DataFrame.

    Expected columns:
        - timestamp    : time of window switch
        - window_title : name of the active application/window
        - duration     : time spent in window (seconds)
    """
    if df is None or df.empty:
        return _empty_window_features()

    features = {}

    # --- Switch rate ---
    if "timestamp" in df.columns and len(df) > 1:
        duration_minutes = _duration_minutes(df["timestamp"])
        features["window_switch_rate"] = len(df) / duration_minutes if duration_minutes > 0 else 0.0
    else:
        features["window_switch_rate"] = 0.0

    # --- Unique apps visited ---
    if "window_title" in df.columns:
        features["unique_windows"] = float(df["window_title"].nunique())
    else:
        features["unique_windows"] = 0.0

    # --- Time-per-window stats ---
    if "duration" in df.columns:
        dur = df["duration"].dropna()
        features["window_duration_mean"] = float(dur.mean()) if len(dur) > 0 else 0.0
        features["window_duration_std"]  = float(dur.std())  if len(dur) > 1 else 0.0
    else:
        features.update({"window_duration_mean": 0.0, "window_duration_std": 0.0})

    return features


def _empty_window_features() -> dict:
    return {k: 0.0 for k in [
        "window_switch_rate", "unique_windows",
        "window_duration_mean", "window_duration_std"
    ]}


# ---------------------------------------------------------------------------
# Baseline-Aware (Deviation) Features  — NEW
# ---------------------------------------------------------------------------

def compute_deviation_features(
    current: dict,
    baseline_mean: dict,
    baseline_std: dict,
    feature_keys: Optional[list] = None,
) -> dict:
    """
    Compute z-score deviation features relative to a user's personal baseline.

    The baseline (mean + std) is computed CLIENT-SIDE and sent alongside the
    current window features. The backend never stores it.

    Args:
        current       : feature dict for the current time window
        baseline_mean : per-feature mean computed over user's baseline period
        baseline_std  : per-feature std  computed over user's baseline period
        feature_keys  : which features to compute deviations for
                        (defaults to all numeric keys in current)

    Returns a dict with additional `*_z` keys for each deviation feature.
    """
    if feature_keys is None:
        feature_keys = list(current.keys())

    deviation = {}
    for key in feature_keys:
        val  = current.get(key, 0.0)
        mean = baseline_mean.get(key, 0.0)
        std  = baseline_std.get(key, 1.0)

        # Avoid division by zero; if std ≈ 0 the user is very consistent
        safe_std = std if std > 1e-6 else 1.0
        deviation[f"{key}_z"] = (val - mean) / safe_std

    return deviation


def build_full_feature_vector(
    keystroke_df:   Optional[pd.DataFrame],
    mouse_df:       Optional[pd.DataFrame],
    inactivity_df:  Optional[pd.DataFrame],
    window_df:      Optional[pd.DataFrame],
    baseline_mean:  Optional[dict] = None,
    baseline_std:   Optional[dict] = None,
) -> dict:
    """
    Combine all modality features into a single flat feature vector.

    If baseline dicts are provided, also appends z-score deviation features.
    This is the single entry-point for feature assembly used by both the
    training pipeline and the inference pipeline.
    """
    raw = {}
    raw.update(extract_keystroke_features(keystroke_df))
    raw.update(extract_mouse_features(mouse_df))
    raw.update(extract_inactivity_features(inactivity_df))
    raw.update(extract_window_features(window_df))

    features = dict(raw)  # copy

    if baseline_mean is not None and baseline_std is not None:
        deviations = compute_deviation_features(raw, baseline_mean, baseline_std)
        features.update(deviations)

    return features


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _duration_minutes(timestamps: pd.Series) -> float:
    """Return the span of a timestamp series in minutes."""
    try:
        ts = pd.to_datetime(timestamps).sort_values()
        delta = (ts.iloc[-1] - ts.iloc[0]).total_seconds()
        return max(delta / 60.0, 1e-6)
    except Exception:
        return 1.0


def normalize_label(label: str) -> str:
    """
    Canonicalize condition labels to LOW / AVERAGE / HIGH.
    Handles common variants from annotation tools.
    """
    mapping = {
        "low":     "LOW",
        "l":       "LOW",
        "avg":     "AVERAGE",
        "average": "AVERAGE",
        "med":     "AVERAGE",
        "medium":  "AVERAGE",
        "high":    "HIGH",
        "h":       "HIGH",
    }
    return mapping.get(str(label).strip().lower(), "AVERAGE")
