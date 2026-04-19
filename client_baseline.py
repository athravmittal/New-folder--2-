"""
client_baseline.py
------------------
Client-side (edge / frontend) baseline computation.

⚠️  THIS CODE RUNS ON THE USER'S DEVICE — NOT ON THE BACKEND SERVER.

Purpose:
  - Collect behavioral events locally
  - Compute personal baseline (mean + std) from the first N sessions
  - Aggregate current window into features
  - Send ONLY aggregated numbers to the backend
  - Never transmit raw keystrokes, raw mouse data, or personal identifiers

This design provides:
  - Privacy: raw behavioral data never leaves the device
  - Personalization: backend receives deviation features, not raw values
  - Security: no PII in transit or at rest on the server

Usage (pseudocode for JS/Python edge client):
    baseline_engine = UserBaselineEngine()
    baseline_engine.record_keystroke(timestamp, hold_time, ikd)
    baseline_engine.record_mouse(timestamp, x, y)
    ...
    payload = baseline_engine.build_payload()
    requests.post("https://api/predict", json=payload)
"""

import numpy as np
import json
from collections import deque
from typing import Optional
from datetime import datetime

from feature_engineering import (
    extract_keystroke_features,
    extract_mouse_features,
    extract_inactivity_features,
    extract_window_features,
    build_full_feature_vector,
)


# ---------------------------------------------------------------------------
# Raw Event Buffers (in-memory only, never persisted)
# ---------------------------------------------------------------------------

class EventBuffer:
    """
    In-memory ring buffer for raw behavioral events.
    Automatically evicts oldest events beyond max_size.
    Never written to disk or sent to server.
    """

    def __init__(self, max_size: int = 10_000):
        self._buffer = deque(maxlen=max_size)

    def append(self, event: dict) -> None:
        self._buffer.append(event)

    def to_list(self) -> list:
        return list(self._buffer)

    def clear(self) -> None:
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)


# ---------------------------------------------------------------------------
# User Baseline Engine
# ---------------------------------------------------------------------------

class UserBaselineEngine:
    """
    Manages local behavioral baseline for a single user session.

    Flow:
      1. Collect raw events in-memory (never persisted)
      2. After BASELINE_SESSIONS windows, compute mean + std per feature
      3. For each subsequent window, compute features + deviation from baseline
      4. Build a payload dict (aggregated features + baseline stats) to send
         to the backend prediction API

    The backend receives only the payload — it never sees raw events.
    """

    BASELINE_WINDOWS = 6   # number of 5-min windows needed to establish baseline

    def __init__(self):
        # Raw event buffers (cleared per window, never sent to server)
        self._ks_buffer   = EventBuffer()
        self._ms_buffer   = EventBuffer()
        self._inac_buffer = EventBuffer()
        self._win_buffer  = EventBuffer()

        # Baseline state (stored locally only)
        self._baseline_history: list[dict] = []   # per-window feature dicts
        self._baseline_mean: Optional[dict] = None
        self._baseline_std:  Optional[dict] = None
        self._baseline_established = False

        self._window_start = datetime.utcnow()

    # ------------------------------------------------------------------
    # Event Recording (called by OS-level hooks / JS listeners)
    # ------------------------------------------------------------------

    def record_keystroke(self, timestamp: str, hold_time: float, ikd: float) -> None:
        """Record a single keystroke event locally."""
        self._ks_buffer.append({
            "timestamp": timestamp,
            "hold_time": hold_time,
            "ikd":       ikd,
        })

    def record_mouse(self, timestamp: str, x: float, y: float,
                     speed: Optional[float] = None) -> None:
        """Record a mouse movement sample locally."""
        self._ms_buffer.append({
            "timestamp": timestamp,
            "x":  x,
            "y":  y,
            "speed": speed,
        })

    def record_inactivity(self, timestamp: str, duration: float) -> None:
        """Record an inactivity period locally."""
        self._inac_buffer.append({
            "timestamp": timestamp,
            "duration":  duration,
        })

    def record_window_switch(self, timestamp: str, window_title: str,
                              duration: float) -> None:
        """Record a window/app switch locally."""
        self._win_buffer.append({
            "timestamp":    timestamp,
            "window_title": window_title,
            "duration":     duration,
        })

    # ------------------------------------------------------------------
    # Baseline Computation
    # ------------------------------------------------------------------

    def _compute_window_features(self) -> dict:
        """
        Aggregate current in-memory buffers into a feature dict.
        This is purely local — the DataFrames are never sent to the server.
        """
        import pandas as pd

        ks_df   = pd.DataFrame(self._ks_buffer.to_list())   if len(self._ks_buffer)   else None
        ms_df   = pd.DataFrame(self._ms_buffer.to_list())   if len(self._ms_buffer)   else None
        inac_df = pd.DataFrame(self._inac_buffer.to_list()) if len(self._inac_buffer) else None
        win_df  = pd.DataFrame(self._win_buffer.to_list())  if len(self._win_buffer)  else None

        return build_full_feature_vector(
            keystroke_df  = ks_df,
            mouse_df      = ms_df,
            inactivity_df = inac_df,
            window_df     = win_df,
            baseline_mean = None,   # no deviations yet at this stage
            baseline_std  = None,
        )

    def _update_baseline(self, features: dict) -> None:
        """
        Add new window features to the baseline history.
        Recomputes mean + std when BASELINE_WINDOWS samples are available.
        Uses a rolling window of the last 2× BASELINE_WINDOWS for drift adaptation.
        """
        self._baseline_history.append(features)

        # Keep rolling window (adapt to long-term drift)
        max_history = self.BASELINE_WINDOWS * 2
        if len(self._baseline_history) > max_history:
            self._baseline_history = self._baseline_history[-max_history:]

        if len(self._baseline_history) >= self.BASELINE_WINDOWS:
            keys = list(features.keys())
            self._baseline_mean = {
                k: float(np.mean([h.get(k, 0.0) for h in self._baseline_history]))
                for k in keys
            }
            self._baseline_std = {
                k: float(np.std([h.get(k, 0.0) for h in self._baseline_history]) + 1e-6)
                for k in keys
            }
            self._baseline_established = True

    # ------------------------------------------------------------------
    # Payload Builder — the only data that leaves the device
    # ------------------------------------------------------------------

    def end_window(self) -> dict:
        """
        Called at the end of each 5-minute window.

        1. Computes aggregated features from local buffers
        2. Updates baseline if still in warm-up phase
        3. Builds and returns a privacy-safe payload for the backend
        4. Clears raw event buffers

        Returns a payload dict ready to POST to /predict or /feedback.
        """
        features = self._compute_window_features()
        self._update_baseline(features)

        # Clear raw buffers — data is gone after this point
        self._ks_buffer.clear()
        self._ms_buffer.clear()
        self._inac_buffer.clear()
        self._win_buffer.clear()

        payload = {
            "features": features,           # aggregated numbers only
            "window_start": self._window_start.isoformat(),
            "baseline_established": self._baseline_established,
        }

        # Include baseline stats only if established
        if self._baseline_established:
            payload["baseline_mean"] = self._baseline_mean
            payload["baseline_std"]  = self._baseline_std

        self._window_start = datetime.utcnow()
        return payload

    # ------------------------------------------------------------------
    # Persistence (optional, local device only)
    # ------------------------------------------------------------------

    def save_baseline_locally(self, path: str) -> None:
        """
        Save baseline to a local file (device only — never synced to server).
        Useful for persisting between app sessions.
        """
        if not self._baseline_established:
            print("Baseline not yet established. Collect more data first.")
            return
        data = {
            "baseline_mean":    self._baseline_mean,
            "baseline_std":     self._baseline_std,
            "history_count":    len(self._baseline_history),
            "saved_at":         datetime.utcnow().isoformat(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Baseline saved locally to {path}")

    def load_baseline_locally(self, path: str) -> None:
        """Load a previously saved local baseline."""
        with open(path, "r") as f:
            data = json.load(f)
        self._baseline_mean = data["baseline_mean"]
        self._baseline_std  = data["baseline_std"]
        self._baseline_established = True
        print(f"Baseline loaded from {path} (saved at {data.get('saved_at', 'unknown')})")

    def baseline_status(self) -> dict:
        return {
            "established":    self._baseline_established,
            "windows_seen":   len(self._baseline_history),
            "windows_needed": self.BASELINE_WINDOWS,
        }
