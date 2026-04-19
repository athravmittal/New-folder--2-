"""
tests_and_examples.py
---------------------
Integration tests and usage examples demonstrating the full system.

Run:  python tests_and_examples.py

Tests:
  1. Feature extraction from synthetic data
  2. Baseline computation (client-side)
  3. Deviation feature generation
  4. Full training pipeline on synthetic data
  5. Real-time inference
  6. Incremental model update (SGD path)
  7. Batch retrain trigger (RandomForest path)
  8. End-to-end simulate: client → payload → backend predict
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Make imports work from any directory
sys.path.insert(0, os.path.dirname(__file__))

from feature_engineering import (
    extract_keystroke_features,
    extract_mouse_features,
    extract_inactivity_features,
    extract_window_features,
    compute_deviation_features,
    build_full_feature_vector,
    normalize_label,
)
from training_pipeline import (
    build_training_dataset,
    train_global_model,
    update_model,
    schedule_batch_retrain,
    load_model,
)
from inference_pipeline import (
    InferenceEngine,
    predict_realtime,
    submit_label_feedback,
)
from client_baseline import UserBaselineEngine


# ---------------------------------------------------------------------------
# Synthetic Data Generators
# ---------------------------------------------------------------------------

def make_keystroke_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = datetime(2024, 1, 15, 9, 0, 0)
    timestamps = [base + timedelta(seconds=i * 1.5) for i in range(n)]
    return pd.DataFrame({
        "timestamp": timestamps,
        "hold_time": rng.normal(80, 15, n).clip(10),
        "ikd":       rng.normal(120, 30, n).clip(20),
    })


def make_mouse_df(n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = datetime(2024, 1, 15, 9, 0, 0)
    timestamps = [base + timedelta(seconds=i * 0.6) for i in range(n)]
    return pd.DataFrame({
        "timestamp": timestamps,
        "x":     rng.uniform(0, 1920, n),
        "y":     rng.uniform(0, 1080, n),
        "speed": rng.lognormal(3.5, 0.8, n),
    })


def make_inactivity_df(n: int = 10, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = datetime(2024, 1, 15, 9, 2, 0)
    timestamps = [base + timedelta(minutes=i * 0.8) for i in range(n)]
    return pd.DataFrame({
        "timestamp": timestamps,
        "duration":  rng.exponential(20, n),
    })


def make_window_df(n: int = 30, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    apps = ["Chrome", "VSCode", "Slack", "Terminal", "Excel"]
    base = datetime(2024, 1, 15, 9, 0, 0)
    timestamps = [base + timedelta(seconds=i * 10) for i in range(n)]
    return pd.DataFrame({
        "timestamp":    timestamps,
        "window_title": rng.choice(apps, n),
        "duration":     rng.exponential(45, n),
    })


def make_full_training_csvs(tmp_dir: str = "/tmp/fatigue_test") -> dict:
    """Write synthetic CSVs for training pipeline test."""
    os.makedirs(tmp_dir, exist_ok=True)
    base = datetime(2024, 1, 15, 8, 0, 0)
    n_hours = 4

    # Generate 4 hours of synthetic data
    all_ks, all_ms, all_inac, all_win = [], [], [], []
    labels = []

    for hour in range(n_hours):
        hour_base = base + timedelta(hours=hour)
        condition = ["LOW", "AVERAGE", "HIGH"][hour % 3]

        seed_offset = hour * 10

        ks = make_keystroke_df(200, seed=seed_offset)
        ks["timestamp"] = ks["timestamp"] + timedelta(hours=hour)
        if condition == "HIGH":
            ks["ikd"] *= 1.5        # fatigue → slower typing
            ks["hold_time"] *= 1.3
        all_ks.append(ks)

        ms = make_mouse_df(300, seed=seed_offset + 1)
        ms["timestamp"] = ms["timestamp"] + timedelta(hours=hour)
        all_ms.append(ms)

        inac = make_inactivity_df(8, seed=seed_offset + 2)
        inac["timestamp"] = inac["timestamp"] + timedelta(hours=hour)
        if condition == "HIGH":
            inac["duration"] *= 2.0   # more inactivity when fatigued
        all_inac.append(inac)

        win = make_window_df(20, seed=seed_offset + 3)
        win["timestamp"] = win["timestamp"] + timedelta(hours=hour)
        all_win.append(win)

        # One label per 5-min window within this hour
        for minute_offset in range(0, 60, 5):
            ts = hour_base + timedelta(minutes=minute_offset)
            labels.append({"timestamp": ts, "label": condition})

    paths = {
        "keystroke":  os.path.join(tmp_dir, "keystrokes.csv"),
        "mouse":      os.path.join(tmp_dir, "mouse.csv"),
        "inactivity": os.path.join(tmp_dir, "inactivity.csv"),
        "windows":    os.path.join(tmp_dir, "windows.csv"),
        "labels":     os.path.join(tmp_dir, "labels.csv"),
    }

    pd.concat(all_ks).to_csv(paths["keystroke"],  index=False)
    pd.concat(all_ms).to_csv(paths["mouse"],       index=False)
    pd.concat(all_inac).to_csv(paths["inactivity"], index=False)
    pd.concat(all_win).to_csv(paths["windows"],    index=False)
    pd.DataFrame(labels).to_csv(paths["labels"],   index=False)

    return paths


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _header(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def test_feature_extraction():
    _header("TEST 1: Feature Extraction")

    ks   = make_keystroke_df()
    ms   = make_mouse_df()
    inac = make_inactivity_df()
    win  = make_window_df()

    ks_feat   = extract_keystroke_features(ks)
    ms_feat   = extract_mouse_features(ms)
    inac_feat = extract_inactivity_features(inac)
    win_feat  = extract_window_features(win)

    print(f"  Keystroke features  ({len(ks_feat)}):  {list(ks_feat.keys())}")
    print(f"  Mouse features      ({len(ms_feat)}):  {list(ms_feat.keys())}")
    print(f"  Inactivity features ({len(inac_feat)}): {list(inac_feat.keys())}")
    print(f"  Window features     ({len(win_feat)}):  {list(win_feat.keys())}")
    print(f"  typing_speed = {ks_feat['typing_speed']:.1f} kpm")
    print(f"  ikd_mean     = {ks_feat['ikd_mean']:.1f} ms")
    print("  ✓ PASS")


def test_baseline_computation():
    _header("TEST 2: Client-Side Baseline Engine")

    engine = UserBaselineEngine()
    status = engine.baseline_status()
    assert not status["established"], "Baseline should not be established yet"

    # Simulate 6 windows of data
    for window_idx in range(8):
        ks   = make_keystroke_df(100, seed=window_idx)
        ms   = make_mouse_df(200, seed=window_idx + 10)
        inac = make_inactivity_df(5,  seed=window_idx + 20)
        win  = make_window_df(10,     seed=window_idx + 30)

        for _, row in ks.iterrows():
            engine.record_keystroke(row["timestamp"].isoformat(),
                                    row["hold_time"], row["ikd"])
        for _, row in ms.iterrows():
            engine.record_mouse(row["timestamp"].isoformat(),
                                row["x"], row["y"], row.get("speed"))
        for _, row in inac.iterrows():
            engine.record_inactivity(row["timestamp"].isoformat(), row["duration"])
        for _, row in win.iterrows():
            engine.record_window_switch(row["timestamp"].isoformat(),
                                        row["window_title"], row["duration"])

        payload = engine.end_window()

    status = engine.baseline_status()
    assert status["established"], "Baseline should be established after 6 windows"
    print(f"  Baseline established after {status['windows_seen']} windows")
    print(f"  Baseline keys: {list(engine._baseline_mean.keys())[:5]}...")
    print(f"  typing_speed mean = {engine._baseline_mean.get('typing_speed', 0):.2f}")

    payload = engine.end_window()
    assert "baseline_mean" in payload
    assert "baseline_std"  in payload
    assert "features" in payload
    print(f"  Payload keys: {list(payload.keys())}")
    print("  ✓ PASS")


def test_deviation_features():
    _header("TEST 3: Deviation / Z-Score Features")

    current = {"typing_speed": 30.0, "ikd_mean": 180.0, "hold_time_mean": 110.0}
    bsl_mean = {"typing_speed": 55.0, "ikd_mean": 120.0, "hold_time_mean": 80.0}
    bsl_std  = {"typing_speed": 10.0, "ikd_mean": 25.0,  "hold_time_mean": 15.0}

    deviations = compute_deviation_features(current, bsl_mean, bsl_std)
    print(f"  typing_speed_z = {deviations['typing_speed_z']:.2f}  (expected ≈ -2.5)")
    print(f"  ikd_mean_z     = {deviations['ikd_mean_z']:.2f}  (expected ≈ +2.4)")
    assert abs(deviations["typing_speed_z"] - (-2.5)) < 0.01
    print("  ✓ PASS")


def test_training_pipeline():
    _header("TEST 4: Training Pipeline (Batch, RandomForest)")

    paths = make_full_training_csvs()
    print(f"  Training data written to {list(paths.values())[0]}")

    df = build_training_dataset(
        keystroke_path  = paths["keystroke"],
        mouse_path      = paths["mouse"],
        inactivity_path = paths["inactivity"],
        window_path     = paths["windows"],
        label_path      = paths["labels"],
    )
    print(f"  Dataset shape: {df.shape}")
    print(f"  Labels: {df['label'].value_counts().to_dict()}")
    assert len(df) > 0, "Dataset must not be empty"

    os.makedirs("artifacts", exist_ok=True)
    result = train_global_model(df, model_type="random_forest", save=True)
    print(f"  Trained at: {result['trained_at']}")
    print(f"  Features:   {len(result['feature_names'])}")
    print(f"  Test report:\n{result['test_report']}")
    print("  ✓ PASS")


def test_sgd_training():
    _header("TEST 5: Training Pipeline (Incremental, SGDClassifier)")

    paths = make_full_training_csvs()
    df = build_training_dataset(
        keystroke_path  = paths["keystroke"],
        mouse_path      = paths["mouse"],
        inactivity_path = paths["inactivity"],
        window_path     = paths["windows"],
        label_path      = paths["labels"],
    )
    result = train_global_model(df, model_type="sgd", save=False)
    print(f"  SGD model trained. Features: {len(result['feature_names'])}")
    print("  ✓ PASS")


def test_inference():
    _header("TEST 6: Real-Time Inference")

    # Load the model saved in test 4
    eng = InferenceEngine()
    eng.load()

    # Build a synthetic feature vector
    ks_feat   = extract_keystroke_features(make_keystroke_df(100))
    ms_feat   = extract_mouse_features(make_mouse_df(200))
    inac_feat = extract_inactivity_features(make_inactivity_df(5))
    win_feat  = extract_window_features(make_window_df(10))

    features = {**ks_feat, **ms_feat, **inac_feat, **win_feat}

    # Without baseline
    result = eng.predict(features)
    print(f"  Prediction (no baseline): {result['label']} (confidence={result['confidence']:.2f})")

    # With baseline deviation
    bsl_mean = {k: v * 1.2 for k, v in features.items()}
    bsl_std  = {k: max(v * 0.15, 0.01) for k, v in features.items()}
    result2  = eng.predict(features, bsl_mean, bsl_std)
    print(f"  Prediction (with baseline): {result2['label']} (confidence={result2['confidence']:.2f})")
    print(f"  Probabilities: {result2['probabilities']}")
    assert result2["label"] in ["LOW", "AVERAGE", "HIGH"]
    print("  ✓ PASS")


def test_incremental_update():
    _header("TEST 7: Incremental Model Update (SGD)")

    paths = make_full_training_csvs()
    df = build_training_dataset(
        keystroke_path  = paths["keystroke"],
        mouse_path      = paths["mouse"],
        inactivity_path = paths["inactivity"],
        window_path     = paths["windows"],
        label_path      = paths["labels"],
    )
    artifact = train_global_model(df, model_type="sgd", save=False)

    features = {k: 0.0 for k in artifact["feature_names"]}
    features["typing_speed"] = 25.0
    features["ikd_mean"]     = 200.0

    updated = update_model(artifact, features, "HIGH")
    print(f"  Last updated: {updated.get('last_updated', 'N/A')}")
    print("  ✓ PASS")


def test_end_to_end_client_backend():
    _header("TEST 8: End-to-End Client → Payload → Backend")

    # 1. Client side: warm up baseline
    client = UserBaselineEngine()
    for i in range(8):
        for _, row in make_keystroke_df(50, seed=i).iterrows():
            client.record_keystroke(row["timestamp"].isoformat(),
                                    row["hold_time"], row["ikd"])
        for _, row in make_mouse_df(100, seed=i+10).iterrows():
            client.record_mouse(row["timestamp"].isoformat(),
                                row["x"], row["y"], row.get("speed"))
        client.end_window()

    # 2. Client builds one "fatigue" window payload
    for _, row in make_keystroke_df(50, seed=99).iterrows():
        # Simulate high fatigue: very slow typing
        client.record_keystroke(row["timestamp"].isoformat(),
                                row["hold_time"] * 2, row["ikd"] * 2)
    payload = client.end_window()

    print(f"  Client payload keys: {list(payload.keys())}")
    print(f"  Baseline established: {payload['baseline_established']}")
    print(f"  Feature count: {len(payload['features'])}")

    # 3. Backend prediction
    eng = InferenceEngine()
    eng.load()
    result = eng.predict(
        payload["features"],
        payload.get("baseline_mean"),
        payload.get("baseline_std"),
    )
    print(f"  Backend prediction: {result['label']} ({result['confidence']:.2%} confidence)")

    # 4. Feedback loop
    feedback = eng.submit_feedback(
        payload["features"],
        true_label="HIGH",
        baseline_mean=payload.get("baseline_mean"),
        baseline_std=payload.get("baseline_std"),
    )
    print(f"  Feedback status: {feedback}")
    print("  ✓ PASS")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n🧠 Real-Time Fatigue Detection System — Integration Tests")

    test_feature_extraction()
    test_baseline_computation()
    test_deviation_features()
    test_training_pipeline()
    test_sgd_training()
    test_inference()
    test_incremental_update()
    test_end_to_end_client_backend()

    print(f"\n{'='*60}")
    print("  ALL TESTS PASSED ✓")
    print('='*60)
