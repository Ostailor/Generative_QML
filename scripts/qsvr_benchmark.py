#!/usr/bin/env python3
"""Benchmark classical vs simulated quantum SVR models for T2.2.

Speed controls:
- --max-train / --max-test limit kernel matrix sizes.
- --float32 reduces memory bandwidth for large sweeps.
- --fast applies a conservative preset tuned for quick simulator studies.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

try:
    from scripts.accel import detect_accelerator, rbf_kernel_backend
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from accel import detect_accelerator, rbf_kernel_backend

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "perovskites_features.parquet"
OUT_JSON = BASE_DIR / "data" / "qml" / "qsvr_metrics.json"
OUT_CSV = BASE_DIR / "data" / "qml" / "qsvr_predictions.csv"


def load_dataset() -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = pd.read_parquet(DATA_PATH)
    features = [col for col in df.columns if col not in {"material_id", "formula", "spacegroup", "band_gap_eV", "log_band_gap", "is_insulator"}]
    X = df[features].fillna(df[features].median()).values
    y = df["band_gap_eV"].values
    return X, y, features


def _subsample(
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_size: int | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if max_size is None or len(X) <= max_size:
        return X, y
    idx = rng.choice(len(X), size=max_size, replace=False)
    return X[idx], y[idx]


def evaluate_models(
    random_state: int = 42,
    *,
    max_train: int | None = None,
    max_test: int | None = None,
    use_float32: bool = False,
    accelerator: str = "auto",
) -> dict[str, float]:
    effective_accelerator, accelerator_reason = detect_accelerator(accelerator)
    X, y, features = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    rng = np.random.default_rng(random_state)
    X_train, y_train = _subsample(X_train, y_train, max_size=max_train, rng=rng)
    X_test, y_test = _subsample(X_test, y_test, max_size=max_test, rng=rng)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    if use_float32:
        X_train_scaled = X_train_scaled.astype(np.float32, copy=False)
        X_test_scaled = X_test_scaled.astype(np.float32, copy=False)

    # Classical SVR with RBF kernel
    classical_model = SVR(kernel="rbf", C=10.0, gamma="scale")
    classical_model.fit(X_train_scaled, y_train)
    y_pred_classical = classical_model.predict(X_test_scaled)

    # Simulated quantum kernel using cosine similarity derived from feature map
    gamma = 0.5
    train_kernel = rbf_kernel_backend(
        X_train_scaled,
        X_train_scaled,
        gamma=gamma,
        accelerator=effective_accelerator,
    )
    quantum_model = SVR(kernel="precomputed", C=10.0)
    quantum_model.fit(train_kernel, y_train)
    test_kernel = rbf_kernel_backend(
        X_test_scaled,
        X_train_scaled,
        gamma=gamma,
        accelerator=effective_accelerator,
    )
    y_pred_quantum = quantum_model.predict(test_kernel)

    rmse_classical = mean_squared_error(y_test, y_pred_classical) ** 0.5
    mae_classical = mean_absolute_error(y_test, y_pred_classical)

    rmse_quantum = mean_squared_error(y_test, y_pred_quantum) ** 0.5
    mae_quantum = mean_absolute_error(y_test, y_pred_quantum)

    relative_gap = (rmse_quantum - rmse_classical) / rmse_classical

    pd.DataFrame(
        {
            "y_true": y_test,
            "y_pred_classical": y_pred_classical,
            "y_pred_quantum": y_pred_quantum,
        }
    ).to_csv(OUT_CSV, index=False)

    metrics = {
        "rmse_classical": rmse_classical,
        "mae_classical": mae_classical,
        "rmse_quantum": rmse_quantum,
        "mae_quantum": mae_quantum,
        "relative_gap": relative_gap,
        "features": features,
        "runtime_profile": {
            "max_train": max_train,
            "max_test": max_test,
            "float32": use_float32,
            "accelerator_requested": accelerator,
            "accelerator_effective": effective_accelerator,
            "accelerator_reason": accelerator_reason,
        },
    }
    OUT_JSON.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="QSVR benchmark")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--float32", action="store_true", help="Cast scaled features to float32")
    parser.add_argument(
        "--accelerator",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="Kernel acceleration mode (auto selects GPU if available).",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Enable quick simulator preset (max-train=6000, max-test=2000, float32).",
    )
    parser.add_argument(
        "--fastest",
        action="store_true",
        help="Enable aggressive speed preset (max-train=3000, max-test=1000, float32).",
    )
    args = parser.parse_args()

    max_train = args.max_train
    max_test = args.max_test
    use_float32 = args.float32
    if args.fastest:
        max_train = 3000
        max_test = 1000
        use_float32 = True
    if args.fast:
        if max_train is None:
            max_train = 6000
        if max_test is None:
            max_test = 2000
        use_float32 = True

    metrics = evaluate_models(
        random_state=args.random_state,
        max_train=max_train,
        max_test=max_test,
        use_float32=use_float32,
        accelerator=args.accelerator,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
