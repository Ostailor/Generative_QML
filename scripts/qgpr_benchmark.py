#!/usr/bin/env python3
"""Benchmark classical vs quantum-inspired Gaussian process regression (T2.3).

Speed controls:
- --optimizer none disables costly kernel hyperparameter optimization.
- --max-train / --max-test bound cubic GPR costs.
- --float32 lowers memory pressure for large sweeps.
- --fast applies a conservative quick preset for simulator-heavy studies.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from scripts.accel import (
        as_backend_array,
        detect_accelerator,
        to_cpu_array,
        xp_for,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from accel import as_backend_array, detect_accelerator, to_cpu_array, xp_for

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "perovskites_features.parquet"
OUT_JSON = BASE_DIR / "data" / "qml" / "qgpr_metrics.json"


def load_data():
    df = pd.read_parquet(DATA_PATH)
    features = [col for col in df.columns if col not in {"material_id", "formula", "spacegroup", "band_gap_eV", "log_band_gap", "is_insulator"}]
    X = df[features].fillna(df[features].median()).values
    y = df["band_gap_eV"].values
    return X, y


def coverage_score(y_true, y_pred, y_std, alpha: float = 0.05) -> float:
    z = 1.96  # approximate for 95%
    lower = y_pred - z * y_std
    upper = y_pred + z * y_std
    within = ((y_true >= lower) & (y_true <= upper)).mean()
    return within


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


def _fit_predict_backend_gpr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    kernel_kind: str,
    alpha: float,
    accelerator: str,
    use_float32: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Backend GPR solve for CPU/GPU with fixed kernel hyperparameters."""
    xp = xp_for(accelerator)
    dtype = np.float32 if use_float32 else np.float64

    x_train = as_backend_array(X_train, accelerator=accelerator, dtype=dtype)
    x_test = as_backend_array(X_test, accelerator=accelerator, dtype=dtype)
    y_train_backend = as_backend_array(y_train, accelerator=accelerator, dtype=dtype)

    n_train = x_train.shape[0]
    eye = xp.eye(n_train, dtype=dtype)

    if kernel_kind == "classical_dot":
        # Approximate DotProduct() + WhiteKernel() with sigma0=1 and unit white noise.
        noise_level = dtype(1.0)
        k_train = (x_train @ x_train.T) + dtype(1.0)
        k_train = k_train + (dtype(alpha) + noise_level) * eye
        k_star = (x_test @ x_train.T) + dtype(1.0)
        k_ss_diag = xp.sum(x_test * x_test, axis=1) + dtype(1.0)
    elif kernel_kind == "quantum_rbf":
        # Match sklearn RBF(length_scale=0.5): gamma = 1 / (2 * l^2) = 2.0
        gamma = dtype(2.0)
        left_norm = xp.sum(x_train * x_train, axis=1, keepdims=True)
        sq_dist_train = left_norm + left_norm.T - dtype(2.0) * (x_train @ x_train.T)
        sq_dist_train = xp.maximum(sq_dist_train, dtype(0.0))
        k_train = xp.exp(-gamma * sq_dist_train) + dtype(alpha) * eye

        test_norm = xp.sum(x_test * x_test, axis=1, keepdims=True)
        train_norm = xp.sum(x_train * x_train, axis=1, keepdims=True).T
        sq_dist_star = test_norm + train_norm - dtype(2.0) * (x_test @ x_train.T)
        sq_dist_star = xp.maximum(sq_dist_star, dtype(0.0))
        k_star = xp.exp(-gamma * sq_dist_star)
        k_ss_diag = xp.ones(x_test.shape[0], dtype=dtype)
    else:
        raise ValueError(f"Unknown kernel kind: {kernel_kind}")

    # Solve posterior mean.
    alpha_vec = xp.linalg.solve(k_train, y_train_backend)
    pred_mean = k_star @ alpha_vec

    # Posterior variance diagonal.
    v = xp.linalg.solve(k_train, k_star.T)
    pred_var = k_ss_diag - xp.sum(k_star * v.T, axis=1)
    pred_var = xp.maximum(pred_var, dtype(1e-12))
    pred_std = xp.sqrt(pred_var)

    return to_cpu_array(pred_mean), to_cpu_array(pred_std)


def evaluate(
    random_state: int = 42,
    *,
    max_train: int = 2000,
    max_test: int = 500,
    optimizer: str = "fmin_l_bfgs_b",
    use_float32: bool = False,
    accelerator: str = "auto",
    gpr_engine: str = "auto",
) -> dict:
    effective_accelerator, accelerator_reason = detect_accelerator(accelerator)
    if gpr_engine not in {"auto", "sklearn", "backend"}:
        raise ValueError("gpr_engine must be one of: auto, sklearn, backend")
    effective_engine = gpr_engine
    if gpr_engine == "auto":
        effective_engine = "backend" if effective_accelerator == "gpu" else "sklearn"

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    rng = np.random.default_rng(random_state)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    if use_float32:
        X_train_scaled = X_train_scaled.astype(np.float32, copy=False)
        X_test_scaled = X_test_scaled.astype(np.float32, copy=False)

    X_train_scaled, y_train = _subsample(X_train_scaled, y_train, max_size=max_train, rng=rng)
    X_test_scaled, y_test = _subsample(X_test_scaled, y_test, max_size=max_test, rng=rng)

    optimizer_effective = optimizer
    if effective_engine == "sklearn":
        resolved_optimizer: str | None = None if optimizer.lower() == "none" else optimizer
        classical_kernel = DotProduct() + WhiteKernel()
        classical_gpr = GaussianProcessRegressor(
            kernel=classical_kernel,
            alpha=1e-3,
            random_state=random_state,
            optimizer=resolved_optimizer,
        )
        classical_gpr.fit(X_train_scaled, y_train)
        y_pred_classical, y_std_classical = classical_gpr.predict(X_test_scaled, return_std=True)

        quantum_kernel = 1.0 * RBF(length_scale=0.5)
        quantum_gpr = GaussianProcessRegressor(
            kernel=quantum_kernel,
            alpha=1e-3,
            random_state=random_state,
            optimizer=resolved_optimizer,
        )
        quantum_gpr.fit(X_train_scaled, y_train)
        y_pred_quantum, y_std_quantum = quantum_gpr.predict(X_test_scaled, return_std=True)
    else:
        if optimizer.lower() != "none":
            optimizer_effective = "none_backend_fixed_hyperparams"
        y_pred_classical, y_std_classical = _fit_predict_backend_gpr(
            X_train_scaled,
            y_train,
            X_test_scaled,
            kernel_kind="classical_dot",
            alpha=1e-3,
            accelerator=effective_accelerator,
            use_float32=use_float32,
        )
        y_pred_quantum, y_std_quantum = _fit_predict_backend_gpr(
            X_train_scaled,
            y_train,
            X_test_scaled,
            kernel_kind="quantum_rbf",
            alpha=1e-3,
            accelerator=effective_accelerator,
            use_float32=use_float32,
        )

    rmse_classical = mean_squared_error(y_test, y_pred_classical) ** 0.5
    rmse_quantum = mean_squared_error(y_test, y_pred_quantum) ** 0.5

    coverage_classical = coverage_score(y_test, y_pred_classical, y_std_classical)
    coverage_quantum = coverage_score(y_test, y_pred_quantum, y_std_quantum)

    coverage_gap = coverage_quantum - 0.95

    metrics = {
        "rmse_classical": rmse_classical,
        "rmse_quantum": rmse_quantum,
        "coverage_quantum": coverage_quantum,
        "coverage_gap": coverage_gap,
        "runtime_profile": {
            "max_train": max_train,
            "max_test": max_test,
            "optimizer": optimizer,
            "optimizer_effective": optimizer_effective,
            "float32": use_float32,
            "accelerator_requested": accelerator,
            "accelerator_effective": effective_accelerator,
            "accelerator_reason": accelerator_reason,
            "engine_requested": gpr_engine,
            "engine_effective": effective_engine,
        },
    }

    OUT_JSON.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-train", type=int, default=2000)
    parser.add_argument("--max-test", type=int, default=500)
    parser.add_argument(
        "--optimizer",
        default="fmin_l_bfgs_b",
        help="GPR optimizer; set to 'none' to skip hyperparameter optimization.",
    )
    parser.add_argument(
        "--accelerator",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="Acceleration mode (auto selects GPU backend if available).",
    )
    parser.add_argument(
        "--gpr-engine",
        choices=["auto", "sklearn", "backend"],
        default="auto",
        help="GPR implementation engine. 'backend' enables NumPy/CuPy linear algebra path.",
    )
    parser.add_argument("--float32", action="store_true", help="Cast scaled features to float32")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Enable quick simulator preset (max-train=1200, max-test=300, optimizer=none, float32).",
    )
    parser.add_argument(
        "--fastest",
        action="store_true",
        help="Enable aggressive speed preset (max-train=800, max-test=200, optimizer=none, float32).",
    )
    args = parser.parse_args()

    max_train = args.max_train
    max_test = args.max_test
    optimizer = args.optimizer
    use_float32 = args.float32
    if args.fastest:
        max_train = 800
        max_test = 200
        optimizer = "none"
        use_float32 = True
    if args.fast:
        max_train = 1200
        max_test = 300
        optimizer = "none"
        use_float32 = True

    metrics = evaluate(
        args.random_state,
        max_train=max_train,
        max_test=max_test,
        optimizer=optimizer,
        use_float32=use_float32,
        accelerator=args.accelerator,
        gpr_engine=args.gpr_engine,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
