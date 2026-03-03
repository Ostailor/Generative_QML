#!/usr/bin/env python3
"""Classical active learning baseline suite (T2.4).

Speed controls:
- --optimizer none disables expensive kernel optimization every AL iteration.
- --pool-subsample bounds acquisition scoring cost per iteration.
- --fast applies a conservative simulator-speed preset.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

try:
    from scripts.accel import as_backend_array, detect_accelerator, to_cpu_array, xp_for
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from accel import as_backend_array, detect_accelerator, to_cpu_array, xp_for

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "perovskites_features.parquet"
OUT_JSON = BASE_DIR / "data" / "qml" / "classical_al_metrics.json"


def load_dataset():
    df = pd.read_parquet(DATA_PATH)
    features = [col for col in df.columns if col not in {"material_id", "formula", "spacegroup", "band_gap_eV", "log_band_gap", "is_insulator"}]
    X = df[features].fillna(df[features].median()).values
    y = df["band_gap_eV"].values
    return X, y


def _fit_gp_state_rbf_backend(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    alpha: float,
    noise_level: float,
    length_scale: float,
    accelerator: str,
    use_float32: bool,
) -> dict:
    xp = xp_for(accelerator)
    dtype = np.float32 if use_float32 else np.float64
    x_train = as_backend_array(X_train, accelerator=accelerator, dtype=dtype)
    y_train_backend = as_backend_array(y_train, accelerator=accelerator, dtype=dtype)

    n_train = x_train.shape[0]
    eye = xp.eye(n_train, dtype=dtype)
    gamma = dtype(1.0 / (2.0 * length_scale * length_scale))

    train_norm = xp.sum(x_train * x_train, axis=1, keepdims=True)
    sq_dist = train_norm + train_norm.T - dtype(2.0) * (x_train @ x_train.T)
    sq_dist = xp.maximum(sq_dist, dtype(0.0))
    K = xp.exp(-gamma * sq_dist)
    K = K + dtype(alpha + noise_level) * eye

    alpha_vec = xp.linalg.solve(K, y_train_backend)
    return {
        "K": K,
        "alpha_vec": alpha_vec,
        "x_train": x_train,
        "gamma": gamma,
        "dtype": dtype,
        "accelerator": accelerator,
    }


def _predict_gp_state_rbf_backend(state: dict, X_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xp = xp_for(state["accelerator"])
    x_pred = as_backend_array(
        X_pred,
        accelerator=state["accelerator"],
        dtype=state["dtype"],
    )
    x_train = state["x_train"]
    gamma = state["gamma"]
    dtype = state["dtype"]

    pred_norm = xp.sum(x_pred * x_pred, axis=1, keepdims=True)
    train_norm = xp.sum(x_train * x_train, axis=1, keepdims=True).T
    sq_dist = pred_norm + train_norm - dtype(2.0) * (x_pred @ x_train.T)
    sq_dist = xp.maximum(sq_dist, dtype(0.0))
    k_star = xp.exp(-gamma * sq_dist)

    pred_mean = k_star @ state["alpha_vec"]
    v = xp.linalg.solve(state["K"], k_star.T)
    pred_var = dtype(1.0) - xp.sum(k_star * v.T, axis=1)
    pred_var = xp.maximum(pred_var, dtype(1e-12))
    pred_std = xp.sqrt(pred_var)
    return to_cpu_array(pred_mean), to_cpu_array(pred_std)


def run_simulation(
    random_state: int = 42,
    init_size: int = 50,
    query_batch: int = 25,
    iterations: int = 10,
    *,
    pool_subsample: int | None = None,
    optimizer: str = "fmin_l_bfgs_b",
    max_eval_size: int = 200,
    accelerator: str = "auto",
    gp_engine: str = "auto",
    use_float32: bool = False,
) -> dict:
    effective_accelerator, accelerator_reason = detect_accelerator(accelerator)
    if gp_engine not in {"auto", "sklearn", "backend"}:
        raise ValueError("gp_engine must be one of: auto, sklearn, backend")
    effective_engine = gp_engine
    if gp_engine == "auto":
        effective_engine = "backend" if effective_accelerator == "gpu" else "sklearn"

    X, y = load_dataset()
    X_train, X_pool, y_train, y_pool = train_test_split(X, y, test_size=0.8, random_state=random_state)

    rng = np.random.default_rng(random_state)
    indices = rng.choice(len(X_train), size=init_size, replace=False)
    labeled_X = X_train[indices]
    labeled_y = y_train[indices]

    remaining_mask = np.ones(len(X_pool), dtype=bool)

    model = None
    optimizer_effective = optimizer
    if effective_engine == "sklearn":
        kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-3)
        resolved_optimizer: str | None = None if optimizer.lower() == "none" else optimizer
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-3,
            random_state=random_state,
            optimizer=resolved_optimizer,
        )
    elif optimizer.lower() != "none":
        optimizer_effective = "none_backend_fixed_hyperparams"

    histories = []

    for step in range(iterations):
        available_pool_idx = np.flatnonzero(remaining_mask)
        if len(available_pool_idx) == 0:
            break

        gp_state = None
        if effective_engine == "sklearn":
            assert model is not None
            model.fit(labeled_X, labeled_y)
        else:
            gp_state = _fit_gp_state_rbf_backend(
                labeled_X,
                labeled_y,
                alpha=1e-3,
                noise_level=1e-3,
                length_scale=1.0,
                accelerator=effective_accelerator,
                use_float32=use_float32,
            )
        if pool_subsample is not None and len(available_pool_idx) > pool_subsample:
            candidate_idx = rng.choice(available_pool_idx, size=pool_subsample, replace=False)
        else:
            candidate_idx = available_pool_idx

        if effective_engine == "sklearn":
            assert model is not None
            _, stds = model.predict(X_pool[candidate_idx], return_std=True)
        else:
            assert gp_state is not None
            _, stds = _predict_gp_state_rbf_backend(gp_state, X_pool[candidate_idx])

        # uncertainty sampling (highest std)
        query_size = min(query_batch, len(candidate_idx))
        if query_size == 0:
            break
        std_indices = np.argpartition(stds, -query_size)[-query_size:]
        uncertainty_candidates = candidate_idx[std_indices]

        # Evaluate on hold-out portion of original training set for comparison.
        eval_size = min(max_eval_size, len(X_train))
        eval_idx = rng.choice(len(X_train), size=eval_size, replace=False)
        y_true = y_train[eval_idx]
        if effective_engine == "sklearn":
            assert model is not None
            y_pred = model.predict(X_train[eval_idx])
        else:
            assert gp_state is not None
            y_pred, _ = _predict_gp_state_rbf_backend(gp_state, X_train[eval_idx])
        rmse = mean_squared_error(y_true, y_pred) ** 0.5

        histories.append({
            "iteration": step,
            "rmse": rmse,
            "random_mean_std": float(stds.mean()),
            "uncertainty_max_std": float(stds[std_indices].mean()),
            "pool_evaluated": int(len(candidate_idx)),
        })

        # add uncertainty-selected points to labeled set
        labeled_X = np.vstack([labeled_X, X_pool[uncertainty_candidates]])
        labeled_y = np.concatenate([labeled_y, y_pool[uncertainty_candidates]])
        remaining_mask[uncertainty_candidates] = False

    if not histories:
        raise RuntimeError("Active learning loop produced no iterations. Check init_size/query_batch settings.")

    metrics = {
        "final_rmse": histories[-1]["rmse"],
        "iterations": len(histories),
        "history": histories,
        "runtime_profile": {
            "pool_subsample": pool_subsample,
            "optimizer": optimizer,
            "optimizer_effective": optimizer_effective,
            "max_eval_size": max_eval_size,
            "accelerator_requested": accelerator,
            "accelerator_effective": effective_accelerator,
            "accelerator_reason": accelerator_reason,
            "engine_requested": gp_engine,
            "engine_effective": effective_engine,
            "float32": use_float32,
        },
    }
    OUT_JSON.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--init-size", type=int, default=50)
    parser.add_argument("--query-batch", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--pool-subsample", type=int, default=None)
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
        "--gp-engine",
        choices=["auto", "sklearn", "backend"],
        default="auto",
        help="GP implementation engine. 'backend' enables NumPy/CuPy linear algebra path.",
    )
    parser.add_argument("--float32", action="store_true", help="Cast backend arrays to float32")
    parser.add_argument("--max-eval-size", type=int, default=200)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Enable quick simulator preset (iterations=6, pool-subsample=4000, optimizer=none).",
    )
    parser.add_argument(
        "--fastest",
        action="store_true",
        help="Enable aggressive speed preset (iterations=3, pool-subsample=1000, optimizer=none, max-eval-size=96, query-batch<=16).",
    )
    args = parser.parse_args()

    query_batch = args.query_batch
    iterations = args.iterations
    pool_subsample = args.pool_subsample
    optimizer = args.optimizer
    max_eval_size = args.max_eval_size
    use_float32 = args.float32
    if args.fastest:
        iterations = 3
        pool_subsample = 1000
        optimizer = "none"
        max_eval_size = 96
        query_batch = min(query_batch, 16)
        use_float32 = True
    if args.fast:
        iterations = 6
        if pool_subsample is None:
            pool_subsample = 4000
        optimizer = "none"
        use_float32 = True

    metrics = run_simulation(
        args.random_state,
        init_size=args.init_size,
        query_batch=query_batch,
        iterations=iterations,
        pool_subsample=pool_subsample,
        optimizer=optimizer,
        max_eval_size=max_eval_size,
        accelerator=args.accelerator,
        gp_engine=args.gp_engine,
        use_float32=use_float32,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
