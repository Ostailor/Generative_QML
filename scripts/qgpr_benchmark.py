#!/usr/bin/env python3
"""Benchmark classical vs quantum-inspired Gaussian process regression (T2.3)."""
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


def evaluate(random_state: int = 42) -> dict:
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Subsample training set for efficiency and to mimic moderate data size
    max_train = 2000
    if len(X_train_scaled) > max_train:
        X_train_scaled = X_train_scaled[:max_train]
        y_train = y_train[:max_train]
    max_test = 500
    if len(X_test_scaled) > max_test:
        X_test_scaled = X_test_scaled[:max_test]
        y_test = y_test[:max_test]

    classical_kernel = DotProduct() + WhiteKernel()
    classical_gpr = GaussianProcessRegressor(kernel=classical_kernel, alpha=1e-3, random_state=random_state)
    classical_gpr.fit(X_train_scaled, y_train)
    y_pred_classical, y_std_classical = classical_gpr.predict(X_test_scaled, return_std=True)

    quantum_kernel = 1.0 * RBF(length_scale=0.5)
    quantum_gpr = GaussianProcessRegressor(kernel=quantum_kernel, alpha=1e-3, random_state=random_state)
    quantum_gpr.fit(X_train_scaled, y_train)
    y_pred_quantum, y_std_quantum = quantum_gpr.predict(X_test_scaled, return_std=True)

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
    }

    OUT_JSON.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    metrics = evaluate(args.random_state)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
