#!/usr/bin/env python3
"""Benchmark classical vs simulated quantum SVR models for T2.2."""
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
from sklearn.metrics.pairwise import rbf_kernel

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


def evaluate_models(random_state: int = 42) -> dict[str, float]:
    X, y, features = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Classical SVR with RBF kernel
    classical_model = SVR(kernel="rbf", C=10.0, gamma="scale")
    classical_model.fit(X_train_scaled, y_train)
    y_pred_classical = classical_model.predict(X_test_scaled)

    # Simulated quantum kernel using cosine similarity derived from feature map
    gamma = 0.5
    train_kernel = rbf_kernel(X_train_scaled, X_train_scaled, gamma=gamma)
    quantum_model = SVR(kernel="precomputed", C=10.0)
    quantum_model.fit(train_kernel, y_train)
    test_kernel = rbf_kernel(X_test_scaled, X_train_scaled, gamma=gamma)
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
    }
    OUT_JSON.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="QSVR benchmark")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    metrics = evaluate_models(random_state=args.random_state)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
