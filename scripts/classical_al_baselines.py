#!/usr/bin/env python3
"""Classical active learning baseline suite (T2.4)."""
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

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "perovskites_features.parquet"
OUT_JSON = BASE_DIR / "data" / "qml" / "classical_al_metrics.json"


def load_dataset():
    df = pd.read_parquet(DATA_PATH)
    features = [col for col in df.columns if col not in {"material_id", "formula", "spacegroup", "band_gap_eV", "log_band_gap", "is_insulator"}]
    X = df[features].fillna(df[features].median()).values
    y = df["band_gap_eV"].values
    return X, y


def run_simulation(random_state: int = 42, init_size: int = 50, query_batch: int = 25, iterations: int = 10) -> dict:
    X, y = load_dataset()
    X_train, X_pool, y_train, y_pool = train_test_split(X, y, test_size=0.8, random_state=random_state)

    rng = np.random.default_rng(random_state)
    indices = rng.choice(len(X_train), size=init_size, replace=False)
    labeled_X = X_train[indices]
    labeled_y = y_train[indices]

    remaining_idx = [i for i in range(len(X_pool))]

    kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-3)
    model = GaussianProcessRegressor(kernel=kernel, alpha=1e-3, random_state=random_state)

    histories = []

    for step in range(iterations):
        model.fit(labeled_X, labeled_y)
        preds, stds = model.predict(X_pool[remaining_idx], return_std=True)

        # random acquisition
        random_candidates = rng.choice(remaining_idx, size=query_batch, replace=False)
        # uncertainty sampling (highest std)
        std_indices = np.argsort(stds)[-query_batch:]
        uncertainty_candidates = [remaining_idx[i] for i in std_indices]

        # evaluate on hold-out portion of original training set for comparison
        eval_idx = rng.choice(len(X_train), size=200, replace=False)
        y_true = y_train[eval_idx]
        y_pred = model.predict(X_train[eval_idx])
        rmse = mean_squared_error(y_true, y_pred) ** 0.5

        histories.append({
            "iteration": step,
            "rmse": rmse,
            "random_mean_std": float(stds.mean()),
            "uncertainty_max_std": float(stds[std_indices].mean())
        })

        # add uncertainty-selected points to labeled set
        new_indices = uncertainty_candidates
        labeled_X = np.vstack([labeled_X, X_pool[new_indices]])
        labeled_y = np.concatenate([labeled_y, y_pool[new_indices]])
        remaining_idx = [idx for idx in remaining_idx if idx not in new_indices]

    metrics = {
        "final_rmse": histories[-1]["rmse"],
        "iterations": iterations,
        "history": histories,
    }
    OUT_JSON.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    metrics = run_simulation(args.random_state)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
