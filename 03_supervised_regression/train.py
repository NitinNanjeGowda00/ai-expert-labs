"""
Segment 03 - Supervised Regression (Model Comparison + Plot)

Dataset: California Housing (sklearn)
Models: Ridge vs RandomForestRegressor

Outputs:
  artifacts/seg03/
    best_model.joblib
    metrics.json
    pred_vs_actual.png
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

# Ensure repo root is on PYTHONPATH so "setup" can be imported
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from setup.repro import seed_everything, basic_run_id

ART_DIR = Path("artifacts/seg03")


def rmse(y_true, y_pred) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def make_pipeline(model):
    # Numeric-only dataset, but keep a robust numeric pipeline
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_pipe, slice(0, None))],
        remainder="drop",
    )

    return Pipeline([("preprocess", preprocessor), ("model", model)])


def evaluate(name, pipe, X_test, y_test) -> dict:
    pred = pipe.predict(X_test)
    return {
        "model": name,
        "rmse": rmse(y_test, pred),
        "mae": float(mean_absolute_error(y_test, pred)),
        "r2": float(r2_score(y_test, pred)),
    }


def save_pred_vs_actual(pipe, X_test, y_test, outpath: Path) -> None:
    pred = pipe.predict(X_test)

    plt.figure()
    plt.scatter(y_test, pred, s=12)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs Actual")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def main() -> None:
    seed_everything(42)
    ART_DIR.mkdir(parents=True, exist_ok=True)

    run_id = basic_run_id("seg03")
    print(f"Run: {run_id}")

    data = fetch_california_housing()
    X = data.data
    y = data.target  # median house value in 100k$

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    candidates = [
        ("ridge", Ridge(alpha=1.0, random_state=42)),
        ("rf", RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)),
    ]

    results = []
    fitted = {}

    for name, model in candidates:
        pipe = make_pipeline(model)
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        res = evaluate(name, pipe, X_test, y_test)
        results.append(res)
        print(json.dumps(res, indent=2))

    # Best: lowest RMSE, tie-breaker higher R2
    results_sorted = sorted(results, key=lambda d: (d["rmse"], -d["r2"]))
    best = results_sorted[0]
    best_name = best["model"]
    best_pipe = fitted[best_name]

    save_pred_vs_actual(best_pipe, X_test, y_test, ART_DIR / "pred_vs_actual.png")

    out = {
        "run_id": run_id,
        "results": results_sorted,
        "best_model": best_name,
        "best_metrics": best,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "target_note": "Target is median house value in 100k$ units (sklearn California Housing).",
    }

    with open(ART_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    dump(best_pipe, ART_DIR / "best_model.joblib")

    print("\nSaved to:", ART_DIR)
    print("Best model:", best_name)
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
