"""
Segment 05 - Time Series Forecasting (Baselines + Lag Features)

We generate a synthetic time series:
- trend + seasonality + noise
Then perform a time-based split and compare:
- Naive forecast (last value)
- Moving average forecast
- Linear regression on lag features

Outputs:
  artifacts/seg05/
    metrics.json
    forecast_plot.png
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
from sklearn.linear_model import LinearRegression

from setup.repro import seed_everything, basic_run_id

ART_DIR = Path("artifacts/seg05")


def generate_series(n: int = 600, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n)

    trend = 0.01 * t
    seasonality = 0.8 * np.sin(2 * np.pi * t / 30) + 0.4 * np.sin(2 * np.pi * t / 7)
    noise = rng.normal(0, 0.25, size=n)

    y = 2.0 + trend + seasonality + noise
    return y


def make_supervised(y: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a 1D series into supervised learning format:
      X[i] = [y[i-lags], ..., y[i-1]]
      target = y[i]
    """
    X, t = [], []
    for i in range(lags, len(y)):
        X.append(y[i - lags : i])
        t.append(y[i])
    return np.asarray(X), np.asarray(t)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def naive_forecast(y_train: np.ndarray, horizon: int) -> np.ndarray:
    # Predict the last observed value for all future steps
    return np.full(shape=horizon, fill_value=y_train[-1], dtype=float)


def moving_average_forecast(y_train: np.ndarray, horizon: int, window: int = 7) -> np.ndarray:
    # Use mean of last `window` points as constant forecast
    w = min(window, len(y_train))
    return np.full(shape=horizon, fill_value=float(np.mean(y_train[-w:])), dtype=float)


def autoregressive_lr_forecast(
    y_train: np.ndarray,
    horizon: int,
    lags: int = 14,
) -> np.ndarray:
    """
    Fit LinearRegression on lag features using training data.
    Then forecast horizon steps ahead auto-regressively.
    """
    X_train, t_train = make_supervised(y_train, lags=lags)
    model = LinearRegression()
    model.fit(X_train, t_train)

    history = list(y_train.copy())
    preds = []

    for _ in range(horizon):
        x = np.asarray(history[-lags:]).reshape(1, -1)
        y_hat = float(model.predict(x)[0])
        preds.append(y_hat)
        history.append(y_hat)

    return np.asarray(preds)


def main() -> None:
    seed_everything(42)
    ART_DIR.mkdir(parents=True, exist_ok=True)

    run_id = basic_run_id("seg05")
    print(f"Run: {run_id}")

    y = generate_series(n=600, seed=42)

    # Time split (no shuffling!)
    split = int(len(y) * 0.8)
    y_train = y[:split]
    y_test = y[split:]
    horizon = len(y_test)

    # Forecasts
    pred_naive = naive_forecast(y_train, horizon=horizon)
    pred_ma = moving_average_forecast(y_train, horizon=horizon, window=14)
    pred_lr = autoregressive_lr_forecast(y_train, horizon=horizon, lags=21)

    # Metrics
    results = {
        "naive": {
            "rmse": rmse(y_test, pred_naive),
            "mape": mape(y_test, pred_naive),
        },
        "moving_average": {
            "rmse": rmse(y_test, pred_ma),
            "mape": mape(y_test, pred_ma),
            "window": 14,
        },
        "lag_lr": {
            "rmse": rmse(y_test, pred_lr),
            "mape": mape(y_test, pred_lr),
            "lags": 21,
        },
    }

    # Choose best by RMSE
    best_name = min(results.keys(), key=lambda k: results[k]["rmse"])
    best = results[best_name]

    out = {
        "run_id": run_id,
        "n_total": int(len(y)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "results": results,
        "best_model": best_name,
        "best_metrics": best,
        "note": "Synthetic time series: trend + multi-seasonality + noise. Forecast is done over final 20% horizon.",
    }

    print("\n=== Metrics ===")
    print(json.dumps(out, indent=2))

    with open(ART_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # Plot (show only last part of train + full test for clarity)
    tail = 180
    x_train = np.arange(len(y_train))
    x_test = np.arange(len(y_train), len(y))

    plt.figure()
    plt.plot(x_train[-tail:], y_train[-tail:], label="train (tail)")
    plt.plot(x_test, y_test, label="test (actual)")
    plt.plot(x_test, pred_naive, label="naive")
    plt.plot(x_test, pred_ma, label="moving_avg")
    plt.plot(x_test, pred_lr, label="lag_lr")
    plt.title("Time Series Forecast (Test Horizon)")
    plt.xlabel("time index")
    plt.ylabel("value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ART_DIR / "forecast_plot.png", dpi=160)
    plt.close()

    print(f"\nSaved to: {ART_DIR}")
    print("Best model:", best_name)
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
