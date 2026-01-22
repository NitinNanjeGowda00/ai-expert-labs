# 05 — Time Series Forecasting (Baselines + Lag Features)

A small, end-to-end time series forecasting workflow using a **synthetic series** (trend + seasonality + noise).  
Compares simple baselines with a feature-based model using **lagged values**.

Models included:
- Naive forecast (last observed value)
- Moving average forecast
- Linear regression with lag features (autoregressive rollout)

---

## Quickstart (60 seconds)

```bash
pip install -r requirements.txt
python 05_time_series/train.py
