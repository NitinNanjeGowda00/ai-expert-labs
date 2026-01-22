# AI Expert Labs (Hands-on)

A step-by-step, runnable repo for building end-to-end AI/ML systems across tabular ML, NLP, time series, deep learning, computer vision, RAG, agents, and deployment.

Each segment is **self-contained**, has a clear entry script, prints metrics, and saves outputs under an `artifacts/` folder for easy inspection.

---

## Quickstart (Windows / PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pips
pip install -r requirements.txt

## Run any segment from the repo root
python <segment_folder>/<script_name>.py

## Repo Structure
- `setup/` Reproducible environment + utilities
- `01_ml_basics/` EDA + feature engineering
- `02_supervised_classification/` Classification pipeline
- `03_supervised_regression/` Regression pipeline
- `04_unsupervised_learning/` Clustering + PCA
- `05_time_series/` Forecasting
- `06_nlp_classic/` TF-IDF + linear models
- `07_deep_learning/` PyTorch basics
- `08_computer_vision/` CNN
- `09_llm_rag/` RAG with citations + eval
- `10_agents/` Tool-using agent
- `11_mlops_deploy/` FastAPI + Docker deployment

