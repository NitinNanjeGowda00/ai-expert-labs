# AI Expert Labs (Hands-on)

This repo is a step-by-step hands-on track to build AI/ML systems across major segments:
tabular ML, NLP, time-series, deep learning, computer vision, RAG, agents, and deployment.

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

## Windows Setup (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
