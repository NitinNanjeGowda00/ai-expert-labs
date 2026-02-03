# 🚀 AI Engineering Labs — End-to-End AI Systems (Hands-on)

A production-oriented, runnable repository that demonstrates how to design, train, debug, and deploy real AI systems — from classical ML to LLMs, agentic workflows, and MLOps.

This repo is built to reflect how AI Engineers actually work in industry, not just notebooks or toy demos.

---
What This Repository Demonstrates:-

Classical ML pipelines (tabular data),
NLP (classic + LLM-based),
Time series forecasting,
Deep learning & computer vision,
Retrieval-Augmented Generation (RAG),
Agentic AI (planner + tools),
Model deployment with FastAPI + Docker,
Real debugging and production issues

Each segment is:

Self-contained,
Runnable from CLI,
Saves artifacts (models, metrics, plots),
Written with production constraints in mind

---
## Run any segment from the repo root
python <segment_folder>/<script_name>.py

---
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


---
🧠 Segment-by-Segment Explanation

02 — Supervised Classification (Production ML Pipeline)

Goal: Train, compare, evaluate, and export a deployable classification model.

How it works

Dataset: Breast Cancer Wisconsin (sklearn)

Models compared:

Logistic Regression,
Random Forest

Pipeline includes:

Imputation,
Scaling,
Model

Evaluation:

Accuracy,
ROC-AUC (primary selection metric),
Best model is automatically selected and exported

Outputs

best_model.joblib,
metrics.json,
Confusion matrix plot,
ROC curve plot

Run:
python 02_supervised_classification/train.py

---
03 — Supervised Regression

Goal: Build and evaluate regression pipelines with proper metrics.

What it demonstrates

Train/test split,
Pipeline-based preprocessing,
Regression metrics (RMSE, R²),
Artifact saving for reproducibility

Run:
python 03_supervised_regression/train.py

---
04 — Unsupervised Learning

Goal: Explore structure in unlabeled data.

What it demonstrates

Clustering (e.g., KMeans),
Dimensionality reduction (PCA),
Visualization of clusters,
Interpretation of unsupervised outputs

Run:
python 04_unsupervised_learning/run.py

---
05 — Time Series Forecasting

Goal: Build forecasting models with temporal awareness.

What it demonstrates

Train/validation split by time,
Forecasting logic,
Error metrics,
Plotting predictions vs ground truth

Run:
python 05_time_series/train.py

---
06 — NLP (Classic)

Goal: Show non-LLM NLP pipelines.

What it demonstrates

Text preprocessing,
TF-IDF vectorization,
Linear classifiers,
Explainable NLP models

Run:
python 06_nlp_classic/train.py

---
07 — Deep Learning (PyTorch)

Goal: Implement neural networks from scratch.

What it demonstrates

PyTorch training loops,
Loss tracking,
Model checkpoints,
GPU-ready code structure

Run:
python 07_deep_learning/train.py

---
08 — Computer Vision

Goal: Build and train CNNs for image tasks.

What it demonstrates

CNN architectures,
Image preprocessing,
Training + evaluation,
Visualization of results

Run:
python 08_computer_vision/train.py

---
09 — LLM + RAG (Retrieval-Augmented Generation)

Goal: Prevent hallucinations using document grounding.

How it works

Documents are ingested and indexed into a vector store,
Queries retrieve relevant chunks,
LLM answers only using retrieved context,
Guardrails reject out-of-scope questions

Key Features

Vector store ingestion,
Citation-backed answers,
Deterministic evaluation script

Run:
python 09_llm_rag/ingest.py
python 09_llm_rag/ask.py "Your question here"

---
10 — Agentic AI System

Goal: Build a planner-based agent that chooses tools dynamically.

Architecture

Planner → decides action

Tools:

RAG search,
General LLM,
Calculator

Agent routes questions based on intent

Example:
Question	||  Tool
“What does Grant Thornton do?”	||  RAG
“Capital of France?”	 ||  LLM
“12 * 8 + 4”	||  Calculator

Run:
python 10_agents/agent.py "Your question"

---
11 — MLOps + Deployment (FastAPI + Docker)

Goal: Ship the trained ML model as a production API.

What it demonstrates

Model loading from artifacts,
FastAPI inference endpoint,
Dockerized deployment,
Version mismatch debugging (sklearn)

Build & Run:

docker build -t ai-expert-labs-api .
docker run -p 8000:8000 ai-expert-labs-api

Test API:

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [...]}'

---
🧪 Engineering Practices Highlighted

Reproducibility (seeds, run IDs),
Pipeline-based ML,
Proper evaluation metrics,
Model persistence & compatibility issues,
LLM grounding & hallucination control,
Tool-using agents,
Production debugging,
API + Docker deployment

---
👤 Author

Nitin Nanje Gowda
🎓 MSc Artificial Intelligence & Robotics
🎯 Target Roles: AI Engineer / ML Engineer / LLM Engineer
📍 Bengaluru, India

---

## Quickstart (Windows / PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pips
pip install -r requirements.txt


