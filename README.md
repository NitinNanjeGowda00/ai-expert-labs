🚀 AI Engineering Labs — End-to-End AI Systems (ML → LLMs → Agents → MLOps)

Author: Nitin Nanje Gowda
Role Target: AI Engineer / Applied ML Engineer / LLM Engineer
Stack: Python · Scikit-Learn · PyTorch · FastAPI · Docker · OpenAI · RAG · Agents · MLOps

🧠 What This Repository Demonstrates

This repository is a production-style AI engineering journey, built to showcase how modern AI systems are designed, trained, debugged, deployed, and orchestrated — not just notebooks or toy demos.

It covers:

Classical Machine Learning

Deep Learning & Computer Vision

LLM-based Retrieval Augmented Generation (RAG)

Agentic AI systems (planner + tools)

MLOps & deployment with FastAPI + Docker

Every segment is independently runnable, versioned, and production-oriented.

🧩 Repository Structure
ai-expert-labs/
├── 00_setup/                 # Reproducibility, seeds, run IDs
├── 01_ml_basics/             # ML fundamentals & evaluation
├── 02_supervised_classification/
├── 03_supervised_regression/
├── 04_unsupervised_learning/
├── 05_time_series/
├── 06_nlp_classic/
├── 07_deep_learning/
├── 08_computer_vision/
├── 09_llm_rag/               # RAG with OpenAI + vector stores
├── 10_agents/                # Planner-based agent system
├── 11_mlops_deploy/          # FastAPI + Docker deployment
└── README.md

🧪 Segment 02 — Supervised ML (Production-Ready Pipeline)

Dataset: Breast Cancer Wisconsin
Models: Logistic Regression vs Random Forest
Metrics: Accuracy, ROC-AUC
Artifacts Saved:

Trained pipeline (joblib)

Metrics JSON

Confusion matrix

ROC curve

🏆 Results
Model	Accuracy	ROC-AUC
Logistic Regression	~98.2%	~0.995
Random Forest	~95.6%	~0.993

Key Engineering Practices

Full preprocessing pipeline (imputation + scaling)

Model comparison & selection

Reproducible runs

Exported deployable model artifact

🔍 Segment 09 — LLM RAG System (Enterprise-Style)

Features

Document ingestion & indexing

Vector store retrieval

Guardrails with confidence thresholds

Citation-grounded answers

“I don’t know” behavior for out-of-scope queries

Example

python ask.py "What does Grant Thornton do?"


Result

Answer grounded strictly in documents

No hallucinations

Citations attached

🤖 Segment 10 — Agentic AI System

Built a planner-based AI agent that dynamically routes questions:

Question Type	Tool Used
Company knowledge	RAG
General knowledge	LLM
Math / logic	Calculator

Architecture

Planner → Tool selection → Execution → Response

Clear separation of reasoning and execution

This mirrors real agent systems used in enterprise AI platforms.

🚢 Segment 11 — MLOps & Deployment

What was shipped

FastAPI inference service

Dockerized application

Loaded trained sklearn pipeline

/predict REST endpoint

Tested via curl

Example API Call
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [...]}'


Response

{
  "prediction": 0,
  "probability": 0.29
}

🧠 Real-World Debugging Experience

This repo intentionally reflects real engineering challenges, including:

OpenAI API deprecations (Assistants → Responses)

Vector store indexing failures

Authentication & API key changes

Python import system pitfalls

Sklearn serialization incompatibilities

Docker runtime vs training environment mismatches

Windows vs Linux execution issues

All were debugged and resolved systematically, not by trial and error.

📬 Contact

GitHub: https://github.com/NitinNanjeGowda00

LinkedIn: https://linkedin.com/in/nitinnanjegowda

Email: nitingowda.roan@gmail.com