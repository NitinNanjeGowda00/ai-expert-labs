## Repository Overview (Segmented Learning + Production Patterns)

This repo is a structured, segment-by-segment build-up of practical ML/AI engineering skills.  
Each segment is runnable, produces artifacts, and is written to reflect production habits: reproducibility, pipeline discipline, and clean evaluation.

### Segments

- **01_ml_basics — EDA + Preprocessing Pipeline (Tabular ML)**
  - Builds a reusable preprocessing pipeline (numeric + categorical), runs basic EDA, trains a baseline classifier, and saves artifacts.
  - Outputs: run id, metrics (Accuracy/ROC-AUC), classification report, and saved artifacts under `artifacts/`.
  - Quick run:
    ```powershell
    python 01_ml_basics/train.py
    ```

### What to look for (recruiter scan)
- **Reproducibility**: run identifiers + saved artifacts
- **Pipeline-first design**: preprocessing encapsulated in a reusable pipeline
- **Evaluation clarity**: standard metrics + reports
- **Code organization**: segment folders with clear entrypoints and outputs

### How to run any segment
1. Install dependencies
   ```bash
   pip install -r requirements.txt
