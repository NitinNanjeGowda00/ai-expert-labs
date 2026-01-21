"""
Segment 01 - ML Basics: EDA + Preprocessing Pipeline + Baseline Model

Dataset: Titanic (loaded via OpenML through scikit-learn)
Task: Binary classification (survived)

Run:
  python 01_ml_basics/train.py

Outputs:
  artifacts/seg01/
    model.joblib
    metrics.json
    feature_info.json
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is on PYTHONPATH so "setup" can be imported
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler



from setup.repro import seed_everything, basic_run_id


ART_DIR = Path("artifacts/seg01")


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    # Titanic is small + good for EDA & mixed types.
    data = fetch_openml(name="titanic", version=1, as_frame=True)
    df = data.frame.copy()

    # Target: survived (0/1 as string)
    y = df["survived"].astype(int)
    X = df.drop(columns=["survived"])

    return X, y


def basic_eda(X: pd.DataFrame, y: pd.Series) -> dict:
    info = {
        "rows": int(X.shape[0]),
        "cols": int(X.shape[1]),
        "target_mean": float(y.mean()),
        "missing_by_col_top10": (
            X.isna().mean().sort_values(ascending=False).head(10).to_dict()
        ),
        "dtypes": X.dtypes.astype(str).value_counts().to_dict(),
    }
    return info


def build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    # Detect columns
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    # Pipelines
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor, numeric_cols, categorical_cols


def main() -> None:
    seed_everything(42)

    ART_DIR.mkdir(parents=True, exist_ok=True)

    run_id = basic_run_id("seg01")
    print(f"Run: {run_id}")

    X, y = load_data()

    # EDA
    eda = basic_eda(X, y)
    print("\n=== Basic EDA ===")
    print(json.dumps(eda, indent=2))

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Preprocess + model
    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X_train)

    clf = LogisticRegression(max_iter=2000)

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", clf),
        ]
    )

    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "run_id": run_id,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    print("\n=== Metrics ===")
    print(json.dumps(metrics, indent=2))

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, digits=4))

    # Save artifacts
    dump(model, ART_DIR / "model.joblib")

    with open(ART_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    feature_info = {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "eda": eda,
    }
    with open(ART_DIR / "feature_info.json", "w", encoding="utf-8") as f:
        json.dump(feature_info, f, indent=2)

    print(f"\nSaved to: {ART_DIR}")


if __name__ == "__main__":
    main()
