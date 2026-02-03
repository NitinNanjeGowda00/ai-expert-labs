"""
Segment 02 - Supervised Classification (Model Comparison + Plots)

Dataset: Breast Cancer Wisconsin (sklearn)
Models: Logistic Regression vs Random Forest
Outputs:
  artifacts/seg02/
    best_model.joblib
    metrics.json
    confusion_matrix.png
    roc_curve.png

Also exports deployable model to:
  11_mlops_deploy/models/logreg_model.joblib
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure repo root is on PYTHONPATH so "setup" can be imported
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
from joblib import dump
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from setup.repro import seed_everything, basic_run_id


ART_DIR = Path("artifacts/seg02")
DEPLOY_MODEL_DIR = Path("11_mlops_deploy/models")


def make_pipeline(model):
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
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    return {
        "model": name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }


def save_confusion_matrix(pipe, X_test, y_test, outpath: Path) -> None:
    y_pred = pipe.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def save_roc_curve(pipe, X_test, y_test, outpath: Path) -> None:
    RocCurveDisplay.from_estimator(pipe, X_test, y_test)
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def main() -> None:
    seed_everything(42)
    ART_DIR.mkdir(parents=True, exist_ok=True)
    DEPLOY_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    run_id = basic_run_id("seg02")
    print(f"Run: {run_id}")

    data = load_breast_cancer()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = [
        ("logreg", LogisticRegression(max_iter=5000)),
        ("rf", RandomForestClassifier(n_estimators=400, random_state=42)),
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

    # Pick best by roc_auc, tie-breaker accuracy
    results_sorted = sorted(
        results, key=lambda d: (d["roc_auc"], d["accuracy"]), reverse=True
    )
    best = results_sorted[0]
    best_name = best["model"]
    best_pipe = fitted[best_name]

    # Save plots
    save_confusion_matrix(best_pipe, X_test, y_test, ART_DIR / "confusion_matrix.png")
    save_roc_curve(best_pipe, X_test, y_test, ART_DIR / "roc_curve.png")

    # Save metrics
    out = {
        "run_id": run_id,
        "results": results_sorted,
        "best_model": best_name,
        "best_metrics": best,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    with open(ART_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # Save training artifact
    dump(best_pipe, ART_DIR / "best_model.joblib")

    # Export deployable artifact
    deploy_path = DEPLOY_MODEL_DIR / "logreg_model.joblib"
    dump(best_pipe, deploy_path)

    print("\nSaved training artifacts to:", ART_DIR)
    print("Exported deployable model to:", deploy_path)
    print("Best model:", best_name)
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
