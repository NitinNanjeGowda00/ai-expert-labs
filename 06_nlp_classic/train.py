"""
Segment 06 - Classic NLP (TF-IDF + Linear SVM)

Dataset: 20 Newsgroups (sklearn)
We use 4 categories for speed and clarity.

Pipeline:
- TfidfVectorizer
- LinearSVC (SVM)

Outputs:
  artifacts/seg06/
    model.joblib
    metrics.json
    confusion_matrix.png
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
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from setup.repro import seed_everything, basic_run_id

ART_DIR = Path("artifacts/seg06")


def main() -> None:
    seed_everything(42)
    ART_DIR.mkdir(parents=True, exist_ok=True)

    run_id = basic_run_id("seg06")
    print(f"Run: {run_id}")

    categories = [
        "sci.space",
        "rec.sport.baseball",
        "talk.politics.mideast",
        "comp.graphics",
    ]

    data = fetch_20newsgroups(
        subset="all",
        categories=categories,
        remove=("headers", "footers", "quotes"),
    )

    X = data.data
    y = data.target
    target_names = list(data.target_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                max_features=50000,
                ngram_range=(1, 2),
                min_df=2,
            )),
            ("clf", LinearSVC()),
        ]
    )

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred, target_names=target_names, digits=4)

    print("\n=== Accuracy ===")
    print(acc)

    print("\n=== Classification Report ===")
    print(report)

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(xticks_rotation=45)
    plt.title("Confusion Matrix (TF-IDF + LinearSVC)")
    plt.tight_layout()
    plt.savefig(ART_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    out = {
        "run_id": run_id,
        "dataset": "20newsgroups_subset4",
        "categories": categories,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "accuracy": acc,
        "classification_report": report,
    }

    with open(ART_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    dump(pipe, ART_DIR / "model.joblib")

    print(f"\nSaved to: {ART_DIR}")


if __name__ == "__main__":
    main()
