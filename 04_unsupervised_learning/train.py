"""
Segment 04 - Unsupervised Learning (Clustering + PCA)

Dataset: Wine (sklearn)
Steps:
- Scale features
- KMeans clustering
- Evaluate with silhouette score
- Visualize clusters in 2D using PCA

Outputs:
  artifacts/seg04/
    metrics.json
    pca_clusters.png
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
import numpy as np
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from setup.repro import seed_everything, basic_run_id

ART_DIR = Path("artifacts/seg04")


def main() -> None:
    seed_everything(42)
    ART_DIR.mkdir(parents=True, exist_ok=True)

    run_id = basic_run_id("seg04")
    print(f"Run: {run_id}")

    data = load_wine()
    X = data.data
    feature_names = list(data.feature_names)

    # Pipeline: scale -> kmeans
    k = 3  # Wine dataset has 3 classes, but we treat it as unsupervised clustering
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("kmeans", KMeans(n_clusters=k, n_init=20, random_state=42)),
        ]
    )

    pipe.fit(X)
    labels = pipe.named_steps["kmeans"].labels_

    # Silhouette score (higher is better, range ~[-1, 1])
    # Use scaled features for silhouette
    X_scaled = pipe.named_steps["scaler"].transform(X)
    sil = float(silhouette_score(X_scaled, labels))

    metrics = {
        "run_id": run_id,
        "dataset": "wine",
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "k": int(k),
        "silhouette_score": sil,
    }

    print("\n=== Metrics ===")
    print(json.dumps(metrics, indent=2))

    # PCA 2D projection for visualization
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_scaled)

    plt.figure()
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, s=18)
    plt.title("KMeans Clusters (PCA 2D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(ART_DIR / "pca_clusters.png", dpi=160)
    plt.close()

    # Save metrics + some PCA info
    out = {
        **metrics,
        "pca_explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
        "feature_names": feature_names,
    }

    with open(ART_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"\nSaved to: {ART_DIR}")


if __name__ == "__main__":
    main()
