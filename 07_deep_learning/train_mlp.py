"""
Segment 07 - Deep Learning Basics (PyTorch MLP)

Dataset: Breast Cancer Wisconsin (sklearn)
Model: MLP (2 hidden layers)
Features:
- train/val split
- minibatch training
- early stopping
- save model + metrics + loss curve

Outputs:
  artifacts/seg07/
    model.pt
    metrics.json
    loss_curve.png
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

# Ensure repo root is on PYTHONPATH so "setup" can be imported
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from setup.repro import seed_everything, basic_run_id

ART_DIR = Path("artifacts/seg07")


@dataclass
class Config:
    seed: int = 42
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 80
    patience: int = 10
    hidden1: int = 64
    hidden2: int = 32
    dropout: float = 0.15


class MLP(nn.Module):
    def __init__(self, n_in: int, h1: int, h2: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, 1),  # logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)  # (B,)


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    ds = torch.utils.data.TensorDataset(X_t, y_t)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


@torch.no_grad()
def predict_logits(model: nn.Module, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits = []
    all_y = []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb).detach().cpu().numpy()
        all_logits.append(logits)
        all_y.append(yb.numpy())
    return np.concatenate(all_logits), np.concatenate(all_y)


def train_one_epoch(model, loader, opt, loss_fn, device) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        opt.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        opt.step()

        bs = xb.size(0)
        total_loss += float(loss.item()) * bs
        n += bs
    return total_loss / max(n, 1)


@torch.no_grad()
def eval_loss(model, loader, loss_fn, device) -> float:
    model.eval()
    total_loss = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        bs = xb.size(0)
        total_loss += float(loss.item()) * bs
        n += bs
    return total_loss / max(n, 1)


def main() -> None:
    cfg = Config()
    seed_everything(cfg.seed)
    ART_DIR.mkdir(parents=True, exist_ok=True)

    run_id = basic_run_id("seg07")
    print(f"Run: {run_id}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # Load data
    data = load_breast_cancer()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.float32)

    # Train/val/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=cfg.seed, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=cfg.seed, stratify=y_train
    )

    # Scale (fit on train only)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    train_loader = make_loader(X_train, y_train, cfg.batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, cfg.batch_size, shuffle=False)
    test_loader = make_loader(X_test, y_test, cfg.batch_size, shuffle=False)

    # Model
    model = MLP(n_in=X_train.shape[1], h1=cfg.hidden1, h2=cfg.hidden2, dropout=cfg.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, cfg.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, opt, loss_fn, device)
        va_loss = eval_loss(model, val_loader, loss_fn, device)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)

        print(f"Epoch {epoch:03d} | train_loss={tr_loss:.4f} | val_loss={va_loss:.4f}")

        if va_loss < best_val - 1e-5:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                print(f"Early stopping at epoch {epoch} (patience={cfg.patience})")
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluate on test
    test_logits, test_y = predict_logits(model, test_loader, device)
    test_proba = 1.0 / (1.0 + np.exp(-test_logits))
    test_pred = (test_proba >= 0.5).astype(int)

    acc = float(accuracy_score(test_y.astype(int), test_pred))
    auc = float(roc_auc_score(test_y, test_proba))

    metrics = {
        "run_id": run_id,
        "device": str(device),
        "accuracy": acc,
        "roc_auc": auc,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "config": cfg.__dict__,
        "best_val_loss": float(best_val),
    }

    print("\n=== Test Metrics ===")
    print(json.dumps(metrics, indent=2))

    # Save artifacts
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "feature_names": list(data.feature_names),
            "config": cfg.__dict__,
        },
        ART_DIR / "model.pt",
    )

    with open(ART_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Loss curve
    plt.figure()
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.title("Loss Curve")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ART_DIR / "loss_curve.png", dpi=160)
    plt.close()

    print(f"\nSaved to: {ART_DIR}")


if __name__ == "__main__":
    main()
