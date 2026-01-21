"""
Segment 08 - Computer Vision (CNN on MNIST)

Train a small CNN on MNIST and save:
  artifacts/seg08/
    model.pt
    metrics.json
    loss_curve.png
    sample_preds.png
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from setup.repro import seed_everything, basic_run_id

ART_DIR = Path("artifacts/seg08")


@dataclass
class Config:
    seed: int = 42
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 6
    val_frac: float = 0.1


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)   # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)  # 14x14 -> 14x14 after pool
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # 28->14
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  # 14->7
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # logits
        return x


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == y).float().mean().item())


def train_one_epoch(model, loader, opt, loss_fn, device) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n = 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        opt.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        opt.step()

        bs = xb.size(0)
        total_loss += float(loss.item()) * bs
        total_acc += accuracy_from_logits(logits, yb) * bs
        n += bs

    return total_loss / max(n, 1), total_acc / max(n, 1)


@torch.no_grad()
def eval_one_epoch(model, loader, loss_fn, device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = loss_fn(logits, yb)

        bs = xb.size(0)
        total_loss += float(loss.item()) * bs
        total_acc += accuracy_from_logits(logits, yb) * bs
        n += bs

    return total_loss / max(n, 1), total_acc / max(n, 1)


@torch.no_grad()
def save_sample_predictions(model, dataset, device, outpath: Path, n: int = 16) -> None:
    model.eval()
    idxs = np.random.choice(len(dataset), size=n, replace=False)

    images = []
    titles = []

    for i in idxs:
        x, y = dataset[i]
        logits = model(x.unsqueeze(0).to(device))
        pred = int(logits.argmax(dim=1).cpu().item())
        images.append(x.squeeze(0).numpy())
        titles.append(f"y={y}, p={pred}")

    grid = int(math.sqrt(n))
    plt.figure(figsize=(8, 8))
    for j, (img, title) in enumerate(zip(images, titles), start=1):
        plt.subplot(grid, grid, j)
        plt.imshow(img, cmap="gray")
        plt.title(title, fontsize=9)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def main() -> None:
    cfg = Config()
    seed_everything(cfg.seed)
    ART_DIR.mkdir(parents=True, exist_ok=True)

    run_id = basic_run_id("seg08")
    print(f"Run: {run_id}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    # Download MNIST
    train_full = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root="data", train=False, download=True, transform=transform)

    # Train/val split
    n_val = int(len(train_full) * cfg.val_frac)
    n_train = len(train_full) - n_val
    train_ds, val_ds = random_split(train_full, [n_train, n_val], generator=torch.Generator().manual_seed(cfg.seed))

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    model = SimpleCNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, loss_fn, device)
        va_loss, va_acc = eval_one_epoch(model, val_loader, loss_fn, device)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
            f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}"
        )

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # Test metrics
    te_loss, te_acc = eval_one_epoch(model, test_loader, loss_fn, device)

    metrics = {
        "run_id": run_id,
        "device": str(device),
        "test_loss": float(te_loss),
        "test_accuracy": float(te_acc),
        "best_val_accuracy": float(best_val_acc),
        "config": cfg.__dict__,
    }

    print("\n=== Test Metrics ===")
    print(json.dumps(metrics, indent=2))

    # Save artifacts
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg.__dict__,
        },
        ART_DIR / "model.pt",
    )

    with open(ART_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Loss curve plot
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

    # Save sample predictions from test set
    save_sample_predictions(model, test_ds, device, ART_DIR / "sample_preds.png", n=16)

    print(f"\nSaved to: {ART_DIR}")


if __name__ == "__main__":
    main()
