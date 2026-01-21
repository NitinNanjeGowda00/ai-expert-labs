"""
Segment 00 - Reproducibility Utilities

This module provides:
- seed_everything(seed): sets random seeds for reproducibility
- basic_run_id(prefix): makes a simple timestamped run id

Usage:
  from 00_setup.repro import seed_everything
  seed_everything(42)
"""

from __future__ import annotations

import os
import random
from datetime import datetime
from typing import Optional

import numpy as np


def seed_everything(seed: int = 42, deterministic_torch: bool = True) -> dict:
    """
    Set seeds for Python, NumPy, and (if installed) PyTorch.
    Returns a dict describing what was set.
    """
    info = {"seed": seed, "python": True, "numpy": True, "torch": False}

    # Python's hash seed (helps ensure deterministic hashing behavior)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Python RNG
    random.seed(seed)

    # NumPy RNG
    np.random.seed(seed)

    # Optional: Torch
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        info["torch"] = True

        if deterministic_torch:
            # Best-effort determinism settings
            torch.backends.cudnn.deterministic = True  # type: ignore
            torch.backends.cudnn.benchmark = False  # type: ignore
    except Exception:
        # Torch not installed yet (that's fine for Segment 00)
        pass

    return info


def basic_run_id(prefix: str = "run", now: Optional[datetime] = None) -> str:
    """
    Generate a simple run id like: run_2026-01-21_224501
    """
    now = now or datetime.now()
    return f"{prefix}_{now.strftime('%Y-%m-%d_%H%M%S')}"


if __name__ == "__main__":
    # Quick sanity check
    info = seed_everything(123)
    print("Seed info:", info)
    print("Run id:", basic_run_id("seg00"))
    print("NumPy sample:", np.random.randint(0, 10, size=5))
