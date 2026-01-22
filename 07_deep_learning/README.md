# 07 — Deep Learning (PyTorch MLP)

A minimal deep learning training loop in **PyTorch** for tabular classification:
**MLP → validation tracking → early stopping → test metrics → saved artifacts**.

**Dataset:** Breast Cancer Wisconsin (scikit-learn)  
**Task:** Binary classification

---

## Quickstart (60 seconds)

### Install PyTorch
Pick the correct command for your machine from the official selector:
https://pytorch.org/get-started/locally/

Example (CPU/GPU varies by system):
```bash
pip install torch torchvision torchaudio

pip install -r requirements.txt
python 07_deep_learning/train_mlp.py
