# 08 — Computer Vision (CNN on MNIST)

Train a small **PyTorch CNN** on **MNIST** with a clean training loop:
**device selection → train/val tracking → test metrics → saved artifacts**.

**Dataset:** MNIST (downloaded automatically by torchvision)  
**Task:** Image classification (10 digits)

---

## Quickstart (60 seconds)

### Install PyTorch
Use the official selector to get the right command for your system:
https://pytorch.org/get-started/locally/

Example:
```bash
pip install torch torchvision torchaudio

pip install -r requirements.txt
python 08_computer_vision/train_cnn_mnist.py
