# Environment Guide

## Stable Baseline Environment

Current stable setup targets the tabular pipeline:

- data contract check
- dataset build
- baseline training (LR / LGBM / XGB)
- scan inference
- inspection and model comparison

This path is the default and is expected to be reproducible on Ubuntu and WSL2.

## Why Requirements Are Layered

Dependencies are split into layers to improve reproducibility and cross-machine compatibility:

- `requirements/base.txt`: common runtime dependencies
- `requirements/tabular.txt`: tabular ML stack (LightGBM, XGBoost)
- `requirements/dev.txt`: developer/testing layer
- `requirements/torch-cu126.txt` / `requirements/torch-cu128.txt`: optional deep learning layer

This avoids hard pinning to unavailable versions and keeps optional stacks isolated.

## Torch Is Optional

PyTorch is not part of default installation.

Reason:

- current production path is tabular-only
- deep learning stack should be opt-in
- GPU/CUDA variants differ across machines

Use torch layers only when deep learning experiments are needed.

## CUDA Variant Guidance

Recommended:

- `cu126`: default stable choice
- `cu128`: use for newer GPU/driver stacks

PyTorch official binaries include CUDA runtime. In most cases, you only need a compatible NVIDIA driver and do not need a local CUDA toolkit installation.

## One-Command Bootstrap

Run:

```bash
bash scripts/bootstrap_ubuntu_wsl.sh
```

Optional modes:

- tabular-only install:

```bash
INSTALL_TABULAR_ONLY=1 bash scripts/bootstrap_ubuntu_wsl.sh
```

- tabular + torch cu126:

```bash
TORCH_VARIANT=cu126 bash scripts/bootstrap_ubuntu_wsl.sh
```

- tabular + torch cu128:

```bash
TORCH_VARIANT=cu128 bash scripts/bootstrap_ubuntu_wsl.sh
```
