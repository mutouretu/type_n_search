#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p \
  outputs \
  outputs/models \
  outputs/predictions \
  outputs/analysis \
  data/raw/daily \
  data/labels \
  data/processed

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

pip install --upgrade pip setuptools wheel

if [[ "${INSTALL_TABULAR_ONLY:-0}" == "1" ]]; then
  pip install -r requirements/tabular.txt
else
  pip install -r requirements.txt
fi

case "${TORCH_VARIANT:-}" in
  cu126)
    pip install -r requirements/torch-cu126.txt
    ;;
  cu128)
    pip install -r requirements/torch-cu128.txt
    ;;
  "")
    ;;
  *)
    echo "Unsupported TORCH_VARIANT='${TORCH_VARIANT}'. Use: cu126 or cu128."
    exit 1
    ;;
esac

python - <<'PY'
import importlib


def print_version(module_name, display_name=None):
    name = display_name or module_name
    m = importlib.import_module(module_name)
    print(f"{name}: {m.__version__}")


print_version("numpy")
print_version("pandas")
print_version("sklearn", "sklearn")
print_version("lightgbm")
print_version("xgboost")

try:
    torch = importlib.import_module("torch")
except ModuleNotFoundError:
    print("torch: not installed (optional)")
else:
    print(f"torch: {torch.__version__}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
PY

echo "Bootstrap completed."
