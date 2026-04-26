#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
TORCH_PACKAGE_SPECS="${TORCH_PACKAGE_SPECS:-torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0}"
INSTALL_SYSTEM_PACKAGES="${INSTALL_SYSTEM_PACKAGES:-1}"

if [[ "${INSTALL_SYSTEM_PACKAGES}" == "1" ]] && command -v apt-get >/dev/null 2>&1; then
  if command -v sudo >/dev/null 2>&1; then
    sudo apt-get update || true
    sudo apt-get install -y python3 python3-pip python3-venv libgl1 libglib2.0-0 || true
  else
    apt-get update || true
    apt-get install -y python3 python3-pip python3-venv libgl1 libglib2.0-0 || true
  fi
fi

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
# RTX 5090 (sm_120) 需要 CUDA 12.8+ 对应的 PyTorch binary。
# shellcheck disable=SC2086
python -m pip install --upgrade --force-reinstall --index-url "${TORCH_INDEX_URL}" ${TORCH_PACKAGE_SPECS}
python -m pip install -r "${REPO_ROOT}/experiments/mbu_netpp/requirements.txt"

python - <<'PY'
import platform
import torch

print("Python:", platform.python_version())
print("Torch:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
PY
