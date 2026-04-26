#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
SERVER_ASSETS_ROOT="${SERVER_ASSETS_ROOT:-$REPO_ROOT/server_assets}"
REAL_TRAIN_SOURCE="${REAL_TRAIN_SOURCE:-$SERVER_ASSETS_ROOT/training_sources/real53}"
AUX_PREPARED_ROOT="${AUX_PREPARED_ROOT:-$SERVER_ASSETS_ROOT/prepared_sources/nasa_secondary}"
MICRONET_CHECKPOINT="${MICRONET_CHECKPOINT:-$REPO_ROOT/experiments/mbu_netpp/workdir/weights/se_resnext50_32x4d_pretrained_microscopynet_v1.0.pth.tar}"
DELIVERY_NAME="${DELIVERY_NAME:-opt_real53_suite}"
DELIVERY_ROOT="${DELIVERY_ROOT:-$REPO_ROOT/results/server_delivery}"
PIPELINE_LOG="${PIPELINE_LOG:-$DELIVERY_ROOT/${DELIVERY_NAME}_pipeline.log}"
AUTO_INSTALL_MISSING="${AUTO_INSTALL_MISSING:-0}"
MAX_WORKERS="${MAX_WORKERS:-2}"
POLL_SECONDS="${POLL_SECONDS:-10}"
MAX_RETRIES="${MAX_RETRIES:-2}"
PYTHON_BIN="${PYTHON_BIN:-}"

NON_TORCH_PACKAGES=(
  "segmentation-models-pytorch"
  "timm"
  "albumentations"
  "opencv-python-headless"
  "numpy"
  "PyYAML"
  "scikit-image"
  "Pillow"
  "openpyxl"
  "pydantic"
  "matplotlib"
)

resolve_python() {
  if [[ -n "${PYTHON_BIN}" && -x "${PYTHON_BIN}" ]]; then
    echo "${PYTHON_BIN}"
    return
  fi
  if [[ -x "/root/miniconda3/bin/python" ]]; then
    echo "/root/miniconda3/bin/python"
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return
  fi
  if command -v python >/dev/null 2>&1; then
    command -v python
    return
  fi
  echo "未找到可用 Python" >&2
  exit 1
}

mkdir -p "${DELIVERY_ROOT}"
exec > >(tee -a "${PIPELINE_LOG}") 2>&1

PYTHON_BIN="$(resolve_python)"
echo "使用 Python: ${PYTHON_BIN}"

if [[ ! -d "${REPO_ROOT}" ]]; then
  echo "仓库根目录不存在: ${REPO_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${REAL_TRAIN_SOURCE}" ]]; then
  echo "真实训练源目录不存在: ${REAL_TRAIN_SOURCE}" >&2
  exit 1
fi
if [[ ! -d "${AUX_PREPARED_ROOT}" ]]; then
  echo "NASA prepared_root 不存在: ${AUX_PREPARED_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${MICRONET_CHECKPOINT}" ]]; then
  echo "MicroNet 权重不存在: ${MICRONET_CHECKPOINT}" >&2
  exit 1
fi

"${PYTHON_BIN}" - <<'PY'
import torch
assert torch.cuda.is_available(), "当前环境没有可用 CUDA"
print({"torch": torch.__version__, "cuda": True, "gpu": torch.cuda.get_device_name(0)})
PY

if [[ "${AUTO_INSTALL_MISSING}" == "1" ]]; then
  "${PYTHON_BIN}" -m pip install "${NON_TORCH_PACKAGES[@]}"
fi

cd "${REPO_ROOT}"
"${PYTHON_BIN}" -m experiments.mbu_netpp.run_opt_training_suite \
  --repo-root "${REPO_ROOT}" \
  --python-bin "${PYTHON_BIN}" \
  --real-train-source "${REAL_TRAIN_SOURCE}" \
  --aux-prepared-root "${AUX_PREPARED_ROOT}" \
  --micronet-checkpoint "${MICRONET_CHECKPOINT}" \
  --delivery-name "${DELIVERY_NAME}" \
  --max-workers "${MAX_WORKERS}" \
  --poll-seconds "${POLL_SECONDS}" \
  --max-retries "${MAX_RETRIES}"
