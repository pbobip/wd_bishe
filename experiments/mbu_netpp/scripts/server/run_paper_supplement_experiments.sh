#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-}"
DELIVERY_ROOT="${DELIVERY_ROOT:-$REPO_ROOT/results/server_delivery}"
LOG_PATH="${LOG_PATH:-$DELIVERY_ROOT/paper_supplement_experiments.log}"

SEEDS="${SEEDS:-0 42 2026}"
MAX_WORKERS="${MAX_WORKERS:-2}"
POLL_SECONDS="${POLL_SECONDS:-10}"

RUN_GPU_TIMING="${RUN_GPU_TIMING:-1}"
RUN_AUG_VOLUME="${RUN_AUG_VOLUME:-1}"
RUN_SEED_REPEATS="${RUN_SEED_REPEATS:-1}"

CHECKPOINT="${CHECKPOINT:-$REPO_ROOT/results/opt_real53_suite/experiments/opt_real53_boundary_sampling/best.pt}"
PREPARED_ROOT="${PREPARED_ROOT:-$REPO_ROOT/experiments/mbu_netpp/workdir/prepared_real53_cv5}"
BASE_CONFIG="${BASE_CONFIG:-$REPO_ROOT/experiments/mbu_netpp/configs/opt_real53_boundary_sampling.yaml}"
MICRONET_CHECKPOINT="${MICRONET_CHECKPOINT:-$REPO_ROOT/experiments/mbu_netpp/workdir/weights/se_resnext50_32x4d_pretrained_microscopynet_v1.0.pth.tar}"

MODEL_EFFICIENCY_OUTPUT="${MODEL_EFFICIENCY_OUTPUT:-$REPO_ROOT/results/paper_supplement/model_efficiency_gpu}"
AUG_VOLUME_OUTPUT="${AUG_VOLUME_OUTPUT:-$REPO_ROOT/results/paper_supplement/augmentation_volume}"
SEED_REPEAT_OUTPUT="${SEED_REPEAT_OUTPUT:-$REPO_ROOT/experiments/mbu_netpp/outputs/paper_seed_repeats}"
SEED_REPEAT_DELIVERY_NAME="${SEED_REPEAT_DELIVERY_NAME:-paper_seed_repeats}"
FINAL_DELIVERY_NAME="${FINAL_DELIVERY_NAME:-paper_supplement_all}"

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
exec > >(tee -a "${LOG_PATH}") 2>&1

PYTHON_BIN="$(resolve_python)"
read -r -a SEED_ARGS <<< "${SEEDS}"

echo "REPO_ROOT=${REPO_ROOT}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "SEEDS=${SEEDS}"
echo "MAX_WORKERS=${MAX_WORKERS}"

cd "${REPO_ROOT}"

"${PYTHON_BIN}" - <<'PY'
import torch
assert torch.cuda.is_available(), "当前服务器没有可用 CUDA，不能做 5090 GPU 推理计时或训练"
print({"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)})
PY

if [[ "${RUN_GPU_TIMING}" == "1" ]]; then
  if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "主模型权重不存在: ${CHECKPOINT}" >&2
    exit 1
  fi
  echo "开始 RTX GPU 推理计时"
  "${PYTHON_BIN}" -m experiments.mbu_netpp.model_efficiency \
    --checkpoint "${CHECKPOINT}" \
    --prepared-root "${PREPARED_ROOT}" \
    --fold-manifest-name "folds_5_seed42_holdout10.json" \
    --fold-index 0 \
    --output-dir "${MODEL_EFFICIENCY_OUTPUT}" \
    --device cuda \
    --warmup 10 \
    --repeats 50 \
    --holdout-warmup 3 \
    --holdout-repeats 20 \
    --limit-images 0
fi

if [[ "${RUN_AUG_VOLUME}" == "1" ]]; then
  echo "开始统计在线增强数量和离线等价数量"
  "${PYTHON_BIN}" -m experiments.mbu_netpp.augmentation_volume \
    --config "${BASE_CONFIG}" \
    --prepared-root "${PREPARED_ROOT}" \
    --fold-manifest-name "folds_5_seed42_holdout10.json" \
    --holdout-manifest-name "holdout_10_seed42.json" \
    --output-dir "${AUG_VOLUME_OUTPUT}" \
    --seeds "${SEED_ARGS[@]}"
fi

if [[ "${RUN_SEED_REPEATS}" == "1" ]]; then
  if [[ ! -f "${MICRONET_CHECKPOINT}" ]]; then
    echo "MicroNet 权重不存在: ${MICRONET_CHECKPOINT}" >&2
    exit 1
  fi
  echo "开始主模型 3 seed 重复实验"
  "${PYTHON_BIN}" -m experiments.mbu_netpp.run_seed_repeats \
    --base-config "${BASE_CONFIG}" \
    --prepared-root "${PREPARED_ROOT}" \
    --micronet-checkpoint "${MICRONET_CHECKPOINT}" \
    --seeds "${SEED_ARGS[@]}" \
    --fold-seed 42 \
    --fold-manifest-name "folds_5_seed42_holdout10.json" \
    --holdout-manifest-name "holdout_10_seed42.json" \
    --output-root "${SEED_REPEAT_OUTPUT}" \
    --delivery-root "${DELIVERY_ROOT}" \
    --delivery-name "${SEED_REPEAT_DELIVERY_NAME}" \
    --python-bin "${PYTHON_BIN}" \
    --device cuda \
    --max-workers "${MAX_WORKERS}" \
    --poll-seconds "${POLL_SECONDS}"
fi

FINAL_DELIVERY_DIR="${DELIVERY_ROOT}/${FINAL_DELIVERY_NAME}"
FINAL_ARCHIVE="${DELIVERY_ROOT}/${FINAL_DELIVERY_NAME}.tar.gz"
rm -rf "${FINAL_DELIVERY_DIR}" "${FINAL_ARCHIVE}"
mkdir -p "${FINAL_DELIVERY_DIR}"

if [[ -d "${MODEL_EFFICIENCY_OUTPUT}" ]]; then
  cp -a "${MODEL_EFFICIENCY_OUTPUT}" "${FINAL_DELIVERY_DIR}/model_efficiency_gpu"
fi
if [[ -d "${AUG_VOLUME_OUTPUT}" ]]; then
  cp -a "${AUG_VOLUME_OUTPUT}" "${FINAL_DELIVERY_DIR}/augmentation_volume"
fi
if [[ -d "${DELIVERY_ROOT}/${SEED_REPEAT_DELIVERY_NAME}" ]]; then
  cp -a "${DELIVERY_ROOT}/${SEED_REPEAT_DELIVERY_NAME}" "${FINAL_DELIVERY_DIR}/seed_repeats"
fi
if [[ -f "${DELIVERY_ROOT}/${SEED_REPEAT_DELIVERY_NAME}.tar.gz" ]]; then
  cp -a "${DELIVERY_ROOT}/${SEED_REPEAT_DELIVERY_NAME}.tar.gz" "${FINAL_DELIVERY_DIR}/"
fi

cat > "${FINAL_DELIVERY_DIR}/README.txt" <<EOF
paper_supplement_all

model_efficiency_gpu: RTX GPU 参数量/FLOPs/推理时间
augmentation_volume: 在线增强数量与离线等价 patch 数量
seed_repeats: 主模型 ${SEEDS} 三个随机种子重复实验结果
EOF

tar -czf "${FINAL_ARCHIVE}" -C "${DELIVERY_ROOT}" "${FINAL_DELIVERY_NAME}"
echo "最终补充实验包: ${FINAL_ARCHIVE}"
