#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"
SERVER_ASSETS_ROOT="${SERVER_ASSETS_ROOT:-$REPO_ROOT/server_assets}"

REAL33_RAW="${REAL33_RAW:-$SERVER_ASSETS_ROOT/training_sources/analysis_real33}"
REAL16_RAW="${REAL16_RAW:-$SERVER_ASSETS_ROOT/training_sources/refined_real16}"
FINAL_INFER_INPUT="${FINAL_INFER_INPUT:-$SERVER_ASSETS_ROOT/final_infer_inputs/full_png_cropped_xlsx_images}"

PREPARED_REAL33="${PREPARED_REAL33:-$REPO_ROOT/experiments/mbu_netpp/workdir/prepared_analysis_real33}"
PREPARED_REAL16="${PREPARED_REAL16:-$REPO_ROOT/experiments/mbu_netpp/workdir/prepared_refined_real16}"
MERGED_PREPARED="${MERGED_PREPARED:-$REPO_ROOT/experiments/mbu_netpp/workdir/prepared_real49}"
TRAIN_OUTPUT="${TRAIN_OUTPUT:-$REPO_ROOT/experiments/mbu_netpp/outputs/real49_supervised}"
FINAL_INFER_OUTPUT="${FINAL_INFER_OUTPUT:-$TRAIN_OUTPUT/final_infer_100}"
FILTERED_INFER_OUTPUT="${FILTERED_INFER_OUTPUT:-$TRAIN_OUTPUT/final_infer_excluding_train}"
DELIVERY_ROOT="${DELIVERY_ROOT:-$REPO_ROOT/results/server_delivery}"
DELIVERY_NAME="${DELIVERY_NAME:-real49_final_infer_100}"
DELIVERY_DIR="${DELIVERY_DIR:-$DELIVERY_ROOT/$DELIVERY_NAME}"

BASE_CONFIG_PATH="${BASE_CONFIG_PATH:-$REPO_ROOT/experiments/mbu_netpp/configs/real49_supervised_server_full.yaml}"
CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/experiments/mbu_netpp/workdir/real49_server_runtime.yaml}"
MICRONET_CHECKPOINT="${MICRONET_CHECKPOINT:-$REPO_ROOT/experiments/mbu_netpp/workdir/weights/se_resnext50_32x4d_pretrained_microscopynet_v1.0.pth.tar}"
CLEAN_RUN="${CLEAN_RUN:-1}"

if [[ ! -d "${REPO_ROOT}" ]]; then
  echo "仓库根目录不存在: ${REPO_ROOT}" >&2
  exit 1
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "未找到虚拟环境: ${VENV_DIR}" >&2
  echo "请先执行 experiments/mbu_netpp/scripts/server/install_env.sh" >&2
  exit 1
fi

if [[ ! -f "${MICRONET_CHECKPOINT}" ]]; then
  echo "未找到 MicroNet 权重: ${MICRONET_CHECKPOINT}" >&2
  exit 1
fi

if [[ ! -f "${BASE_CONFIG_PATH}" ]]; then
  echo "未找到基础训练配置: ${BASE_CONFIG_PATH}" >&2
  exit 1
fi

for required_dir in "${REAL33_RAW}" "${REAL16_RAW}" "${FINAL_INFER_INPUT}"; do
  if [[ ! -e "${required_dir}" ]]; then
    echo "缺少输入目录: ${required_dir}" >&2
    exit 1
  fi
done

source "${VENV_DIR}/bin/activate"
cd "${REPO_ROOT}"

if [[ "${CLEAN_RUN}" == "1" ]]; then
  rm -rf "${PREPARED_REAL33}" "${PREPARED_REAL16}" "${MERGED_PREPARED}" "${TRAIN_OUTPUT}" "${DELIVERY_DIR}"
  rm -f "${DELIVERY_ROOT}/${DELIVERY_NAME}.tar.gz"
fi

python -m experiments.mbu_netpp.preparation \
  --images-dir "${REAL33_RAW}" \
  --annotations-dir "${REAL33_RAW}" \
  --output-root "${PREPARED_REAL33}" \
  --no-auto-crop

python -m experiments.mbu_netpp.preparation \
  --images-dir "${REAL16_RAW}" \
  --annotations-dir "${REAL16_RAW}" \
  --output-root "${PREPARED_REAL16}" \
  --no-auto-crop

python -m experiments.mbu_netpp.prepare_merged_supervised \
  --prepared-root "${PREPARED_REAL33}" \
  --source-alias real33 \
  --prepared-root "${PREPARED_REAL16}" \
  --source-alias real16 \
  --output-root "${MERGED_PREPARED}" \
  --num-folds 3 \
  --seed 42

export BASE_CONFIG_PATH CONFIG_PATH TRAIN_OUTPUT MERGED_PREPARED MICRONET_CHECKPOINT
python - <<'PY'
import json
import os
from pathlib import Path

import torch
import yaml

base_config_path = Path(os.environ["BASE_CONFIG_PATH"])
runtime_config_path = Path(os.environ["CONFIG_PATH"])
train_output = os.environ["TRAIN_OUTPUT"]
prepared_root = os.environ["MERGED_PREPARED"]
micronet_checkpoint = os.environ["MICRONET_CHECKPOINT"]

config = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
training = config["training"]
data_cfg = config["data"]
experiment = config["experiment"]

cpu_count = os.cpu_count() or 4
num_workers = max(2, min(8, cpu_count // 2))
batch_size = int(training.get("batch_size", 4))
device_name = "cpu"
gpu_info = {}

if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / (1024 ** 3)
    gpu_info = {"name": props.name, "total_gb": round(total_gb, 2)}
    device_name = "cuda"
    if total_gb >= 20:
        batch_size = 6
    elif total_gb >= 12:
        batch_size = 4
    elif total_gb >= 8:
        batch_size = 2
    else:
        batch_size = 1
else:
    batch_size = 1

training["device"] = device_name
training["batch_size"] = batch_size
training["num_workers"] = num_workers
data_cfg["prepared_root"] = prepared_root
experiment["output_root"] = train_output
config["model"]["micronet_checkpoint"] = micronet_checkpoint
config["server_runtime"] = {
    "cpu_count": cpu_count,
    "gpu_info": gpu_info,
    "resolved_batch_size": batch_size,
    "resolved_num_workers": num_workers,
}

runtime_config_path.parent.mkdir(parents=True, exist_ok=True)
runtime_config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
print(json.dumps(config["server_runtime"], ensure_ascii=False, indent=2))
PY

python -m experiments.mbu_netpp.train \
  --config "${CONFIG_PATH}" \
  --run-all-folds

export TRAIN_OUTPUT
BEST_CHECKPOINT="$(
python - <<'PY'
import json
import os
from pathlib import Path

summary_path = Path(os.environ["TRAIN_OUTPUT"]) / "crossval_summary.json"
summary = json.loads(summary_path.read_text(encoding="utf-8"))
best = max(summary.get("fold_results", []), key=lambda item: float(item.get("best_metric", 0.0)))
print(best["checkpoint_path"])
PY
)"

if [[ "${BEST_CHECKPOINT}" != /* ]]; then
  BEST_CHECKPOINT="${REPO_ROOT}/${BEST_CHECKPOINT}"
fi

python -m experiments.mbu_netpp.infer \
  --checkpoint "${BEST_CHECKPOINT}" \
  --input "${FINAL_INFER_INPUT}" \
  --output-dir "${FINAL_INFER_OUTPUT}" \
  --device auto

python -m experiments.mbu_netpp.filter_inference_outputs \
  --prepared-root "${MERGED_PREPARED}" \
  --inference-root "${FINAL_INFER_OUTPUT}" \
  --output-root "${FILTERED_INFER_OUTPUT}"

mkdir -p "${DELIVERY_ROOT}"
cp -r "${FINAL_INFER_OUTPUT}" "${DELIVERY_DIR}_all_100"
cp -r "${FILTERED_INFER_OUTPUT}" "${DELIVERY_DIR}_excluding_train"
mkdir -p "${DELIVERY_DIR}"
mv "${DELIVERY_DIR}_all_100" "${DELIVERY_DIR}/infer_all_100"
mv "${DELIVERY_DIR}_excluding_train" "${DELIVERY_DIR}/infer_excluding_train"
tar -czf "${DELIVERY_ROOT}/${DELIVERY_NAME}.tar.gz" -C "${DELIVERY_ROOT}" "${DELIVERY_NAME}"

python - <<PY
import json
from pathlib import Path

payload = {
    "best_checkpoint": "${BEST_CHECKPOINT}",
    "final_infer_output": "${FINAL_INFER_OUTPUT}",
    "filtered_infer_output": "${FILTERED_INFER_OUTPUT}",
    "delivery_dir": "${DELIVERY_DIR}",
    "delivery_archive": "${DELIVERY_ROOT}/${DELIVERY_NAME}.tar.gz",
}
Path("${DELIVERY_DIR}/run_summary.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
Path("${DELIVERY_ROOT}/${DELIVERY_NAME}_summary.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY
