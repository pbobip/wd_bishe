#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
SERVER_ASSETS_ROOT="${SERVER_ASSETS_ROOT:-$REPO_ROOT/server_assets}"
TRAIN_SOURCE="${TRAIN_SOURCE:-$SERVER_ASSETS_ROOT/training_sources/real53}"
PREPARED_ROOT="${PREPARED_ROOT:-$REPO_ROOT/experiments/mbu_netpp/workdir/prepared_real53_cv5}"
MICRONET_CHECKPOINT="${MICRONET_CHECKPOINT:-$REPO_ROOT/experiments/mbu_netpp/workdir/weights/se_resnext50_32x4d_pretrained_microscopynet_v1.0.pth.tar}"
RUNTIME_CONFIG_DIR="${RUNTIME_CONFIG_DIR:-$REPO_ROOT/experiments/mbu_netpp/workdir/runtime_configs_real53_holdout10}"
SUITE_OUTPUT_ROOT="${SUITE_OUTPUT_ROOT:-$REPO_ROOT/experiments/mbu_netpp/outputs/real53_holdout10_suite}"
TRADITIONAL_OUTPUT_ROOT="${TRADITIONAL_OUTPUT_ROOT:-$REPO_ROOT/experiments/mbu_netpp/outputs/real53_holdout10_traditional_eval}"
DELIVERY_ROOT="${DELIVERY_ROOT:-$REPO_ROOT/results/server_delivery}"
DELIVERY_NAME="${DELIVERY_NAME:-real53_full_experiments}"
DELIVERY_DIR="${DELIVERY_DIR:-$DELIVERY_ROOT/$DELIVERY_NAME}"
PYTHON_BIN="${PYTHON_BIN:-}"
AUTO_INSTALL_MISSING="${AUTO_INSTALL_MISSING:-1}"
CLEAN_RUN="${CLEAN_RUN:-1}"
NUM_TEST="${NUM_TEST:-10}"
NUM_FOLDS="${NUM_FOLDS:-5}"
SEED="${SEED:-42}"
PIPELINE_LOG="${PIPELINE_LOG:-$DELIVERY_ROOT/${DELIVERY_NAME}_pipeline.log}"

BASE_CONFIGS=(
  "real53_cv5_unet_gpu.yaml"
  "real53_cv5_unetpp_nopretrain.yaml"
  "real53_cv5_unetpp_imagenet.yaml"
  "real53_cv5_unetpp_micronet.yaml"
  "real53_cv5_mbu_edge.yaml"
  "real53_cv5_mbu_edge_deep.yaml"
  "real53_cv5_mbu_edge_deep_vf.yaml"
  "real53_cv5_deeplabv3plus_imagenet.yaml"
  "real53_cv5_deeplabv3plus_micronet.yaml"
)

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

mkdir -p "${DELIVERY_ROOT}"
exec > >(tee -a "${PIPELINE_LOG}") 2>&1

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

log_runtime() {
  local label="$1"
  local seconds="$2"
  local rc="$3"
  printf '%s,%s,%s\n' "${label}" "${seconds}" "${rc}" >> "${DELIVERY_ROOT}/${DELIVERY_NAME}_runtime_records.csv"
}

run_timed() {
  local label="$1"
  shift
  local start_ts end_ts rc
  start_ts="$(date +%s)"
  set +e
  "$@"
  rc=$?
  set -e
  end_ts="$(date +%s)"
  log_runtime "${label}" "$((end_ts - start_ts))" "${rc}"
  if [[ "${rc}" -ne 0 ]]; then
    echo "步骤失败: ${label}" >&2
    exit "${rc}"
  fi
}

PYTHON_BIN="$(resolve_python)"
echo "使用 Python: ${PYTHON_BIN}"

if [[ ! -d "${REPO_ROOT}" ]]; then
  echo "仓库根目录不存在: ${REPO_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${TRAIN_SOURCE}" ]]; then
  echo "训练源目录不存在: ${TRAIN_SOURCE}" >&2
  exit 1
fi
if [[ ! -f "${MICRONET_CHECKPOINT}" ]]; then
  echo "未找到 MicroNet 权重: ${MICRONET_CHECKPOINT}" >&2
  exit 1
fi

printf 'label,seconds,return_code\n' > "${DELIVERY_ROOT}/${DELIVERY_NAME}_runtime_records.csv"

ENV_REPORT_PATH="${DELIVERY_ROOT}/${DELIVERY_NAME}_environment_report.json"
"${PYTHON_BIN}" - <<PY
import json
import os
import platform
from pathlib import Path

payload = {
    "python_executable": r"${PYTHON_BIN}",
    "python_version": platform.python_version(),
    "platform": platform.platform(),
}
try:
    import torch
    payload["torch_version"] = torch.__version__
    payload["cuda_available"] = bool(torch.cuda.is_available())
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        payload["gpu_name"] = props.name
        payload["gpu_total_gb"] = round(props.total_memory / (1024 ** 3), 2)
except Exception as exc:
    payload["torch_import_error"] = repr(exc)

Path(r"${ENV_REPORT_PATH}").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY

if ! "${PYTHON_BIN}" - <<'PY'
import torch
assert torch.cuda.is_available(), "当前环境没有可用 CUDA"
print(torch.__version__)
PY
then
  echo "当前环境不可用 GPU，请先检查远端环境。" >&2
  exit 1
fi

if [[ "${AUTO_INSTALL_MISSING}" == "1" ]]; then
  run_timed "install:non_torch_requirements" "${PYTHON_BIN}" -m pip install "${NON_TORCH_PACKAGES[@]}"
fi

cd "${REPO_ROOT}"

if [[ "${CLEAN_RUN}" == "1" ]]; then
  rm -rf "${PREPARED_ROOT}" "${RUNTIME_CONFIG_DIR}" "${SUITE_OUTPUT_ROOT}" "${TRADITIONAL_OUTPUT_ROOT}" "${DELIVERY_DIR}"
  rm -f "${DELIVERY_ROOT}/${DELIVERY_NAME}.tar.gz"
fi

run_timed \
  "prepare:real53" \
  "${PYTHON_BIN}" -m experiments.mbu_netpp.preparation \
    --images-dir "${TRAIN_SOURCE}" \
    --annotations-dir "${TRAIN_SOURCE}" \
    --output-root "${PREPARED_ROOT}" \
    --labels gamma_prime \
    --num-folds "${NUM_FOLDS}" \
    --seed "${SEED}"

run_timed \
  "split:holdout10" \
  "${PYTHON_BIN}" -m experiments.mbu_netpp.create_holdout_split \
    --prepared-root "${PREPARED_ROOT}" \
    --num-test "${NUM_TEST}" \
    --num-folds "${NUM_FOLDS}" \
    --seed "${SEED}" \
    --preferred-test-stem 4233 \
    --preferred-test-stem 5 \
    --preferred-test-stem 732 \
    --preferred-test-stem 7323

HOLDOUT_FOLD_MANIFEST="folds_${NUM_FOLDS}_seed${SEED}_holdout${NUM_TEST}.json"
HOLDOUT_MANIFEST="holdout_${NUM_TEST}_seed${SEED}.json"

run_timed \
  "eval:traditional" \
  "${PYTHON_BIN}" -m experiments.mbu_netpp.traditional_eval \
    --prepared-root "${PREPARED_ROOT}" \
    --fold-manifest-name "${HOLDOUT_FOLD_MANIFEST}" \
    --fold-index 0 \
    --output-dir "${TRADITIONAL_OUTPUT_ROOT}"

mkdir -p "${RUNTIME_CONFIG_DIR}" "${SUITE_OUTPUT_ROOT}"

for base_config_name in "${BASE_CONFIGS[@]}"; do
  config_stem="${base_config_name%.yaml}"
  base_config_path="${REPO_ROOT}/experiments/mbu_netpp/configs/${base_config_name}"
  runtime_config_path="${RUNTIME_CONFIG_DIR}/${config_stem}.yaml"
  experiment_output_root="${SUITE_OUTPUT_ROOT}/${config_stem}"

  if [[ ! -f "${base_config_path}" ]]; then
    echo "缺少基础配置: ${base_config_path}" >&2
    exit 1
  fi

  "${PYTHON_BIN}" - <<PY
import os
from pathlib import Path
import yaml

base_config_path = Path(r"${base_config_path}")
runtime_config_path = Path(r"${runtime_config_path}")
config = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
config["experiment"]["name"] = "${config_stem}_holdout${NUM_TEST}"
config["experiment"]["output_root"] = r"${experiment_output_root}"
config["data"]["prepared_root"] = r"${PREPARED_ROOT}"
config["data"]["num_folds"] = ${NUM_FOLDS}
config["data"]["fold_seed"] = ${SEED}
config["data"]["fold_manifest_name"] = "${HOLDOUT_FOLD_MANIFEST}"
config["data"]["auto_prepare"] = False
config["data"]["force_prepare"] = False
config["model"]["micronet_checkpoint"] = r"${MICRONET_CHECKPOINT}"
config["training"]["device"] = "cuda"
config["training"]["batch_size"] = 6
config["training"]["num_workers"] = 8
runtime_config_path.parent.mkdir(parents=True, exist_ok=True)
runtime_config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
print(runtime_config_path)
PY

  run_timed "train:${config_stem}" "${PYTHON_BIN}" -m experiments.mbu_netpp.train --config "${runtime_config_path}" --run-all-folds
  run_timed "plot:${config_stem}" "${PYTHON_BIN}" -m experiments.mbu_netpp.plot_training_curves --experiment-root "${experiment_output_root}"

  BEST_CHECKPOINT="$(
  "${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

summary = json.loads(Path(r"${experiment_output_root}") .joinpath("crossval_summary.json").read_text(encoding="utf-8"))
best = max(summary.get("fold_results", []), key=lambda item: float(item.get("best_metric", 0.0)))
print(best["checkpoint_path"])
PY
  )"

  run_timed \
    "holdout:${config_stem}" \
    "${PYTHON_BIN}" -m experiments.mbu_netpp.holdout_eval \
      --checkpoint "${BEST_CHECKPOINT}" \
      --prepared-root "${PREPARED_ROOT}" \
      --fold-manifest-name "${HOLDOUT_FOLD_MANIFEST}" \
      --fold-index 0 \
      --output-dir "${experiment_output_root}/holdout_eval" \
      --config "${runtime_config_path}" \
      --device auto
done

SUITE_SUMMARY_DIR="${SUITE_OUTPUT_ROOT}/suite_summary"
run_timed \
  "summary:suite" \
  "${PYTHON_BIN}" -m experiments.mbu_netpp.summarize_experiment_suite \
    --experiments-root "${SUITE_OUTPUT_ROOT}" \
    --output-dir "${SUITE_SUMMARY_DIR}"

mkdir -p "${DELIVERY_DIR}/manifests" "${DELIVERY_DIR}/experiments"
cp "${PREPARED_ROOT}/manifests/dataset.json" "${DELIVERY_DIR}/manifests/"
cp "${PREPARED_ROOT}/manifests/${HOLDOUT_MANIFEST}" "${DELIVERY_DIR}/manifests/"
cp "${PREPARED_ROOT}/manifests/${HOLDOUT_FOLD_MANIFEST}" "${DELIVERY_DIR}/manifests/"
cp -r "${TRADITIONAL_OUTPUT_ROOT}" "${DELIVERY_DIR}/traditional_eval"
cp -r "${SUITE_SUMMARY_DIR}" "${DELIVERY_DIR}/suite_summary"
cp "${ENV_REPORT_PATH}" "${DELIVERY_DIR}/environment_report.json"
cp "${DELIVERY_ROOT}/${DELIVERY_NAME}_runtime_records.csv" "${DELIVERY_DIR}/runtime_records.csv"
cp "${PIPELINE_LOG}" "${DELIVERY_DIR}/pipeline.log"

for base_config_name in "${BASE_CONFIGS[@]}"; do
  config_stem="${base_config_name%.yaml}"
  experiment_output_root="${SUITE_OUTPUT_ROOT}/${config_stem}"
  runtime_config_path="${RUNTIME_CONFIG_DIR}/${config_stem}.yaml"
  target_root="${DELIVERY_DIR}/experiments/${config_stem}"
  mkdir -p "${target_root}"
  cp "${runtime_config_path}" "${target_root}/runtime_config.yaml"
  cp "${experiment_output_root}/crossval_summary.json" "${target_root}/"
  cp -r "${experiment_output_root}/plots" "${target_root}/plots"
  cp -r "${experiment_output_root}/holdout_eval" "${target_root}/holdout_eval"
  BEST_CHECKPOINT="$(
  "${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

summary = json.loads(Path(r"${experiment_output_root}") .joinpath("crossval_summary.json").read_text(encoding="utf-8"))
best = max(summary.get("fold_results", []), key=lambda item: float(item.get("best_metric", 0.0)))
print(best["checkpoint_path"])
PY
  )"
  cp "${BEST_CHECKPOINT}" "${target_root}/best.pt"
  for fold_dir in "${experiment_output_root}"/fold_*; do
    if [[ -d "${fold_dir}" ]]; then
      fold_name="$(basename "${fold_dir}")"
      mkdir -p "${target_root}/${fold_name}"
      [[ -f "${fold_dir}/summary.json" ]] && cp "${fold_dir}/summary.json" "${target_root}/${fold_name}/"
      [[ -f "${fold_dir}/history.json" ]] && cp "${fold_dir}/history.json" "${target_root}/${fold_name}/"
    fi
  done
done

run_timed "package:tar" tar -czf "${DELIVERY_ROOT}/${DELIVERY_NAME}.tar.gz" -C "${DELIVERY_ROOT}" "${DELIVERY_NAME}"

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

payload = {
    "prepared_root": r"${PREPARED_ROOT}",
    "holdout_manifest": r"${PREPARED_ROOT}/manifests/${HOLDOUT_MANIFEST}",
    "holdout_fold_manifest": r"${PREPARED_ROOT}/manifests/${HOLDOUT_FOLD_MANIFEST}",
    "traditional_eval_root": r"${TRADITIONAL_OUTPUT_ROOT}",
    "suite_output_root": r"${SUITE_OUTPUT_ROOT}",
    "suite_summary_root": r"${SUITE_SUMMARY_DIR}",
    "delivery_dir": r"${DELIVERY_DIR}",
    "delivery_archive": r"${DELIVERY_ROOT}/${DELIVERY_NAME}.tar.gz",
    "environment_report": r"${ENV_REPORT_PATH}",
    "runtime_records_csv": r"${DELIVERY_ROOT}/${DELIVERY_NAME}_runtime_records.csv",
}
Path(r"${DELIVERY_DIR}/run_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
Path(r"${DELIVERY_ROOT}/${DELIVERY_NAME}_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY
