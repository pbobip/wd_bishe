from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import torch

from backend.app.utils.image_io import read_gray

from experiments.mbu_netpp.common import ensure_dir, image_to_tensor, save_json
from experiments.mbu_netpp.dataset import load_prepared_records
from experiments.mbu_netpp.infer import load_model_from_checkpoint
from experiments.mbu_netpp.models import sliding_window_inference
from experiments.mbu_netpp.preprocess import apply_preprocess


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_checkpoint() -> Path:
    return repo_root() / "results" / "opt_real53_suite" / "experiments" / "opt_real53_boundary_sampling" / "best.pt"


def default_prepared_root() -> Path:
    return repo_root() / "experiments" / "mbu_netpp" / "workdir" / "prepared_real53_cv5"


def default_output_dir() -> Path:
    return repo_root() / "results" / "paper_supplement" / "model_efficiency"


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
    }


def count_conv_linear_macs(model: torch.nn.Module, input_shape: tuple[int, int, int, int], device: torch.device) -> dict[str, Any]:
    macs_by_type: dict[str, int] = {"Conv2d": 0, "Linear": 0}
    handles = []

    def conv_hook(module: torch.nn.Conv2d, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        if not isinstance(output, torch.Tensor):
            return
        out = output
        batch_size = int(out.shape[0])
        out_channels = int(out.shape[1])
        out_h = int(out.shape[2])
        out_w = int(out.shape[3])
        kernel_h, kernel_w = module.kernel_size
        in_channels = int(module.in_channels)
        groups = int(module.groups)
        macs = batch_size * out_channels * out_h * out_w * (in_channels // groups) * kernel_h * kernel_w
        macs_by_type["Conv2d"] += int(macs)

    def linear_hook(module: torch.nn.Linear, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        if not isinstance(output, torch.Tensor):
            return
        batch = int(output.numel() // max(1, module.out_features))
        macs = batch * int(module.in_features) * int(module.out_features)
        macs_by_type["Linear"] += int(macs)

    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, torch.nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))

    was_training = model.training
    model.eval()
    dummy = torch.zeros(input_shape, device=device)
    with torch.no_grad():
        model(dummy)
    for handle in handles:
        handle.remove()
    if was_training:
        model.train()

    macs = int(sum(macs_by_type.values()))
    return {
        "input_shape": list(input_shape),
        "macs_conv_linear": macs,
        "flops_conv_linear_multiply_add": int(macs * 2),
        "gmacs_conv_linear": macs / 1e9,
        "gflops_conv_linear_multiply_add": (macs * 2) / 1e9,
        "macs_by_type": macs_by_type,
        "flop_counting_note": "Only Conv2d and Linear layers are counted. FLOPs use multiply-add=2 convention; interpolation, activations and Python overhead are not included.",
    }


def count_sliding_windows(height: int, width: int, patch_size: int, overlap: float) -> dict[str, int]:
    stride = max(1, int(patch_size * (1.0 - overlap)))
    padded_height = max(height, patch_size)
    padded_width = max(width, patch_size)
    y_positions = list(range(0, max(1, padded_height - patch_size + 1), stride))
    x_positions = list(range(0, max(1, padded_width - patch_size + 1), stride))
    if y_positions[-1] != padded_height - patch_size:
        y_positions.append(padded_height - patch_size)
    if x_positions[-1] != padded_width - patch_size:
        x_positions.append(padded_width - patch_size)
    return {
        "stride": stride,
        "num_y_windows": len(y_positions),
        "num_x_windows": len(x_positions),
        "num_windows": len(y_positions) * len(x_positions),
    }


def benchmark_patch_forward(
    model: torch.nn.Module,
    device: torch.device,
    patch_size: int,
    warmup: int,
    repeats: int,
) -> dict[str, float]:
    dummy = torch.zeros((1, 1, patch_size, patch_size), device=device)
    with torch.no_grad():
        for _ in range(max(0, warmup)):
            model(dummy)
        synchronize(device)
        times: list[float] = []
        for _ in range(max(1, repeats)):
            start = time.perf_counter()
            model(dummy)
            synchronize(device)
            times.append(time.perf_counter() - start)
    return {
        "patch_forward_repeats": float(repeats),
        "patch_forward_mean_seconds": float(sum(times) / len(times)),
        "patch_forward_min_seconds": float(min(times)),
        "patch_forward_max_seconds": float(max(times)),
    }


def benchmark_holdout(
    model: torch.nn.Module,
    config: dict[str, Any],
    prepared_root: Path,
    fold_manifest_name: str,
    fold_index: int,
    device: torch.device,
    limit_images: int,
    warmup: int,
    repeats: int,
) -> list[dict[str, Any]]:
    data_cfg = config["data"]
    inference_cfg = config.get("inference", {})
    patch_size = int(data_cfg.get("patch_size", 256))
    overlap = float(inference_cfg.get("overlap", data_cfg.get("overlap", 0.25)))
    normalization = str(data_cfg.get("normalization", "minmax"))
    preprocess_cfg = data_cfg.get("preprocess")

    records = load_prepared_records(
        prepared_root=prepared_root,
        fold_manifest_name=fold_manifest_name,
        fold_index=fold_index,
        stage="test",
    )
    if limit_images > 0:
        records = records[:limit_images]

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for record in records:
            image = read_gray(prepared_root / record["image_path"])
            preprocess_start = time.perf_counter()
            processed = apply_preprocess(image, preprocess_cfg)
            image_tensor = image_to_tensor(processed, normalization=normalization).unsqueeze(0).to(device)
            synchronize(device)
            preprocess_seconds = time.perf_counter() - preprocess_start

            for _ in range(max(0, warmup)):
                logits = sliding_window_inference(model, image_tensor, patch_size=patch_size, overlap=overlap)
                _ = torch.sigmoid(logits)
            synchronize(device)

            inference_times: list[float] = []
            for _ in range(max(1, repeats)):
                infer_start = time.perf_counter()
                logits = sliding_window_inference(model, image_tensor, patch_size=patch_size, overlap=overlap)
                _ = torch.sigmoid(logits)
                synchronize(device)
                inference_times.append(time.perf_counter() - infer_start)
            inference_seconds = average(inference_times)

            window_info = count_sliding_windows(int(image.shape[0]), int(image.shape[1]), patch_size, overlap)
            rows.append(
                {
                    "stem": str(record["stem"]),
                    "height": int(image.shape[0]),
                    "width": int(image.shape[1]),
                    **window_info,
                    "holdout_warmup": int(warmup),
                    "holdout_repeats": int(repeats),
                    "preprocess_seconds": float(preprocess_seconds),
                    "inference_seconds": float(inference_seconds),
                    "inference_min_seconds": float(min(inference_times)),
                    "inference_max_seconds": float(max(inference_times)),
                    "total_seconds": float(preprocess_seconds + inference_seconds),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def average(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute MBU-Net++ paper efficiency metrics")
    parser.add_argument("--checkpoint", default=str(default_checkpoint()))
    parser.add_argument("--config", default="")
    parser.add_argument("--prepared-root", default=str(default_prepared_root()))
    parser.add_argument("--fold-manifest-name", default="folds_5_seed42_holdout10.json")
    parser.add_argument("--fold-index", type=int, default=0)
    parser.add_argument("--output-dir", default=str(default_output_dir()))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--holdout-warmup", type=int, default=0)
    parser.add_argument("--holdout-repeats", type=int, default=1)
    parser.add_argument("--limit-images", type=int, default=0, help="0 means all holdout images")
    parser.add_argument("--skip-holdout-timing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    device = resolve_device(str(args.device))
    model, config = load_model_from_checkpoint(args.checkpoint, args.config or None, device=device)
    data_cfg = config["data"]
    patch_size = int(data_cfg.get("patch_size", 256))

    parameter_summary = count_parameters(model)
    flops_summary = count_conv_linear_macs(model, (1, 1, patch_size, patch_size), device=device)
    patch_timing = benchmark_patch_forward(
        model=model,
        device=device,
        patch_size=patch_size,
        warmup=int(args.warmup),
        repeats=int(args.repeats),
    )

    holdout_rows: list[dict[str, Any]] = []
    if not args.skip_holdout_timing:
        holdout_rows = benchmark_holdout(
            model=model,
            config=config,
            prepared_root=Path(args.prepared_root),
            fold_manifest_name=str(args.fold_manifest_name),
            fold_index=int(args.fold_index),
            device=device,
            limit_images=int(args.limit_images),
            warmup=int(args.holdout_warmup),
            repeats=int(args.holdout_repeats),
        )
        write_csv(output_dir / "holdout_timing.csv", holdout_rows)

    timing_summary = {
        **patch_timing,
        "holdout_images_timed": len(holdout_rows),
        "holdout_mean_inference_seconds": average([float(row["inference_seconds"]) for row in holdout_rows]),
        "holdout_mean_total_seconds": average([float(row["total_seconds"]) for row in holdout_rows]),
        "holdout_mean_windows": average([float(row["num_windows"]) for row in holdout_rows]),
    }
    timing_summary["estimated_patch_gflops_multiply_add"] = float(
        flops_summary["gflops_conv_linear_multiply_add"]
    )
    timing_summary["estimated_mean_full_image_gflops_multiply_add"] = (
        float(timing_summary["holdout_mean_windows"]) * float(flops_summary["gflops_conv_linear_multiply_add"])
    )

    report = {
        "checkpoint": str(Path(args.checkpoint)),
        "config": str(args.config or "checkpoint_embedded_config"),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "parameter_summary": parameter_summary,
        "flops_summary": flops_summary,
        "timing_summary": timing_summary,
    }
    save_json(output_dir / "model_efficiency_summary.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
