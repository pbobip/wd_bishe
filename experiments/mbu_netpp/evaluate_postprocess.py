from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from backend.app.schemas.run import TraditionalSegConfig
from backend.app.services.algorithms.traditional import traditional_service
from backend.app.services.statistics import StatisticsService

from experiments.mbu_netpp.common import ensure_dir, load_yaml, resolve_device, save_json
from experiments.mbu_netpp.dataset import SEMSegmentationDataset
from experiments.mbu_netpp.metrics import average_metric_dicts, boundary_f1_score
from experiments.mbu_netpp.models import build_model_for_checkpoint, sliding_window_inference


def compute_binary_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    boundary_tolerance: int = 2,
) -> dict[str, float]:
    pred = (pred_mask > 0).astype(np.uint8)
    target = (gt_mask > 0).astype(np.uint8)

    intersection = float(np.count_nonzero(pred & target))
    pred_sum = float(np.count_nonzero(pred))
    target_sum = float(np.count_nonzero(target))
    union = pred_sum + target_sum - intersection

    dice = 1.0 if pred_sum == 0 and target_sum == 0 else (2.0 * intersection / max(pred_sum + target_sum, 1.0))
    iou = 1.0 if union == 0 else (intersection / union)
    precision = 1.0 if pred_sum == 0 and target_sum == 0 else (intersection / max(pred_sum, 1.0))
    recall = 1.0 if pred_sum == 0 and target_sum == 0 else (intersection / max(target_sum, 1.0))
    pred_vf = float(pred.mean())
    gt_vf = float(target.mean())
    vf = abs(pred_vf - gt_vf)
    boundary_f1 = boundary_f1_score(pred, target, tolerance=boundary_tolerance)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "vf": float(vf),
        "vf_pred": float(pred_vf),
        "vf_gt": float(gt_vf),
        "boundary_f1": float(boundary_f1),
    }


def apply_postprocess(mask: np.ndarray, strategy: str, min_area: int, kernel_size: int) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8) * 255
    strategy = str(strategy).strip().lower()
    if strategy == "raw":
        return binary
    if strategy == "remove_small":
        stats = StatisticsService().build_object_stats(
            binary,
            config=TraditionalSegConfig(min_area=int(min_area), remove_border=False),
        )
        return stats["kept_mask"].astype(np.uint8)
    if strategy == "morph_smooth":
        processed = traditional_service._fill_holes(binary)
        processed = traditional_service._smooth_boundary(processed, kernel_size)
        processed = traditional_service._morph_cleanup(processed, kernel_size, kernel_size)
        return (processed > 0).astype(np.uint8) * 255
    raise ValueError(f"不支持的后处理策略: {strategy}")


def maybe_write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        return
    headers = sorted({key for row in rows for key in row.keys()})
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in headers})


def load_model(checkpoint_path: Path, model_config: dict[str, Any], device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = build_model_for_checkpoint(model_config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model


def build_val_loader(config: dict[str, Any], fold_index: int) -> DataLoader:
    data_cfg = config["data"]
    training_cfg = config["training"]
    dataset = SEMSegmentationDataset(
        prepared_root=data_cfg["prepared_root"],
        fold_manifest_name=data_cfg["fold_manifest_name"],
        fold_index=fold_index,
        stage="val",
        patch_size=int(data_cfg["patch_size"]),
        edge_kernel=int(data_cfg["edge_kernel"]),
        normalization=str(data_cfg.get("normalization", "minmax")),
        preprocess_config=data_cfg.get("preprocess"),
        augmentation_config=None,
        samples_per_epoch=None,
    )
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=int(training_cfg.get("num_workers", 0)))


def evaluate_strategy(
    config: dict[str, Any],
    strategy: str,
    output_root: Path,
    min_area: int,
    kernel_size: int,
) -> dict[str, Any]:
    device = resolve_device(str(config["training"].get("device", "auto")))
    threshold = float(config["training"].get("threshold", 0.5))
    boundary_tolerance = int(config["training"].get("boundary_tolerance", 2))
    patch_size = int(config["data"].get("patch_size", 256))
    overlap = float(config.get("inference", {}).get("overlap", config["data"].get("overlap", 0.25)))
    num_folds = int(config["data"].get("num_folds", 3))

    fold_results: list[dict[str, Any]] = []
    per_image_rows: list[dict[str, Any]] = []
    stats_service = StatisticsService()

    for fold_index in range(num_folds):
        checkpoint_path = Path(config["experiment"]["output_root"]) / f"fold_{fold_index}" / "best.pt"
        model = load_model(checkpoint_path, config["model"], device=device)
        val_loader = build_val_loader(config, fold_index)

        fold_metrics: list[dict[str, float]] = []
        for batch in val_loader:
            image = batch["image"].to(device)
            gt_mask = (batch["mask"].cpu().numpy()[0, 0] > 0.5).astype(np.uint8) * 255
            stem = str(batch["stem"][0])
            with torch.no_grad():
                logits = sliding_window_inference(model, image, patch_size=patch_size, overlap=overlap)
            pred_mask = (torch.sigmoid(logits).cpu().numpy()[0, 0] >= threshold).astype(np.uint8) * 255
            processed_mask = apply_postprocess(pred_mask, strategy=strategy, min_area=min_area, kernel_size=kernel_size)
            metrics = compute_binary_metrics(processed_mask, gt_mask, boundary_tolerance=boundary_tolerance)
            summary = stats_service.summarize(processed_mask)
            metrics["volume_fraction"] = float(summary["volume_fraction"])
            metrics["particle_count"] = float(summary["particle_count"])
            metrics["object_count"] = float(summary["object_count"])
            fold_metrics.append(metrics)
            per_image_rows.append(
                {
                    "strategy": strategy,
                    "fold_index": fold_index,
                    "stem": stem,
                    **metrics,
                }
            )

        fold_summary = average_metric_dicts(fold_metrics)
        fold_results.append(
            {
                "fold_index": fold_index,
                "checkpoint_path": str(checkpoint_path),
                "summary": fold_summary,
            }
        )

    result = {
        "strategy": strategy,
        "min_area": int(min_area),
        "kernel_size": int(kernel_size),
        "fold_results": fold_results,
        "mean_summary": average_metric_dicts([item["summary"] for item in fold_results]),
    }
    save_json(output_root / f"{strategy}_summary.json", result)
    maybe_write_csv([row for row in per_image_rows if row["strategy"] == strategy], output_root / f"{strategy}_per_image.csv")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估 E5 后处理策略")
    parser.add_argument("--config", required=True, help="已有实验配置文件")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--strategies", nargs="+", default=["raw", "remove_small", "morph_smooth"])
    parser.add_argument("--min-area", type=int, default=30)
    parser.add_argument("--kernel-size", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    output_root = ensure_dir(args.output_dir)
    results = [
        evaluate_strategy(
            config=config,
            strategy=strategy,
            output_root=output_root,
            min_area=int(args.min_area),
            kernel_size=int(args.kernel_size),
        )
        for strategy in args.strategies
    ]
    summary = {
        "experiment": config["experiment"]["name"],
        "strategies": results,
    }
    save_json(output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
