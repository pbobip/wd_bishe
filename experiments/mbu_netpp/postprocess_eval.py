from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from backend.app.schemas.run import TraditionalSegConfig
from backend.app.services.algorithms.traditional import traditional_service
from backend.app.services.statistics import StatisticsService
from backend.app.utils.image_io import read_gray, write_image

from experiments.mbu_netpp.common import build_overlay, ensure_dir, image_to_tensor, save_json
from experiments.mbu_netpp.dataset import load_prepared_records
from experiments.mbu_netpp.infer import load_model_from_checkpoint
from experiments.mbu_netpp.metrics import average_metric_dicts, boundary_f1_score
from experiments.mbu_netpp.models import sliding_window_inference
from experiments.mbu_netpp.preprocess import apply_preprocess


def compute_mask_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    boundary_tolerance: int = 2,
) -> dict[str, float]:
    pred = (pred_mask > 0).astype(np.uint8)
    gt = (gt_mask > 0).astype(np.uint8)

    intersection = float(np.count_nonzero(pred & gt))
    pred_sum = float(np.count_nonzero(pred))
    gt_sum = float(np.count_nonzero(gt))
    union = pred_sum + gt_sum - intersection

    def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
        if denominator <= 0:
            return default
        return float(numerator / denominator)

    dice = safe_divide(2.0 * intersection, pred_sum + gt_sum, default=1.0 if pred_sum == gt_sum == 0 else 0.0)
    iou = safe_divide(intersection, union, default=1.0 if union == 0 else 0.0)
    precision = safe_divide(intersection, pred_sum, default=1.0 if pred_sum == 0 and gt_sum == 0 else 0.0)
    recall = safe_divide(intersection, gt_sum, default=1.0 if pred_sum == 0 and gt_sum == 0 else 0.0)
    pred_vf = float(pred.mean())
    gt_vf = float(gt.mean())
    vf_error = abs(pred_vf - gt_vf)
    boundary_f1 = boundary_f1_score(pred, gt, tolerance=boundary_tolerance)
    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "vf": float(vf_error),
        "vf_pred": float(pred_vf),
        "vf_gt": float(gt_vf),
        "boundary_f1": float(boundary_f1),
    }


def build_filter_config(min_area: int = 30, remove_border: bool = True) -> TraditionalSegConfig:
    return TraditionalSegConfig(
        method="threshold",
        fill_holes=False,
        watershed=False,
        boundary_smoothing=False,
        boundary_smoothing_kernel=1,
        min_area=int(min_area),
        remove_border=bool(remove_border),
        open_kernel=1,
        close_kernel=1,
    )


def apply_variant(
    raw_mask: np.ndarray,
    base_image: np.ndarray,
    variant: str,
    statistics_service: StatisticsService,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    normalized = variant.strip().lower()
    if normalized == "raw":
        object_stats = statistics_service.build_object_stats(raw_mask, base_image=base_image)
        final_mask = (raw_mask > 0).astype(np.uint8) * 255
        return final_mask, object_stats, {"variant": "raw"}

    filter_config = build_filter_config(min_area=30, remove_border=True)
    if normalized == "filter_small":
        object_stats = statistics_service.build_object_stats(raw_mask, config=filter_config, base_image=base_image)
        final_mask = object_stats["kept_mask"].astype(np.uint8)
        return final_mask, object_stats, {"variant": "filter_small", "min_area": 30, "remove_border": True}

    if normalized == "openclose_smooth":
        binary = (raw_mask > 0).astype(np.uint8) * 255
        binary = traditional_service._morph_cleanup(binary, open_kernel=3, close_kernel=3)
        binary = traditional_service._smooth_boundary(binary, kernel_size=3)
        object_stats = statistics_service.build_object_stats(binary, config=filter_config, base_image=base_image)
        final_mask = object_stats["kept_mask"].astype(np.uint8)
        return final_mask, object_stats, {
            "variant": "openclose_smooth",
            "open_kernel": 3,
            "close_kernel": 3,
            "smooth_kernel": 3,
            "min_area": 30,
            "remove_border": True,
        }

    raise ValueError(f"不支持的后处理变体: {variant}")


def maybe_write_xlsx(rows: list[dict[str, Any]], output_path: Path) -> bool:
    try:
        from openpyxl import Workbook
    except Exception:
        return False

    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "summary"
    if not rows:
        workbook.save(output_path)
        return True

    headers = [key for key in rows[0].keys() if not isinstance(rows[0].get(key), (list, dict))]
    worksheet.append(headers)
    for row in rows:
        worksheet.append([row.get(header) for header in headers])
    workbook.save(output_path)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估 E5 后处理对分割与统计的影响")
    parser.add_argument("--experiment-root", required=True, help="包含 fold_0..fold_n checkpoint 的实验目录")
    parser.add_argument("--prepared-root", required=True, help="prepared_main 目录")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--device", default="cuda", help="cuda/cpu/auto")
    parser.add_argument("--variants", nargs="+", default=["raw", "filter_small", "openclose_smooth"], help="后处理变体列表")
    parser.add_argument("--boundary-tolerance", type=int, default=2, help="Boundary F1 容忍像素")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    experiment_root = Path(args.experiment_root)
    prepared_root = Path(args.prepared_root)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)

    fold_dirs = sorted([item for item in experiment_root.iterdir() if item.is_dir() and item.name.startswith("fold_")])
    if not fold_dirs:
        raise FileNotFoundError(f"未在 {experiment_root} 找到 fold 目录")

    statistics_service = StatisticsService()
    rows: list[dict[str, Any]] = []

    for fold_dir in fold_dirs:
        fold_index = int(fold_dir.name.split("_")[-1])
        checkpoint_path = fold_dir / "best.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"缺少 checkpoint: {checkpoint_path}")

        model, config = load_model_from_checkpoint(checkpoint_path, config_path=None, device=device)
        data_cfg = config["data"]
        patch_size = int(data_cfg.get("patch_size", 256))
        overlap = float(config.get("inference", {}).get("overlap", data_cfg.get("overlap", 0.25)))
        normalization = str(data_cfg.get("normalization", "minmax"))
        preprocess_cfg = data_cfg.get("preprocess")
        fold_manifest_name = str(data_cfg.get("fold_manifest_name", "folds_3_seed42.json"))
        val_records = load_prepared_records(prepared_root, fold_manifest_name, fold_index, stage="val")

        for record in val_records:
            image = read_gray(prepared_root / record["image_path"])
            gt_mask = (read_gray(prepared_root / record["mask_path"]) > 127).astype(np.uint8) * 255
            processed = apply_preprocess(image, preprocess_cfg)
            image_tensor = image_to_tensor(processed, normalization=normalization).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = sliding_window_inference(model, image_tensor, patch_size=patch_size, overlap=overlap)
            raw_mask = (torch.sigmoid(logits).cpu().numpy()[0, 0] >= 0.5).astype(np.uint8) * 255

            gt_object_stats = statistics_service.build_object_stats(gt_mask, base_image=image)
            gt_summary = statistics_service.summarize(gt_mask, objects=gt_object_stats["objects"])

            for variant in args.variants:
                final_mask, object_stats, variant_meta = apply_variant(raw_mask, image, variant, statistics_service)
                summary = statistics_service.summarize(final_mask, objects=object_stats["objects"])
                metrics = compute_mask_metrics(final_mask, gt_mask, boundary_tolerance=int(args.boundary_tolerance))

                variant_dir = ensure_dir(output_dir / variant / f"fold_{fold_index}")
                write_image(variant_dir / f"{record['stem']}_mask.png", final_mask)
                write_image(variant_dir / f"{record['stem']}_overlay.png", build_overlay(image, final_mask))
                save_json(variant_dir / f"{record['stem']}.json", {"metrics": metrics, "summary": summary, "meta": variant_meta})

                row = {
                    "fold_index": fold_index,
                    "stem": record["stem"],
                    "variant": variant,
                    **metrics,
                    "pred_volume_fraction": float(summary["volume_fraction"]),
                    "gt_volume_fraction": float(gt_summary["volume_fraction"]),
                    "pred_particle_count": int(summary["particle_count"]),
                    "gt_particle_count": int(gt_summary["particle_count"]),
                    "abs_particle_count_error": abs(int(summary["particle_count"]) - int(gt_summary["particle_count"])),
                    "pred_filtered_object_count": int(summary["filtered_object_count"]),
                }
                rows.append(row)

    csv_path = output_dir / "per_image_metrics.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    maybe_write_xlsx(rows, output_dir / "per_image_metrics.xlsx")

    summaries: dict[str, dict[str, Any]] = {}
    for variant in args.variants:
        variant_rows = [row for row in rows if row["variant"] == variant]
        metric_keys = [
            "dice",
            "iou",
            "precision",
            "recall",
            "vf",
            "boundary_f1",
            "abs_particle_count_error",
            "pred_particle_count",
            "pred_filtered_object_count",
        ]
        summaries[variant] = {
            "num_images": len(variant_rows),
            "mean": average_metric_dicts([{key: row[key] for key in metric_keys} for row in variant_rows]),
        }

    save_json(output_dir / "summary.json", {"experiment_root": str(experiment_root), "variants": summaries})
    print(json.dumps({"num_rows": len(rows), "output_dir": str(output_dir), "variants": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
