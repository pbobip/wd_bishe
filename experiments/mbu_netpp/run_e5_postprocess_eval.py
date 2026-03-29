from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from backend.app.services.statistics import StatisticsService
from backend.app.utils.image_io import read_gray, write_image

from experiments.mbu_netpp.common import build_overlay, ensure_dir, image_to_tensor, save_json
from experiments.mbu_netpp.dataset import load_prepared_records
from experiments.mbu_netpp.infer import load_model_from_checkpoint, maybe_write_xlsx
from experiments.mbu_netpp.metrics import average_metric_dicts, compute_binary_segmentation_metrics
from experiments.mbu_netpp.models import sliding_window_inference
from experiments.mbu_netpp.preprocess import apply_preprocess


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    kept = np.zeros_like(binary, dtype=np.uint8)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= int(min_area):
            kept[labels == label] = 255
    return kept


def morph_open_close_smooth(mask: np.ndarray, open_kernel: int, close_kernel: int, smooth_kernel: int) -> np.ndarray:
    result = (mask > 0).astype(np.uint8) * 255
    if int(open_kernel) > 1:
        open_kernel = max(1, int(open_kernel))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel, iterations=1)
    if int(close_kernel) > 1:
        close_kernel = max(1, int(close_kernel))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=1)
    if int(smooth_kernel) > 1:
        smooth_kernel = max(1, int(smooth_kernel))
        if smooth_kernel % 2 == 0:
            smooth_kernel += 1
        result = cv2.medianBlur(result, smooth_kernel)
        result = ((result > 127).astype(np.uint8) * 255)
    return result


def variant_masks(raw_mask: np.ndarray, min_area: int, open_kernel: int, close_kernel: int, smooth_kernel: int) -> dict[str, np.ndarray]:
    return {
        "raw": raw_mask,
        "remove_small_cc": remove_small_components(raw_mask, min_area=min_area),
        "open_close_smooth": morph_open_close_smooth(
            raw_mask,
            open_kernel=open_kernel,
            close_kernel=close_kernel,
            smooth_kernel=smooth_kernel,
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 E5 后处理与统计稳定性评估")
    parser.add_argument("--prepared-root", required=True, help="监督数据 prepared_root")
    parser.add_argument("--fold-manifest-name", default="folds_3_seed42.json", help="fold manifest 名称")
    parser.add_argument("--checkpoint-root", required=True, help="含 fold_0/fold_1/fold_2 的实验输出目录")
    parser.add_argument("--output-dir", required=True, help="E5 输出目录")
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda")
    parser.add_argument("--threshold", type=float, default=None, help="可选覆盖阈值")
    parser.add_argument("--min-area", type=int, default=30, help="去小连通域阈值")
    parser.add_argument("--open-kernel", type=int, default=3, help="开运算核")
    parser.add_argument("--close-kernel", type=int, default=3, help="闭运算核")
    parser.add_argument("--smooth-kernel", type=int, default=3, help="平滑核")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    prepared_root = Path(args.prepared_root)
    checkpoint_root = Path(args.checkpoint_root)
    output_dir = ensure_dir(args.output_dir)
    predictions_dir = ensure_dir(output_dir / "predictions")
    overlays_dir = ensure_dir(output_dir / "overlays")

    fold_manifest = json.loads((prepared_root / "manifests" / args.fold_manifest_name).read_text(encoding="utf-8"))
    num_folds = int(fold_manifest["num_folds"])

    rows: list[dict[str, Any]] = []
    statistics_service = StatisticsService()

    for fold_index in range(num_folds):
        checkpoint_path = checkpoint_root / f"fold_{fold_index}" / "best.pt"
        model, config = load_model_from_checkpoint(checkpoint_path, config_path=None, device=device)

        data_cfg = config["data"]
        training_cfg = config["training"]
        threshold = float(args.threshold if args.threshold is not None else training_cfg.get("threshold", 0.5))
        patch_size = int(data_cfg.get("patch_size", 256))
        overlap = float(data_cfg.get("overlap", 0.25))
        normalization = str(data_cfg.get("normalization", "minmax"))
        preprocess_cfg = data_cfg.get("preprocess")
        boundary_tolerance = int(training_cfg.get("boundary_tolerance", 2))

        records = load_prepared_records(prepared_root, args.fold_manifest_name, fold_index=fold_index, stage="val")
        for record in records:
            image = read_gray(prepared_root / record["image_path"])
            gt_mask = read_gray(prepared_root / record["mask_path"])
            processed = apply_preprocess(image, preprocess_cfg)
            image_tensor = image_to_tensor(processed, normalization=normalization).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = sliding_window_inference(model, image_tensor, patch_size=patch_size, overlap=overlap)
            raw_mask = (torch.sigmoid(logits).cpu().numpy()[0, 0] >= threshold).astype(np.uint8) * 255

            for variant_name, variant_mask in variant_masks(
                raw_mask,
                min_area=int(args.min_area),
                open_kernel=int(args.open_kernel),
                close_kernel=int(args.close_kernel),
                smooth_kernel=int(args.smooth_kernel),
            ).items():
                metrics = compute_binary_segmentation_metrics(
                    pred_mask=variant_mask,
                    target_mask=(gt_mask > 127).astype(np.uint8),
                    boundary_tolerance=boundary_tolerance,
                )
                object_stats = statistics_service.build_object_stats(variant_mask, base_image=image)
                summary = statistics_service.summarize(variant_mask, objects=object_stats["objects"])

                prediction_path = predictions_dir / variant_name / f"fold_{fold_index}"
                overlay_path = overlays_dir / variant_name / f"fold_{fold_index}"
                ensure_dir(prediction_path)
                ensure_dir(overlay_path)
                pred_file = prediction_path / f"{record['stem']}.png"
                overlay_file = overlay_path / f"{record['stem']}_overlay.png"
                write_image(pred_file, variant_mask)
                write_image(overlay_file, build_overlay(image, variant_mask))

                rows.append(
                    {
                        "fold_index": fold_index,
                        "variant": variant_name,
                        "stem": record["stem"],
                        "dice": float(metrics["dice"]),
                        "iou": float(metrics["iou"]),
                        "precision": float(metrics["precision"]),
                        "recall": float(metrics["recall"]),
                        "vf": float(metrics["vf"]),
                        "vf_pred": float(metrics["vf_pred"]),
                        "vf_gt": float(metrics["vf_gt"]),
                        "boundary_f1": float(metrics["boundary_f1"]),
                        "particle_count": int(summary["particle_count"]),
                        "filtered_object_count": int(summary["filtered_object_count"]),
                        "volume_fraction": float(summary["volume_fraction"]),
                        "prediction_path": str(pred_file),
                        "overlay_path": str(overlay_file),
                    }
                )

    headers = [
        "fold_index",
        "variant",
        "stem",
        "dice",
        "iou",
        "precision",
        "recall",
        "vf",
        "vf_pred",
        "vf_gt",
        "boundary_f1",
        "particle_count",
        "filtered_object_count",
        "volume_fraction",
        "prediction_path",
        "overlay_path",
    ]
    csv_path = output_dir / "per_image_metrics.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in headers})
    maybe_write_xlsx(rows, output_dir / "per_image_metrics.xlsx")

    summary = {
        "prepared_root": str(prepared_root),
        "checkpoint_root": str(checkpoint_root),
        "num_rows": len(rows),
        "variants": {
            variant: {
                "num_images": len([row for row in rows if row["variant"] == variant]),
                "metrics": average_metric_dicts(
                    [
                        {
                            "dice": row["dice"],
                            "iou": row["iou"],
                            "precision": row["precision"],
                            "recall": row["recall"],
                            "vf": row["vf"],
                            "vf_pred": row["vf_pred"],
                            "vf_gt": row["vf_gt"],
                            "boundary_f1": row["boundary_f1"],
                            "particle_count": row["particle_count"],
                            "filtered_object_count": row["filtered_object_count"],
                            "volume_fraction": row["volume_fraction"],
                        }
                        for row in rows
                        if row["variant"] == variant
                    ]
                ),
            }
            for variant in sorted({row["variant"] for row in rows})
        },
        "per_image_csv": str(csv_path),
    }
    save_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
