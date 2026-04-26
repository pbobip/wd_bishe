from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from backend.app.services.statistics import StatisticsService
from backend.app.utils.image_io import read_gray, write_image

from experiments.mbu_netpp.common import build_overlay, ensure_dir, image_to_tensor, save_json
from experiments.mbu_netpp.dataset import load_prepared_records
from experiments.mbu_netpp.infer import load_model_from_checkpoint, maybe_write_xlsx
from experiments.mbu_netpp.metrics import average_metric_dicts, compute_segmentation_metrics
from experiments.mbu_netpp.models import sliding_window_inference
from experiments.mbu_netpp.preprocess import apply_preprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对固定 holdout 测试集进行模型评估")
    parser.add_argument("--checkpoint", required=True, help="best.pt 路径")
    parser.add_argument("--prepared-root", required=True, help="prepared_root 路径")
    parser.add_argument("--fold-manifest-name", required=True, help="包含 test_stems 的 fold manifest 文件名")
    parser.add_argument("--fold-index", type=int, default=0, help="读取 test_stems 时使用的 fold index")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--config", default=None, help="可选 YAML 配置，缺省时从 checkpoint 读取")
    parser.add_argument("--threshold", type=float, default=None, help="覆盖默认阈值")
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    prepared_root = Path(args.prepared_root)
    output_dir = ensure_dir(args.output_dir)
    predictions_dir = ensure_dir(output_dir / "predictions")
    overlays_dir = ensure_dir(output_dir / "overlays")
    stats_dir = ensure_dir(output_dir / "stats")

    model, config = load_model_from_checkpoint(args.checkpoint, args.config, device=device)
    data_cfg = config["data"]
    training_cfg = config["training"]
    inference_cfg = config.get("inference", {})
    threshold = float(args.threshold if args.threshold is not None else training_cfg.get("threshold", 0.5))
    patch_size = int(data_cfg.get("patch_size", 256))
    overlap = float(inference_cfg.get("overlap", data_cfg.get("overlap", 0.25)))
    normalization = str(data_cfg.get("normalization", "minmax"))
    preprocess_cfg = data_cfg.get("preprocess")
    boundary_tolerance = int(training_cfg.get("boundary_tolerance", 2))

    statistics_service = StatisticsService()
    records = load_prepared_records(
        prepared_root=prepared_root,
        fold_manifest_name=args.fold_manifest_name,
        fold_index=int(args.fold_index),
        stage="test",
    )
    if not records:
        raise ValueError("当前 fold manifest 没有 test_stems")

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for record in records:
            image = read_gray(prepared_root / record["image_path"])
            mask = read_gray(prepared_root / record["mask_path"])
            processed = apply_preprocess(image, preprocess_cfg)

            image_tensor = image_to_tensor(processed, normalization=normalization).unsqueeze(0).to(device)
            target_tensor = torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

            logits = sliding_window_inference(model, image_tensor, patch_size=patch_size, overlap=overlap)
            metrics = compute_segmentation_metrics(
                logits=logits,
                target_mask=target_tensor,
                threshold=threshold,
                boundary_tolerance=boundary_tolerance,
            )
            pred_mask = (torch.sigmoid(logits).cpu().numpy()[0, 0] >= threshold).astype(np.uint8) * 255
            overlay = build_overlay(image, pred_mask)

            stem = str(record["stem"])
            write_image(predictions_dir / f"{stem}_pred.png", pred_mask)
            write_image(overlays_dir / f"{stem}_overlay.png", overlay)

            pred_stats = statistics_service.summarize(pred_mask)
            save_json(stats_dir / f"{stem}.json", pred_stats)

            row = {
                "stem": stem,
                "dice": float(metrics["dice"]),
                "iou": float(metrics["iou"]),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "vf": float(metrics["vf"]),
                "vf_pred": float(metrics["vf_pred"]),
                "vf_gt": float(metrics["vf_gt"]),
                "boundary_f1": float(metrics["boundary_f1"]),
                "pred_volume_fraction": float(pred_stats.get("volume_fraction", 0.0)),
                "particle_count": int(pred_stats.get("particle_count", 0)),
                "prediction_path": str(predictions_dir / f"{stem}_pred.png"),
                "overlay_path": str(overlays_dir / f"{stem}_overlay.png"),
            }
            rows.append(row)

    headers = [
        "stem",
        "dice",
        "iou",
        "precision",
        "recall",
        "vf",
        "vf_pred",
        "vf_gt",
        "boundary_f1",
        "pred_volume_fraction",
        "particle_count",
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
        "checkpoint": str(args.checkpoint),
        "prepared_root": str(prepared_root),
        "fold_manifest_name": args.fold_manifest_name,
        "fold_index": int(args.fold_index),
        "num_images": len(rows),
        "metrics": average_metric_dicts(
            [
                {
                    "dice": float(row["dice"]),
                    "iou": float(row["iou"]),
                    "precision": float(row["precision"]),
                    "recall": float(row["recall"]),
                    "vf": float(row["vf"]),
                    "vf_pred": float(row["vf_pred"]),
                    "vf_gt": float(row["vf_gt"]),
                    "boundary_f1": float(row["boundary_f1"]),
                }
                for row in rows
            ]
        ),
        "per_image_csv": str(csv_path),
    }
    save_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
