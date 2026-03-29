from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from backend.app.utils.image_io import read_gray, write_image

from experiments.mbu_netpp.common import build_overlay, ensure_dir, image_to_tensor, save_json
from experiments.mbu_netpp.external_data import collect_paired_records, convert_mask_to_binary, detect_crop_height, load_mask_array, parse_rgb_spec
from experiments.mbu_netpp.infer import load_model_from_checkpoint, maybe_write_xlsx
from experiments.mbu_netpp.metrics import average_metric_dicts, compute_segmentation_metrics
from experiments.mbu_netpp.models import sliding_window_inference
from experiments.mbu_netpp.preprocess import apply_preprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用现有 checkpoint 做外部验证")
    parser.add_argument("--checkpoint", required=True, help="best.pt 路径")
    parser.add_argument("--prepared-root", default=None, help="prepared 外部数据目录")
    parser.add_argument("--images-dir", default=None, help="原图目录")
    parser.add_argument("--masks-dir", default=None, help="掩膜目录")
    parser.add_argument("--output-dir", required=True, help="验证输出目录")
    parser.add_argument("--config", default=None, help="可选配置文件，缺省时从 checkpoint 读取")
    parser.add_argument("--threshold", type=float, default=None, help="覆盖默认阈值")
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda")
    parser.add_argument("--subset", nargs="*", default=None, help="只评估指定 subset")
    parser.add_argument("--split", nargs="*", default=None, help="只评估指定 split")
    parser.add_argument("--mask-mode", default=None, help="原始掩膜二值化策略: colored/nonzero/alpha_nonzero/exact_color")
    parser.add_argument("--foreground-colors", nargs="*", default=None, help="mask-mode=exact_color 时使用的 RGB/HEX 颜色")
    return parser.parse_args()


def load_items(prepared_root: Path, subsets: set[str] | None, splits: set[str] | None) -> list[dict[str, Any]]:
    dataset_manifest = json.loads((prepared_root / "manifests" / "dataset.json").read_text(encoding="utf-8"))
    items = dataset_manifest.get("items", [])
    filtered: list[dict[str, Any]] = []
    for item in items:
        subset = str(item.get("subset", ""))
        split = str(item.get("split", ""))
        if subsets and subset not in subsets:
            continue
        if splits and split not in splits:
            continue
        filtered.append(item)
    if not filtered:
        raise ValueError("筛选后没有可评估图像")
    return filtered


def load_raw_items(images_dir: str | Path, masks_dir: str | Path) -> list[dict[str, Any]]:
    records = collect_paired_records(images_dir, masks_dir)
    items: list[dict[str, Any]] = []
    for record in records:
        items.append(
            {
                "stem": record["stem"],
                "subset": "raw",
                "split": "all",
                "image_path": record["image_path"],
                "mask_path": record["mask_path"],
            }
        )
    return items


def summarize_groups(rows: list[dict[str, Any]], group_keys: list[str]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        group_name = "/".join(str(row.get(key, "")) for key in group_keys)
        grouped[group_name].append(
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
        )
    return {
        name: {
            "num_images": len(metrics),
            "metrics": average_metric_dicts(metrics),
        }
        for name, metrics in grouped.items()
    }


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    output_dir = ensure_dir(args.output_dir)
    predictions_dir = ensure_dir(output_dir / "predictions")
    overlays_dir = ensure_dir(output_dir / "overlays")

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
    auto_crop = bool(data_cfg.get("auto_crop_sem_region", True))
    crop_detection_ratio = float(data_cfg.get("crop_detection_ratio", 0.75))
    raw_mask_mode = str(args.mask_mode or "colored")
    raw_foreground_colors = [parse_rgb_spec(value) for value in args.foreground_colors] if args.foreground_colors else []

    if args.prepared_root:
        prepared_root = Path(args.prepared_root)
        subset_filter = set(args.subset or [])
        split_filter = set(args.split or [])
        items = load_items(prepared_root, subsets=subset_filter or None, splits=split_filter or None)
        data_source = "prepared"
    else:
        if not args.images_dir or not args.masks_dir:
            raise ValueError("请提供 --prepared-root 或同时提供 --images-dir 和 --masks-dir")
        prepared_root = None
        items = load_raw_items(args.images_dir, args.masks_dir)
        data_source = "raw"

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for item in items:
            if data_source == "prepared":
                assert prepared_root is not None
                image = read_gray(prepared_root / item["image_path"])
                mask = read_gray(prepared_root / item["mask_path"])
                crop_height = image.shape[0]
            else:
                image = read_gray(Path(item["image_path"]))
                raw_mask = load_mask_array(Path(item["mask_path"]))
                mask = convert_mask_to_binary(raw_mask, mode=raw_mask_mode, foreground_colors=raw_foreground_colors)
                if image.shape[:2] != mask.shape[:2]:
                    raise ValueError(f"图像与掩膜尺寸不一致: {item['image_path']} vs {item['mask_path']}")
                crop_height = image.shape[0]
                if auto_crop:
                    crop_height = detect_crop_height(image, start_ratio=crop_detection_ratio)
                    image = image[:crop_height, :]
                    mask = mask[:crop_height, :]
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

            file_stub = f"{item.get('subset', 'dataset')}__{item.get('split', 'all')}__{item['stem']}"
            write_image(predictions_dir / f"{file_stub}_pred.png", pred_mask)
            write_image(overlays_dir / f"{file_stub}_overlay.png", overlay)

            row = {
                "stem": item["stem"],
                "subset": item.get("subset", ""),
                "split": item.get("split", ""),
                "crop_height": int(crop_height),
                "dice": float(metrics["dice"]),
                "iou": float(metrics["iou"]),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "vf": float(metrics["vf"]),
                "vf_pred": float(metrics["vf_pred"]),
                "vf_gt": float(metrics["vf_gt"]),
                "boundary_f1": float(metrics["boundary_f1"]),
                "prediction_path": str(predictions_dir / f"{file_stub}_pred.png"),
                "overlay_path": str(overlays_dir / f"{file_stub}_overlay.png"),
            }
            rows.append(row)

    csv_path = output_dir / "per_image_metrics.csv"
    headers = [
        "stem",
        "subset",
        "split",
        "crop_height",
        "dice",
        "iou",
        "precision",
        "recall",
        "vf",
        "vf_pred",
        "vf_gt",
        "boundary_f1",
        "prediction_path",
        "overlay_path",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in headers})
    maybe_write_xlsx(rows, output_dir / "per_image_metrics.xlsx")

    summary = {
        "checkpoint": str(args.checkpoint),
        "prepared_root": str(prepared_root) if prepared_root is not None else "",
        "data_source": data_source,
        "mask_mode": raw_mask_mode if data_source == "raw" else "prepared_binary",
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
        "by_subset": summarize_groups(rows, ["subset"]),
        "by_split": summarize_groups(rows, ["split"]),
        "by_subset_split": summarize_groups(rows, ["subset", "split"]),
        "per_image_csv": str(csv_path),
    }
    save_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
