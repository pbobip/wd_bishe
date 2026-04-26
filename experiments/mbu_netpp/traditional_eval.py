from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from backend.app.schemas.run import PreprocessConfig, TraditionalSegConfig
from backend.app.services.algorithms.traditional import TraditionalService
from backend.app.utils.image_io import read_gray, write_image

from experiments.mbu_netpp.common import ensure_dir, save_json
from experiments.mbu_netpp.dataset import load_prepared_records
from experiments.mbu_netpp.infer import maybe_write_xlsx
from experiments.mbu_netpp.metrics import average_metric_dicts, compute_binary_segmentation_metrics
from experiments.mbu_netpp.preprocess import apply_preprocess


def build_presets() -> dict[str, dict[str, Any]]:
    return {
        "otsu_raw": {
            "preprocess": PreprocessConfig(enabled=False),
            "traditional": TraditionalSegConfig(
                method="threshold",
                threshold_mode="otsu",
                foreground_target="dark",
                fill_holes=False,
                open_kernel=3,
                close_kernel=3,
                min_area=30,
            ),
        },
        "otsu_clahe": {
            "preprocess": PreprocessConfig.model_validate(
                {
                    "enabled": True,
                    "background": {"method": "none", "radius": 25},
                    "denoise": {"method": "none", "gaussian_kernel": 3, "median_kernel": 3},
                    "enhance": {"method": "clahe", "clahe_clip_limit": 2.0, "clahe_tile_size": 8},
                    "extras": {"unsharp": False, "unsharp_radius": 3, "unsharp_amount": 1.0},
                }
            ),
            "traditional": TraditionalSegConfig(
                method="threshold",
                threshold_mode="otsu",
                foreground_target="dark",
                fill_holes=False,
                open_kernel=3,
                close_kernel=3,
                min_area=30,
            ),
        },
        "adaptive_clahe": {
            "preprocess": PreprocessConfig.model_validate(
                {
                    "enabled": True,
                    "background": {"method": "none", "radius": 25},
                    "denoise": {"method": "none", "gaussian_kernel": 3, "median_kernel": 3},
                    "enhance": {"method": "clahe", "clahe_clip_limit": 2.0, "clahe_tile_size": 8},
                    "extras": {"unsharp": False, "unsharp_radius": 3, "unsharp_amount": 1.0},
                }
            ),
            "traditional": TraditionalSegConfig(
                method="adaptive",
                adaptive_method="gaussian",
                adaptive_block_size=35,
                adaptive_c=5,
                foreground_target="dark",
                fill_holes=False,
                open_kernel=3,
                close_kernel=3,
                min_area=30,
            ),
        },
        "edge_canny_clahe": {
            "preprocess": PreprocessConfig.model_validate(
                {
                    "enabled": True,
                    "background": {"method": "none", "radius": 25},
                    "denoise": {"method": "gaussian", "gaussian_kernel": 3, "median_kernel": 3},
                    "enhance": {"method": "clahe", "clahe_clip_limit": 2.0, "clahe_tile_size": 8},
                    "extras": {"unsharp": False, "unsharp_radius": 3, "unsharp_amount": 1.0},
                }
            ),
            "traditional": TraditionalSegConfig(
                method="edge",
                edge_operator="canny",
                edge_blur_kernel=3,
                edge_threshold1=60,
                edge_threshold2=180,
                edge_dilate_iterations=1,
                foreground_target="dark",
                fill_holes=False,
                open_kernel=3,
                close_kernel=3,
                min_area=30,
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对固定 holdout 测试集运行传统基线评估")
    parser.add_argument("--prepared-root", required=True, help="prepared_root 路径")
    parser.add_argument("--fold-manifest-name", required=True, help="包含 test_stems 的 fold manifest 文件名")
    parser.add_argument("--fold-index", type=int, default=0, help="读取 test_stems 时使用的 fold index")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument(
        "--preset",
        action="append",
        default=[],
        help="只运行指定 preset，可重复传入；默认运行全部",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepared_root = Path(args.prepared_root)
    output_dir = ensure_dir(args.output_dir)

    presets = build_presets()
    selected_names = list(args.preset) or list(presets.keys())
    selected = {name: presets[name] for name in selected_names}

    records = load_prepared_records(
        prepared_root=prepared_root,
        fold_manifest_name=args.fold_manifest_name,
        fold_index=int(args.fold_index),
        stage="test",
    )
    if not records:
        raise ValueError("当前 fold manifest 没有 test_stems")

    service = TraditionalService()
    all_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for preset_name, preset in selected.items():
        preset_dir = ensure_dir(output_dir / preset_name)
        masks_dir = ensure_dir(preset_dir / "masks")
        overlays_dir = ensure_dir(preset_dir / "overlays")

        preprocess_cfg = preset["preprocess"]
        traditional_cfg = preset["traditional"]
        rows: list[dict[str, Any]] = []

        for record in records:
            image = read_gray(prepared_root / record["image_path"])
            gt_mask = read_gray(prepared_root / record["mask_path"])
            processed = apply_preprocess(image, preprocess_cfg.model_dump())
            result = service.segment(processed, traditional_cfg)
            pred_mask = result["mask"]
            overlay = result["overlay"]

            stem = str(record["stem"])
            write_image(masks_dir / f"{stem}_mask.png", pred_mask)
            write_image(overlays_dir / f"{stem}_overlay.png", overlay)

            metrics = compute_binary_segmentation_metrics(pred_mask, gt_mask, boundary_tolerance=2)
            row = {
                "preset": preset_name,
                "stem": stem,
                "dice": float(metrics["dice"]),
                "iou": float(metrics["iou"]),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "vf": float(metrics["vf"]),
                "vf_pred": float(metrics["vf_pred"]),
                "vf_gt": float(metrics["vf_gt"]),
                "boundary_f1": float(metrics["boundary_f1"]),
                "mask_path": str(masks_dir / f"{stem}_mask.png"),
                "overlay_path": str(overlays_dir / f"{stem}_overlay.png"),
            }
            rows.append(row)
            all_rows.append(row)

        csv_path = preset_dir / "per_image_metrics.csv"
        headers = [
            "preset",
            "stem",
            "dice",
            "iou",
            "precision",
            "recall",
            "vf",
            "vf_pred",
            "vf_gt",
            "boundary_f1",
            "mask_path",
            "overlay_path",
        ]
        with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in headers})
        maybe_write_xlsx(rows, preset_dir / "per_image_metrics.xlsx")

        summary = {
            "preset": preset_name,
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
        save_json(preset_dir / "summary.json", summary)
        summary_rows.append(
            {
                "preset": preset_name,
                "num_images": len(rows),
                **summary["metrics"],
            }
        )

    if summary_rows:
        summary_csv = output_dir / "traditional_summary.csv"
        headers = list(summary_rows[0].keys())
        with summary_csv.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers)
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)
        maybe_write_xlsx(summary_rows, output_dir / "traditional_summary.xlsx")

    save_json(
        output_dir / "traditional_summary.json",
        {
            "prepared_root": str(prepared_root),
            "fold_manifest_name": args.fold_manifest_name,
            "fold_index": int(args.fold_index),
            "presets": summary_rows,
            "num_rows": len(all_rows),
        },
    )
    print(json.dumps({"num_presets": len(summary_rows), "output_dir": str(output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
