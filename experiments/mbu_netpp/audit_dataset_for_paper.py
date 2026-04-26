from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from statistics import mean
from typing import Any

import cv2
import numpy as np

from backend.app.utils.image_io import read_gray

from experiments.mbu_netpp.common import ensure_dir, load_yaml, save_json
from experiments.mbu_netpp.preparation import SUPPORTED_IMAGE_SUFFIXES, build_binary_mask, find_image_path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_images_dir() -> Path:
    return Path(r"C:\Users\pyd111\Desktop\analysis_same_teacher_nocap_095\1")


def default_prepared_root() -> Path:
    return repo_root() / "experiments" / "mbu_netpp" / "workdir" / "prepared_real53_cv5"


def default_config() -> Path:
    return repo_root() / "experiments" / "mbu_netpp" / "configs" / "opt_real53_boundary_sampling.yaml"


def default_output_dir() -> Path:
    return repo_root() / "results" / "paper_supplement" / "dataset_audit"


def list_images(root: Path) -> list[Path]:
    return sorted([item for item in root.iterdir() if item.is_file() and item.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES])


def numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "mean": None, "max": None}
    return {
        "count": len(values),
        "min": float(min(values)),
        "mean": float(mean(values)),
        "max": float(max(values)),
    }


def audit_labelme(images_dir: Path, annotations_dir: Path, positive_labels: list[str]) -> dict[str, Any]:
    annotation_files = sorted(annotations_dir.glob("*.json"))
    image_files = list_images(images_dir)
    positive = {label.strip().lower() for label in positive_labels}

    label_counts: collections.Counter[str] = collections.Counter()
    shape_type_counts: collections.Counter[str] = collections.Counter()
    point_counts: list[float] = []
    polygon_areas: list[float] = []
    foreground_ratios: list[float] = []
    missing_images: list[str] = []
    invalid_polygons: list[dict[str, Any]] = []
    out_of_bounds_points = 0
    empty_masks: list[str] = []

    for json_path in annotation_files:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        stem = json_path.stem
        try:
            image_path = find_image_path(images_dir, stem)
        except FileNotFoundError:
            missing_images.append(stem)
            continue

        image = read_gray(image_path)
        height, width = image.shape[:2]
        shapes = list(payload.get("shapes") or [])
        mask = build_binary_mask((height, width), shapes, positive)
        foreground_ratio = float(np.count_nonzero(mask)) / float(mask.size)
        foreground_ratios.append(foreground_ratio)
        if np.count_nonzero(mask) == 0:
            empty_masks.append(stem)

        for shape_index, shape in enumerate(shapes):
            label = str(shape.get("label", "")).strip().lower()
            shape_type = str(shape.get("shape_type") or "polygon")
            points = np.asarray(shape.get("points") or [], dtype=np.float32)
            label_counts[label] += 1
            shape_type_counts[shape_type] += 1
            point_counts.append(float(len(points)))

            if label not in positive:
                continue
            if shape_type not in {"polygon", "None", "none"} and shape.get("shape_type") is not None:
                continue
            if len(points) < 3:
                invalid_polygons.append({"stem": stem, "shape_index": shape_index, "reason": "less_than_3_points"})
                continue

            outside = (
                (points[:, 0] < 0)
                | (points[:, 0] >= width)
                | (points[:, 1] < 0)
                | (points[:, 1] >= height)
            )
            out_of_bounds_points += int(np.count_nonzero(outside))
            clipped = points.copy()
            clipped[:, 0] = np.clip(clipped[:, 0], 0, width - 1)
            clipped[:, 1] = np.clip(clipped[:, 1], 0, height - 1)
            area = float(abs(cv2.contourArea(np.round(clipped).astype(np.int32))))
            polygon_areas.append(area)
            if area <= 0:
                invalid_polygons.append({"stem": stem, "shape_index": shape_index, "reason": "zero_area"})

    paired_stems = {path.stem for path in image_files} & {path.stem for path in annotation_files}
    image_without_json = sorted({path.stem for path in image_files} - {path.stem for path in annotation_files})
    critical_pass = len(missing_images) == 0 and len(image_without_json) == 0 and len(invalid_polygons) == 0 and len(empty_masks) == 0
    warnings: list[str] = []
    if out_of_bounds_points:
        warnings.append(
            f"{out_of_bounds_points} polygon points are outside image bounds and are clipped during mask generation."
        )

    return {
        "images_dir": str(images_dir),
        "annotations_dir": str(annotations_dir),
        "image_file_count": len(image_files),
        "json_file_count": len(annotation_files),
        "paired_image_json_count": len(paired_stems),
        "missing_images_for_json": missing_images,
        "images_without_json": image_without_json,
        "label_counts": dict(label_counts),
        "shape_type_counts": dict(shape_type_counts),
        "total_shapes": int(sum(label_counts.values())),
        "positive_labels": sorted(positive),
        "point_count_summary": numeric_summary(point_counts),
        "polygon_area_px_summary": numeric_summary(polygon_areas),
        "source_foreground_ratio_summary": numeric_summary(foreground_ratios),
        "invalid_polygon_count": len(invalid_polygons),
        "invalid_polygon_examples": invalid_polygons[:20],
        "out_of_bounds_point_count": out_of_bounds_points,
        "empty_mask_count": len(empty_masks),
        "empty_mask_stems": empty_masks,
        "programmatic_quality_critical_pass": critical_pass,
        "programmatic_quality_warning_count": len(warnings),
        "programmatic_quality_warnings": warnings,
        "quality_note": "This is a programmatic audit of file pairing, label consistency and polygon geometry. It is not an inter-annotator agreement test.",
    }


def summarize_prepared_root(prepared_root: Path) -> dict[str, Any]:
    dataset_path = prepared_root / "manifests" / "dataset.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    items = list(dataset.get("items") or [])
    heights = [float(item.get("height", 0)) for item in items]
    widths = [float(item.get("width", 0)) for item in items]
    crop_heights = [float(item.get("crop_height", 0)) for item in items if item.get("crop_height") is not None]
    foreground_ratios = [
        float(item.get("foreground_pixels", 0)) / max(1.0, float(item.get("height", 0)) * float(item.get("width", 0)))
        for item in items
    ]

    fold_summaries: list[dict[str, Any]] = []
    for fold_path in sorted((prepared_root / "manifests").glob("folds*.json")):
        payload = json.loads(fold_path.read_text(encoding="utf-8"))
        fold_summaries.append(
            {
                "manifest": fold_path.name,
                "num_folds": payload.get("num_folds"),
                "strategy": payload.get("strategy"),
                "folds": [
                    {
                        "fold_index": fold.get("fold_index"),
                        "train_count": len(fold.get("train_stems", [])),
                        "val_count": len(fold.get("val_stems", [])),
                        "test_count": len(fold.get("test_stems", [])),
                    }
                    for fold in payload.get("folds", [])
                ],
            }
        )

    holdout_summaries: list[dict[str, Any]] = []
    for holdout_path in sorted((prepared_root / "manifests").glob("holdout*.json")):
        payload = json.loads(holdout_path.read_text(encoding="utf-8"))
        holdout_summaries.append(
            {
                "manifest": holdout_path.name,
                "num_test": len(payload.get("test_stems", [])),
                "num_trainval": len(payload.get("trainval_stems", [])),
                "test_stems": payload.get("test_stems", []),
            }
        )

    return {
        "prepared_root": str(prepared_root),
        "num_samples_after_crop": len(items),
        "height_summary": numeric_summary(heights),
        "width_summary": numeric_summary(widths),
        "crop_height_summary": numeric_summary(crop_heights),
        "foreground_ratio_summary": numeric_summary(foreground_ratios),
        "fold_summaries": fold_summaries,
        "holdout_summaries": holdout_summaries,
    }


def summarize_training_sample_counts(config_path: Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    data_cfg = config["data"]
    training_cfg = config["training"]
    num_folds = int(data_cfg.get("num_folds", 1))
    samples_per_epoch = int(data_cfg.get("samples_per_epoch", 0))
    epochs = int(training_cfg.get("epochs", 0))
    batch_size = int(training_cfg.get("batch_size", 0))
    return {
        "config": str(config_path),
        "augmentation_is_online": True,
        "patch_size": int(data_cfg.get("patch_size", 256)),
        "samples_per_epoch_per_fold": samples_per_epoch,
        "epochs_per_fold": epochs,
        "num_folds": num_folds,
        "batch_size": batch_size,
        "dynamic_training_patch_samples_per_fold": samples_per_epoch * epochs,
        "dynamic_training_patch_samples_all_folds": samples_per_epoch * epochs * num_folds,
        "augmentation_config": data_cfg.get("augmentation", {}),
        "explanation": "Augmentation is applied online in __getitem__; it does not create a fixed expanded image folder. The count is therefore the number of dynamically sampled training patches seen by the optimizer.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit dataset and annotation facts for the paper")
    parser.add_argument("--images-dir", default=str(default_images_dir()))
    parser.add_argument("--annotations-dir", default=str(default_images_dir()))
    parser.add_argument("--prepared-root", default=str(default_prepared_root()))
    parser.add_argument("--config", default=str(default_config()))
    parser.add_argument("--output-dir", default=str(default_output_dir()))
    parser.add_argument("--positive-label", action="append", default=["gamma_prime"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    images_dir = Path(args.images_dir)
    annotations_dir = Path(args.annotations_dir)
    prepared_root = Path(args.prepared_root)
    config_path = Path(args.config)

    report = {
        "labelme_audit": audit_labelme(images_dir, annotations_dir, list(args.positive_label)),
        "prepared_dataset_summary": summarize_prepared_root(prepared_root),
        "training_sample_count_summary": summarize_training_sample_counts(config_path),
    }
    save_json(output_dir / "dataset_annotation_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
