from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from experiments.mbu_netpp.common import ensure_dir, load_yaml, save_json, seed_everything
from experiments.mbu_netpp.dataset import SEMSegmentationDataset


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_config() -> Path:
    return repo_root() / "experiments" / "mbu_netpp" / "configs" / "opt_real53_boundary_sampling.yaml"


def default_output_dir() -> Path:
    return repo_root() / "results" / "paper_supplement" / "augmentation_volume"


def resolve_path(value: str | Path, root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_dataset_items(prepared_root: Path) -> list[dict[str, Any]]:
    payload = load_json(prepared_root / "manifests" / "dataset.json")
    return list(payload.get("items", []))


def read_fold_manifest(prepared_root: Path, fold_manifest_name: str) -> dict[str, Any]:
    return load_json(prepared_root / "manifests" / fold_manifest_name)


def read_holdout_manifest(prepared_root: Path, holdout_manifest_name: str) -> dict[str, Any] | None:
    path = prepared_root / "manifests" / holdout_manifest_name
    if not path.exists():
        return None
    return load_json(path)


def summarize_fold_counts(fold_manifest: dict[str, Any]) -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    for fold in fold_manifest.get("folds", []):
        rows.append(
            {
                "fold_index": int(fold["fold_index"]),
                "train_images": len(fold.get("train_stems", [])),
                "val_images": len(fold.get("val_stems", [])),
                "test_images": len(fold.get("test_stems", [])),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def tensor_to_uint8(array: Any) -> np.ndarray:
    data = array.detach().cpu().numpy()
    if data.ndim == 3:
        data = data[0]
    data = np.clip(data, 0.0, 1.0)
    return (data * 255.0).astype(np.uint8)


def build_train_dataset(config: dict[str, Any], fold_index: int, samples_per_epoch: int) -> SEMSegmentationDataset:
    data_cfg = config["data"]
    training_cfg = config["training"]
    return SEMSegmentationDataset(
        prepared_root=data_cfg["prepared_root"],
        fold_manifest_name=data_cfg["fold_manifest_name"],
        fold_index=fold_index,
        stage="train",
        patch_size=int(data_cfg["patch_size"]),
        edge_kernel=int(data_cfg["edge_kernel"]),
        normalization=str(data_cfg.get("normalization", "minmax")),
        preprocess_config=data_cfg.get("preprocess"),
        augmentation_config=data_cfg.get("augmentation"),
        sampling_config=data_cfg.get("sampling"),
        samples_per_epoch=samples_per_epoch,
    )


def export_offline_samples(
    config: dict[str, Any],
    seeds: list[int],
    fold_indices: list[int],
    patches_per_fold: int,
    output_dir: Path,
) -> dict[str, Any]:
    export_root = ensure_dir(output_dir / "offline_patch_preview")
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        seed_everything(seed)
        for fold_index in fold_indices:
            dataset = build_train_dataset(config, fold_index=fold_index, samples_per_epoch=patches_per_fold)
            fold_dir = ensure_dir(export_root / f"seed_{seed}" / f"fold_{fold_index}")
            image_dir = ensure_dir(fold_dir / "images")
            mask_dir = ensure_dir(fold_dir / "masks")
            edge_dir = ensure_dir(fold_dir / "edges")
            for index in range(patches_per_fold):
                item = dataset[index]
                name = f"{index:06d}_{item['source_stem']}.png"
                image_path = image_dir / name
                mask_path = mask_dir / name
                edge_path = edge_dir / name
                cv2.imwrite(str(image_path), tensor_to_uint8(item["image"]))
                cv2.imwrite(str(mask_path), tensor_to_uint8(item["mask"]))
                cv2.imwrite(str(edge_path), tensor_to_uint8(item["edge"]))
                rows.append(
                    {
                        "seed": seed,
                        "fold_index": fold_index,
                        "index": index,
                        "source_stem": str(item["source_stem"]),
                        "image_path": str(image_path),
                        "mask_path": str(mask_path),
                        "edge_path": str(edge_path),
                    }
                )
    write_csv(export_root / "offline_patch_preview_manifest.csv", rows)
    return {
        "export_root": str(export_root),
        "num_patch_triplets": len(rows),
        "num_files_written": len(rows) * 3,
        "manifest_csv": str(export_root / "offline_patch_preview_manifest.csv"),
        "note": "Each patch triplet contains image, mask and edge files. This is a fixed preview/export of the online augmentation process, not an extra training split.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report online augmentation volume and optionally export fixed augmented patches")
    parser.add_argument("--config", default=str(default_config()))
    parser.add_argument("--prepared-root", default="")
    parser.add_argument("--fold-manifest-name", default="")
    parser.add_argument("--holdout-manifest-name", default="")
    parser.add_argument("--output-dir", default=str(default_output_dir()))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--write-offline-samples", action="store_true")
    parser.add_argument("--offline-seeds", type=int, nargs="+", default=[])
    parser.add_argument("--offline-folds", type=int, nargs="+", default=[])
    parser.add_argument("--max-patches-per-fold", type=int, default=64)
    return parser.parse_args()


def expected_event_count(total_samples: int, probability: float) -> float:
    return float(total_samples) * float(probability)


def main() -> None:
    args = parse_args()
    root = repo_root()
    config_path = resolve_path(args.config, root)
    config = load_yaml(config_path)
    data_cfg = config["data"]
    training_cfg = config["training"]

    prepared_root = resolve_path(args.prepared_root or data_cfg["prepared_root"], root)
    fold_manifest_name = str(args.fold_manifest_name or data_cfg["fold_manifest_name"])
    holdout_manifest_name = str(args.holdout_manifest_name or data_cfg.get("holdout_manifest_name", ""))
    data_cfg["prepared_root"] = str(prepared_root)
    data_cfg["fold_manifest_name"] = fold_manifest_name
    output_dir = ensure_dir(args.output_dir)

    dataset_items = read_dataset_items(prepared_root)
    fold_manifest = read_fold_manifest(prepared_root, fold_manifest_name)
    holdout_manifest = read_holdout_manifest(prepared_root, holdout_manifest_name) if holdout_manifest_name else None
    fold_rows = summarize_fold_counts(fold_manifest)

    num_folds = len(fold_rows)
    samples_per_epoch = int(data_cfg.get("samples_per_epoch", 0))
    epochs = int(training_cfg.get("epochs", 0))
    batch_size = int(training_cfg.get("batch_size", 1))
    patch_size = int(data_cfg.get("patch_size", 256))
    seeds = list(args.seeds)
    per_fold_patches = samples_per_epoch * epochs
    all_fold_patches = per_fold_patches * num_folds
    all_seed_patches = all_fold_patches * len(seeds)
    optimizer_steps_per_epoch = math.ceil(samples_per_epoch / max(1, batch_size))
    optimizer_steps_per_fold = optimizer_steps_per_epoch * epochs
    optimizer_steps_all_folds = optimizer_steps_per_fold * num_folds
    optimizer_steps_all_seeds = optimizer_steps_all_folds * len(seeds)

    trainval_unique = sorted({stem for fold in fold_manifest.get("folds", []) for stem in fold.get("train_stems", []) + fold.get("val_stems", [])})
    holdout_stems = list((holdout_manifest or {}).get("test_stems", []))
    uncompressed_patch_bytes = patch_size * patch_size * 3
    aug_cfg = data_cfg.get("augmentation", {})
    hflip_p = float(aug_cfg.get("horizontal_flip_p", 0.0))
    vflip_p = float(aug_cfg.get("vertical_flip_p", 0.0))
    rotate90_p = float(aug_cfg.get("rotate90_p", 0.0))
    scale_p = float(aug_cfg.get("scale_p", 0.0))
    brightness_contrast_p = float(aug_cfg.get("brightness_contrast_p", 0.0))
    gaussian_noise_p = float(aug_cfg.get("gaussian_noise_p", 0.0))

    report: dict[str, Any] = {
        "config": str(config_path),
        "prepared_root": str(prepared_root),
        "fold_manifest_name": fold_manifest_name,
        "holdout_manifest_name": holdout_manifest_name,
        "independent_data_counts": {
            "prepared_full_images": len(dataset_items),
            "trainval_unique_images": len(trainval_unique),
            "holdout_test_images": len(holdout_stems),
            "fold_count": num_folds,
            "patch_size": patch_size,
        },
        "fold_counts": fold_rows,
        "online_training_volume": {
            "augmentation_is_online": True,
            "physical_augmented_image_files_created_by_training": 0,
            "samples_per_epoch_per_fold": samples_per_epoch,
            "epochs_per_fold": epochs,
            "batch_size": batch_size,
            "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
            "dynamic_patch_samples_per_fold": per_fold_patches,
            "dynamic_patch_samples_per_5fold_run": all_fold_patches,
            "seeds": seeds,
            "num_seed_runs": len(seeds),
            "dynamic_patch_samples_all_seed_runs": all_seed_patches,
            "optimizer_steps_per_fold": optimizer_steps_per_fold,
            "optimizer_steps_per_5fold_run": optimizer_steps_all_folds,
            "optimizer_steps_all_seed_runs": optimizer_steps_all_seeds,
            "augmentation_event_expected_counts_per_5fold_run": {
                "pipeline_invocations": all_fold_patches,
                "horizontal_flip": expected_event_count(all_fold_patches, hflip_p),
                "vertical_flip": expected_event_count(all_fold_patches, vflip_p),
                "random_rotate90_called": expected_event_count(all_fold_patches, rotate90_p),
                "random_rotate90_nonzero_angle": expected_event_count(all_fold_patches, rotate90_p * 0.75),
                "scale_affine": expected_event_count(all_fold_patches, scale_p),
                "brightness_contrast": expected_event_count(all_fold_patches, brightness_contrast_p),
                "gaussian_noise": expected_event_count(all_fold_patches, gaussian_noise_p),
            },
            "augmentation_event_expected_counts_all_seed_runs": {
                "pipeline_invocations": all_seed_patches,
                "horizontal_flip": expected_event_count(all_seed_patches, hflip_p),
                "vertical_flip": expected_event_count(all_seed_patches, vflip_p),
                "random_rotate90_called": expected_event_count(all_seed_patches, rotate90_p),
                "random_rotate90_nonzero_angle": expected_event_count(all_seed_patches, rotate90_p * 0.75),
                "scale_affine": expected_event_count(all_seed_patches, scale_p),
                "brightness_contrast": expected_event_count(all_seed_patches, brightness_contrast_p),
                "gaussian_noise": expected_event_count(all_seed_patches, gaussian_noise_p),
            },
        },
        "offline_equivalent_volume": {
            "patch_triplets_per_fold": per_fold_patches,
            "patch_triplets_per_5fold_run": all_fold_patches,
            "patch_triplets_all_seed_runs": all_seed_patches,
            "files_per_patch_triplet": 3,
            "image_mask_edge_files_per_5fold_run": all_fold_patches * 3,
            "image_mask_edge_files_all_seed_runs": all_seed_patches * 3,
            "uncompressed_bytes_per_patch_triplet": uncompressed_patch_bytes,
            "estimated_uncompressed_gib_per_5fold_run": (all_fold_patches * uncompressed_patch_bytes) / (1024**3),
            "estimated_uncompressed_gib_all_seed_runs": (all_seed_patches * uncompressed_patch_bytes) / (1024**3),
            "note": "Offline equivalent means saving the online sampled image/mask/edge patch triplets as files. PNG compression may reduce actual disk usage, but the independent source images remain 53.",
        },
        "augmentation_config": aug_cfg,
    }

    offline_export = None
    if args.write_offline_samples:
        offline_seeds = args.offline_seeds or seeds[:1]
        offline_folds = args.offline_folds or [0]
        patches_per_fold = int(args.max_patches_per_fold)
        if patches_per_fold <= 0:
            patches_per_fold = per_fold_patches
        offline_export = export_offline_samples(
            config=config,
            seeds=offline_seeds,
            fold_indices=offline_folds,
            patches_per_fold=patches_per_fold,
            output_dir=output_dir,
        )
        report["offline_export"] = offline_export

    write_csv(output_dir / "fold_counts.csv", fold_rows)
    save_json(output_dir / "augmentation_volume_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
