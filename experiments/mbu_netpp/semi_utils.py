from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch

from backend.app.utils.image_io import read_gray, write_image

from experiments.mbu_netpp.common import (
    build_overlay,
    ensure_dir,
    image_to_tensor,
    load_yaml,
    resolve_device,
    save_json,
)
from experiments.mbu_netpp.models import build_model_for_checkpoint, sliding_window_inference
from experiments.mbu_netpp.preparation import SUPPORTED_IMAGE_SUFFIXES, build_edge_mask, detect_crop_height
from experiments.mbu_netpp.preprocess import apply_preprocess
from experiments.mbu_netpp.semi_scoring import (
    compute_prediction_scores,
    select_active_learning_candidates,
    select_pseudo_candidates,
)


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    if not rows:
        target.write_text("", encoding="utf-8-sig")
        return
    fieldnames = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if not isinstance(value, (dict, list))
        }
    )
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def save_stem_list(path: str | Path, stems: list[str]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    target.write_text("\n".join(stems), encoding="utf-8")


def clone_config(config: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(config))


def load_dataset_manifest(prepared_root: str | Path) -> dict[str, Any]:
    return load_json(Path(prepared_root) / "manifests" / "dataset.json")


def load_fold_manifest(prepared_root: str | Path, fold_manifest_name: str) -> dict[str, Any]:
    return load_json(Path(prepared_root) / "manifests" / fold_manifest_name)


def list_image_paths(images_dir: str | Path) -> list[Path]:
    root = Path(images_dir)
    if not root.exists():
        raise FileNotFoundError(f"未标注图像目录不存在: {root}")
    if root.is_file():
        return [root.resolve()]
    return sorted(
        [
            path.resolve()
            for path in root.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        ]
    )


def build_unlabeled_items(images_dir: str | Path, excluded_stems: set[str]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for image_path in list_image_paths(images_dir):
        if image_path.stem in excluded_stems:
            continue
        items.append(
            {
                "stem": image_path.stem,
                "image_path": str(image_path),
            }
        )
    return items


def load_runtime_config(checkpoint_path: str | Path, config_path: str | None) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if config_path:
        config = load_yaml(config_path)
    else:
        config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("未能从 checkpoint 或配置文件读取有效 config")
    return checkpoint, config


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    config_path: str | None,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint, config = load_runtime_config(checkpoint_path, config_path)
    model = build_model_for_checkpoint(config["model"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, config


def _compute_tta_probability(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    patch_size: int,
    overlap: float,
) -> tuple[np.ndarray, float]:
    base_logits = sliding_window_inference(model, image_tensor, patch_size=patch_size, overlap=overlap)
    base_probs = torch.sigmoid(base_logits)

    flipped_image = torch.flip(image_tensor, dims=[3])
    flipped_logits = sliding_window_inference(model, flipped_image, patch_size=patch_size, overlap=overlap)
    flipped_probs = torch.flip(torch.sigmoid(flipped_logits), dims=[3])

    merged_probs = 0.5 * (base_probs + flipped_probs)
    base_mask = (base_probs >= 0.5).float()
    flipped_mask = (flipped_probs >= 0.5).float()
    consistency = 1.0 - float(torch.mean(torch.abs(base_mask - flipped_mask)).cpu().item())
    return merged_probs.cpu().numpy()[0, 0], consistency


def predict_probability_map(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    patch_size: int,
    overlap: float,
    use_tta: bool,
) -> tuple[np.ndarray, dict[str, float]]:
    with torch.no_grad():
        if use_tta:
            probs, tta_consistency = _compute_tta_probability(
                model=model,
                image_tensor=image_tensor,
                patch_size=patch_size,
                overlap=overlap,
            )
            return probs, {"tta_consistency_score": float(tta_consistency)}
        logits = sliding_window_inference(model, image_tensor, patch_size=patch_size, overlap=overlap)
        probs = torch.sigmoid(logits).cpu().numpy()[0, 0]
        return probs, {"tta_consistency_score": 1.0}


def infer_unlabeled_items(
    checkpoint_path: str | Path,
    image_items: list[dict[str, Any]],
    output_dir: str | Path,
    device_name: str,
    binary_threshold: float,
    use_tta: bool,
    config_path: str | None = None,
) -> list[dict[str, Any]]:
    output_root = ensure_dir(output_dir)
    cropped_images_dir = ensure_dir(output_root / "cropped_images")
    cropped_masks_dir = ensure_dir(output_root / "cropped_masks")
    full_masks_dir = ensure_dir(output_root / "full_masks")
    overlays_dir = ensure_dir(output_root / "overlays")
    probabilities_dir = ensure_dir(output_root / "probabilities")
    stats_dir = ensure_dir(output_root / "stats")

    device = resolve_device(device_name)
    model, config = load_model_from_checkpoint(checkpoint_path, config_path=config_path, device=device)

    data_cfg = config["data"]
    inference_cfg = config.get("inference", {})
    patch_size = int(data_cfg.get("patch_size", 256))
    overlap = float(inference_cfg.get("overlap", data_cfg.get("overlap", 0.25)))
    normalization = str(data_cfg.get("normalization", "minmax"))
    preprocess_cfg = data_cfg.get("preprocess")
    auto_crop = bool(data_cfg.get("auto_crop_sem_region", True))
    crop_detection_ratio = float(data_cfg.get("crop_detection_ratio", 0.75))

    prediction_rows: list[dict[str, Any]] = []
    for image_item in image_items:
        image_path = Path(str(image_item["image_path"]))
        original = read_gray(image_path)
        crop_height = int(original.shape[0])
        cropped_image = original
        if auto_crop:
            crop_height = detect_crop_height(original, start_ratio=crop_detection_ratio)
            cropped_image = original[:crop_height, :]

        processed = apply_preprocess(cropped_image, preprocess_cfg)
        image_tensor = image_to_tensor(processed, normalization=normalization).unsqueeze(0).to(device)
        prob_map, extra_scores = predict_probability_map(
            model=model,
            image_tensor=image_tensor,
            patch_size=patch_size,
            overlap=overlap,
            use_tta=use_tta,
        )
        cropped_mask = (prob_map >= float(binary_threshold)).astype(np.uint8) * 255
        full_mask = np.zeros_like(original, dtype=np.uint8)
        full_mask[:crop_height, :] = cropped_mask

        cropped_image_path = cropped_images_dir / f"{image_path.stem}.png"
        cropped_mask_path = cropped_masks_dir / f"{image_path.stem}.png"
        full_mask_path = full_masks_dir / f"{image_path.stem}_mask.png"
        overlay_path = overlays_dir / f"{image_path.stem}_overlay.png"
        probability_path = probabilities_dir / f"{image_path.stem}.npy"

        write_image(cropped_image_path, cropped_image)
        write_image(cropped_mask_path, cropped_mask)
        write_image(full_mask_path, full_mask)
        write_image(overlay_path, build_overlay(original, full_mask))
        np.save(probability_path, prob_map.astype(np.float32))

        scores = compute_prediction_scores(prob_map, binary_threshold=binary_threshold)
        row = {
            "stem": image_path.stem,
            "image_path": str(image_path.resolve()),
            "cropped_image_path": str(cropped_image_path.resolve()),
            "cropped_mask_path": str(cropped_mask_path.resolve()),
            "full_mask_path": str(full_mask_path.resolve()),
            "overlay_path": str(overlay_path.resolve()),
            "probability_path": str(probability_path.resolve()),
            "crop_height": crop_height,
            "height": int(cropped_image.shape[0]),
            "width": int(cropped_image.shape[1]),
            "binary_threshold": float(binary_threshold),
            **scores,
            **extra_scores,
        }
        save_json(stats_dir / f"{image_path.stem}.json", row)
        prediction_rows.append(row)

    save_json(
        output_root / "prediction_manifest.json",
        {
            "checkpoint_path": str(Path(checkpoint_path).resolve()),
            "num_images": len(prediction_rows),
            "items": prediction_rows,
        },
    )
    save_csv(output_root / "prediction_scores.csv", prediction_rows)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return prediction_rows


def _copy_file(source: str | Path, target: str | Path) -> Path:
    source_path = Path(source)
    target_path = Path(target)
    ensure_dir(target_path.parent)
    shutil.copy2(source_path, target_path)
    return target_path


def load_pseudo_pool(pseudo_pool_root: str | Path) -> list[dict[str, Any]]:
    manifest_path = Path(pseudo_pool_root) / "manifests" / "pseudo_pool.json"
    if not manifest_path.exists():
        return []
    manifest = load_json(manifest_path)
    return list(manifest.get("items", []))


def save_pseudo_pool(pseudo_pool_root: str | Path, items: list[dict[str, Any]]) -> None:
    manifest_root = ensure_dir(Path(pseudo_pool_root) / "manifests")
    save_json(
        manifest_root / "pseudo_pool.json",
        {
            "num_items": len(items),
            "items": items,
        },
    )


def append_pseudo_pool(
    pseudo_pool_root: str | Path,
    selected_items: list[dict[str, Any]],
    selected_iteration: int,
) -> list[dict[str, Any]]:
    pseudo_root = ensure_dir(pseudo_pool_root)
    images_dir = ensure_dir(pseudo_root / "images")
    masks_dir = ensure_dir(pseudo_root / "masks")
    previews_dir = ensure_dir(pseudo_root / "previews")
    edge_dirs = {
        "k3": ensure_dir(pseudo_root / "edges" / "k3"),
        "k5": ensure_dir(pseudo_root / "edges" / "k5"),
    }

    existing_items = load_pseudo_pool(pseudo_root)
    existing_stems = {str(item.get("source_stem", item.get("stem"))) for item in existing_items}
    merged_items = list(existing_items)

    for item in selected_items:
        source_stem = str(item["stem"])
        if source_stem in existing_stems:
            continue

        image_target = _copy_file(item["cropped_image_path"], images_dir / f"{source_stem}.png")
        mask_target = _copy_file(item["cropped_mask_path"], masks_dir / f"{source_stem}.png")
        overlay_target = _copy_file(item["overlay_path"], previews_dir / f"{source_stem}_overlay.png")

        mask_image = read_gray(mask_target)
        edge_paths: dict[str, str] = {}
        for edge_key, edge_dir in edge_dirs.items():
            kernel = int(edge_key[1:])
            edge_mask = build_edge_mask(mask_image, kernel_size=kernel)
            edge_target = edge_dir / f"{source_stem}.png"
            write_image(edge_target, edge_mask)
            edge_paths[edge_key] = str(edge_target.resolve())

        merged_items.append(
            {
                "stem": f"pseudo__{source_stem}",
                "source_stem": source_stem,
                "source_type": "pseudo",
                "selected_iteration": int(selected_iteration),
                "sample_weight": 1.0,
                "image_path": str(image_target.resolve()),
                "mask_path": str(mask_target.resolve()),
                "overlay_path": str(overlay_target.resolve()),
                "edge_paths": edge_paths,
                "confidence_score": float(item.get("confidence_score", 0.0)),
                "uncertainty_score": float(item.get("uncertainty_score", 0.0)),
                "tta_consistency_score": float(item.get("tta_consistency_score", 1.0)),
                "height": int(item.get("height", 0)),
                "width": int(item.get("width", 0)),
                "crop_height": int(item.get("crop_height", 0)),
                "source_image_path": str(item.get("image_path", "")),
                "probability_path": str(item.get("probability_path", "")),
            }
        )
        existing_stems.add(source_stem)

    save_pseudo_pool(pseudo_root, merged_items)
    return merged_items


def load_query_pending(query_root: str | Path) -> list[dict[str, Any]]:
    manifest_path = Path(query_root) / "pending_queries.json"
    if not manifest_path.exists():
        return []
    payload = load_json(manifest_path)
    return list(payload.get("items", []))


def save_query_pending(query_root: str | Path, items: list[dict[str, Any]]) -> None:
    root = ensure_dir(query_root)
    save_json(root / "pending_queries.json", {"num_items": len(items), "items": items})


def export_active_learning_selection(
    query_root: str | Path,
    selected_items: list[dict[str, Any]],
    iteration_index: int,
) -> None:
    root = ensure_dir(query_root)
    history_dir = ensure_dir(root / "history")
    save_csv(history_dir / f"iter_{iteration_index:02d}_active_learning.csv", selected_items)
    save_stem_list(
        history_dir / f"iter_{iteration_index:02d}_active_learning.txt",
        [str(item["stem"]) for item in selected_items],
    )


def to_absolute_supervised_record(
    prepared_root: str | Path,
    item: dict[str, Any],
    source_type: str,
    sample_weight: float,
    stem_override: str | None = None,
) -> dict[str, Any]:
    root = Path(prepared_root)
    record = dict(item)
    record["stem"] = str(stem_override or item["stem"])
    record["image_path"] = str((root / str(item["image_path"])).resolve())
    record["mask_path"] = str((root / str(item["mask_path"])).resolve())
    record["edge_paths"] = {
        key: str((root / str(value)).resolve())
        for key, value in dict(item.get("edge_paths") or {}).items()
    }
    record["source_type"] = source_type
    record["sample_weight"] = float(sample_weight)
    record["source_stem"] = str(item.get("stem", record["stem"]))
    return record


def load_reviewed_supervised_items(
    reviewed_roots: list[str],
    sample_weight: float,
) -> tuple[list[dict[str, Any]], list[str], set[str]]:
    items: list[dict[str, Any]] = []
    train_stems: list[str] = []
    source_stems: set[str] = set()
    for root_index, root in enumerate(reviewed_roots):
        dataset_manifest = load_dataset_manifest(root)
        for item in dataset_manifest.get("items", []):
            stem = str(item["stem"])
            merged_stem = f"reviewed__{root_index}__{stem}"
            items.append(
                to_absolute_supervised_record(
                    prepared_root=root,
                    item=item,
                    source_type="reviewed_supervised",
                    sample_weight=sample_weight,
                    stem_override=merged_stem,
                )
            )
            train_stems.append(merged_stem)
            source_stems.add(stem)
    return items, train_stems, source_stems


def build_iteration_training_root(
    base_supervised_root: str | Path,
    fold_manifest_name: str,
    fold_index: int,
    reviewed_labeled_roots: list[str],
    pseudo_pool_items: list[dict[str, Any]],
    output_root: str | Path,
    real_loss_weight: float,
    pseudo_loss_weight: float,
    supervised_repeat: int,
) -> dict[str, Any]:
    base_root = Path(base_supervised_root)
    output_root = ensure_dir(output_root)
    manifests_dir = ensure_dir(output_root / "manifests")

    base_dataset = load_dataset_manifest(base_root)
    base_folds = load_fold_manifest(base_root, fold_manifest_name)
    base_items_by_stem = {str(item["stem"]): item for item in base_dataset.get("items", [])}
    target_fold = next(
        fold for fold in base_folds.get("folds", []) if int(fold["fold_index"]) == int(fold_index)
    )

    dataset_items: list[dict[str, Any]] = []
    for stem, item in base_items_by_stem.items():
        dataset_items.append(
            to_absolute_supervised_record(
                prepared_root=base_root,
                item=item,
                source_type="supervised",
                sample_weight=real_loss_weight,
            )
        )

    reviewed_items, reviewed_train_stems, reviewed_source_stems = load_reviewed_supervised_items(
        reviewed_roots=reviewed_labeled_roots,
        sample_weight=real_loss_weight,
    )
    dataset_items.extend(reviewed_items)

    filtered_pseudo_items = [
        {
            **item,
            "sample_weight": float(pseudo_loss_weight),
        }
        for item in pseudo_pool_items
        if str(item.get("source_stem", item.get("stem"))) not in reviewed_source_stems
    ]
    dataset_items.extend(filtered_pseudo_items)

    base_train_stems = [str(stem) for stem in target_fold.get("train_stems", [])]
    repeated_train_stems = base_train_stems * max(1, int(supervised_repeat))
    repeated_train_stems.extend(reviewed_train_stems * max(1, int(supervised_repeat)))
    repeated_train_stems.extend([str(item["stem"]) for item in filtered_pseudo_items])
    val_stems = [str(stem) for stem in target_fold.get("val_stems", [])]

    save_json(
        manifests_dir / "dataset.json",
        {
            "num_samples": len(dataset_items),
            "fold_index": int(fold_index),
            "num_supervised": len(base_items_by_stem),
            "num_reviewed": len(reviewed_items),
            "num_pseudo": len(filtered_pseudo_items),
            "items": dataset_items,
        },
    )
    save_json(
        manifests_dir / "folds_1_seed42.json",
        {
            "num_folds": 1,
            "seed": 42,
            "folds": [
                {
                    "fold_index": 0,
                    "train_stems": repeated_train_stems,
                    "val_stems": val_stems,
                }
            ],
        },
    )
    return {
        "prepared_root": str(output_root),
        "num_supervised": len(base_items_by_stem),
        "num_reviewed": len(reviewed_items),
        "num_pseudo": len(filtered_pseudo_items),
        "num_train_records": len(repeated_train_stems),
        "num_val_records": len(val_stems),
    }


def filter_items_by_stems(items: list[dict[str, Any]], excluded_source_stems: set[str]) -> list[dict[str, Any]]:
    return [
        item
        for item in items
        if str(item.get("source_stem", item.get("stem"))) not in excluded_source_stems
    ]


def summarize_iteration(
    iteration_index: int,
    checkpoint_path: str | Path,
    pseudo_selected: list[dict[str, Any]],
    active_selected: list[dict[str, Any]],
    training_result: dict[str, Any] | None,
    pool_info: dict[str, Any],
) -> dict[str, Any]:
    return {
        "iteration_index": int(iteration_index),
        "teacher_checkpoint": str(Path(checkpoint_path).resolve()),
        "num_pseudo_selected": len(pseudo_selected),
        "num_active_selected": len(active_selected),
        "pseudo_selected_stems": [str(item["stem"]) for item in pseudo_selected],
        "active_selected_stems": [str(item["stem"]) for item in active_selected],
        "training_result": training_result or {},
        "pool_info": pool_info,
    }
