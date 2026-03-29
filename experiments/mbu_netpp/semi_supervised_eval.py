from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from backend.app.utils.image_io import read_gray, write_image

from experiments.mbu_netpp.common import build_overlay, ensure_dir, image_to_tensor, load_yaml, save_json, seed_everything
from experiments.mbu_netpp.dataset import load_prepared_records
from experiments.mbu_netpp.infer import load_model_from_checkpoint
from experiments.mbu_netpp.metrics import average_metric_dicts
from experiments.mbu_netpp.models import sliding_window_inference
from experiments.mbu_netpp.preparation import build_edge_mask
from experiments.mbu_netpp.preprocess import apply_preprocess
from experiments.mbu_netpp.train import train_one_fold


def load_dataset_manifest(prepared_root: str | Path) -> dict[str, Any]:
    return json.loads((Path(prepared_root) / "manifests" / "dataset.json").read_text(encoding="utf-8"))


def load_fold_manifest(prepared_root: str | Path, fold_manifest_name: str) -> dict[str, Any]:
    return json.loads((Path(prepared_root) / "manifests" / fold_manifest_name).read_text(encoding="utf-8"))


def list_unlabeled_images(images_dir: str | Path, exclude_stems: set[str]) -> list[Path]:
    images_dir = Path(images_dir)
    candidates = sorted(images_dir.glob("*.png"))
    return [path for path in candidates if path.stem not in exclude_stems]


def generate_pseudo_masks(
    checkpoint_path: str | Path,
    image_paths: list[Path],
    output_dir: str | Path,
    device_name: str,
    threshold: float = 0.5,
) -> list[dict[str, Any]]:
    output_dir = ensure_dir(output_dir)
    masks_dir = ensure_dir(output_dir / "masks")
    previews_dir = ensure_dir(output_dir / "previews")
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else device_name)
    model, config = load_model_from_checkpoint(checkpoint_path, config_path=None, device=device)

    data_cfg = config["data"]
    patch_size = int(data_cfg.get("patch_size", 256))
    overlap = float(config.get("inference", {}).get("overlap", data_cfg.get("overlap", 0.25)))
    normalization = str(data_cfg.get("normalization", "minmax"))
    preprocess_cfg = data_cfg.get("preprocess")

    items: list[dict[str, Any]] = []
    for image_path in image_paths:
        image = read_gray(image_path)
        processed = apply_preprocess(image, preprocess_cfg)
        image_tensor = image_to_tensor(processed, normalization=normalization).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = sliding_window_inference(model, image_tensor, patch_size=patch_size, overlap=overlap)
        mask = (torch.sigmoid(logits).cpu().numpy()[0, 0] >= threshold).astype(np.uint8) * 255
        write_image(masks_dir / f"{image_path.stem}_mask.png", mask)
        write_image(previews_dir / f"{image_path.stem}_overlay.png", build_overlay(image, mask))
        items.append(
            {
                "stem": image_path.stem,
                "image_path": str(image_path),
                "mask_path": str(masks_dir / f"{image_path.stem}_mask.png"),
                "foreground_pixels": int(np.count_nonzero(mask)),
                "height": int(mask.shape[0]),
                "width": int(mask.shape[1]),
            }
        )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    save_json(output_dir / "pseudo_manifest.json", {"num_images": len(items), "items": items, "checkpoint": str(checkpoint_path)})
    return items


def build_pseudo_prepared_root(
    image_paths: list[Path],
    pseudo_output_dir: str | Path,
    output_root: str | Path,
    edge_kernels: tuple[int, ...] = (3, 5),
) -> dict[str, Any]:
    pseudo_output_dir = Path(pseudo_output_dir)
    output_root = ensure_dir(output_root)
    images_out = ensure_dir(output_root / "images")
    masks_out = ensure_dir(output_root / "masks")
    previews_out = ensure_dir(output_root / "previews")
    manifests_out = ensure_dir(output_root / "manifests")
    edge_dirs = {kernel: ensure_dir(output_root / "edges" / f"k{kernel}") for kernel in edge_kernels}

    items: list[dict[str, Any]] = []
    for image_path in image_paths:
        stem = image_path.stem
        source_image = read_gray(image_path)
        source_mask = read_gray(pseudo_output_dir / "masks" / f"{stem}_mask.png")
        image_rel = Path("images") / f"{stem}.png"
        mask_rel = Path("masks") / f"{stem}.png"
        preview_rel = Path("previews") / f"{stem}_overlay.png"
        write_image(output_root / image_rel, source_image)
        write_image(output_root / mask_rel, source_mask)
        write_image(output_root / preview_rel, build_overlay(source_image, source_mask))

        edge_rel_map: dict[str, str] = {}
        for kernel in edge_kernels:
            edge_mask = build_edge_mask(source_mask, kernel)
            edge_rel = Path("edges") / f"k{kernel}" / f"{stem}.png"
            write_image(output_root / edge_rel, edge_mask)
            edge_rel_map[f"k{kernel}"] = edge_rel.as_posix()

        items.append(
            {
                "stem": stem,
                "image_path": image_rel.as_posix(),
                "mask_path": mask_rel.as_posix(),
                "edge_paths": edge_rel_map,
                "source_image_path": str(image_path),
                "source_mask_path": str(pseudo_output_dir / "masks" / f"{stem}_mask.png"),
                "height": int(source_image.shape[0]),
                "width": int(source_image.shape[1]),
                "foreground_pixels": int(np.count_nonzero(source_mask)),
                "crop_height": int(source_image.shape[0]),
            }
        )

    save_json(
        manifests_out / "dataset.json",
        {
            "num_samples": len(items),
            "source": "pseudo_labels",
            "items": items,
        },
    )
    return {"prepared_root": str(output_root), "num_samples": len(items)}


def absolutize_item(prepared_root: Path, item: dict[str, Any], stem_override: str | None = None) -> dict[str, Any]:
    cloned = dict(item)
    if stem_override:
        cloned["stem"] = stem_override
    cloned["image_path"] = str((prepared_root / item["image_path"]).resolve())
    cloned["mask_path"] = str((prepared_root / item["mask_path"]).resolve())
    cloned["edge_paths"] = {key: str((prepared_root / value).resolve()) for key, value in dict(item["edge_paths"]).items()}
    return cloned


def build_merged_fold_root(
    supervised_root: str | Path,
    fold_manifest_name: str,
    fold_index: int,
    pseudo_prepared_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    supervised_root = Path(supervised_root)
    pseudo_prepared_root = Path(pseudo_prepared_root)
    output_root = ensure_dir(output_root)
    manifests_out = ensure_dir(output_root / "manifests")

    supervised_dataset = load_dataset_manifest(supervised_root)
    supervised_folds = load_fold_manifest(supervised_root, fold_manifest_name)
    pseudo_dataset = load_dataset_manifest(pseudo_prepared_root)

    sup_items_by_stem = {item["stem"]: item for item in supervised_dataset["items"]}
    target_fold = next(fold for fold in supervised_folds["folds"] if int(fold["fold_index"]) == int(fold_index))
    train_stems = list(target_fold["train_stems"])
    val_stems = list(target_fold["val_stems"])

    items: list[dict[str, Any]] = []
    for stem in train_stems + val_stems:
        items.append(absolutize_item(supervised_root, sup_items_by_stem[stem]))

    pseudo_train_stems: list[str] = []
    for pseudo_item in pseudo_dataset["items"]:
        pseudo_stem = f"pseudo__{pseudo_item['stem']}"
        items.append(absolutize_item(pseudo_prepared_root, pseudo_item, stem_override=pseudo_stem))
        pseudo_train_stems.append(pseudo_stem)

    folds_manifest = {
        "num_folds": 1,
        "seed": 42,
        "folds": [
            {
                "fold_index": 0,
                "train_stems": train_stems + pseudo_train_stems,
                "val_stems": val_stems,
            }
        ],
    }
    dataset_manifest = {
        "num_samples": len(items),
        "source": "semi_supervised_merged",
        "supervised_root": str(supervised_root),
        "pseudo_root": str(pseudo_prepared_root),
        "fold_index": int(fold_index),
        "items": items,
    }
    save_json(manifests_out / "dataset.json", dataset_manifest)
    save_json(manifests_out / "folds_1_seed42.json", folds_manifest)
    return {
        "prepared_root": str(output_root),
        "num_train_supervised": len(train_stems),
        "num_train_pseudo": len(pseudo_train_stems),
        "num_val": len(val_stems),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="执行 E4 半监督伪标签实验")
    parser.add_argument("--base-config", required=True, help="监督基线配置，例如 e2_micronet_edge_deep_gpu.yaml")
    parser.add_argument("--teacher-root", required=True, help="监督基线输出目录，包含 fold_0..fold_n best.pt")
    parser.add_argument("--supervised-prepared-root", required=True, help="prepared_main 目录")
    parser.add_argument("--unlabeled-images-dir", required=True, help="未标注图像目录，建议用 full_png 修正裁切后的 images")
    parser.add_argument("--output-root", required=True, help="E4 输出根目录")
    parser.add_argument("--device", default="cuda", help="cuda/cpu/auto")
    parser.add_argument("--epochs", type=int, default=30, help="半监督微调 epoch")
    parser.add_argument("--learning-rate", type=float, default=3e-5, help="半监督微调学习率")
    parser.add_argument("--threshold", type=float, default=0.5, help="伪标签二值化阈值")
    parser.add_argument("--folds", nargs="*", type=int, default=None, help="仅运行指定 teacher fold 列表，例如 --folds 1 2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = load_yaml(args.base_config)
    seed_everything(int(base_config["experiment"].get("seed", 42)))

    supervised_root = Path(args.supervised_prepared_root)
    output_root = ensure_dir(args.output_root)
    teacher_root = Path(args.teacher_root)
    fold_manifest_name = str(base_config["data"].get("fold_manifest_name", "folds_3_seed42.json"))
    supervised_dataset = load_dataset_manifest(supervised_root)
    supervised_stems = {item["stem"] for item in supervised_dataset["items"]}
    unlabeled_images = list_unlabeled_images(args.unlabeled_images_dir, exclude_stems=supervised_stems)

    if not unlabeled_images:
        raise ValueError("未找到可用未标注图像，当前 unlabeled 池为空")

    baseline_summary = json.loads((teacher_root / "crossval_summary.json").read_text(encoding="utf-8"))
    fold_results: list[dict[str, Any]] = []

    target_folds = args.folds if args.folds else list(range(int(base_config["data"].get("num_folds", 3))))

    for fold_index in target_folds:
        teacher_checkpoint = teacher_root / f"fold_{fold_index}" / "best.pt"
        if not teacher_checkpoint.exists():
            raise FileNotFoundError(f"缺少 teacher checkpoint: {teacher_checkpoint}")

        pseudo_dir = output_root / "pseudo_predictions" / f"fold_{fold_index}"
        pseudo_prepared_root = output_root / "pseudo_prepared" / f"fold_{fold_index}"
        merged_root = output_root / "merged_prepared" / f"fold_{fold_index}"
        train_output_root = output_root / "training" / f"fold_{fold_index}"

        train_summary_path = train_output_root / "fold_0" / "summary.json"
        if train_summary_path.exists():
            semi_result = json.loads(train_summary_path.read_text(encoding="utf-8"))
            merge_info = {
                "prepared_root": str(merged_root),
                "num_train_supervised": len(load_prepared_records(supervised_root, fold_manifest_name, fold_index, stage="train")),
                "num_train_pseudo": len(unlabeled_images),
                "num_val": len(load_prepared_records(supervised_root, fold_manifest_name, fold_index, stage="val")),
            }
        else:
            generate_pseudo_masks(
                checkpoint_path=teacher_checkpoint,
                image_paths=unlabeled_images,
                output_dir=pseudo_dir,
                device_name=args.device,
                threshold=float(args.threshold),
            )
            build_pseudo_prepared_root(
                image_paths=unlabeled_images,
                pseudo_output_dir=pseudo_dir,
                output_root=pseudo_prepared_root,
            )
            merge_info = build_merged_fold_root(
                supervised_root=supervised_root,
                fold_manifest_name=fold_manifest_name,
                fold_index=fold_index,
                pseudo_prepared_root=pseudo_prepared_root,
                output_root=merged_root,
            )

            fold_config = json.loads(json.dumps(base_config))
            fold_config["experiment"]["name"] = f"{base_config['experiment']['name']}_semi_fold{fold_index}"
            fold_config["experiment"]["output_root"] = str(train_output_root)
            fold_config["data"]["prepared_root"] = str(merged_root)
            fold_config["data"]["fold_manifest_name"] = "folds_1_seed42.json"
            fold_config["data"]["num_folds"] = 1
            fold_config["training"]["epochs"] = int(args.epochs)
            fold_config["training"]["learning_rate"] = float(args.learning_rate)
            fold_config["training"]["init_checkpoint"] = str(teacher_checkpoint)

            semi_result = train_one_fold(fold_config, fold_index=0)
        semi_result["teacher_checkpoint"] = str(teacher_checkpoint)
        semi_result["merge_info"] = merge_info

        baseline_fold = next(item for item in baseline_summary["fold_results"] if int(item["fold_index"]) == int(fold_index))
        fold_results.append(
            {
                "teacher_fold_index": fold_index,
                "baseline": baseline_fold["best_summary"],
                "semi_supervised": semi_result["best_summary"],
                "teacher_checkpoint": str(teacher_checkpoint),
                "semi_checkpoint": semi_result["checkpoint_path"],
                "merge_info": merge_info,
            }
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    comparison_rows = []
    for item in fold_results:
        baseline = item["baseline"]
        semi = item["semi_supervised"]
        comparison_rows.append(
            {
                "dice_gain": float(semi.get("dice", 0.0) - baseline.get("dice", 0.0)),
                "iou_gain": float(semi.get("iou", 0.0) - baseline.get("iou", 0.0)),
                "precision_gain": float(semi.get("precision", 0.0) - baseline.get("precision", 0.0)),
                "recall_gain": float(semi.get("recall", 0.0) - baseline.get("recall", 0.0)),
                "vf_gain": float(semi.get("vf", 0.0) - baseline.get("vf", 0.0)),
                "boundary_f1_gain": float(semi.get("boundary_f1", 0.0) - baseline.get("boundary_f1", 0.0)),
            }
        )

    summary = {
        "experiment": "e4_semi_supervised_eval",
        "teacher_root": str(teacher_root),
        "unlabeled_pool_size": len(unlabeled_images),
        "fold_results": fold_results,
        "mean_gain": average_metric_dicts(comparison_rows),
    }
    save_json(output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
