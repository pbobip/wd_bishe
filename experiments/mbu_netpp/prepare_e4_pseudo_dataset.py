from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from experiments.mbu_netpp.common import ensure_dir, save_json
from experiments.mbu_netpp.preparation import prepare_paired_dataset


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def copy_with_prefix(source_root: Path, rel_path: str, dest_root: Path, stem_prefix: str) -> str:
    source = source_root / rel_path
    relative = Path(rel_path)
    target_name = f"{stem_prefix}{relative.name}"
    target = dest_root / relative.parent / target_name
    ensure_dir(target.parent)
    shutil.copy2(source, target)
    return target.relative_to(dest_root).as_posix()


def build_combined_dataset(
    supervised_prepared_root: str | Path,
    pseudo_images_dir: str | Path,
    pseudo_masks_dir: str | Path,
    output_root: str | Path,
    supervised_repeat: int = 12,
    pseudo_workdir_name: str = "_pseudo_prepared",
) -> dict[str, Any]:
    supervised_root = Path(supervised_prepared_root)
    output_root = ensure_dir(output_root)
    pseudo_root = output_root / pseudo_workdir_name

    prepare_paired_dataset(
        images_dir=pseudo_images_dir,
        masks_dir=pseudo_masks_dir,
        output_root=pseudo_root,
        auto_crop_sem_region=False,
        crop_detection_ratio=0.75,
        edge_kernels=(3, 5),
        mask_mode="non_gray",
    )

    combined_images = ensure_dir(output_root / "images")
    combined_masks = ensure_dir(output_root / "masks")
    combined_previews = ensure_dir(output_root / "previews")
    combined_manifests = ensure_dir(output_root / "manifests")
    combined_edges = {
        "k3": ensure_dir(output_root / "edges" / "k3"),
        "k5": ensure_dir(output_root / "edges" / "k5"),
    }

    supervised_dataset = load_json(supervised_root / "manifests" / "dataset.json")
    supervised_folds = load_json(supervised_root / "manifests" / "folds_3_seed42.json")
    pseudo_dataset = load_json(pseudo_root / "manifests" / "dataset.json")

    combined_items: list[dict[str, Any]] = []

    for item in supervised_dataset["items"]:
        combined_items.append(
            {
                **item,
                "source_kind": "supervised",
                "image_path": copy_with_prefix(supervised_root, item["image_path"], output_root, ""),
                "mask_path": copy_with_prefix(supervised_root, item["mask_path"], output_root, ""),
                "edge_paths": {
                    key: copy_with_prefix(supervised_root, rel_path, output_root, "")
                    for key, rel_path in item["edge_paths"].items()
                },
            }
        )
        preview_path = Path("previews") / f"{item['stem']}_overlay.png"
        source_preview = supervised_root / preview_path
        if source_preview.exists():
            copy_with_prefix(supervised_root, preview_path.as_posix(), output_root, "")

    pseudo_stems: list[str] = []
    for item in pseudo_dataset["items"]:
        pseudo_stem = f"pseudo__{item['stem']}"
        pseudo_stems.append(pseudo_stem)
        combined_items.append(
            {
                **item,
                "stem": pseudo_stem,
                "source_kind": "pseudo",
                "teacher_mask_dir": str(pseudo_masks_dir),
                "image_path": copy_with_prefix(pseudo_root, item["image_path"], output_root, "pseudo__"),
                "mask_path": copy_with_prefix(pseudo_root, item["mask_path"], output_root, "pseudo__"),
                "edge_paths": {
                    key: copy_with_prefix(pseudo_root, rel_path, output_root, "pseudo__")
                    for key, rel_path in item["edge_paths"].items()
                },
            }
        )
        preview_path = Path("previews") / f"{item['stem']}_overlay.png"
        source_preview = pseudo_root / preview_path
        if source_preview.exists():
            copy_with_prefix(pseudo_root, preview_path.as_posix(), output_root, "pseudo__")

    combined_dataset_manifest = {
        "num_samples": len(combined_items),
        "num_supervised_samples": len(supervised_dataset["items"]),
        "num_pseudo_samples": len(pseudo_stems),
        "pseudo_images_dir": str(pseudo_images_dir),
        "pseudo_masks_dir": str(pseudo_masks_dir),
        "supervised_repeat": int(supervised_repeat),
        "items": combined_items,
    }

    combined_folds = []
    for fold in supervised_folds["folds"]:
        train_stems = list(fold["train_stems"]) * int(supervised_repeat)
        train_stems.extend(pseudo_stems)
        combined_folds.append(
            {
                "fold_index": int(fold["fold_index"]),
                "train_stems": train_stems,
                "val_stems": list(fold["val_stems"]),
            }
        )

    combined_fold_manifest = {
        "num_folds": int(supervised_folds["num_folds"]),
        "seed": int(supervised_folds["seed"]),
        "supervised_repeat": int(supervised_repeat),
        "pseudo_only_in_train": True,
        "folds": combined_folds,
    }

    save_json(combined_manifests / "dataset.json", combined_dataset_manifest)
    save_json(combined_manifests / "folds_3_seed42_pseudo.json", combined_fold_manifest)
    return {
        "output_root": str(output_root),
        "num_supervised_samples": len(supervised_dataset["items"]),
        "num_pseudo_samples": len(pseudo_stems),
        "supervised_repeat": int(supervised_repeat),
        "dataset_manifest_path": str(combined_manifests / "dataset.json"),
        "fold_manifest_path": str(combined_manifests / "folds_3_seed42_pseudo.json"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 E4 伪标签组合训练集")
    parser.add_argument("--supervised-prepared-root", required=True, help="原始监督 prepared_root")
    parser.add_argument("--pseudo-images-dir", required=True, help="伪标签原图目录")
    parser.add_argument("--pseudo-masks-dir", required=True, help="伪标签 mask 目录")
    parser.add_argument("--output-root", required=True, help="E4 组合数据集输出目录")
    parser.add_argument("--supervised-repeat", type=int, default=12, help="监督样本在训练清单里的重复次数")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_combined_dataset(
        supervised_prepared_root=args.supervised_prepared_root,
        pseudo_images_dir=args.pseudo_images_dir,
        pseudo_masks_dir=args.pseudo_masks_dir,
        output_root=args.output_root,
        supervised_repeat=int(args.supervised_repeat),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
