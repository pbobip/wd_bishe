from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.mbu_netpp.common import ensure_dir, save_json
from experiments.mbu_netpp.preparation import prepare_paired_dataset


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def to_absolute_record_paths(root: Path, item: dict[str, Any]) -> dict[str, Any]:
    record = dict(item)
    for key in ("image_path", "mask_path"):
        record[key] = str((root / str(item[key])).resolve())
    record["edge_paths"] = {
        edge_key: str((root / str(edge_path)).resolve())
        for edge_key, edge_path in dict(item.get("edge_paths") or {}).items()
    }
    return record


def build_pseudo_items(
    pseudo_root: Path,
    pseudo_manifest: dict[str, Any],
    excluded_stems: set[str],
    prefix: str,
    teacher_name: str,
) -> list[dict[str, Any]]:
    pseudo_items: list[dict[str, Any]] = []
    for item in pseudo_manifest.get("items", []):
        stem = str(item["stem"])
        if stem in excluded_stems:
            continue
        record = to_absolute_record_paths(pseudo_root, item)
        record["stem"] = f"{prefix}{stem}"
        record["source_type"] = "pseudo"
        record["teacher_name"] = teacher_name
        record["source_stem"] = stem
        pseudo_items.append(record)
    return pseudo_items


def build_supervised_items(supervised_root: Path, supervised_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for item in supervised_manifest.get("items", []):
        record = to_absolute_record_paths(supervised_root, item)
        record["source_type"] = "supervised"
        record["source_stem"] = str(item["stem"])
        items.append(record)
    return items


def build_combined_folds(
    supervised_folds: dict[str, Any],
    pseudo_stems: list[str],
    supervised_repeat: int,
) -> dict[str, Any]:
    folds: list[dict[str, Any]] = []
    for fold in supervised_folds.get("folds", []):
        train_stems = [str(stem) for stem in fold["train_stems"]]
        repeated_supervised = train_stems * max(1, int(supervised_repeat))
        combined_train = repeated_supervised + list(pseudo_stems)
        folds.append(
            {
                "fold_index": int(fold["fold_index"]),
                "train_stems": combined_train,
                "val_stems": [str(stem) for stem in fold["val_stems"]],
            }
        )
    return {
        "num_folds": int(supervised_folds.get("num_folds", len(folds))),
        "seed": int(supervised_folds.get("seed", 42)),
        "supervised_repeat": int(supervised_repeat),
        "pseudo_count": len(pseudo_stems),
        "folds": folds,
    }


def prepare_e4_pseudo_mix(
    supervised_prepared_root: str | Path,
    pseudo_images_dir: str | Path,
    pseudo_masks_dir: str | Path,
    pseudo_prepared_root: str | Path,
    output_root: str | Path,
    teacher_name: str,
    pseudo_prefix: str = "pseudo__",
    supervised_repeat: int = 16,
) -> dict[str, Any]:
    supervised_root = Path(supervised_prepared_root)
    pseudo_images_dir = Path(pseudo_images_dir)
    pseudo_masks_dir = Path(pseudo_masks_dir)
    pseudo_root = Path(pseudo_prepared_root)
    output_root = ensure_dir(output_root)
    manifests_root = ensure_dir(output_root / "manifests")

    prepare_paired_dataset(
        images_dir=pseudo_images_dir,
        masks_dir=pseudo_masks_dir,
        output_root=pseudo_root,
        auto_crop_sem_region=False,
        edge_kernels=(3, 5),
        mask_mode="non_gray",
    )

    supervised_manifest = load_json(supervised_root / "manifests" / "dataset.json")
    supervised_folds = load_json(supervised_root / "manifests" / "folds_3_seed42.json")
    pseudo_manifest = load_json(pseudo_root / "manifests" / "dataset.json")

    supervised_stems = {str(item["stem"]) for item in supervised_manifest.get("items", [])}
    supervised_items = build_supervised_items(supervised_root, supervised_manifest)
    pseudo_items = build_pseudo_items(
        pseudo_root=pseudo_root,
        pseudo_manifest=pseudo_manifest,
        excluded_stems=supervised_stems,
        prefix=pseudo_prefix,
        teacher_name=teacher_name,
    )

    combined_manifest = {
        "num_samples": len(supervised_items) + len(pseudo_items),
        "num_supervised": len(supervised_items),
        "num_pseudo": len(pseudo_items),
        "pseudo_prefix": pseudo_prefix,
        "teacher_name": teacher_name,
        "items": supervised_items + pseudo_items,
    }
    save_json(manifests_root / "dataset.json", combined_manifest)

    combined_folds = build_combined_folds(
        supervised_folds=supervised_folds,
        pseudo_stems=[item["stem"] for item in pseudo_items],
        supervised_repeat=supervised_repeat,
    )
    save_json(manifests_root / "folds_3_seed42.json", combined_folds)

    return {
        "output_root": str(output_root),
        "pseudo_prepared_root": str(pseudo_root),
        "num_supervised": len(supervised_items),
        "num_pseudo": len(pseudo_items),
        "excluded_overlap_count": len(pseudo_manifest.get("items", [])) - len(pseudo_items),
        "dataset_manifest_path": str(manifests_root / "dataset.json"),
        "fold_manifest_path": str(manifests_root / "folds_3_seed42.json"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="准备 E4 伪标签半监督混合数据集")
    parser.add_argument("--supervised-prepared-root", required=True)
    parser.add_argument("--pseudo-images-dir", required=True)
    parser.add_argument("--pseudo-masks-dir", required=True)
    parser.add_argument("--pseudo-prepared-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--teacher-name", required=True)
    parser.add_argument("--pseudo-prefix", default="pseudo__")
    parser.add_argument("--supervised-repeat", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = prepare_e4_pseudo_mix(
        supervised_prepared_root=args.supervised_prepared_root,
        pseudo_images_dir=args.pseudo_images_dir,
        pseudo_masks_dir=args.pseudo_masks_dir,
        pseudo_prepared_root=args.pseudo_prepared_root,
        output_root=args.output_root,
        teacher_name=args.teacher_name,
        pseudo_prefix=args.pseudo_prefix,
        supervised_repeat=int(args.supervised_repeat),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
