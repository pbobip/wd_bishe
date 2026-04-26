from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.mbu_netpp.prepare_merged_supervised import (
    build_source_balanced_folds,
    copy_record_assets,
    ensure_dir,
    load_json,
    normalize_alias,
    save_json,
)


def prefix_stems(stems: list[str], alias: str) -> list[str]:
    return [f"{alias}__{stem}" for stem in stems]


def prepare_opt_training_dataset(
    real_prepared_root: str | Path,
    aux_prepared_root: str | Path,
    output_root: str | Path,
    mode: str,
    num_folds: int,
    seed: int,
    holdout_manifest_name: str,
    real_fold_manifest_name: str,
    real_alias: str = "real",
    aux_alias: str = "nasa",
    real_sample_weight: float = 1.0,
    real_sampling_weight: float = 1.0,
    aux_sample_weight: float = 0.25,
    aux_sampling_weight: float = 0.15,
    copy_previews: bool = True,
) -> dict[str, Any]:
    if mode not in {"train_only_aux", "merged_trainval_with_external_holdout"}:
        raise ValueError(f"不支持的 mode: {mode}")

    real_root = Path(real_prepared_root).resolve()
    aux_root = Path(aux_prepared_root).resolve()
    output_root = ensure_dir(output_root).resolve()
    ensure_dir(output_root / "images")
    ensure_dir(output_root / "masks")
    ensure_dir(output_root / "manifests")
    if copy_previews:
        ensure_dir(output_root / "previews")

    real_alias = normalize_alias(real_alias)
    aux_alias = normalize_alias(aux_alias)

    real_dataset = load_json(real_root / "manifests" / "dataset.json")
    aux_dataset = load_json(aux_root / "manifests" / "dataset.json")
    real_holdout = load_json(real_root / "manifests" / holdout_manifest_name)
    real_folds = load_json(real_root / "manifests" / real_fold_manifest_name)

    merged_items: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    edge_keys: set[str] = set()

    for source_root, source_alias, dataset_payload, sample_weight, sampling_weight in [
        (real_root, real_alias, real_dataset, real_sample_weight, real_sampling_weight),
        (aux_root, aux_alias, aux_dataset, aux_sample_weight, aux_sampling_weight),
    ]:
        source_items = list(dataset_payload.get("items") or [])
        source_counts[source_alias] = len(source_items)
        for item in source_items:
            merged_record = copy_record_assets(
                source_root=source_root,
                output_root=output_root,
                source_alias=source_alias,
                item=item,
                sample_weight=float(sample_weight),
                copy_previews=copy_previews,
            )
            merged_record["sampling_weight"] = float(sampling_weight)
            merged_items.append(merged_record)
            edge_keys.update(dict(merged_record.get("edge_paths") or {}).keys())

    real_test_stems = prefix_stems(list(real_holdout["test_stems"]), real_alias)
    real_trainval_stems = prefix_stems(list(real_holdout["trainval_stems"]), real_alias)
    aux_all_stems = [item["stem"] for item in merged_items if str(item["stem"]).startswith(f"{aux_alias}__")]

    if mode == "train_only_aux":
        folds: list[dict[str, Any]] = []
        for real_fold in real_folds["folds"]:
            folds.append(
                {
                    "fold_index": int(real_fold["fold_index"]),
                    "train_stems": prefix_stems(list(real_fold["train_stems"]), real_alias) + list(aux_all_stems),
                    "val_stems": prefix_stems(list(real_fold["val_stems"]), real_alias),
                    "test_stems": list(real_test_stems),
                }
            )
        strategy = "real_holdout_plus_aux_train_only"
        holdout_trainval_stems = list(real_trainval_stems) + list(aux_all_stems)
    else:
        folds = build_source_balanced_folds(
            stems_by_source={
                real_alias: list(real_trainval_stems),
                aux_alias: list(aux_all_stems),
            },
            num_folds=int(num_folds),
            seed=int(seed),
        )
        for fold in folds:
            fold["test_stems"] = list(real_test_stems)
        strategy = "merged_trainval_with_external_real_holdout"
        holdout_trainval_stems = list(real_trainval_stems) + list(aux_all_stems)

    dataset_manifest = {
        "num_samples": len(merged_items),
        "num_sources": 2,
        "source_aliases": [real_alias, aux_alias],
        "source_counts": source_counts,
        "sample_weights": {
            real_alias: float(real_sample_weight),
            aux_alias: float(aux_sample_weight),
        },
        "sampling_weights": {
            real_alias: float(real_sampling_weight),
            aux_alias: float(aux_sampling_weight),
        },
        "edge_kernels": sorted(edge_keys),
        "copy_previews": bool(copy_previews),
        "items": merged_items,
    }
    holdout_manifest = {
        "num_samples": len(merged_items),
        "num_test": len(real_test_stems),
        "num_trainval": len(holdout_trainval_stems),
        "seed": int(seed),
        "mode": mode,
        "test_stems": list(real_test_stems),
        "trainval_stems": holdout_trainval_stems,
    }
    fold_manifest = {
        "num_folds": int(num_folds),
        "seed": int(seed),
        "strategy": strategy,
        "holdout_manifest_name": holdout_manifest_name,
        "test_stems": list(real_test_stems),
        "folds": folds,
    }

    dataset_manifest_path = output_root / "manifests" / "dataset.json"
    holdout_manifest_path = output_root / "manifests" / holdout_manifest_name
    fold_manifest_path = output_root / "manifests" / real_fold_manifest_name
    save_json(dataset_manifest_path, dataset_manifest)
    save_json(holdout_manifest_path, holdout_manifest)
    save_json(fold_manifest_path, fold_manifest)
    return {
        "output_root": str(output_root),
        "dataset_manifest_path": str(dataset_manifest_path),
        "holdout_manifest_path": str(holdout_manifest_path),
        "fold_manifest_path": str(fold_manifest_path),
        "mode": mode,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构造优化实验用的合并训练集")
    parser.add_argument("--real-prepared-root", required=True)
    parser.add_argument("--aux-prepared-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--mode",
        required=True,
        choices=["train_only_aux", "merged_trainval_with_external_holdout"],
    )
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-manifest-name", required=True)
    parser.add_argument("--real-fold-manifest-name", required=True)
    parser.add_argument("--real-alias", default="real")
    parser.add_argument("--aux-alias", default="nasa")
    parser.add_argument("--real-sample-weight", type=float, default=1.0)
    parser.add_argument("--real-sampling-weight", type=float, default=1.0)
    parser.add_argument("--aux-sample-weight", type=float, default=0.25)
    parser.add_argument("--aux-sampling-weight", type=float, default=0.15)
    parser.add_argument("--no-copy-previews", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = prepare_opt_training_dataset(
        real_prepared_root=args.real_prepared_root,
        aux_prepared_root=args.aux_prepared_root,
        output_root=args.output_root,
        mode=args.mode,
        num_folds=args.num_folds,
        seed=args.seed,
        holdout_manifest_name=args.holdout_manifest_name,
        real_fold_manifest_name=args.real_fold_manifest_name,
        real_alias=args.real_alias,
        aux_alias=args.aux_alias,
        real_sample_weight=args.real_sample_weight,
        real_sampling_weight=args.real_sampling_weight,
        aux_sample_weight=args.aux_sample_weight,
        aux_sampling_weight=args.aux_sampling_weight,
        copy_previews=not args.no_copy_previews,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
