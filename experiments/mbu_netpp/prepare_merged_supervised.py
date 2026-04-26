from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def save_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def normalize_alias(value: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z_-]+", "_", str(value).strip())
    normalized = re.sub(r"_+", "_", normalized).strip("_").lower()
    if not normalized:
        raise ValueError(f"无法从输入生成有效 source alias: {value!r}")
    return normalized


def normalize_manifest_path(raw_path: str | Path) -> str:
    return str(raw_path).strip().replace("\\", "/")


def resolve_source_path(source_root: Path, raw_path: str | Path) -> Path:
    path = Path(normalize_manifest_path(raw_path))
    if path.is_absolute():
        return path
    return (source_root / path).resolve()


def derive_preview_source_path(item: dict[str, Any]) -> str | None:
    explicit = str(item.get("preview_path", "")).strip()
    if explicit:
        return explicit

    image_path = normalize_manifest_path(item.get("image_path", ""))
    if not image_path:
        return None

    rel = Path(image_path)
    parts = list(rel.parts)
    if "images" in parts:
        index = parts.index("images")
        preview_parts = list(parts[:index]) + ["previews"] + list(parts[index + 1 : -1]) + [f"{rel.stem}_overlay.png"]
        return Path(*preview_parts).as_posix()
    return (Path("previews") / f"{rel.stem}_overlay.png").as_posix()


def copy_file(source: Path, target: Path) -> None:
    ensure_dir(target.parent)
    shutil.copy2(source, target)


def copy_record_assets(
    source_root: Path,
    output_root: Path,
    source_alias: str,
    item: dict[str, Any],
    sample_weight: float,
    copy_previews: bool,
) -> dict[str, Any]:
    source_stem = str(item["stem"])
    merged_stem = f"{source_alias}__{source_stem}"

    source_image = resolve_source_path(source_root, item["image_path"])
    source_mask = resolve_source_path(source_root, item["mask_path"])

    image_target_rel = Path("images") / source_alias / f"{merged_stem}.png"
    mask_target_rel = Path("masks") / source_alias / f"{merged_stem}.png"
    copy_file(source_image, output_root / image_target_rel)
    copy_file(source_mask, output_root / mask_target_rel)

    edge_rel_map: dict[str, str] = {}
    for edge_key, edge_path in dict(item.get("edge_paths") or {}).items():
        source_edge = resolve_source_path(source_root, edge_path)
        edge_target_rel = Path("edges") / edge_key / source_alias / f"{merged_stem}.png"
        copy_file(source_edge, output_root / edge_target_rel)
        edge_rel_map[edge_key] = edge_target_rel.as_posix()

    preview_target_rel = ""
    if copy_previews:
        preview_source_path = derive_preview_source_path(item)
        if preview_source_path:
            source_preview = resolve_source_path(source_root, preview_source_path)
            if source_preview.exists():
                preview_target_rel = (Path("previews") / source_alias / f"{merged_stem}_overlay.png").as_posix()
                copy_file(source_preview, output_root / preview_target_rel)

    record = dict(item)
    record["stem"] = merged_stem
    record["source_stem"] = source_stem
    record["source_dataset_alias"] = source_alias
    record["source_prepared_root"] = str(source_root.resolve())
    record["source_type"] = str(item.get("source_type", "supervised"))
    record["sample_weight"] = float(sample_weight)
    record["image_path"] = image_target_rel.as_posix()
    record["mask_path"] = mask_target_rel.as_posix()
    record["edge_paths"] = edge_rel_map
    if preview_target_rel:
        record["preview_path"] = preview_target_rel
    return record


def build_source_balanced_folds(
    stems_by_source: dict[str, list[str]],
    num_folds: int,
    seed: int,
) -> list[dict[str, Any]]:
    all_stems: list[str] = []
    per_fold_val: list[list[str]] = [[] for _ in range(num_folds)]

    for source_alias in sorted(stems_by_source):
        source_stems = list(stems_by_source[source_alias])
        all_stems.extend(source_stems)
        rng = random.Random(f"{seed}:{source_alias}")
        rng.shuffle(source_stems)
        splits = np.array_split(np.asarray(source_stems, dtype=object), num_folds)
        for fold_index, split in enumerate(splits):
            per_fold_val[fold_index].extend([str(item) for item in split.tolist()])

    folds: list[dict[str, Any]] = []
    for fold_index, val_stems in enumerate(per_fold_val):
        val_set = set(val_stems)
        train_stems = [stem for stem in all_stems if stem not in val_set]
        folds.append(
            {
                "fold_index": int(fold_index),
                "train_stems": train_stems,
                "val_stems": val_stems,
            }
        )
    return folds


def prepare_merged_supervised_dataset(
    prepared_roots: list[str | Path],
    output_root: str | Path,
    source_aliases: list[str] | None = None,
    sample_weights: list[float] | None = None,
    num_folds: int = 3,
    seed: int = 42,
    copy_previews: bool = True,
) -> dict[str, Any]:
    if not prepared_roots:
        raise ValueError("至少需要一个 prepared_root")
    if num_folds < 2:
        raise ValueError("num_folds 必须 >= 2")

    roots = [Path(path).resolve() for path in prepared_roots]
    aliases = list(source_aliases or [])
    if aliases and len(aliases) != len(roots):
        raise ValueError("source_aliases 数量必须与 prepared_roots 一致")
    if not aliases:
        aliases = [normalize_alias(root.name) for root in roots]
    else:
        aliases = [normalize_alias(value) for value in aliases]
    if len(set(aliases)) != len(aliases):
        raise ValueError(f"source_alias 重复: {aliases}")

    weights = list(sample_weights or [])
    if weights and len(weights) != len(roots):
        raise ValueError("sample_weights 数量必须与 prepared_roots 一致")
    if not weights:
        weights = [1.0] * len(roots)

    output_root = ensure_dir(output_root).resolve()
    ensure_dir(output_root / "images")
    ensure_dir(output_root / "masks")
    ensure_dir(output_root / "manifests")
    if copy_previews:
        ensure_dir(output_root / "previews")

    merged_items: list[dict[str, Any]] = []
    stems_by_source: dict[str, list[str]] = defaultdict(list)
    source_counts: dict[str, int] = {}
    edge_keys: set[str] = set()

    for source_root, source_alias, sample_weight in zip(roots, aliases, weights):
        dataset_manifest_path = source_root / "manifests" / "dataset.json"
        if not dataset_manifest_path.exists():
            raise FileNotFoundError(f"未找到 dataset manifest: {dataset_manifest_path}")
        dataset_manifest = load_json(dataset_manifest_path)
        source_items = list(dataset_manifest.get("items") or [])
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
            merged_items.append(merged_record)
            stems_by_source[source_alias].append(str(merged_record["stem"]))
            edge_keys.update(dict(merged_record.get("edge_paths") or {}).keys())

    merged_stems = [str(item["stem"]) for item in merged_items]
    if len(merged_stems) != len(set(merged_stems)):
        raise ValueError("合并后 stem 出现重复，请检查 source_alias 或原始数据")

    folds = build_source_balanced_folds(
        stems_by_source={alias: list(stems) for alias, stems in stems_by_source.items()},
        num_folds=int(num_folds),
        seed=int(seed),
    )

    fold_manifest_name = f"folds_{int(num_folds)}_seed{int(seed)}.json"
    dataset_manifest = {
        "num_samples": len(merged_items),
        "num_sources": len(roots),
        "source_aliases": aliases,
        "source_counts": source_counts,
        "sample_weights": {alias: float(weight) for alias, weight in zip(aliases, weights)},
        "edge_kernels": sorted(edge_keys),
        "copy_previews": bool(copy_previews),
        "items": merged_items,
    }
    fold_manifest = {
        "num_folds": int(num_folds),
        "seed": int(seed),
        "strategy": "source_balanced_random_split",
        "folds": folds,
    }

    save_json(output_root / "manifests" / "dataset.json", dataset_manifest)
    save_json(output_root / "manifests" / fold_manifest_name, fold_manifest)
    return {
        "output_root": str(output_root),
        "num_samples": len(merged_items),
        "num_sources": len(roots),
        "source_counts": source_counts,
        "dataset_manifest_path": str(output_root / "manifests" / "dataset.json"),
        "fold_manifest_path": str(output_root / "manifests" / fold_manifest_name),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将多个 prepared_root 合并成统一监督训练集")
    parser.add_argument("--prepared-root", action="append", required=True, help="输入 prepared_root，可重复传入")
    parser.add_argument("--source-alias", action="append", default=[], help="每个 prepared_root 对应的来源别名")
    parser.add_argument("--sample-weight", action="append", default=[], type=float, help="每个来源的样本权重")
    parser.add_argument("--output-root", required=True, help="合并后的 prepared_root 输出目录")
    parser.add_argument("--num-folds", type=int, default=3, help="fold 数量")
    parser.add_argument("--seed", type=int, default=42, help="fold 随机种子")
    parser.add_argument("--no-copy-previews", action="store_true", help="不复制 previews 目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = prepare_merged_supervised_dataset(
        prepared_roots=list(args.prepared_root),
        output_root=args.output_root,
        source_aliases=list(args.source_alias) or None,
        sample_weights=list(args.sample_weight) or None,
        num_folds=int(args.num_folds),
        seed=int(args.seed),
        copy_previews=not bool(args.no_copy_previews),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
