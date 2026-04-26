from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np


def save_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_folds(stems: list[str], num_folds: int, seed: int) -> list[dict[str, Any]]:
    shuffled = list(stems)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    splits = np.array_split(np.asarray(shuffled, dtype=object), num_folds)
    folds: list[dict[str, Any]] = []
    for fold_index, val_split in enumerate(splits):
        val_stems = [str(item) for item in val_split.tolist()]
        val_set = set(val_stems)
        train_stems = [stem for stem in shuffled if stem not in val_set]
        folds.append(
            {
                "fold_index": int(fold_index),
                "train_stems": train_stems,
                "val_stems": val_stems,
            }
        )
    return folds


def create_holdout_split(
    prepared_root: str | Path,
    num_test: int = 10,
    num_folds: int = 5,
    seed: int = 42,
    preferred_test_stems: list[str] | None = None,
    holdout_manifest_name: str | None = None,
    fold_manifest_name: str | None = None,
) -> dict[str, Any]:
    prepared_root = Path(prepared_root)
    dataset_manifest_path = prepared_root / "manifests" / "dataset.json"
    if not dataset_manifest_path.exists():
        raise FileNotFoundError(f"未找到 dataset manifest: {dataset_manifest_path}")

    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    items = list(dataset_manifest.get("items") or [])
    stems = [str(item["stem"]) for item in items]
    if len(stems) < 2:
        raise ValueError("样本数太少，无法划分 holdout")
    if num_test <= 0 or num_test >= len(stems):
        raise ValueError(f"num_test 必须在 1 到 {len(stems) - 1} 之间")

    preferred = []
    existing = set(stems)
    for stem in preferred_test_stems or []:
        if stem in existing and stem not in preferred:
            preferred.append(stem)
    if len(preferred) > num_test:
        raise ValueError("preferred_test_stems 数量超过 num_test")

    remaining = [stem for stem in stems if stem not in set(preferred)]
    rng = random.Random(seed)
    rng.shuffle(remaining)
    sampled = remaining[: max(0, num_test - len(preferred))]
    test_stems = preferred + sampled
    test_set = set(test_stems)
    trainval_stems = [stem for stem in stems if stem not in test_set]

    folds = build_folds(trainval_stems, num_folds=num_folds, seed=seed)
    for fold in folds:
        fold["test_stems"] = list(test_stems)

    holdout_manifest_name = holdout_manifest_name or f"holdout_{num_test}_seed{seed}.json"
    fold_manifest_name = fold_manifest_name or f"folds_{num_folds}_seed{seed}_holdout{num_test}.json"

    holdout_manifest = {
        "num_samples": len(stems),
        "num_test": int(num_test),
        "num_trainval": len(trainval_stems),
        "seed": int(seed),
        "preferred_test_stems": preferred,
        "sampled_additional_test_stems": sampled,
        "test_stems": test_stems,
        "trainval_stems": trainval_stems,
    }
    fold_manifest = {
        "num_folds": int(num_folds),
        "seed": int(seed),
        "strategy": "random_trainval_with_fixed_holdout",
        "holdout_manifest_name": holdout_manifest_name,
        "test_stems": test_stems,
        "folds": folds,
    }

    holdout_manifest_path = prepared_root / "manifests" / holdout_manifest_name
    fold_manifest_path = prepared_root / "manifests" / fold_manifest_name
    save_json(holdout_manifest_path, holdout_manifest)
    save_json(fold_manifest_path, fold_manifest)
    return {
        "prepared_root": str(prepared_root),
        "dataset_manifest_path": str(dataset_manifest_path),
        "holdout_manifest_path": str(holdout_manifest_path),
        "fold_manifest_path": str(fold_manifest_path),
        "num_test": int(num_test),
        "num_trainval": len(trainval_stems),
        "preferred_test_stems": preferred,
        "sampled_additional_test_stems": sampled,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="为 prepared_root 生成固定测试集 + train/val folds")
    parser.add_argument("--prepared-root", required=True, help="prepared_root 路径")
    parser.add_argument("--num-test", type=int, default=10, help="固定测试集样本数")
    parser.add_argument("--num-folds", type=int, default=5, help="train/val 交叉验证 fold 数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--preferred-test-stem",
        action="append",
        default=[],
        help="优先放入测试集的 stem，可重复传入",
    )
    parser.add_argument("--holdout-manifest-name", default="", help="固定测试集 manifest 文件名")
    parser.add_argument("--fold-manifest-name", default="", help="train/val fold manifest 文件名")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = create_holdout_split(
        prepared_root=args.prepared_root,
        num_test=int(args.num_test),
        num_folds=int(args.num_folds),
        seed=int(args.seed),
        preferred_test_stems=list(args.preferred_test_stem),
        holdout_manifest_name=args.holdout_manifest_name or None,
        fold_manifest_name=args.fold_manifest_name or None,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
