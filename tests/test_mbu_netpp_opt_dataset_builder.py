from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.mbu_netpp.prepare_opt_training_sets import prepare_opt_training_dataset


def write_gray(path: Path, value: int) -> None:
    image = np.full((4, 4), fill_value=value, dtype=np.uint8)
    ok = cv2.imwrite(str(path), image)
    assert ok


def build_prepared_root(root: Path, stems: list[str], source_type: str) -> None:
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "masks").mkdir(parents=True, exist_ok=True)
    (root / "edges" / "k3").mkdir(parents=True, exist_ok=True)
    (root / "manifests").mkdir(parents=True, exist_ok=True)

    items = []
    for index, stem in enumerate(stems):
        image_rel = Path("images") / f"{stem}.png"
        mask_rel = Path("masks") / f"{stem}.png"
        edge_rel = Path("edges") / "k3" / f"{stem}.png"
        write_gray(root / image_rel, value=32 + index)
        write_gray(root / mask_rel, value=255)
        write_gray(root / edge_rel, value=255)
        items.append(
            {
                "stem": stem,
                "image_path": image_rel.as_posix(),
                "mask_path": mask_rel.as_posix(),
                "edge_paths": {"k3": edge_rel.as_posix()},
                "height": 4,
                "width": 4,
                "source_type": source_type,
            }
        )

    (root / "manifests" / "dataset.json").write_text(
        json.dumps({"num_samples": len(items), "items": items}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_real_manifests(root: Path) -> None:
    holdout_payload = {
        "num_samples": 6,
        "num_test": 2,
        "num_trainval": 4,
        "seed": 42,
        "preferred_test_stems": ["r5"],
        "sampled_additional_test_stems": ["r6"],
        "test_stems": ["r5", "r6"],
        "trainval_stems": ["r1", "r2", "r3", "r4"],
    }
    folds_payload = {
        "num_folds": 2,
        "seed": 42,
        "strategy": "random_trainval_with_fixed_holdout",
        "holdout_manifest_name": "holdout_2_seed42.json",
        "test_stems": ["r5", "r6"],
        "folds": [
            {
                "fold_index": 0,
                "train_stems": ["r1", "r2"],
                "val_stems": ["r3", "r4"],
                "test_stems": ["r5", "r6"],
            },
            {
                "fold_index": 1,
                "train_stems": ["r3", "r4"],
                "val_stems": ["r1", "r2"],
                "test_stems": ["r5", "r6"],
            },
        ],
    }
    (root / "manifests" / "holdout_2_seed42.json").write_text(
        json.dumps(holdout_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (root / "manifests" / "folds_2_seed42_holdout2.json").write_text(
        json.dumps(folds_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def test_prepare_opt_training_dataset_train_only_adds_aux_only_to_train(tmp_path: Path) -> None:
    real_root = tmp_path / "real"
    nasa_root = tmp_path / "nasa"
    build_prepared_root(real_root, ["r1", "r2", "r3", "r4", "r5", "r6"], source_type="real")
    build_prepared_root(nasa_root, ["n1", "n2"], source_type="nasa_super")
    build_real_manifests(real_root)

    result = prepare_opt_training_dataset(
        real_prepared_root=real_root,
        aux_prepared_root=nasa_root,
        output_root=tmp_path / "train_only",
        mode="train_only_aux",
        num_folds=2,
        seed=42,
        holdout_manifest_name="holdout_2_seed42.json",
        real_fold_manifest_name="folds_2_seed42_holdout2.json",
        aux_sample_weight=0.25,
        aux_sampling_weight=0.15,
    )

    dataset_manifest = json.loads(Path(result["dataset_manifest_path"]).read_text(encoding="utf-8"))
    fold_manifest = json.loads(Path(result["fold_manifest_path"]).read_text(encoding="utf-8"))

    stems = {item["stem"] for item in dataset_manifest["items"]}
    assert {"real__r1", "real__r6", "nasa__n1", "nasa__n2"}.issubset(stems)

    nasa_items = [item for item in dataset_manifest["items"] if item["stem"].startswith("nasa__")]
    assert {item["sample_weight"] for item in nasa_items} == {0.25}
    assert {item["sampling_weight"] for item in nasa_items} == {0.15}

    for fold in fold_manifest["folds"]:
        assert set(fold["test_stems"]) == {"real__r5", "real__r6"}
        assert set(fold["val_stems"]).issubset({"real__r1", "real__r2", "real__r3", "real__r4"})
        assert {"nasa__n1", "nasa__n2"}.issubset(set(fold["train_stems"]))
        assert not any(stem.startswith("nasa__") for stem in fold["val_stems"])


def test_prepare_opt_training_dataset_merged_trainval_rebuilds_folds_but_keeps_real_holdout(tmp_path: Path) -> None:
    real_root = tmp_path / "real"
    nasa_root = tmp_path / "nasa"
    build_prepared_root(real_root, ["r1", "r2", "r3", "r4", "r5", "r6"], source_type="real")
    build_prepared_root(nasa_root, ["n1", "n2"], source_type="nasa_super")
    build_real_manifests(real_root)

    result = prepare_opt_training_dataset(
        real_prepared_root=real_root,
        aux_prepared_root=nasa_root,
        output_root=tmp_path / "merged",
        mode="merged_trainval_with_external_holdout",
        num_folds=2,
        seed=42,
        holdout_manifest_name="holdout_2_seed42.json",
        real_fold_manifest_name="folds_2_seed42_holdout2.json",
        aux_sample_weight=0.5,
        aux_sampling_weight=0.25,
    )

    fold_manifest = json.loads(Path(result["fold_manifest_path"]).read_text(encoding="utf-8"))
    dataset_manifest = json.loads(Path(result["dataset_manifest_path"]).read_text(encoding="utf-8"))

    assert dataset_manifest["num_samples"] == 8
    for fold in fold_manifest["folds"]:
        assert set(fold["test_stems"]) == {"real__r5", "real__r6"}
        assert len(set(fold["train_stems"]) & set(fold["val_stems"])) == 0
        assert len(set(fold["val_stems"]) & {"real__r5", "real__r6"}) == 0
    assert any(any(stem.startswith("nasa__") for stem in fold["val_stems"]) for fold in fold_manifest["folds"])
