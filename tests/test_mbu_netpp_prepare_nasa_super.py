from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.mbu_netpp.preparation import prepare_nasa_super_dataset


def write_gray(path: Path, value: int) -> None:
    image = np.full((4, 4), fill_value=value, dtype=np.uint8)
    ok = cv2.imwrite(str(path), image)
    assert ok


def write_mask(path: Path, color_bgr: tuple[int, int, int]) -> None:
    mask = np.zeros((4, 4, 3), dtype=np.uint8)
    mask[1:3, 1:3] = np.asarray(color_bgr, dtype=np.uint8)
    ok = cv2.imwrite(str(path), mask)
    assert ok


def build_nasa_pair(root: Path, subset: str, split: str, stem: str, color_bgr: tuple[int, int, int]) -> None:
    images_dir = root / subset / split
    masks_dir = root / subset / f"{split}_annot"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    write_gray(images_dir / f"{stem}.png", value=127)
    write_mask(masks_dir / f"{stem}_mask.png", color_bgr=color_bgr)


def test_prepare_nasa_super_dataset_generates_unique_stems_and_folds(tmp_path: Path) -> None:
    dataset_root = tmp_path / "nasa"
    build_nasa_pair(dataset_root, "Super1", "train", "spot1_0d_0", (255, 0, 0))
    build_nasa_pair(dataset_root, "Super2", "train", "spot1_0d_0", (0, 0, 255))

    output_root = tmp_path / "prepared"
    result = prepare_nasa_super_dataset(
        dataset_root=dataset_root,
        output_root=output_root,
        subsets=["Super1", "Super2"],
        splits=["train"],
        foreground_colors=[[255, 0, 0]],
        num_folds=2,
        seed=7,
    )

    dataset_manifest = json.loads(Path(result["dataset_manifest_path"]).read_text(encoding="utf-8"))
    fold_manifest = json.loads(Path(result["fold_manifest_path"]).read_text(encoding="utf-8"))

    stems = [item["stem"] for item in dataset_manifest["items"]]
    assert stems == ["Super1__train__spot1_0d_0", "Super2__train__spot1_0d_0"]
    assert len(stems) == len(set(stems))
    assert [item["source_stem"] for item in dataset_manifest["items"]] == ["spot1_0d_0", "spot1_0d_0"]

    first_mask = cv2.imread(str(output_root / dataset_manifest["items"][0]["mask_path"]), cv2.IMREAD_GRAYSCALE)
    second_mask = cv2.imread(str(output_root / dataset_manifest["items"][1]["mask_path"]), cv2.IMREAD_GRAYSCALE)
    assert first_mask is not None and int(first_mask.sum()) > 0
    assert second_mask is not None and int(second_mask.sum()) == 0

    fold_stems = []
    for fold in fold_manifest["folds"]:
        fold_stems.extend(fold["train_stems"])
        fold_stems.extend(fold["val_stems"])
    assert set(fold_stems) == set(stems)
