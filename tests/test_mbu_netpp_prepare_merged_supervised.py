from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.mbu_netpp.prepare_merged_supervised import prepare_merged_supervised_dataset


def write_gray(path: Path, value: int) -> None:
    image = np.full((4, 4), fill_value=value, dtype=np.uint8)
    ok = cv2.imwrite(str(path), image)
    assert ok


def build_prepared_root(root: Path, stems: list[str]) -> None:
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "masks").mkdir(parents=True, exist_ok=True)
    (root / "previews").mkdir(parents=True, exist_ok=True)
    (root / "edges" / "k3").mkdir(parents=True, exist_ok=True)
    (root / "edges" / "k5").mkdir(parents=True, exist_ok=True)
    (root / "manifests").mkdir(parents=True, exist_ok=True)

    items = []
    for index, stem in enumerate(stems):
        image_rel = Path("images") / f"{stem}.png"
        mask_rel = Path("masks") / f"{stem}.png"
        preview_rel = Path("previews") / f"{stem}_overlay.png"
        edge_k3_rel = Path("edges") / "k3" / f"{stem}.png"
        edge_k5_rel = Path("edges") / "k5" / f"{stem}.png"

        write_gray(root / image_rel, value=50 + index)
        write_gray(root / mask_rel, value=200)
        write_gray(root / preview_rel, value=100)
        write_gray(root / edge_k3_rel, value=255)
        write_gray(root / edge_k5_rel, value=255)

        items.append(
            {
                "stem": stem,
                "image_path": image_rel.as_posix(),
                "mask_path": mask_rel.as_posix(),
                "preview_path": preview_rel.as_posix(),
                "edge_paths": {
                    "k3": edge_k3_rel.as_posix(),
                    "k5": edge_k5_rel.as_posix(),
                },
                "height": 4,
                "width": 4,
                "foreground_pixels": 16,
            }
        )

    (root / "manifests" / "dataset.json").write_text(
        json.dumps({"num_samples": len(items), "items": items}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def test_prepare_merged_supervised_dataset_copies_files_and_builds_balanced_folds(tmp_path: Path) -> None:
    source_a = tmp_path / "prepared_a"
    source_b = tmp_path / "prepared_b"
    build_prepared_root(source_a, stems=["a1", "a2"])
    build_prepared_root(source_b, stems=["a1", "b2"])

    output_root = tmp_path / "merged"
    result = prepare_merged_supervised_dataset(
        prepared_roots=[source_a, source_b],
        output_root=output_root,
        source_aliases=["realA", "nasaB"],
        sample_weights=[1.0, 1.0],
        num_folds=2,
        seed=42,
    )

    dataset_manifest = json.loads(Path(result["dataset_manifest_path"]).read_text(encoding="utf-8"))
    fold_manifest = json.loads(Path(result["fold_manifest_path"]).read_text(encoding="utf-8"))

    stems = [item["stem"] for item in dataset_manifest["items"]]
    assert stems == ["reala__a1", "reala__a2", "nasab__a1", "nasab__b2"]
    assert len(stems) == len(set(stems))
    assert dataset_manifest["source_counts"] == {"reala": 2, "nasab": 2}

    for item in dataset_manifest["items"]:
        assert (output_root / item["image_path"]).exists()
        assert (output_root / item["mask_path"]).exists()
        assert (output_root / item["preview_path"]).exists()
        assert (output_root / item["edge_paths"]["k3"]).exists()
        assert item["sample_weight"] == 1.0

    assert fold_manifest["strategy"] == "source_balanced_random_split"
    assert len(fold_manifest["folds"]) == 2
    for fold in fold_manifest["folds"]:
        assert len(fold["val_stems"]) == 2
        assert any(stem.startswith("reala__") for stem in fold["val_stems"])
        assert any(stem.startswith("nasab__") for stem in fold["val_stems"])


def test_prepare_merged_supervised_dataset_accepts_windows_style_manifest_paths(tmp_path: Path) -> None:
    source_root = tmp_path / "prepared_windows"
    build_prepared_root(source_root, stems=["w1"])

    manifest_path = source_root / "manifests" / "dataset.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    item = manifest["items"][0]
    item["image_path"] = item["image_path"].replace("/", "\\")
    item["mask_path"] = item["mask_path"].replace("/", "\\")
    item["preview_path"] = item["preview_path"].replace("/", "\\")
    item["edge_paths"] = {key: value.replace("/", "\\") for key, value in item["edge_paths"].items()}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    output_root = tmp_path / "merged"
    result = prepare_merged_supervised_dataset(
        prepared_roots=[source_root],
        output_root=output_root,
        source_aliases=["nasa"],
        num_folds=2,
        seed=42,
    )

    dataset_manifest = json.loads(Path(result["dataset_manifest_path"]).read_text(encoding="utf-8"))
    merged_item = dataset_manifest["items"][0]
    assert (output_root / merged_item["image_path"]).exists()
    assert (output_root / merged_item["mask_path"]).exists()
    assert (output_root / merged_item["preview_path"]).exists()
    assert (output_root / merged_item["edge_paths"]["k3"]).exists()
