from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.mbu_netpp.create_holdout_split import create_holdout_split


def build_dataset_manifest(root: Path, stems: list[str]) -> None:
    manifests = root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    payload = {
        "num_samples": len(stems),
        "items": [{"stem": stem, "image_path": f"images/{stem}.png", "mask_path": f"masks/{stem}.png"} for stem in stems],
    }
    (manifests / "dataset.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_create_holdout_split_reserves_preferred_stems_and_builds_trainval_folds(tmp_path: Path) -> None:
    prepared_root = tmp_path / "prepared"
    stems = [f"s{i}" for i in range(1, 13)]
    build_dataset_manifest(prepared_root, stems)

    result = create_holdout_split(
        prepared_root=prepared_root,
        num_test=4,
        num_folds=4,
        seed=7,
        preferred_test_stems=["s2", "s9"],
    )

    holdout = json.loads(Path(result["holdout_manifest_path"]).read_text(encoding="utf-8"))
    folds = json.loads(Path(result["fold_manifest_path"]).read_text(encoding="utf-8"))

    assert holdout["preferred_test_stems"] == ["s2", "s9"]
    assert set(["s2", "s9"]).issubset(set(holdout["test_stems"]))
    assert len(holdout["test_stems"]) == 4
    assert len(holdout["trainval_stems"]) == 8

    assert folds["strategy"] == "random_trainval_with_fixed_holdout"
    assert folds["test_stems"] == holdout["test_stems"]
    assert len(folds["folds"]) == 4
    for fold in folds["folds"]:
        assert fold["test_stems"] == holdout["test_stems"]
        assert len(set(fold["train_stems"]) & set(fold["val_stems"])) == 0
        assert len(set(fold["val_stems"]) & set(holdout["test_stems"])) == 0
