from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.mbu_netpp.filter_inference_outputs import filter_inference_outputs_by_prepared_root


def write_gray(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((4, 4), fill_value=value, dtype=np.uint8)
    ok = cv2.imwrite(str(path), image)
    assert ok


def build_prepared_root(root: Path) -> None:
    (root / "manifests").mkdir(parents=True, exist_ok=True)
    payload = {
        "num_samples": 2,
        "items": [
            {"stem": "real33__01-10", "source_stem": "01-10"},
            {"stem": "real16__13", "source_stem": "13"},
        ],
    }
    (root / "manifests" / "dataset.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_inference_root(root: Path) -> None:
    for stem, value in [("01-10", 50), ("13", 120), ("999", 200)]:
        write_gray(root / "masks" / f"{stem}_mask.png", value=value)
        write_gray(root / "overlays" / f"{stem}_overlay.png", value=value)
        (root / "stats").mkdir(parents=True, exist_ok=True)
        (root / "stats" / f"{stem}.json").write_text(
            json.dumps({"stem": stem, "foreground_pixels": value}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    with (root / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stem", "foreground_pixels"])
        writer.writeheader()
        writer.writerow({"stem": "01-10", "foreground_pixels": 50})
        writer.writerow({"stem": "13", "foreground_pixels": 120})
        writer.writerow({"stem": "999", "foreground_pixels": 200})


def test_filter_inference_outputs_by_prepared_root_keeps_only_non_training_results(tmp_path: Path) -> None:
    prepared_root = tmp_path / "prepared_real49"
    inference_root = tmp_path / "infer_100"
    filtered_root = tmp_path / "infer_excluding_train"

    build_prepared_root(prepared_root)
    build_inference_root(inference_root)

    result = filter_inference_outputs_by_prepared_root(
        prepared_root=prepared_root,
        inference_root=inference_root,
        output_root=filtered_root,
    )

    assert result["total_input_stems"] == 3
    assert result["excluded_training_stems"] == ["01-10", "13"]
    assert result["kept_stems"] == ["999"]

    assert not (filtered_root / "masks" / "01-10_mask.png").exists()
    assert not (filtered_root / "overlays" / "13_overlay.png").exists()
    assert (filtered_root / "masks" / "999_mask.png").exists()
    assert (filtered_root / "overlays" / "999_overlay.png").exists()
    assert (filtered_root / "stats" / "999.json").exists()

    rows = list(csv.DictReader((filtered_root / "summary.csv").open("r", encoding="utf-8-sig")))
    assert [row["stem"] for row in rows] == ["999"]

    manifest = json.loads((filtered_root / "inference_manifest.json").read_text(encoding="utf-8"))
    assert manifest["kept_count"] == 1
    assert manifest["excluded_count"] == 2
