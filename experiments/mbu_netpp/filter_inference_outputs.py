from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def save_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_dataset_manifest(prepared_root: str | Path) -> dict[str, Any]:
    manifest_path = Path(prepared_root) / "manifests" / "dataset.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"未找到训练集 manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def collect_training_source_stems(prepared_root: str | Path) -> list[str]:
    dataset_manifest = load_dataset_manifest(prepared_root)
    stems = {
        str(item.get("source_stem") or item.get("stem", "")).strip()
        for item in list(dataset_manifest.get("items") or [])
    }
    stems.discard("")
    return sorted(stems)


def read_summary_rows(inference_root: str | Path) -> list[dict[str, Any]]:
    summary_path = Path(inference_root) / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"未找到推理汇总文件: {summary_path}")
    with summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def maybe_write_xlsx(rows: list[dict[str, Any]], xlsx_path: Path) -> bool:
    try:
        from openpyxl import Workbook
    except Exception:
        return False

    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "summary"
    if rows:
        headers = list(rows[0].keys())
        worksheet.append(headers)
        for row in rows:
            worksheet.append([row.get(header) for header in headers])
    workbook.save(xlsx_path)
    return True


def copy_if_exists(source: Path, target: Path) -> bool:
    if not source.exists():
        return False
    ensure_dir(target.parent)
    shutil.copy2(source, target)
    return True


def filter_inference_outputs(
    inference_root: str | Path,
    output_root: str | Path,
    excluded_stems: list[str] | set[str],
) -> dict[str, Any]:
    inference_root = Path(inference_root)
    output_root = ensure_dir(output_root)
    excluded = {str(stem).strip() for stem in excluded_stems if str(stem).strip()}

    all_rows = read_summary_rows(inference_root)
    kept_rows = [row for row in all_rows if str(row.get("stem", "")).strip() not in excluded]
    kept_stems = [str(row.get("stem", "")).strip() for row in kept_rows if str(row.get("stem", "")).strip()]

    ensure_dir(output_root / "masks")
    ensure_dir(output_root / "overlays")
    ensure_dir(output_root / "stats")

    for stem in kept_stems:
        copy_if_exists(
            inference_root / "masks" / f"{stem}_mask.png",
            output_root / "masks" / f"{stem}_mask.png",
        )
        copy_if_exists(
            inference_root / "overlays" / f"{stem}_overlay.png",
            output_root / "overlays" / f"{stem}_overlay.png",
        )
        copy_if_exists(
            inference_root / "stats" / f"{stem}.json",
            output_root / "stats" / f"{stem}.json",
        )

    summary_csv_path = output_root / "summary.csv"
    headers = list(all_rows[0].keys()) if all_rows else ["stem"]
    with summary_csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in kept_rows:
            writer.writerow({key: row.get(key) for key in headers})

    maybe_write_xlsx(kept_rows, output_root / "summary.xlsx")

    manifest = {
        "source_inference_root": str(inference_root.resolve()),
        "output_root": str(output_root.resolve()),
        "total_input_stems": len(all_rows),
        "excluded_training_stems": sorted(excluded),
        "excluded_count": len([row for row in all_rows if str(row.get("stem", "")).strip() in excluded]),
        "kept_stems": kept_stems,
        "kept_count": len(kept_stems),
    }
    save_json(output_root / "inference_manifest.json", manifest)
    return manifest


def filter_inference_outputs_by_prepared_root(
    prepared_root: str | Path,
    inference_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    return filter_inference_outputs(
        inference_root=inference_root,
        output_root=output_root,
        excluded_stems=collect_training_source_stems(prepared_root),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按训练集 stem 剔除推理结果")
    parser.add_argument("--prepared-root", required=True, help="训练集 prepared_root")
    parser.add_argument("--inference-root", required=True, help="infer.py 输出目录")
    parser.add_argument("--output-root", required=True, help="过滤后的输出目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = filter_inference_outputs_by_prepared_root(
        prepared_root=args.prepared_root,
        inference_root=args.inference_root,
        output_root=args.output_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
