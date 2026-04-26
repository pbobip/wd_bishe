from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from experiments.mbu_netpp.common import ensure_dir, save_json
from experiments.mbu_netpp.infer import maybe_write_xlsx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总一组实验目录的 crossval 与 holdout 结果")
    parser.add_argument("--experiments-root", required=True, help="实验输出根目录，子目录为各实验")
    parser.add_argument("--output-dir", required=True, help="汇总输出目录")
    return parser.parse_args()


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    experiments_root = Path(args.experiments_root)
    output_dir = ensure_dir(args.output_dir)

    rows: list[dict[str, Any]] = []
    for experiment_dir in sorted([item for item in experiments_root.iterdir() if item.is_dir()]):
        crossval = load_json_if_exists(experiment_dir / "crossval_summary.json")
        holdout = load_json_if_exists(experiment_dir / "holdout_eval" / "summary.json")
        if not crossval and not holdout:
            continue

        crossval_metrics = (crossval or {}).get("mean_best_summary", {})
        holdout_metrics = (holdout or {}).get("metrics", {})
        rows.append(
            {
                "experiment": experiment_dir.name,
                "crossval_dice": float(crossval_metrics.get("dice", 0.0)),
                "crossval_iou": float(crossval_metrics.get("iou", 0.0)),
                "crossval_precision": float(crossval_metrics.get("precision", 0.0)),
                "crossval_recall": float(crossval_metrics.get("recall", 0.0)),
                "crossval_vf": float(crossval_metrics.get("vf", 0.0)),
                "crossval_boundary_f1": float(crossval_metrics.get("boundary_f1", 0.0)),
                "holdout_dice": float(holdout_metrics.get("dice", 0.0)),
                "holdout_iou": float(holdout_metrics.get("iou", 0.0)),
                "holdout_precision": float(holdout_metrics.get("precision", 0.0)),
                "holdout_recall": float(holdout_metrics.get("recall", 0.0)),
                "holdout_vf": float(holdout_metrics.get("vf", 0.0)),
                "holdout_boundary_f1": float(holdout_metrics.get("boundary_f1", 0.0)),
                "crossval_summary_path": str(experiment_dir / "crossval_summary.json"),
                "holdout_summary_path": str(experiment_dir / "holdout_eval" / "summary.json"),
            }
        )

    csv_path = output_dir / "suite_summary.csv"
    if rows:
        headers = list(rows[0].keys())
        with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
        maybe_write_xlsx(rows, output_dir / "suite_summary.xlsx")

    payload = {
        "experiments_root": str(experiments_root),
        "num_experiments": len(rows),
        "rows": rows,
        "suite_summary_csv": str(csv_path),
    }
    save_json(output_dir / "suite_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
