from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.mbu_netpp.common import save_json

METRIC_KEYS = ["dice", "iou", "precision", "recall", "vf", "vf_pred", "vf_gt", "boundary_f1"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总 NASA 外部验证多 fold 结果")
    parser.add_argument("--eval-root", required=True, help="external_eval 实验目录，例如 outputs/external_eval/e3_micronet_edge_deep_vf_gpu/nasa_super_all")
    parser.add_argument("--output", default=None, help="可选输出 JSON 路径")
    return parser.parse_args()


def mean_metric_dict(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    return {key: float(np.mean([item[key] for item in items])) for key in METRIC_KEYS if key in items[0]}


def load_fold_summary(fold_dir: Path) -> dict[str, Any]:
    summary_path = fold_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"缺少 summary.json: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def load_unique_metrics(fold_dir: Path) -> dict[str, float]:
    csv_path = fold_dir / "per_image_metrics.csv"
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8-sig")))
    unique_rows = {}
    for row in rows:
        unique_rows[row["stem"]] = row
    return {
        key: float(np.mean([float(item[key]) for item in unique_rows.values()]))
        for key in METRIC_KEYS
    }


def main() -> None:
    args = parse_args()
    eval_root = Path(args.eval_root)
    fold_dirs = sorted([item for item in eval_root.iterdir() if item.is_dir() and item.name.startswith("fold_")])
    if not fold_dirs:
        raise FileNotFoundError(f"未在 {eval_root} 找到 fold 目录")

    fold_payloads = [load_fold_summary(fold_dir) for fold_dir in fold_dirs]
    summary = {
        "eval_root": str(eval_root),
        "num_folds": len(fold_payloads),
        "mean_overall": mean_metric_dict([payload["metrics"] for payload in fold_payloads]),
        "mean_test": mean_metric_dict([payload["by_split"]["test"]["metrics"] for payload in fold_payloads if "test" in payload.get("by_split", {})]),
        "mean_different_test": mean_metric_dict(
            [
                payload["by_split"]["different_test"]["metrics"]
                for payload in fold_payloads
                if "different_test" in payload.get("by_split", {})
            ]
        ),
        "mean_super4_test": mean_metric_dict(
            [
                payload["by_subset_split"]["Super4/test"]["metrics"]
                for payload in fold_payloads
                if "Super4/test" in payload.get("by_subset_split", {})
            ]
        ),
        "mean_unique_stems": mean_metric_dict([load_unique_metrics(fold_dir) for fold_dir in fold_dirs]),
    }
    if args.output:
        save_json(args.output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
