from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_experiment(output_root: Path) -> dict[str, Any]:
    experiment_name = output_root.name
    crossval_path = output_root / "crossval_summary.json"
    fold_summaries: list[dict[str, Any]] = []
    for fold_dir in sorted(output_root.glob("fold_*")):
        summary_path = fold_dir / "summary.json"
        history_path = fold_dir / "history.json"
        best_path = fold_dir / "best.pt"
        item: dict[str, Any] = {
            "fold": fold_dir.name,
            "has_best": best_path.exists(),
            "has_history": history_path.exists(),
            "has_summary": summary_path.exists(),
        }
        if summary_path.exists():
            summary = load_json(summary_path)
            best_summary = summary.get("best_summary", {})
            item.update(
                {
                    "dice": best_summary.get("dice"),
                    "vf": best_summary.get("vf"),
                    "boundary_f1": best_summary.get("boundary_f1"),
                }
            )
        fold_summaries.append(item)

    payload: dict[str, Any] = {
        "experiment": experiment_name,
        "has_crossval": crossval_path.exists(),
        "folds": fold_summaries,
    }
    if crossval_path.exists():
        crossval = load_json(crossval_path)
        payload["mean_best_summary"] = crossval.get("mean_best_summary", {})
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="查看 MBU-Net++ 基线实验状态")
    parser.add_argument(
        "--outputs-root",
        default=r"C:\Users\pyd111\Desktop\中期\wd_bishe\experiments\mbu_netpp\outputs",
        help="实验输出根目录",
    )
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root)
    target_names = [
        "e1a_unetpp_noaug_gpu",
        "e1a_unetpp_aug_gpu",
        "e1_unet_gpu",
        "e1_unetpp_gpu",
        "e1_micronet_unetpp_gpu",
        "e2_micronet_edge_gpu",
        "e2_micronet_edge_deep_gpu",
        "e3_micronet_edge_deep_vf_gpu",
    ]
    results = []
    for name in target_names:
        output_root = outputs_root / name
        if output_root.exists():
            results.append(summarize_experiment(output_root))

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
