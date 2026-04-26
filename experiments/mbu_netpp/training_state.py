from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from experiments.mbu_netpp.common import save_json
from experiments.mbu_netpp.metrics import average_metric_dicts


def load_resume_state(checkpoint_path: str | Path) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    return {
        "epoch": int(payload.get("epoch", 0)),
        "start_epoch": int(payload.get("epoch", 0)) + 1,
        "history": list(payload.get("history") or []),
        "best_metric": payload.get("best_metric"),
        "best_summary": dict(payload.get("best_summary") or {}),
        "model_state_dict": payload.get("model_state_dict"),
        "optimizer_state_dict": payload.get("optimizer_state_dict"),
        "scheduler_state_dict": payload.get("scheduler_state_dict"),
        "scaler_state_dict": payload.get("scaler_state_dict"),
    }


def collect_crossval_summary(
    experiment_name: str,
    experiment_root: str | Path,
    fold_indices: list[int],
) -> dict[str, Any]:
    root = Path(experiment_root)
    fold_results: list[dict[str, Any]] = []
    for fold_index in fold_indices:
        summary_path = root / f"fold_{fold_index}" / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"缺少 fold summary: {summary_path}")
        fold_results.append(json.loads(summary_path.read_text(encoding="utf-8")))
    summary = {
        "experiment": experiment_name,
        "fold_results": fold_results,
        "mean_best_summary": average_metric_dicts(
            [item["best_summary"] for item in fold_results if item.get("best_summary")]
        ),
    }
    save_json(root / "crossval_summary.json", summary)
    return summary
