from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.mbu_netpp.training_state import collect_crossval_summary, load_resume_state


def test_collect_crossval_summary_reads_existing_fold_summaries(tmp_path: Path) -> None:
    experiment_root = tmp_path / "experiment"
    (experiment_root / "fold_0").mkdir(parents=True, exist_ok=True)
    (experiment_root / "fold_1").mkdir(parents=True, exist_ok=True)

    (experiment_root / "fold_0" / "summary.json").write_text(
        json.dumps(
            {
                "fold_index": 0,
                "best_metric_name": "dice",
                "best_metric": 0.8,
                "best_summary": {"dice": 0.8, "vf": 0.1},
                "checkpoint_path": "fold_0/best.pt",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (experiment_root / "fold_1" / "summary.json").write_text(
        json.dumps(
            {
                "fold_index": 1,
                "best_metric_name": "dice",
                "best_metric": 0.6,
                "best_summary": {"dice": 0.6, "vf": 0.3},
                "checkpoint_path": "fold_1/best.pt",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = collect_crossval_summary(
        experiment_name="opt_suite_demo",
        experiment_root=experiment_root,
        fold_indices=[0, 1],
    )

    assert summary["experiment"] == "opt_suite_demo"
    assert len(summary["fold_results"]) == 2
    assert summary["mean_best_summary"]["dice"] == 0.7
    assert summary["mean_best_summary"]["vf"] == 0.2


def test_load_resume_state_returns_next_epoch_and_history(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "last.pt"
    torch.save(
        {
            "epoch": 3,
            "history": [{"epoch": 1}, {"epoch": 2}, {"epoch": 3}],
            "best_metric": 0.9,
            "best_summary": {"dice": 0.9},
            "optimizer_state_dict": {"state": {}, "param_groups": []},
            "scheduler_state_dict": {"last_epoch": 3},
            "scaler_state_dict": {"scale": 1.0},
        },
        checkpoint_path,
    )

    state = load_resume_state(checkpoint_path)

    assert state["start_epoch"] == 4
    assert len(state["history"]) == 3
    assert state["best_metric"] == 0.9
    assert state["best_summary"] == {"dice": 0.9}
