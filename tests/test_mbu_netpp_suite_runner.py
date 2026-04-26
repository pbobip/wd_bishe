from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.mbu_netpp.run_opt_training_suite import build_experiment_tasks


def test_build_experiment_tasks_creates_fold_and_finalize_steps() -> None:
    tasks = build_experiment_tasks(experiment_name="opt_real53_baseline", num_folds=5)

    assert len(tasks) == 8
    assert [task["kind"] for task in tasks[:5]] == ["train_fold"] * 5
    assert [task["fold_index"] for task in tasks[:5]] == [0, 1, 2, 3, 4]
    assert tasks[5]["kind"] == "summarize"
    assert tasks[6]["kind"] == "plot"
    assert tasks[7]["kind"] == "holdout"
