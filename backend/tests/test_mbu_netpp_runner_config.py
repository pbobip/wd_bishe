from __future__ import annotations

from pathlib import Path

from backend.app.db.init_db import DEFAULT_RUNNERS
from backend.app.schemas.run import DlModelConfig


def test_mbu_netpp_is_default_dl_model() -> None:
    config = DlModelConfig()

    assert config.model_slot == "mbu_netpp"
    assert config.input_size == 256


def test_mbu_netpp_default_runner_points_to_final_model() -> None:
    runner = next(item for item in DEFAULT_RUNNERS if item["slot"] == "mbu_netpp")

    assert runner["display_name"] == "MBU-Net++ 主模型"
    assert runner["extra_config"]["model_kind"] == "mbu_netpp"
    assert Path(runner["weight_path"]).as_posix().endswith(
        "results/opt_real53_suite/experiments/opt_real53_boundary_sampling/best.pt"
    )
