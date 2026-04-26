from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.mbu_netpp.sampling import compute_record_sampling_weights


def test_compute_record_sampling_weights_respects_record_and_hard_stem_boost() -> None:
    records = [
        {"stem": "real__easy", "source_stem": "easy", "sampling_weight": 1.0},
        {"stem": "real__hard", "source_stem": "hard", "sampling_weight": 0.5},
        {"stem": "nasa__sample", "source_stem": "sample", "sampling_weight": 0.25},
    ]
    sampling_config = {
        "hard_stems": ["hard", "nasa__sample"],
        "hard_stem_weight": 4.0,
    }

    weights = compute_record_sampling_weights(records, sampling_config)

    assert weights == [1.0, 2.0, 1.0]


def test_compute_record_sampling_weights_defaults_to_one_when_missing() -> None:
    records = [
        {"stem": "a"},
        {"stem": "b", "source_stem": "b"},
    ]

    weights = compute_record_sampling_weights(records, {})

    assert weights == [1.0, 1.0]
