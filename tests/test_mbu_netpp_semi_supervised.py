from __future__ import annotations

import torch

from experiments.mbu_netpp.losses import MBULoss
from experiments.mbu_netpp.semi_scoring import (
    select_active_learning_candidates,
    select_pseudo_candidates,
)


def build_batch(sample_weights: list[float]) -> dict[str, torch.Tensor]:
    batch_size = len(sample_weights)
    return {
        "mask": torch.ones((batch_size, 1, 2, 2), dtype=torch.float32),
        "edge": torch.zeros((batch_size, 1, 2, 2), dtype=torch.float32),
        "sample_weight": torch.tensor(sample_weights, dtype=torch.float32),
    }


def build_outputs(logit_values: list[float]) -> dict[str, torch.Tensor | list[torch.Tensor] | None]:
    logits = torch.stack(
        [
            torch.full((1, 2, 2), fill_value=float(value), dtype=torch.float32)
            for value in logit_values
        ],
        dim=0,
    )
    return {
        "seg_logits": logits,
        "edge_logits": None,
        "deep_logits": [],
    }


def test_mbu_loss_respects_sample_weight() -> None:
    criterion = MBULoss(
        {
            "use_edge_loss": False,
            "use_vf_loss": False,
        }
    )
    outputs = build_outputs(logit_values=[2.0, -2.0])

    full_weight_loss, _ = criterion(outputs, build_batch([1.0, 1.0]))
    down_weighted_loss, _ = criterion(outputs, build_batch([1.0, 0.2]))

    assert down_weighted_loss.item() < full_weight_loss.item()


def test_select_pseudo_candidates_filters_and_sorts_by_confidence() -> None:
    predictions = [
        {"stem": "a", "confidence_score": 0.91, "uncertainty_score": 0.10},
        {"stem": "b", "confidence_score": 0.76, "uncertainty_score": 0.30},
        {"stem": "c", "confidence_score": 0.83, "uncertainty_score": 0.22},
    ]

    selected = select_pseudo_candidates(predictions, confidence_threshold=0.8, max_items=2)

    assert [item["stem"] for item in selected] == ["a", "c"]


def test_select_active_learning_candidates_prefers_high_uncertainty() -> None:
    predictions = [
        {"stem": "a", "confidence_score": 0.91, "uncertainty_score": 0.10},
        {"stem": "b", "confidence_score": 0.76, "uncertainty_score": 0.30},
        {"stem": "c", "confidence_score": 0.83, "uncertainty_score": 0.22},
        {"stem": "d", "confidence_score": 0.55, "uncertainty_score": 0.44},
    ]

    selected = select_active_learning_candidates(predictions, top_k=2)

    assert [item["stem"] for item in selected] == ["d", "b"]
def test_active_learning_selection_handles_empty_inputs() -> None:
    assert select_active_learning_candidates([], top_k=3) == []


def test_active_learning_selection_handles_small_inputs() -> None:
    predictions = [{"stem": "x", "confidence_score": 0.3, "uncertainty_score": 0.5}]
    selected = select_active_learning_candidates(predictions, top_k=3)
    assert [item["stem"] for item in selected] == ["x"]
