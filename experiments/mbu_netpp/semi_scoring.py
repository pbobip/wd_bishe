from __future__ import annotations

import math
from typing import Any

import numpy as np


def compute_prediction_scores(prob_map: np.ndarray, binary_threshold: float) -> dict[str, float]:
    eps = 1e-6
    clipped = np.clip(prob_map.astype(np.float32), eps, 1.0 - eps)
    binary_mask = clipped >= float(binary_threshold)

    entropy = -(
        clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped)
    ) / math.log(2.0)
    confidence = np.maximum(clipped, 1.0 - clipped)

    foreground_probs = clipped[binary_mask]
    background_probs = clipped[~binary_mask]
    return {
        "confidence_score": float(np.mean(confidence)),
        "uncertainty_score": float(np.mean(entropy)),
        "mean_foreground_prob": float(np.mean(foreground_probs)) if foreground_probs.size > 0 else 0.0,
        "mean_background_prob": float(np.mean(background_probs)) if background_probs.size > 0 else 0.0,
        "mask_probability_mean": float(np.mean(clipped)),
        "foreground_ratio": float(np.mean(binary_mask.astype(np.float32))),
    }


def _sort_candidates(
    items: list[dict[str, Any]],
    score_name: str,
    descending: bool,
) -> list[dict[str, Any]]:
    return sorted(
        items,
        key=lambda item: (float(item.get(score_name, 0.0)), item.get("stem", "")),
        reverse=descending,
    )


def select_pseudo_candidates(
    predictions: list[dict[str, Any]],
    confidence_threshold: float,
    max_items: int | None = None,
    score_name: str = "confidence_score",
) -> list[dict[str, Any]]:
    candidates = [
        item
        for item in predictions
        if float(item.get(score_name, 0.0)) >= float(confidence_threshold)
    ]
    ranked = _sort_candidates(candidates, score_name=score_name, descending=True)
    if max_items is not None and max_items >= 0:
        ranked = ranked[: int(max_items)]
    return ranked


def select_active_learning_candidates(
    predictions: list[dict[str, Any]],
    top_k: int,
    score_name: str = "uncertainty_score",
    excluded_stems: set[str] | None = None,
) -> list[dict[str, Any]]:
    if top_k <= 0:
        return []
    excluded = excluded_stems or set()
    candidates = [item for item in predictions if str(item.get("stem", "")) not in excluded]
    ranked = _sort_candidates(candidates, score_name=score_name, descending=True)
    return ranked[: int(top_k)]
