from __future__ import annotations

from typing import Any


def compute_record_sampling_weights(
    records: list[dict[str, Any]],
    sampling_config: dict[str, Any] | None,
) -> list[float]:
    cfg = sampling_config or {}
    hard_stems = {str(value) for value in cfg.get("hard_stems", [])}
    hard_stem_weight = float(cfg.get("hard_stem_weight", 1.0))

    weights: list[float] = []
    for record in records:
        weight = float(record.get("sampling_weight", 1.0))
        stem = str(record.get("stem", ""))
        source_stem = str(record.get("source_stem", stem))
        if stem in hard_stems or source_stem in hard_stems:
            weight *= hard_stem_weight
        weights.append(max(weight, 1e-6))
    return weights
