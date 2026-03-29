from __future__ import annotations

import numpy as np

from backend.app.schemas.run import PreprocessConfig
from backend.app.services.preprocess import preprocess_service


def build_preprocess_config(raw_config: dict | None) -> PreprocessConfig:
    payload = raw_config or {"enabled": False}
    return PreprocessConfig.model_validate(payload)


def apply_preprocess(image: np.ndarray, raw_config: dict | None) -> np.ndarray:
    config = build_preprocess_config(raw_config)
    if not config.enabled:
        return image.copy()
    return preprocess_service.apply(image, config)
