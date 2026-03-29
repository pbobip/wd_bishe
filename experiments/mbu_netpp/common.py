from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def load_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def save_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(preferred: str) -> torch.device:
    if preferred == "cpu":
        return torch.device("cpu")
    if preferred == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("配置要求使用 CUDA，但当前环境不可用")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def minmax_normalize(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    min_value = float(image.min())
    max_value = float(image.max())
    if max_value <= min_value:
        return np.zeros_like(image, dtype=np.float32)
    return (image - min_value) / (max_value - min_value)


def zscore_normalize(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    mean = float(image.mean())
    std = float(image.std())
    if std < 1e-6:
        return np.zeros_like(image, dtype=np.float32)
    normalized = (image - mean) / std
    normalized = np.clip((normalized + 3.0) / 6.0, 0.0, 1.0)
    return normalized.astype(np.float32)


def image_to_tensor(image: np.ndarray, normalization: str = "minmax") -> torch.Tensor:
    if normalization == "zscore":
        normalized = zscore_normalize(image)
    else:
        normalized = minmax_normalize(image)
    return torch.from_numpy(normalized).unsqueeze(0).float()


def mask_to_tensor(mask: np.ndarray) -> torch.Tensor:
    binary = (mask > 0).astype(np.float32)
    return torch.from_numpy(binary).unsqueeze(0).float()


def build_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        base = image.copy()
    overlay = base.copy()
    overlay[mask > 0] = (48, 132, 103)
    return cv2.addWeighted(base, 0.68, overlay, 0.32, 0)
