from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def decode_gray_bytes(data: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("无法从上传数据解析灰度图像")
    return image


def read_gray(path: str | Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"无法读取图像: {path}")
    return image


def write_image(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or ".png"
    success, encoded = cv2.imencode(suffix, image)
    if not success:
        raise ValueError(f"无法编码图像: {path}")
    encoded.tofile(str(path))


def ensure_odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1
