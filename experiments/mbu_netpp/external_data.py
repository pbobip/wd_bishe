from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image

from backend.app.utils.image_io import read_gray, write_image

from experiments.mbu_netpp.common import build_overlay, ensure_dir, save_json

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
SUPPORTED_MASK_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def detect_crop_height(image: np.ndarray, start_ratio: float = 0.75) -> int:
    height, width = image.shape[:2]
    bottom_h = int(height * start_ratio)
    edges = cv2.Canny(image[bottom_h:, :], 50, 150)
    edge_sum = np.sum(edges, axis=1)
    candidates = np.where(edge_sum > width * 0.5 * 255)[0]
    if len(candidates) > 0:
        return bottom_h + int(candidates[0])
    return int(height * 0.85)


def find_existing_file(directory: str | Path, stem: str, suffixes: set[str], extra_stems: Sequence[str] = ()) -> Path:
    root = Path(directory)
    candidates = [stem, *extra_stems]
    for candidate_stem in candidates:
        for suffix in suffixes:
            candidate = root / f"{candidate_stem}{suffix}"
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"未找到 {stem} 对应文件，搜索目录: {root}")


def find_image_path(images_dir: str | Path, stem: str) -> Path:
    return find_existing_file(images_dir, stem, SUPPORTED_IMAGE_SUFFIXES)


def find_mask_path(masks_dir: str | Path, stem: str) -> Path:
    return find_existing_file(masks_dir, stem, SUPPORTED_MASK_SUFFIXES, extra_stems=(f"{stem}_mask",))


def load_mask_array(mask_path: str | Path) -> np.ndarray:
    return np.array(Image.open(mask_path))


def parse_rgb_spec(raw: str) -> tuple[int, int, int]:
    text = raw.strip()
    if text.startswith("#"):
        value = text[1:]
        if len(value) == 6:
            return tuple(int(value[idx : idx + 2], 16) for idx in (0, 2, 4))
        raise ValueError(f"无效的十六进制颜色: {raw}")
    normalized = text.replace("(", "").replace(")", "")
    parts = [part.strip() for part in normalized.split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError(f"无法解析 RGB 颜色: {raw}")
    rgb = tuple(int(part) for part in parts)
    for channel in rgb:
        if channel < 0 or channel > 255:
            raise ValueError(f"颜色通道越界: {raw}")
    return rgb


def convert_mask_to_binary(
    mask: np.ndarray,
    mode: str = "colored",
    foreground_colors: Sequence[tuple[int, int, int]] | None = None,
) -> np.ndarray:
    normalized_mode = str(mode or "colored").strip().lower()
    mask_array = np.asarray(mask)

    if mask_array.ndim == 2:
        foreground = mask_array > 0
        return foreground.astype(np.uint8) * 255

    if mask_array.ndim != 3:
        raise ValueError(f"不支持的掩膜维度: {mask_array.shape}")

    channels = mask_array.shape[-1]
    if channels not in (3, 4):
        mask_array = mask_array[..., :3]
        channels = mask_array.shape[-1]

    rgb = mask_array[..., :3]
    alpha = mask_array[..., 3] if channels == 4 else np.ones(mask_array.shape[:2], dtype=mask_array.dtype)

    if normalized_mode in {"colored", "non_gray", "non-grey", "color"}:
        foreground = (alpha > 0) & (
            (rgb[..., 0] != rgb[..., 1]) | (rgb[..., 1] != rgb[..., 2]) | (rgb[..., 0] != rgb[..., 2])
        )
    elif normalized_mode in {"nonzero", "positive", "value_nonzero"}:
        foreground = (alpha > 0) & np.any(rgb > 0, axis=-1)
    elif normalized_mode == "alpha_nonzero":
        foreground = alpha > 0
    elif normalized_mode == "exact_color":
        if not foreground_colors:
            raise ValueError("mask_mode=exact_color 时必须提供 foreground_colors")
        foreground = np.zeros(mask_array.shape[:2], dtype=bool)
        for color in foreground_colors:
            foreground |= np.all(rgb[..., :3] == np.asarray(color, dtype=rgb.dtype), axis=-1)
    else:
        raise ValueError(f"不支持的 mask_mode: {mode}")

    return foreground.astype(np.uint8) * 255


def build_edge_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(mask, kernel, iterations=1)
    eroded = cv2.erode(mask, kernel, iterations=1)
    edge = cv2.subtract(dilated, eroded)
    edge[edge > 0] = 255
    return edge


def build_folds(stems: list[str], num_folds: int, seed: int) -> list[dict[str, Any]]:
    shuffled = list(stems)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    splits = np.array_split(np.asarray(shuffled, dtype=object), num_folds)
    folds: list[dict[str, Any]] = []
    for fold_index, val_split in enumerate(splits):
        val_stems = [str(item) for item in val_split.tolist()]
        val_set = set(val_stems)
        train_stems = [stem for stem in shuffled if stem not in val_set]
        folds.append(
            {
                "fold_index": fold_index,
                "train_stems": train_stems,
                "val_stems": val_stems,
            }
        )
    return folds


def collect_paired_records(images_dir: str | Path, masks_dir: str | Path) -> list[dict[str, Any]]:
    image_root = Path(images_dir)
    mask_root = Path(masks_dir)
    image_files = sorted([item for item in image_root.iterdir() if item.is_file() and item.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES])
    if not image_files:
        raise FileNotFoundError(f"未在 {image_root} 找到可用原图")

    records: list[dict[str, Any]] = []
    for image_path in image_files:
        mask_path = find_mask_path(mask_root, image_path.stem)
        records.append(
            {
                "stem": image_path.stem,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
            }
        )
    return records


def prepare_paired_dataset(
    images_dir: str | Path,
    masks_dir: str | Path,
    output_root: str | Path,
    mask_mode: str = "colored",
    foreground_colors: Sequence[tuple[int, int, int]] | None = None,
    auto_crop_sem_region: bool = False,
    crop_detection_ratio: float = 0.75,
    edge_kernels: tuple[int, ...] = (3, 5),
    num_folds: int = 3,
    seed: int = 42,
) -> dict[str, Any]:
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    output_root = ensure_dir(output_root)

    images_out = ensure_dir(output_root / "images")
    masks_out = ensure_dir(output_root / "masks")
    previews_out = ensure_dir(output_root / "previews")
    manifests_out = ensure_dir(output_root / "manifests")
    edge_dirs = {kernel: ensure_dir(output_root / "edges" / f"k{kernel}") for kernel in edge_kernels}

    records = collect_paired_records(images_dir, masks_dir)
    items: list[dict[str, Any]] = []

    for record in records:
        image_path = Path(record["image_path"])
        mask_path = Path(record["mask_path"])
        image = read_gray(image_path)
        raw_mask = load_mask_array(mask_path)
        mask = convert_mask_to_binary(raw_mask, mode=mask_mode, foreground_colors=foreground_colors)

        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(f"图像与掩膜尺寸不一致: {image_path.name} vs {mask_path.name}")

        crop_height = image.shape[0]
        if auto_crop_sem_region:
            crop_height = detect_crop_height(image, start_ratio=crop_detection_ratio)
            image = image[:crop_height, :]
            mask = mask[:crop_height, :]

        image_rel = Path("images") / f"{record['stem']}.png"
        mask_rel = Path("masks") / f"{record['stem']}.png"
        preview_rel = Path("previews") / f"{record['stem']}_overlay.png"

        write_image(output_root / image_rel, image)
        write_image(output_root / mask_rel, mask)
        write_image(output_root / preview_rel, build_overlay(image, mask))

        edge_rel_map: dict[str, str] = {}
        for kernel in edge_kernels:
            edge_mask = build_edge_mask(mask, kernel)
            edge_rel = Path("edges") / f"k{kernel}" / f"{record['stem']}.png"
            write_image(output_root / edge_rel, edge_mask)
            edge_rel_map[f"k{kernel}"] = edge_rel.as_posix()

        items.append(
            {
                "stem": record["stem"],
                "image_path": image_rel.as_posix(),
                "mask_path": mask_rel.as_posix(),
                "edge_paths": edge_rel_map,
                "source_image_path": record["image_path"],
                "source_mask_path": record["mask_path"],
                "mask_mode": mask_mode,
                "height": int(image.shape[0]),
                "width": int(image.shape[1]),
                "foreground_pixels": int(np.count_nonzero(mask)),
                "crop_height": int(crop_height),
            }
        )

    stems = [item["stem"] for item in items]
    folds = build_folds(stems, num_folds=num_folds, seed=seed)

    dataset_manifest = {
        "num_samples": len(items),
        "mask_mode": mask_mode,
        "foreground_colors": [list(color) for color in foreground_colors] if foreground_colors else [],
        "auto_crop_sem_region": auto_crop_sem_region,
        "crop_detection_ratio": crop_detection_ratio,
        "edge_kernels": list(edge_kernels),
        "items": items,
    }
    folds_manifest = {
        "num_folds": num_folds,
        "seed": seed,
        "folds": folds,
    }

    save_json(manifests_out / "dataset.json", dataset_manifest)
    save_json(manifests_out / f"folds_{num_folds}_seed{seed}.json", folds_manifest)
    return {
        "dataset_manifest_path": str(manifests_out / "dataset.json"),
        "fold_manifest_path": str(manifests_out / f"folds_{num_folds}_seed{seed}.json"),
        "num_samples": len(items),
        "output_root": str(output_root),
    }
