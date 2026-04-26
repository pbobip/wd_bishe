from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from backend.app.utils.image_io import read_gray, write_image

from experiments.mbu_netpp.common import build_overlay, ensure_dir, save_json
from experiments.mbu_netpp.external_data import parse_rgb_spec, prepare_paired_dataset as prepare_paired_dataset_external

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def detect_crop_height(image: np.ndarray, start_ratio: float = 0.75) -> int:
    height, width = image.shape[:2]
    bottom_h = int(height * start_ratio)
    edges = cv2.Canny(image[bottom_h:, :], 50, 150)
    edge_sum = np.sum(edges, axis=1)
    candidates = np.where(edge_sum > width * 0.5 * 255)[0]
    if len(candidates) > 0:
        return bottom_h + int(candidates[0])
    return int(height * 0.85)


def find_image_path(images_dir: Path, stem: str) -> Path:
    for suffix in SUPPORTED_IMAGE_SUFFIXES:
        candidate = images_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"未找到 {stem} 对应原图，搜索目录: {images_dir}")


def find_mask_path(masks_dir: Path, stem: str, suffix_candidates: list[str] | None = None) -> Path:
    candidates = suffix_candidates or ["_mask", ""]
    for suffix in candidates:
        for image_suffix in SUPPORTED_IMAGE_SUFFIXES:
            candidate = masks_dir / f"{stem}{suffix}{image_suffix}"
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"未找到 {stem} 对应掩膜，搜索目录: {masks_dir}")


def load_shapes(json_path: Path) -> list[dict[str, Any]]:
    return json.loads(json_path.read_text(encoding="utf-8")).get("shapes", [])


def build_binary_mask(image_shape: tuple[int, int], shapes: list[dict[str, Any]], positive_labels: set[str]) -> np.ndarray:
    mask = np.zeros(image_shape, dtype=np.uint8)
    for shape in shapes:
        label = str(shape.get("label", "")).strip().lower()
        if label not in positive_labels:
            continue
        shape_type = shape.get("shape_type")
        if shape_type not in (None, "polygon"):
            continue
        points = np.asarray(shape.get("points") or [], dtype=np.float32)
        if len(points) < 3:
            continue
        polygon = np.round(points).astype(np.int32)
        polygon[:, 0] = np.clip(polygon[:, 0], 0, image_shape[1] - 1)
        polygon[:, 1] = np.clip(polygon[:, 1], 0, image_shape[0] - 1)
        cv2.fillPoly(mask, [polygon], 255)
    return mask


def build_edge_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(mask, kernel, iterations=1)
    eroded = cv2.erode(mask, kernel, iterations=1)
    edge = cv2.subtract(dilated, eroded)
    edge[edge > 0] = 255
    return edge


def parse_color_tuples(raw_colors: list[list[int]] | None) -> list[tuple[int, int, int]]:
    parsed: list[tuple[int, int, int]] = []
    for color in raw_colors or []:
        if len(color) < 3:
            continue
        parsed.append((int(color[0]), int(color[1]), int(color[2])))
    return parsed


def build_binary_mask_from_paired_mask(
    mask_image: np.ndarray,
    mode: str = "non_gray",
    foreground_colors: list[tuple[int, int, int]] | None = None,
) -> np.ndarray:
    if mask_image.ndim == 2:
        return ((mask_image > 127).astype(np.uint8) * 255)

    color = mask_image[:, :, :3]
    if foreground_colors:
        foreground = np.zeros(color.shape[:2], dtype=bool)
        for bgr in foreground_colors:
            foreground |= np.all(color == np.asarray(bgr, dtype=np.uint8), axis=2)
    elif mode == "non_black":
        foreground = np.any(color > 0, axis=2)
    else:
        # NASA 的彩色叠加掩膜会保留灰度背景，所有非灰度像素视为标注区域。
        foreground = np.logical_or(color[:, :, 0] != color[:, :, 1], color[:, :, 1] != color[:, :, 2])
    return foreground.astype(np.uint8) * 255


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


def prepare_supervised_dataset(
    images_dir: str | Path,
    annotations_dir: str | Path,
    output_root: str | Path,
    positive_labels: list[str] | None = None,
    auto_crop_sem_region: bool = True,
    crop_detection_ratio: float = 0.75,
    edge_kernels: tuple[int, ...] = (3, 5),
    num_folds: int = 3,
    seed: int = 42,
) -> dict[str, Any]:
    images_dir = Path(images_dir)
    annotations_dir = Path(annotations_dir)
    output_root = ensure_dir(output_root)

    images_out = ensure_dir(output_root / "images")
    masks_out = ensure_dir(output_root / "masks")
    previews_out = ensure_dir(output_root / "previews")
    manifests_out = ensure_dir(output_root / "manifests")
    edge_dirs = {kernel: ensure_dir(output_root / "edges" / f"k{kernel}") for kernel in edge_kernels}

    positive = {label.strip().lower() for label in (positive_labels or ["gamma_prime"])}
    annotation_files = sorted(annotations_dir.glob("*.json"))
    if not annotation_files:
        raise FileNotFoundError(f"未在 {annotations_dir} 找到 JSON 标注文件")

    items: list[dict[str, Any]] = []
    for json_path in annotation_files:
        stem = json_path.stem
        image_path = find_image_path(images_dir, stem)
        image = read_gray(image_path)
        mask = build_binary_mask(image.shape[:2], load_shapes(json_path), positive)

        crop_height = image.shape[0]
        if auto_crop_sem_region:
            crop_height = detect_crop_height(image, start_ratio=crop_detection_ratio)
            image = image[:crop_height, :]
            mask = mask[:crop_height, :]

        image_rel = Path("images") / f"{stem}.png"
        mask_rel = Path("masks") / f"{stem}.png"
        preview_rel = Path("previews") / f"{stem}_overlay.png"

        write_image(output_root / image_rel, image)
        write_image(output_root / mask_rel, mask)
        write_image(output_root / preview_rel, build_overlay(image, mask))

        edge_rel_map: dict[str, str] = {}
        for kernel in edge_kernels:
            edge_mask = build_edge_mask(mask, kernel)
            edge_rel = Path("edges") / f"k{kernel}" / f"{stem}.png"
            write_image(output_root / edge_rel, edge_mask)
            edge_rel_map[f"k{kernel}"] = edge_rel.as_posix()

        items.append(
            {
                "stem": stem,
                "image_path": image_rel.as_posix(),
                "mask_path": mask_rel.as_posix(),
                "edge_paths": edge_rel_map,
                "source_image_path": str(image_path),
                "source_annotation_path": str(json_path),
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
        "positive_labels": sorted(positive),
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


def prepare_paired_dataset(
    images_dir: str | Path,
    masks_dir: str | Path,
    output_root: str | Path,
    pair_suffix_candidates: list[str] | None = None,
    auto_crop_sem_region: bool = False,
    crop_detection_ratio: float = 0.75,
    edge_kernels: tuple[int, ...] = (3, 5),
    mask_mode: str = "non_gray",
    foreground_colors: list[list[int]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    output_root = ensure_dir(output_root)

    images_out = ensure_dir(output_root / "images")
    masks_out = ensure_dir(output_root / "masks")
    previews_out = ensure_dir(output_root / "previews")
    manifests_out = ensure_dir(output_root / "manifests")
    edge_dirs = {kernel: ensure_dir(output_root / "edges" / f"k{kernel}") for kernel in edge_kernels}

    image_files = sorted([item for item in images_dir.iterdir() if item.is_file() and item.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES])
    if not image_files:
        raise FileNotFoundError(f"未在 {images_dir} 找到图像文件")

    parsed_colors = parse_color_tuples(foreground_colors)
    items: list[dict[str, Any]] = []
    for image_path in image_files:
        stem = image_path.stem
        mask_path = find_mask_path(masks_dir, stem, suffix_candidates=pair_suffix_candidates)
        image = read_gray(image_path)
        mask_image = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask_image is None:
            raise FileNotFoundError(f"无法读取掩膜文件: {mask_path}")
        mask = build_binary_mask_from_paired_mask(
            mask_image=mask_image,
            mode=mask_mode,
            foreground_colors=parsed_colors,
        )

        crop_height = image.shape[0]
        if auto_crop_sem_region:
            crop_height = detect_crop_height(image, start_ratio=crop_detection_ratio)
            image = image[:crop_height, :]
            mask = mask[:crop_height, :]

        image_rel = Path("images") / f"{stem}.png"
        mask_rel = Path("masks") / f"{stem}.png"
        preview_rel = Path("previews") / f"{stem}_overlay.png"
        write_image(output_root / image_rel, image)
        write_image(output_root / mask_rel, mask)
        write_image(output_root / preview_rel, build_overlay(image, mask))

        edge_rel_map: dict[str, str] = {}
        for kernel in edge_kernels:
            edge_mask = build_edge_mask(mask, kernel)
            edge_rel = Path("edges") / f"k{kernel}" / f"{stem}.png"
            write_image(output_root / edge_rel, edge_mask)
            edge_rel_map[f"k{kernel}"] = edge_rel.as_posix()

        item = {
            "stem": stem,
            "image_path": image_rel.as_posix(),
            "mask_path": mask_rel.as_posix(),
            "edge_paths": edge_rel_map,
            "source_image_path": str(image_path),
            "source_mask_path": str(mask_path),
            "height": int(image.shape[0]),
            "width": int(image.shape[1]),
            "foreground_pixels": int(np.count_nonzero(mask)),
            "crop_height": int(crop_height),
            "mask_mode": mask_mode,
            "foreground_colors_bgr": [list(color) for color in parsed_colors],
        }
        if metadata:
            item.update(metadata)
        items.append(item)

    dataset_manifest = {
        "num_samples": len(items),
        "source_type": "paired_images_masks",
        "pair_suffix_candidates": list(pair_suffix_candidates or ["_mask", ""]),
        "auto_crop_sem_region": auto_crop_sem_region,
        "crop_detection_ratio": crop_detection_ratio,
        "edge_kernels": list(edge_kernels),
        "mask_mode": mask_mode,
        "foreground_colors_bgr": [list(color) for color in parsed_colors],
        "items": items,
    }
    save_json(manifests_out / "dataset.json", dataset_manifest)
    return {
        "dataset_manifest_path": str(manifests_out / "dataset.json"),
        "num_samples": len(items),
        "output_root": str(output_root),
    }


def prepare_nasa_super_dataset(
    dataset_root: str | Path,
    output_root: str | Path,
    subsets: list[str] | None = None,
    splits: list[str] | None = None,
    include_different_test: bool = False,
    auto_crop_sem_region: bool = False,
    crop_detection_ratio: float = 0.75,
    edge_kernels: tuple[int, ...] = (3, 5),
    foreground_colors: list[list[int]] | None = None,
    num_folds: int = 3,
    seed: int = 42,
) -> dict[str, Any]:
    dataset_root = Path(dataset_root)
    output_root = ensure_dir(output_root)
    manifests_out = ensure_dir(output_root / "manifests")

    target_subsets = subsets or ["Super1", "Super2", "Super3", "Super4"]
    target_splits = splits or ["train", "val", "test"]
    if include_different_test and "different_test" not in target_splits:
        target_splits = [*target_splits, "different_test"]

    items: list[dict[str, Any]] = []
    group_counts: dict[str, int] = {}
    seen_stems: set[str] = set()
    for subset in target_subsets:
        for split in target_splits:
            images_dir = dataset_root / subset / split
            masks_dir = dataset_root / subset / f"{split}_annot"
            if not images_dir.exists() or not masks_dir.exists():
                continue

            paired_output = output_root / subset / split
            result = prepare_paired_dataset(
                images_dir=images_dir,
                masks_dir=masks_dir,
                output_root=paired_output,
                pair_suffix_candidates=["_mask"],
                auto_crop_sem_region=auto_crop_sem_region,
                crop_detection_ratio=crop_detection_ratio,
                edge_kernels=edge_kernels,
                mask_mode="non_gray" if not foreground_colors else "colors",
                foreground_colors=foreground_colors or [[255, 0, 0], [0, 0, 255]],
                metadata={"subset": subset, "split": split},
            )
            dataset_manifest = json.loads((Path(result["dataset_manifest_path"])).read_text(encoding="utf-8"))
            for item in dataset_manifest["items"]:
                source_stem = str(item["stem"])
                unique_stem = f"{subset}__{split}__{source_stem}"
                if unique_stem in seen_stems:
                    raise ValueError(f"NASA Super 数据存在重复样本键: {unique_stem}")
                seen_stems.add(unique_stem)
                item["source_stem"] = source_stem
                item["stem"] = unique_stem
                item["prepared_group_root"] = str(paired_output)
                # 统一成相对总输出目录的路径，便于外部评估脚本读取
                item["image_path"] = str(Path(subset) / split / item["image_path"])
                item["mask_path"] = str(Path(subset) / split / item["mask_path"])
                item["edge_paths"] = {
                    key: str(Path(subset) / split / value)
                    for key, value in item["edge_paths"].items()
                }
                items.append(item)
            group_counts[f"{subset}/{split}"] = int(result["num_samples"])

    if not items:
        raise FileNotFoundError(f"未在 {dataset_root} 找到可用的 NASA Super 数据")

    folds = build_folds([item["stem"] for item in items], num_folds=num_folds, seed=seed)
    dataset_manifest = {
        "num_samples": len(items),
        "source_type": "nasa_super",
        "subsets": target_subsets,
        "splits": target_splits,
        "include_different_test": include_different_test,
        "auto_crop_sem_region": auto_crop_sem_region,
        "crop_detection_ratio": crop_detection_ratio,
        "edge_kernels": list(edge_kernels),
        "mask_mode": "colors",
        "foreground_colors_bgr": foreground_colors or [[255, 0, 0], [0, 0, 255]],
        "num_folds": num_folds,
        "fold_seed": seed,
        "group_counts": group_counts,
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
        "group_counts": group_counts,
        "output_root": str(output_root),
    }


def prepare_external_paired_dataset(
    images_dir: str | Path,
    masks_dir: str | Path,
    output_root: str | Path,
    mask_mode: str = "colored",
    foreground_colors: list[tuple[int, int, int]] | None = None,
    auto_crop_sem_region: bool = False,
    crop_detection_ratio: float = 0.75,
    edge_kernels: tuple[int, ...] = (3, 5),
    num_folds: int = 3,
    seed: int = 42,
) -> dict[str, Any]:
    return prepare_paired_dataset_external(
        images_dir=images_dir,
        masks_dir=masks_dir,
        output_root=output_root,
        mask_mode=mask_mode,
        foreground_colors=foreground_colors,
        auto_crop_sem_region=auto_crop_sem_region,
        crop_detection_ratio=crop_detection_ratio,
        edge_kernels=edge_kernels,
        num_folds=num_folds,
        seed=seed,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="准备 MBU-Net++ 监督学习数据")
    parser.add_argument("--images-dir", required=True, help="原图目录")
    parser.add_argument("--annotations-dir", default=None, help="LabelMe JSON 目录")
    parser.add_argument("--masks-dir", default=None, help="成对掩膜目录")
    parser.add_argument("--output-root", required=True, help="准备后数据输出目录")
    parser.add_argument("--labels", nargs="+", default=["gamma_prime"], help="前景标签名称")
    parser.add_argument("--no-auto-crop", action="store_true", help="关闭 SEM 底栏自动裁切")
    parser.add_argument("--crop-detection-ratio", type=float, default=0.75, help="底栏检测起始比例")
    parser.add_argument("--edge-kernels", nargs="+", type=int, default=[3, 5], help="边界带核大小")
    parser.add_argument("--num-folds", type=int, default=3, help="fold 数量")
    parser.add_argument("--seed", type=int, default=42, help="fold 随机种子")
    parser.add_argument("--mask-mode", default="colored", help="成对掩膜二值化策略: colored/nonzero/alpha_nonzero/exact_color")
    parser.add_argument("--foreground-colors", nargs="*", default=[], help="mask-mode=exact_color 时使用的 RGB/HEX 颜色")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    has_labelme = bool(args.annotations_dir)
    has_paired_masks = bool(args.masks_dir)
    if has_labelme == has_paired_masks:
        raise ValueError("请二选一提供 --annotations-dir 或 --masks-dir")

    if has_labelme:
        result = prepare_supervised_dataset(
            images_dir=args.images_dir,
            annotations_dir=args.annotations_dir,
            output_root=args.output_root,
            positive_labels=args.labels,
            auto_crop_sem_region=not args.no_auto_crop,
            crop_detection_ratio=args.crop_detection_ratio,
            edge_kernels=tuple(args.edge_kernels),
            num_folds=args.num_folds,
            seed=args.seed,
        )
    else:
        parsed_colors = [parse_rgb_spec(value) for value in args.foreground_colors]
        result = prepare_external_paired_dataset(
            images_dir=args.images_dir,
            masks_dir=args.masks_dir,
            output_root=args.output_root,
            mask_mode=args.mask_mode,
            foreground_colors=parsed_colors,
            auto_crop_sem_region=not args.no_auto_crop,
            crop_detection_ratio=args.crop_detection_ratio,
            edge_kernels=tuple(args.edge_kernels),
            num_folds=args.num_folds,
            seed=args.seed,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
