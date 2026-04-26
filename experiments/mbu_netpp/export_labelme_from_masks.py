from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def list_overlay_files(overlays_dir: Path) -> list[Path]:
    files = sorted(
        [
            path
            for path in overlays_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".png"
        ]
    )
    if not files:
        raise FileNotFoundError(f"未在 {overlays_dir} 找到 overlay PNG")
    return files


def build_mask_path(masks_dir: Path, overlay_name: str) -> Path:
    if not overlay_name.endswith("_overlay.png"):
        raise ValueError(f"overlay 文件名不符合约定: {overlay_name}")
    stem = overlay_name[: -len("_overlay.png")]
    return masks_dir / f"{stem}_mask.png"


def contour_to_points(contour: np.ndarray) -> list[list[float]]:
    points: list[list[float]] = []
    for point in contour.reshape(-1, 2):
        x = float(point[0]) + 0.5
        y = float(point[1]) + 0.5
        points.append([x, y])
    return points


def extract_shapes(
    mask: np.ndarray,
    label: str,
    min_area: float,
    simplify_epsilon: float,
) -> list[dict[str, Any]]:
    binary = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    shapes: list[dict[str, Any]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        working = contour
        if simplify_epsilon > 0:
            working = cv2.approxPolyDP(contour, epsilon=simplify_epsilon, closed=True)
        if len(working) < 3:
            continue
        shapes.append(
            {
                "label": label,
                "points": contour_to_points(working),
                "group_id": None,
                "description": "",
                "shape_type": "polygon",
                "flags": {
                    "done": False,
                    "checked": False,
                    "uncertain": False,
                },
                "mask": None,
            }
        )
    return shapes


def build_labelme_payload(
    image_name: str,
    image_shape: tuple[int, int],
    shapes: list[dict[str, Any]],
) -> dict[str, Any]:
    height, width = image_shape
    return {
        "version": "5.11.4",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": int(height),
        "imageWidth": int(width),
    }


def export_labelme_review_bundle(
    overlays_dir: str | Path,
    masks_dir: str | Path,
    output_dir: str | Path,
    label: str = "gamma_prime",
    min_area: float = 16.0,
    simplify_epsilon: float = 0.0,
    clean_output: bool = True,
) -> dict[str, Any]:
    overlays_root = Path(overlays_dir)
    masks_root = Path(masks_dir)
    output_root = Path(output_dir)

    if clean_output and output_root.exists():
        shutil.rmtree(output_root)
    ensure_dir(output_root)

    overlay_files = list_overlay_files(overlays_root)
    num_shapes = 0
    exported_items: list[dict[str, Any]] = []

    for overlay_path in overlay_files:
        mask_path = build_mask_path(masks_root, overlay_path.name)
        if not mask_path.exists():
            raise FileNotFoundError(f"缺少与 {overlay_path.name} 匹配的 mask: {mask_path}")

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"无法读取 mask: {mask_path}")

        overlay_image = cv2.imread(str(overlay_path), cv2.IMREAD_UNCHANGED)
        if overlay_image is None:
            raise RuntimeError(f"无法读取 overlay: {overlay_path}")

        shapes = extract_shapes(
            mask=mask,
            label=label,
            min_area=float(min_area),
            simplify_epsilon=float(simplify_epsilon),
        )
        num_shapes += len(shapes)

        target_image_path = output_root / overlay_path.name
        target_json_path = output_root / f"{overlay_path.stem}.json"
        shutil.copy2(overlay_path, target_image_path)
        target_json_path.write_text(
            json.dumps(
                build_labelme_payload(
                    image_name=overlay_path.name,
                    image_shape=overlay_image.shape[:2],
                    shapes=shapes,
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        exported_items.append(
            {
                "image": str(target_image_path),
                "json": str(target_json_path),
                "num_shapes": len(shapes),
            }
        )

    summary = {
        "output_dir": str(output_root.resolve()),
        "num_images": len(overlay_files),
        "num_json": len(overlay_files),
        "label": label,
        "min_area": float(min_area),
        "simplify_epsilon": float(simplify_epsilon),
        "total_shapes": int(num_shapes),
        "items": exported_items,
    }
    (output_root / "_export_summary.txt").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将预测 mask 导出成可在 LabelMe 中直接修改的 polygon json")
    parser.add_argument("--overlays-dir", required=True, help="overlay 图片目录")
    parser.add_argument("--masks-dir", required=True, help="二值 mask 目录")
    parser.add_argument("--output-dir", required=True, help="导出的 LabelMe 审核目录")
    parser.add_argument("--label", default="gamma_prime", help="LabelMe 标签名")
    parser.add_argument("--min-area", type=float, default=16.0, help="最小轮廓面积，低于该值将忽略")
    parser.add_argument("--simplify-epsilon", type=float, default=0.0, help="轮廓简化 epsilon，0 表示不额外简化")
    parser.add_argument("--keep-output", action="store_true", help="保留已有输出目录，不先清空")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = export_labelme_review_bundle(
        overlays_dir=args.overlays_dir,
        masks_dir=args.masks_dir,
        output_dir=args.output_dir,
        label=args.label,
        min_area=float(args.min_area),
        simplify_epsilon=float(args.simplify_epsilon),
        clean_output=not bool(args.keep_output),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
