from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def detect_crop_height(image: np.ndarray) -> int:
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    bottom_h = int(h * 0.75)
    edges = cv2.Canny(gray[bottom_h:, :], 50, 150)
    edge_sum = np.sum(edges, axis=1)
    candidates = np.where(edge_sum > w * 0.5 * 255)[0]
    if len(candidates) > 0:
        return bottom_h + int(candidates[0])
    return int(h * 0.85)


def read_color(path: str) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图像: {path}")
    return image


def write_image(path: str, image: np.ndarray) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(target.suffix or ".png", image)
    if not success:
        raise ValueError(f"无法写入图像: {path}")
    encoded.tofile(path)


def save_overlay(gray: np.ndarray, mask: np.ndarray, output_path: str) -> None:
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    color = overlay.copy()
    color[mask == 255] = [48, 132, 103]
    blended = cv2.addWeighted(overlay, 0.68, color, 0.32, 0)
    write_image(output_path, blended)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


def emit_progress(
    config: dict[str, Any],
    processed: int,
    total: int,
    image_name: str | None = None,
    status: str = "running",
    error: str | None = None,
) -> None:
    progress_path = config.get("progress_path")
    if not progress_path:
        return
    write_json_atomic(
        Path(progress_path),
        {
            "processed": int(processed),
            "total": int(total),
            "image_name": image_name,
            "status": status,
            "error": error,
        },
    )


def run_matsam(config: dict[str, Any]) -> dict[str, Any]:
    try:
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("当前运行环境缺少 segment_anything 或 torch") from exc

    checkpoint = config.get("weight_path")
    if not checkpoint or not Path(checkpoint).exists():
        raise RuntimeError("MatSAM 需要有效的 SAM 权重路径")

    device = "cuda" if config.get("device") != "cpu" and torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_h"](checkpoint=checkpoint)
    sam.to(device)
    generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
    )

    items = []
    total_items = len(config["items"])
    emit_progress(config, 0, total_items)
    for index, item in enumerate(config["items"], start=1):
        image = read_color(item["image_path"])
        crop_h = detect_crop_height(image)
        cropped = image[:crop_h, :]
        rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        masks = generator.generate(rgb)
        combined = np.zeros(cropped.shape[:2], dtype=np.uint8)
        total_area = combined.size
        min_area = total_area * 0.0001
        max_area = total_area * 0.05
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        for mask_data in masks:
            mask = mask_data["segmentation"]
            area = mask_data["area"]
            if min_area < area < max_area and np.mean(gray[mask]) > 100:
                combined[mask] = 255

        full_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        full_mask[:crop_h, :] = combined
        output_dir = Path(config["output_dir"])
        stem = Path(item["image_name"]).stem
        mask_path = output_dir / f"{stem}_mask.png"
        overlay_path = output_dir / f"{stem}_overlay.png"
        write_image(str(mask_path), full_mask)
        save_overlay(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), full_mask, str(overlay_path))
        items.append(
            {
                "image_id": item["image_id"],
                "image_name": item["image_name"],
                "relative_input_path": item["relative_input_path"],
                "mask_path": str(mask_path),
                "overlay_path": str(overlay_path),
            }
        )
        emit_progress(config, index, total_items, image_name=item["image_name"])
    return {"status": "completed", "items": items}


def run_custom(config: dict[str, Any]) -> dict[str, Any]:
    items = []
    threshold = int(config.get("extra_params", {}).get("threshold", 120))
    total_items = len(config["items"])
    emit_progress(config, 0, total_items)
    for index, item in enumerate(config["items"], start=1):
        image = read_color(item["image_path"])
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        output_dir = Path(config["output_dir"])
        stem = Path(item["image_name"]).stem
        mask_path = output_dir / f"{stem}_mask.png"
        overlay_path = output_dir / f"{stem}_overlay.png"
        write_image(str(mask_path), mask)
        save_overlay(gray, mask, str(overlay_path))
        items.append(
            {
                "image_id": item["image_id"],
                "image_name": item["image_name"],
                "relative_input_path": item["relative_input_path"],
                "mask_path": str(mask_path),
                "overlay_path": str(overlay_path),
            }
        )
        emit_progress(config, index, total_items, image_name=item["image_name"])
    return {"status": "completed", "items": items}


def main() -> None:
    config_path = Path(sys.argv[1])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_kind = config.get("model_kind")
    try:
        if model_kind == "matsam":
            manifest = run_matsam(config)
        elif model_kind == "custom":
            manifest = run_custom(config)
        else:
            raise RuntimeError(f"模型 {model_kind} 尚未绑定真实推理实现，请先配置 runner 或权重")
        emit_progress(config, len(manifest["items"]), len(config["items"]), status="completed")
        Path(config["manifest_path"]).write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        last_processed = 0
        total_items = len(config.get("items", []))
        progress_path = Path(config["progress_path"]) if config.get("progress_path") else None
        if progress_path and progress_path.exists():
            try:
                progress_payload = json.loads(progress_path.read_text(encoding="utf-8"))
                last_processed = int(progress_payload.get("processed", 0))
                total_items = int(progress_payload.get("total", total_items))
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                pass
        emit_progress(config, last_processed, total_items, status="failed", error=str(exc))
        raise


if __name__ == "__main__":
    main()
