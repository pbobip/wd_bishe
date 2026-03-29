from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from backend.app.services.statistics import StatisticsService
from backend.app.utils.image_io import read_gray, write_image

from experiments.mbu_netpp.common import build_overlay, ensure_dir, image_to_tensor, load_yaml, save_json
from experiments.mbu_netpp.models import build_model_for_checkpoint, sliding_window_inference
from experiments.mbu_netpp.preparation import detect_crop_height
from experiments.mbu_netpp.preprocess import apply_preprocess


def load_runtime_config(checkpoint_path: str | Path, config_path: str | None) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if config_path:
        config = load_yaml(config_path)
    else:
        config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("推理时未能从 checkpoint 或显式配置中获取 config")
    return checkpoint, config


def load_model_from_checkpoint(checkpoint_path: str | Path, config_path: str | None, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint, config = load_runtime_config(checkpoint_path, config_path)
    model = build_model_for_checkpoint(config["model"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, config


def list_inputs(path: str | Path) -> list[Path]:
    root = Path(path)
    if root.is_file():
        return [root]
    supported = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    return sorted([item for item in root.iterdir() if item.is_file() and item.suffix.lower() in supported])


def maybe_write_xlsx(rows: list[dict[str, Any]], xlsx_path: Path) -> bool:
    try:
        from openpyxl import Workbook
    except Exception:
        return False

    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "summary"
    if not rows:
        workbook.save(xlsx_path)
        return True

    headers = [key for key in rows[0].keys() if not isinstance(rows[0].get(key), (list, dict))]
    worksheet.append(headers)
    for row in rows:
        worksheet.append([row.get(header) for header in headers])
    workbook.save(xlsx_path)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 MBU-Net++ checkpoint 推理")
    parser.add_argument("--checkpoint", required=True, help="best.pt 路径")
    parser.add_argument("--input", required=True, help="输入单图或目录")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--config", default=None, help="可选配置文件，缺省时从 checkpoint 读取")
    parser.add_argument("--threshold", type=float, default=None, help="覆盖默认阈值")
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model, config = load_model_from_checkpoint(args.checkpoint, args.config, device=device)

    data_cfg = config["data"]
    training_cfg = config["training"]
    inference_cfg = config.get("inference", {})
    threshold = float(args.threshold if args.threshold is not None else training_cfg.get("threshold", 0.5))
    patch_size = int(data_cfg.get("patch_size", 256))
    overlap = float(inference_cfg.get("overlap", data_cfg.get("overlap", 0.25)))
    normalization = str(data_cfg.get("normalization", "minmax"))
    preprocess_cfg = data_cfg.get("preprocess")
    auto_crop = bool(data_cfg.get("auto_crop_sem_region", True))
    crop_detection_ratio = float(data_cfg.get("crop_detection_ratio", 0.75))

    output_dir = ensure_dir(args.output_dir)
    masks_dir = ensure_dir(output_dir / "masks")
    overlays_dir = ensure_dir(output_dir / "overlays")
    stats_dir = ensure_dir(output_dir / "stats")

    statistics_service = StatisticsService()
    summary_rows: list[dict[str, Any]] = []

    for image_path in list_inputs(args.input):
        original = read_gray(image_path)
        crop_height = original.shape[0]
        cropped = original
        if auto_crop:
            crop_height = detect_crop_height(original, start_ratio=crop_detection_ratio)
            cropped = original[:crop_height, :]

        processed = apply_preprocess(cropped, preprocess_cfg)
        image_tensor = image_to_tensor(processed, normalization=normalization).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = sliding_window_inference(model, image_tensor, patch_size=patch_size, overlap=overlap)
        mask = (torch.sigmoid(logits).cpu().numpy()[0, 0] >= threshold).astype(np.uint8) * 255

        full_mask = np.zeros_like(original, dtype=np.uint8)
        full_mask[:crop_height, :] = mask
        overlay = build_overlay(original, full_mask)

        write_image(masks_dir / f"{image_path.stem}_mask.png", full_mask)
        write_image(overlays_dir / f"{image_path.stem}_overlay.png", overlay)

        object_stats = statistics_service.build_object_stats(mask, base_image=cropped)
        summary = statistics_service.summarize(mask, objects=object_stats["objects"])
        summary["stem"] = image_path.stem
        summary["crop_height"] = int(crop_height)
        save_json(stats_dir / f"{image_path.stem}.json", summary)
        summary_rows.append(summary)

    if summary_rows:
        headers = sorted({key for row in summary_rows for key in row.keys() if not isinstance(row.get(key), (list, dict))})
        csv_path = output_dir / "summary.csv"
        with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers)
            writer.writeheader()
            for row in summary_rows:
                writer.writerow({key: row.get(key) for key in headers})
        maybe_write_xlsx(summary_rows, output_dir / "summary.xlsx")

    print(json.dumps({"num_images": len(summary_rows), "output_dir": str(output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
