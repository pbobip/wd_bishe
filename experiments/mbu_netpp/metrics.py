from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch


def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator <= 0:
        return default
    return float(numerator / denominator)


def _to_numpy_binary(mask: torch.Tensor | np.ndarray, threshold: float = 0.5) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        data = mask.detach().cpu().numpy()
    else:
        data = np.asarray(mask)
    if data.ndim == 4:
        data = data[0, 0]
    elif data.ndim == 3:
        data = data[0]
    return (data >= threshold).astype(np.uint8)


def boundary_f1_score(pred_mask: np.ndarray, gt_mask: np.ndarray, tolerance: int = 2) -> float:
    kernel = np.ones((3, 3), dtype=np.uint8)
    pred_edge = cv2.morphologyEx(pred_mask.astype(np.uint8) * 255, cv2.MORPH_GRADIENT, kernel) > 0
    gt_edge = cv2.morphologyEx(gt_mask.astype(np.uint8) * 255, cv2.MORPH_GRADIENT, kernel) > 0

    pred_count = int(np.count_nonzero(pred_edge))
    gt_count = int(np.count_nonzero(gt_edge))
    if pred_count == 0 and gt_count == 0:
        return 1.0
    if pred_count == 0 or gt_count == 0:
        return 0.0

    tol_kernel = np.ones((2 * tolerance + 1, 2 * tolerance + 1), dtype=np.uint8)
    pred_dilated = cv2.dilate(pred_edge.astype(np.uint8), tol_kernel, iterations=1) > 0
    gt_dilated = cv2.dilate(gt_edge.astype(np.uint8), tol_kernel, iterations=1) > 0

    precision = _safe_divide(np.count_nonzero(pred_edge & gt_dilated), pred_count, default=0.0)
    recall = _safe_divide(np.count_nonzero(gt_edge & pred_dilated), gt_count, default=0.0)
    if precision + recall <= 0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def compute_binary_segmentation_metrics(
    pred_mask: torch.Tensor | np.ndarray,
    target_mask: torch.Tensor | np.ndarray,
    boundary_tolerance: int = 2,
) -> dict[str, float]:
    pred_np = _to_numpy_binary(pred_mask, threshold=0.5)
    gt_np = _to_numpy_binary(target_mask, threshold=0.5)

    intersection = float(np.count_nonzero(pred_np & gt_np))
    pred_sum = float(np.count_nonzero(pred_np))
    target_sum = float(np.count_nonzero(gt_np))
    union = pred_sum + target_sum - intersection

    dice = _safe_divide(2.0 * intersection, pred_sum + target_sum, default=1.0 if pred_sum == target_sum == 0 else 0.0)
    iou = _safe_divide(intersection, union, default=1.0 if union == 0 else 0.0)
    precision = _safe_divide(intersection, pred_sum, default=1.0 if pred_sum == 0 and target_sum == 0 else 0.0)
    recall = _safe_divide(intersection, target_sum, default=1.0 if pred_sum == 0 and target_sum == 0 else 0.0)
    pred_vf = _safe_divide(pred_sum, float(pred_np.size), default=0.0)
    gt_vf = _safe_divide(target_sum, float(gt_np.size), default=0.0)
    vf_error = abs(pred_vf - gt_vf)
    boundary_f1 = boundary_f1_score(pred_np, gt_np, tolerance=boundary_tolerance)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "vf": float(vf_error),
        "vf_pred": float(pred_vf),
        "vf_gt": float(gt_vf),
        "boundary_f1": float(boundary_f1),
    }


def compute_segmentation_metrics(
    logits: torch.Tensor,
    target_mask: torch.Tensor,
    threshold: float = 0.5,
    boundary_tolerance: int = 2,
) -> dict[str, float]:
    probs = torch.sigmoid(logits)
    pred = (probs >= threshold).float()
    target = (target_mask >= 0.5).float()

    return compute_binary_segmentation_metrics(pred, target, boundary_tolerance=boundary_tolerance)


def average_metric_dicts(items: list[dict[str, Any]]) -> dict[str, float]:
    if not items:
        return {}
    keys = sorted({key for item in items for key in item.keys()})
    summary: dict[str, float] = {}
    for key in keys:
        values = [float(item[key]) for item in items if key in item]
        summary[key] = float(np.mean(values)) if values else 0.0
    return summary
