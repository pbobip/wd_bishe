from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from backend.app.schemas.run import TraditionalSegConfig
from backend.app.services.statistics import statistics_service
from backend.app.utils.image_io import ensure_odd


class TraditionalService:
    def segment(self, source: np.ndarray, config: TraditionalSegConfig) -> dict[str, Any]:
        gray = self._ensure_gray(source)
        binary, route_details = self._segment_binary(gray, config)

        if config.fill_holes:
            binary = self._fill_holes(binary)

        if config.boundary_smoothing:
            binary = self._smooth_boundary(binary, config.boundary_smoothing_kernel)

        binary = self._morph_cleanup(binary, config.open_kernel, config.close_kernel)

        if config.watershed:
            binary = self._apply_watershed(gray, binary)

        binary = (binary > 0).astype(np.uint8) * 255
        object_stats = statistics_service.build_object_stats(binary, config=config, base_image=gray)
        kept_mask = object_stats["kept_mask"].astype(np.uint8)
        overlay = self._build_overlay(gray, kept_mask)
        edges = self._edge_map(gray, config.edge_operator, config)

        return {
            "mask": kept_mask,
            "edges": edges,
            "overlay": overlay,
            "object_overlay": object_stats["object_overlay"],
            "route": config.method,
            "route_details": route_details,
            "objects": object_stats["objects"],
            "particles": object_stats["particles"],
        }

    def _ensure_gray(self, source: np.ndarray) -> np.ndarray:
        if source.ndim != 2:
            raise ValueError("传统分割仅支持灰度图像")
        if source.dtype != np.uint8:
            clipped = np.clip(source, 0, 255)
            return clipped.astype(np.uint8)
        return source

    def _segment_binary(self, source: np.ndarray, config: TraditionalSegConfig) -> tuple[np.ndarray, dict[str, Any]]:
        if config.method == "adaptive":
            return self._adaptive_segment(source, config)
        if config.method == "edge":
            return self._edge_segment(source, config)
        if config.method == "clustering":
            return self._cluster_segment(source, config)
        return self._threshold_segment(source, config)

    def _threshold_segment(self, source: np.ndarray, config: TraditionalSegConfig) -> tuple[np.ndarray, dict[str, Any]]:
        binary, threshold_value = self._threshold(source, config)
        return binary.astype(np.uint8), {
            "label": "阈值分割",
            "threshold_mode": config.threshold_mode,
            "threshold": float(threshold_value),
        }

    def _adaptive_segment(self, source: np.ndarray, config: TraditionalSegConfig) -> tuple[np.ndarray, dict[str, Any]]:
        block_size = max(3, ensure_odd(int(config.adaptive_block_size)))
        adaptive_method = (
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C
            if config.adaptive_method == "gaussian"
            else cv2.ADAPTIVE_THRESH_MEAN_C
        )
        binary = cv2.adaptiveThreshold(
            source,
            255,
            adaptive_method,
            cv2.THRESH_BINARY,
            block_size,
            float(config.adaptive_c),
        )
        return binary.astype(np.uint8), {
            "label": "自适应阈值",
            "adaptive_method": config.adaptive_method,
            "adaptive_block_size": block_size,
            "adaptive_c": float(config.adaptive_c),
        }

    def _edge_segment(self, source: np.ndarray, config: TraditionalSegConfig) -> tuple[np.ndarray, dict[str, Any]]:
        blur_kernel = max(1, ensure_odd(int(config.edge_blur_kernel)))
        working = source
        if blur_kernel > 1:
            working = cv2.GaussianBlur(source, (blur_kernel, blur_kernel), 0)

        edges = self._edge_map(working, config.edge_operator, config)
        if config.edge_dilate_iterations > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges = cv2.dilate(edges, kernel, iterations=max(1, int(config.edge_dilate_iterations)))

        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_kernel, iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        binary = np.zeros_like(source, dtype=np.uint8)
        if contours:
            cv2.drawContours(binary, contours, -1, 255, thickness=cv2.FILLED)

        return binary.astype(np.uint8), {
            "label": "边缘分割",
            "edge_operator": config.edge_operator,
            "edge_blur_kernel": blur_kernel,
            "edge_threshold1": int(config.edge_threshold1),
            "edge_threshold2": int(config.edge_threshold2),
            "edge_dilate_iterations": int(config.edge_dilate_iterations),
        }

    def _cluster_segment(self, source: np.ndarray, config: TraditionalSegConfig) -> tuple[np.ndarray, dict[str, Any]]:
        flat = source.reshape((-1, 1)).astype(np.float32)
        clusters = max(2, int(config.kmeans_clusters))
        attempts = max(1, int(config.kmeans_attempts))
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            20,
            0.2,
        )
        _compactness, labels, centers = cv2.kmeans(
            flat,
            clusters,
            None,
            criteria,
            attempts,
            cv2.KMEANS_PP_CENTERS,
        )
        centers = centers.reshape(-1)
        label_grid = labels.reshape(source.shape)
        if config.cluster_target == "dark":
            target = int(np.argmin(centers))
        elif config.cluster_target == "largest":
            counts = np.bincount(labels.flatten(), minlength=clusters)
            target = int(np.argmax(counts))
        else:
            target = int(np.argmax(centers))
        binary = np.where(label_grid == target, 255, 0).astype(np.uint8)
        return binary, {
            "label": "聚类分割",
            "kmeans_clusters": clusters,
            "kmeans_attempts": attempts,
            "cluster_target": config.cluster_target,
            "cluster_centers": [float(value) for value in centers.tolist()],
        }

    def _threshold(self, source: np.ndarray, config: TraditionalSegConfig) -> tuple[np.ndarray, float]:
        if config.threshold_mode == "otsu":
            threshold_value, binary = cv2.threshold(source, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif config.threshold_mode == "global":
            threshold_value = float(config.global_threshold)
            threshold_value = max(0.0, min(255.0, threshold_value))
            _, binary = cv2.threshold(source, threshold_value, 255, cv2.THRESH_BINARY)
        else:
            threshold_value = float(config.fixed_threshold)
            threshold_value = max(0.0, min(255.0, threshold_value))
            _, binary = cv2.threshold(source, threshold_value, 255, cv2.THRESH_BINARY)
        return binary.astype(np.uint8), float(threshold_value)

    def _fill_holes(self, binary: np.ndarray) -> np.ndarray:
        foreground = (binary > 0).astype(np.uint8) * 255
        inverted = cv2.bitwise_not(foreground)
        height, width = inverted.shape[:2]
        mask = np.zeros((height + 2, width + 2), np.uint8)
        flooded = inverted.copy()
        cv2.floodFill(flooded, mask, (0, 0), 0)
        holes = cv2.bitwise_not(flooded)
        filled = cv2.bitwise_or(foreground, holes)
        return (filled > 0).astype(np.uint8) * 255

    def _smooth_boundary(self, binary: np.ndarray, kernel_size: int) -> np.ndarray:
        kernel_size = max(1, ensure_odd(int(kernel_size)))
        if kernel_size <= 1:
            return binary
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        smoothed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel, iterations=1)
        return (smoothed > 0).astype(np.uint8) * 255

    def _morph_cleanup(self, binary: np.ndarray, open_kernel: int, close_kernel: int) -> np.ndarray:
        open_kernel = max(1, ensure_odd(int(open_kernel)))
        close_kernel = max(1, ensure_odd(int(close_kernel)))
        result = binary
        if open_kernel > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
            result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel, iterations=1)
        if close_kernel > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
            result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=1)
        return (result > 0).astype(np.uint8) * 255

    def _apply_watershed(self, gray: np.ndarray, binary: np.ndarray) -> np.ndarray:
        foreground = (binary > 0).astype(np.uint8) * 255
        if int(foreground.sum()) == 0:
            return foreground
        kernel = np.ones((3, 3), np.uint8)
        sure_bg = cv2.dilate(foreground, kernel, iterations=1)
        dist = cv2.distanceTransform(foreground, cv2.DIST_L2, 5)
        if dist.max() <= 0:
            return foreground
        _, sure_fg = cv2.threshold(dist, 0.35 * dist.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(color, markers)
        separated = np.zeros_like(foreground)
        separated[markers > 1] = 255
        return separated

    def _edge_map(self, gray: np.ndarray, operator: str, config: TraditionalSegConfig) -> np.ndarray:
        if operator == "sobel":
            grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
            magnitude = cv2.convertScaleAbs(cv2.magnitude(grad_x.astype(np.float32), grad_y.astype(np.float32)))
            _, edges = cv2.threshold(magnitude, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return edges
        if operator == "laplacian":
            lap = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
            lap = cv2.convertScaleAbs(lap)
            _, edges = cv2.threshold(lap, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return edges
        return cv2.Canny(gray, int(config.edge_threshold1), int(config.edge_threshold2))

    def _build_overlay(self, gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
        base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        color_mask = np.zeros_like(base)
        color_mask[mask > 0] = (0, 180, 0)
        overlay = cv2.addWeighted(base, 0.75, color_mask, 0.25, 0)
        return overlay


traditional_service = TraditionalService()
