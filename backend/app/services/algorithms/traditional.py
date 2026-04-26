from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from backend.app.schemas.run import PostprocessConfig, TraditionalSegConfig
from backend.app.services.statistics import statistics_service
from backend.app.utils.image_io import ensure_odd


class TraditionalService:
    def segment(self, source: np.ndarray, config: TraditionalSegConfig) -> dict[str, Any]:
        gray = self._ensure_gray(source)
        binary, route_details = self._segment_binary(gray, config)
        return self._finalize_result(gray, binary, config, route=config.method, route_details=route_details)

    def build_postprocess_config(
        self,
        base_config: TraditionalSegConfig,
        postprocess: PostprocessConfig,
    ) -> TraditionalSegConfig:
        payload = base_config.model_dump()
        shape_filter = postprocess.shape_filter
        smoothing = postprocess.smoothing
        morphology = postprocess.morphology
        watershed_params = postprocess.watershed_params
        payload.update(
            {
                "fill_holes": bool(postprocess.fill_holes),
                "watershed": bool(postprocess.watershed),
                "watershed_separation": int(watershed_params.separation),
                "watershed_bg_iterations": int(watershed_params.background_iterations),
                "watershed_min_marker_area": int(watershed_params.min_marker_area),
                "boundary_smoothing": bool(smoothing.enabled),
                "boundary_smoothing_method": smoothing.method if smoothing.enabled else "morphology",
                "boundary_smoothing_kernel": int(smoothing.kernel),
                "min_area": int(shape_filter.min_area) if shape_filter.enabled else 0,
                "max_area": int(shape_filter.max_area) if shape_filter.enabled and shape_filter.max_area is not None else None,
                "min_solidity": float(shape_filter.min_solidity) if shape_filter.enabled else 0.0,
                "min_circularity": float(shape_filter.min_circularity) if shape_filter.enabled else 0.0,
                "min_roundness": float(shape_filter.min_roundness) if shape_filter.enabled else 0.0,
                "max_aspect_ratio": (
                    float(shape_filter.max_aspect_ratio)
                    if shape_filter.enabled and shape_filter.max_aspect_ratio is not None
                    else None
                ),
                "remove_border": bool(postprocess.remove_border),
                "open_kernel": int(morphology.opening_kernel) if morphology.opening_enabled else 1,
                "close_kernel": int(morphology.closing_kernel) if morphology.closing_enabled else 1,
            }
        )
        return TraditionalSegConfig.model_validate(payload)

    def apply_postprocess(self, source: np.ndarray, mask: np.ndarray, config: TraditionalSegConfig) -> dict[str, Any]:
        gray = self._ensure_gray(source)
        binary = (self._ensure_gray(mask) > 0).astype(np.uint8) * 255
        route_details = {"label": "现有掩码后处理"}
        return self._finalize_result(
            gray,
            binary,
            config,
            route="postprocess",
            route_details=route_details,
        )

    def _finalize_result(
        self,
        gray: np.ndarray,
        binary: np.ndarray,
        config: TraditionalSegConfig,
        *,
        route: str,
        route_details: dict[str, Any],
    ) -> dict[str, Any]:
        if config.fill_holes:
            binary = self._fill_holes(binary)

        if config.boundary_smoothing:
            binary = self._smooth_boundary(binary, config.boundary_smoothing_kernel, config.boundary_smoothing_method)

        binary = self._morph_cleanup(binary, config.open_kernel, config.close_kernel)

        if config.watershed:
            binary = self._apply_watershed(
                gray,
                binary,
                separation=config.watershed_separation,
                background_iterations=config.watershed_bg_iterations,
                min_marker_area=config.watershed_min_marker_area,
            )

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
            "route": route,
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
            "foreground_target": config.foreground_target,
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
        threshold_type = cv2.THRESH_BINARY_INV if config.foreground_target == "dark" else cv2.THRESH_BINARY
        binary = cv2.adaptiveThreshold(
            source,
            255,
            adaptive_method,
            threshold_type,
            block_size,
            float(config.adaptive_c),
        )
        return binary.astype(np.uint8), {
            "label": "自适应阈值",
            "foreground_target": config.foreground_target,
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
        threshold_type = cv2.THRESH_BINARY_INV if config.foreground_target == "dark" else cv2.THRESH_BINARY
        if config.threshold_mode == "otsu":
            threshold_value, binary = cv2.threshold(source, 0, 255, threshold_type + cv2.THRESH_OTSU)
        elif config.threshold_mode == "global":
            threshold_value = float(config.global_threshold)
            threshold_value = max(0.0, min(255.0, threshold_value))
            _, binary = cv2.threshold(source, threshold_value, 255, threshold_type)
        else:
            threshold_value = float(config.fixed_threshold)
            threshold_value = max(0.0, min(255.0, threshold_value))
            _, binary = cv2.threshold(source, threshold_value, 255, threshold_type)
        return binary.astype(np.uint8), float(threshold_value)

    def _fill_holes(self, binary: np.ndarray) -> np.ndarray:
        foreground = (binary > 0).astype(np.uint8) * 255
        padded = cv2.copyMakeBorder(foreground, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        flooded = padded.copy()
        height, width = flooded.shape[:2]
        mask = np.zeros((height + 2, width + 2), np.uint8)
        cv2.floodFill(flooded, mask, (0, 0), 255)
        holes = cv2.bitwise_not(flooded)[1:-1, 1:-1]
        filled = cv2.bitwise_or(foreground, holes)
        return (filled > 0).astype(np.uint8) * 255

    def _effective_iterations(self, kernel_size: int) -> int:
        kernel_size = max(1, ensure_odd(int(kernel_size)))
        if kernel_size >= 17:
            return 3
        if kernel_size >= 11:
            return 2
        return 1

    def _smooth_boundary(self, binary: np.ndarray, kernel_size: int, method: str = "morphology") -> np.ndarray:
        kernel_size = max(1, ensure_odd(int(kernel_size)))
        if kernel_size <= 1:
            return binary

        foreground = (binary > 0).astype(np.uint8) * 255
        foreground_area = cv2.countNonZero(foreground)
        if foreground_area == 0:
            return foreground
        if method in {"mean", "gaussian", "median"}:
            passes = self._effective_iterations(kernel_size)
            blurred = foreground.copy()
            for _ in range(passes):
                if method == "mean":
                    blurred = cv2.blur(blurred, (kernel_size, kernel_size))
                elif method == "median":
                    blurred = cv2.medianBlur(blurred, kernel_size)
                else:
                    blurred = cv2.GaussianBlur(blurred, (kernel_size, kernel_size), 0)
            _, smoothed = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
            refine_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            if kernel_size >= 5:
                smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, refine_kernel, iterations=1)
                smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_CLOSE, refine_kernel, iterations=1)
            smoothed_area = cv2.countNonZero(smoothed)
            if smoothed_area > foreground_area * 1.10:
                _, smoothed = cv2.threshold(blurred, 143, 255, cv2.THRESH_BINARY)
            elif smoothed_area < foreground_area * 0.90:
                _, smoothed = cv2.threshold(blurred, 111, 255, cv2.THRESH_BINARY)
            smoothed_area = cv2.countNonZero(smoothed)
            if smoothed_area > foreground_area * 1.20 or smoothed_area < foreground_area * 0.80:
                return foreground
            return (smoothed > 0).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        iterations = self._effective_iterations(kernel_size)
        smoothed = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel, iterations=max(1, iterations - 1))
        smoothed_area = cv2.countNonZero(smoothed)
        if smoothed_area > foreground_area * 1.15:
            support_iterations = max(1, min(3, kernel_size // 5))
            support_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            support = cv2.dilate(foreground, support_kernel, iterations=support_iterations)
            smoothed = cv2.bitwise_and(smoothed, support)
        elif smoothed_area < foreground_area * 0.85:
            smoothed = cv2.bitwise_or(smoothed, cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel, iterations=1))
        return (smoothed > 0).astype(np.uint8) * 255

    def _morph_cleanup(self, binary: np.ndarray, open_kernel: int, close_kernel: int) -> np.ndarray:
        open_kernel = max(1, ensure_odd(int(open_kernel)))
        close_kernel = max(1, ensure_odd(int(close_kernel)))
        result = (binary > 0).astype(np.uint8) * 255
        if open_kernel > 1:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((result > 0).astype(np.uint8), connectivity=8)
            filtered = np.zeros_like(result)
            min_noise_area = max(2, int(open_kernel * open_kernel * 0.6))
            for label in range(1, num_labels):
                if int(stats[label, cv2.CC_STAT_AREA]) >= min_noise_area:
                    filtered[labels == label] = 255
            result = filtered
        if close_kernel > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
            closed = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=self._effective_iterations(close_kernel))
            support_iterations = max(1, min(4, close_kernel // 4))
            support_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            support = cv2.dilate(result, support_kernel, iterations=support_iterations)
            result = cv2.bitwise_and(closed, support)
        return (result > 0).astype(np.uint8) * 255

    def _apply_watershed(
        self,
        gray: np.ndarray,
        binary: np.ndarray,
        *,
        separation: int = 35,
        background_iterations: int = 1,
        min_marker_area: int = 12,
    ) -> np.ndarray:
        foreground = (binary > 0).astype(np.uint8) * 255
        if int(foreground.sum()) == 0:
            return foreground
        kernel = np.ones((3, 3), np.uint8)
        bg_iterations = max(1, min(5, int(background_iterations)))
        sure_bg = cv2.dilate(foreground, kernel, iterations=bg_iterations)
        dist = cv2.distanceTransform(foreground, cv2.DIST_L2, 5)
        if dist.max() <= 0:
            return foreground
        dist = cv2.GaussianBlur(dist, (3, 3), 0)
        separation_ratio = max(0.05, min(0.85, int(separation) / 100.0))
        _, sure_fg = cv2.threshold(dist, separation_ratio * dist.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        seed_kernel_size = 3 if separation <= 45 else 5
        seed_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (seed_kernel_size, seed_kernel_size))
        sure_fg = cv2.morphologyEx(sure_fg, cv2.MORPH_OPEN, seed_kernel, iterations=1)
        marker_area = max(0, int(min_marker_area))
        if marker_area > 0:
            num_labels, marker_labels, stats, _ = cv2.connectedComponentsWithStats(sure_fg)
            filtered_fg = np.zeros_like(sure_fg)
            for label in range(1, num_labels):
                if int(stats[label, cv2.CC_STAT_AREA]) >= marker_area:
                    filtered_fg[marker_labels == label] = 255
            sure_fg = filtered_fg
            if int(sure_fg.sum()) == 0:
                return foreground
        unknown = cv2.subtract(sure_bg, sure_fg)
        _, markers = cv2.connectedComponents(sure_fg)
        if int(markers.max()) <= 1:
            return foreground
        markers = markers + 1
        markers[unknown == 255] = 0
        color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(color, markers)
        separated = np.zeros_like(foreground)
        separated[(markers > 1) & (foreground > 0)] = 255
        separated_area = cv2.countNonZero(separated)
        foreground_area = cv2.countNonZero(foreground)
        if separated_area < foreground_area * 0.70:
            return foreground
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
