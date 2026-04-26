from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from backend.app.schemas.run import TraditionalSegConfig


def _skeletonize(binary: np.ndarray) -> np.ndarray:
    """Zhang-Suen 形态学骨架化（纯 OpenCV 实现，无需 skimage）。"""
    skeleton = np.zeros_like(binary, dtype=np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    remaining = binary.copy()
    while True:
        eroded = cv2.erode(remaining, element)
        dilated = cv2.dilate(eroded, element)
        diff = cv2.subtract(remaining, dilated)
        skeleton = cv2.bitwise_or(skeleton, diff)
        remaining = eroded.copy()
        if cv2.countNonZero(remaining) == 0:
            break
    return skeleton


class StatisticsService:
    def build_object_stats(
        self,
        mask: np.ndarray,
        um_per_px: float | None = None,
        config: TraditionalSegConfig | None = None,
        base_image: np.ndarray | None = None,
    ) -> dict[str, Any]:
        binary = (mask > 0).astype(np.uint8)
        scale = um_per_px if um_per_px and um_per_px > 0 else None
        unit = "um" if scale else "px"
        height, width = binary.shape[:2]

        canvas = self._prepare_canvas(base_image, binary)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        objects: list[dict[str, Any]] = []
        kept_mask = np.zeros_like(binary, dtype=np.uint8)

        for label in range(1, num_labels):
            x, y, w, h, area_px = (int(value) for value in stats[label])
            if area_px <= 0:
                continue
            component_mask = (labels == label).astype(np.uint8) * 255
            contour = self._extract_largest_contour(component_mask)
            centroid_x, centroid_y = (float(value) for value in centroids[label])
            metrics = self._measure_object(
                contour=contour,
                area_px=area_px,
                centroid_x=centroid_x,
                centroid_y=centroid_y,
                scale=scale,
            )
            reasons = self._filter_reasons(
                config=config,
                area_px=area_px,
                bbox=(x, y, w, h),
                image_shape=(height, width),
                metrics=metrics,
            )
            filtered = bool(reasons)
            reason_text = ";".join(reasons) if reasons else None
            # ------ Lantuéjoul 权重：P = (W-w)*(H-h) / (W*H), weight = 1/P ------
            if not filtered and w < width and h < height:
                survival_prob = float((width - w) * (height - h)) / float(width * height)
                lantueoul_weight = 1.0 / survival_prob if survival_prob > 0 else 1.0
            else:
                lantueoul_weight = 1.0

            object_record = {
                "label": int(label),
                "bbox_x": int(x),
                "bbox_y": int(y),
                "bbox_w": int(w),
                "bbox_h": int(h),
                "area_px": int(area_px),
                "area_value": float(metrics["area_value"]),
                "size_value": float(metrics["size_value"]),
                "equiv_diameter": float(metrics["equiv_diameter"]),
                "perimeter": float(metrics["perimeter_value"]),
                "perimeter_value": float(metrics["perimeter_value"]),
                "major": float(metrics["major_value"]),
                "major_value": float(metrics["major_value"]),
                "minor": float(metrics["minor_value"]),
                "minor_value": float(metrics["minor_value"]),
                "feret": float(metrics["feret_value"]),
                "feret_value": float(metrics["feret_value"]),
                "minferet": float(metrics["minferet_value"]),
                "minferet_value": float(metrics["minferet_value"]),
                "aspect_ratio": float(metrics["aspect_ratio"]),
                "circularity": float(metrics["circularity"]),
                "roundness": float(metrics["roundness"]),
                "solidity": float(metrics["solidity"]),
                "centroid_x": float(centroid_x),
                "centroid_y": float(centroid_y),
                "weight": float(lantueoul_weight),
                "filtered": filtered,
                "filter_reason": reason_text,
                "length_unit": unit,
                "area_unit": f"{unit}^2",
                "um_per_px": float(scale) if scale else None,
            }
            objects.append(object_record)
            if not filtered:
                kept_mask[labels == label] = 255
            self._draw_object(canvas, contour, object_record, filtered)

        kept_objects = [item for item in objects if not item["filtered"]]
        filtered_objects = [item for item in objects if item["filtered"]]
        return {
            "objects": objects,
            "particles": kept_objects,
            "kept_mask": kept_mask,
            "object_overlay": canvas,
            "object_count": len(objects),
            "kept_object_count": len(kept_objects),
            "filtered_object_count": len(filtered_objects),
            "length_unit": unit,
            "area_unit": f"{unit}^2",
        }

    def summarize(
        self,
        mask: np.ndarray,
        um_per_px: float | None = None,
        objects: list[dict[str, Any]] | None = None,
        config: TraditionalSegConfig | None = None,
    ) -> dict[str, object]:
        binary = (mask > 0).astype(np.uint8)
        valid_pixels = int(binary.size)
        fg_pixels = int(np.count_nonzero(binary))
        volume_fraction = float(fg_pixels / valid_pixels) if valid_pixels else 0.0

        scale = um_per_px if um_per_px and um_per_px > 0 else None
        unit = "um" if scale else "px"
        if objects is None or not self._objects_match_scale(objects, scale):
            object_stats = self.build_object_stats(binary, um_per_px=scale, config=config)
            objects = object_stats["objects"]

        all_objects = list(objects or [])
        kept_objects = [item for item in all_objects if not bool(item.get("filtered"))]
        filtered_objects = [item for item in all_objects if bool(item.get("filtered"))]

        areas_units = [float(item["area_value"]) for item in kept_objects]
        diameters_units = [
            float(item["equiv_diameter"]) if "equiv_diameter" in item else float(item["size_value"])
            for item in kept_objects
        ]
        sizes_units = diameters_units
        perimeters_units = [float(item["perimeter_value"]) for item in kept_objects]
        majors_units = [float(item["major_value"]) for item in kept_objects]
        minors_units = [float(item["minor_value"]) for item in kept_objects]
        ferets_units = [float(item["feret_value"]) for item in kept_objects]
        minferets_units = [float(item["minferet_value"]) for item in kept_objects]
        aspect_ratios = [float(item["aspect_ratio"]) for item in kept_objects]
        circularities = [float(item["circularity"]) for item in kept_objects]
        roundness_values = [float(item["roundness"]) for item in kept_objects]
        solidity_values = [float(item["solidity"]) for item in kept_objects]

        # 空前景或全前景时不存在稳定通道，直接跳过，避免预览在极端阈值下卡死。
        if fg_pixels == 0 or fg_pixels == valid_pixels:
            channel_widths_x = []
            channel_widths_y = []
        else:
            # ------ 通道宽度：水平 / 垂直截线法 ------
            # W_x: 逐行统计被前景从左右夹住的背景段；W_y: 逐列统计被前景从上下夹住的背景段。
            channel_widths_x = self._collect_channel_widths(binary, axis="x", scale=scale)
            channel_widths_y = self._collect_channel_widths(binary, axis="y", scale=scale)
        channel_widths = list(channel_widths_x) + list(channel_widths_y)

        def safe_stat(values: list[float], fn, default: float = 0.0) -> float:
            clean = [float(value) for value in values if value is not None and np.isfinite(value)]
            return float(fn(clean)) if clean else default

        # ------ Lantuéjoul 加权均值 ------
        weights = [float(item.get("weight", 1.0)) for item in kept_objects]

        def weighted_mean(values: list[float], w: list[float]) -> float:
            """Lantuéjoul 偏置修正加权均值。"""
            pairs = [
                (v, wi) for v, wi in zip(values, w)
                if v is not None and np.isfinite(v) and wi is not None and np.isfinite(wi) and wi > 0
            ]
            if not pairs:
                return 0.0
            total_w = sum(wi for _, wi in pairs)
            return float(sum(v * wi for v, wi in pairs) / total_w) if total_w > 0 else 0.0

        particle_count = len(kept_objects)
        object_count = len(all_objects)
        filtered_count = len(filtered_objects)

        return {
            "volume_fraction": volume_fraction,
            "foreground_pixels": fg_pixels,
            "total_pixels": valid_pixels,
            "background_pixels": int(valid_pixels - fg_pixels),
            "object_count": object_count,
            "particle_count": particle_count,
            "filtered_object_count": filtered_count,
            "filtered_ratio": float(filtered_count / object_count) if object_count else 0.0,
            "area_unit": f"{unit}^2",
            "size_unit": unit,
            "length_unit": unit,
            "diameter_unit": unit,
            "perimeter_unit": unit,
            "channel_width_unit": unit,
            # --- 使用 Lantuéjoul 加权均值修正边界偏置 ---
            "mean_area": weighted_mean(areas_units, weights),
            "median_area": safe_stat(areas_units, np.median),
            "std_area": safe_stat(areas_units, np.std),
            "mean_size": weighted_mean(sizes_units, weights),
            "median_size": safe_stat(sizes_units, np.median),
            "std_size": safe_stat(sizes_units, np.std),
            "mean_diameter": weighted_mean(diameters_units, weights),
            "median_diameter": safe_stat(diameters_units, np.median),
            "std_diameter": safe_stat(diameters_units, np.std),
            "mean_perimeter": weighted_mean(perimeters_units, weights),
            "median_perimeter": safe_stat(perimeters_units, np.median),
            "std_perimeter": safe_stat(perimeters_units, np.std),
            "mean_major": weighted_mean(majors_units, weights),
            "median_major": safe_stat(majors_units, np.median),
            "std_major": safe_stat(majors_units, np.std),
            "mean_minor": weighted_mean(minors_units, weights),
            "median_minor": safe_stat(minors_units, np.median),
            "std_minor": safe_stat(minors_units, np.std),
            "mean_feret": weighted_mean(ferets_units, weights),
            "median_feret": safe_stat(ferets_units, np.median),
            "std_feret": safe_stat(ferets_units, np.std),
            "mean_minferet": weighted_mean(minferets_units, weights),
            "median_minferet": safe_stat(minferets_units, np.median),
            "std_minferet": safe_stat(minferets_units, np.std),
            "mean_aspect_ratio": weighted_mean(aspect_ratios, weights),
            "median_aspect_ratio": safe_stat(aspect_ratios, np.median),
            "std_aspect_ratio": safe_stat(aspect_ratios, np.std),
            "mean_circularity": weighted_mean(circularities, weights),
            "median_circularity": safe_stat(circularities, np.median),
            "std_circularity": safe_stat(circularities, np.std),
            "mean_roundness": weighted_mean(roundness_values, weights),
            "median_roundness": safe_stat(roundness_values, np.median),
            "std_roundness": safe_stat(roundness_values, np.std),
            "mean_solidity": weighted_mean(solidity_values, weights),
            "median_solidity": safe_stat(solidity_values, np.median),
            "std_solidity": safe_stat(solidity_values, np.std),
            # --- 通道宽度（水平 / 垂直截线）---
            "mean_channel_width_x": safe_stat(channel_widths_x, np.mean),
            "median_channel_width_x": safe_stat(channel_widths_x, np.median),
            "std_channel_width_x": safe_stat(channel_widths_x, np.std),
            "mean_channel_width_y": safe_stat(channel_widths_y, np.mean),
            "median_channel_width_y": safe_stat(channel_widths_y, np.median),
            "std_channel_width_y": safe_stat(channel_widths_y, np.std),
            "channel_widths": channel_widths,
            "channel_widths_x": channel_widths_x,
            "channel_widths_y": channel_widths_y,
            "channel_width_count_x": len(channel_widths_x),
            "channel_width_count_y": len(channel_widths_y),
            "channel_width_count": len(channel_widths),
            "areas": areas_units,
            "sizes": sizes_units,
            "diameters": diameters_units,
            "perimeters": perimeters_units,
            "majors": majors_units,
            "minors": minors_units,
            "ferets": ferets_units,
            "minferets": minferets_units,
            "aspect_ratios": aspect_ratios,
            "circularities": circularities,
            "roundnesses": roundness_values,
            "solidities": solidity_values,
            "objects": all_objects,
            "particles": kept_objects,
            "filtered_objects": filtered_objects,
        }

    def _objects_match_scale(self, objects: list[dict[str, Any]], scale: float | None) -> bool:
        if not objects:
            return True

        expected_unit = "um" if scale else "px"
        for item in objects:
            length_unit = item.get("length_unit")
            if length_unit is None:
                # Legacy object records were generated in pixels because they
                # carried no unit metadata.
                if scale:
                    return False
                continue
            if str(length_unit) != expected_unit:
                return False

            item_scale = item.get("um_per_px")
            if scale:
                if item_scale is None or not math.isclose(float(item_scale), float(scale), rel_tol=1e-9, abs_tol=1e-12):
                    return False
            elif item_scale not in (None, 0, 0.0):
                return False

        return True

    def _measure_object(
        self,
        contour: np.ndarray,
        area_px: int,
        centroid_x: float,
        centroid_y: float,
        scale: float | None,
    ) -> dict[str, float]:
        area_value = float(area_px * (scale**2)) if scale else float(area_px)
        perimeter_px = float(cv2.arcLength(contour, True)) if contour is not None and len(contour) >= 2 else 0.0
        major_px, minor_px = self._axis_lengths(contour, area_px)
        feret_px = self._feret_diameter(contour)
        minferet_px = self._min_feret(contour, major_px, minor_px)
        hull_area_px = self._hull_area(contour)
        solidity = float(area_px / hull_area_px) if hull_area_px > 0 else 0.0
        circularity = float(4.0 * math.pi * area_px / (perimeter_px**2)) if perimeter_px > 0 else 0.0
        roundness = float(4.0 * area_px / (math.pi * (major_px**2))) if major_px > 0 else 0.0
        aspect_ratio = float(major_px / minor_px) if minor_px > 0 else 0.0
        equiv_diameter_value = float(2.0 * math.sqrt(area_value / math.pi)) if area_value > 0 else 0.0

        if scale:
            perimeter_value = perimeter_px * scale
            major_value = major_px * scale
            minor_value = minor_px * scale
            feret_value = feret_px * scale
            minferet_value = minferet_px * scale
            size_value = equiv_diameter_value
        else:
            perimeter_value = perimeter_px
            major_value = major_px
            minor_value = minor_px
            feret_value = feret_px
            minferet_value = minferet_px
            size_value = equiv_diameter_value

        return {
            "area_value": float(area_value),
            "size_value": float(size_value),
            "equiv_diameter": float(equiv_diameter_value),
            "perimeter_value": float(perimeter_value),
            "major_value": float(major_value),
            "minor_value": float(minor_value),
            "feret_value": float(feret_value),
            "minferet_value": float(minferet_value),
            "aspect_ratio": float(aspect_ratio),
            "circularity": float(circularity),
            "roundness": float(roundness),
            "solidity": float(solidity),
            "centroid_x": float(centroid_x),
            "centroid_y": float(centroid_y),
        }

    def _filter_reasons(
        self,
        config: TraditionalSegConfig | None,
        area_px: int,
        bbox: tuple[int, int, int, int],
        image_shape: tuple[int, int],
        metrics: dict[str, float],
    ) -> list[str]:
        if config is None:
            return []
        reasons: list[str] = []
        x, y, w, h = bbox
        height, width = image_shape
        touches_border = x <= 0 or y <= 0 or x + w >= width or y + h >= height
        if config.remove_border and touches_border:
            reasons.append("border")
        if area_px < int(config.min_area):
            reasons.append("area_below_min")
        if config.max_area is not None and area_px > int(config.max_area):
            reasons.append("area_above_max")
        if config.min_solidity > 0 and metrics["solidity"] < float(config.min_solidity):
            reasons.append("solidity_below_min")
        if config.min_circularity > 0 and metrics["circularity"] < float(config.min_circularity):
            reasons.append("circularity_below_min")
        if config.min_roundness > 0 and metrics["roundness"] < float(config.min_roundness):
            reasons.append("roundness_below_min")
        if config.max_aspect_ratio is not None and metrics["aspect_ratio"] > float(config.max_aspect_ratio):
            reasons.append("aspect_ratio_above_max")
        return reasons

    def _axis_lengths(self, contour: np.ndarray | None, area_px: int) -> tuple[float, float]:
        if contour is None or len(contour) == 0:
            side = math.sqrt(max(float(area_px), 1.0))
            return side, side
        try:
            if len(contour) >= 5:
                (_, _), (w, h), _ = cv2.fitEllipse(contour)
                major_px = float(max(w, h))
                minor_px = float(min(w, h))
            else:
                (_, _), (w, h), _ = cv2.minAreaRect(contour)
                major_px = float(max(w, h))
                minor_px = float(min(w, h))
        except cv2.error:
            side = math.sqrt(max(float(area_px), 1.0))
            major_px = side
            minor_px = side
        if major_px <= 0 or not np.isfinite(major_px):
            major_px = math.sqrt(max(float(area_px), 1.0))
        if minor_px <= 0 or not np.isfinite(minor_px):
            minor_px = major_px
        if major_px < minor_px:
            major_px, minor_px = minor_px, major_px
        return float(major_px), float(minor_px)

    def _feret_diameter(self, contour: np.ndarray | None) -> float:
        if contour is None or len(contour) == 0:
            return 0.0
        hull = cv2.convexHull(contour, returnPoints=True).reshape(-1, 2)
        if hull.size == 0:
            return 0.0
        if len(hull) == 1:
            return 0.0
        if len(hull) == 2:
            return float(np.linalg.norm(hull[0] - hull[1]))
        diff = hull[:, None, :] - hull[None, :, :]
        distances = np.sqrt(np.sum(diff.astype(np.float32) ** 2, axis=2))
        return float(np.max(distances))

    def _min_feret(self, contour: np.ndarray | None, major_px: float, minor_px: float) -> float:
        if contour is None or len(contour) == 0:
            return 0.0
        hull = cv2.convexHull(contour, returnPoints=True)
        if hull is None or len(hull) == 0:
            return float(min(major_px, minor_px))
        try:
            rect = cv2.minAreaRect(hull)
            width, height = rect[1]
            candidate = min(float(width), float(height))
            if candidate > 0:
                return candidate
        except cv2.error:
            pass
        return float(min(major_px, minor_px))

    def _hull_area(self, contour: np.ndarray | None) -> float:
        if contour is None or len(contour) == 0:
            return 0.0
        try:
            hull = cv2.convexHull(contour)
            return float(cv2.contourArea(hull))
        except cv2.error:
            return 0.0

    def _extract_largest_contour(self, component_mask: np.ndarray) -> np.ndarray | None:
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        return max(contours, key=cv2.contourArea)

    def _prepare_canvas(self, base_image: np.ndarray | None, binary: np.ndarray) -> np.ndarray:
        if base_image is None:
            canvas = cv2.cvtColor((binary * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        else:
            canvas = np.asarray(base_image).copy()
            if canvas.ndim == 2:
                canvas = cv2.cvtColor(canvas.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            elif canvas.ndim == 3 and canvas.shape[2] == 4:
                canvas = cv2.cvtColor(canvas.astype(np.uint8), cv2.COLOR_BGRA2BGR)
            elif canvas.ndim == 3 and canvas.shape[2] == 3:
                canvas = canvas.astype(np.uint8)
            else:
                canvas = cv2.cvtColor((binary * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        if canvas.dtype != np.uint8:
            if np.issubdtype(canvas.dtype, np.floating):
                canvas = np.clip(canvas, 0.0, 1.0 if float(canvas.max() or 0) <= 1.0 else 255.0)
                if float(canvas.max() or 0) <= 1.0:
                    canvas = (canvas * 255.0).astype(np.uint8)
                else:
                    canvas = canvas.astype(np.uint8)
            else:
                canvas = np.clip(canvas, 0, 255).astype(np.uint8)
        return canvas

    def _draw_object(self, canvas: np.ndarray, contour: np.ndarray | None, obj: dict[str, Any], filtered: bool) -> None:
        if contour is None or len(contour) == 0:
            return
        color = (0, 64, 255) if filtered else (0, 200, 0)
        cv2.drawContours(canvas, [contour], -1, color, 1)
        cx = int(round(float(obj.get("centroid_x", 0.0))))
        cy = int(round(float(obj.get("centroid_y", 0.0))))
        label = str(obj.get("label", ""))
        if label:
            cv2.putText(canvas, label, (cx + 2, cy + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(canvas, label, (cx + 2, cy + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    def _channel_widths_distance_transform(
        self, binary: np.ndarray, scale: float | None = None
    ) -> list[float]:
        """基于欧氏距离变换 + 骨架中轴线提取通道宽度（旋转不变）。

        原理：
        1. 对背景通道（binary == 0）做距离变换，得到每个通道像素到最近前景边缘的距离。
        2. 提取通道骨架（中轴线），骨架点上的距离值即为该处通道的半宽。
        3. 半宽 × 2 = 局部通道宽度（法向距离，旋转不变）。
        """
        if binary.size == 0:
            return []

        # 通道掩码：将前景 (>0) 取反
        channel_mask = ((binary == 0) * 255).astype(np.uint8)
        if cv2.countNonZero(channel_mask) == 0:
            return []

        # 欧氏距离变换
        dist = cv2.distanceTransform(channel_mask, cv2.DIST_L2, 5)

        # 骨架化
        skeleton = _skeletonize(channel_mask)

        # 在骨架点上采样半宽
        skeleton_coords = np.argwhere(skeleton > 0)  # (row, col)
        if skeleton_coords.size == 0:
            return []

        half_widths = dist[skeleton_coords[:, 0], skeleton_coords[:, 1]]
        # 过滤掉距离为 0 的点（骨架端点等退化情况）
        valid = half_widths[half_widths > 0.5]
        if valid.size == 0:
            return []

        widths_px = (valid * 2.0).tolist()
        if scale and scale > 0:
            return [float(w * scale) for w in widths_px]
        return [float(w) for w in widths_px]

    def _collect_channel_widths(self, binary: np.ndarray, axis: str, scale: float | None = None) -> list[float]:
        """按水平/垂直截线统计被前景夹住的背景通道宽度。"""
        if binary.size == 0:
            return []

        normalized = (binary > 0).astype(np.uint8)
        if axis == "x":
            lines = normalized
        elif axis == "y":
            lines = normalized.T
        else:
            raise ValueError(f"未知通道宽度方向: {axis}")

        widths_px: list[int] = []
        for line in lines:
            widths_px.extend(self._interior_zero_runs(line))

        if scale and scale > 0:
            return [float(width * scale) for width in widths_px]
        return [float(width) for width in widths_px]

    def _interior_zero_runs(self, line: np.ndarray) -> list[int]:
        """返回一条截线上被前景从两侧夹住的背景段长度。"""
        values = [int(item) for item in np.asarray(line).ravel()]
        if len(values) < 3:
            return []

        runs: list[tuple[int, int]] = []
        current_value = values[0]
        run_length = 1
        for value in values[1:]:
            if value == current_value:
                run_length += 1
            else:
                runs.append((current_value, run_length))
                current_value = value
                run_length = 1
        runs.append((current_value, run_length))

        gaps: list[int] = []
        for index in range(1, len(runs) - 1):
            value, length = runs[index]
            if value == 0 and runs[index - 1][0] == 1 and runs[index + 1][0] == 1:
                gaps.append(length)
        return gaps


statistics_service = StatisticsService()
