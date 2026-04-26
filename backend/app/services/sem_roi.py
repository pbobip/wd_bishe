from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SemRoiResult:
    analysis_image: np.ndarray
    footer_image: np.ndarray | None
    analysis_bbox: tuple[int, int, int, int]
    footer_bbox: tuple[int, int, int, int] | None
    source_shape: tuple[int, int]
    cropped_footer: bool
    cropped_background: bool


@dataclass
class ScaleBarHint:
    pixel_length: int
    bbox: tuple[int, int, int, int]
    confidence: float


@dataclass
class AnnotationBarHint:
    side: str
    bbox: tuple[int, int, int, int]
    score: float


class SemRoiService:
    MIN_SCALE_BAR_CONFIDENCE = 0.55
    MIN_ANNOTATION_BAR_SCORE = 2.2

    def extract(self, image: np.ndarray) -> SemRoiResult:
        height, width = image.shape[:2]
        annotation_bar = self._detect_annotation_bar(image)
        roi, footer_image, footer_bbox = self._split_image(image, annotation_bar)
        left, top, right, bottom = self._detect_content_bbox(roi)

        analysis_image = roi[top : bottom + 1, left : right + 1].copy()
        analysis_bbox = (left, top, right - left + 1, bottom - top + 1)

        return SemRoiResult(
            analysis_image=analysis_image,
            footer_image=footer_image,
            analysis_bbox=analysis_bbox,
            footer_bbox=footer_bbox,
            source_shape=(height, width),
            cropped_footer=annotation_bar is not None,
            cropped_background=analysis_bbox != (0, 0, roi.shape[1], roi.shape[0]),
        )

    def detect_scale_bar(self, footer_image: np.ndarray | None) -> ScaleBarHint | None:
        if footer_image is None or footer_image.size == 0:
            return None

        blurred = cv2.GaussianBlur(footer_image, (3, 3), 0)
        threshold_value = int(max(160, np.percentile(blurred, 88)))
        _, thresholded = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)
        thresholded = cv2.morphologyEx(
            thresholded,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1)),
        )

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(thresholded, connectivity=8)
        best: ScaleBarHint | None = None
        footer_height, footer_width = footer_image.shape[:2]

        for label in range(1, num_labels):
            x, y, width, height, area = stats[label]
            if width < 40 or height <= 0 or area <= 0:
                continue

            aspect_ratio = float(width / max(height, 1))
            normalized_y = y / max(footer_height, 1)
            normalized_x = x / max(footer_width, 1)
            width_ratio = width / max(footer_width, 1)
            score = aspect_ratio

            if aspect_ratio < 8:
                continue
            if width < max(56, int(footer_width * 0.06)):
                continue
            if height > max(12, int(footer_height * 0.16)):
                continue
            if height > 16:
                score *= 0.75
            if normalized_x < 0.4:
                score *= 0.7
            if normalized_y < 0.1 or normalized_y > 0.75:
                score *= 0.7
            if width_ratio < 0.12:
                score *= 0.6
            if normalized_x > 0.92:
                score *= 0.85

            candidate = ScaleBarHint(
                pixel_length=int(width),
                bbox=(int(x), int(y), int(width), int(height)),
                confidence=min(1.0, round(score / 45.0, 3)),
            )
            if best is None or (candidate.confidence, candidate.pixel_length) > (
                best.confidence,
                best.pixel_length,
            ):
                best = candidate

        if best is None or best.confidence < self.MIN_SCALE_BAR_CONFIDENCE:
            return None
        return best

    def _detect_annotation_bar(self, image: np.ndarray) -> AnnotationBarHint | None:
        height, width = image.shape[:2]
        candidates: list[AnnotationBarHint] = []

        bottom_candidate = self._detect_horizontal_bar(image)
        if bottom_candidate is not None:
            candidates.append(bottom_candidate)
        else:
            bottom_fallback = self._detect_side_band_fallback(image, side="bottom")
            if bottom_fallback is not None:
                candidates.append(bottom_fallback)

        flipped_top = self._detect_horizontal_bar(np.flipud(image))
        if flipped_top is not None:
            _, _, _, bar_height = flipped_top.bbox
            candidates.append(
                AnnotationBarHint(
                    side="top",
                    bbox=(0, 0, width, bar_height),
                    score=round(flipped_top.score * 0.75, 4),
                )
            )
        else:
            top_fallback = self._detect_side_band_fallback(image, side="top")
            if top_fallback is not None:
                candidates.append(top_fallback)

        if not candidates:
            return None

        bottom_only = [item for item in candidates if item.side == "bottom"]
        if bottom_only:
            best = max(bottom_only, key=lambda item: item.score)
        else:
            best = max(candidates, key=lambda item: item.score)
        if best.score < self.MIN_ANNOTATION_BAR_SCORE:
            return None
        return best

    def _detect_horizontal_bar(self, image: np.ndarray) -> AnnotationBarHint | None:
        height, width = image.shape[:2]
        if height < 120 or width < 120:
            return None

        image_f = image.astype(np.float32)
        row_mean = cv2.GaussianBlur(image_f.mean(axis=1).reshape(-1, 1), (1, 11), 0).ravel()
        row_std = cv2.GaussianBlur(image_f.std(axis=1).reshape(-1, 1), (1, 11), 0).ravel()
        dark_threshold = float(min(110.0, np.percentile(image_f, 35)))
        row_dark = cv2.GaussianBlur(
            (image_f < dark_threshold).mean(axis=1).astype(np.float32).reshape(-1, 1),
            (1, 11),
            0,
        ).ravel()
        row_grad = np.abs(np.diff(image.astype(np.int16), axis=0)).mean(axis=1).astype(np.float32)
        row_grad = cv2.GaussianBlur(row_grad.reshape(-1, 1), (1, 11), 0).ravel()

        search_start = int(height * 0.55)
        min_bar_height = max(60, int(height * 0.08))
        max_bar_height = int(height * 0.28)
        search_end = height - min_bar_height
        if search_end <= search_start:
            return None

        baseline_mean = float(np.median(row_mean[:search_start]))
        baseline_std = float(np.median(row_std[:search_start]))
        baseline_dark = float(np.median(row_dark[:search_start]))
        grad_ref = float(max(np.percentile(row_grad[search_start:search_end], 90), 1.0))

        candidates: list[AnnotationBarHint] = []
        for index in range(search_start, search_end):
            bar_height = height - index
            if bar_height < min_bar_height or bar_height > max_bar_height:
                continue

            head_end = min(height, index + min(90, bar_height))
            future_slice = slice(index + 2, head_end)
            if future_slice.start >= future_slice.stop:
                continue

            full_slice = slice(index + 2, height)
            lower_mean = float(np.median(row_mean[full_slice]))
            lower_std = float(np.median(row_std[full_slice]))
            lower_dark = float(np.median(row_dark[full_slice]))
            head_mean = float(np.median(row_mean[future_slice]))
            head_dark = float(np.median(row_dark[future_slice]))

            mean_drop = max(0.0, (baseline_mean - lower_mean) / max(baseline_mean, 1.0))
            std_drop = max(0.0, (baseline_std - lower_std) / max(baseline_std, 1.0))
            dark_gain = max(0.0, lower_dark - baseline_dark)
            head_drop = max(0.0, (baseline_mean - head_mean) / max(baseline_mean, 1.0))
            head_dark_gain = max(0.0, head_dark - baseline_dark)
            grad_score = max(0.0, float(row_grad[index]) / grad_ref)

            bar_ratio = bar_height / max(height, 1)
            if 0.11 <= bar_ratio <= 0.24:
                height_score = 1.0
            elif 0.08 <= bar_ratio <= 0.28:
                height_score = 0.65
            else:
                height_score = 0.0

            score = (
                1.5 * mean_drop
                + 1.2 * dark_gain
                + 0.8 * std_drop
                + 0.6 * head_drop
                + 0.4 * head_dark_gain
                + 0.9 * grad_score
                + 0.5 * height_score
            )
            if mean_drop < 0.08 and dark_gain < 0.06:
                continue

            candidate = AnnotationBarHint(
                side="bottom",
                bbox=(0, int(index), width, int(bar_height)),
                score=round(score, 4),
            )
            candidates.append(candidate)

        if not candidates:
            return None

        best_score = max(item.score for item in candidates)
        near_best = [item for item in candidates if item.score >= best_score * 0.9]
        return min(near_best, key=lambda item: item.bbox[1])

    def _detect_side_band_fallback(self, image: np.ndarray, side: str) -> AnnotationBarHint | None:
        height, width = image.shape[:2]
        ratios = [0.09, 0.11, 0.13, 0.15, 0.18, 0.2, 0.22, 0.25, 0.28]
        best: AnnotationBarHint | None = None

        for ratio in ratios:
            band_height = max(60, int(height * ratio))
            if band_height >= height:
                continue
            if side == "bottom":
                y0 = height - band_height
                band = image[y0:, :]
            else:
                y0 = 0
                band = image[:band_height, :]

            scale = self.detect_scale_bar(band)
            edge_density = float((cv2.Canny(band, 60, 160) > 0).mean())
            dark_ratio = float((band < max(110, np.percentile(band, 35))).mean())

            score = 0.0
            if scale is not None:
                score += 1.3
                score += min(0.6, scale.pixel_length / max(width, 1))
                if scale.confidence >= 0.9:
                    score += 0.25
            score += min(0.35, edge_density * 4.5)
            score += min(0.35, dark_ratio * 0.7)

            if side == "top":
                score *= 0.72

            if score < 1.6:
                continue

            candidate = AnnotationBarHint(
                side=side,
                bbox=(0, int(y0), int(width), int(band_height)),
                score=round(score, 4),
            )
            if best is None or candidate.score > best.score:
                best = candidate

        return best

    def _split_image(
        self,
        image: np.ndarray,
        annotation_bar: AnnotationBarHint | None,
    ) -> tuple[np.ndarray, np.ndarray | None, tuple[int, int, int, int] | None]:
        height, width = image.shape[:2]
        if annotation_bar is None:
            return image, None, None

        _, y, _, bar_height = annotation_bar.bbox
        if annotation_bar.side == "bottom":
            roi = image[:y, :]
            footer = image[y:, :]
            footer_bbox = (0, y, width, height - y)
            return roi, footer.copy(), footer_bbox

        if annotation_bar.side == "top":
            roi = image[bar_height:, :]
            footer = image[:bar_height, :]
            footer_bbox = (0, 0, width, bar_height)
            return roi, footer.copy(), footer_bbox

        return image, None, None

    def _detect_content_bbox(self, roi: np.ndarray) -> tuple[int, int, int, int]:
        height, width = roi.shape[:2]
        if height == 0 or width == 0:
            return 0, 0, max(0, width - 1), max(0, height - 1)

        edges = cv2.Canny(roi, 30, 120)
        edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
        points = np.column_stack(np.where(edges > 0))

        if points.size == 0:
            return 0, 0, width - 1, height - 1

        top, left = points.min(axis=0)
        bottom, right = points.max(axis=0)
        pad = 8
        left = max(0, int(left) - pad)
        top = max(0, int(top) - pad)
        right = min(width - 1, int(right) + pad)
        bottom = min(height - 1, int(bottom) + pad)

        min_keep_width = int(width * 0.75)
        min_keep_height = int(height * 0.75)
        if right - left + 1 < min_keep_width:
            left, right = 0, width - 1
        if bottom - top + 1 < min_keep_height:
            top, bottom = 0, height - 1

        return left, top, right, bottom


sem_roi_service = SemRoiService()
