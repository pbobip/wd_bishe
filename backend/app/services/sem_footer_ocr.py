from __future__ import annotations

import re
from dataclasses import dataclass

import cv2
import numpy as np

try:
    from rapidocr_onnxruntime import RapidOCR
except Exception:  # pragma: no cover - optional dependency
    RapidOCR = None


@dataclass
class OcrTextBox:
    text: str
    confidence: float
    bbox: tuple[float, float, float, float]


@dataclass
class SemFooterOcrResult:
    raw_texts: list[OcrTextBox]
    scale_bar_um: float | None
    fov_um: float | None
    magnification_text: str | None
    wd_mm: float | None
    detector: str | None
    scan_mode: str | None
    vacuum_mode: str | None
    date_text: str | None
    time_text: str | None


class SemFooterOcrService:
    def __init__(self) -> None:
        self._ocr = RapidOCR() if RapidOCR is not None else None

    @property
    def available(self) -> bool:
        return self._ocr is not None

    def analyze(
        self,
        footer_image: np.ndarray | None,
        scale_bar_hint_bbox: tuple[int, int, int, int] | None = None,
    ) -> SemFooterOcrResult | None:
        if not self.available or footer_image is None or footer_image.size == 0:
            return None

        raw_boxes = self._recognize(footer_image)
        enhanced_boxes = self._recognize(self._preprocess(footer_image))
        boxes = self._merge_boxes(raw_boxes, enhanced_boxes)
        if not boxes:
            boxes = self._recognize_region_variants(footer_image)
            if not boxes:
                return None

        footer_height, footer_width = footer_image.shape[:2]
        initial_result = SemFooterOcrResult(
            raw_texts=boxes,
            scale_bar_um=self._extract_scale_bar_um(
                boxes,
                footer_width,
                footer_height,
                scale_bar_hint_bbox,
            ),
            fov_um=self._extract_fov_um(boxes, footer_width),
            magnification_text=self._extract_magnification_text(boxes),
            wd_mm=self._extract_numeric(boxes, r"(\d+(?:\.\d+)?)\s*mm", lambda box: self._center(box)[0] > footer_width * 0.45),
            detector=self._extract_exact(boxes, {"E-T", "ET", "BSE", "SE"}),
            scan_mode=self._extract_text(boxes, lambda text: text.upper() in {"ANALYSIS", "SPOT", "LINE", "MAP"}),
            vacuum_mode=self._extract_text(boxes, lambda text: "vac" in text.lower()),
            date_text=self._extract_text(boxes, lambda text: re.fullmatch(r"\d{2}/\d{2}/\d{2,4}", text) is not None),
            time_text=self._extract_text(boxes, lambda text: re.fullmatch(r"\d{2}:\d{2}:\d{2}", text) is not None),
        )

        needs_regional_retry = (
            footer_height <= 110
            or initial_result.scale_bar_um is None
            or initial_result.fov_um is None
            or initial_result.magnification_text is None
        )
        if not needs_regional_retry:
            return initial_result

        regional_boxes = self._recognize_region_variants(footer_image)
        boxes = self._merge_boxes(boxes, regional_boxes)
        if not boxes:
            return initial_result

        return SemFooterOcrResult(
            raw_texts=boxes,
            scale_bar_um=self._extract_scale_bar_um(
                boxes,
                footer_width,
                footer_height,
                scale_bar_hint_bbox,
            ),
            fov_um=self._extract_fov_um(boxes, footer_width),
            magnification_text=self._extract_magnification_text(boxes),
            wd_mm=self._extract_numeric(boxes, r"(\d+(?:\.\d+)?)\s*mm", lambda box: self._center(box)[0] > footer_width * 0.45),
            detector=self._extract_exact(boxes, {"E-T", "ET", "BSE", "SE"}),
            scan_mode=self._extract_text(boxes, lambda text: text.upper() in {"ANALYSIS", "SPOT", "LINE", "MAP"}),
            vacuum_mode=self._extract_text(boxes, lambda text: "vac" in text.lower()),
            date_text=self._extract_text(boxes, lambda text: re.fullmatch(r"\d{2}/\d{2}/\d{2,4}", text) is not None),
            time_text=self._extract_text(boxes, lambda text: re.fullmatch(r"\d{2}:\d{2}:\d{2}", text) is not None),
        )

    def _recognize(self, image: np.ndarray) -> list[OcrTextBox]:
        if self._ocr is None:
            return []
        result, _ = self._ocr(image)
        if not result:
            return []

        boxes: list[OcrTextBox] = []
        for item in result:
            quad, text, confidence = item
            if not text:
                continue
            xs = [point[0] for point in quad]
            ys = [point[1] for point in quad]
            boxes.append(
                OcrTextBox(
                    text=str(text).strip(),
                    confidence=float(confidence),
                    bbox=(min(xs), min(ys), max(xs), max(ys)),
                )
            )
        return boxes

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        upscaled = cv2.resize(image, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        blurred = cv2.GaussianBlur(upscaled, (3, 3), 0)
        _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresholded

    def _merge_boxes(self, *groups: list[OcrTextBox]) -> list[OcrTextBox]:
        merged: dict[str, OcrTextBox] = {}
        for group in groups:
            for box in group:
                key = box.text.lower()
                if key not in merged or box.confidence > merged[key].confidence:
                    merged[key] = box
        return sorted(merged.values(), key=lambda item: (item.bbox[1], item.bbox[0]))

    def _recognize_region_variants(self, footer_image: np.ndarray) -> list[OcrTextBox]:
        footer_height, footer_width = footer_image.shape[:2]
        crops: list[tuple[str, tuple[int, int, int, int], int]] = [
            ("top_left", (0, 0, int(footer_width * 0.6), max(28, int(footer_height * 0.55))), 6),
            ("top_right", (int(footer_width * 0.42), 0, footer_width, max(36, int(footer_height * 0.65))), 8),
            ("bottom_left", (0, int(footer_height * 0.25), int(footer_width * 0.7), footer_height), 6),
        ]
        if footer_height > 110:
            crops.append(("wide_full", (0, 0, footer_width, footer_height), 4))

        boxes: list[OcrTextBox] = []
        for _, (x1, y1, x2, y2), upscale in crops:
            crop = footer_image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            boxes.extend(self._recognize_crop_variants(crop, x1, y1, upscale))
        return boxes

    def _recognize_crop_variants(
        self,
        crop: np.ndarray,
        offset_x: int,
        offset_y: int,
        upscale: int,
    ) -> list[OcrTextBox]:
        upscaled = cv2.resize(crop, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_image = clahe.apply(upscaled)
        binary = cv2.threshold(
            cv2.GaussianBlur(clahe_image, (3, 3), 0),
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )[1]
        variants = [upscaled, clahe_image, binary, 255 - binary]

        boxes: list[OcrTextBox] = []
        for variant in variants:
            for box in self._recognize(variant):
                x1, y1, x2, y2 = box.bbox
                boxes.append(
                    OcrTextBox(
                        text=box.text,
                        confidence=box.confidence,
                        bbox=(
                            x1 / upscale + offset_x,
                            y1 / upscale + offset_y,
                            x2 / upscale + offset_x,
                            y2 / upscale + offset_y,
                        ),
                    )
                )
        return boxes

    def _extract_scale_bar_um(
        self,
        boxes: list[OcrTextBox],
        footer_width: int,
        footer_height: int,
        scale_bar_hint_bbox: tuple[int, int, int, int] | None,
    ) -> float | None:
        candidates: list[tuple[float, float]] = []
        hint_center = None
        if scale_bar_hint_bbox is not None:
            x, y, width, height = scale_bar_hint_bbox
            hint_center = (x + width / 2.0, y + height / 2.0)
        for box in boxes:
            matches = list(re.finditer(r"(\d+(?:\.\d+)?)\s*[uμ]m", box.text, re.IGNORECASE))
            if not matches:
                continue
            center_x, center_y = self._center(box)
            for order, match in enumerate(matches):
                score = box.confidence + order * 0.2
                if len(matches) > 1 and order == len(matches) - 1:
                    score += 0.2
                if center_x > footer_width * 0.55:
                    score += 0.5
                if 0.08 * footer_height <= center_y <= 0.75 * footer_height:
                    score += 0.2
                if hint_center is not None:
                    horizontal_gap = abs(center_x - hint_center[0])
                    vertical_gap = abs(center_y - hint_center[1])
                    if horizontal_gap <= footer_width * 0.22:
                        score += 0.75
                    elif horizontal_gap <= footer_width * 0.34:
                        score += 0.35
                    if vertical_gap <= footer_height * 0.22:
                        score += 0.25
                value = float(match.group(1))
                normalized_value = self._snap_scale_value(value)
                if normalized_value is not None:
                    score += 0.15
                    value = normalized_value
                candidates.append((score, value))
        if not candidates:
            return None
        return max(candidates, key=lambda item: item[0])[1]

    def _extract_fov_um(self, boxes: list[OcrTextBox], footer_width: int) -> float | None:
        keyword_boxes = [
            box
            for box in boxes
            if "fov" in box.text.lower() or "view field" in box.text.lower()
        ]
        value_candidates: list[tuple[float, float]] = []
        for box in boxes:
            match = re.search(r"(\d+(?:\.\d+)?)\s*[uμ]m", box.text, re.IGNORECASE)
            if not match:
                continue
            center_x, center_y = self._center(box)
            score = box.confidence
            if center_x < footer_width * 0.68:
                score += 0.2
            for keyword in keyword_boxes:
                keyword_center_x, keyword_center_y = self._center(keyword)
                if abs(center_y - keyword_center_y) <= 20:
                    distance = abs(center_x - keyword_center_x)
                    if distance <= footer_width * 0.18:
                        score += 0.8
                    elif distance <= footer_width * 0.3:
                        score += 0.4
            value_candidates.append((score, float(match.group(1))))

        if not value_candidates:
            return None
        return max(value_candidates, key=lambda item: item[0])[1]

    def _extract_numeric(self, boxes: list[OcrTextBox], pattern: str, box_filter) -> float | None:
        candidates: list[tuple[float, float]] = []
        for box in boxes:
            if not box_filter(box):
                continue
            match = re.search(pattern, box.text, re.IGNORECASE)
            if match:
                candidates.append((box.confidence, float(match.group(1))))
        if not candidates:
            return None
        return max(candidates, key=lambda item: item[0])[1]

    def _extract_text(self, boxes: list[OcrTextBox], matcher) -> str | None:
        candidates = [box for box in boxes if matcher(box.text)]
        if not candidates:
            return None
        return max(candidates, key=lambda item: item.confidence).text

    def _extract_magnification_text(self, boxes: list[OcrTextBox]) -> str | None:
        candidates: list[tuple[float, str]] = []
        for box in boxes:
            match = re.search(r"(\d+(?:\.\d+)?)\s*kx", box.text, re.IGNORECASE)
            if match:
                candidates.append((box.confidence, f"{match.group(1)} kx"))
        if not candidates:
            return None
        return max(candidates, key=lambda item: item[0])[1]

    def _extract_exact(self, boxes: list[OcrTextBox], allowed: set[str]) -> str | None:
        normalized = {item.upper(): item for item in allowed}
        candidates = []
        for box in boxes:
            text = box.text.strip().upper()
            if text in normalized:
                candidates.append(box)
        if not candidates:
            return None
        return max(candidates, key=lambda item: item.confidence).text

    def _center(self, box: OcrTextBox) -> tuple[float, float]:
        x1, y1, x2, y2 = box.bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def _snap_scale_value(self, value: float) -> float | None:
        canonical = [0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
        best = min(canonical, key=lambda candidate: abs(candidate - value))
        if abs(best - value) / max(best, 1e-6) <= 0.18:
            return float(best)
        return None


sem_footer_ocr_service = SemFooterOcrService()
