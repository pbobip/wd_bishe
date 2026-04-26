from __future__ import annotations

import cv2
import numpy as np
import pytest
import sys
import types

sem_footer_ocr_stub = types.ModuleType("backend.app.services.sem_footer_ocr")


class _DummySemFooterOcrService:
    available = False

    def analyze(self, *args: object, **kwargs: object) -> None:
        return None


sem_footer_ocr_stub.sem_footer_ocr_service = _DummySemFooterOcrService()
sys.modules.setdefault("backend.app.services.sem_footer_ocr", sem_footer_ocr_stub)

from backend.app.models.entities import ImageAsset
from backend.app.schemas.run import RunCreate
from backend.app.services.pipeline import pipeline_service
from backend.app.services.statistics import statistics_service


def test_summarize_rebuilds_pixel_objects_when_um_scale_is_provided() -> None:
    mask = np.zeros((12, 12), dtype=np.uint8)
    cv2.rectangle(mask, (4, 4), (7, 7), 255, thickness=-1)
    pixel_objects = statistics_service.build_object_stats(mask)["objects"]

    assert pixel_objects[0]["area_value"] == pytest.approx(16.0)
    assert pixel_objects[0]["size_value"] == pytest.approx(2.0 * np.sqrt(16.0 / np.pi))

    summary = statistics_service.summarize(mask, um_per_px=0.5, objects=pixel_objects)

    assert summary["area_unit"] == "um^2"
    assert summary["size_unit"] == "um"
    assert summary["areas"] == pytest.approx([4.0])
    assert summary["sizes"] == pytest.approx([2.0 * np.sqrt(4.0 / np.pi)])
    assert summary["mean_area"] == pytest.approx(4.0)
    assert summary["mean_size"] == pytest.approx(2.0 * np.sqrt(4.0 / np.pi))
    assert summary["objects"][0]["area_value"] == pytest.approx(4.0)
    assert summary["objects"][0]["size_value"] == pytest.approx(2.0 * np.sqrt(4.0 / np.pi))


def test_channel_width_uses_horizontal_and_vertical_scanlines() -> None:
    vertical_stripes = np.zeros((10, 12), dtype=np.uint8)
    vertical_stripes[:, 1:3] = 255
    vertical_stripes[:, 7:9] = 255

    vertical_summary = statistics_service.summarize(vertical_stripes)

    assert vertical_summary["channel_widths_x"] == pytest.approx([4.0] * 10)
    assert vertical_summary["channel_widths_y"] == []
    assert vertical_summary["mean_channel_width_x"] == pytest.approx(4.0)
    assert vertical_summary["mean_channel_width_y"] == pytest.approx(0.0)

    horizontal_stripes = np.zeros((12, 10), dtype=np.uint8)
    horizontal_stripes[1:3, :] = 255
    horizontal_stripes[8:10, :] = 255

    horizontal_summary = statistics_service.summarize(horizontal_stripes, um_per_px=0.5)

    assert horizontal_summary["channel_widths_x"] == []
    assert horizontal_summary["channel_widths_y"] == pytest.approx([2.5] * 10)
    assert horizontal_summary["mean_channel_width_x"] == pytest.approx(0.0)
    assert horizontal_summary["mean_channel_width_y"] == pytest.approx(2.5)
    assert horizontal_summary["channel_width_unit"] == "um"


@pytest.mark.parametrize(
    ("calibration_item", "image_relative_path", "image_original_name"),
    [
        ({"relative_path": "runs/1/input/a.png", "um_per_px": 0.25}, "runs/1/input/a.png", "raw-a.png"),
        (
            {"relative_path": "unused-key", "original_name": "raw-b.png", "um_per_px": 0.5},
            "runs/1/input/b.png",
            "raw-b.png",
        ),
        ({"relative_path": "uploaded/c.png", "um_per_px": 0.75}, "runs/1/input/c.png", "raw-c.png"),
    ],
)
def test_image_calibration_matches_relative_path_original_name_or_basename(
    calibration_item: dict[str, object],
    image_relative_path: str,
    image_original_name: str,
) -> None:
    config = RunCreate(
        name="calibration-match",
        input_mode="batch",
        segmentation_mode="traditional",
        input_config={"image_calibrations": [calibration_item]},
    )
    image = ImageAsset(
        original_name=image_original_name,
        relative_path=image_relative_path,
        sort_index=0,
    )

    calibration_map = pipeline_service._build_image_calibration_map(config)
    resolved = pipeline_service._resolve_image_calibration(image, calibration_map, config)

    assert resolved["calibrated"] is True
    assert resolved["source"] == "image_override"
    assert resolved["um_per_px"] == pytest.approx(float(calibration_item["um_per_px"]))
