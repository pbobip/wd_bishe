from __future__ import annotations

from pathlib import Path
import sys
import types
from typing import Any, Iterator

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

sem_footer_ocr_stub = types.ModuleType("backend.app.services.sem_footer_ocr")


class _DummySemFooterOcrService:
    available = False

    def analyze(self, *args: Any, **kwargs: Any) -> None:
        return None


sem_footer_ocr_stub.sem_footer_ocr_service = _DummySemFooterOcrService()
sys.modules.setdefault("backend.app.services.sem_footer_ocr", sem_footer_ocr_stub)

from backend.app.api.routes import router
from backend.app.db.base import Base
from backend.app.db.session import get_db
from backend.app.models.entities import ExportRecord, ImageAsset, MetricRecord, RunTask
from backend.app.schemas.run import RunCreate, TraditionalSegConfig
from backend.app.services.algorithms.traditional import traditional_service
from backend.app.services.pipeline import pipeline_service
from backend.app.services.statistics import statistics_service
from backend.app.services.storage import storage_service
from backend.app.utils.image_io import read_gray, write_image


@pytest.fixture()
def api_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[TestClient, sessionmaker]]:
    storage_root = tmp_path / "storage"
    runs_root = storage_root / "runs"
    exports_root = storage_root / "exports"
    runs_root.mkdir(parents=True, exist_ok=True)
    exports_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(storage_service, "root", storage_root)
    monkeypatch.setattr(storage_service, "runs_root", runs_root)
    monkeypatch.setattr(storage_service, "exports_root", exports_root)

    engine = create_engine(
        f"sqlite:///{(tmp_path / 'test.db').as_posix()}",
        connect_args={"check_same_thread": False},
        future=True,
    )
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    Base.metadata.create_all(bind=engine)

    app = FastAPI()
    app.mount("/static", StaticFiles(directory=storage_root), name="static")
    app.include_router(router, prefix="/api")

    def override_get_db() -> Iterator[Session]:
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    try:
        yield client, SessionLocal
    finally:
        client.close()
        engine.dispose()


def _build_metric_summary(
    image_name: str,
    result: dict[str, Any],
    config: TraditionalSegConfig,
    um_per_px: float | None = None,
) -> dict[str, Any]:
    summary = statistics_service.summarize(
        result["mask"],
        um_per_px=um_per_px,
        objects=result["objects"],
        config=config,
    )
    summary.update(
        {
            "traditional_method": config.method,
            "traditional_route_details": result.get("route_details", {}),
            "applied_um_per_px": um_per_px,
            "calibrated": bool(um_per_px),
            "calibration_source": "task_default" if um_per_px else "pixels_only",
            "image_name": image_name,
        }
    )
    return summary


@pytest.fixture()
def seeded_run(api_env: tuple[TestClient, sessionmaker]) -> dict[str, Any]:
    client, SessionLocal = api_env
    with SessionLocal() as db:
        config = RunCreate(
            name="postprocess-preview",
            input_mode="single",
            segmentation_mode="traditional",
            input_config={"um_per_px": 0.5},
            traditional_seg=TraditionalSegConfig(
                fill_holes=False,
                watershed=False,
                boundary_smoothing=False,
                open_kernel=1,
                close_kernel=1,
                remove_border=False,
            ),
        )
        run = RunTask(
            name=config.name,
            input_mode=config.input_mode,
            segmentation_mode=config.segmentation_mode,
            status="completed",
            progress=1.0,
            config=config.model_dump(),
            summary={},
            chart_data={},
        )
        db.add(run)
        db.flush()

        input_dir = storage_service.run_subdir(run.id, "input")
        output_dir = storage_service.run_subdir(run.id, "outputs", "traditional")
        input_path = input_dir / "sample.png"
        mask_path = output_dir / "sample_mask.png"
        overlay_path = output_dir / "sample_overlay.png"
        edge_path = output_dir / "sample_edges.png"
        object_overlay_path = output_dir / "sample_objects.png"

        source = np.full((32, 32), 96, dtype=np.uint8)
        confirmed_mask = np.zeros((32, 32), dtype=np.uint8)
        cv2.rectangle(confirmed_mask, (6, 6), (25, 25), 255, thickness=-1)
        cv2.rectangle(confirmed_mask, (12, 12), (19, 19), 0, thickness=-1)
        confirmed_result = traditional_service.apply_postprocess(source, confirmed_mask, config.traditional_seg)

        write_image(input_path, source)
        write_image(mask_path, confirmed_result["mask"])
        write_image(overlay_path, confirmed_result["overlay"])
        write_image(edge_path, confirmed_result["edges"])
        write_image(object_overlay_path, confirmed_result["object_overlay"])

        image = ImageAsset(
            run_id=run.id,
            original_name="sample.png",
            relative_path=storage_service.relative_path(input_path),
            sort_index=0,
        )
        db.add(image)
        db.flush()

        metric = MetricRecord(
            run_id=run.id,
            image_id=image.id,
            mode="traditional",
            summary=_build_metric_summary(
                image.original_name,
                confirmed_result,
                config.traditional_seg,
                um_per_px=config.input_config.um_per_px,
            ),
            artifacts=pipeline_service.build_metric_artifacts(
                input_relative=storage_service.relative_path(input_path),
                processed_relative=None,
                analysis_relative=None,
                footer_relative=None,
                mask_relative=storage_service.relative_path(mask_path),
                overlay_relative=storage_service.relative_path(overlay_path),
                edge_relative=storage_service.relative_path(edge_path),
                object_overlay_relative=storage_service.relative_path(object_overlay_path),
            ),
        )
        db.add(metric)
        db.commit()
        db.refresh(metric)

        run = pipeline_service.rebuild_confirmed_outputs(db, run.id)
        db.refresh(metric)

        export_count = len(db.scalars(select(ExportRecord).where(ExportRecord.run_id == run.id)).all())
        baseline_summary = dict(metric.summary)
        baseline_mask = read_gray(mask_path)
        run_id = run.id
        image_id = image.id

    return {
        "client": client,
        "SessionLocal": SessionLocal,
        "run_id": run_id,
        "image_id": image_id,
        "baseline_summary": baseline_summary,
        "baseline_mask": baseline_mask,
        "baseline_export_count": export_count,
    }


def _preview_payload(image_id: int) -> dict[str, Any]:
    return {
        "mode": "traditional",
        "selected_image_id": image_id,
        "postprocess": {
            "fill_holes": True,
            "watershed": False,
            "watershed_params": {
                "separation": 35,
                "background_iterations": 1,
                "min_marker_area": 12,
            },
            "smoothing": {
                "enabled": False,
                "method": "gaussian",
                "kernel": 3,
            },
            "shape_filter": {
                "enabled": False,
                "min_area": 30,
                "max_area": None,
                "min_solidity": 0.0,
                "min_circularity": 0.0,
                "min_roundness": 0.0,
                "max_aspect_ratio": None,
            },
            "morphology": {
                "opening_enabled": False,
                "opening_kernel": 3,
                "closing_enabled": False,
                "closing_kernel": 3,
            },
            "remove_border": False,
        },
    }


def test_postprocess_preview_does_not_mutate_confirmed_result(seeded_run: dict[str, Any]) -> None:
    client: TestClient = seeded_run["client"]
    run_id = seeded_run["run_id"]
    image_id = seeded_run["image_id"]
    baseline_summary = seeded_run["baseline_summary"]

    response = client.post(f"/api/runs/{run_id}/postprocess/preview", json=_preview_payload(image_id))
    assert response.status_code == 200
    payload = response.json()

    assert payload["preview_token"]
    assert payload["mode"] == "traditional"
    assert payload["image_name"] == "sample.png"
    assert payload["before_mask_url"].endswith("sample_mask.png")
    assert "/tmp/postprocess_preview/" in payload["after_mask_url"]
    assert payload["after_summary"]["volume_fraction"] > payload["before_summary"]["volume_fraction"]

    results_response = client.get(f"/api/runs/{run_id}/results")
    assert results_response.status_code == 200
    results = results_response.json()
    image_row = results["images"][0]["modes"]["traditional"]
    assert image_row["summary"]["volume_fraction"] == baseline_summary["volume_fraction"]
    assert image_row["summary"]["foreground_pixels"] == baseline_summary["foreground_pixels"]


def test_postprocess_confirm_overwrites_confirmed_result_and_rebuilds_outputs(
    seeded_run: dict[str, Any],
) -> None:
    client: TestClient = seeded_run["client"]
    SessionLocal = seeded_run["SessionLocal"]
    run_id = seeded_run["run_id"]
    image_id = seeded_run["image_id"]
    baseline_mask = seeded_run["baseline_mask"]
    baseline_export_count = seeded_run["baseline_export_count"]

    preview_response = client.post(f"/api/runs/{run_id}/postprocess/preview", json=_preview_payload(image_id))
    assert preview_response.status_code == 200
    preview_payload = preview_response.json()

    confirm_response = client.post(
        f"/api/runs/{run_id}/postprocess/confirm",
        json={"preview_token": preview_payload["preview_token"]},
    )
    assert confirm_response.status_code == 200
    confirm_payload = confirm_response.json()
    assert confirm_payload["mode"] == "traditional"
    assert confirm_payload["image_count"] == 1
    assert confirm_payload["run"]["export_bundle_path"]

    results_response = client.get(f"/api/runs/{run_id}/results")
    assert results_response.status_code == 200
    results = results_response.json()
    image_row = results["images"][0]["modes"]["traditional"]
    assert image_row["summary"]["foreground_pixels"] > int(np.count_nonzero(baseline_mask))
    assert results["run"]["summary"]["batch"]["traditional"]["avg_volume_fraction"] == image_row["summary"]["volume_fraction"]
    assert image_row["summary"]["calibrated"] is True
    assert image_row["summary"]["applied_um_per_px"] == pytest.approx(0.5)
    assert image_row["summary"]["size_unit"] == "um"
    assert image_row["summary"]["area_unit"] == "um^2"
    assert results["run"]["summary"]["batch"]["traditional"]["calibrated_image_count"] == 1
    assert results["exports"]

    with SessionLocal() as db:
        metric = db.scalar(select(MetricRecord).where(MetricRecord.run_id == run_id, MetricRecord.mode == "traditional"))
        assert metric is not None
        confirmed_mask_path = storage_service.absolute_path(metric.artifacts["mask_path"])
        exports = db.scalars(select(ExportRecord).where(ExportRecord.run_id == run_id)).all()
        assert len(exports) >= baseline_export_count

    confirmed_mask = read_gray(confirmed_mask_path)
    assert int(np.count_nonzero(confirmed_mask)) > int(np.count_nonzero(baseline_mask))
    preview_dir = storage_service.run_dir(run_id) / "tmp" / "postprocess_preview" / preview_payload["preview_token"]
    assert not preview_dir.exists()


def test_postprocess_preview_restores_calibration_from_run_config_when_metric_summary_is_degraded(
    seeded_run: dict[str, Any],
) -> None:
    client: TestClient = seeded_run["client"]
    SessionLocal = seeded_run["SessionLocal"]
    run_id = seeded_run["run_id"]
    image_id = seeded_run["image_id"]

    with SessionLocal() as db:
        metric = db.scalar(select(MetricRecord).where(MetricRecord.run_id == run_id, MetricRecord.mode == "traditional"))
        assert metric is not None
        summary = dict(metric.summary or {})
        summary.pop("applied_um_per_px", None)
        summary.pop("calibrated", None)
        summary.pop("calibration_source", None)
        summary["area_unit"] = "px^2"
        summary["size_unit"] = "px"
        metric.summary = summary
        db.commit()

    preview_response = client.post(f"/api/runs/{run_id}/postprocess/preview", json=_preview_payload(image_id))
    assert preview_response.status_code == 200
    preview_payload = preview_response.json()
    assert preview_payload["after_summary"]["calibrated"] is True
    assert preview_payload["after_summary"]["applied_um_per_px"] == pytest.approx(0.5)
    assert preview_payload["after_summary"]["size_unit"] == "um"
    assert preview_payload["after_summary"]["area_unit"] == "um^2"
