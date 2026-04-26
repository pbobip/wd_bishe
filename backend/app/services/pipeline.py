from __future__ import annotations

from datetime import datetime
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from sqlalchemy.orm import Session

from backend.app.models.entities import ImageAsset, MetricRecord, RunStep, RunTask
from backend.app.schemas.run import RunCreate
from backend.app.services.algorithms.traditional import traditional_service
from backend.app.services.exporter import export_service
from backend.app.services.model_runner import model_runner_service
from backend.app.services.sem_footer_ocr import sem_footer_ocr_service
from backend.app.services.preprocess import preprocess_service
from backend.app.services.sem_roi import sem_roi_service
from backend.app.services.statistics import statistics_service
from backend.app.services.storage import storage_service
from backend.app.utils.charts import bar_png, histogram_png
from backend.app.utils.image_io import read_gray, write_image


STEP_ORDER = ["input", "preprocess", "traditional", "dl", "stats", "export"]


class PipelineService:
    _STEP_LABELS = {
        "input": "整理输入图像",
        "preprocess": "执行图像预处理",
        "traditional": "执行传统分割",
        "dl": "执行深度学习分割",
        "stats": "生成统计摘要与图表",
        "export": "导出结果",
    }

    _T_CRITICAL_975 = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        21: 2.080,
        22: 2.074,
        23: 2.069,
        24: 2.064,
        25: 2.060,
        26: 2.056,
        27: 2.052,
        28: 2.048,
        29: 2.045,
        30: 2.042,
    }

    def execute(self, db: Session, run_id: int) -> None:
        run = db.get(RunTask, run_id)
        if run is None:
            raise ValueError(f"任务不存在: {run_id}")

        run.status = "running"
        run.progress = 0.05
        run.started_at = datetime.utcnow()
        run.error_message = None
        db.commit()
        db.refresh(run)

        self._sync_steps(db, run)
        db.refresh(run)
        config = RunCreate.model_validate(run.config)
        images = sorted(run.images, key=lambda item: item.sort_index)
        if not images:
            raise ValueError("任务还没有导入图像")
        total_images = len(images)

        processed_paths = {image.id: image.relative_path for image in images}
        layout_context: dict[int, dict[str, Any]] = {}
        batch_summaries: dict[str, dict[str, Any]] = {}
        chart_paths: dict[str, str] = {}
        image_calibration_map = self._build_image_calibration_map(config)
        self._init_execution_summary(db, run, total_images)

        self._start_step(db, run, "input", self._STEP_LABELS["input"], total_images=total_images)
        if config.input_config.auto_crop_sem_region or config.input_config.save_sem_footer:
            layout_dir = storage_service.run_subdir(run_id, "outputs", "layout")
            for index, image in enumerate(images, start=1):
                source = read_gray(storage_service.absolute_path(image.relative_path))
                roi = sem_roi_service.extract(source)
                scale_bar = sem_roi_service.detect_scale_bar(roi.footer_image)
                footer_ocr = sem_footer_ocr_service.analyze(
                    roi.footer_image,
                    scale_bar_hint_bbox=scale_bar.bbox if scale_bar is not None else None,
                )
                analysis_relative = image.relative_path
                footer_relative = None
                calibration_hint = None

                if scale_bar is not None and footer_ocr and footer_ocr.scale_bar_um:
                    suggested_um_per_px = round(footer_ocr.scale_bar_um / scale_bar.pixel_length, 8)
                    calibration_hint = (
                        f"已自动识别比例尺约 {footer_ocr.scale_bar_um:g} μm，对应 {scale_bar.pixel_length} px；"
                        f"建议 um_per_px ≈ {suggested_um_per_px:.8f}。"
                    )
                elif scale_bar is not None:
                    calibration_hint = (
                        f"已检测到底部比例尺横线约 {scale_bar.pixel_length} px；"
                        "请结合标尺文字确认 um_per_px。"
                    )

                if config.input_config.auto_crop_sem_region and roi.cropped_footer:
                    analysis_path = layout_dir / f"{Path(image.original_name).stem}_analysis.png"
                    write_image(analysis_path, roi.analysis_image)
                    analysis_relative = storage_service.relative_path(analysis_path)
                    processed_paths[image.id] = analysis_relative

                if config.input_config.save_sem_footer and roi.footer_image is not None and roi.footer_image.size:
                    footer_path = layout_dir / f"{Path(image.original_name).stem}_footer.png"
                    write_image(footer_path, roi.footer_image)
                    footer_relative = storage_service.relative_path(footer_path)

                layout_context[image.id] = {
                    "analysis_relative_path": analysis_relative,
                    "analysis_url": storage_service.browser_preview_url(analysis_relative),
                    "footer_relative_path": footer_relative,
                    "footer_url": storage_service.static_url(footer_relative),
                    "analysis_bbox": {
                        "x": int(roi.analysis_bbox[0]),
                        "y": int(roi.analysis_bbox[1]),
                        "width": int(roi.analysis_bbox[2]),
                        "height": int(roi.analysis_bbox[3]),
                    },
                    "footer_bbox": (
                        {
                            "x": int(roi.footer_bbox[0]),
                            "y": int(roi.footer_bbox[1]),
                            "width": int(roi.footer_bbox[2]),
                            "height": int(roi.footer_bbox[3]),
                        }
                        if roi.footer_bbox
                        else None
                    ),
                    "source_shape": {"height": int(roi.source_shape[0]), "width": int(roi.source_shape[1])},
                    "cropped_footer": bool(roi.cropped_footer),
                    "cropped_background": bool(roi.cropped_background),
                    "scale_bar_px": int(scale_bar.pixel_length) if scale_bar is not None else None,
                    "scale_bar_confidence": float(scale_bar.confidence) if scale_bar is not None else None,
                    "scale_bar_bbox": (
                        {
                            "x": int(scale_bar.bbox[0]),
                            "y": int(scale_bar.bbox[1]),
                            "width": int(scale_bar.bbox[2]),
                            "height": int(scale_bar.bbox[3]),
                        }
                        if scale_bar is not None
                        else None
                    ),
                    "ocr_scale_bar_um": float(footer_ocr.scale_bar_um) if footer_ocr and footer_ocr.scale_bar_um else None,
                    "ocr_fov_um": float(footer_ocr.fov_um) if footer_ocr and footer_ocr.fov_um else None,
                    "ocr_magnification_text": footer_ocr.magnification_text if footer_ocr else None,
                    "ocr_wd_mm": float(footer_ocr.wd_mm) if footer_ocr and footer_ocr.wd_mm else None,
                    "ocr_detector": footer_ocr.detector if footer_ocr else None,
                    "ocr_scan_mode": footer_ocr.scan_mode if footer_ocr else None,
                    "ocr_vacuum_mode": footer_ocr.vacuum_mode if footer_ocr else None,
                    "ocr_date_text": footer_ocr.date_text if footer_ocr else None,
                    "ocr_time_text": footer_ocr.time_text if footer_ocr else None,
                    "calibration_hint": calibration_hint,
                }
                self._update_step_progress(
                    db,
                    run,
                    "input",
                    processed=index,
                    total=total_images,
                    current_image_name=image.original_name,
                )
        else:
            self._update_step_progress(db, run, "input", processed=total_images, total=total_images)
        self._finish_step(db, run, "input", total_images=total_images)

        if config.preprocess.enabled:
            self._start_step(db, run, "preprocess", self._STEP_LABELS["preprocess"], total_images=total_images)
            preprocess_dir = storage_service.run_subdir(run_id, "processed")
            for index, image in enumerate(images, start=1):
                source = read_gray(storage_service.absolute_path(processed_paths[image.id]))
                processed = preprocess_service.apply(source, config.preprocess)
                target = preprocess_dir / f"{Path(image.original_name).stem}_processed.png"
                write_image(target, processed)
                processed_paths[image.id] = storage_service.relative_path(target)
                self._update_step_progress(
                    db,
                    run,
                    "preprocess",
                    processed=index,
                    total=total_images,
                    current_image_name=image.original_name,
                )
            run.progress = 0.2
            db.commit()
            self._finish_step(db, run, "preprocess", total_images=total_images)

        if config.segmentation_mode == "traditional":
            self._start_step(db, run, "traditional", self._STEP_LABELS["traditional"], total_images=total_images)
            batch_summaries["traditional"] = self._run_traditional(
                db, run, images, processed_paths, layout_context, image_calibration_map, config
            )
            run.progress = 0.55
            db.commit()
            self._finish_step(db, run, "traditional", total_images=total_images)

        if config.segmentation_mode == "dl":
            self._start_step(db, run, "dl", self._STEP_LABELS["dl"], total_images=total_images)
            batch_summaries["dl"] = self._run_dl(
                db, run, images, processed_paths, layout_context, image_calibration_map, config
            )
            run.progress = 0.7
            self._finish_step(db, run, "dl", total_images=total_images)
            db.commit()

        self._start_step(db, run, "stats", self._STEP_LABELS["stats"], total_images=total_images)
        self._update_step_progress(db, run, "stats", processed=total_images, total=total_images)
        run_metrics = db.query(MetricRecord).filter(MetricRecord.run_id == run_id).all()
        execution_summary = self._get_execution_summary(run)
        run.summary = self._build_summary(run, run_metrics, batch_summaries, execution_summary)
        run.chart_data = self._build_charts(run_id, batch_summaries, chart_paths)
        run.progress = 0.88
        db.commit()
        self._finish_step(db, run, "stats", total_images=total_images)

        self._start_step(db, run, "export", self._STEP_LABELS["export"], total_images=total_images)
        self._update_step_progress(db, run, "export", processed=total_images, total=total_images)
        exports = export_service.export_run(db, run_id, run.config, chart_paths, replace_existing=True)
        run.export_bundle_path = exports["bundle"]
        run.progress = 1.0
        run.status = "completed"
        run.finished_at = datetime.utcnow()
        db.commit()
        self._finish_step(db, run, "export", total_images=total_images)

    def _build_image_calibration_map(self, config: RunCreate) -> dict[str, float]:
        mapping: dict[str, float] = {}
        for item in config.input_config.image_calibrations:
            value = item.um_per_px
            if value is None or value <= 0:
                continue
            for key in (
                item.relative_path,
                item.original_name,
                Path(item.relative_path).name,
                Path(item.original_name).name if item.original_name else None,
            ):
                normalized = (key or "").strip()
                if normalized:
                    mapping[normalized] = float(value)
        return mapping

    def _resolve_image_calibration(self, image: ImageAsset, image_calibration_map: dict[str, float], config: RunCreate) -> dict[str, Any]:
        image_value = next(
            (
                image_calibration_map[key]
                for key in (
                    image.relative_path,
                    image.original_name,
                    Path(image.relative_path).name,
                    Path(image.original_name).name,
                )
                if key in image_calibration_map
            ),
            None,
        )
        if image_value is not None and image_value > 0:
            return {
                "um_per_px": float(image_value),
                "calibrated": True,
                "source": "image_override",
            }

        default_value = config.input_config.um_per_px
        if default_value is not None and default_value > 0:
            return {
                "um_per_px": float(default_value),
                "calibrated": True,
                "source": "task_default",
            }

        return {
            "um_per_px": None,
            "calibrated": False,
            "source": "pixels_only",
        }

    def _run_traditional(
        self,
        db: Session,
        run: RunTask,
        images: list[ImageAsset],
        source_paths: dict[int, str],
        layout_context: dict[int, dict[str, Any]],
        image_calibration_map: dict[str, float],
        config: RunCreate,
    ) -> dict[str, Any]:
        output_dir = storage_service.run_subdir(run.id, "outputs", "traditional")
        rows: list[dict[str, Any]] = []
        total_images = len(images)
        for index, image in enumerate(images, start=1):
            source = read_gray(storage_service.absolute_path(source_paths[image.id]))
            result = traditional_service.segment(source, config.traditional_seg)
            calibration = self._resolve_image_calibration(image, image_calibration_map, config)
            effective_um_per_px = calibration["um_per_px"]
            stem = Path(image.original_name).stem
            mask_path = output_dir / f"{stem}_mask.png"
            overlay_path = output_dir / f"{stem}_overlay.png"
            edge_path = output_dir / f"{stem}_edges.png"
            object_overlay_path = output_dir / f"{stem}_objects.png"
            write_image(mask_path, result["mask"])
            write_image(overlay_path, result["overlay"])
            write_image(edge_path, result["edges"])
            write_image(object_overlay_path, result["object_overlay"])

            stats = statistics_service.summarize(
                result["mask"],
                um_per_px=effective_um_per_px,
                objects=result.get("objects"),
                config=config.traditional_seg,
            )
            stats["traditional_method"] = config.traditional_seg.method
            stats["traditional_route_details"] = result.get("route_details", {})
            stats["applied_um_per_px"] = effective_um_per_px
            stats["calibrated"] = bool(calibration["calibrated"])
            stats["calibration_source"] = calibration["source"]
            threshold_value = result.get("route_details", {}).get("threshold")
            if threshold_value is not None:
                stats["threshold"] = float(threshold_value)
            stats["image_name"] = image.original_name
            if layout_context.get(image.id, {}).get("calibration_hint"):
                stats["calibration_hint"] = layout_context[image.id]["calibration_hint"]
            if image.id in layout_context:
                stats["roi"] = {
                    key: value
                    for key, value in layout_context[image.id].items()
                    if key not in {"analysis_relative_path", "analysis_url", "footer_relative_path", "footer_url"}
                }
            artifact = self.build_metric_artifacts(
                input_relative=image.relative_path,
                processed_relative=source_paths[image.id] if source_paths[image.id] != image.relative_path else None,
                analysis_relative=layout_context.get(image.id, {}).get("analysis_relative_path"),
                footer_relative=layout_context.get(image.id, {}).get("footer_relative_path"),
                mask_relative=storage_service.relative_path(mask_path),
                overlay_relative=storage_service.relative_path(overlay_path),
                edge_relative=storage_service.relative_path(edge_path),
                object_overlay_relative=storage_service.relative_path(object_overlay_path),
            )
            rows.append({"summary": stats, "artifact": artifact})
            db.add(MetricRecord(run_id=run.id, image_id=image.id, mode="traditional", summary=stats, artifacts=artifact))
            self._update_step_progress(
                db,
                run,
                "traditional",
                processed=index,
                total=total_images,
                current_image_name=image.original_name,
            )
        return self._aggregate_mode(rows)

    def _run_dl(
        self,
        db: Session,
        run: RunTask,
        images: list[ImageAsset],
        source_paths: dict[int, str],
        layout_context: dict[int, dict[str, Any]],
        image_calibration_map: dict[str, float],
        config: RunCreate,
    ) -> dict[str, Any]:
        output_dir = storage_service.run_subdir(run.id, "outputs", "dl")
        tmp_dir = storage_service.run_subdir(run.id, "tmp")
        manifest_path = tmp_dir / "dl_manifest.json"
        progress_path = tmp_dir / "dl_progress.json"
        payload = {
            "run_id": run.id,
            "runner_id": config.dl_model.runner_id,
            "weight_path": config.dl_model.weight_path,
            "input_size": config.dl_model.input_size,
            "device": config.dl_model.device,
            "extra_params": config.dl_model.extra_params,
            "manifest_path": str(manifest_path),
            "progress_path": str(progress_path),
            "output_dir": str(output_dir),
            "auto_crop_sem_region": False,  # always skip crop in inference; input step already handled footer removal
            "items": [
                {
                    "image_id": image.id,
                    "image_name": image.original_name,
                    "image_path": str(storage_service.absolute_path(source_paths[image.id])),
                    "relative_input_path": source_paths[image.id],
                }
                for image in images
            ],
        }
        total_images = len(images)

        def report_progress(progress: dict[str, Any]) -> None:
            self._update_step_progress(
                db,
                run,
                "dl",
                processed=int(progress.get("processed", 0)),
                total=int(progress.get("total", total_images) or total_images),
                current_image_name=progress.get("image_name"),
                current_batch=progress.get("current_batch"),
            )

        manifest = model_runner_service.run_inference(
            db,
            config.dl_model.model_slot,
            payload,
            runner_id=config.dl_model.runner_id,
            progress_callback=report_progress,
        )
        rows: list[dict[str, Any]] = []
        for item in manifest["items"]:
            image_id = int(item["image_id"])
            image = next((candidate for candidate in images if candidate.id == image_id), None)
            if image is None:
                continue
            calibration = self._resolve_image_calibration(image, image_calibration_map, config)
            effective_um_per_px = calibration["um_per_px"]
            mask_image = read_gray(item["mask_path"])
            source_image = read_gray(item["image_path"])
            object_stats = statistics_service.build_object_stats(
                mask_image,
                um_per_px=effective_um_per_px,
                config=config.traditional_seg,
                base_image=source_image,
            )
            refined_mask = object_stats["kept_mask"]
            write_image(Path(item["mask_path"]), refined_mask)
            refined_overlay = self._build_overlay_image(source_image, refined_mask)
            write_image(Path(item["overlay_path"]), refined_overlay)
            mask_relative = storage_service.relative_path(Path(item["mask_path"]))
            overlay_relative = storage_service.relative_path(Path(item["overlay_path"]))
            object_overlay_path = output_dir / f"{Path(item['image_name']).stem}_objects.png"
            write_image(object_overlay_path, object_stats["object_overlay"])
            stats = statistics_service.summarize(
                refined_mask,
                um_per_px=effective_um_per_px,
                objects=object_stats["objects"],
                config=config.traditional_seg,
            )
            stats["image_name"] = item["image_name"]
            stats["applied_um_per_px"] = effective_um_per_px
            stats["calibrated"] = bool(calibration["calibrated"])
            stats["calibration_source"] = calibration["source"]
            if layout_context.get(image_id, {}).get("calibration_hint"):
                stats["calibration_hint"] = layout_context[image_id]["calibration_hint"]
            if image_id in layout_context:
                stats["roi"] = {
                    key: value
                    for key, value in layout_context[image_id].items()
                    if key not in {"analysis_relative_path", "analysis_url", "footer_relative_path", "footer_url"}
                }
            processed_relative = source_paths.get(image_id, item["relative_input_path"])
            artifact = self.build_metric_artifacts(
                input_relative=item["relative_input_path"],
                processed_relative=processed_relative if processed_relative != item["relative_input_path"] else None,
                analysis_relative=layout_context.get(image_id, {}).get("analysis_relative_path"),
                footer_relative=layout_context.get(image_id, {}).get("footer_relative_path"),
                mask_relative=mask_relative,
                overlay_relative=overlay_relative,
                edge_relative=None,
                object_overlay_relative=storage_service.relative_path(object_overlay_path),
            )
            rows.append({"summary": stats, "artifact": artifact})
            db.add(
                MetricRecord(
                    run_id=run.id,
                    image_id=image_id,
                    mode="dl",
                    summary=stats,
                    artifacts=artifact,
                )
            )
        return self._aggregate_mode(rows)

    def _build_overlay_image(self, gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
        base = np.asarray(gray)
        if base.ndim == 2:
            base = np.stack([base, base, base], axis=-1)
        else:
            base = base.copy()
        base = np.clip(base, 0, 255).astype(np.uint8)
        color_mask = np.zeros_like(base)
        color_mask[mask > 0] = (0, 180, 0)
        return cv2.addWeighted(base, 0.75, color_mask, 0.25, 0)

    def build_metric_artifacts(
        self,
        *,
        input_relative: str | None,
        processed_relative: str | None,
        analysis_relative: str | None,
        footer_relative: str | None,
        mask_relative: str | None,
        overlay_relative: str | None,
        edge_relative: str | None,
        object_overlay_relative: str | None,
    ) -> dict[str, Any]:
        return {
            "input_path": input_relative,
            "input_url": storage_service.static_url(input_relative),
            "processed_path": processed_relative,
            "processed_url": storage_service.static_url(processed_relative),
            "analysis_input_path": analysis_relative,
            "analysis_input_url": storage_service.static_url(analysis_relative),
            "footer_panel_path": footer_relative,
            "footer_panel_url": storage_service.static_url(footer_relative),
            "mask_path": mask_relative,
            "mask_url": storage_service.static_url(mask_relative),
            "overlay_path": overlay_relative,
            "overlay_url": storage_service.static_url(overlay_relative),
            "edge_path": edge_relative,
            "edge_url": storage_service.static_url(edge_relative),
            "object_overlay_path": object_overlay_relative,
            "object_overlay_url": storage_service.static_url(object_overlay_relative),
        }

    def rebuild_confirmed_outputs(self, db: Session, run_id: int) -> RunTask:
        run = db.get(RunTask, run_id)
        if run is None:
            raise ValueError(f"任务不存在: {run_id}")

        metrics = db.query(MetricRecord).filter(MetricRecord.run_id == run_id).all()
        if not metrics:
            raise ValueError("当前任务还没有可重建的确认结果")

        batch_summaries = self._collect_batch_summaries(metrics)
        chart_paths: dict[str, str] = {}
        execution_summary = self._get_execution_summary(run)

        run.summary = self._build_summary(run, metrics, batch_summaries, execution_summary)
        run.chart_data = self._build_charts(run.id, batch_summaries, chart_paths)
        exports = export_service.export_run(
            db,
            run.id,
            run.config,
            chart_paths,
            replace_existing=True,
        )
        run.export_bundle_path = exports["bundle"]
        db.commit()
        db.refresh(run)
        return run

    def _collect_batch_summaries(self, metrics: list[MetricRecord]) -> dict[str, dict[str, Any]]:
        rows_by_mode: dict[str, list[dict[str, Any]]] = {}
        for metric in metrics:
            rows_by_mode.setdefault(metric.mode, []).append(
                {
                    "summary": dict(metric.summary or {}),
                    "artifact": dict(metric.artifacts or {}),
                }
            )
        return {mode: self._aggregate_mode(rows) for mode, rows in rows_by_mode.items() if rows}

    def _aggregate_mode(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        summaries = [row["summary"] for row in rows]
        calibrated_summaries = [item for item in summaries if bool(item.get("calibrated"))]
        physical_summaries = calibrated_summaries or summaries
        calibrated_image_count = len(calibrated_summaries)
        physical_stats_image_count = len(physical_summaries)
        total_image_count = len(summaries)
        mixed_calibration = bool(calibrated_summaries) and calibrated_image_count < total_image_count
        if calibrated_summaries:
            physical_stats_scope = "calibrated_subset" if mixed_calibration else "all_calibrated"
        else:
            physical_stats_scope = "pixels_only"

        areas = [float(area) for item in physical_summaries for area in item["areas"]]
        sizes = [float(size) for item in physical_summaries for size in item.get("sizes", item["diameters"])]
        diameters = [float(diam) for item in physical_summaries for diam in item["diameters"]]
        perimeters = [float(value) for item in physical_summaries for value in item.get("perimeters", [])]
        majors = [float(value) for item in physical_summaries for value in item.get("majors", [])]
        minors = [float(value) for item in physical_summaries for value in item.get("minors", [])]
        ferets = [float(value) for item in physical_summaries for value in item.get("ferets", [])]
        minferets = [float(value) for item in physical_summaries for value in item.get("minferets", [])]
        aspect_ratios = [float(value) for item in physical_summaries for value in item.get("aspect_ratios", [])]
        circularities = [float(value) for item in physical_summaries for value in item.get("circularities", [])]
        roundnesses = [float(value) for item in physical_summaries for value in item.get("roundnesses", [])]
        solidities = [float(value) for item in physical_summaries for value in item.get("solidities", [])]
        channel_widths_x = [float(width) for item in physical_summaries for width in item.get("channel_widths_x", [])]
        channel_widths_y = [float(width) for item in physical_summaries for width in item.get("channel_widths_y", [])]
        object_counts = [int(item.get("object_count", item.get("particle_count", 0))) for item in summaries]
        filtered_object_counts = [int(item.get("filtered_object_count", 0)) for item in summaries]
        volume_fractions = [float(item["volume_fraction"]) for item in summaries]
        particle_counts = [int(item["particle_count"]) for item in summaries]
        image_mean_areas = [float(item["mean_area"]) for item in physical_summaries if int(item.get("particle_count", 0)) > 0]
        image_mean_sizes = [float(item["mean_size"]) for item in physical_summaries if int(item.get("particle_count", 0)) > 0]
        image_mean_channel_widths_x = [
            float(item["mean_channel_width_x"]) for item in physical_summaries if int(item.get("channel_width_count_x", 0)) > 0
        ]
        image_mean_channel_widths_y = [
            float(item["mean_channel_width_y"]) for item in physical_summaries if int(item.get("channel_width_count_y", 0)) > 0
        ]
        image_names = [str(item.get("image_name", f"image_{index + 1}")) for index, item in enumerate(summaries)]
        physical_image_names = [
            str(item.get("image_name", f"image_{index + 1}")) for index, item in enumerate(physical_summaries)
        ]
        area_unit = next((str(item.get("area_unit")) for item in physical_summaries if item.get("area_unit")), "px^2")
        size_unit = next((str(item.get("size_unit")) for item in physical_summaries if item.get("size_unit")), "px")
        diameter_unit = next((str(item.get("diameter_unit")) for item in physical_summaries if item.get("diameter_unit")), "px")
        channel_width_unit = next((str(item.get("channel_width_unit")) for item in physical_summaries if item.get("channel_width_unit")), "px")
        total_pixels = sum(int(item.get("total_pixels", 0)) for item in summaries)
        foreground_pixels = sum(int(item.get("foreground_pixels", 0)) for item in summaries)
        calibration_hint = next(
            (str(item.get("calibration_hint")) for item in summaries if item.get("calibration_hint")),
            None,
        )
        calibration_probe = next(
            (
                {
                    "footer_detected": bool(item["roi"].get("cropped_footer")),
                    "scale_bar_detected": item["roi"].get("scale_bar_px") is not None,
                    "scale_bar_pixels": item["roi"].get("scale_bar_px"),
                    "analysis_width_px": item["roi"].get("analysis_bbox", {}).get("width"),
                    "source_width_px": item["roi"].get("source_shape", {}).get("width"),
                    "analysis_height_px": item["roi"].get("analysis_bbox", {}).get("height"),
                    "ocr_scale_bar_um": item["roi"].get("ocr_scale_bar_um"),
                    "ocr_fov_um": item["roi"].get("ocr_fov_um"),
                    "ocr_magnification_text": item["roi"].get("ocr_magnification_text"),
                    "ocr_wd_mm": item["roi"].get("ocr_wd_mm"),
                    "ocr_detector": item["roi"].get("ocr_detector"),
                    "ocr_vacuum_mode": item["roi"].get("ocr_vacuum_mode"),
                }
                for item in summaries
                if isinstance(item.get("roi"), dict)
            ),
            None,
        )

        def safe_stat(values: list[float], fn) -> float:
            return float(fn(values)) if values else 0.0

        return {
            "image_count": len(rows),
            "volume_fraction": float(foreground_pixels / total_pixels) if total_pixels else 0.0,
            "avg_volume_fraction": safe_stat(volume_fractions, np.mean),
            "avg_particle_count": safe_stat([float(count) for count in particle_counts], np.mean),
            "avg_image_mean_area": safe_stat(image_mean_areas, np.mean),
            "avg_image_mean_size": safe_stat(image_mean_sizes, np.mean),
            "avg_image_mean_channel_width_x": safe_stat(image_mean_channel_widths_x, np.mean),
            "avg_image_mean_channel_width_y": safe_stat(image_mean_channel_widths_y, np.mean),
            "avg_object_count": safe_stat([float(count) for count in object_counts], np.mean),
            "avg_filtered_object_count": safe_stat([float(count) for count in filtered_object_counts], np.mean),
            "particle_count_total": int(sum(particle_counts)),
            "object_count_total": int(sum(object_counts)),
            "filtered_object_count_total": int(sum(filtered_object_counts)),
            "calibrated_image_count": calibrated_image_count,
            "physical_stats_image_count": physical_stats_image_count,
            "physical_stats_scope": physical_stats_scope,
            "mixed_calibration": mixed_calibration,
            "particle_counts": particle_counts,
            "object_counts": object_counts,
            "image_mean_areas": image_mean_areas,
            "image_mean_sizes": image_mean_sizes,
            "image_mean_channel_widths_x": image_mean_channel_widths_x,
            "image_mean_channel_widths_y": image_mean_channel_widths_y,
            "areas": areas,
            "sizes": sizes,
            "diameters": diameters,
            "perimeters": perimeters,
            "majors": majors,
            "minors": minors,
            "ferets": ferets,
            "minferets": minferets,
            "aspect_ratios": aspect_ratios,
            "circularities": circularities,
            "roundnesses": roundnesses,
            "solidities": solidities,
            "channel_widths_x": channel_widths_x,
            "channel_widths_y": channel_widths_y,
            "channel_width_count_x": len(channel_widths_x),
            "channel_width_count_y": len(channel_widths_y),
            "volume_fractions": volume_fractions,
            "mean_area": safe_stat(areas, np.mean),
            "median_area": safe_stat(areas, np.median),
            "std_area": safe_stat(areas, np.std),
            "mean_size": safe_stat(sizes, np.mean),
            "median_size": safe_stat(sizes, np.median),
            "std_size": safe_stat(sizes, np.std),
            "mean_diameter": safe_stat(diameters, np.mean),
            "median_diameter": safe_stat(diameters, np.median),
            "std_diameter": safe_stat(diameters, np.std),
            "mean_perimeter": safe_stat(perimeters, np.mean),
            "median_perimeter": safe_stat(perimeters, np.median),
            "std_perimeter": safe_stat(perimeters, np.std),
            "mean_major": safe_stat(majors, np.mean),
            "median_major": safe_stat(majors, np.median),
            "std_major": safe_stat(majors, np.std),
            "mean_minor": safe_stat(minors, np.mean),
            "median_minor": safe_stat(minors, np.median),
            "std_minor": safe_stat(minors, np.std),
            "mean_feret": safe_stat(ferets, np.mean),
            "median_feret": safe_stat(ferets, np.median),
            "std_feret": safe_stat(ferets, np.std),
            "mean_minferet": safe_stat(minferets, np.mean),
            "median_minferet": safe_stat(minferets, np.median),
            "std_minferet": safe_stat(minferets, np.std),
            "mean_aspect_ratio": safe_stat(aspect_ratios, np.mean),
            "median_aspect_ratio": safe_stat(aspect_ratios, np.median),
            "std_aspect_ratio": safe_stat(aspect_ratios, np.std),
            "mean_circularity": safe_stat(circularities, np.mean),
            "median_circularity": safe_stat(circularities, np.median),
            "std_circularity": safe_stat(circularities, np.std),
            "mean_roundness": safe_stat(roundnesses, np.mean),
            "median_roundness": safe_stat(roundnesses, np.median),
            "std_roundness": safe_stat(roundnesses, np.std),
            "mean_solidity": safe_stat(solidities, np.mean),
            "median_solidity": safe_stat(solidities, np.median),
            "std_solidity": safe_stat(solidities, np.std),
            "mean_channel_width_x": safe_stat(channel_widths_x, np.mean),
            "median_channel_width_x": safe_stat(channel_widths_x, np.median),
            "std_channel_width_x": safe_stat(channel_widths_x, np.std),
            "mean_channel_width_y": safe_stat(channel_widths_y, np.mean),
            "median_channel_width_y": safe_stat(channel_widths_y, np.median),
            "std_channel_width_y": safe_stat(channel_widths_y, np.std),
            "area_unit": area_unit,
            "size_unit": size_unit,
            "diameter_unit": diameter_unit,
            "channel_width_unit": channel_width_unit,
            "image_names": image_names,
            "physical_stats_image_names": physical_image_names,
            "foreground_pixels": int(foreground_pixels),
            "total_pixels": int(total_pixels),
            "calibration_hint": calibration_hint,
            "calibration_probe": calibration_probe,
            "volume_fraction_ci95": self._confidence_interval_95(volume_fractions, unit="fraction"),
            "particle_count_ci95": self._confidence_interval_95([float(count) for count in particle_counts], unit="count"),
            "image_mean_area_ci95": self._confidence_interval_95(image_mean_areas, unit=area_unit),
            "image_mean_size_ci95": self._confidence_interval_95(image_mean_sizes, unit=size_unit),
            "image_mean_channel_width_x_ci95": self._confidence_interval_95(
                image_mean_channel_widths_x, unit=channel_width_unit
            ),
            "image_mean_channel_width_y_ci95": self._confidence_interval_95(
                image_mean_channel_widths_y, unit=channel_width_unit
            ),
        }

    def _confidence_interval_95(self, values: list[float], unit: str) -> dict[str, Any]:
        clean_values = [float(value) for value in values if value is not None and np.isfinite(value)]
        sample_count = len(clean_values)
        if sample_count == 0:
            return {
                "available": False,
                "n": 0,
                "mean": 0.0,
                "lower": None,
                "upper": None,
                "half_width": None,
                "critical_value": None,
                "unit": unit,
                "method": "student_t",
                "reason": "无有效图像级样本",
            }

        sample_mean = float(np.mean(clean_values))
        if sample_count == 1:
            return {
                "available": False,
                "n": 1,
                "mean": sample_mean,
                "lower": None,
                "upper": None,
                "half_width": None,
                "critical_value": None,
                "unit": unit,
                "method": "student_t",
                "reason": "图像级样本数不足 2，无法估计 95% CI",
            }

        sample_std = float(np.std(clean_values, ddof=1))
        critical_value = self._student_t_critical_95(sample_count - 1)
        half_width = critical_value * sample_std / math.sqrt(sample_count)
        return {
            "available": True,
            "n": sample_count,
            "mean": sample_mean,
            "lower": float(sample_mean - half_width),
            "upper": float(sample_mean + half_width),
            "half_width": float(half_width),
            "critical_value": float(critical_value),
            "unit": unit,
            "method": "student_t",
            "reason": None,
        }

    def _student_t_critical_95(self, degrees_of_freedom: int) -> float:
        if degrees_of_freedom <= 0:
            return 0.0
        if degrees_of_freedom in self._T_CRITICAL_975:
            return self._T_CRITICAL_975[degrees_of_freedom]
        if degrees_of_freedom <= 40:
            return 2.021
        if degrees_of_freedom <= 60:
            return 2.000
        if degrees_of_freedom <= 120:
            return 1.980
        return 1.960

    def _build_summary(
        self,
        run: RunTask,
        metrics: list[MetricRecord],
        batch_summaries: dict[str, dict[str, Any]],
        execution: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        image_map: dict[int, dict[str, Any]] = {}
        for metric in metrics:
            if metric.image_id is None:
                continue
            image_row = image_map.setdefault(metric.image_id, {"image_id": metric.image_id, "modes": {}})
            image_row["modes"][metric.mode] = {"summary": metric.summary, "artifacts": metric.artifacts}
        return {
            "batch": batch_summaries,
            "execution": execution,
            "failed_items": (execution or {}).get("failed_items", []),
            "calibration_hint": next(
                (summary.get("calibration_hint") for summary in batch_summaries.values() if summary.get("calibration_hint")),
                None,
            ),
            "calibration_probe": next(
                (summary.get("calibration_probe") for summary in batch_summaries.values() if summary.get("calibration_probe")),
                None,
            ),
        }

    def _build_charts(self, run_id: int, batch_summaries: dict[str, dict[str, Any]], chart_paths: dict[str, str]) -> dict[str, Any]:
        chart_dir = storage_service.run_subdir(run_id, "charts")
        response: dict[str, Any] = {}
        for mode, summary in batch_summaries.items():
            area_file = chart_dir / f"{mode}_area_hist.png"
            size_file = chart_dir / f"{mode}_size_hist.png"
            vf_file = chart_dir / f"{mode}_vf_bar.png"
            histogram_png(summary["areas"], f"{mode} area distribution", "area", area_file)
            histogram_png(summary["sizes"], f"{mode} size distribution", "equiv size", size_file)
            labels = [f"img{i + 1}" for i in range(len(summary["volume_fractions"]))]
            vf_values = [value * 100 for value in summary["volume_fractions"]]
            bar_png(labels, vf_values, f"{mode} vf comparison", "Vf (%)", vf_file)
            chart_paths[f"{mode}_area_hist"] = storage_service.relative_path(area_file)
            chart_paths[f"{mode}_size_hist"] = storage_service.relative_path(size_file)
            chart_paths[f"{mode}_vf_bar"] = storage_service.relative_path(vf_file)
            response[mode] = {
                "area_hist_url": storage_service.static_url(chart_paths[f"{mode}_area_hist"]),
                "size_hist_url": storage_service.static_url(chart_paths[f"{mode}_size_hist"]),
                "vf_bar_url": storage_service.static_url(chart_paths[f"{mode}_vf_bar"]),
                "volume_fractions": summary["volume_fractions"],
                "sizes": summary["sizes"][:300],
                "areas": summary["areas"][:300],
            }
        return response

    def _sync_steps(self, db: Session, run: RunTask) -> None:
        existing = {step.step_key for step in run.steps}
        for step_key in STEP_ORDER:
            if step_key not in existing:
                db.add(RunStep(run_id=run.id, step_key=step_key, status="pending"))
        db.commit()

    def _init_execution_summary(self, db: Session, run: RunTask, total_images: int) -> None:
        run.summary = {
            **dict(run.summary or {}),
            "execution": {
                "total_images": total_images,
                "processed_images": 0,
                "remaining_images": total_images,
                "current_step": None,
                "current_batch": None,
                "current_image_name": None,
                "updated_at": datetime.utcnow().isoformat(),
                "step_details": {},
                "failed_items": [],
            },
        }
        db.commit()

    def _get_execution_summary(self, run: RunTask) -> dict[str, Any] | None:
        summary = dict(run.summary or {})
        execution = summary.get("execution")
        return dict(execution) if isinstance(execution, dict) else None

    def _build_step_message(
        self,
        step_key: str,
        processed: int,
        total: int,
        current_image_name: str | None = None,
    ) -> str:
        base_message = self._STEP_LABELS.get(step_key, step_key)
        message = f"{base_message}（已处理 {processed}/{total}）"
        if current_image_name:
            message = f"{message} · {current_image_name}"
        return message

    def _build_step_details(
        self,
        step_key: str,
        processed: int,
        total: int,
        current_image_name: str | None = None,
        current_batch: str | None = None,
        status: str = "running",
    ) -> dict[str, Any]:
        processed_images = max(0, min(processed, total))
        return {
            "step_key": step_key,
            "status": status,
            "processed_images": processed_images,
            "total_images": total,
            "remaining_images": max(total - processed_images, 0),
            "progress_ratio": float(processed_images / total) if total else 1.0,
            "current_image_name": current_image_name,
            "current_batch": current_batch or step_key,
            "failed_items": [],
            "message": self._build_step_message(step_key, processed_images, total, current_image_name),
            "updated_at": datetime.utcnow().isoformat(),
        }

    def _set_execution_state(self, run: RunTask, step_key: str, details: dict[str, Any]) -> None:
        summary = dict(run.summary or {})
        execution = dict(summary.get("execution") or {})
        step_details = dict(execution.get("step_details") or {})
        step_details[step_key] = details
        failed_items = list(execution.get("failed_items") or [])
        execution.update(
            {
                "total_images": int(details.get("total_images", execution.get("total_images", 0) or 0)),
                "processed_images": int(details.get("processed_images", execution.get("processed_images", 0) or 0)),
                "remaining_images": int(details.get("remaining_images", execution.get("remaining_images", 0) or 0)),
                "current_step": step_key,
                "current_batch": details.get("current_batch") or step_key,
                "current_image_name": details.get("current_image_name"),
                "updated_at": details.get("updated_at", datetime.utcnow().isoformat()),
                "step_details": step_details,
                "failed_items": failed_items,
            }
        )
        summary["execution"] = execution
        run.summary = summary

    def _update_step_progress(
        self,
        db: Session,
        run: RunTask,
        step_key: str,
        processed: int,
        total: int,
        current_image_name: str | None = None,
        current_batch: str | None = None,
        status: str = "running",
    ) -> None:
        step = next(step for step in run.steps if step.step_key == step_key)
        details = self._build_step_details(
            step_key,
            processed,
            total,
            current_image_name,
            current_batch=current_batch,
            status=status,
        )
        step.message = str(details["message"])
        step.details = details
        if status == "running":
            step.status = "running"
        self._set_execution_state(run, step_key, details)
        db.commit()

    def _start_step(self, db: Session, run: RunTask, step_key: str, message: str, total_images: int | None = None) -> None:
        step = next(step for step in run.steps if step.step_key == step_key)
        step.status = "running"
        step.started_at = datetime.utcnow()
        if total_images is None:
            step.message = message
            db.commit()
            return
        details = self._build_step_details(step_key, 0, total_images, status="running")
        details["message"] = message if total_images == 0 else details["message"]
        step.message = str(details["message"])
        step.details = details
        self._set_execution_state(run, step_key, details)
        db.commit()

    def _finish_step(self, db: Session, run: RunTask, step_key: str, total_images: int | None = None) -> None:
        step = next(step for step in run.steps if step.step_key == step_key)
        step.status = "completed"
        step.finished_at = datetime.utcnow()
        if total_images is not None:
            current_details = dict(step.details or {})
            details = self._build_step_details(
                step_key,
                processed=total_images,
                total=total_images,
                current_image_name=current_details.get("current_image_name"),
                current_batch=current_details.get("current_batch"),
                status="completed",
            )
            step.message = str(details["message"])
            step.details = details
            self._set_execution_state(run, step_key, details)
        db.commit()

    def _fail_step(self, db: Session, run: RunTask, step_key: str, message: str) -> None:
        step = next(step for step in run.steps if step.step_key == step_key)
        step.status = "failed"
        step.message = message
        step.finished_at = datetime.utcnow()
        current_details = dict(step.details or {})
        total_images = int(current_details.get("total_images", 0) or 0)
        processed_images = int(current_details.get("processed_images", 0) or 0)
        details = self._build_step_details(
            step_key,
            processed=processed_images,
            total=total_images,
            current_image_name=current_details.get("current_image_name"),
            current_batch=current_details.get("current_batch"),
            status="failed",
        )
        details["message"] = message
        step.details = details
        self._set_execution_state(run, step_key, details)
        self._append_failed_item(
            run,
            {
                "stage": step_key,
                "current_batch": details.get("current_batch") or step_key,
                "current_image_name": details.get("current_image_name"),
                "message": message,
            },
        )
        db.commit()

    def _append_failed_item(self, run: RunTask, item: dict[str, Any]) -> None:
        summary = dict(run.summary or {})
        execution = dict(summary.get("execution") or {})
        failed_items = list(execution.get("failed_items") or [])
        failed_items.append(item)
        execution["failed_items"] = failed_items
        execution["updated_at"] = datetime.utcnow().isoformat()
        summary["execution"] = execution
        run.summary = summary
        return None


pipeline_service = PipelineService()
