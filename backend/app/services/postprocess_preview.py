from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.core.config import SETTINGS
from backend.app.models.entities import ImageAsset, MetricRecord, RunTask
from backend.app.schemas.run import PostprocessConfig, RunCreate
from backend.app.services.algorithms.traditional import traditional_service
from backend.app.services.pipeline import pipeline_service
from backend.app.services.statistics import statistics_service
from backend.app.services.storage import storage_service
from backend.app.utils.image_io import read_gray, write_image


class PreviewNotFoundError(FileNotFoundError):
    pass


class PreviewConflictError(RuntimeError):
    pass


class PostprocessPreviewService:
    def create_preview(
        self,
        db: Session,
        run_id: int,
        mode: str,
        postprocess: PostprocessConfig,
        selected_image_id: int | None = None,
    ) -> dict[str, Any]:
        run = db.get(RunTask, run_id)
        if run is None:
            raise ValueError("任务不存在")

        config = RunCreate.model_validate(run.config)
        effective_config = traditional_service.build_postprocess_config(config.traditional_seg, postprocess)
        metrics = self._list_mode_metrics(db, run_id, mode)
        if not metrics:
            raise ValueError("当前模式还没有可预览的确认结果")

        image_map = {image.id: image for image in run.images}
        selected_metric = None
        if selected_image_id is None:
            selected_metric = metrics[0]
        else:
            selected_metric = next((metric for metric in metrics if metric.image_id == selected_image_id), None)
        if selected_metric is None:
            raise ValueError("当前图像不在该模式的确认结果中")

        image = image_map.get(selected_metric.image_id or -1)
        if image is None:
            raise ValueError("当前图像不存在")

        preview_token = uuid4().hex
        preview_root = storage_service.run_subdir(run_id, "tmp", "postprocess_preview")
        shutil.rmtree(preview_root, ignore_errors=True)
        preview_dir = storage_service.run_subdir(run_id, "tmp", "postprocess_preview", preview_token)
        current_artifacts = dict(selected_metric.artifacts or {})
        source_relative = self._resolve_source_relative(current_artifacts)
        if source_relative is None:
            raise ValueError(f"图像 {image.original_name} 缺少可用于后处理的源图")
        mask_relative = self._resolve_relative_path(current_artifacts, "mask_path", "mask_url")
        if mask_relative is None:
            raise ValueError(f"图像 {image.original_name} 缺少当前确认 mask")

        source_image = read_gray(storage_service.absolute_path(source_relative))
        current_mask = read_gray(storage_service.absolute_path(mask_relative))
        result = traditional_service.apply_postprocess(source_image, current_mask, effective_config)

        stem = Path(image.original_name).stem
        preview_mask_path = preview_dir / f"{stem}_mask.png"
        preview_overlay_path = preview_dir / f"{stem}_overlay.png"
        preview_edge_path = preview_dir / f"{stem}_edges.png"
        preview_object_overlay_path = preview_dir / f"{stem}_objects.png"
        write_image(preview_mask_path, result["mask"])
        write_image(preview_overlay_path, result["overlay"])
        write_image(preview_edge_path, result["edges"])
        write_image(preview_object_overlay_path, result["object_overlay"])

        preview_artifact = pipeline_service.build_metric_artifacts(
            input_relative=self._resolve_relative_path(current_artifacts, "input_path", "input_url"),
            processed_relative=self._resolve_relative_path(current_artifacts, "processed_path", "processed_url"),
            analysis_relative=self._resolve_relative_path(
                current_artifacts, "analysis_input_path", "analysis_input_url"
            ),
            footer_relative=self._resolve_relative_path(current_artifacts, "footer_panel_path", "footer_panel_url"),
            mask_relative=storage_service.relative_path(preview_mask_path),
            overlay_relative=storage_service.relative_path(preview_overlay_path),
            edge_relative=storage_service.relative_path(preview_edge_path),
            object_overlay_relative=storage_service.relative_path(preview_object_overlay_path),
        )
        preview_summary = self._build_preview_summary(
            current_summary=selected_metric.summary,
            mask=result["mask"],
            objects=result["objects"],
            segment_config=effective_config,
            run_config=config,
            image=image,
            route_details=result.get("route_details"),
        )
        selected_payload = self._build_selected_payload(
            {
                "image_id": image.id,
                "image_name": image.original_name,
                "baseline_summary": dict(selected_metric.summary or {}),
                "preview_artifacts": preview_artifact,
                "summary": preview_summary,
            },
            current_artifacts,
        )

        manifest = {
            "run_id": run_id,
            "mode": mode,
            "postprocess": postprocess.model_dump(),
            "selected_image_id": selected_payload["image_id"],
        }
        manifest_path = preview_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        return {
            "preview_token": preview_token,
            "mode": mode,
            "image_id": selected_payload["image_id"],
            "image_name": selected_payload["image_name"],
            "before_mask_url": selected_payload["before_mask_url"],
            "after_mask_url": selected_payload["after_mask_url"],
            "before_overlay_url": selected_payload["before_overlay_url"],
            "after_overlay_url": selected_payload["after_overlay_url"],
            "before_object_overlay_url": selected_payload["before_object_overlay_url"],
            "after_object_overlay_url": selected_payload["after_object_overlay_url"],
            "before_summary": selected_payload["before_summary"],
            "after_summary": selected_payload["after_summary"],
            "image_count": len(metrics),
        }

    def confirm_preview(self, db: Session, run_id: int, preview_token: str) -> dict[str, Any]:
        run = db.get(RunTask, run_id)
        if run is None:
            raise ValueError("任务不存在")

        preview_dir = storage_service.run_dir(run_id) / "tmp" / "postprocess_preview" / preview_token
        manifest_path = preview_dir / "manifest.json"
        if not manifest_path.exists():
            raise PreviewNotFoundError("预览不存在或已过期")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest.get("run_id", -1)) != run_id:
            raise PreviewConflictError("预览任务与当前任务不一致")

        config = RunCreate.model_validate(run.config)
        postprocess = PostprocessConfig.model_validate(manifest.get("postprocess") or {})
        effective_config = traditional_service.build_postprocess_config(config.traditional_seg, postprocess)
        metrics = self._list_mode_metrics(db, run_id, str(manifest["mode"]))
        image_map = {image.id: image for image in run.images}

        for metric in metrics:
            image = image_map.get(metric.image_id or -1)
            if image is None:
                continue

            current_artifacts = dict(metric.artifacts or {})
            source_relative = self._resolve_source_relative(current_artifacts)
            if source_relative is None:
                raise ValueError(f"图像 {image.original_name} 缺少可用于后处理的源图")
            mask_relative = self._resolve_relative_path(current_artifacts, "mask_path", "mask_url")
            if mask_relative is None:
                raise ValueError(f"图像 {image.original_name} 缺少当前确认 mask")

            source_image = read_gray(storage_service.absolute_path(source_relative))
            current_mask = read_gray(storage_service.absolute_path(mask_relative))
            result = traditional_service.apply_postprocess(source_image, current_mask, effective_config)

            stem = Path(image.original_name).stem
            confirmed_paths = self._resolve_target_paths(current_artifacts, run_id, str(manifest["mode"]), stem)
            if confirmed_paths.get("mask_path"):
                write_image(storage_service.absolute_path(confirmed_paths["mask_path"]), result["mask"])
            if confirmed_paths.get("overlay_path"):
                write_image(storage_service.absolute_path(confirmed_paths["overlay_path"]), result["overlay"])
            if confirmed_paths.get("edge_path"):
                write_image(storage_service.absolute_path(confirmed_paths["edge_path"]), result["edges"])
            if confirmed_paths.get("object_overlay_path"):
                write_image(storage_service.absolute_path(confirmed_paths["object_overlay_path"]), result["object_overlay"])

            metric.summary = self._build_preview_summary(
                current_summary=metric.summary,
                mask=result["mask"],
                objects=result["objects"],
                segment_config=effective_config,
                run_config=config,
                image=image,
                route_details=result.get("route_details"),
            )
            metric.artifacts = pipeline_service.build_metric_artifacts(
                input_relative=confirmed_paths.get("input_path"),
                processed_relative=confirmed_paths.get("processed_path"),
                analysis_relative=confirmed_paths.get("analysis_input_path"),
                footer_relative=confirmed_paths.get("footer_panel_path"),
                mask_relative=confirmed_paths.get("mask_path"),
                overlay_relative=confirmed_paths.get("overlay_path"),
                edge_relative=confirmed_paths.get("edge_path"),
                object_overlay_relative=confirmed_paths.get("object_overlay_path"),
            )

        updated_config = dict(run.config or {})
        updated_config["postprocess"] = dict(manifest.get("postprocess") or {})
        run.config = updated_config

        db.commit()
        run = pipeline_service.rebuild_confirmed_outputs(db, run_id)
        shutil.rmtree(preview_dir, ignore_errors=True)
        return {
            "run": run,
            "mode": str(manifest["mode"]),
            "image_count": len(metrics),
        }

    def _list_mode_metrics(self, db: Session, run_id: int, mode: str) -> list[MetricRecord]:
        metrics = db.scalars(
            select(MetricRecord).where(
                MetricRecord.run_id == run_id,
                MetricRecord.mode == mode,
            )
        ).all()
        image_order = {image.id: image.sort_index for image in db.scalars(select(ImageAsset).where(ImageAsset.run_id == run_id)).all()}
        return sorted(metrics, key=lambda item: image_order.get(item.image_id or -1, 0))

    def _resolve_source_relative(self, artifacts: dict[str, Any]) -> str | None:
        for path_key, url_key in (
            ("processed_path", "processed_url"),
            ("analysis_input_path", "analysis_input_url"),
            ("input_path", "input_url"),
        ):
            relative = self._resolve_relative_path(artifacts, path_key, url_key)
            if relative:
                return relative
        return None

    def _resolve_relative_path(self, artifacts: dict[str, Any], path_key: str, url_key: str) -> str | None:
        path_value = artifacts.get(path_key)
        if isinstance(path_value, str) and path_value.strip():
            return path_value
        url_value = artifacts.get(url_key)
        if not isinstance(url_value, str) or not url_value.strip():
            return None
        prefix = f"{SETTINGS.static_url_prefix}/"
        if url_value.startswith(prefix):
            return url_value.removeprefix(prefix)
        return None

    def _resolve_target_paths(self, artifacts: dict[str, Any], run_id: int, mode: str, stem: str) -> dict[str, str | None]:
        output_dir = storage_service.run_subdir(run_id, "outputs", mode)

        def pick(path_key: str, url_key: str, fallback_name: str) -> str:
            relative = self._resolve_relative_path(artifacts, path_key, url_key)
            if relative:
                return relative
            return storage_service.relative_path(output_dir / fallback_name)

        return {
            "input_path": self._resolve_relative_path(artifacts, "input_path", "input_url"),
            "processed_path": self._resolve_relative_path(artifacts, "processed_path", "processed_url"),
            "analysis_input_path": self._resolve_relative_path(artifacts, "analysis_input_path", "analysis_input_url"),
            "footer_panel_path": self._resolve_relative_path(artifacts, "footer_panel_path", "footer_panel_url"),
            "mask_path": pick("mask_path", "mask_url", f"{stem}_mask.png"),
            "overlay_path": pick("overlay_path", "overlay_url", f"{stem}_overlay.png"),
            "edge_path": pick("edge_path", "edge_url", f"{stem}_edges.png"),
            "object_overlay_path": pick("object_overlay_path", "object_overlay_url", f"{stem}_objects.png"),
        }

    def _build_preview_summary(
        self,
        current_summary: dict[str, Any] | None,
        mask: Any,
        objects: list[dict[str, Any]],
        segment_config: Any,
        run_config: RunCreate,
        image: ImageAsset,
        route_details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        current_summary = dict(current_summary or {})
        calibration = self._resolve_effective_calibration(current_summary, run_config, image)
        effective_um_per_px = calibration["um_per_px"]
        summary = statistics_service.summarize(
            mask,
            um_per_px=float(effective_um_per_px) if effective_um_per_px else None,
            objects=objects,
            config=segment_config,
        )
        for key in (
            "traditional_method",
            "traditional_route_details",
            "applied_um_per_px",
            "calibrated",
            "calibration_source",
            "calibration_hint",
            "roi",
            "threshold",
        ):
            if key in current_summary:
                summary[key] = current_summary[key]
        summary["applied_um_per_px"] = effective_um_per_px
        summary["calibrated"] = bool(calibration["calibrated"])
        summary["calibration_source"] = calibration["source"]
        summary["image_name"] = current_summary.get("image_name") or image.original_name
        summary["postprocess_applied"] = True
        if route_details is not None:
            summary["traditional_route_details"] = route_details
        return summary

    def _resolve_effective_calibration(
        self,
        current_summary: dict[str, Any],
        config: RunCreate,
        image: ImageAsset,
    ) -> dict[str, Any]:
        current_value = current_summary.get("applied_um_per_px")
        if current_value is not None and float(current_value) > 0:
            return {
                "um_per_px": float(current_value),
                "calibrated": True,
                "source": str(current_summary.get("calibration_source") or "task_default"),
            }

        image_calibration_map = pipeline_service._build_image_calibration_map(config)
        return pipeline_service._resolve_image_calibration(image, image_calibration_map, config)

    def _build_selected_payload(self, item: dict[str, Any], current_artifacts: dict[str, Any]) -> dict[str, Any]:
        preview_artifacts = dict(item["preview_artifacts"])
        return {
            "image_id": int(item["image_id"]),
            "image_name": str(item["image_name"]),
            "before_mask_url": current_artifacts.get("mask_url"),
            "after_mask_url": preview_artifacts.get("mask_url"),
            "before_overlay_url": current_artifacts.get("overlay_url"),
            "after_overlay_url": preview_artifacts.get("overlay_url"),
            "before_object_overlay_url": current_artifacts.get("object_overlay_url"),
            "after_object_overlay_url": preview_artifacts.get("object_overlay_url"),
            "before_summary": dict(item.get("baseline_summary") or {}),
            "after_summary": dict(item["summary"] or {}),
        }
postprocess_preview_service = PostprocessPreviewService()
