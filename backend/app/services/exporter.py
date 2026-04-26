from __future__ import annotations

import json
import pathlib
import zipfile
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from backend.app.models.entities import ExportRecord, MetricRecord, RunTask
from backend.app.services.report_service import generate_docx_report
from backend.app.services.storage import storage_service


class ExportService:
    _BATCH_CI_FIELDS = {
        "vf": "volume_fraction_ci95",
        "particle_count": "particle_count_ci95",
        "image_mean_area": "image_mean_area_ci95",
        "image_mean_size": "image_mean_size_ci95",
        "image_mean_channel_width_x": "image_mean_channel_width_x_ci95",
        "image_mean_channel_width_y": "image_mean_channel_width_y_ci95",
    }

    def export_run(
        self,
        db: Session,
        run_id: int,
        config_snapshot: dict[str, Any],
        chart_paths: dict[str, str],
        *,
        replace_existing: bool = False,
    ) -> dict[str, str]:
        export_dir = storage_service.run_subdir(run_id, "exports")
        run = db.get(RunTask, run_id)
        metrics = db.query(MetricRecord).filter(MetricRecord.run_id == run_id).all()
        rows: list[dict[str, Any]] = []
        particle_rows: list[dict[str, Any]] = []
        batch_rows: list[dict[str, Any]] = []
        for record in metrics:
            summary = record.summary.copy()
            summary["mode"] = record.mode
            summary["image_id"] = record.image_id
            rows.append(summary)
            objects = summary.get("objects") or summary.get("particles", [])
            for particle in objects:
                particle_rows.append(
                    {
                        "run_id": run_id,
                        "image_id": record.image_id,
                        "image_name": summary.get("image_name"),
                        "mode": record.mode,
                        "object_label": particle.get("label"),
                        "particle_label": particle.get("label"),
                        "area_px": particle.get("area_px"),
                        "area_value": particle.get("area_value"),
                        "area_unit": summary.get("area_unit"),
                        "size_value": particle.get("size_value"),
                        "size_unit": summary.get("size_unit"),
                        "equiv_diameter": particle.get("equiv_diameter"),
                        "diameter_unit": summary.get("diameter_unit"),
                        "perimeter": particle.get("perimeter"),
                        "perimeter_value": particle.get("perimeter_value"),
                        "perimeter_unit": summary.get("perimeter_unit"),
                        "major": particle.get("major"),
                        "major_value": particle.get("major_value"),
                        "minor": particle.get("minor"),
                        "minor_value": particle.get("minor_value"),
                        "feret": particle.get("feret"),
                        "feret_value": particle.get("feret_value"),
                        "minferet": particle.get("minferet"),
                        "minferet_value": particle.get("minferet_value"),
                        "aspect_ratio": particle.get("aspect_ratio"),
                        "circularity": particle.get("circularity"),
                        "roundness": particle.get("roundness"),
                        "solidity": particle.get("solidity"),
                        "bbox_x": particle.get("bbox_x"),
                        "bbox_y": particle.get("bbox_y"),
                        "bbox_w": particle.get("bbox_w"),
                        "bbox_h": particle.get("bbox_h"),
                        "centroid_x": particle.get("centroid_x"),
                        "centroid_y": particle.get("centroid_y"),
                        "filtered": particle.get("filtered"),
                        "filter_reason": particle.get("filter_reason"),
                    }
                )

        xlsx_path = export_dir / "metrics.xlsx"
        batch_xlsx_path = export_dir / "batch_summary.xlsx"
        particle_xlsx_path = export_dir / "particles.xlsx"
        json_path = export_dir / "config_snapshot.json"

        if run and isinstance(run.summary, dict):
            for mode, summary in (run.summary.get("batch") or {}).items():
                if not isinstance(summary, dict):
                    continue
                row = {
                    "run_id": run_id,
                    "mode": mode,
                    "image_count": summary.get("image_count"),
                    "avg_volume_fraction": summary.get("avg_volume_fraction"),
                    "avg_particle_count": summary.get("avg_particle_count"),
                    "avg_image_mean_area": summary.get("avg_image_mean_area"),
                    "avg_image_mean_size": summary.get("avg_image_mean_size"),
                    "avg_image_mean_channel_width_x": summary.get("avg_image_mean_channel_width_x"),
                    "avg_image_mean_channel_width_y": summary.get("avg_image_mean_channel_width_y"),
                    "avg_object_count": summary.get("avg_object_count"),
                    "avg_filtered_object_count": summary.get("avg_filtered_object_count"),
                    "particle_count_total": summary.get("particle_count_total"),
                    "object_count_total": summary.get("object_count_total"),
                    "filtered_object_count_total": summary.get("filtered_object_count_total"),
                    "area_unit": summary.get("area_unit"),
                    "size_unit": summary.get("size_unit"),
                    "channel_width_unit": summary.get("channel_width_unit"),
                }
                for prefix, field_name in self._BATCH_CI_FIELDS.items():
                    ci = summary.get(field_name) or {}
                    row[f"{prefix}_ci95_available"] = ci.get("available")
                    row[f"{prefix}_ci95_n"] = ci.get("n")
                    row[f"{prefix}_ci95_mean"] = ci.get("mean")
                    row[f"{prefix}_ci95_lower"] = ci.get("lower")
                    row[f"{prefix}_ci95_upper"] = ci.get("upper")
                    row[f"{prefix}_ci95_half_width"] = ci.get("half_width")
                    row[f"{prefix}_ci95_method"] = ci.get("method")
                    row[f"{prefix}_ci95_reason"] = ci.get("reason")
                batch_rows.append(row)

        df = pd.DataFrame(rows)
        df.to_excel(xlsx_path, index=False)
        batch_df = pd.DataFrame(batch_rows)
        batch_df.to_excel(batch_xlsx_path, index=False)
        particle_df = pd.DataFrame(particle_rows)
        particle_df.to_excel(particle_xlsx_path, index=False)
        json_path.write_text(json.dumps(config_snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

        outputs_dir = storage_service.run_subdir(run_id, "outputs")
        overlay_files: list[str] = []
        mask_files: list[str] = []

        if outputs_dir.exists():
            for file in outputs_dir.rglob("*"):
                if file.is_file() and file.suffix.lower() in (".png", ".jpg", ".jpeg"):
                    name = file.stem.lower()
                    if "_overlay" in name or name.endswith("overlay"):
                        overlay_files.append(str(file))
                    elif "_mask" in name or name.endswith("mask"):
                        mask_files.append(str(file))

        overlay_zip_path: pathlib.Path | None = None
        mask_zip_path: pathlib.Path | None = None

        if overlay_files:
            overlay_zip_path = export_dir / "segmentation_overlays.zip"
            with zipfile.ZipFile(overlay_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for file_path in sorted(overlay_files):
                    p = pathlib.Path(file_path)
                    zf.write(p, arcname=f"segmentation_overlays/{p.name}")

        if mask_files:
            mask_zip_path = export_dir / "segmentation_masks.zip"
            with zipfile.ZipFile(mask_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for file_path in sorted(mask_files):
                    p = pathlib.Path(file_path)
                    zf.write(p, arcname=f"segmentation_masks/{p.name}")

        bundle_path = export_dir / f"run_{run_id}_bundle.zip"
        with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(xlsx_path, arcname=xlsx_path.name)
            archive.write(batch_xlsx_path, arcname=batch_xlsx_path.name)
            archive.write(particle_xlsx_path, arcname=particle_xlsx_path.name)
            archive.write(json_path, arcname=json_path.name)
            for chart_name, relative in chart_paths.items():
                absolute = storage_service.absolute_path(relative)
                if absolute.exists():
                    archive.write(absolute, arcname=f"charts/{chart_name}{absolute.suffix}")
            if outputs_dir.exists():
                for file in outputs_dir.rglob("*"):
                    if file.is_file():
                        archive.write(file, arcname=f"outputs/{file.relative_to(outputs_dir).as_posix()}")

        exports = {
            "xlsx": storage_service.relative_path(xlsx_path),
            "batch_xlsx": storage_service.relative_path(batch_xlsx_path),
            "particles_xlsx": storage_service.relative_path(particle_xlsx_path),
            "config": storage_service.relative_path(json_path),
            "bundle": storage_service.relative_path(bundle_path),
        }
        if overlay_zip_path:
            exports["overlay"] = storage_service.relative_path(overlay_zip_path)
        if mask_zip_path:
            exports["mask"] = storage_service.relative_path(mask_zip_path)

        # 将 Word 报告也加入 bundle
        try:
            docx_relative = generate_docx_report(
                db=db,
                run=run,
                metrics=metrics,
                config_snapshot=config_snapshot,
                chart_paths=chart_paths,
            )
            exports["docx_report"] = docx_relative
            # 写入 bundle
            docx_abs = storage_service.absolute_path(docx_relative)
            if docx_abs.exists():
                archive = zipfile.ZipFile(bundle_path, "a", zipfile.ZIP_DEFLATED)
                archive.write(docx_abs, arcname="experiment_report.docx")
                archive.close()
        except Exception:
            # Word 报告生成失败不影响其他导出
            pass

        if replace_existing:
            db.query(ExportRecord).filter(ExportRecord.run_id == run_id).delete()
        for kind, relative in exports.items():
            db.add(ExportRecord(run_id=run_id, kind=kind, relative_path=relative))
        db.commit()
        return exports


export_service = ExportService()
