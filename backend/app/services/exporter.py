from __future__ import annotations

import json
import zipfile
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from backend.app.models.entities import ExportRecord, MetricRecord, RunTask
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

        csv_path = export_dir / "metrics.csv"
        xlsx_path = export_dir / "metrics.xlsx"
        batch_csv_path = export_dir / "batch_summary.csv"
        batch_xlsx_path = export_dir / "batch_summary.xlsx"
        particle_csv_path = export_dir / "particles.csv"
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
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        df.to_excel(xlsx_path, index=False)
        batch_df = pd.DataFrame(batch_rows)
        batch_df.to_csv(batch_csv_path, index=False, encoding="utf-8-sig")
        batch_df.to_excel(batch_xlsx_path, index=False)
        particle_df = pd.DataFrame(particle_rows)
        particle_df.to_csv(particle_csv_path, index=False, encoding="utf-8-sig")
        particle_df.to_excel(particle_xlsx_path, index=False)
        json_path.write_text(json.dumps(config_snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

        bundle_path = export_dir / f"run_{run_id}_bundle.zip"
        with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(csv_path, arcname=csv_path.name)
            archive.write(xlsx_path, arcname=xlsx_path.name)
            archive.write(batch_csv_path, arcname=batch_csv_path.name)
            archive.write(batch_xlsx_path, arcname=batch_xlsx_path.name)
            archive.write(particle_csv_path, arcname=particle_csv_path.name)
            archive.write(particle_xlsx_path, arcname=particle_xlsx_path.name)
            archive.write(json_path, arcname=json_path.name)
            for chart_name, relative in chart_paths.items():
                absolute = storage_service.absolute_path(relative)
                if absolute.exists():
                    archive.write(absolute, arcname=f"charts/{chart_name}{absolute.suffix}")
            outputs_dir = storage_service.run_subdir(run_id, "outputs")
            if outputs_dir.exists():
                for file in outputs_dir.rglob("*"):
                    if file.is_file():
                        archive.write(file, arcname=f"outputs/{file.relative_to(outputs_dir).as_posix()}")

        exports = {
            "csv": storage_service.relative_path(csv_path),
            "xlsx": storage_service.relative_path(xlsx_path),
            "batch_csv": storage_service.relative_path(batch_csv_path),
            "batch_xlsx": storage_service.relative_path(batch_xlsx_path),
            "particles_csv": storage_service.relative_path(particle_csv_path),
            "particles_xlsx": storage_service.relative_path(particle_xlsx_path),
            "config": storage_service.relative_path(json_path),
            "bundle": storage_service.relative_path(bundle_path),
        }
        for kind, relative in exports.items():
            db.add(ExportRecord(run_id=run_id, kind=kind, relative_path=relative))
        db.commit()
        return exports


export_service = ExportService()
