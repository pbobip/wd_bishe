"""
补生成 Word 实验报告脚本

用法（服务器上运行）:
  # 补生成全部已完成任务的 Word 报告
  python scripts/regenerate_report.py

  # 指定任务 ID 补生成
  python scripts/regenerate_report.py --run-id 42

  # 仅预览，不写入
  python scripts/regenerate_report.py --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.db.session import SessionLocal
from backend.app.models.entities import ExportRecord, MetricRecord, RunTask
from backend.app.services.report_service import generate_docx_report
from backend.app.services.storage import storage_service


def _rebuild_chart_paths(run: RunTask) -> dict[str, str]:
    """
    从 run.chart_data 中重建 chart_paths。
    chart_data 结构:
      {
        "traditional": {
            "area_hist_url": "...",
            "size_hist_url": "...",
            "vf_bar_url": "...",
        },
        "dl": { ... }
      }
    """
    chart_data = run.chart_data or {}
    chart_paths: dict[str, str] = {}
    for mode, mode_data in chart_data.items():
        if not isinstance(mode_data, dict):
            continue
        for key_suffix, url_key in [
            ("area_hist", "area_hist_url"),
            ("size_hist", "size_hist_url"),
            ("vf_bar", "vf_bar_url"),
        ]:
            url = mode_data.get(url_key)
            if url and isinstance(url, str):
                # url 格式: /static/runs/run_{id}/charts/{mode}_{key_suffix}.png
                # 转换为相对路径: runs/run_{id}/charts/{mode}_{key_suffix}.png
                if url.startswith("/static/"):
                    rel = url[len("/static/"):]
                else:
                    rel = url.lstrip("/")
                chart_paths[f"{mode}_{key_suffix}"] = rel
    return chart_paths


def regenerate_for_run(run_id: int, dry_run: bool = False) -> bool:
    db = SessionLocal()
    try:
        run = db.get(RunTask, run_id)
        if run is None:
            print(f"[ERROR] 任务 {run_id} 不存在")
            return False

        if run.status not in ("completed", "partial_success"):
            print(f"[SKIP] 任务 {run_id} 状态={run.status}，跳过（需先完成执行）")
            return False

        metrics = db.query(MetricRecord).filter(MetricRecord.run_id == run_id).all()
        config_snapshot = run.config or {}
        chart_paths = _rebuild_chart_paths(run)

        print(f"[INFO] 任务 {run_id}：{run.name}")
        print(f"       统计记录数：{len(metrics)}，图表路径数：{len(chart_paths)}")

        if dry_run:
            print(f"       [DRY-RUN] 跳过实际生成")
            return True

        docx_relative = generate_docx_report(
            db=db,
            run=run,
            metrics=metrics,
            config_snapshot=config_snapshot,
            chart_paths=chart_paths,
        )
        print(f"[OK]   Word 报告已生成: {docx_relative}")

        # 写入 ExportRecord 到数据库
        db.add(ExportRecord(run_id=run_id, kind="docx_report", relative_path=docx_relative))
        db.commit()

        # 追加到 bundle.zip
        docx_abs = storage_service.absolute_path(docx_relative)
        if docx_abs.exists():
            bundle_path = storage_service.run_subdir(run_id, "exports") / f"run_{run_id}_bundle.zip"
            if bundle_path.exists():
                import zipfile
                with zipfile.ZipFile(bundle_path, "a", zipfile.ZIP_DEFLATED) as archive:
                    archive.write(docx_abs, arcname="experiment_report.docx")
                print(f"[OK]   已追加到 bundle: {bundle_path.name}")

        return True

    except Exception as exc:
        print(f"[ERROR] 生成失败: {exc}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="补生成 Word 实验报告")
    parser.add_argument("--run-id", type=int, default=None, help="指定任务 ID（不指定则遍历全部已完成任务）")
    parser.add_argument("--dry-run", action="store_true", help="仅预览，不实际生成文件")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        if args.run_id is not None:
            regenerate_for_run(args.run_id, dry_run=args.dry_run)
            return

        runs = db.query(RunTask).filter(
            RunTask.status.in_(["completed", "partial_success"])
        ).order_by(RunTask.id.desc()).all()

        if not runs:
            print("[INFO] 没有已完成的任务")
            return

        print(f"[INFO] 找到 {len(runs)} 个已完成任务，开始补生成...\n")
        success = 0
        failed = 0
        skipped = 0

        for run in runs:
            result = regenerate_for_run(run.id, dry_run=args.dry_run)
            if result:
                success += 1
            elif run.status not in ("completed", "partial_success"):
                skipped += 1
            else:
                failed += 1

        print(f"\n[DONE] 成功 {success}，失败 {failed}，跳过 {skipped}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
