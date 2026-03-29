from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.core.config import SETTINGS
from backend.app.models.entities import ModelRunner


class ModelRunnerService:
    def get_runner(self, db: Session, slot: str, runner_id: int | None = None) -> ModelRunner:
        if runner_id is not None:
            runner = db.get(ModelRunner, runner_id)
        else:
            runner = db.scalar(select(ModelRunner).where(ModelRunner.slot == slot, ModelRunner.is_active.is_(True)))
        if runner is None:
            raise ValueError(f"未找到可用模型运行器: {slot}")
        return runner

    def run_inference(
        self,
        db: Session,
        slot: str,
        payload: dict[str, Any],
        runner_id: int | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        runner = self.get_runner(db, slot, runner_id)
        config_path = SETTINGS.storage_dir / "tmp" / f"runner_{payload['run_id']}_{slot}.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            **payload,
            "weight_path": payload.get("weight_path") or runner.weight_path,
            "model_kind": runner.extra_config.get("model_kind", slot),
        }
        config_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

        script_path = Path(__file__).resolve().parent / "runners" / "dl_infer.py"
        process = subprocess.Popen(
            [runner.python_path, str(script_path), str(config_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            cwd=str(SETTINGS.backend_dir),
        )
        stdout = ""
        stderr = ""
        progress_signature: tuple[int, int] | None = None
        progress_path = Path(payload["progress_path"]) if payload.get("progress_path") else None
        while process.poll() is None:
            progress_signature = self._emit_progress(progress_path, progress_callback, progress_signature)
            time.sleep(0.2)
        stdout, stderr = process.communicate()
        progress_signature = self._emit_progress(progress_path, progress_callback, progress_signature)
        if process.returncode != 0:
            raise RuntimeError(stderr.strip() or stdout.strip() or "深度学习推理失败")
        return json.loads(Path(payload["manifest_path"]).read_text(encoding="utf-8"))

    def _emit_progress(
        self,
        progress_path: Path | None,
        progress_callback: Callable[[dict[str, Any]], None] | None,
        last_signature: tuple[int, int] | None,
    ) -> tuple[int, int] | None:
        if progress_path is None or progress_callback is None or not progress_path.exists():
            return last_signature
        try:
            stat = progress_path.stat()
            signature = (int(stat.st_mtime_ns), int(stat.st_size))
            if signature == last_signature:
                return last_signature
            payload = json.loads(progress_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return last_signature
        progress_callback(payload)
        return signature


model_runner_service = ModelRunnerService()
