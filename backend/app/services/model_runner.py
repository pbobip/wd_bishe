from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.core.config import SETTINGS
from backend.app.models.entities import ModelRunner

logger = logging.getLogger(__name__)

# Default 10 minutes per image for deep learning inference
_DEFAULT_TIMEOUT_PER_IMAGE_SEC = 600


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
        cmd = [runner.python_path, str(script_path), str(config_path)]
        logger.info("[ModelRunner] Starting inference: run_id=%s, model_kind=%s, cmd=%s", payload["run_id"], payload["model_kind"], cmd)

        stdout_lines: list[str] = []
        stderr_lines: list[str] = []

        def stream_logger(stream, lines_list, label):
            for line in iter(stream.readline, ""):
                if line:
                    lines_list.append(line.rstrip())
                    logger.debug("[ModelRunner][%s] %s", label, line.rstrip())

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(SETTINGS.backend_dir),
        )

        stdout_thread = threading.Thread(target=stream_logger, args=(process.stdout, stdout_lines, "stdout"), daemon=True)
        stderr_thread = threading.Thread(target=stream_logger, args=(process.stderr, stderr_lines, "stderr"), daemon=True)
        stdout_thread.start()
        stderr_thread.start()

        total_items = len(payload.get("items", []))
        timeout_per_image = _DEFAULT_TIMEOUT_PER_IMAGE_SEC
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            timeout_per_image = int(cfg.get("timeout_per_image", _DEFAULT_TIMEOUT_PER_IMAGE_SEC))
        except (OSError, json.JSONDecodeError, ValueError):
            pass
        timeout_sec = timeout_per_image * max(total_items, 1)
        timeout_sec = max(timeout_sec, 60)

        progress_signature: tuple[int, int] | None = None
        progress_path = Path(payload["progress_path"]) if payload.get("progress_path") else None

        try:
            start_time = time.monotonic()
            while process.poll() is None:
                if time.monotonic() - start_time > timeout_sec:
                    logger.error("[ModelRunner] Inference timed out after %ds (cmd=%s)", timeout_sec, cmd)
                    process.kill()
                    process.wait()
                    raise TimeoutError(f"深度学习推理超时（{timeout_sec}秒），进程被强制终止。常见原因：模型加载卡住、CUDA OOM、大图像滑窗推理过慢。请检查权重文件路径是否正确，或尝试缩小图像/增大 timeout_per_image 配置。")
                progress_signature = self._emit_progress(progress_path, progress_callback, progress_signature)
                time.sleep(0.2)

            process.wait()
        except TimeoutError:
            raise
        except Exception as exc:
            logger.error("[ModelRunner] Unexpected error during inference: %s", exc)
            if process.poll() is None:
                process.kill()
                process.wait()
            raise

        stdout_thread.join(timeout=2)
        stderr_thread.join(timeout=2)

        if process.returncode != 0:
            stderr_text = "\n".join(stderr_lines[-50:]) if stderr_lines else ""
            stdout_text = "\n".join(stdout_lines[-20:]) if stdout_lines else ""
            logger.error("[ModelRunner] Inference failed (code=%d):\nstderr: %s\nstdout: %s", process.returncode, stderr_text, stdout_text)
            raise RuntimeError(stderr_text.strip() or stdout_text.strip() or f"深度学习推理失败 (exit code: {process.returncode})")

        logger.info("[ModelRunner] Inference completed successfully for run_id=%s", payload["run_id"])
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
