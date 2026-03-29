from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime

from backend.app.core.config import SETTINGS
from backend.app.db.session import SessionLocal
from backend.app.models.entities import RunTask
from backend.app.services.pipeline import pipeline_service


class TaskManager:
    def __init__(self) -> None:
        self.executor = ThreadPoolExecutor(max_workers=SETTINGS.executor_workers, thread_name_prefix="run-task")
        self.futures: dict[int, Future[None]] = {}

    def submit(self, run_id: int) -> None:
        future = self.executor.submit(self._execute, run_id)
        self.futures[run_id] = future

    def _execute(self, run_id: int) -> None:
        with SessionLocal() as db:
            try:
                pipeline_service.execute(db, run_id)
            except Exception as exc:
                run = db.get(RunTask, run_id)
                if run is not None:
                    run.status = "failed"
                    run.error_message = str(exc)
                    run.finished_at = datetime.utcnow()
                    db.commit()
                raise


task_manager = TaskManager()
