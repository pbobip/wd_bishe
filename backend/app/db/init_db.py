from __future__ import annotations

from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.core.config import SETTINGS
from backend.app.db.base import Base
from backend.app.db.session import engine
from backend.app.models.entities import ModelRunner, Project


DEFAULT_RUNNERS = [
    {
        "slot": "sam_lora",
        "display_name": "SAM LoRA",
        "python_path": r"C:\Users\pyd111\anaconda3\envs\sam_gpu\python.exe",
        "env_name": "sam_gpu",
        "weight_path": "",
        "extra_config": {"model_kind": "sam_lora"},
    },
    {
        "slot": "resnext50",
        "display_name": "ResNeXt50",
        "python_path": r"C:\Users\pyd111\anaconda3\envs\torch311\python.exe",
        "env_name": "torch311",
        "weight_path": "",
        "extra_config": {"model_kind": "resnext50"},
    },
    {
        "slot": "matsam",
        "display_name": "MatSAM",
        "python_path": r"C:\Users\pyd111\anaconda3\envs\sam_gpu\python.exe",
        "env_name": "sam_gpu",
        "weight_path": "",
        "extra_config": {"model_kind": "matsam"},
    },
    {
        "slot": "custom",
        "display_name": "Custom Runner",
        "python_path": r"C:\Users\pyd111\anaconda3\envs\aavt_gpu\python.exe",
        "env_name": "aavt_gpu",
        "weight_path": "",
        "extra_config": {"model_kind": "custom"},
    },
]


def init_storage() -> None:
    for path in [
        SETTINGS.storage_dir,
        SETTINGS.storage_dir / "runs",
        SETTINGS.storage_dir / "exports",
        SETTINGS.storage_dir / "charts",
        SETTINGS.storage_dir / "tmp",
    ]:
        Path(path).mkdir(parents=True, exist_ok=True)


def init_db() -> None:
    init_storage()
    Base.metadata.create_all(bind=engine)
    with Session(engine) as session:
        seed_defaults(session)


def seed_defaults(session: Session) -> None:
    if session.scalar(select(Project.id).limit(1)) is None:
        session.add(Project(name="默认项目", description="毕设系统默认项目"))
    existing_slots = {slot for slot in session.scalars(select(ModelRunner.slot)).all()}
    for runner in DEFAULT_RUNNERS:
        if runner["slot"] in existing_slots:
            continue
        session.add(ModelRunner(**runner))
    session.commit()
