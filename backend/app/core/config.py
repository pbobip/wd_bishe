from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    backend_dir: Path
    storage_dir: Path
    db_path: Path
    static_url_prefix: str = "/static"
    api_prefix: str = "/api"
    executor_workers: int = 2

    @property
    def database_url(self) -> str:
        return f"sqlite:///{self.db_path.as_posix()}"


BACKEND_DIR = Path(__file__).resolve().parents[2]
STORAGE_DIR = BACKEND_DIR / "storage"
SETTINGS = Settings(
    backend_dir=BACKEND_DIR,
    storage_dir=STORAGE_DIR,
    db_path=STORAGE_DIR / "platform.db",
)
