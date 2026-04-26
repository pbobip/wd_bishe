from __future__ import annotations

import logging
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.api.routes import router
from backend.app.core.config import SETTINGS
from backend.app.db.init_db import init_db


def _configure_logging() -> None:
    log_dir = SETTINGS.storage_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "backend.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def create_app() -> FastAPI:
    _configure_logging()
    init_db()
    app = FastAPI(title="SEM 分割统计平台", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.mount(SETTINGS.static_url_prefix, StaticFiles(directory=SETTINGS.storage_dir), name="static")
    app.include_router(router, prefix=SETTINGS.api_prefix)
    return app


app = create_app()
