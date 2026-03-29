from __future__ import annotations

import shutil
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
from fastapi import UploadFile

from backend.app.core.config import SETTINGS


class StorageService:
    def __init__(self) -> None:
        self.root = SETTINGS.storage_dir
        self.runs_root = self.root / "runs"
        self.exports_root = self.root / "exports"

    def run_dir(self, run_id: int) -> Path:
        return self.runs_root / f"run_{run_id}"

    def run_subdir(self, run_id: int, *parts: str) -> Path:
        path = self.run_dir(run_id).joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def delete_run_dir(self, run_id: int) -> None:
        target = self.run_dir(run_id)
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)

    def _sanitize_relative_upload_path(self, raw_path: str | None, fallback_name: str) -> PurePosixPath:
        candidate = (raw_path or fallback_name or "image.png").replace("\\", "/").strip()
        if not candidate:
            candidate = fallback_name or "image.png"
        candidate = candidate.lstrip("/").replace(":", "_")
        parts = [part for part in PurePosixPath(candidate).parts if part not in {"", ".", ".."}]
        if not parts:
            parts = [fallback_name or "image.png"]
        return PurePosixPath(*parts)

    def _dedupe_target_path(self, target: Path) -> Path:
        if not target.exists():
            return target
        stem = target.stem
        suffix = target.suffix
        parent = target.parent
        index = 2
        while True:
            candidate = parent / f"{stem}_{index}{suffix}"
            if not candidate.exists():
                return candidate
            index += 1

    def save_uploads(
        self,
        run_id: int,
        files: list[UploadFile],
        relative_paths: list[str] | None = None,
    ) -> list[tuple[str, str]]:
        saved: list[tuple[str, str]] = []
        input_dir = self.run_subdir(run_id, "input")
        for index, upload in enumerate(files):
            filename = upload.filename or f"image_{index}.png"
            requested_relative = relative_paths[index] if relative_paths and index < len(relative_paths) else filename
            safe_relative = self._sanitize_relative_upload_path(requested_relative, filename)
            target = self._dedupe_target_path(input_dir / Path(safe_relative.as_posix()))
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as buffer:
                shutil.copyfileobj(upload.file, buffer)
            display_name = safe_relative.as_posix()
            if target.name != safe_relative.name:
                display_name = str(PurePosixPath(*safe_relative.parts[:-1], target.name))
            saved.append((display_name, self.relative_path(target)))
        return saved

    def relative_path(self, path: str | Path) -> str:
        return Path(path).resolve().relative_to(self.root.resolve()).as_posix()

    def absolute_path(self, relative_path: str) -> Path:
        return self.root / relative_path

    def static_url(self, relative_path: str | None) -> str | None:
        if not relative_path:
            return None
        return f"{SETTINGS.static_url_prefix}/{relative_path.replace('\\', '/')}"

    def browser_preview_url(self, relative_path: str | None) -> str | None:
        if not relative_path:
            return None
        source = self.absolute_path(relative_path)
        suffix = source.suffix.lower()
        if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}:
            return self.static_url(relative_path)

        preview_relative = self._ensure_preview(relative_path)
        return self.static_url(preview_relative)

    def _ensure_preview(self, relative_path: str) -> str:
        source = self.absolute_path(relative_path)
        preview_dir = source.parent / "_preview"
        preview_dir.mkdir(parents=True, exist_ok=True)
        preview_path = preview_dir / f"{source.stem}.png"
        if not preview_path.exists():
            image = self._read_any_image(source)
            preview = self._to_uint8_preview(image)
            success, encoded = cv2.imencode(".png", preview)
            if not success:
                raise ValueError(f"无法生成预览图: {source}")
            encoded.tofile(str(preview_path))
        return self.relative_path(preview_path)

    def _read_any_image(self, path: Path) -> np.ndarray:
        data = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"无法读取图像: {path}")
        return image

    def _to_uint8_preview(self, image: np.ndarray) -> np.ndarray:
        preview = image
        if preview.dtype != np.uint8:
            preview = cv2.normalize(preview, None, 0, 255, cv2.NORM_MINMAX)
            preview = preview.astype(np.uint8)
        if preview.ndim == 2:
            return preview
        if preview.shape[2] == 4:
            return cv2.cvtColor(preview, cv2.COLOR_BGRA2BGR)
        return preview


storage_service = StorageService()
