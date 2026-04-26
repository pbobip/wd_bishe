from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.base import Base


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )


class RunTask(TimestampMixin, Base):
    __tablename__ = "run_tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    input_mode: Mapped[str] = mapped_column(String(20), nullable=False)
    segmentation_mode: Mapped[str] = mapped_column(String(20), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="draft", nullable=False)
    progress: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    summary: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    chart_data: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    export_bundle_path: Mapped[str | None] = mapped_column(String(400), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    images: Mapped[list["ImageAsset"]] = relationship(back_populates="run", cascade="all, delete-orphan")
    steps: Mapped[list["RunStep"]] = relationship(back_populates="run", cascade="all, delete-orphan")
    metrics: Mapped[list["MetricRecord"]] = relationship(back_populates="run", cascade="all, delete-orphan")
    exports: Mapped[list["ExportRecord"]] = relationship(back_populates="run", cascade="all, delete-orphan")


class ImageAsset(Base):
    __tablename__ = "image_assets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("run_tasks.id"), nullable=False)
    original_name: Mapped[str] = mapped_column(String(255), nullable=False)
    relative_path: Mapped[str] = mapped_column(String(400), nullable=False)
    sort_index: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)

    run: Mapped["RunTask"] = relationship(back_populates="images")
    metrics: Mapped[list["MetricRecord"]] = relationship(back_populates="image")


class RunStep(Base):
    __tablename__ = "run_steps"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("run_tasks.id"), nullable=False)
    step_key: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    details: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    run: Mapped["RunTask"] = relationship(back_populates="steps")


class ModelRunner(TimestampMixin, Base):
    __tablename__ = "model_runners"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    slot: Mapped[str] = mapped_column(String(32), unique=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String(120), nullable=False)
    python_path: Mapped[str] = mapped_column(String(400), nullable=False)
    env_name: Mapped[str | None] = mapped_column(String(80), nullable=True)
    weight_path: Mapped[str | None] = mapped_column(String(400), nullable=True)
    extra_config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)


class MetricRecord(Base):
    __tablename__ = "metric_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("run_tasks.id"), nullable=False)
    image_id: Mapped[int | None] = mapped_column(ForeignKey("image_assets.id"), nullable=True)
    mode: Mapped[str] = mapped_column(String(32), nullable=False)
    summary: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    artifacts: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)

    run: Mapped["RunTask"] = relationship(back_populates="metrics")
    image: Mapped["ImageAsset"] = relationship(back_populates="metrics")


class ExportRecord(Base):
    __tablename__ = "export_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("run_tasks.id"), nullable=False)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    relative_path: Mapped[str] = mapped_column(String(400), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)

    run: Mapped["RunTask"] = relationship(back_populates="exports")
