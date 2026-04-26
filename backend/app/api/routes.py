from __future__ import annotations

import base64
import json

import cv2
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.db.session import get_db
from backend.app.models.entities import ExportRecord, ImageAsset, MetricRecord, ModelRunner, RunTask
from backend.app.schemas.run import (
    ImageAssetRead,
    ModelRunnerCreate,
    ModelRunnerRead,
    PostprocessConfirmCreate,
    PostprocessConfirmRead,
    PreprocessConfig,
    PostprocessPreviewCreate,
    PostprocessPreviewRead,
    RunCreate,
    RunRead,
    RunResultResponse,
    RunStepRead,
)
from backend.app.services.postprocess_preview import (
    PreviewConflictError,
    PreviewNotFoundError,
    postprocess_preview_service,
)
from backend.app.services.sem_footer_ocr import sem_footer_ocr_service
from backend.app.services.preprocess import preprocess_service
from backend.app.services.sem_roi import sem_roi_service
from backend.app.services.storage import storage_service
from backend.app.services.task_manager import task_manager
from backend.app.utils.image_io import decode_gray_bytes


router = APIRouter()


def _encode_preview_data_url(image, max_edge: int | None = None) -> str:
    preview = image
    if max_edge and max_edge > 0:
        height, width = image.shape[:2]
        longest_edge = max(height, width)
        if longest_edge > max_edge:
            scale = max_edge / float(longest_edge)
            preview = cv2.resize(
                image,
                (max(1, int(width * scale)), max(1, int(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
    ok, encoded = cv2.imencode(".png", preview)
    if not ok:
        raise ValueError("无法编码预览图像")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/png;base64,{payload}"


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/runs", response_model=list[RunRead])
def list_runs(db: Session = Depends(get_db)) -> list[RunTask]:
    return db.scalars(select(RunTask).order_by(RunTask.created_at.desc())).all()


@router.post("/runs", response_model=RunRead)
def create_run(payload: RunCreate, db: Session = Depends(get_db)) -> RunTask:
    run = RunTask(
        name=payload.name,
        input_mode=payload.input_mode,
        segmentation_mode=payload.segmentation_mode,
        config=payload.model_dump(),
        status="draft",
        progress=0.0,
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    storage_service.run_subdir(run.id, "input")
    return run


@router.put("/runs/{run_id}", response_model=RunRead)
def update_run(run_id: int, payload: RunCreate, db: Session = Depends(get_db)) -> RunTask:
    run = db.get(RunTask, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    if run.status not in {"draft", "queued"}:
        raise HTTPException(status_code=400, detail="当前任务状态不允许更新配置")

    run.name = payload.name
    run.input_mode = payload.input_mode
    run.segmentation_mode = payload.segmentation_mode
    run.config = payload.model_dump()
    run.error_message = None
    if run.status == "queued":
        run.status = "draft"
        run.progress = 0.0
    db.commit()
    db.refresh(run)
    return run


@router.post("/runs/{run_id}/images", response_model=list[ImageAssetRead])
async def upload_images(
    run_id: int,
    files: list[UploadFile] = File(...),
    relative_paths: list[str] = Form(default=[]),
    db: Session = Depends(get_db),
) -> list[ImageAsset]:
    run = db.get(RunTask, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    if run.status != "draft":
        raise HTTPException(status_code=400, detail="只有草稿任务允许继续上传图像")
    if not files:
        raise HTTPException(status_code=400, detail="未选择图像")
    saved = storage_service.save_uploads(run_id, files, relative_paths)
    existing_count = db.query(ImageAsset).filter(ImageAsset.run_id == run_id).count()
    created: list[ImageAsset] = []
    for index, (name, relative_path) in enumerate(saved, start=existing_count):
        asset = ImageAsset(run_id=run_id, original_name=name, relative_path=relative_path, sort_index=index)
        db.add(asset)
        created.append(asset)
    db.commit()
    for asset in created:
        db.refresh(asset)
    return created


@router.post("/calibration/inspect")
@router.post("/calibration-hint")
async def calibration_hint(file: UploadFile = File(...)) -> dict[str, object]:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="未读取到图像内容")

    try:
        image = decode_gray_bytes(content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    roi = sem_roi_service.extract(image)
    scale_bar = sem_roi_service.detect_scale_bar(roi.footer_image)
    footer_ocr = sem_footer_ocr_service.analyze(
        roi.footer_image,
        scale_bar_hint_bbox=scale_bar.bbox if scale_bar is not None else None,
    )
    common_scales = []
    if scale_bar and scale_bar.pixel_length > 0:
        for scale_um in (0.5, 1.0, 2.0, 5.0, 10.0):
            common_scales.append(
                {
                    "scale_um": scale_um,
                    "um_per_px": round(scale_um / scale_bar.pixel_length, 8),
                }
            )
    suggested_um_per_px = None
    if scale_bar and footer_ocr and footer_ocr.scale_bar_um:
        suggested_um_per_px = round(footer_ocr.scale_bar_um / scale_bar.pixel_length, 8)

    return {
        "file_name": file.filename,
        "preview_url": _encode_preview_data_url(image, max_edge=320),
        "footer_detected": roi.cropped_footer,
        "background_cropped": roi.cropped_background,
        "source_width_px": int(roi.source_shape[1]),
        "source_height_px": int(roi.source_shape[0]),
        "analysis_width_px": int(roi.analysis_bbox[2]),
        "analysis_height_px": int(roi.analysis_bbox[3]),
        "source_shape": {"width": int(roi.source_shape[1]), "height": int(roi.source_shape[0])},
        "analysis_bbox": {
            "x": int(roi.analysis_bbox[0]),
            "y": int(roi.analysis_bbox[1]),
            "width": int(roi.analysis_bbox[2]),
            "height": int(roi.analysis_bbox[3]),
        },
        "footer_bbox": (
            {
                "x": int(roi.footer_bbox[0]),
                "y": int(roi.footer_bbox[1]),
                "width": int(roi.footer_bbox[2]),
                "height": int(roi.footer_bbox[3]),
            }
            if roi.footer_bbox
            else None
        ),
        "scale_bar_detected": bool(scale_bar),
        "scale_bar_pixels": int(scale_bar.pixel_length) if scale_bar else None,
        "scale_bar_bbox": (
            {
                "x": int(scale_bar.bbox[0]),
                "y": int(scale_bar.bbox[1]),
                "width": int(scale_bar.bbox[2]),
                "height": int(scale_bar.bbox[3]),
            }
            if scale_bar
            else None
        ),
        "scale_bar_confidence": float(scale_bar.confidence) if scale_bar else 0.0,
        "ocr_available": sem_footer_ocr_service.available,
        "ocr_scale_bar_um": footer_ocr.scale_bar_um if footer_ocr else None,
        "ocr_fov_um": footer_ocr.fov_um if footer_ocr else None,
        "ocr_magnification_text": footer_ocr.magnification_text if footer_ocr else None,
        "ocr_wd_mm": footer_ocr.wd_mm if footer_ocr else None,
        "ocr_detector": footer_ocr.detector if footer_ocr else None,
        "ocr_scan_mode": footer_ocr.scan_mode if footer_ocr else None,
        "ocr_vacuum_mode": footer_ocr.vacuum_mode if footer_ocr else None,
        "ocr_date_text": footer_ocr.date_text if footer_ocr else None,
        "ocr_time_text": footer_ocr.time_text if footer_ocr else None,
        "suggested_um_per_px": suggested_um_per_px,
        "common_scale_candidates": common_scales,
        "message": (
            "已自动标定，可直接使用建议值。"
            if scale_bar and footer_ocr and footer_ocr.scale_bar_um
            else "未自动标定，请手动填写 um/px。"
        ),
    }


@router.post("/preprocess/preview")
async def preprocess_preview(
    file: UploadFile = File(...),
    preprocess: str = Form(...),
    auto_crop_sem_region: bool = Form(True),
) -> dict[str, object]:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="未读取到图像内容")

    try:
        image = decode_gray_bytes(content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        preprocess_config = PreprocessConfig.model_validate(json.loads(preprocess))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="预处理配置无法解析") from exc

    roi = sem_roi_service.extract(image)
    preview_source = roi.analysis_image if auto_crop_sem_region and roi.cropped_footer else image
    processed = preprocess_service.apply(preview_source, preprocess_config)

    return {
        "source_label": "分析区域" if auto_crop_sem_region and roi.cropped_footer else "原图",
        "footer_detected": roi.cropped_footer,
        "original_preview_url": _encode_preview_data_url(preview_source),
        "processed_preview_url": _encode_preview_data_url(processed),
        "message": "当前预处理尚未启用，预览图保持与输入一致。"
        if not preprocess_config.enabled
        else "已按当前配置生成预处理预览。",
    }


@router.post("/runs/{run_id}/execute", response_model=RunRead)
def execute_run(run_id: int, db: Session = Depends(get_db)) -> RunTask:
    run = db.get(RunTask, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    if not run.images:
        raise HTTPException(status_code=400, detail="请先导入图像")
    if run.status != "draft":
        raise HTTPException(status_code=400, detail="当前任务状态不允许直接执行")
    run.status = "queued"
    run.progress = 0.01
    db.commit()
    task_manager.submit(run_id)
    db.refresh(run)
    return run


@router.get("/runs/{run_id}", response_model=RunRead)
def get_run(run_id: int, db: Session = Depends(get_db)) -> RunTask:
    run = db.get(RunTask, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return run


@router.get("/runs/{run_id}/results", response_model=RunResultResponse)
def get_run_results(run_id: int, db: Session = Depends(get_db)) -> RunResultResponse:
    run = db.get(RunTask, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    image_assets = db.scalars(select(ImageAsset).where(ImageAsset.run_id == run_id).order_by(ImageAsset.sort_index)).all()
    metrics = db.scalars(select(MetricRecord).where(MetricRecord.run_id == run_id)).all()
    exports = db.scalars(select(ExportRecord).where(ExportRecord.run_id == run_id)).all()

    image_map = {
        image.id: {
            "image_id": image.id,
            "image_name": image.original_name,
            "input_url": storage_service.browser_preview_url(image.relative_path),
            "modes": {},
        }
        for image in image_assets
    }
    for metric in metrics:
        if metric.image_id is None or metric.image_id not in image_map:
            continue
        image_map[metric.image_id]["modes"][metric.mode] = {"summary": metric.summary, "artifacts": metric.artifacts}

    return RunResultResponse(
        run=RunRead.model_validate(run),
        images=list(image_map.values()),
        steps=[RunStepRead.model_validate(step) for step in run.steps],
        exports=[
            {
                "id": export.id,
                "kind": export.kind,
                "url": storage_service.static_url(export.relative_path),
                "path": export.relative_path,
            }
            for export in exports
        ],
    )


@router.post("/runs/{run_id}/postprocess/preview", response_model=PostprocessPreviewRead)
def create_postprocess_preview(
    run_id: int,
    payload: PostprocessPreviewCreate,
    db: Session = Depends(get_db),
) -> PostprocessPreviewRead:
    try:
        preview = postprocess_preview_service.create_preview(
            db,
            run_id,
            payload.mode,
            payload.postprocess,
            selected_image_id=payload.selected_image_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PostprocessPreviewRead.model_validate(preview)


@router.post("/runs/{run_id}/postprocess/confirm", response_model=PostprocessConfirmRead)
def confirm_postprocess_preview(
    run_id: int,
    payload: PostprocessConfirmCreate,
    db: Session = Depends(get_db),
) -> PostprocessConfirmRead:
    try:
        result = postprocess_preview_service.confirm_preview(db, run_id, payload.preview_token)
    except PreviewNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PreviewConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return PostprocessConfirmRead(
        run=RunRead.model_validate(result["run"]),
        mode=str(result["mode"]),
        image_count=int(result["image_count"]),
    )


@router.get("/model-runners", response_model=list[ModelRunnerRead])
def list_model_runners(db: Session = Depends(get_db)) -> list[ModelRunner]:
    return db.scalars(select(ModelRunner).order_by(ModelRunner.slot.asc())).all()


@router.post("/model-runners", response_model=ModelRunnerRead)
def create_or_update_model_runner(payload: ModelRunnerCreate, db: Session = Depends(get_db)) -> ModelRunner:
    runner = db.scalar(select(ModelRunner).where(ModelRunner.slot == payload.slot))
    if runner is None:
        runner = ModelRunner(**payload.model_dump())
        db.add(runner)
    else:
        for key, value in payload.model_dump().items():
            setattr(runner, key, value)
    db.commit()
    db.refresh(runner)
    return runner


@router.get("/exports/{export_id}")
def get_export(export_id: int, db: Session = Depends(get_db)) -> dict[str, str]:
    export = db.get(ExportRecord, export_id)
    if export is None:
        raise HTTPException(status_code=404, detail="导出文件不存在")
    return {"kind": export.kind, "path": export.relative_path, "url": storage_service.static_url(export.relative_path) or ""}
