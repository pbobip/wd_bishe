from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ImageCalibrationConfig(BaseModel):
    relative_path: str
    original_name: str | None = None
    um_per_px: float | None = None


class InputConfig(BaseModel):
    result_dir_name: str = "default"
    um_per_px: float | None = None
    auto_crop_sem_region: bool = True
    save_sem_footer: bool = True
    image_calibrations: list[ImageCalibrationConfig] = Field(default_factory=list)


class PreprocessBackgroundConfig(BaseModel):
    method: Literal["none", "tophat", "rolling_ball"] = "none"
    radius: int = 25


class PreprocessDenoiseConfig(BaseModel):
    method: Literal["none", "wavelet", "gaussian", "median", "bilateral", "mean"] = "none"
    wavelet_strength: float = 0.12
    mean_kernel: int = 3
    gaussian_kernel: int = 3
    median_kernel: int = 3
    bilateral_diameter: int = 5
    bilateral_sigma_color: float = 45.0
    bilateral_sigma_space: float = 9.0


class PreprocessEnhanceConfig(BaseModel):
    method: Literal["none", "clahe", "hist_equalization", "gamma"] = "none"
    clahe_clip_limit: float = 2.0
    clahe_tile_size: int = 8
    gamma: float = 1.0


class PreprocessExtrasConfig(BaseModel):
    unsharp: bool = False
    unsharp_radius: int = 3
    unsharp_amount: float = 1.0


class PreprocessConfig(BaseModel):
    enabled: bool = False
    background: PreprocessBackgroundConfig = Field(default_factory=PreprocessBackgroundConfig)
    denoise: PreprocessDenoiseConfig = Field(default_factory=PreprocessDenoiseConfig)
    enhance: PreprocessEnhanceConfig = Field(default_factory=PreprocessEnhanceConfig)
    extras: PreprocessExtrasConfig = Field(default_factory=PreprocessExtrasConfig)

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_operations(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        if any(key in data for key in ("background", "denoise", "enhance", "extras")):
            return data

        operations = list(data.get("operations") or [])
        denoise_method = "none"
        for operation in operations:
            if operation in {"wavelet", "mean", "gaussian", "median"}:
                denoise_method = operation

        enhance_method = "hist_equalization" if any(
            operation in {"hist_equalization", "equalize"} for operation in operations
        ) else "none"

        return {
            "enabled": data.get("enabled", False),
            "background": {
                "method": "none",
                "radius": 25,
            },
            "denoise": {
                "method": denoise_method,
                "wavelet_strength": data.get("wavelet_strength", 0.12),
                "mean_kernel": data.get("mean_kernel", 3),
                "gaussian_kernel": data.get("gaussian_kernel", 3),
                "median_kernel": data.get("median_kernel", 3),
                "bilateral_diameter": data.get("bilateral_diameter", 5),
                "bilateral_sigma_color": data.get("bilateral_sigma_color", 45.0),
                "bilateral_sigma_space": data.get("bilateral_sigma_space", 9.0),
            },
            "enhance": {
                "method": enhance_method,
                "clahe_clip_limit": data.get("clahe_clip_limit", 2.0),
                "clahe_tile_size": data.get("clahe_tile_size", 8),
                "gamma": data.get("gamma", 1.0),
            },
            "extras": {
                "unsharp": data.get("unsharp", False),
                "unsharp_radius": data.get("unsharp_radius", 3),
                "unsharp_amount": data.get("unsharp_amount", 1.0),
            },
        }


class TraditionalSegConfig(BaseModel):
    method: Literal["threshold", "adaptive", "edge", "clustering"] = "threshold"
    foreground_target: Literal["dark", "bright"] = "dark"
    threshold_mode: Literal["otsu", "global", "fixed"] = "otsu"
    global_threshold: int = 120
    fixed_threshold: int = 120
    adaptive_method: Literal["mean", "gaussian"] = "gaussian"
    adaptive_block_size: int = 35
    adaptive_c: int = 5
    edge_operator: Literal["canny", "sobel", "laplacian"] = "canny"
    edge_blur_kernel: int = 3
    edge_threshold1: int = 60
    edge_threshold2: int = 180
    edge_dilate_iterations: int = 1
    kmeans_clusters: int = 2
    kmeans_attempts: int = 5
    cluster_target: Literal["bright", "dark", "largest"] = "dark"
    fill_holes: bool = False
    watershed: bool = False
    watershed_separation: int = 35
    watershed_bg_iterations: int = 1
    watershed_min_marker_area: int = 12
    boundary_smoothing: bool = False
    boundary_smoothing_method: Literal["morphology", "mean", "gaussian", "median"] = "morphology"
    boundary_smoothing_kernel: int = 3
    min_area: int = 30
    max_area: int | None = None
    min_solidity: float = 0.0
    min_circularity: float = 0.0
    min_roundness: float = 0.0
    max_aspect_ratio: float | None = None
    remove_border: bool = False
    open_kernel: int = 3
    close_kernel: int = 3


class DlModelConfig(BaseModel):
    model_slot: Literal["mbu_netpp", "sam_lora", "resnext50", "matsam", "custom"] = "mbu_netpp"
    runner_id: int | None = None
    weight_path: str | None = None
    input_size: int = 256
    device: Literal["auto", "cuda", "cpu"] = "auto"
    extra_params: dict[str, Any] = Field(default_factory=dict)


class PostprocessSmoothingConfig(BaseModel):
    enabled: bool = False
    method: Literal["mean", "gaussian", "median"] = "gaussian"
    kernel: int = 3


class PostprocessShapeFilterConfig(BaseModel):
    enabled: bool = False
    min_area: int = 30
    max_area: int | None = None
    min_solidity: float = 0.0
    min_circularity: float = 0.0
    min_roundness: float = 0.0
    max_aspect_ratio: float | None = None


class PostprocessMorphologyConfig(BaseModel):
    opening_enabled: bool = False
    opening_kernel: int = 3
    closing_enabled: bool = False
    closing_kernel: int = 3


class PostprocessWatershedConfig(BaseModel):
    separation: int = Field(default=35, ge=5, le=85)
    background_iterations: int = Field(default=1, ge=1, le=5)
    min_marker_area: int = Field(default=12, ge=0, le=500)


class PostprocessConfig(BaseModel):
    fill_holes: bool = False
    watershed: bool = False
    watershed_params: PostprocessWatershedConfig = Field(default_factory=PostprocessWatershedConfig)
    smoothing: PostprocessSmoothingConfig = Field(default_factory=PostprocessSmoothingConfig)
    shape_filter: PostprocessShapeFilterConfig = Field(default_factory=PostprocessShapeFilterConfig)
    morphology: PostprocessMorphologyConfig = Field(default_factory=PostprocessMorphologyConfig)
    remove_border: bool = False


class MeasurementConfig(BaseModel):
    vf: bool = True
    particle_count: bool = True
    area: bool = True
    size: bool = True
    channel_width: bool = True
    mean: bool = True
    median: bool = True
    std: bool = True


class StatsConfig(BaseModel):
    enabled: bool = True
    measurements: MeasurementConfig = Field(default_factory=MeasurementConfig)
    export_csv: bool = True
    export_xlsx: bool = True


class ExportConfig(BaseModel):
    include_masks: bool = True
    include_overlays: bool = True
    include_tables: bool = True
    include_charts: bool = True
    include_config_snapshot: bool = True


class RunCreate(BaseModel):
    name: str
    input_mode: Literal["single", "batch"]
    segmentation_mode: Literal["traditional", "dl"]
    input_config: InputConfig = Field(default_factory=InputConfig)
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    traditional_seg: TraditionalSegConfig = Field(default_factory=TraditionalSegConfig)
    dl_model: DlModelConfig = Field(default_factory=DlModelConfig)
    postprocess: PostprocessConfig = Field(default_factory=PostprocessConfig)
    stats: StatsConfig = Field(default_factory=StatsConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)


class RunStepRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    step_key: str
    status: str
    message: str | None
    details: dict[str, Any] | None
    started_at: datetime | None
    finished_at: datetime | None


class ImageAssetRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    original_name: str
    relative_path: str
    sort_index: int


class RunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    input_mode: str
    segmentation_mode: str
    status: str
    progress: float
    error_message: str | None
    config: dict[str, Any]
    summary: dict[str, Any] | None
    chart_data: dict[str, Any] | None
    export_bundle_path: str | None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class RunResultResponse(BaseModel):
    run: RunRead
    images: list[dict[str, Any]]
    steps: list[RunStepRead]
    exports: list[dict[str, Any]]


class PostprocessPreviewCreate(BaseModel):
    mode: Literal["traditional", "dl"]
    postprocess: PostprocessConfig = Field(default_factory=PostprocessConfig)
    selected_image_id: int | None = None


class PostprocessPreviewRead(BaseModel):
    preview_token: str
    mode: str
    image_id: int
    image_name: str
    before_mask_url: str | None = None
    after_mask_url: str | None = None
    before_overlay_url: str | None = None
    after_overlay_url: str | None = None
    before_object_overlay_url: str | None = None
    after_object_overlay_url: str | None = None
    before_summary: dict[str, Any]
    after_summary: dict[str, Any]
    image_count: int


class PostprocessConfirmCreate(BaseModel):
    preview_token: str


class PostprocessConfirmRead(BaseModel):
    run: RunRead
    mode: str
    image_count: int


class ModelRunnerCreate(BaseModel):
    slot: str
    display_name: str
    python_path: str
    env_name: str | None = None
    weight_path: str | None = None
    extra_config: dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True


class ModelRunnerRead(ModelRunnerCreate):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
    updated_at: datetime
