from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from backend.app.utils.image_io import read_gray

from experiments.mbu_netpp.common import image_to_tensor, mask_to_tensor
from experiments.mbu_netpp.preprocess import apply_preprocess
from experiments.mbu_netpp.sampling import compute_record_sampling_weights


def load_prepared_records(
    prepared_root: str | Path,
    fold_manifest_name: str,
    fold_index: int,
    stage: str,
) -> list[dict[str, Any]]:
    prepared_root = Path(prepared_root)
    dataset_manifest = json.loads((prepared_root / "manifests" / "dataset.json").read_text(encoding="utf-8"))
    fold_manifest = json.loads((prepared_root / "manifests" / fold_manifest_name).read_text(encoding="utf-8"))

    items_by_stem = {item["stem"]: item for item in dataset_manifest["items"]}
    target_fold = None
    for fold in fold_manifest["folds"]:
        if int(fold["fold_index"]) == int(fold_index):
            target_fold = fold
            break
    if target_fold is None:
        raise ValueError(f"未找到 fold_index={fold_index} 的划分")

    if stage == "train":
        stems = target_fold["train_stems"]
    elif stage == "val":
        stems = target_fold["val_stems"]
    elif stage == "test":
        stems = target_fold.get("test_stems", [])
    else:
        raise ValueError(f"未知阶段: {stage}")

    return [items_by_stem[stem] for stem in stems]


class SEMSegmentationDataset(Dataset):
    def __init__(
        self,
        prepared_root: str | Path,
        fold_manifest_name: str,
        fold_index: int,
        stage: str,
        patch_size: int,
        edge_kernel: int,
        normalization: str,
        preprocess_config: dict[str, Any] | None,
        augmentation_config: dict[str, Any] | None,
        sampling_config: dict[str, Any] | None = None,
        samples_per_epoch: int | None = None,
    ) -> None:
        self.prepared_root = Path(prepared_root)
        self.records = load_prepared_records(prepared_root, fold_manifest_name, fold_index, stage)
        self.stage = stage
        self.patch_size = int(patch_size)
        self.edge_key = f"k{int(edge_kernel)}"
        self.normalization = normalization
        self.preprocess_config = preprocess_config
        self.augmentation_config = augmentation_config or {}
        self.sampling_config = sampling_config or {}
        self.samples_per_epoch = int(samples_per_epoch or len(self.records))
        self.transforms = self._build_transforms() if stage == "train" else None
        self.record_sampling_weights = (
            compute_record_sampling_weights(self.records, self.sampling_config) if stage == "train" else None
        )

    def _build_transforms(self) -> A.Compose:
        cfg = self.augmentation_config
        scale_limit = float(cfg.get("scale_limit", 0.1))
        noise_limit = float(cfg.get("noise_std_limit", 10.0)) / 255.0
        transforms: list[Any] = [
            A.HorizontalFlip(p=float(cfg.get("horizontal_flip_p", 0.5))),
            A.VerticalFlip(p=float(cfg.get("vertical_flip_p", 0.5))),
            A.RandomRotate90(p=float(cfg.get("rotate90_p", 0.5))),
            A.Affine(
                scale=(1.0 - scale_limit, 1.0 + scale_limit),
                translate_percent=None,
                rotate=0.0,
                shear=0.0,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                fit_output=False,
                keep_ratio=False,
                border_mode=cv2.BORDER_REFLECT_101,
                fill=0,
                fill_mask=0,
                p=float(cfg.get("scale_p", 0.3)),
            ),
            A.RandomBrightnessContrast(
                brightness_limit=float(cfg.get("brightness_limit", 0.08)),
                contrast_limit=float(cfg.get("contrast_limit", 0.08)),
                p=float(cfg.get("brightness_contrast_p", 0.25)),
            ),
            A.GaussNoise(
                std_range=(0.0, max(0.01, noise_limit)),
                mean_range=(0.0, 0.0),
                p=float(cfg.get("gaussian_noise_p", 0.2)),
            ),
        ]
        return A.Compose(transforms, additional_targets={"edge": "mask"})

    def __len__(self) -> int:
        if self.stage == "train":
            return self.samples_per_epoch
        return len(self.records)

    def _pad_to_min_size(self, image: np.ndarray, mask: np.ndarray, edge: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        height, width = image.shape[:2]
        pad_bottom = max(0, self.patch_size - height)
        pad_right = max(0, self.patch_size - width)
        if pad_bottom == 0 and pad_right == 0:
            return image, mask, edge
        image = cv2.copyMakeBorder(image, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT_101)
        mask = cv2.copyMakeBorder(mask, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=0)
        edge = cv2.copyMakeBorder(edge, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=0)
        return image, mask, edge

    def _random_crop(self, image: np.ndarray, mask: np.ndarray, edge: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        image, mask, edge = self._pad_to_min_size(image, mask, edge)
        height, width = image.shape[:2]
        max_y = height - self.patch_size
        max_x = width - self.patch_size

        attempts = int(self.augmentation_config.get("crop_attempts", 12))
        min_foreground_ratio = float(self.augmentation_config.get("min_foreground_ratio", 0.01))
        min_edge_ratio = float(self.augmentation_config.get("min_edge_ratio", 0.002))

        best_candidate: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        best_score = -1.0
        for _ in range(attempts):
            top = random.randint(0, max(0, max_y))
            left = random.randint(0, max(0, max_x))
            image_crop = image[top : top + self.patch_size, left : left + self.patch_size]
            mask_crop = mask[top : top + self.patch_size, left : left + self.patch_size]
            edge_crop = edge[top : top + self.patch_size, left : left + self.patch_size]

            fg_ratio = float(np.count_nonzero(mask_crop)) / float(mask_crop.size)
            edge_ratio = float(np.count_nonzero(edge_crop)) / float(edge_crop.size)
            score = fg_ratio + edge_ratio
            if score > best_score:
                best_score = score
                best_candidate = (image_crop, mask_crop, edge_crop)
            if fg_ratio >= min_foreground_ratio or edge_ratio >= min_edge_ratio:
                return image_crop, mask_crop, edge_crop

        assert best_candidate is not None
        return best_candidate

    def _load_record(self, record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        image = read_gray(self.prepared_root / record["image_path"])
        mask = read_gray(self.prepared_root / record["mask_path"])
        edge_path = record["edge_paths"].get(self.edge_key)
        if edge_path is None:
            edge = np.zeros_like(mask, dtype=np.uint8)
        else:
            edge = read_gray(self.prepared_root / edge_path)
        image = apply_preprocess(image, self.preprocess_config)
        return image, mask, edge

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.stage == "train":
            assert self.record_sampling_weights is not None
            record = random.choices(self.records, weights=self.record_sampling_weights, k=1)[0]
        else:
            record = self.records[index % len(self.records)]
        image, mask, edge = self._load_record(record)

        if self.stage == "train":
            image, mask, edge = self._random_crop(image, mask, edge)
            if self.transforms is not None:
                augmented = self.transforms(image=image, mask=mask, edge=edge)
                image = augmented["image"]
                mask = augmented["mask"]
                edge = augmented["edge"]

        mask = (mask > 127).astype(np.uint8) * 255
        edge = (edge > 127).astype(np.uint8) * 255
        return {
            "image": image_to_tensor(image, normalization=self.normalization),
            "mask": mask_to_tensor(mask),
            "edge": mask_to_tensor(edge),
            "sample_weight": torch.tensor(float(record.get("sample_weight", 1.0)), dtype=torch.float32),
            "source_type": str(record.get("source_type", "supervised")),
            "source_stem": str(record.get("source_stem", record["stem"])),
            "stem": record["stem"],
            "height": int(image.shape[0]),
            "width": int(image.shape[1]),
        }
