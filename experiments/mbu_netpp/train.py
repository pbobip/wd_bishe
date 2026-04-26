from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from experiments.mbu_netpp.common import ensure_dir, load_yaml, save_json, seed_everything, resolve_device
from experiments.mbu_netpp.dataset import SEMSegmentationDataset
from experiments.mbu_netpp.losses import MBULoss
from experiments.mbu_netpp.metrics import average_metric_dicts, compute_segmentation_metrics
from experiments.mbu_netpp.models import build_model, sliding_window_inference
from experiments.mbu_netpp.preparation import prepare_supervised_dataset
from experiments.mbu_netpp.training_state import collect_crossval_summary, load_resume_state


def build_grad_scaler(use_amp: bool) -> Any:
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    return torch.cuda.amp.GradScaler(enabled=use_amp)


def build_autocast_context(use_amp: bool) -> Any:
    if not use_amp:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda", enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


def maybe_prepare_data(config: dict[str, Any]) -> None:
    data_cfg = config["data"]
    prepared_root = Path(data_cfg["prepared_root"])
    dataset_manifest_path = prepared_root / "manifests" / "dataset.json"
    should_prepare = bool(data_cfg.get("force_prepare", False)) or not dataset_manifest_path.exists()
    if not should_prepare:
        return
    if not bool(data_cfg.get("auto_prepare", True)):
        raise FileNotFoundError(f"未找到 prepared 数据: {dataset_manifest_path}")
    result = prepare_supervised_dataset(
        images_dir=data_cfg["images_dir"],
        annotations_dir=data_cfg["annotations_dir"],
        output_root=prepared_root,
        positive_labels=list(data_cfg.get("positive_labels", ["gamma_prime"])),
        auto_crop_sem_region=bool(data_cfg.get("auto_crop_sem_region", True)),
        crop_detection_ratio=float(data_cfg.get("crop_detection_ratio", 0.75)),
        edge_kernels=(3, 5),
        num_folds=int(data_cfg.get("num_folds", 3)),
        seed=int(data_cfg.get("fold_seed", 42)),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def build_dataloaders(config: dict[str, Any], fold_index: int) -> tuple[DataLoader, DataLoader]:
    data_cfg = config["data"]
    training_cfg = config["training"]
    train_dataset = SEMSegmentationDataset(
        prepared_root=data_cfg["prepared_root"],
        fold_manifest_name=data_cfg["fold_manifest_name"],
        fold_index=fold_index,
        stage="train",
        patch_size=int(data_cfg["patch_size"]),
        edge_kernel=int(data_cfg["edge_kernel"]),
        normalization=str(data_cfg.get("normalization", "minmax")),
        preprocess_config=data_cfg.get("preprocess"),
        augmentation_config=data_cfg.get("augmentation"),
        sampling_config=data_cfg.get("sampling"),
        samples_per_epoch=int(data_cfg.get("samples_per_epoch", 96)),
    )
    val_dataset = SEMSegmentationDataset(
        prepared_root=data_cfg["prepared_root"],
        fold_manifest_name=data_cfg["fold_manifest_name"],
        fold_index=fold_index,
        stage="val",
        patch_size=int(data_cfg["patch_size"]),
        edge_kernel=int(data_cfg["edge_kernel"]),
        normalization=str(data_cfg.get("normalization", "minmax")),
        preprocess_config=data_cfg.get("preprocess"),
        augmentation_config=None,
        samples_per_epoch=None,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training_cfg.get("batch_size", 4)),
        shuffle=True,
        num_workers=int(training_cfg.get("num_workers", 0)),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(training_cfg.get("num_workers", 0)),
    )
    return train_loader, val_loader


def should_improve(metric_name: str, new_value: float, best_value: float | None) -> bool:
    if best_value is None:
        return True
    if metric_name in {"vf", "loss"}:
        return new_value < best_value
    return new_value > best_value


def maybe_load_init_checkpoint(model: torch.nn.Module, training_cfg: dict[str, Any], fold_index: int) -> str | None:
    init_checkpoint_template = str(training_cfg.get("init_checkpoint_template", "")).strip()
    if init_checkpoint_template:
        init_checkpoint = init_checkpoint_template.format(fold_index=fold_index)
    else:
        init_checkpoint = str(training_cfg.get("init_checkpoint", "")).strip()
    if not init_checkpoint:
        return None
    checkpoint = torch.load(init_checkpoint, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError(f"初始化 checkpoint 缺少 model_state_dict: {init_checkpoint}")
    model.load_state_dict(state_dict, strict=True)
    print(f"已加载初始化 checkpoint: {init_checkpoint}")
    return init_checkpoint


def train_one_fold(config: dict[str, Any], fold_index: int) -> dict[str, Any]:
    experiment_cfg = config["experiment"]
    training_cfg = config["training"]
    data_cfg = config["data"]
    device = resolve_device(str(training_cfg.get("device", "auto")))
    output_root = ensure_dir(Path(experiment_cfg["output_root"]) / f"fold_{fold_index}")
    summary_path = output_root / "summary.json"
    history_path = output_root / "history.json"
    last_checkpoint_path = output_root / "last.pt"

    if summary_path.exists():
        print(f"[fold={fold_index}] 已存在 summary.json，跳过训练")
        return json.loads(summary_path.read_text(encoding="utf-8"))

    train_loader, val_loader = build_dataloaders(config, fold_index)
    model = build_model(config["model"]).to(device)
    maybe_load_init_checkpoint(model, training_cfg, fold_index=fold_index)
    criterion = MBULoss(config["loss"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get("learning_rate", 1e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
    )
    scheduler_name = str(training_cfg.get("scheduler", "cosine")).strip().lower()
    if scheduler_name == "reduce":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(training_cfg.get("epochs", 80))))

    use_amp = bool(training_cfg.get("amp", True)) and device.type == "cuda"
    scaler = build_grad_scaler(use_amp)

    epochs = int(training_cfg.get("epochs", 80))
    threshold = float(training_cfg.get("threshold", 0.5))
    boundary_tolerance = int(training_cfg.get("boundary_tolerance", 2))
    patch_size = int(data_cfg.get("patch_size", 256))
    overlap = float(data_cfg.get("overlap", 0.25))
    primary_metric = str(experiment_cfg.get("save_best_by", "dice"))

    history: list[dict[str, Any]] = []
    best_metric: float | None = None
    best_summary: dict[str, Any] | None = None
    start_epoch = 1

    if bool(training_cfg.get("resume", False)) and last_checkpoint_path.exists():
        try:
            resume_state = load_resume_state(last_checkpoint_path)
        except Exception as exc:
            print(f"[fold={fold_index}] last.pt 损坏或不可读，删除后从头开始: {exc!r}")
            last_checkpoint_path.unlink(missing_ok=True)
        else:
            if resume_state.get("model_state_dict"):
                model.load_state_dict(resume_state["model_state_dict"], strict=True)
            if resume_state.get("optimizer_state_dict"):
                optimizer.load_state_dict(resume_state["optimizer_state_dict"])
            if resume_state.get("scheduler_state_dict"):
                scheduler.load_state_dict(resume_state["scheduler_state_dict"])
            if use_amp and resume_state.get("scaler_state_dict"):
                scaler.load_state_dict(resume_state["scaler_state_dict"])
            history = list(resume_state.get("history") or [])
            best_metric = resume_state.get("best_metric")
            best_summary = dict(resume_state.get("best_summary") or {})
            start_epoch = int(resume_state.get("start_epoch", 1))
            print(f"[fold={fold_index}] 从 last.pt 恢复，继续 epoch={start_epoch}")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_components: list[dict[str, float]] = []
        for batch in train_loader:
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)
            edge = batch["edge"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with build_autocast_context(use_amp):
                outputs = model(image)
                loss, components = criterion(outputs, {"mask": mask, "edge": edge})
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_components.append(components)

        model.eval()
        val_metrics: list[dict[str, float]] = []
        with torch.no_grad():
            for batch in val_loader:
                image = batch["image"].to(device)
                mask = batch["mask"].to(device)
                logits = sliding_window_inference(model, image, patch_size=patch_size, overlap=overlap)
                metrics = compute_segmentation_metrics(
                    logits=logits,
                    target_mask=mask,
                    threshold=threshold,
                    boundary_tolerance=boundary_tolerance,
                )
                val_metrics.append(metrics)

        train_summary = average_metric_dicts(train_components)
        val_summary = average_metric_dicts(val_metrics)
        epoch_summary = {
            "epoch": epoch,
            "train": train_summary,
            "val": val_summary,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(epoch_summary)
        print(
            f"[fold={fold_index}] epoch={epoch} "
            f"train_total={train_summary.get('total', 0.0):.4f} "
            f"val_dice={val_summary.get('dice', 0.0):.4f} "
            f"val_vf={val_summary.get('vf', 0.0):.4f} "
            f"val_boundary_f1={val_summary.get('boundary_f1', 0.0):.4f}"
        )

        if scheduler_name == "reduce":
            scheduler.step(float(val_summary.get("dice", 0.0)))
        else:
            scheduler.step()

        current_value = float(val_summary.get(primary_metric, 0.0))
        if should_improve(primary_metric, current_value, best_metric):
            best_metric = current_value
            best_summary = val_summary
            checkpoint = {
                "fold_index": fold_index,
                "epoch": epoch,
                "config": config,
                "model_state_dict": model.state_dict(),
                "best_summary": best_summary,
            }
            torch.save(checkpoint, output_root / "best.pt")

        save_json(history_path, history)
        torch.save(
            {
                "fold_index": fold_index,
                "epoch": epoch,
                "config": config,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict() if use_amp else None,
                "history": history,
                "best_metric": best_metric,
                "best_summary": best_summary or {},
            },
            last_checkpoint_path,
        )

    fold_result = {
        "fold_index": fold_index,
        "best_metric_name": primary_metric,
        "best_metric": best_metric,
        "best_summary": best_summary or {},
        "history_path": str(history_path),
        "checkpoint_path": str(output_root / "best.pt"),
    }
    save_json(summary_path, fold_result)
    last_checkpoint_path.unlink(missing_ok=True)
    return fold_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 MBU-Net++ 监督学习模型")
    parser.add_argument("--config", required=True, help="YAML 配置路径")
    parser.add_argument("--fold", type=int, default=None, help="只运行指定 fold")
    parser.add_argument("--run-all-folds", action="store_true", help="运行全部 fold")
    parser.add_argument("--resume", action="store_true", help="若存在 last.pt，则从断点继续训练")
    parser.add_argument("--summarize-only", action="store_true", help="只汇总已完成 fold 结果")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    seed_everything(int(config["experiment"].get("seed", 42)))
    maybe_prepare_data(config)

    num_folds = int(config["data"].get("num_folds", 3))
    output_root = Path(config["experiment"]["output_root"])
    experiment_name = str(config["experiment"]["name"])
    if args.run_all_folds:
        fold_indices = list(range(num_folds))
    elif args.fold is not None:
        fold_indices = [int(args.fold)]
    else:
        fold_indices = [0]

    if args.resume:
        config.setdefault("training", {})["resume"] = True

    if args.summarize_only:
        summary = collect_crossval_summary(
            experiment_name=experiment_name,
            experiment_root=output_root,
            fold_indices=fold_indices,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    fold_results = [train_one_fold(config, fold_index=fold_index) for fold_index in fold_indices]
    if len(fold_results) == num_folds and set(fold_indices) == set(range(num_folds)):
        summary = collect_crossval_summary(
            experiment_name=experiment_name,
            experiment_root=output_root,
            fold_indices=fold_indices,
        )
    else:
        summary = {
            "experiment": experiment_name,
            "fold_results": fold_results,
            "mean_best_summary": average_metric_dicts(
                [item["best_summary"] for item in fold_results if item.get("best_summary")]
            ),
        }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
