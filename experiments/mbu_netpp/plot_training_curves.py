from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from experiments.mbu_netpp.common import ensure_dir, save_json


def load_history(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="为实验输出目录绘制训练/验证曲线")
    parser.add_argument("--experiment-root", required=True, help="训练输出根目录，如 outputs/real53_xxx")
    return parser.parse_args()


def plot_single_fold(history: list[dict[str, Any]], output_path: Path, title: str) -> None:
    epochs = [int(item["epoch"]) for item in history]
    train_total = [float(item.get("train", {}).get("total", 0.0)) for item in history]
    val_dice = [float(item.get("val", {}).get("dice", 0.0)) for item in history]
    val_vf = [float(item.get("val", {}).get("vf", 0.0)) for item in history]
    val_boundary = [float(item.get("val", {}).get("boundary_f1", 0.0)) for item in history]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=140)
    fig.suptitle(title)

    axes[0, 0].plot(epochs, train_total, color="#d1495b", linewidth=1.8)
    axes[0, 0].set_title("Train Total Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].plot(epochs, val_dice, color="#00798c", linewidth=1.8)
    axes[0, 1].set_title("Val Dice")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].grid(alpha=0.25)

    axes[1, 0].plot(epochs, val_vf, color="#edae49", linewidth=1.8)
    axes[1, 0].set_title("Val VF Error")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].grid(alpha=0.25)

    axes[1, 1].plot(epochs, val_boundary, color="#30638e", linewidth=1.8)
    axes[1, 1].set_title("Val Boundary F1")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_mean_curves(histories: list[list[dict[str, Any]]], output_path: Path, csv_path: Path) -> None:
    min_len = min(len(history) for history in histories)
    epochs = [int(histories[0][idx]["epoch"]) for idx in range(min_len)]

    rows: list[dict[str, float]] = []
    for idx in range(min_len):
        row = {
            "epoch": float(epochs[idx]),
            "train_total_mean": float(np.mean([history[idx].get("train", {}).get("total", 0.0) for history in histories])),
            "val_dice_mean": float(np.mean([history[idx].get("val", {}).get("dice", 0.0) for history in histories])),
            "val_vf_mean": float(np.mean([history[idx].get("val", {}).get("vf", 0.0) for history in histories])),
            "val_boundary_f1_mean": float(np.mean([history[idx].get("val", {}).get("boundary_f1", 0.0) for history in histories])),
        }
        rows.append(row)

    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=140)
    fig.suptitle("Crossval Mean Curves")
    axes[0, 0].plot(epochs, [row["train_total_mean"] for row in rows], color="#d1495b", linewidth=1.8)
    axes[0, 0].set_title("Mean Train Total Loss")
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].plot(epochs, [row["val_dice_mean"] for row in rows], color="#00798c", linewidth=1.8)
    axes[0, 1].set_title("Mean Val Dice")
    axes[0, 1].grid(alpha=0.25)

    axes[1, 0].plot(epochs, [row["val_vf_mean"] for row in rows], color="#edae49", linewidth=1.8)
    axes[1, 0].set_title("Mean Val VF Error")
    axes[1, 0].grid(alpha=0.25)

    axes[1, 1].plot(epochs, [row["val_boundary_f1_mean"] for row in rows], color="#30638e", linewidth=1.8)
    axes[1, 1].set_title("Mean Val Boundary F1")
    axes[1, 1].grid(alpha=0.25)

    for ax in axes.flat:
        ax.set_xlabel("Epoch")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    experiment_root = Path(args.experiment_root)
    plots_dir = ensure_dir(experiment_root / "plots")

    histories: list[list[dict[str, Any]]] = []
    fold_summaries: list[dict[str, Any]] = []
    for fold_dir in sorted(experiment_root.glob("fold_*")):
        history_path = fold_dir / "history.json"
        if not history_path.exists():
            continue
        history = load_history(history_path)
        histories.append(history)
        fold_index = int(fold_dir.name.split("_")[-1])
        plot_single_fold(history, plots_dir / f"{fold_dir.name}_training_curves.png", title=f"Fold {fold_index} Curves")
        fold_summaries.append(
            {
                "fold_index": fold_index,
                "history_path": str(history_path),
                "plot_path": str(plots_dir / f"{fold_dir.name}_training_curves.png"),
                "epochs": len(history),
            }
        )

    if not histories:
        raise FileNotFoundError(f"未在 {experiment_root} 找到任何 fold history.json")

    mean_csv_path = plots_dir / "crossval_mean_curves.csv"
    mean_plot_path = plots_dir / "crossval_mean_curves.png"
    plot_mean_curves(histories, mean_plot_path, mean_csv_path)

    payload = {
        "experiment_root": str(experiment_root),
        "num_folds": len(histories),
        "folds": fold_summaries,
        "mean_plot_path": str(mean_plot_path),
        "mean_csv_path": str(mean_csv_path),
    }
    save_json(plots_dir / "plots_manifest.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
