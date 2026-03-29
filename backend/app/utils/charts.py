from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def histogram_png(values: list[float], title: str, xlabel: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    if values:
        plt.hist(values, bins=min(20, max(5, len(values) // 2)), color="#d1672d", edgecolor="#23303b")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def bar_png(labels: list[str], values: list[float], title: str, ylabel: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.bar(labels, values, color="#176087")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
