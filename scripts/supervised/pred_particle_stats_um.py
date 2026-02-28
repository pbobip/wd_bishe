import argparse
import sys
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


VAL_FG = 255
VAL_IGNORE = 128

# 过滤策略（与 07 一致）
MIN_AREA_PX = 30
REMOVE_BORDER = True
CONNECTIVITY = 8


def touches_border(x, y, w, h, W, H):
    return x == 0 or y == 0 or (x + w) >= W or (y + h) >= H


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path, default=Path("data/meta/patch_manifest.csv"))
    parser.add_argument("--pred_dir", type=Path, default=Path("data/preds/binary"))
    parser.add_argument("--out_dir", type=Path, default=Path("data/pred_stats"))
    args = parser.parse_args()

    root = args.root
    pred_dir = root / args.pred_dir
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_patch = out_dir / "patch_level_pred_stats.csv"
    out_particles = out_dir / "particle_level_pred_stats.csv"

    man = pd.read_csv(root / args.manifest, encoding="utf-8-sig")
    man = man.dropna(subset=["um_per_px"])
    man["um_per_px"] = man["um_per_px"].astype(float)

    particle_rows = []
    patch_rows = []

    for _, r in tqdm(man.iterrows(), total=len(man)):
        stem = Path(r["patch_file"]).stem
        mp = pred_dir / f"{stem}.png"
        if not mp.exists():
            continue

        mask = cv2.imdecode(np.fromfile(str(mp), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        valid = mask != VAL_IGNORE
        if valid.sum() == 0:
            continue
        binm = ((mask == VAL_FG) & valid).astype(np.uint8)

        num, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binm, connectivity=CONNECTIVITY
        )
        H, W = binm.shape[:2]
        um_per_px = float(r["um_per_px"])

        areas_um2 = []
        diams_um = []
        kept = 0

        for lab in range(1, num):
            x, y, w, h, area_px = stats[lab]
            if area_px < MIN_AREA_PX:
                continue
            if REMOVE_BORDER and touches_border(x, y, w, h, W, H):
                continue

            area_um2 = area_px * (um_per_px ** 2)
            equiv_d_um = 2.0 * math.sqrt(area_um2 / math.pi)

            cx, cy = centroids[lab]
            particle_rows.append({
                "patch_stem": stem,
                "parent_file": r.get("parent_file", ""),
                "Mag(kx)": r.get("Mag(kx)", np.nan),
                "um_per_px": um_per_px,
                "label": lab,
                "area_px": int(area_px),
                "area_um2": area_um2,
                "equiv_d_um": equiv_d_um,
                "centroid_x_px": float(cx),
                "centroid_y_px": float(cy),
            })

            areas_um2.append(area_um2)
            diams_um.append(equiv_d_um)
            kept += 1

        fg_ratio = float(binm.sum() / valid.sum())
        patch_rows.append({
            "patch_stem": stem,
            "parent_file": r.get("parent_file", ""),
            "Mag(kx)": r.get("Mag(kx)", np.nan),
            "um_per_px": um_per_px,
            "n_particles": kept,
            "mean_equiv_d_um": float(np.mean(diams_um)) if kept > 0 else np.nan,
            "median_equiv_d_um": float(np.median(diams_um)) if kept > 0 else np.nan,
            "mean_area_um2": float(np.mean(areas_um2)) if kept > 0 else np.nan,
            "fg_ratio": fg_ratio,
        })

    pd.DataFrame(patch_rows).to_csv(out_patch, index=False, encoding="utf-8-sig")
    pd.DataFrame(particle_rows).to_csv(out_particles, index=False, encoding="utf-8-sig")
    print(f"[OK] 预测 patch 级统计：{out_patch}")
    print(f"[OK] 预测 粒子级统计：{out_particles}")


if __name__ == "__main__":
    main()
