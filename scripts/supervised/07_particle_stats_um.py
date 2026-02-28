import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import math

MANIFEST = Path("C:/Users/pyd111/Desktop/标注/project/data/meta/patch_manifest.csv")
MASK_DIR = Path("C:/Users/pyd111/Desktop/标注/project/data/masks/binary")
OUT_DIR = Path("C:/Users/pyd111/Desktop/标注/project/data/stats")
OUT_PATCH = OUT_DIR / "patch_level_stats.csv"
OUT_PARTICLES = OUT_DIR / "particle_level_stats.csv"

# 过滤策略（建议固定写入论文“标注/统计规范”）
MIN_AREA_PX = 30          # 过小连通域当噪声
REMOVE_BORDER = True      # 触边颗粒是否剔除（推荐 True）
CONNECTIVITY = 8          # 4 或 8

# mask 取值约定：gamma_prime=255，ignore=128，背景=0（与 04 脚本一致）
VAL_FG = 255
VAL_IGNORE = 128

def touches_border(x, y, w, h, W, H):
    return x == 0 or y == 0 or (x + w) >= W or (y + h) >= H

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    man = pd.read_csv(MANIFEST, encoding="utf-8-sig")
    man = man.dropna(subset=["um_per_px"])
    man["um_per_px"] = man["um_per_px"].astype(float)

    particle_rows = []
    patch_rows = []

    for _, r in tqdm(man.iterrows(), total=len(man)):
        stem = Path(r["patch_file"]).stem
        mp = MASK_DIR / f"{stem}.png"
        if not mp.exists():
            continue

        mask = cv2.imdecode(np.fromfile(str(mp), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        valid = (mask != VAL_IGNORE)
        if valid.sum() == 0:
            continue  # 整张都是 ignore
        binm = ((mask == VAL_FG) & valid).astype(np.uint8)

        # 连通域
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binm, connectivity=CONNECTIVITY
        )
        # stats: [label, x, y, w, h, area]
        H, W = binm.shape[:2]
        um_per_px = float(r["um_per_px"])

        areas_um2 = []
        diams_um = []
        kept = 0

        for lab in range(1, num):  # 0 是背景
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
                "parent_file": r["parent_file"],
                "Mag(kx)": r["Mag(kx)"],
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

        # patch 级统计
        if kept > 0:
            fg_ratio = float(binm.sum() / valid.sum())
            patch_rows.append({
                "patch_stem": stem,
                "parent_file": r["parent_file"],
                "Mag(kx)": r["Mag(kx)"],
                "um_per_px": um_per_px,
                "n_particles": kept,
                "mean_equiv_d_um": float(np.mean(diams_um)),
                "median_equiv_d_um": float(np.median(diams_um)),
                "mean_area_um2": float(np.mean(areas_um2)),
                "fg_ratio": fg_ratio
            })
        else:
            fg_ratio = float(binm.sum() / valid.sum())
            patch_rows.append({
                "patch_stem": stem,
                "parent_file": r["parent_file"],
                "Mag(kx)": r["Mag(kx)"],
                "um_per_px": um_per_px,
                "n_particles": 0,
                "mean_equiv_d_um": np.nan,
                "median_equiv_d_um": np.nan,
                "mean_area_um2": np.nan,
                "fg_ratio": fg_ratio
            })

    pd.DataFrame(patch_rows).to_csv(OUT_PATCH, index=False, encoding="utf-8-sig")
    pd.DataFrame(particle_rows).to_csv(OUT_PARTICLES, index=False, encoding="utf-8-sig")
    print(f"[OK] patch 级统计：{OUT_PATCH}")
    print(f"[OK] 粒子级统计：{OUT_PARTICLES}")

if __name__ == "__main__":
    main()
