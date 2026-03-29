from __future__ import annotations

import argparse
import json

from experiments.mbu_netpp.preparation import prepare_nasa_super_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="准备 NASA Super1~4 外部验证数据")
    parser.add_argument("--dataset-root", required=True, help="NASA benchmark_segmentation_data 根目录")
    parser.add_argument("--output-root", required=True, help="准备后输出目录")
    parser.add_argument("--subsets", nargs="+", default=["Super1", "Super2", "Super3", "Super4"], help="需要处理的 Super 子集")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], help="需要处理的 split")
    parser.add_argument("--include-different-test", action="store_true", help="包含 different_test 及其标注")
    parser.add_argument("--auto-crop", action="store_true", help="对 NASA 图像也尝试自动裁掉底栏")
    parser.add_argument("--crop-detection-ratio", type=float, default=0.75, help="底栏检测起始比例")
    parser.add_argument("--edge-kernels", nargs="+", type=int, default=[3, 5], help="边界带核大小")
    parser.add_argument(
        "--foreground-colors-bgr",
        nargs="+",
        default=["255,0,0", "0,0,255"],
        help="前景颜色，BGR 逗号分隔，默认把两类析出相并成前景",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    foreground_colors = []
    for item in args.foreground_colors_bgr:
        values = [int(value.strip()) for value in str(item).split(",") if value.strip()]
        if len(values) >= 3:
            foreground_colors.append(values[:3])

    result = prepare_nasa_super_dataset(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        subsets=list(args.subsets),
        splits=list(args.splits),
        include_different_test=bool(args.include_different_test),
        auto_crop_sem_region=bool(args.auto_crop),
        crop_detection_ratio=float(args.crop_detection_ratio),
        edge_kernels=tuple(args.edge_kernels),
        foreground_colors=foreground_colors,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
