# 数据目录说明

本目录保存的是当前仓库附带的数据内容，分为两类：

## 1. `full_png/`

- 包含完整 `100` 张实验输入 `PNG` 原图。
- 可作为统一输入数据集查看、复现推理流程或补充展示。
- 该数据对应你当前实验链路中反复使用的单晶图像 `PNG` 集。

## 2. 与 `samples/` 的区别

- `dataset/full_png/` 是完整输入数据集。
- `samples/annotated_eval_set/` 是带 `JSON` 标注的精标样本。
- `samples/showcase_set/` 是从完整数据中抽出的 10 张展示样例。

如果后续需要进一步公开原始 `TIF`、压缩包或更大规模数据，建议单独使用网盘、Releases 或 Git LFS。
