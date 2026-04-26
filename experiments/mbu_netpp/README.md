# MBU-Net++ 实验区

本目录承载 `MBU-Net++` 的独立实验代码，和现有展示型后端解耦。

如果你是先想快速定位训练、推理、数据准备、配置和服务器脚本分别在哪里，先看：

- [`CODEMAP.md`](CODEMAP.md)

当前实现覆盖：

- `LabelMe JSON -> mask` 数据准备
- 固定 `3-fold` 清单生成
- `3x3 / 5x5` 边界标签生成
- 小样本 patch 训练数据集
- `U-Net / U-Net++ / MicroNet-U-Net++ / MBU-Net++` 统一模型接口
- `Dice + BCE + Edge + Deep Supervision + VF Loss` 训练损失
- `Dice / IoU / Precision / Recall / VF / Boundary F1` 验证指标
- 监督训练入口
- 推理与统计导出入口
- NASA `Super1~4` 外部验证数据准备
- 外部标注集评估入口

## 1. 依赖

推荐在仓库根目录执行：

```bash
python -m pip install -r experiments/mbu_netpp/requirements.txt
```

## 2. 先准备数据

### 使用桌面精修数据

```bash
python -m experiments.mbu_netpp.preparation ^
  --images-dir "C:\Users\pyd111\Desktop\已完成标注图像\全部已完成标注图像_精修底图" ^
  --annotations-dir "C:\Users\pyd111\Desktop\已完成标注图像\全部已完成标注图像_精修底图" ^
  --output-root "C:\Users\pyd111\Desktop\中期\wd_bishe\experiments\mbu_netpp\workdir\prepared_main"
```

### 使用成对图像目录 + 掩膜目录

```bash
python -m experiments.mbu_netpp.preparation ^
  --images-dir "C:\path\to\images" ^
  --masks-dir "C:\path\to\masks" ^
  --mask-mode colored ^
  --output-root "C:\Users\pyd111\Desktop\中期\wd_bishe\experiments\mbu_netpp\workdir\prepared_external"
```

`--mask-mode` 可选值包括：

- `colored`：默认，把所有有色标注区域并为前景
- `nonzero`：把所有非零像素并为前景
- `alpha_nonzero`：把所有不透明区域并为前景
- `exact_color`：仅保留指定颜色，需配合 `--foreground-colors`

当 `mask-mode=exact_color` 时，颜色可写成 `255,0,0` 或 `#ff0000`，例如：

```bash
python -m experiments.mbu_netpp.preparation ^
  --images-dir "C:\path\to\images" ^
  --masks-dir "C:\path\to\masks" ^
  --mask-mode exact_color ^
  --foreground-colors 255,0,0 0,0,255 ^
  --output-root "C:\Users\pyd111\Desktop\中期\wd_bishe\experiments\mbu_netpp\workdir\prepared_external"
```

### 使用仓库内 9 张样本做烟雾验证

```bash
python -m experiments.mbu_netpp.preparation ^
  --images-dir "C:\Users\pyd111\Desktop\中期\wd_bishe\samples\annotated_eval_set\raw_images" ^
  --annotations-dir "C:\Users\pyd111\Desktop\中期\wd_bishe\samples\annotated_eval_set\annotations" ^
  --output-root "C:\Users\pyd111\Desktop\中期\wd_bishe\experiments\mbu_netpp\workdir\prepared_smoke"
```

准备完成后，输出目录会生成：

- `images/`
- `masks/`
- `edges/k3/`
- `edges/k5/`
- `manifests/dataset.json`
- `manifests/folds_3_seed42.json`
- `previews/`

## 3. 训练

### 轻量烟雾训练

```bash
python -m experiments.mbu_netpp.train --config experiments/mbu_netpp/configs/repo_sample_smoke.yaml --run-all-folds
```

### 正式监督训练

先在配置文件里填好 `prepared_root` 和 `micronet_checkpoint`，再运行：

```bash
python -m experiments.mbu_netpp.train --config experiments/mbu_netpp/configs/default_supervised.yaml --run-all-folds
```

## 4. 外部评估

对 NASA 这类“原图目录 + 掩膜目录”的外部数据集做逐图评估：

```bash
python -m experiments.mbu_netpp.external_eval ^
  --checkpoint "C:\path\to\best.pt" ^
  --images-dir "C:\path\to\images" ^
  --masks-dir "C:\path\to\masks" ^
  --mask-mode colored ^
  --output-dir "C:\Users\pyd111\Desktop\中期\wd_bishe\experiments\mbu_netpp\outputs\nasa_eval"
```

如果已经先执行了 `preparation.py`，也可以直接对准备后的目录评估：

```bash
python -m experiments.mbu_netpp.external_eval ^
  --checkpoint "C:\path\to\best.pt" ^
  --prepared-root "C:\Users\pyd111\Desktop\中期\wd_bishe\experiments\mbu_netpp\workdir\prepared_external" ^
  --output-dir "C:\Users\pyd111\Desktop\中期\wd_bishe\experiments\mbu_netpp\outputs\nasa_eval"
```

输出会包含：

- `predictions/`
- `overlays/`
- `per_image_metrics.csv`
- `per_image_metrics.xlsx`
- `summary.json`

## 5. 推理

```bash
python -m experiments.mbu_netpp.infer ^
  --checkpoint "C:\path\to\best.pt" ^
  --input "C:\Users\pyd111\Desktop\中期\wd_bishe\dataset\full_png" ^
  --output-dir "C:\Users\pyd111\Desktop\中期\wd_bishe\experiments\mbu_netpp\outputs\inference"
```

推理输出默认包括：

- 二值 mask
- overlay
- 每图统计 `json`
- 汇总 `csv`
- 若环境支持 `openpyxl`，则额外导出 `xlsx`

## 6. 设计边界

- 训练代码独立于 `backend/app`
- 统计导出复用现有 `StatisticsService`
- `MicroNet` 权重通过 `se_resnext50_32x4d` encoder 的兼容加载器接入
- 当前优先完成监督学习主线；半监督伪标签后续再扩

## 6.1 迭代式半监督训练

当前仓库已新增一个“小样本监督训练 + 伪标签 + 主动学习”的迭代入口：

```bash
python -m experiments.mbu_netpp.iterative_semi_supervised ^
  --config experiments/mbu_netpp/configs/default_semi_supervised_iterative.yaml ^
  --fold 0
```

如果要按当前 `3-fold` 逻辑全部执行：

```bash
python -m experiments.mbu_netpp.iterative_semi_supervised ^
  --config experiments/mbu_netpp/configs/default_semi_supervised_iterative.yaml ^
  --run-all-folds
```

### 半监督流程说明

每一轮会自动执行：

1. 用 `labeled` 数据训练或加载初始 `teacher`
2. 对 `unlabeled` 图像推理并计算每图分数
3. 按 `confidence_threshold` 挑选高置信样本，写入 `pseudo_pool`
4. 按 `uncertainty_score` 导出 `top-k` 待人工补标清单
5. 用 `labeled + pseudo_labeled` 重建训练清单并重训下一轮模型

### 输出目录

在 `semi_supervised.workspace_root` 下，每个 `fold` 会生成：

- `iterations/iter_00_supervised/`：初始监督训练结果
- `iterations/iter_xx_predict/`：未标注池预测结果、分数表、伪标签候选
- `iterations/iter_xx_prepared/`：本轮混合训练 manifest
- `iterations/iter_xx_train/`：本轮重训练模型与日志
- `pseudo_pool/`：累计伪标签池
- `query_pool/`：主动学习待补标样本列表
- `state.json`：当前迭代状态，便于继续运行

### 人工补标后继续下一轮

如果你已经把主动学习导出的样本补标完成，建议先用现有 `preparation.py`
把这些新标注整理成一个新的 `prepared_root`，然后把该目录加入：

```yaml
semi_supervised:
  reviewed_labeled_roots:
    - experiments/mbu_netpp/workdir/reviewed_iter1
```

再次运行 `iterative_semi_supervised.py` 时，脚本会自动：

- 将这些新样本视为真实标签训练样本
- 从 `query_pending` / `pseudo_pool` 中剔除同 stem 项
- 继续后续迭代

## 7. NASA Super1~4 外部验证

### 准备 NASA 数据

当前 NASA 的 `Super1~4` 标注在官方 notebook 中定义为：

- `secondary = [255, 0, 0]`
- `tertiary = [0, 0, 255]`
- `matrix = 其他区域`

针对当前二分类 `γ′ vs matrix` 任务，外部验证默认把 `secondary + tertiary` 合并为前景。

```bash
python -m experiments.mbu_netpp.prepare_nasa_super ^
  --dataset-root "C:\Users\pyd111\Desktop\中期\wd_bishe\external_data\nasa_benchmark_segmentation_data" ^
  --output-root "C:\Users\pyd111\Desktop\中期\wd_bishe\experiments\mbu_netpp\workdir\prepared_nasa_super_standard"
```

如需把 `different_test` 也纳入：

```bash
python -m experiments.mbu_netpp.prepare_nasa_super ^
  --dataset-root "C:\Users\pyd111\Desktop\中期\wd_bishe\external_data\nasa_benchmark_segmentation_data" ^
  --output-root "C:\Users\pyd111\Desktop\中期\wd_bishe\experiments\mbu_netpp\workdir\prepared_nasa_super_all" ^
  --include-different-test
```

### 评估已有 checkpoint 的外部泛化

```bash
python -m experiments.mbu_netpp.external_eval ^
  --checkpoint "C:\Users\pyd111\Desktop\中期\wd_bishe\experiments\mbu_netpp\outputs\e2_micronet_edge_deep_gpu\fold_0\best.pt" ^
  --prepared-root "C:\Users\pyd111\Desktop\中期\wd_bishe\experiments\mbu_netpp\workdir\prepared_nasa_super_all" ^
  --output-dir "C:\Users\pyd111\Desktop\中期\wd_bishe\experiments\mbu_netpp\outputs\external_eval\demo" ^
  --device cuda ^
  --split test different_test
```

批量跑 3-fold 并汇总：

```powershell
powershell -ExecutionPolicy Bypass -File experiments\mbu_netpp\scripts\run_nasa_external_eval.ps1
python -m experiments.mbu_netpp.summarize_external_eval ^
  --eval-root "C:\Users\pyd111\Desktop\中期\wd_bishe\experiments\mbu_netpp\outputs\external_eval\e3_micronet_edge_deep_vf_gpu\nasa_super_all"
```

### 用 NASA Super 训练红色 `secondary` 二分类

如果你要把 NASA `Super1~4` 当成训练集，而不是只做外部验证，推荐先只准备红色
`secondary` 前景版本，再与本地真实标签数据统一整合。

## 8. 三源真实标签合并训练 + 服务器流程

当前仓库已支持把下面三份数据合并成一个统一监督训练集：

- `analysis_same_teacher_nocap_095\1`：`26` 张真实 LabelMe 标注
- `全部已完成标注图像_精修底图`：`16` 张真实 LabelMe 标注
- `prepared_nasa_super_secondary_train`：`35` 张 NASA prepared 训练样本

### 8.1 本地先准备两份 LabelMe 数据

```bash
python -m experiments.mbu_netpp.preparation ^
  --images-dir "C:\Users\pyd111\Desktop\analysis_same_teacher_nocap_095\1" ^
  --annotations-dir "C:\Users\pyd111\Desktop\analysis_same_teacher_nocap_095\1" ^
  --output-root "D:\中期\wd_bishe\experiments\mbu_netpp\workdir\prepared_analysis_real26" ^
  --no-auto-crop

python -m experiments.mbu_netpp.preparation ^
  --images-dir "C:\Users\pyd111\Desktop\已完成标注图像\全部已完成标注图像_精修底图" ^
  --annotations-dir "C:\Users\pyd111\Desktop\已完成标注图像\全部已完成标注图像_精修底图" ^
  --output-root "D:\中期\wd_bishe\experiments\mbu_netpp\workdir\prepared_refined_real16" ^
  --no-auto-crop
```

### 8.2 合并三份 prepared 数据

```bash
python -m experiments.mbu_netpp.prepare_merged_supervised ^
  --prepared-root "D:\中期\wd_bishe\experiments\mbu_netpp\workdir\prepared_analysis_real26" ^
  --source-alias real26 ^
  --prepared-root "D:\中期\wd_bishe\experiments\mbu_netpp\workdir\prepared_refined_real16" ^
  --source-alias real16 ^
  --prepared-root "C:\Users\pyd111\Desktop\prepared_nasa_super_secondary_train" ^
  --source-alias nasa ^
  --output-root "D:\中期\wd_bishe\experiments\mbu_netpp\workdir\prepared_merged_real77"
```

训练配置已提供：

- `experiments/mbu_netpp/configs/merged_real77_supervised.yaml`

```bash
python -m experiments.mbu_netpp.train ^
  --config experiments/mbu_netpp/configs/merged_real77_supervised.yaml ^
  --run-all-folds
```

### 8.3 服务器流程

新增脚本位于：

- `experiments/mbu_netpp/scripts/server/install_env.sh`
- `experiments/mbu_netpp/scripts/server/run_merged_real77_pipeline.sh`
- `experiments/mbu_netpp/scripts/server/upload_bundle.ps1`
- `experiments/mbu_netpp/scripts/server/download_results.ps1`

默认服务器流程是：

1. 本地执行 `upload_bundle.ps1`，上传最小运行 bundle
2. 服务器执行 `install_env.sh`
3. 服务器执行 `run_merged_real77_pipeline.sh`
4. 本地执行 `download_results.ps1`

### 8.4 最终交付测试集

服务器训练完成后，最终不是只看 NASA `test`，而是固定对这 `100` 张已裁掉信息栏的图做推理：

- `dataset/full_png_cropped_xlsx/images`

最终会回传：

- `masks/`
- `overlays/`
- `stats/`
- `summary.csv`
- `summary.xlsx`
`train + val`，并且只保留红色 `secondary` 前景：

```bash
python -m experiments.mbu_netpp.prepare_nasa_super ^
  --dataset-root "D:\中期\wd_bishe\external_data\nasa_benchmark_segmentation_data" ^
  --output-root "D:\中期\wd_bishe\experiments\mbu_netpp\workdir\prepared_nasa_super_secondary_train" ^
  --splits train val ^
  --foreground-colors-bgr 255,0,0 ^
  --num-folds 3 ^
  --seed 42
```

这条命令会同时生成：

- 训练用二值 mask
- 边界标签
- `manifests/dataset.json`
- `manifests/folds_3_seed42.json`

随后可直接使用：

```bash
python -m experiments.mbu_netpp.train ^
  --config experiments/mbu_netpp/configs/nasa_super_secondary_supervised.yaml ^
  --run-all-folds
```

说明：

- 生成的 NASA prepared 数据会自动把 `stem` 改成带 `subset/split` 前缀的唯一键，避免不同子集里的同名样本互相覆盖。
- 如果你只想训练某几个 `Super` 子集，可以额外传 `--subsets Super1 Super2`。
