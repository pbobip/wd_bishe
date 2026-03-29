# MBU-Net++ 实验区

本目录承载 `MBU-Net++` 的独立实验代码，和现有展示型后端解耦。

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
