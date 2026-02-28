# 镍基单晶 SEM 图像分割统计项目

这是毕业设计《面向扫描电镜图像的镍基单晶微观结构分割统计系统开发》的 GitHub 整理提交包。仓库保留了当前阶段最关键的文档、核心脚本、代表性样本和各方法的对比结果，用于快速说明技术路线与阶段成果。

## 项目结论

- 最佳高精度模型：`SAM LoRA V2`，`Dice = 0.9499`
- 最适合体积分数统计的模型：`ResNeXt50`，`Dice = 0.9398`，`VF = 60.44%`
- 零样本对照方法：`MatSAM`，可作为未微调大模型基线

## 技术路线

1. 原始 SEM 图像整理与 `TIF -> PNG` 预处理
2. 制定 γ′ 相标注规范，完成小样本精标
3. 建立监督学习基线与迁移学习流程
4. 引入 `SAM LoRA`、`ResNeXt`、`MicroNet`、`Swin Transformer` 等多模型对比
5. 引入 `MatSAM` 作为零样本对照路线
6. 输出分割结果，并进行 `Dice` 和 `VF` 评价

![技术路线流程图](docs/charts/技术路线流程图.png)

## 方法效果概览

| 方法 | 仓库目录 | Dice | VF | 定位 |
| --- | --- | ---: | ---: | --- |
| SAM LoRA V2 | `results/sam_lora/` | 0.9499 | 38.36% | 最佳高精度模型 |
| ResNeXt50 | `results/resnext/` | 0.9398 | 60.44% | 最适合体积分数统计 |
| MicroNet | `results/micronet/` | 0.9233 | 57.58% | 轻量且均衡 |
| ConvNeXt 512 | `results/convnext/` | 0.9070 | 52.32% | 中等偏稳 |
| Swin Transformer | `results/swint/` | 0.9004 | 58.41% | Transformer 对照 |
| SMP | `results/smp/` | 0.8707 | 74.84% | 存在过分割 |
| SAM (冻结) | `results/sam_frozen/` | 0.8100 | 52.28% | 只训练解码器 |
| UNet | `results/unet/` | 0.7892 | 61.77% | 经典基准模型 |
| MatSAM | `results/matsam/` | - | 30.38% | 零样本基线 |
| SAM V1 | `results/sam_v1_metrics_only/` | 0.6534 | 86.02% | 仅保留指标记录 |

![各方法效果对比](docs/charts/方法效果对比.png)

## 仓库里有什么

- [`docs/当前工作总结报告.md`](docs/当前工作总结报告.md)
  当前阶段总结、技术路线、指标解读与图表。
- [`docs/FINAL_REPORT.md`](docs/FINAL_REPORT.md)
  各模型最终评价与推荐结论。
- [`samples/annotated_eval_set/`](samples/annotated_eval_set/)
  9 份精标样本原图与 `LabelMe JSON`。
- [`samples/showcase_set/`](samples/showcase_set/)
  用于展示各方法结果的 10 张统一样例原图。
- [`results/`](results/)
  9 个可展示方法的统一 10 样例预测结果，每张图包含 `mask` 和 `overlay`。
- [`scripts/`](scripts/)
  数据预处理、分割预测、后处理和统计相关核心脚本。

## 推荐阅读顺序

1. [`docs/当前工作总结报告.md`](docs/当前工作总结报告.md)
2. [`docs/FINAL_REPORT.md`](docs/FINAL_REPORT.md)
3. [`results/README.md`](results/README.md)
4. [`samples/showcase_set/README.md`](samples/showcase_set/README.md)
5. [`docs/manifests/未纳入大文件清单.md`](docs/manifests/未纳入大文件清单.md)

## 目录结构

```text
毕设_github提交包/
├── README.md
├── .gitignore
├── docs/
│   ├── 当前工作总结报告.md
│   ├── FINAL_REPORT.md
│   ├── comparison_report_v2_v3.md
│   ├── charts/
│   └── manifests/
├── scripts/
│   ├── data_prep/
│   ├── supervised/
│   ├── zero_shot/
│   ├── postprocess/
│   └── model_selection/
├── samples/
│   ├── annotated_eval_set/
│   └── showcase_set/
└── results/
    ├── convnext/
    ├── matsam/
    ├── micronet/
    ├── resnext/
    ├── sam_frozen/
    ├── sam_lora/
    ├── sam_v1_metrics_only/
    ├── smp/
    ├── swint/
    └── unet/
```

## 为什么不是全量原始目录

这个仓库是面向 GitHub 展示和答辩说明的整理包，不是原始实验工作区的完整镜像。以下内容没有直接纳入：

- 模型权重文件，如 `*.pth`
- 全量预测结果和中间缓存
- Kaggle 训练中间产物
- 全量原始数据压缩包

相关说明见 [`docs/manifests/未纳入大文件清单.md`](docs/manifests/未纳入大文件清单.md)。
