# 镍基单晶 SEM 图像分割统计项目

本仓库对应毕业设计《面向扫描电镜图像的镍基单晶微观结构分割统计系统开发》的整理提交版，目标是把当前阶段最重要的成果集中到一个可读、可展示、可复用的 GitHub 仓库中。

最近一次更新：`2026-03-24`

仓库包含：
- 完整 `100` 张实验输入 `PNG` 数据
- 监督学习主结果与当前自动预标注结果
- 精标样本、总结报告、图表与关键脚本

## 核心结论

- 最佳高精度模型：`SAM LoRA V2`，`Dice = 0.9499`
- 最适合体积分数统计的模型：`ResNeXt50`，`Dice = 0.9398`，`VF = 60.44%`
- 零样本对照方法：`MatSAM`
- 当前自动预标注推荐结果：`MatSAM + SAM2 + localfix_strict_otsu`

## 近期更新（2026-03-24）

本轮新增了一条更贴近实际精修需求的自动预标注路线：

- `MatSAM + SAM2 + localfix_strict_otsu`

这条路线的定位不是替代前面已经完成的监督学习模型结论，而是解决“当前如何更高效地产生可精修底稿”的问题。

当前判断如下：

- 它比早期零样本 `MatSAM` 更适合作为预标注起点；
- 它比激进的 `hybrid_debond` 更稳，不容易把单块切碎；
- 它仍然存在漏检，但在“宁可漏一点，也尽量减少明显粘连”这个目标下更实用；
- 现阶段最合理的使用方式是：先用它出草稿，再人工精修，再训练专门监督模型。

## 快速导航

| 想看什么 | 入口 |
| --- | --- |
| 最新进展 | [`docs/LATEST_STATUS_2026-03-24.md`](docs/LATEST_STATUS_2026-03-24.md) |
| 总结报告 | [`docs/当前工作总结报告.md`](docs/当前工作总结报告.md) |
| 最终模型评价 | [`docs/FINAL_REPORT.md`](docs/FINAL_REPORT.md) |
| 完整输入数据 | [`dataset/full_png/`](dataset/full_png/) |
| 精标样本 | [`samples/annotated_eval_set/`](samples/annotated_eval_set/) |
| 展示样例 | [`samples/showcase_set/`](samples/showcase_set/) |
| 方法结果对比 | [`results/README.md`](results/README.md) |
| 当前自动预标注结果 | [`results/matsam_sam2_localfix_strict_otsu/`](results/matsam_sam2_localfix_strict_otsu/) |
| 关键脚本 | [`scripts/`](scripts/) |
| 未纳入内容说明 | [`docs/manifests/未纳入大文件清单.md`](docs/manifests/未纳入大文件清单.md) |

## 一眼看懂结果

- `SAM LoRA V2`：边界最干净，最适合高精度形貌分析。
- `ResNeXt50`：`Dice` 高且 `VF` 更接近统计需求，最适合体积分数统计。
- `MicroNet`、`ConvNeXt`、`Swin Transformer`：可用，但不是当前最优解。
- `SMP`、`SAM V1`：`VF` 偏高，存在明显过分割风险。
- `MatSAM`：零样本可直接出结果，但明显弱于微调后的监督学习模型。
- `MatSAM + SAM2 + localfix_strict_otsu`：当前最适合作为人工精修前的自动预标注底稿。

![各方法指标定位图](docs/charts/model_positioning.png)

## 代表性样例对比

下图统一展示 3 张代表性样例，从左到右依次为：原图、标注真值、`SAM LoRA V2`、`ResNeXt50`、`SMP`、`MatSAM`。

![代表性样例分割对比](docs/charts/sample_showcase.png)

## 方法效果概览

| 方法 | 仓库目录 | Dice | VF | 定位 |
| --- | --- | ---: | ---: | --- |
| SAM LoRA V2 | `results/sam_lora/` | 0.9499 | 38.36% | 最佳高精度模型 |
| ResNeXt50 | `results/resnext/` | 0.9398 | 60.44% | 最适合体积分数统计 |
| MicroNet | `results/micronet/` | 0.9233 | 57.58% | 轻量且均衡 |
| ConvNeXt 512 | `results/convnext/` | 0.9070 | 52.32% | 中上水平 |
| Swin Transformer | `results/swint/` | 0.9004 | 58.41% | Transformer 对照 |
| SMP | `results/smp/` | 0.8707 | 74.84% | 过分割 |
| SAM (冻结) | `results/sam_frozen/` | 0.8100 | 52.28% | 冻结编码器对照 |
| UNet | `results/unet/` | 0.7892 | 61.77% | 经典基准模型 |
| MatSAM | `results/matsam/` | - | 30.38% | 零样本基线 |
| MatSAM + SAM2 + localfix_strict_otsu | `results/matsam_sam2_localfix_strict_otsu/` | - | 58.53%* | 当前自动预标注推荐结果 |
| SAM V1 | `results/sam_v1_metrics_only/` | 0.6534 | 86.02% | 失败对照，仅保留指标 |

![各方法效果对比](docs/charts/方法效果对比.png)

\* `58.53%` 为当前 100 张图自动预标注结果的 `vf_mean`，不与前述监督学习模型的 `Dice` 直接横向等价比较。详见 [`docs/LATEST_STATUS_2026-03-24.md`](docs/LATEST_STATUS_2026-03-24.md)。

## 技术路线

1. 原始 SEM 图像整理与 `TIF -> PNG` 预处理
2. 制定 γ′ 相标注规范，完成小样本精标
3. 建立监督学习基线与迁移学习流程
4. 引入 `SAM LoRA`、`ResNeXt`、`MicroNet`、`Swin Transformer` 等多模型对比
5. 引入 `MatSAM` 作为零样本对照路线
6. 在零样本路线基础上继续引入 `SAM2 + localfix`，探索更适合人工精修的自动预标注方案
7. 输出分割结果，并进行 `Dice` 和 `VF` 评价

![技术路线流程图](docs/charts/技术路线流程图.png)

## 仓库内容

- [`dataset/full_png/`](dataset/full_png/)
  完整 `100` 张实验输入 `PNG` 原图。
- [`samples/annotated_eval_set/`](samples/annotated_eval_set/)
  `9` 份精标样本原图与 `LabelMe JSON`。
- [`samples/showcase_set/`](samples/showcase_set/)
  用于横向展示的 `10` 张统一样例。
- [`results/`](results/)
  监督学习方法结果与零样本基线的统一展示结果。
- [`results/matsam_sam2_localfix_strict_otsu/`](results/matsam_sam2_localfix_strict_otsu/)
  当前自动预标注推荐结果的代表性 `preview`。
- [`docs/`](docs/)
  总结报告、最终报告、图表与内容清单。
- [`scripts/`](scripts/)
  数据预处理、预测、后处理与统计脚本。

## 使用说明

### 克隆仓库

本仓库使用 `Git LFS` 管理图片。首次克隆前请确保本机已安装 `git lfs`。

```bash
git lfs install
git clone git@github.com:pbobip/wd_bishe.git
cd wd_bishe
git lfs pull
```

### 推荐阅读顺序

1. [`docs/当前工作总结报告.md`](docs/当前工作总结报告.md)
2. [`docs/LATEST_STATUS_2026-03-24.md`](docs/LATEST_STATUS_2026-03-24.md)
3. [`docs/FINAL_REPORT.md`](docs/FINAL_REPORT.md)
4. [`results/README.md`](results/README.md)
5. [`samples/showcase_set/README.md`](samples/showcase_set/README.md)
6. [`docs/manifests/未纳入大文件清单.md`](docs/manifests/未纳入大文件清单.md)

## 目录结构

```text
wd_bishe/
├── README.md
├── .gitattributes
├── .gitignore
├── dataset/
│   ├── README.md
│   └── full_png/
├── docs/
│   ├── 当前工作总结报告.md
│   ├── LATEST_STATUS_2026-03-24.md
│   ├── FINAL_REPORT.md
│   ├── comparison_report_v2_v3.md
│   ├── charts/
│   └── manifests/
├── results/
│   ├── matsam_sam2_localfix_strict_otsu/
├── samples/
└── scripts/
```

## 仓库边界

这个仓库是面向 GitHub 展示和答辩说明的整理包，不是原始实验工作区的完整镜像。以下内容没有直接纳入：

- 模型权重文件，如 `*.pth`
- 全量预测结果和中间缓存
- Kaggle 训练中间产物
- 原始 `TIF` 数据与压缩包
- 当前 `SAM2` 路线的完整 100 图输出目录，仅保留代表性预览图

详细说明见 [`docs/manifests/未纳入大文件清单.md`](docs/manifests/未纳入大文件清单.md)。
