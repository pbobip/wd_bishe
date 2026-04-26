# 结果目录说明

本目录主要按“方法名 -> 统一 10 张展示样例”的方式组织，便于在 GitHub 中直接横向比较不同模型的预测效果。

## 2026-04-26 补充说明

论文当前主结果不再以本目录早期 `sam_lora`、`resnext`、`matsam` 等探索路线作为最终模型结论。最终论文主模型采用 `opt_real53_boundary_sampling`，完整训练结果和权重体积较大，保留在本地归档，不直接纳入 git。

当前纳入仓库、可直接用于论文的轻量图表位于：

- `results/paper_figures/holdout_traditional_vs_mbu_metrics_bar.png`
- `results/paper_figures/holdout_traditional_vs_mbu_typical_comparison.png`
- `results/paper_figures/holdout_typical_samples_selection.csv`

服务器下载的完整结果目录、`.tar.gz` 大包和模型权重不纳入 git，避免仓库体积失控。

## 展示样例

统一抽取的 10 张样例如下：

- `01-10`
- `01-14`
- `104`
- `13`
- `3`
- `32`
- `3233`
- `36`
- `37`
- `48`

大多数监督学习/对照方法目录中均包含上述 10 张图对应的：

- `_mask.png`
- `_overlay.png`

例外说明：

- `matsam_sam2_localfix_strict_otsu/` 保存的是当前自动预标注路线的代表性 `preview`，不是统一 10 张展示样例全集。

## 方法与目录映射

| 目录 | 方法 | 说明 |
| --- | --- | --- |
| `sam_lora/` | SAM LoRA V2 | 当前最佳高精度模型 |
| `resnext/` | ResNeXt50 | 最适合体积分数统计 |
| `micronet/` | MicroNet | 轻量均衡 |
| `convnext/` | ConvNeXt 512 | 现代 CNN 对照 |
| `swint/` | Swin Transformer | Transformer 对照 |
| `smp/` | SMP | 存在过分割倾向 |
| `sam_frozen/` | SAM (冻结) | 仅训练解码器 |
| `unet/` | UNet | 经典基准模型 |
| `matsam/` | MatSAM | 零样本基线 |
| `matsam_sam2_localfix_strict_otsu/` | MatSAM + SAM2 + localfix strict_otsu | 当前自动预标注推荐结果，优先减少粘连 |
| `sam_v1_metrics_only/` | SAM V1 | 仅保留指标说明，未保留独立预测图 |

## 使用建议

- 如果要看最佳轮廓质量，先看 `sam_lora/`
- 如果要看体积分数统计更接近经验值的结果，先看 `resnext/`
- 如果要看零样本和微调模型差异，直接对比 `matsam/` 与 `sam_lora/`
- 如果要看当前最适合继续人工精修的自动预标注底稿，先看 `matsam_sam2_localfix_strict_otsu/`

## 当前自动预标注结果说明

`matsam_sam2_localfix_strict_otsu/` 与前面的监督学习模型目录用途不同：

- 它不是最终模型效果展示目录；
- 它保存的是最近一轮 `MatSAM + SAM2` 路线的代表性 `preview`；
- 每张 `preview` 从左到右依次为：`raw / base / localfix`。

当前放入了 4 张代表性样例：

- `100_preview.png`
- `73_preview.png`
- `112_preview.png`
- `262_preview.png`

其中：

- `100`、`73` 更能体现局部拆粘连的收益；
- `112`、`262` 更能体现当前方法在低对比和漏检样本上的局限。
