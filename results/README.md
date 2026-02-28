# 结果目录说明

本目录按“方法名 -> 统一 10 张展示样例”的方式组织，便于在 GitHub 中直接横向比较不同模型的预测效果。

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

每个方法目录中均包含上述 10 张图对应的：

- `_mask.png`
- `_overlay.png`

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
| `sam_v1_metrics_only/` | SAM V1 | 仅保留指标说明，未保留独立预测图 |

## 使用建议

- 如果要看最佳轮廓质量，先看 `sam_lora/`
- 如果要看体积分数统计更接近经验值的结果，先看 `resnext/`
- 如果要看零样本和微调模型差异，直接对比 `matsam/` 与 `sam_lora/`
