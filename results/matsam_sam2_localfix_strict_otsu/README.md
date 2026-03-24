# MatSAM + SAM2 + localfix strict_otsu

这是当前阶段推荐用于**自动预标注**的结果目录。

它的定位不是最终最优模型，而是：

- 优先减少明显粘连；
- 尽量保持单块形状完整；
- 为后续人工精修提供更友好的起点。

## 当前结果摘要

- 模型：`SAM2.1 base_plus`
- 主策略：`B_balance_otsu`
- 局部修复：`localfix strict`
- 样本数：`100`
- `vf_mean = 0.585334`
- `components_mean = 118.77`
- `split_count_mean = 1.10`

补充观察：

- `51/100` 张图触发了局部拆分；
- 它比激进的 `hybrid_debond` 更稳；
- 它仍然存在漏检，因此更适合“自动预标注 + 人工精修”，而不是直接作为最终结果。

## 当前放入的代表图

目录中保存了 4 张 `preview`：

- `100_preview.png`
- `73_preview.png`
- `112_preview.png`
- `262_preview.png`

每张图从左到右依次为：

- `raw`
- `base`
- `localfix`

其中：

- `100`、`73`：代表性较好的 anti-merge 样本；
- `112`、`262`：代表性较差的低对比/漏检样本。

## 样例展示

### 有效样例

![100 preview](100_preview.png)

![73 preview](73_preview.png)

### 困难样例

![112 preview](112_preview.png)

![262 preview](262_preview.png)

## 怎么使用这套结果

当前建议：

1. 用它作为第一轮自动预标注底稿。
2. 优先修正低对比、边界断裂、漏检严重样本。
3. 累积一批高质量精修标签后，训练专门的监督学习分割模型。

如果要看完整背景和下一步方向，回到：

- [`../../docs/LATEST_STATUS_2026-03-24.md`](../../docs/LATEST_STATUS_2026-03-24.md)
