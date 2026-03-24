# 最新进展（2026-03-24）

本文档用于补充仓库当前阶段的最新状态，重点说明最近完成的 `MatSAM + SAM2` 自动预标注路线及其定位。

## 本次更新聚焦什么

本轮工作的目标不是替代前面已经完成的监督学习模型比较，而是解决一个更现实的问题：

- 现有零样本 `MatSAM` 路线粘连明显，直接用于精修不够友好；
- 需要一条更适合“先自动出草稿，再人工精修”的预标注主线；
- 评价重点从单纯追求召回，转为优先减少明显粘连，并尽量保持单块形状完整。

因此，本轮主要尝试并比较了以下几条 `SAM2` 路线：

- `localfix`
- `best_fusion`
- `hybrid_debond`
- `localfix_strict`
- `localfix_strict_otsu`

## 当前推荐结果

基于目前已经完成的 100 张图实验，当前最适合作为**自动预标注主线**的结果是：

- `MatSAM + SAM2 + localfix_strict_otsu`

对应本地结果摘要如下：

| 方案 | image_count | vf_mean | components_mean | split_count_mean | 当前判断 |
| --- | ---: | ---: | ---: | ---: | --- |
| `localfix_strict` | 100 | 0.591883 | 120.10 | 1.29 | 更积极拆分，适合进一步排查 |
| `localfix_strict_otsu` | 100 | 0.585334 | 118.77 | 1.10 | 当前推荐的自动预标注版本 |

补充观察：

- `localfix_strict_otsu` 在 100 张图中有 `51/100` 张触发了局部拆分；
- `vf < 0.5` 的样本有 `17/100` 张，说明它仍然偏保守；
- 它没有像 `hybrid_debond` 那样出现“全图过度切割”的问题，整体更可控；
- 它擅长修复“已经被 base 检测出来的大块粘连”，但对本来就没有进入前景的漏检区域帮助有限。

## 为什么当前选它

选择 `localfix_strict_otsu` 不是因为它在所有指标上都最好，而是因为它更符合当前阶段的真实需求：

1. 当前最需要的是**给后续人工精修提供更干净的起点**，而不是单纯提高覆盖率。
2. `hybrid_debond` 虽然更敢拆，但已经出现明显过度切割，不适合直接作为精修底稿。
3. `best_fusion` 过于保守，很多粘连几乎没有真正被改掉。
4. `localfix_strict_otsu` 虽然会带来一定漏检，但整体更接近“宁可漏一点，也不要把明显分开的块粘在一起”的目标。

## 代表性样例

当前仓库同步放入了 4 张代表性 `preview`，位于：

- [`results/matsam_sam2_localfix_strict_otsu/100_preview.png`](../results/matsam_sam2_localfix_strict_otsu/100_preview.png)
- [`results/matsam_sam2_localfix_strict_otsu/73_preview.png`](../results/matsam_sam2_localfix_strict_otsu/73_preview.png)
- [`results/matsam_sam2_localfix_strict_otsu/112_preview.png`](../results/matsam_sam2_localfix_strict_otsu/112_preview.png)
- [`results/matsam_sam2_localfix_strict_otsu/262_preview.png`](../results/matsam_sam2_localfix_strict_otsu/262_preview.png)

它们分别对应：

- `100`、`73`：代表性较好的 anti-merge 样本；
- `112`、`262`：代表性较差的低对比/漏检样本。

## 当前结论

到目前为止，可以明确得出的判断是：

- `SAM2` 这条路线更适合承担**自动预标注工具**角色；
- 它已经足够用于生成当前阶段的精修底稿；
- 但它还不是最终生产模型，更不适合继续靠堆后处理规则来同时解决所有粘连和漏检问题。

## 下一步方向

下一步不建议继续无止境地叠加后处理规则，而应当把当前结果转化为更高价值的监督数据。

推荐顺序如下：

1. 以 `localfix_strict_otsu` 作为当前自动预标注底稿。
2. 优先人工精修低对比、断边、漏检严重样本。
3. 形成一批高质量精修标签后，训练专门的监督学习分割模型。
4. 后续模型方向优先考虑：
   - 边界感知的语义分割模型；
   - 前景分割 + 边界分割双头结构；
   - 若最终要做实例级统计，再进一步转向实例分割路线。

一句话总结：

> 当前阶段，`localfix_strict_otsu` 的价值不在于成为最终最优模型，而在于成为“最适合继续精修和扩充标注数据”的自动预标注起点。
