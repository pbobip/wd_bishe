# Model Comparison Report: V2 (Baseline) vs V3 (Evolved)

This report highlights the differences between the original model (V2) and the fine-tuned model (V3) on the `标注2` dataset (100 images).

## 1. Executive Summary

We performed a **Volatility Audit** to identify where the model changed the most.
- **Total Images Audited**: 100
- **Comparisons Generated**: 100 (Full Set)

**Key Finding**: The high change rates in the top images suggest the V3 model has significantly altered its prediction strategy for difficult samples.

## 2. 核心数据解释 (Metrics Explanation)

报告中的表格包含两列关键数据，含义如下：

*   **Change Rate (变化率)**: 
    *   **定义**: 像素发生改变的比例。
    *   **解读**: 
        *   `0.01` (1%): 几乎没变。
        *   `0.50` (50%): 模型完全推翻了之前的判断（"变脸"了）。
    *   **用途**: 帮您快速找到“模型学到了新东西”或者“模型学坏了”的图片。

*   **IoU (Intersection over Union, 交并比)**:
    *   **定义**: 两个掩膜重叠的程度。
    *   **解读**: 
        *   `1.0`: 完美重合。
        *   `0.0`: 完全不沾边。
    *   **用途**: IoU 越低，说明两个模型的判断差异越大。

## 3. Top 5 Most Changed Images (Top 5 剧变与分歧最大的图)

这些图片是您审查的重点。

| Rank | Image ID | Change Rate | IoU (Overlap) | Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| 1 | **35** | 66.5% | 0.03 | **完全不一致**。模型看到了完全不同的东西。 |
| 2 | **37** | 61.2% | 0.09 | 几乎无重叠。 |
| 3 | **172** | 54.6% | 0.20 | 结构发生重大改变。 |
| 4 | **42** | 52.2% | 0.30 | 主要边界发生偏移。 |
| 5 | **32** | 51.9% | 0.13 | 极低的一致性。 |

## 4. How to Review (如何审查)

请查看以下文件夹中的对比图：
`c:\Users\pyd111\Desktop\标注1\project_v3_transfer\audit_report\all_comparisons\`

图片文件名格式：`compare_{ImageID}.jpg`

**图片布局 (从左到右)**:
1.  **Original Raw**: 原始图片（最真实的背景）。
2.  **V2 (Baseline)**: 旧模型预测的覆盖层（红色）。
3.  **V3 (Evolved)**: 新模型预测的覆盖层（绿色）。

**审查标准**:
- 对比 **Raw** 和 **V3 (绿色)**。
- 绿色覆盖的区域是否真的是颗粒？
- 绿色是否比红色（旧模型）漏检更少、误检更少？
