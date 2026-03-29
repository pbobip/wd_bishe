# 镍基单晶 SEM 图像分割统计项目

本仓库对应毕业设计《面向扫描电镜图像的镍基单晶微观结构分割统计系统开发》的 GitHub 整理版。  
截至 `2026-03-29`，仓库中的**当前主方案**已经切换为 `MBU-Net++` 路线，前期的 `SAM LoRA`、`ResNeXt50`、`MatSAM + SAM2` 等结果保留为**尝试记录、对照路线或预标注路线**，不再作为当前论文主线。

## 当前主线

当前论文主线基于开题报告里已经明确提出的四类问题展开：

- `SEM` 图像灰度差异小，γ/γ′ 相对比弱
- 颗粒边界复杂，单纯区域分割容易粘连或漏分
- 高质量标注获取成本高，监督样本少
- 课题目标不只是分割，还要做面积分数和尺寸统计，要求结果稳定可用

针对这些问题，当前主方案定为：

- `MicroNet-pretrained SE-ResNeXt50 + U-Net++ + Edge Head + Deep Supervision`
- 在统计约束版本中进一步加入 `VF Loss`

当前主监督数据来源为：

- 前期以 `MatSAM` 自动粗标注为底稿
- 人工逐张精修后形成的 `9` 张高质量标注图

这 `9` 张精修图是当前阶段最核心、最可信的监督数据来源，也是当前 `E1a-E5` 全部实验的基础。

如果要看这条链是怎么形成的，可以直接看：

- [`docs/MatSAM_粗标注思路.md`](docs/MatSAM_粗标注思路.md)

## 当前结论

### 当前最值得保留的模型

- 分割与边界最优：`E2 = MicroNet-U-Net++ + Edge Head + Deep Supervision`
  - `Dice = 0.9321`
  - `Boundary F1 = 0.7276`
  - `VF = 0.0371`
- 统计更均衡的版本：`E3 = E2 + VF Loss`
  - `Dice = 0.9309`
  - `Boundary F1 = 0.7211`
  - `VF = 0.0332`

### E4 与 E5 的真实结论

- `E4` 半监督伪标签：有小幅正收益，但主要体现在分割和边界，`VF` 误差反而变差，暂时不能直接升格为最终方案。
- `E5` 后处理：`remove_small` 只有极小收益；激进的形态学平滑会明显打坏结果，应该否掉。

### 当前阶段判断

- 当前主方案已经从“前期多路线尝试”转入“围绕开题报告问题做针对性改进”的阶段。
- 目前更值得继续做的是：
  - 基于修正裁切后的 `full_png` 挑难例精标
  - 在当前主方案上做更干净的增量训练
  - 最后收口到系统演示和论文图表

## 毕设进度

当前整体进度大致可判断为 `85%` 左右。

已经完成：

- 开题报告、任务书和题目目标梳理
- 原始数据整理、`TIF/PNG` 处理与标注规范
- `MatSAM` 粗标注与人工精修，形成 `9` 张主监督样本
- `E1a-E5` 的核心实验主线
- `full_png` 的底部信息栏清理、批量预测和异常排查
- 公开数据检索、NASA 数据接入与外部验证

还需要继续收口：

- 难例精标后的增量训练
- 最终系统整合与展示
- 论文正文、图表和答辩材料收束

## 前期尝试与其定位

仓库中保留的以下路线，仍然有价值，但当前定位已经变化：

- `SAM LoRA V2`
  - 前期高精度尝试结果，保留为历史比较材料
- `ResNeXt50`
  - 前期统计型尝试结果，保留为历史比较材料
- `MatSAM`
  - 零样本对照方法
- `MatSAM + SAM2 + localfix_strict_otsu`
  - 自动预标注路线，用于生成粗底稿和扩充精修入口

这些内容现在的作用是：

- 说明前期已经做过较充分的模型和路线探索
- 为当前主方案提供对照背景
- 为后续扩标和难例精修继续服务

而不是继续作为当前论文主模型结论。

## 快速导航

| 想看什么 | 入口 |
| --- | --- |
| 当前最新状态 | [`docs/LATEST_STATUS_2026-03-29.md`](docs/LATEST_STATUS_2026-03-29.md) |
| `MatSAM` 思路与数据来源 | [`docs/MatSAM_粗标注思路.md`](docs/MatSAM_粗标注思路.md) |
| 当前主方案研究区 | [`docs/mbu_netpp_research/README.md`](docs/mbu_netpp_research/README.md) |
| `E4/E5` 最新结果 | [`docs/mbu_netpp_research/08_E4_E5结果.md`](docs/mbu_netpp_research/08_E4_E5结果.md) |
| 当前总结报告 | [`docs/当前工作总结报告.md`](docs/当前工作总结报告.md) |
| 前一轮自动预标注状态 | [`docs/LATEST_STATUS_2026-03-24.md`](docs/LATEST_STATUS_2026-03-24.md) |
| 完整输入数据 | [`dataset/full_png/`](dataset/full_png/) |
| 修正裁切后的 `full_png` 数据集 | [`dataset/full_png_cropped_xlsx/`](dataset/full_png_cropped_xlsx/) |
| 精标样本 | [`samples/annotated_eval_set/`](samples/annotated_eval_set/) |
| 前期方法结果 | [`results/README.md`](results/README.md) |
| 当前实验代码 | [`experiments/mbu_netpp/`](experiments/mbu_netpp/) |

## 仓库内容说明

- [`dataset/full_png/`](dataset/full_png/)
  - `100` 张原始 `PNG` 输入图
- [`dataset/full_png_cropped_xlsx/`](dataset/full_png_cropped_xlsx/)
  - 基于 `xlsx` 类型信息完成底部信息栏清理后的版本
- [`samples/annotated_eval_set/`](samples/annotated_eval_set/)
  - 当前最核心的 `9` 张精修图和对应标注
- [`docs/mbu_netpp_research/`](docs/mbu_netpp_research/)
  - 当前主方案的设计、实验、外部验证和 `E4/E5` 结果
- [`results/`](results/)
  - 前期对比方法和自动预标注路线的展示材料
- [`experiments/mbu_netpp/`](experiments/mbu_netpp/)
  - 当前主方案训练、评估、推理与数据处理代码

## 推荐阅读顺序

1. [`docs/LATEST_STATUS_2026-03-29.md`](docs/LATEST_STATUS_2026-03-29.md)
2. [`docs/mbu_netpp_research/README.md`](docs/mbu_netpp_research/README.md)
3. [`docs/mbu_netpp_research/06_阶段性结果汇总.md`](docs/mbu_netpp_research/06_阶段性结果汇总.md)
4. [`docs/mbu_netpp_research/08_E4_E5结果.md`](docs/mbu_netpp_research/08_E4_E5结果.md)
5. [`docs/当前工作总结报告.md`](docs/当前工作总结报告.md)

## 说明

这个仓库现在同时承担两件事：

- 记录当前毕业设计主线的真实进展
- 保存前期探索、尝试和自动预标注路线的阶段性产物

因此，阅读时请优先以 `2026-03-29` 之后的 `MBU-Net++` 路线为主，不要再把更早的 `SAM LoRA / ResNeXt / MatSAM + SAM2` 结果理解为当前论文主方案。
