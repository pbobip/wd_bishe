# 最新状态说明（2026-04-26）

本文件记录当前毕业设计仓库的论文收束口径，避免继续沿用 2026-03 旧实验结论。

## 当前主线

毕业设计题目为“面向扫描电镜图像的镍基单晶微观结构分割统计系统开发”。当前主线由两部分组成：

1. 模型实验：使用 `MBU-Net++ boundary sampling` 完成 SEM 图像中 `gamma_prime` 相二值分割。
2. 系统实现：使用 FastAPI + Vue 3 实现 SEM 图像导入、分割、后处理、统计分析和结果导出。

## 模型与实验口径

| 项目 | 当前结论 |
| --- | --- |
| 数据规模 | 53 张 SEM 图像 + 53 个 LabelMe JSON，4268 个 `gamma_prime` polygon |
| 数据划分 | 10 张固定 holdout 测试集，43 张进行 5-fold 训练验证 |
| 输入尺寸 | 训练 patch size 为 256，推理 overlap 为 0.25 |
| 主模型 | `MBU-Net++`，MicroNet 预训练 `se_resnext50_32x4d` 编码器，U-Net++ 解码器，边界辅助监督，深监督，边界采样 |
| 主模型结果 | holdout Dice 0.964984，IoU 0.932976，Precision 0.954685，Recall 0.976702，Boundary F1 0.929954，VF Error 0.024045 |
| 传统 baseline | Otsu+CLAHE 为最佳传统路线，holdout Dice 约 0.893 |
| 重复实验 | 已完成 3 个随机种子重复实验，固定 holdout10 和 fold 划分，仅改变训练随机性 |
| 效率指标 | 已补充参数量、FLOPs/MACs、本地 CPU 推理耗时和 RTX 5090 GPU 推理耗时 |

## 系统实现口径

系统采用前后端分离架构：

| 层级 | 技术与职责 |
| --- | --- |
| 前端 | Vue 3、Vite、TypeScript、Element Plus、Pinia、ECharts，负责任务创建、参数配置、运行状态、结果查看和统计图表 |
| 后端 | FastAPI、SQLite、SQLAlchemy、OpenCV、scikit-image、RapidOCR、PyTorch 子进程，负责任务调度、图像处理、分割推理、统计和导出 |
| 分割模块 | 传统分割支持 threshold、adaptive、edge、clustering；深度学习 runner 真实绑定 `mbu_netpp`、`matsam`、`custom` |
| 统计模块 | 计算体积分数、颗粒数量、面积、等效直径、Feret、圆度、长宽比、通道宽度和 Lantuéjoul 加权统计 |
| 导出模块 | 支持 XLSX、JSON、ZIP bundle 和 Word 报告 |

系统代码的核心入口：

- 后端 API：`backend/app/api/routes.py`
- 流水线：`backend/app/services/pipeline.py`
- 深度学习 runner：`backend/app/services/model_runner.py`、`backend/app/services/runners/dl_infer.py`
- 统计：`backend/app/services/statistics.py`
- 导出：`backend/app/services/exporter.py`、`backend/app/services/report_service.py`
- 前端主流程：`frontend/src/views/TaskConfigView.vue`、`frontend/src/views/RunDetailView.vue`、`frontend/src/views/RunStatisticsView.vue`

## 论文材料入口

| 材料 | 路径 |
| --- | --- |
| 模型与实验部分说明 | `毕业设计整理/04_论文资料/模型与实验部分说明.md` |
| 模型与实验材料清单 | `毕业设计整理/04_论文资料/毕设模型与实验材料和说明清单.md` |
| 系统材料清单 | `毕业设计整理/毕设系统材料和说明清单_2026-04-26.md` |
| 论文图表示例 | `results/paper_figures/` |

## 不纳入 git 的内容

以下内容保留在本地或归档目录，不直接纳入 GitHub：

- 模型权重文件，如 `.pt`、`.pth`、`.ckpt`
- 服务器下载的完整训练结果包和 `.tar.gz` 大文件
- 后端运行数据库和 `backend/storage/`
- `experiments/mbu_netpp/outputs/` 与 `workdir/`
- 批量预测中间结果和可由脚本重新生成的运行产物

## 写论文时不要夸大的内容

- 不要把 SAM LoRA、ResNeXt50、SegFormer 等写成当前最终主模型。
- 不要把系统预留的 runner 槽位写成已经完成完整实验。
- 不要把 100 张无人工真值预测图写成独立测试集。
- 不要写“完全自动物理标定可靠”，SEM 底部信息栏和 OCR 结果仍需要人工确认。
- 不要写“工业级部署”，当前定位是毕业设计实验验证与辅助分析平台。
