# 镍基单晶 SEM 图像分割统计系统

本仓库对应毕业设计《面向扫描电镜图像的镍基单晶微观结构分割统计系统开发》。  
截至 `2026-03-29`，**当前主方案**为 `MBU-Net++` 路线，前期的 `SAM LoRA`、`ResNeXt50`、`MatSAM + SAM2` 等结果保留为**尝试记录、对照路线或预标注工具**。

---

## 项目总览

本项目的核心目标是：**针对 SEM 图像中 γ/γ′ 相灰度对比弱、边界复杂、标注匮乏的问题，构建一套从分割到统计的完整系统**。

项目包含三条主线：

| 主线 | 核心内容 | 完成度 |
| --- | --- | :---: |
| **研究主线** | MBU-Net++ 模型设计与 E1a-E5 消融实验 | ✅ 完成 |
| **数据主线** | MatSAM 粗标注 → 人工精修 → 9 张金标准 | ✅ 完成 |
| **系统主线** | FastAPI + Vue 3 的 B/S 分割统计系统 | 🔧 ~80% |

---

## 研究链路

```
原始 SEM 图像 (100张)
  → 格式转换 + 底部信息栏裁切
  → MatSAM 零样本粗标注（粗 ROI + Grid Prompt → SAM → NMS）
  → 人工逐张精修 → 9 张高质量标注图
  → MBU-Net++ 主方案（MicroNet-pretrained SE-ResNeXt50 + U-Net++ + Edge Head + Deep Supervision + VF Loss）
  → E1a-E5 消融实验验证
  → 系统整合为「分割 + 统计 + 可视化 + 导出」的 Web 平台
```

详细流程图见 [`docs/diagrams/`](docs/diagrams/)。

---

## 当前主方案与实验结论

### 模型架构

当前主方案针对开题报告的四类核心问题逐一设计：

| 组件 | 解决的问题 | 设计原理 |
| --- | --- | --- |
| MicroNet 预训练 | 灰度差异小 | 显微图像域预训练，增强纹理和微观形貌表征 |
| U-Net++ | 边界复杂 | 密集跳连融合多尺度特征，提升细粒度结构恢复 |
| Edge Head + Deep Supervision | 边界复杂 + 标注匮乏 | 独立建模边界位置，多尺度输出约束稳定小样本训练 |
| VF Loss | 统计稳定性 | `L_vf = (mean(Ŷ) - mean(Y))²`，训练目标对齐统计目标 |

### 实验结果

| 实验 | 内容 | Dice | BF1 | VF↓ | 判定 |
| --- | --- | ---: | ---: | ---: | --- |
| E1a | 增强对照 | 0.9296 | — | 0.0425 | ✅ |
| E1 | Baseline 三组 | 0.9305 | 0.7188 | — | ✅ |
| **E2** | **+ Edge + Deep** | **0.9321** | **0.7276** | 0.0371 | **⭐ 分割/边界最优** |
| **E3** | **E2 + VF Loss** | 0.9309 | 0.7211 | **0.0332** | **⭐ 最均衡** |
| E4 | 半监督伪标签 | +0.002 | +0.011 | +0.002 | ⚠ 有限 |
| E5 | 后处理评估 | — | — | — | ❌ morph_smooth 否定 |

---

## 系统完成情况

系统采用 **FastAPI (后端) + Vue 3 / Vite (前端)** 的 B/S 架构，实现「分割 → 可视化 → 统计分析 → 结果导出」的完整闭环。

### 系统架构

```
前端交互层 (Vue 3 + Element Plus)
  ↓ HTTP REST API
API 路由层 (FastAPI)
  ↓ task_manager 调度
主执行流水线 (pipeline.py)
  → Step 1: Input — SEM ROI 检测 + 底栏裁切
  → Step 2: Preprocess — CLAHE / 去噪
  → Step 3: Traditional — 传统分割 (Otsu / 自适应阈值)
  → Step 4: DL — 深度学习分割 (MBU-Net++ 滑窗推理)
  → Step 5: Stats — 统计摘要 + 图表
  → Step 6: Export — 结果导出 (PNG/CSV/XLSX)
  ↓
核心服务模块 + 数据持久层 (SQLAlchemy + SQLite)
```

### 后端模块完成度

| 模块 | 文件 | 功能 | 状态 |
| --- | --- | --- | :---: |
| API 路由 | `api/routes.py` | 项目/任务/图像/标定/模型管理 9 个端点 | ✅ |
| 执行流水线 | `services/pipeline.py` | 六步串行流水线 + 进度报告 + 批量聚合 | ✅ |
| SEM ROI 检测 | `services/sem_roi.py` | 底部信息栏自动分离 + 分析区域提取 + 比例尺横线定位 | ✅ |
| 底栏 OCR | `services/sem_footer_ocr.py` | 比例尺数值/放大倍率/WD/检测器类型识别 → 自动算 um_per_px | ✅ |
| 图像预处理 | `services/preprocess.py` | CLAHE 对比度增强、轻量去噪 | ✅ |
| 传统分割 | `services/algorithms/traditional.py` | Otsu / 自适应阈值 → mask + overlay + edges | ✅ |
| DL 推理调度 | `services/model_runner.py` | 加载权重、滑窗推理、GPU/CPU 调度 | ✅ |
| 统计分析 ⭐ | `services/statistics.py` | VF / 面积 / 等效直径 / Feret 径 / 形态学指标 / Distance Transform 通道宽度 / Lantuéjoul 加权 | ✅ |
| 结果导出 | `services/exporter.py` | PNG 可视化 / CSV / XLSX 报表 / 打包下载 | ✅ |
| 文件存储 | `services/storage.py` | 路径映射 / 静态文件服务 | ✅ |
| 任务管理 | `services/task_manager.py` | 后台线程调度执行 | ✅ |
| 数据模型 | `models/entities.py` | Project / RunTask / ImageAsset / MetricRecord / ExportRecord | ✅ |

### 前端页面完成度

| 页面 | 文件 | 功能 | 状态 |
| --- | --- | --- | :---: |
| 仪表盘 | `DashboardView.vue` | 项目概览 + 快速入口 | ✅ |
| 项目管理 | `ProjectManagementView.vue` | 项目 CRUD | ✅ |
| 任务配置 | `TaskConfigView.vue` | 分割模式/预处理/标定/模型选择 | ✅ |
| 任务执行 | `TaskRunView.vue` | 进度跟踪 + 实时状态 | ✅ |
| 结果浏览 | `ResultsView.vue` | 分割结果查看 + 对比模式 | ✅ |
| 统计图表 | `RunStatisticsView.vue` | 面积/尺寸分布直方图 + VF 柱状图 | ✅ |
| 运行详情 | `RunDetailView.vue` | 单次运行结果总览 | ✅ |
| 历史记录 | `HistoryView.vue` | 历史任务列表 | ✅ |
| 标定提示横幅 | `CalibrationStatusBanner.vue` | 标定状态提示 | ✅ |
| 预处理预览 | `PreviewWorkspace.vue` | 预处理效果预览 | ✅ |
| 统计侧边栏 | `StatsSidebar.vue` | 统计摘要展示 | ✅ |
| 图表画布 | `ChartCanvas.vue` | 动态图表渲染 | ✅ |

### 系统待完成项

| 项目 | 说明 | 优先级 |
| --- | --- | :---: |
| MBU-Net++ 权重接入 | 将训练产出的最优权重 (.pth) 注册到系统 model-runner | 🔴 高 |
| 端到端演示验证 | 完整走通「上传 → 分割 → 统计 → 导出」流程 | 🔴 高 |
| 一键启动脚本 | `start.bat` 同时启动前后端 | 🟡 中 |
| 结果对比页面增强 | 传统/DL 分割结果并排对比展示 | 🟡 中 |

---

## 数据来源

| 数据集 | 内容 | 定位 |
| --- | --- | --- |
| `dataset/full_png/` | 100 张原始 PNG 输入图 | 未标注池 |
| `dataset/full_png_cropped_xlsx/` | 基于 xlsx 类型信息裁切后的版本 | 清洗后数据 |
| `samples/annotated_eval_set/` | **9 张高质量精修标注图** | **主监督数据** |

当前 9 张精修图来自：

1. `MatSAM` 零样本粗标注生成初始 mask 底稿
2. 人工逐张修正漏检/过分割/边界误差

详见 [`docs/MatSAM_粗标注思路.md`](docs/MatSAM_粗标注思路.md)

---

## 前期探索路线定位

| 路线 | 当前定位 | 说明 |
| --- | --- | --- |
| SAM LoRA V2 | 历史比较材料 | Dice=0.9499，前期最高精度尝试 |
| ResNeXt50 | 历史比较材料 | VF=60.44%，前期最佳统计型模型 |
| MatSAM | 零样本对照 + 粗标注工具 | 为人工精修提供标注起点 |
| MatSAM + SAM2 | 自动预标注路线 | 生成粗底稿、支持扩标 |

这些路线的价值：
- 说明前期做过充分的模型和路线探索
- 为当前主方案提供实验对照背景
- 为后续扩标和难例精修继续服务

---

## 毕设整体进度

整体完成度：**~85%**

### 已完成

- [x] 开题报告、任务书和题目目标梳理
- [x] 原始数据整理、TIF/PNG 处理与标注规范
- [x] MatSAM 粗标注与人工精修，形成 9 张主监督样本
- [x] MBU-Net++ 主方案设计
- [x] E1a-E5 核心实验主线
- [x] full_png 底部信息栏清理、批量预测和异常排查
- [x] 后端 API + 六步执行流水线
- [x] 统计分析模块（Distance Transform 通道宽度 + Lantuéjoul 加权）
- [x] 前端 8 个页面 + 4 个组件
- [x] SEM ROI 自动检测 + OCR 自动标定

### 待完成

- [ ] 难例精标后的增量训练
- [ ] MBU-Net++ 最优权重接入系统 + 端到端演示
- [ ] 论文正文、图表和答辩材料收束

---

## 仓库结构

```
wd_bishe/
├── backend/                    # 后端服务
│   └── app/
│       ├── api/routes.py       # FastAPI 路由（9 个端点）
│       ├── models/entities.py  # ORM 数据模型
│       ├── schemas/            # Pydantic 请求/响应模型
│       ├── services/
│       │   ├── pipeline.py     # 六步主执行流水线
│       │   ├── sem_roi.py      # SEM ROI 检测
│       │   ├── sem_footer_ocr.py # 底栏 OCR
│       │   ├── preprocess.py   # 图像预处理
│       │   ├── statistics.py   # 统计分析（核心）
│       │   ├── model_runner.py # DL 推理调度
│       │   ├── exporter.py     # 结果导出
│       │   ├── storage.py      # 文件存储
│       │   └── algorithms/     # 传统分割算法
│       ├── db/                 # 数据库会话管理
│       └── utils/              # 工具函数
├── frontend/                   # 前端应用
│   └── src/
│       ├── views/              # 8 个页面视图
│       ├── components/         # 4 个功能组件
│       ├── router.ts           # 路由配置
│       ├── stores/             # 状态管理
│       └── composables/        # 组合式函数
├── experiments/
│   └── mbu_netpp/              # 当前主方案训练代码
│       ├── models.py           # MBU-Net++ 模型定义
│       ├── train.py            # 训练流程
│       ├── dataset.py          # 数据加载 + 增强
│       ├── losses.py           # 复合损失函数
│       ├── metrics.py          # 评估指标
│       └── infer.py            # 滑窗推理
├── dataset/                    # 数据集
│   ├── full_png/               # 100 张原始 PNG
│   └── full_png_cropped_xlsx/  # 裁切清洗后版本
├── samples/
│   └── annotated_eval_set/     # 9 张精修标注
├── docs/                       # 文档
│   ├── mbu_netpp_research/     # 主方案研究文档
│   ├── diagrams/               # draw.io 流程图
│   ├── LATEST_STATUS_2026-03-29.md
│   ├── MatSAM_粗标注思路.md
│   └── 当前工作总结报告.md
├── results/                    # 前期方法结果
├── scripts/                    # 数据处理脚本
└── start.bat                   # 启动脚本
```

---

## 快速导航

| 想看什么 | 入口 |
| --- | --- |
| 当前最新状态 | [`docs/LATEST_STATUS_2026-03-29.md`](docs/LATEST_STATUS_2026-03-29.md) |
| 研究链路流程图 | [`docs/diagrams/`](docs/diagrams/) |
| MatSAM 思路与数据来源 | [`docs/MatSAM_粗标注思路.md`](docs/MatSAM_粗标注思路.md) |
| 当前主方案研究区 | [`docs/mbu_netpp_research/README.md`](docs/mbu_netpp_research/README.md) |
| E4/E5 最新结果 | [`docs/mbu_netpp_research/08_E4_E5结果.md`](docs/mbu_netpp_research/08_E4_E5结果.md) |
| 当前总结报告 | [`docs/当前工作总结报告.md`](docs/当前工作总结报告.md) |
| 完整输入数据 | [`dataset/full_png/`](dataset/full_png/) |
| 精标样本 | [`samples/annotated_eval_set/`](samples/annotated_eval_set/) |
| 前期方法结果 | [`results/README.md`](results/README.md) |
| 当前实验代码 | [`experiments/mbu_netpp/`](experiments/mbu_netpp/) |
| 系统后端代码 | [`backend/app/`](backend/app/) |
| 系统前端代码 | [`frontend/src/`](frontend/src/) |

---

## 推荐阅读顺序

1. 本 README（总览）
2. [`docs/LATEST_STATUS_2026-03-29.md`](docs/LATEST_STATUS_2026-03-29.md)（最新进展）
3. [`docs/mbu_netpp_research/README.md`](docs/mbu_netpp_research/README.md)（主方案研究区）
4. [`docs/mbu_netpp_research/06_阶段性结果汇总.md`](docs/mbu_netpp_research/06_阶段性结果汇总.md)（实验结果）
5. [`docs/当前工作总结报告.md`](docs/当前工作总结报告.md)（完整总结）

---

## 说明

本仓库同时记录两件事：

- **当前毕业设计主线的真实进展**（MBU-Net++ + 系统开发）
- **前期探索、尝试和预标注路线的阶段性产物**

阅读时请以 `2026-03-29` 之后的 MBU-Net++ 路线和系统开发内容为主。
