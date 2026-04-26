# MBU-Net++ 代码地图

这个目录只放**训练、评估、推理、数据准备**相关代码，不放 Web 平台业务逻辑。

## 1. 入口分层

### 训练入口

- `train.py`
  监督训练主入口。
- `iterative_semi_supervised.py`
  迭代式半监督训练入口。

### 推理与评估入口

- `infer.py`
  单独推理入口。
- `external_eval.py`
  外部数据集评估。
- `evaluate_postprocess.py`
  后处理评估。
- `semi_supervised_eval.py`
  半监督结果评估。

### 数据准备入口

- `preparation.py`
  通用准备入口。
- `prepare_merged_supervised.py`
  合并监督数据准备。
- `prepare_nasa_super.py`
  NASA Super1~4 数据准备。
- `prepare_full_png_dataset.py`
  full_png 系列数据准备。
- `export_labelme_from_masks.py`
  从 mask 导出 LabelMe。

## 2. 核心模块

- `models.py`
  模型定义。
- `dataset.py`
  数据集与采样逻辑。
- `losses.py`
  损失函数。
- `metrics.py`
  指标计算。
- `common.py`
  公共训练辅助。
- `semi_scoring.py`
  半监督样本打分。
- `semi_utils.py`
  半监督辅助逻辑。

## 3. 配置与脚本

- `configs/`
  实验配置文件，按监督、半监督、NASA、服务器方案分开。
- `scripts/run_gpu_*.ps1`
  本地 GPU 批量实验脚本。
- `scripts/server/`
  服务器安装、上传、执行、下载脚本。

## 4. 目录边界

- `outputs/`
  实验输出目录，属于生成物，不属于源码。
- `workdir/`
  数据准备与中间产物目录，属于生成物，不属于源码。
- `__pycache__/`
  Python 缓存。

如果只是改 Web 平台分割流程，不应该碰 `outputs/`、`workdir/`，也不应该把训练逻辑塞回 `backend/app/`。

## 5. 与系统后端的接口边界

当前系统后端真正会接触模型实验区的，原则上只有两类东西：

1. 训练产出的权重文件
2. 推理所需的最小配置与运行约定

也就是说：

- 训练策略、半监督迭代、外部评估，留在 `experiments/mbu_netpp/`
- Web 运行态调度、结果展示、统计与导出，留在 `backend/ + frontend/`

这条边界后面不要再打穿。
