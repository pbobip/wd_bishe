# 优化实验可恢复训练套件设计

## 背景

当前 `experiments/mbu_netpp` 已具备 `real53` 数据的 `5-fold + holdout10` 训练、传统方法评估、结果汇总与打包能力，但现有服务器脚本是串行批处理，存在三个核心问题：

1. 训练任务按实验整组串行执行，无法稳定吃满单卡 `RTX 5090` 的算力。
2. 训练进度只在 fold 结束后落盘，服务器断电或用户定时关机会导致当前 fold 从头开始。
3. 现有流程无法表达本轮新增的优化实验：
   - `VF weight`
   - `edge/deep weight`
   - 难样本/边界采样
   - `NASA secondary` 辅助训练与混合数据集实验

本设计的目标是在不推翻现有训练主线的前提下，构建一套可恢复、可重入、可并发的完整实验套件。

## 目标

1. 固化一套 10 组完整实验，避免组合爆炸，同时覆盖本轮所有研究问题。
2. 为训练脚本增加 fold 级断点续训能力，支持 `last.pt` 恢复。
3. 为数据集增加真正的 `sampling_weight` 采样能力，支持更强边界采样与难样本采样。
4. 新增两类扩展数据构造：
   - `NASA train-only`
   - `merged78_cv5_ext10`
5. 新增服务器端总控，支持：
   - 双并发 worker
   - 任务状态文件
   - 心跳检测
   - 已完成任务跳过
   - 断电后重启续跑
6. 保留现有结果产物形式：
   - `crossval_summary.json`
   - `holdout_eval`
   - `plots`
   - `suite_summary`
   - 最终打包下载

## 实验矩阵

主评估口径固定为：

- 基础真实数据：`real53`
- 固定测试集：`holdout10`
- 交叉验证：`5-fold`
- 主指标优先级：
  1. `holdout boundary_f1`
  2. `holdout dice`
  3. `holdout vf`

本轮完整实验共 10 组：

1. `baseline`
   - `MBU + Edge + Deep`
   - `edge=0.3, deep=0.2, vf=0`
2. `edge_up`
   - `edge=0.4, deep=0.2`
3. `deep_up`
   - `edge=0.3, deep=0.3`
4. `edge_deep_up`
   - `edge=0.4, deep=0.3`
5. `vf_002`
   - `vf=0.02`
6. `vf_005`
   - `vf=0.05`
7. `boundary_sampling`
   - 更强边界采样
8. `boundary_hard_sampling`
   - 更强边界采样 + 难样本采样
9. `nasa_train_only`
   - `train = real53_train_fold + nasa35`
   - `val = real53_val_fold`
   - `test = real53 holdout10`
10. `merged78_cv5_ext10`
   - `trainval = real53_trainval43 + nasa35 = 78`
   - 在 `78` 上重做 `5-fold`
   - 最终测试仍用 `real53 holdout10`

## 数据构造设计

### 1. real53 主线

沿用现有：

- `prepared_real53_cv5`
- `holdout_10_seed42.json`
- `folds_5_seed42_holdout10.json`

### 2. NASA train-only

新增一个合并脚本，输出新的 `prepared_root`：

- 数据集包含：
  - `real53` 全部样本
  - `nasa35` 全部样本
- fold 定义：
  - `train_stems = real_train_stems + nasa_all_stems`
  - `val_stems = real_val_stems`
  - `test_stems = real_holdout10`

NASA 样本的 `sample_weight` 与 `sampling_weight` 由配置控制。

### 3. merged78_cv5_ext10

新增一个混合训练集构造：

- 数据集包含：
  - `real53` 全部样本
  - `nasa35` 全部样本
- fold 生成范围：
  - `real53` 的 `trainval43`
  - `nasa35`
- 固定测试集：
  - `real53 holdout10`
- fold 中：
  - `train_stems + val_stems` 只来自 `real43 + nasa35`
  - `test_stems` 固定为 `real holdout10`

该实验用于回答“NASA 真正并入数据集后是否有帮助”，但它不替代 `real53` 主表。

## 采样设计

### 1. loss 权重

现有 `sample_weight` 仅影响 loss。该逻辑保留。

### 2. sampling_weight

新增真正的训练样本采样权重：

- 训练阶段根据每条 record 的 `sampling_weight` 做加权抽样
- 若未提供，则默认 `1.0`

### 3. 难样本采样

在配置中增加：

- `hard_stems`
- `hard_stem_weight`

若样本 stem 命中 `hard_stems`，则额外乘上难样本采样权重。

### 4. 边界采样

继续沿用随机裁块，但允许通过配置强化：

- `min_edge_ratio`
- `crop_attempts`

`boundary_sampling` 与 `boundary_hard_sampling` 只通过这些参数和 `sampling_weight` 改行为，不再引入新模型分支。

## 训练恢复设计

### 1. fold 级 checkpoint

每个 fold 目录下新增：

- `last.pt`
- `best.pt`
- `history.json`
- `summary.json`

`last.pt` 包含：

- `epoch`
- `model_state_dict`
- `optimizer_state_dict`
- `scheduler_state_dict`
- `scaler_state_dict`
- `history`
- `best_metric`
- `best_summary`

### 2. resume 逻辑

`train.py` 新增：

- `--resume`
- `--summarize-only`

行为：

- 若 fold 已存在 `summary.json`，直接跳过
- 若 `--resume` 且存在 `last.pt`，从下一 epoch 继续训练
- `--summarize-only` 只汇总已有 fold 结果，生成 `crossval_summary.json`

## 服务器端总控设计

### 1. 任务粒度

任务以 fold 为最小执行单元：

- `train:<experiment>:fold_<i>`
- `summarize:<experiment>`
- `plot:<experiment>`
- `holdout:<experiment>`
- `suite_summary`
- `package`

### 2. 状态文件

总控维护：

- `suite_state/tasks.json`
- `suite_state/heartbeats/worker_<n>.json`
- `suite_state/logs/*.log`

每个任务状态包含：

- `pending`
- `running`
- `done`
- `failed`

### 3. 可重入

总控每次启动时先做状态对账：

- 若目标产物已存在，则标 `done`
- 若任务没有完成产物，则标 `pending`
- 若 fold 有 `last.pt` 但无 `summary.json`，则保留为可恢复训练任务

### 4. 并发

默认 `2` 个 worker，支持环境变量调整：

- `MAX_WORKERS=2`

不默认上 `3`，以夜间稳定性优先。

### 5. 心跳检测

worker 定时写心跳。若总控重启后发现：

- 任务状态为 `running`
- 但无存活进程

则重置为 `pending`，等待重新调度。

## 产物与下载

结果结构延续现有模式，额外补充：

- 状态与日志目录
- 运行总表
- 每实验运行配置快照

最终交付仍包含：

- `manifests/`
- `traditional_eval/`
- `suite_summary/`
- `experiments/<name>/best.pt`
- `experiments/<name>/holdout_eval/`
- `experiments/<name>/plots/`
- `pipeline.log`
- `environment_report.json`

## 运行时间预估

在单卡 `RTX 5090`、双并发 worker 模式下，按上一轮真实训练速度估计：

- 单组完整实验：约 `2.0 - 2.8` 小时
- 10 组完整实验：核心 wall time 约 `12 - 16` 小时
- 加数据准备、传统评估、汇总、打包：总 wall time 约 `14 - 18` 小时

建议按 **约 16 小时** 做夜间运行准备。

## 风险与边界

1. 断电恢复依赖 `last.pt` 周期性落盘；若单个 epoch 很长，则本 epoch 内进度仍会丢失。
2. 若代码级错误导致任务重复失败，总控不会无限盲重试，而是保留失败记录供人工介入。
3. `merged78_cv5_ext10` 是扩展实验，不应替代 `real53` 主结果表。
