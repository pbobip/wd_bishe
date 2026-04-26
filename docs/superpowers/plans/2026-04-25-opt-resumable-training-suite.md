# 优化实验可恢复训练套件 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 `MBU-Net++` 新一轮优化实验补齐可恢复训练、加权采样、NASA 扩展数据构造和双并发服务器总控。

**Architecture:** 在现有 `train.py` 和 `dataset.py` 上做最小必要扩展；新增一个数据构造脚本和一个 Python 任务调度器，再由薄 shell 包装脚本负责服务器调用。训练状态与任务状态分别落盘，以支持断电后重新执行同一入口直接续跑。

**Tech Stack:** Python, PyTorch, YAML, JSON, Bash, pytest

---

### Task 1: 固化设计与实现范围

**Files:**
- Create: `docs/superpowers/specs/2026-04-25-opt-resumable-training-suite-design.md`
- Create: `docs/superpowers/plans/2026-04-25-opt-resumable-training-suite.md`

- [ ] 记录本轮 10 组实验、数据口径、恢复与调度约束
- [ ] 固化实现计划，避免边做边改范围

### Task 2: 为采样与合并数据构造补测试

**Files:**
- Create: `tests/test_mbu_netpp_opt_dataset_builder.py`
- Create: `tests/test_mbu_netpp_sampling_weights.py`
- Create: `tests/test_mbu_netpp_train_resume.py`

- [ ] 先写 failing tests，覆盖：
  - `NASA train-only` fold 构造
  - `merged78_cv5_ext10` fold 构造
  - `sampling_weight + hard_stem_weight`
  - `train.py --summarize-only`
  - `last.pt` 断点恢复辅助函数

### Task 3: 实现优化数据集构造脚本

**Files:**
- Create: `experiments/mbu_netpp/prepare_opt_training_sets.py`

- [ ] 新增数据构造脚本，支持：
  - `real53`
  - `nasa_train_only`
  - `merged78_cv5_ext10`
- [ ] 输出 `dataset.json` 与 fold manifest
- [ ] 复用现有文件复制与 manifest 结构，保持兼容 `holdout_eval`

### Task 4: 实现真正的 sampling_weight 与难样本采样

**Files:**
- Modify: `experiments/mbu_netpp/dataset.py`

- [ ] 在训练集记录上构建采样权重
- [ ] 支持 `sampling_weight`
- [ ] 支持 `hard_stems` 与 `hard_stem_weight`
- [ ] 保留现有前景/边界裁块逻辑，并允许通过配置强化

### Task 5: 实现 train.py 断点恢复与 summary-only

**Files:**
- Modify: `experiments/mbu_netpp/train.py`

- [ ] 新增 `last.pt` 周期性保存
- [ ] 新增 `--resume`
- [ ] 新增 `--summarize-only`
- [ ] fold 已完成时自动跳过
- [ ] 汇总逻辑独立成可复用函数

### Task 6: 新增 10 组优化实验配置

**Files:**
- Create: `experiments/mbu_netpp/configs/opt_real53_baseline.yaml`
- Create: `experiments/mbu_netpp/configs/opt_real53_edge_up.yaml`
- Create: `experiments/mbu_netpp/configs/opt_real53_deep_up.yaml`
- Create: `experiments/mbu_netpp/configs/opt_real53_edge_deep_up.yaml`
- Create: `experiments/mbu_netpp/configs/opt_real53_vf_002.yaml`
- Create: `experiments/mbu_netpp/configs/opt_real53_vf_005.yaml`
- Create: `experiments/mbu_netpp/configs/opt_real53_boundary_sampling.yaml`
- Create: `experiments/mbu_netpp/configs/opt_real53_boundary_hard_sampling.yaml`
- Create: `experiments/mbu_netpp/configs/opt_real53_nasa_train_only.yaml`
- Create: `experiments/mbu_netpp/configs/opt_real53_merged78_cv5_ext10.yaml`

- [ ] 统一配置格式
- [ ] 只修改必要差异项
- [ ] 对 NASA 两组配置指向新 prepared_root

### Task 7: 实现服务器端可恢复任务调度器

**Files:**
- Create: `experiments/mbu_netpp/run_opt_training_suite.py`

- [ ] 任务拆为 fold 级别
- [ ] 维护 `tasks.json`
- [ ] 实现双并发 worker
- [ ] 写心跳
- [ ] 已完成任务跳过
- [ ] 失败任务留痕并支持重启后继续

### Task 8: 新增服务器入口脚本

**Files:**
- Create: `experiments/mbu_netpp/scripts/server/run_opt_training_suite.sh`
- Create: `experiments/mbu_netpp/scripts/server/upload_opt_suite_bundle.ps1`
- Create: `experiments/mbu_netpp/scripts/server/download_opt_suite_results.ps1`

- [ ] shell 入口只负责环境准备与调用 Python 调度器
- [ ] 上传脚本带新配置与新脚本
- [ ] 下载脚本回传完整结果包

### Task 9: 本地验证

**Files:**
- Modify: `tests/...` as needed

- [ ] 跑新增 pytest
- [ ] 用临时目录验证新数据构造脚本
- [ ] 用单 fold 冒烟验证 `--resume` / `--summarize-only`
- [ ] 检查新配置 YAML 可加载

### Task 10: 交付说明

**Files:**
- Modify: `experiments/mbu_netpp/README.md` if needed

- [ ] 记录运行命令
- [ ] 记录恢复方式
- [ ] 记录结果目录结构
