# Merged Real77 Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在本仓库内补齐三源真实标签合并、服务器训练与最终结果回传的完整脚本链路。

**Architecture:** 先把两份 `LabelMe` 数据转成标准 `prepared_root`，再和 NASA prepared 训练集合并为新的 `prepared_merged_real77`。服务器侧通过最小运行 bundle 解包后完成环境安装、训练、最佳 fold 选择、最终 `100` 张图推理，以及结果打包下载。

**Tech Stack:** Python, PowerShell, Bash, PyTorch, OpenCV

---

### Task 1: 合并 prepared 数据

**Files:**
- Create: `experiments/mbu_netpp/prepare_merged_supervised.py`
- Create: `tests/test_mbu_netpp_prepare_merged_supervised.py`

- [ ] 新增多来源 prepared 合并逻辑，复制 images/masks/edges/previews 到统一输出根目录。
- [ ] 生成统一 `dataset.json` 和 `folds_3_seed42.json`。
- [ ] 补一个最小单测，验证 stem 去重、文件复制与 fold 生成。

### Task 2: 新增训练配置

**Files:**
- Create: `experiments/mbu_netpp/configs/merged_real77_supervised.yaml`

- [ ] 提供统一监督训练配置，默认指向 `prepared_merged_real77`。

### Task 3: 新增服务器脚本

**Files:**
- Create: `experiments/mbu_netpp/scripts/server/install_env.sh`
- Create: `experiments/mbu_netpp/scripts/server/run_merged_real77_pipeline.sh`
- Create: `experiments/mbu_netpp/scripts/server/upload_bundle.ps1`
- Create: `experiments/mbu_netpp/scripts/server/download_results.ps1`

- [ ] 编写 Linux 环境安装脚本。
- [ ] 编写远端准备、训练、最佳 fold 推理、结果打包脚本。
- [ ] 编写本地上传 bundle 脚本。
- [ ] 编写本地下载最终结果脚本。

### Task 4: 文档与依赖

**Files:**
- Modify: `experiments/mbu_netpp/README.md`
- Modify: `experiments/mbu_netpp/requirements.txt`

- [ ] 补 `pydantic` 依赖，保证统计模块在服务器可导入。
- [ ] 更新 README，写清楚本地准备、服务器执行与最终测试集口径。

### Task 5: 本地验证

**Files:**
- Generate: `experiments/mbu_netpp/workdir/prepared_analysis_real26`
- Generate: `experiments/mbu_netpp/workdir/prepared_refined_real16`
- Generate: `experiments/mbu_netpp/workdir/prepared_merged_real77`

- [ ] 运行新单测。
- [ ] 在本地生成三源合并训练集。
- [ ] 核对样本数是否为 `77`，并确认 manifest 生成正常。
