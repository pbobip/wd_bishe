# 可提交分组清单

目的：把当前工作区里的改动拆成 4 个可控提交组，避免前端流程重构、后端运行链路、模型实验和文档工具混在一个提交里。

## 分组 1：前端主流程重构

适合提交主题：

`feat(frontend): 重构 01/02/03/04 主流程与结果工作台`

建议范围：

- `frontend/src/App.vue`
- `frontend/src/router.ts`
- `frontend/src/composables/useTaskWorkflow.ts`
- `frontend/src/views/TaskConfigView.vue`
- `frontend/src/views/TaskRunView.vue`
- `frontend/src/views/RunDetailView.vue`
- `frontend/src/views/RunStatisticsView.vue`
- `frontend/src/views/HistoryView.vue`
- `frontend/src/views/ResultsView.vue`
- `frontend/src/views/DashboardView.vue`
- `frontend/src/views/ProjectManagementView.vue`
- `frontend/src/components/PreviewWorkspace.vue`
- `frontend/src/components/PostprocessPreviewDialog.vue`
- `frontend/src/components/StatsSidebar.vue`
- `frontend/src/components/CalibrationStatusBanner.vue`
- `frontend/src/components/ZoomableImageDialog.vue`
- `frontend/src/utils/runStatistics.ts`
- `frontend/src/utils/calibration.ts`
- `frontend/src/types.ts`
- `frontend/src/styles.css`
- `frontend/package.json`
- `frontend/package-lock.json`
- `frontend/components.d.ts`
- `frontend/README.md`

直接暂存：

```powershell
git -C D:\中期\wd_bishe add --pathspec-from-file=docs/pathspecs/01-frontend-flow.txt
```

## 分组 2：后端运行链路与后处理确认

适合提交主题：

`feat(backend): 增加后处理预览确认与确认后结果重建`

建议范围：

- `backend/app/api/routes.py`
- `backend/app/schemas/run.py`
- `backend/app/schemas/project.py`
- `backend/app/models/entities.py`
- `backend/app/db/init_db.py`
- `backend/app/services/pipeline.py`
- `backend/app/services/postprocess_preview.py`
- `backend/app/services/exporter.py`
- `backend/app/services/algorithms/traditional.py`
- `backend/app/services/sem_roi.py`
- `backend/tests/test_postprocess_preview_confirm.py`
- `backend/README.md`

直接暂存：

```powershell
git -C D:\中期\wd_bishe add --pathspec-from-file=docs/pathspecs/02-backend-runflow.txt
```

## 分组 3：模型实验与训练辅助扩展

适合提交主题：

`feat(model): 增补半监督、服务器脚本与数据准备链路`

建议范围：

- `experiments/mbu_netpp/train.py`
- `experiments/mbu_netpp/dataset.py`
- `experiments/mbu_netpp/losses.py`
- `experiments/mbu_netpp/preparation.py`
- `experiments/mbu_netpp/prepare_nasa_super.py`
- `experiments/mbu_netpp/prepare_merged_supervised.py`
- `experiments/mbu_netpp/iterative_semi_supervised.py`
- `experiments/mbu_netpp/export_labelme_from_masks.py`
- `experiments/mbu_netpp/semi_scoring.py`
- `experiments/mbu_netpp/semi_utils.py`
- `experiments/mbu_netpp/requirements.txt`
- `experiments/mbu_netpp/README.md`
- `experiments/mbu_netpp/CODEMAP.md`
- `experiments/mbu_netpp/configs/default_semi_supervised_iterative.yaml`
- `experiments/mbu_netpp/configs/merged_real77_supervised.yaml`
- `experiments/mbu_netpp/configs/merged_real77_supervised_server_full.yaml`
- `experiments/mbu_netpp/configs/nasa_super_secondary_supervised.yaml`
- `experiments/mbu_netpp/scripts/server/*`
- `tests/test_mbu_netpp_prepare_merged_supervised.py`
- `tests/test_mbu_netpp_prepare_nasa_super.py`
- `tests/test_mbu_netpp_semi_supervised.py`

直接暂存：

```powershell
git -C D:\中期\wd_bishe add --pathspec-from-file=docs/pathspecs/03-model-experiments.txt
```

## 分组 4：文档与开发辅助

适合提交主题：

`docs: 补代码地图与工作区分组索引`

建议范围：

- `.gitignore`
- `README.md`
- `docs/CODEBASE_STREAMS_2026-04-23.md`
- `docs/COMMIT_GROUPS_2026-04-23.md`
- `docs/WORKTREE_REVIEW_SPLIT_2026-04-23.md`
- `docs/pathspecs/01-frontend-flow.txt`
- `docs/pathspecs/02-backend-runflow.txt`
- `docs/pathspecs/03-model-experiments.txt`
- `docs/pathspecs/04-docs-and-tooling.txt`
- `docs/superpowers/plans/2026-04-20-shape-based-material-classification.md`
- `docs/superpowers/plans/2026-04-22-merged-real77-server-plan.md`
- `docs/superpowers/plans/2026-04-22-run-flow-redesign-plan.md`
- `docs/superpowers/specs/2026-04-22-merged-real77-server-design.md`
- `scripts/dev-up.ps1`
- `start.bat`

直接暂存：

```powershell
git -C D:\中期\wd_bishe add --pathspec-from-file=docs/pathspecs/04-docs-and-tooling.txt
```

## 推荐提交顺序

1. 前端主流程重构
2. 后端运行链路与后处理确认
3. 模型实验与训练辅助扩展
4. 文档与开发辅助

## 不建议混提

下面两类不要混在同一个提交里：

- `frontend/ + backend/` 的 Web 平台流程改造
- `experiments/mbu_netpp/ + tests/` 的模型实验扩展

理由很简单：前者是产品运行态，后者是研究实验线，review 目标完全不同。
