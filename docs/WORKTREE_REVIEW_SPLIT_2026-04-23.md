# 工作区审阅分流

目的：把当前脏工作区拆成“这次流程重构必须先看”的核心集合，以及“可以暂时忽略、单独 review”的非核心集合。

## A. 本次流程重构必看

这部分直接决定 `01 任务创建 -> 02 结果展示与后处理 -> 03 统计分析 -> 04 历史记录` 是否成立。

### 前端核心

- `frontend/src/App.vue`
- `frontend/src/router.ts`
- `frontend/src/composables/useTaskWorkflow.ts`
- `frontend/src/views/TaskConfigView.vue`
- `frontend/src/views/TaskRunView.vue`
- `frontend/src/views/RunDetailView.vue`
- `frontend/src/views/RunStatisticsView.vue`
- `frontend/src/views/HistoryView.vue`
- `frontend/src/components/PreviewWorkspace.vue`
- `frontend/src/components/PostprocessPreviewDialog.vue`
- `frontend/src/components/StatsSidebar.vue`
- `frontend/src/components/CalibrationStatusBanner.vue`
- `frontend/src/components/ZoomableImageDialog.vue`
- `frontend/src/types.ts`
- `frontend/src/utils/calibration.ts`
- `frontend/src/utils/runStatistics.ts`

### 后端核心

- `backend/app/api/routes.py`
- `backend/app/schemas/run.py`
- `backend/app/services/pipeline.py`
- `backend/app/services/postprocess_preview.py`
- `backend/app/services/exporter.py`
- `backend/tests/test_postprocess_preview_confirm.py`

### 审阅顺序

1. `frontend/src/router.ts`
2. `frontend/src/composables/useTaskWorkflow.ts`
3. `frontend/src/views/TaskConfigView.vue`
4. `frontend/src/views/RunDetailView.vue`
5. `frontend/src/views/RunStatisticsView.vue`
6. `backend/app/api/routes.py`
7. `backend/app/services/postprocess_preview.py`
8. `backend/app/services/pipeline.py`
9. `backend/app/schemas/run.py`
10. `backend/tests/test_postprocess_preview_confirm.py`

## B. 与流程重构强相关，但不是第一批阻塞点

这部分主要是支撑 UI 落地、状态展示和兼容老入口，不是第一眼必须读，但第二轮应该补看。

- `frontend/src/styles.css`
- `frontend/src/views/ResultsView.vue`
- `frontend/src/views/DashboardView.vue`
- `frontend/components.d.ts`
- `frontend/package.json`
- `frontend/package-lock.json`
- `backend/app/db/init_db.py`
- `backend/app/models/entities.py`
- `backend/app/services/algorithms/traditional.py`
- `backend/app/services/sem_roi.py`
- `backend/app/schemas/project.py`

## C. 模型实验线，可暂时忽略

如果你当前的目标是审 Web 平台流程，这一组可以先不看。它们是训练、数据准备、半监督和服务器脚本线，不影响当前前后端主流程是否跑通。

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
- `experiments/mbu_netpp/configs/default_semi_supervised_iterative.yaml`
- `experiments/mbu_netpp/configs/merged_real77_supervised.yaml`
- `experiments/mbu_netpp/configs/merged_real77_supervised_server_full.yaml`
- `experiments/mbu_netpp/configs/nasa_super_secondary_supervised.yaml`
- `experiments/mbu_netpp/scripts/server/download_results.ps1`
- `experiments/mbu_netpp/scripts/server/install_env.sh`
- `experiments/mbu_netpp/scripts/server/run_merged_real77_pipeline.sh`
- `experiments/mbu_netpp/scripts/server/upload_bundle.ps1`
- `tests/test_mbu_netpp_prepare_merged_supervised.py`
- `tests/test_mbu_netpp_prepare_nasa_super.py`
- `tests/test_mbu_netpp_semi_supervised.py`

## D. 文档与工程辅助，可按需阅读

这部分的作用是把工作区变得可读、可分组、可追踪，不会改变运行逻辑。

- `.gitignore`
- `README.md`
- `frontend/README.md`
- `backend/README.md`
- `experiments/mbu_netpp/README.md`
- `experiments/mbu_netpp/CODEMAP.md`
- `docs/CODEBASE_STREAMS_2026-04-23.md`
- `docs/COMMIT_GROUPS_2026-04-23.md`
- `docs/superpowers/plans/2026-04-20-shape-based-material-classification.md`
- `docs/superpowers/plans/2026-04-22-merged-real77-server-plan.md`
- `docs/superpowers/plans/2026-04-22-run-flow-redesign-plan.md`
- `docs/superpowers/specs/2026-04-22-merged-real77-server-design.md`
- `scripts/dev-up.ps1`
- `start.bat`

## E. 最省时间的 review 路径

如果你只想先搞清楚“这次到底改了什么”，按下面顺序最快：

1. 先看 [docs/CODEBASE_STREAMS_2026-04-23.md](docs/CODEBASE_STREAMS_2026-04-23.md)
2. 再看 [docs/COMMIT_GROUPS_2026-04-23.md](docs/COMMIT_GROUPS_2026-04-23.md)
3. 只读上面 A 组核心文件
4. 确认主流程没问题后，再决定要不要读模型实验线
