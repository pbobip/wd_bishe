# 代码流分层与本次改动索引

目的：把当前杂乱工作区里的改动按**前端主流程重构**、**后端运行链路**、**模型实验线**分开看，避免 review 时互相污染。

## 1. 本次主流程重构优先看这里

### 前端

- `frontend/src/App.vue`
  顶部 4 页主导航。
- `frontend/src/router.ts`
  01/02/03/04 路由映射。
- `frontend/src/composables/useTaskWorkflow.ts`
  当前任务与 run 状态主线。
- `frontend/src/views/TaskConfigView.vue`
  `01 任务创建`
- `frontend/src/views/RunDetailView.vue`
  `02 结果与后处理`
- `frontend/src/views/RunStatisticsView.vue`
  `03 统计分析`
- `frontend/src/views/HistoryView.vue`
  `04 历史记录`
- `frontend/src/views/TaskRunView.vue`
  旧入口兼容页
- `frontend/src/components/PreviewWorkspace.vue`
- `frontend/src/components/PostprocessPreviewDialog.vue`
- `frontend/src/components/StatsSidebar.vue`
- `frontend/src/types.ts`

### 后端

- `backend/app/api/routes.py`
  新增后处理预览/确认接口。
- `backend/app/schemas/run.py`
  运行态请求/响应结构。
- `backend/app/services/pipeline.py`
  确认结果后的汇总与导出重建。
- `backend/app/services/postprocess_preview.py`
  后处理预览、确认覆盖、标定回写。
- `backend/app/services/exporter.py`
  导出结果重建。

### 这次新增的回归测试

- `backend/tests/test_postprocess_preview_confirm.py`

## 2. 模型实验线单独看这里

这些文件属于模型与实验，不属于 Web 流程重构：

- `experiments/mbu_netpp/train.py`
- `experiments/mbu_netpp/iterative_semi_supervised.py`
- `experiments/mbu_netpp/dataset.py`
- `experiments/mbu_netpp/losses.py`
- `experiments/mbu_netpp/preparation.py`
- `experiments/mbu_netpp/prepare_merged_supervised.py`
- `experiments/mbu_netpp/prepare_nasa_super.py`
- `experiments/mbu_netpp/semi_scoring.py`
- `experiments/mbu_netpp/semi_utils.py`
- `experiments/mbu_netpp/configs/*`
- `experiments/mbu_netpp/scripts/server/*`

模型区入口和边界说明见：

- [experiments/mbu_netpp/CODEMAP.md](/D:/中期/wd_bishe/experiments/mbu_netpp/CODEMAP.md)

## 3. 当前工作区里的噪音来源

下面这些大多是生成物或历史改动，不要和本次主线混在一起 review：

- `experiments/mbu_netpp/outputs/`
- `experiments/mbu_netpp/workdir/`
- `dataset/full_png_cropped_xlsx/images/processed/`
- `results/merged_real77_*`
- `output/`
- 各类 `__pycache__/`

## 4. 建议 review 顺序

如果你只想 review 这次流程重构，按这个顺序看：

1. `frontend/src/router.ts`
2. `frontend/src/App.vue`
3. `frontend/src/composables/useTaskWorkflow.ts`
4. `frontend/src/views/TaskConfigView.vue`
5. `frontend/src/views/RunDetailView.vue`
6. `frontend/src/views/RunStatisticsView.vue`
7. `backend/app/api/routes.py`
8. `backend/app/services/postprocess_preview.py`
9. `backend/app/services/pipeline.py`
10. `backend/tests/test_postprocess_preview_confirm.py`

## 5. 这次整理后的代码导航

- 前端导航说明：[frontend/README.md](/D:/中期/wd_bishe/frontend/README.md)
- 后端导航说明：[backend/README.md](/D:/中期/wd_bishe/backend/README.md)
- 模型区代码地图：[experiments/mbu_netpp/CODEMAP.md](/D:/中期/wd_bishe/experiments/mbu_netpp/CODEMAP.md)
