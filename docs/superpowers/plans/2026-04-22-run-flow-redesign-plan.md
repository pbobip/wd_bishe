# Run Flow Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把当前“任务创建 → 后处理 → 结果展示 → 历史记录”的流程，重构为云东确认的新四页流程：`01 任务创建`、`02 结果展示与后处理`、`03 统计分析`、`04 历史记录`，并保证后处理采用“先预览、再确认覆盖”的交互。

**Architecture:** 采用 KISS 方案，不引入完整结果版本表。主分割完成后，当前持久化的 `MetricRecord + artifacts + run.summary/chart_data/exports` 即为当前 `confirmed_result`；后处理预览只生成临时 `preview_result` 文件和对比响应，用户确认后再覆盖当前已确认结果并重建统计/导出。这样既满足“先看结果再决定后处理”，又避免多版本数据库设计。

**Tech Stack:** Vue 3, TypeScript, Vue Router, Element Plus, FastAPI, SQLAlchemy, OpenCV

---

### Task 1: 重排全局导航与路由语义

**Files:**
- Modify: `frontend/src/App.vue`
- Modify: `frontend/src/router.ts`

- [ ] 把顶部四个入口统一为 `01 任务创建`、`02 结果展示与后处理`、`03 统计分析`、`04 历史记录`，移除当前“后处理 / 结果展示”旧语义。
- [ ] 调整 `route.meta.sectionKey/sectionTitle/sectionSubtitle`，让当前页标题和导航高亮与新职责一致。
- [ ] 取消 `App.vue` 中“点击 02 自动帮用户准备草稿”的旧 guard，改为只在有活动任务或显式 `runId` 时允许进入结果页；否则提示“请先在任务创建页开始处理”。
- [ ] 为历史兼容保留旧路径跳转：
  - `/results` 重定向到 `/history`
  - `/runs/:id` 作为指定任务的结果展示与后处理页入口
  - `/runs/:id/statistics` 继续作为统计页，但归属新的 `03`

### Task 2: 拆分工作流状态机，让 01 只负责建任务并启动主分割

**Files:**
- Modify: `frontend/src/composables/useTaskWorkflow.ts`
- Modify: `frontend/src/types.ts`

- [ ] 把当前 `prepareCurrentTask / ensurePostprocessDraftReady / launchCurrentTask` 的语义改成“准备草稿”和“启动主分割”两段式，但都归 `01` 页面驱动。
- [ ] 从 `buildRunPayload()` 中移除“把 `form.postprocess` 回写进 `traditional_seg`”的旧行为，保证主分割阶段不再提前混入后处理。
- [ ] 补充当前活动任务上下文，至少明确：
  - 当前 `runId`
  - 当前结果是否可展示
  - 当前统计是否可进入
  - 当前是否有后处理预览进行中
- [ ] 保留并兼容现有逐图标定、批量上传、恢复草稿、轮询进度逻辑，避免打坏已完成的逐图标定改造。

### Task 3: 把 01 页重构为“任务创建 + 主分割启动 + 紧凑进度”

**Files:**
- Modify: `frontend/src/views/TaskConfigView.vue`

- [ ] 保留现有图像导入、逐图标定、预处理配置、分割模式配置。
- [ ] 把页面主操作从“下一步：后处理”改成“开始处理”或等价语义，由这里直接创建草稿、上传图像并触发主分割。
- [ ] 在 `01` 页保留紧凑运行进度区，只展示：
  - 上传/登记进度
  - 当前执行阶段
  - 当前任务状态
- [ ] 主分割启动后，允许跳去 `02` 查看运行中的结果页；如果主分割尚未完成，`02` 先显示运行状态和占位结果，不显示可操作的后处理确认区。
- [ ] 清理旧文案，确保页面不再暗示“先进入后处理页再启动执行”。

### Task 4: 用现有结果工作台能力重建 02 页

**Files:**
- Modify: `frontend/src/views/TaskRunView.vue`
- Modify: `frontend/src/views/RunDetailView.vue`
- Modify: `frontend/src/components/PreviewWorkspace.vue`
- Modify: `frontend/src/components/StatsSidebar.vue`
- Create: `frontend/src/components/PostprocessPreviewDialog.vue`

- [ ] 以当前 `TaskRunView` 为壳，但改造成“结果展示与后处理”页面，而不是“运行前配置面板”。
- [ ] 复用现有 `RunDetailView` 的结果展示组件能力，把原图、掩码、叠加图、步骤轨迹和导出信息聚合到 `02` 页，不再让结果展示分散在另一套页面中。
- [ ] 把当前 `RunDetailView` 缩成兼容入口或薄包装，避免维护两套结果展示 UI。
- [ ] 在 `02` 页增加后处理面板，只保留必要的传统后处理项：
  - 填孔
  - watershed
  - 平滑
  - 形状过滤
  - 触边剔除
- [ ] `02` 页点击“应用后处理”时，先请求后端生成预览，再弹出统一对比弹窗：
  - 展示当前选中图像的前后对比
  - 仅保留“取消 / 确认应用”
  - 不直接覆盖主界面结果
- [ ] 用户确认后，刷新 `02` 页当前结果、步骤说明和导出摘要；用户取消后，主界面保持不变。
- [ ] 删掉当前“统计与测量项”整块、`点击放大查看` 提示等旧布局遗留，让 02 页 UI 聚焦在结果与后处理本身。

### Task 5: 后端新增“后处理预览 / 确认覆盖”链路

**Files:**
- Modify: `backend/app/api/routes.py`
- Modify: `backend/app/schemas/run.py`
- Modify: `backend/app/services/pipeline.py`
- Modify: `backend/app/services/exporter.py`
- Modify: `backend/app/services/algorithms/traditional.py`
- Create: `backend/app/services/postprocess_preview.py`

- [ ] 把传统后处理从“主分割内部顺手做掉”中拆出来，至少暴露一条可复用的“对现有 mask 应用后处理”的服务入口。
- [ ] 新增预览接口，例如：
  - `POST /runs/{run_id}/postprocess/preview`
  - 输入：当前模式、后处理参数、当前选中图像
  - 输出：预览 token、前后对比图 URL、适配弹窗的文案与预览状态
- [ ] 新增确认接口，例如：
  - `POST /runs/{run_id}/postprocess/confirm`
  - 输入：预览 token
  - 行为：把预览结果提升为当前已确认结果，覆盖当前 mode 的 mask / overlay / objects / summary / charts / exports
- [ ] 不做完整多版本数据库设计；预览文件放到 `runs/{id}/tmp/postprocess_preview/...` 之类的临时目录，确认后覆盖正式输出目录。
- [ ] 确认后重建当前运行的：
  - `MetricRecord.summary/artifacts`
  - `run.summary`
  - `run.chart_data`
  - `ExportRecord` 与导出压缩包
- [ ] 让 `/runs/{id}/results` 始终返回“当前确认结果”，从而保证前端只需重新拉 payload，不需要自己拼统计值。

### Task 6: 让 03 页只读当前确认结果，并明确不可提前进入

**Files:**
- Modify: `frontend/src/views/RunStatisticsView.vue`
- Modify: `frontend/src/utils/runStatistics.ts`
- Modify: `frontend/src/App.vue`

- [ ] 统一 03 页标题、导航归属和返回入口，去掉当前残留的“结果展示”归类。
- [ ] 进入统计页时，如果任务尚未产出可确认结果，则展示空态提示并阻止用户误以为统计已就绪。
- [ ] 保持现有按 `confirmed_result` 拉取 `/runs/{id}/results` 的策略，不在前端本地改统计值。
- [ ] 后处理确认成功后，`03` 页再次进入或当前页刷新时，应直接反映新的统计卡片、图表和表格。
- [ ] 保留现有“逐图已标定 / 部分按像素统计”的退化口径，不因为流程重构打坏之前的混合标定统计逻辑。

### Task 7: 历史页与深链回看链路统一到新流程

**Files:**
- Modify: `frontend/src/views/HistoryView.vue`
- Modify: `frontend/src/router.ts`

- [ ] 历史页“查看”按钮默认进入新的 `02` 结果展示与后处理页，而不是旧的独立结果详情页。
- [ ] 对历史任务进入 `02` 时，允许只读查看；只有当前模式支持且结果已就绪时才开放后处理预览与确认。
- [ ] 历史任务进入 `03` 时，直接展示该任务当前确认结果的统计，不再依赖旧的 `results` 列表页中转。

### Task 8: 验证与回归

**Files:**
- Modify: `backend/tests/test_init_db_legacy.py`
- Modify: `backend/tests/test_runs_without_project.py`
- Create: `backend/tests/test_postprocess_preview_confirm.py`

- [ ] 为新的后处理预览 / 确认接口补后端测试，至少覆盖：
  - 预览成功
  - 取消不落库
  - 确认后结果与统计刷新
- [ ] 如果需要补数据库兼容逻辑，更新 `init_db` 相关测试，确保旧库升级后仍能创建/运行任务。
- [ ] 运行后端测试：

Run: `python -m pytest backend/tests -q`
Expected: PASS

- [ ] 运行前端构建验证：

Run: `npm run build`
Workdir: `frontend`
Expected: PASS

- [ ] 做一轮人工联调，至少覆盖以下链路：
  - `01` 导入图像、逐图标定、开始主分割
  - `02` 运行中查看结果占位，完成后查看主分割结果
  - `02` 应用后处理，弹出预览对比，取消不覆盖
  - `02` 再次应用并确认，主界面结果刷新
  - `03` 查看与最新确认结果一致的统计
  - `04` 从历史页回看同一任务仍能进入新的 02/03
