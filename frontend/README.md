# Frontend

前端是当前 Web 平台的交互壳层，职责是：

- 承载全局导航与页面切换
- 维护任务创建草稿与当前运行上下文
- 展示结果、后处理预览、统计图表和导出入口
- 调用后端 API，但**不自己计算分割与统计**

## 开发启动

```bash
cd frontend
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

如果要和后端一起联调，推荐直接在仓库根目录运行 `start.bat`。

## 代码地图

### 入口与壳层

- `src/main.ts`
  Vue 挂载入口。
- `src/App.vue`
  全局页面壳层与顶部主流程导航。
- `src/router.ts`
  4 个主页面入口和运行态路由。

### 工作流状态

- `src/composables/useTaskWorkflow.ts`
  任务创建草稿、当前 run、启动处理、页面间共享状态。

### 主页面

- `src/views/TaskConfigView.vue`
  `01 任务创建`，负责导入、逐图标定、预处理配置、分割模式选择、启动主分割。
- `src/views/RunDetailView.vue`
  `02 结果与后处理`，负责结果核查、图层切换、后处理预览与确认。
- `src/views/RunStatisticsView.vue`
  `03 统计分析`，负责当前确认结果的只读统计。
- `src/views/HistoryView.vue`
  `04 历史记录`。
- `src/views/TaskRunView.vue`
  兼容重定向页，旧 `/postprocess` 入口现在导向 run 结果页。

### 核心组件

- `src/components/PreviewWorkspace.vue`
  图像核查工作区、图层与模式切换。
- `src/components/PostprocessPreviewDialog.vue`
  后处理前后对比确认弹窗。
- `src/components/StatsSidebar.vue`
  导出文件与统计摘要侧栏。
- `src/components/CalibrationStatusBanner.vue`
  标定状态提示横幅。
- `src/components/ZoomableImageDialog.vue`
  放大查看图像。
- `src/components/ChartCanvas.vue`
  图表渲染。

### 工具与类型

- `src/types.ts`
  API 结构定义。
- `src/api.ts`
  请求封装。
- `src/utils/calibration.ts`
  标定辅助格式化。
- `src/utils/runStatistics.ts`
  统计页展示逻辑。

## 当前页面职责边界

- `01` 只做任务准备和主分割启动。
- `02` 只做结果核查与后处理确认。
- `03` 只读当前确认结果，不承载配置。
- `04` 只做历史记录与回看入口。

如果一个需求同时涉及任务配置、结果核查和统计刷新，优先沿着这条链路判断归属，而不是把逻辑堆进单个大页面。
