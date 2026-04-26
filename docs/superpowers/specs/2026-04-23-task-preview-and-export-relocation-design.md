# 任务创建缩略图放大与结果文件迁移设计

日期：2026-04-23

## 背景

当前前端存在两个交互问题：

1. 任务创建页的逐图导入卡片已经展示了缩略图，但用户无法点击查看大图，不利于在批量导入后快速核对图像内容。
2. 结果页承担了“分割结果查看 + 后处理 + 导出文件”三类职责，`结果文件` 模块与页面主任务不一致，干扰结果核查与后处理操作。

## 目标

本次只解决两个明确问题，不额外扩 scope：

1. 在任务创建页复用已有大图预览弹窗，让每张导入缩略图都可点击放大。
2. 将 `结果文件` 模块完全从结果页移除，迁移到统计页，并把导出 UI 收敛到统计语境下。

## 非目标

- 不改后端导出接口和导出文件格式。
- 不新增新的图片查看组件，直接复用现有 `ZoomableImageDialog.vue`。
- 不重构统计页的数据模型，只调整导出模块挂载位置与展示方式。

## 设计决策

### 1. 任务创建页缩略图放大

位置：`frontend/src/views/TaskConfigView.vue`

方案：

- 复用当前页面已接入的 `ZoomableImageDialog.vue`。
- 将 `.selection-item__preview` 变成可点击预览入口。
- 当 `entry.previewUrl` 存在时：
  - 整个缩略图区显示点击态与 hover 态；
  - 点击后打开弹窗；
  - 弹窗标题使用文件名；
  - 弹窗副标题优先显示相对路径，否则显示“导入图像预览”。
- 当没有 `previewUrl` 时，继续显示占位块 `SEM`，但不触发弹窗。

收益：

- 批量导入后，用户可以直接在任务创建页逐图核对图像。
- 与预处理预览页的“点击放大”交互保持一致，避免新增认知成本。

### 2. 结果文件迁移到统计页

当前位置：

- 组件：`frontend/src/components/StatsSidebar.vue`
- 挂载点：`frontend/src/views/RunDetailView.vue`

目标位置：

- 挂载到 `frontend/src/views/RunStatisticsView.vue`
- 放在统计页头部控制区之后、汇总卡片之前。

理由：

- 结果页的主任务是“看分割效果 + 做后处理 + 确认当前结果”，不应该再混入导出操作。
- 统计页天然承接“看统计 + 导出统计结果”的闭环，导出放在这里更符合用户心智。

### 3. 导出 UI 布局

保留现有 `StatsSidebar.vue` 的导出逻辑，不改选择、全选、ZIP 打包行为，只改容器语义和视觉布局。

新布局原则：

- 不再做结果页侧栏卡片。
- 改成统计页中的横向内容卡片，命名为“统计导出”。
- 结构分两层：
  - 第一层：说明文案 + 选择状态 + 全选/取消全选 + 导出所选 ZIP
  - 第二层：导出文件列表
- 文件列表保持复选能力，但压缩视觉重量，避免抢占统计图表的注意力。

## 组件边界

### 保留

- `StatsSidebar.vue` 中的导出数据处理逻辑、选择逻辑、ZIP 打包逻辑。
- `RunStatisticsView.vue` 中现有统计数据拉取与图表逻辑。

### 调整

- `StatsSidebar.vue` 将从“结果页侧栏组件”转为“统计页导出组件”。
- `RunDetailView.vue` 删除对 `StatsSidebar` 的挂载。
- `RunStatisticsView.vue` 新增对 `StatsSidebar` 的挂载，并基于页面结构调整样式。

## 数据流

数据流保持不变：

1. `/runs/:id/results` 返回 `payload.exports`
2. 结果页与统计页本来都基于同一个 results payload
3. 只是将 `payload.exports` 的消费位置从结果页移动到统计页

因此这次改动是前端视图重排，不涉及接口调整。

## 错误处理

- 若统计页 `payload.exports` 为空，继续显示空状态提示。
- 若单文件下载失败或 ZIP 生成失败，继续使用现有 `ElMessage.error`。
- 若缩略图没有可用 `previewUrl`，不启用点击放大，避免空弹窗。

## 验证点

1. 任务创建页点击任意导入缩略图，可打开大图弹窗。
2. 无预览图的占位块不会误触发弹窗。
3. 结果页不再出现 `结果文件` 模块。
4. 统计页出现 `统计导出` 模块，并能：
   - 勾选文件；
   - 全选/取消全选；
   - 导出所选 ZIP。
5. 统计页整体阅读顺序仍然清晰：控制区 -> 导出区 -> 汇总卡片 -> 图表 -> 明细 -> 说明。

## 实现范围

预计只涉及以下前端文件：

- `frontend/src/views/TaskConfigView.vue`
- `frontend/src/views/RunDetailView.vue`
- `frontend/src/views/RunStatisticsView.vue`
- `frontend/src/components/StatsSidebar.vue`

不改后端。
