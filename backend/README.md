# Backend

后端是当前 Web 平台的执行与统计中枢，职责是：

- 暴露任务、结果、统计、后处理预览确认等 API
- 串起 `输入 -> 预处理 -> 分割 -> 统计 -> 导出` 主流水线
- 管理运行记录、图像资产、统计记录与导出文件
- 调用传统分割与深度学习推理，但**不承载训练代码**

## 开发启动

```bash
python -m uvicorn backend.main:app --reload
```

如果要与前端固定联调，推荐在仓库根目录直接运行 `start.bat`。

## 代码地图

### 入口

- `app/main.py`
  FastAPI 应用入口。
- `app/api/routes.py`
  所有前端可见 API 路由都从这里进入。

### 核心执行链路

- `app/services/pipeline.py`
  主执行流水线，负责分步推进、聚合统计、重建导出结果。
- `app/services/postprocess_preview.py`
  后处理预览与确认覆盖链路。
- `app/services/exporter.py`
  CSV / XLSX / JSON / ZIP 导出。

### 图像处理与统计

- `app/services/algorithms/traditional.py`
  传统分割与后处理算法。
- `app/services/preprocess.py`
  预处理。
- `app/services/sem_roi.py`
  分析区域与底栏分离。
- `app/services/sem_footer_ocr.py`
  底栏 OCR 与标定提示。
- `app/services/statistics.py`
  Vf、面积、尺寸、通道宽度等统计。

### 运行与存储

- `app/services/model_runner.py`
  深度学习推理调度入口。
- `app/services/task_manager.py`
  后台任务调度。
- `app/services/storage.py`
  运行目录、静态文件、导出文件路径管理。

### 数据边界

- `app/models/entities.py`
  ORM 数据模型。
- `app/schemas/run.py`
  运行相关请求/响应结构。
- `app/db/`
  数据库初始化与会话管理。

## 当前前后端分界

- 前端负责：路由、页面状态、交互、预览确认。
- 后端负责：真正的分割、统计、导出与确认覆盖。
- 模型训练代码全部放在 `experiments/mbu_netpp/`，不要继续塞进 `backend/app/`。
