# ============================================================
# 后端 Dockerfile — GPU (CUDA) / CPU 双模式
#
# ▶ GPU 服务器（推荐）:
#   docker build \
#     --build-arg BASE_IMAGE=nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 \
#     -t sem-backend:gpu \
#     -f deployment/backend/Dockerfile .
#
# ▶ CPU-only 服务器:
#   docker build \
#     --build-arg BASE_IMAGE=ubuntu:22.04 \
#     --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu \
#     -t sem-backend:cpu \
#     -f deployment/backend/Dockerfile .
#
# ▶ 运行:
#   GPU: docker run -d --gpus all -p 8000:8000 \
#          -v /data/sem-platform/models:/models:ro \
#          -v /data/sem-platform/storage:/app/storage \
#          sem-backend:gpu
#
#   CPU: docker run -d -p 8000:8000 \
#          -v /data/sem-platform/models:/models:ro \
#          -v /data/sem-platform/storage:/app/storage \
#          sem-backend:cpu
# ============================================================

# 默认使用 CUDA 12.1 基础镜像（runtime = 无 NVCC 编译器，仅运行时库）
ARG BASE_IMAGE=nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
# PyTorch 下载源：GPU 用 cu121，CPU 用 cpu
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121

# ---------- Stage 1: Base ----------
FROM ${BASE_IMAGE} AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# 安装 Python、OpenCV 依赖和必要工具
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3-pip \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
        libgomp1 libhdf5-103-1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/bin/pip3 /usr/local/bin/pip

WORKDIR /app

# ---------- Stage 2: Python deps ----------
# 先装轻量依赖，不含 torch（torch 在下一个 stage 装以优化镜像层）
COPY deployment/backend/requirements.txt /tmp/requirements.txt

# 过滤掉 torch/segment-anything 相关的行（单独处理）
RUN grep -v -E "^#|^\s*$|^torch|^segment-anything|^--|pip install torch" \
    /tmp/requirements.txt > /tmp/req_filtered.txt \
    && pip install --no-cache-dir --break-system-packages -r /tmp/req_filtered.txt

# 安装 PyTorch（GPU: cu121, CPU: cpu）
# nvidia/cuda 镜像中已有 CUDA 库，所以 --extra-index-url 找到 cu121 wheel
# CPU 镜像无 CUDA，用 cpu index
RUN pip install --no-cache-dir --break-system-packages \
        "torch>=2.2.0" "torchvision>=0.17.0" \
        --index-url "${TORCH_INDEX_URL}" \
    || pip install --no-cache-dir --break-system-packages \
        "torch>=2.2.0" "torchvision>=0.17.0"

# ---------- Stage 3: Copy code ----------
COPY backend/ /app/backend/
COPY experiments/ /app/experiments/

# 创建存储目录
RUN mkdir -p /app/storage/runs /app/storage/exports \
             /app/storage/tmp /app/storage/logs

# 创建非 root 用户
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app/storage /app/backend /app/experiments
USER appuser

ENV PYTHONPATH=/app:/app/backend:/app/experiments:$PYTHONPATH
ENV STATIC_URL_PREFIX=/static
ENV API_PREFIX=/api

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

# 启动 uvicorn
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
