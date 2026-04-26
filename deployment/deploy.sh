#!/bin/bash
# ============================================================
# 部署脚本 — SEM 分割统计平台
#
# 用法（服务器上直接运行）:
#   chmod +x deployment/deploy.sh
#   ./deployment/deploy.sh
#
# 或配合 systemd 服务使用（推荐）:
#   sudo cp deployment/sem-platform.service /etc/systemd/system/
#   sudo systemctl enable sem-platform
#   sudo systemctl start sem-platform
# ============================================================

set -euo pipefail

# ---------- 颜色输出 ----------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ---------- 加载环境变量 ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"

if [[ -f "$ENV_FILE" ]]; then
    info "加载环境变量: $ENV_FILE"
    set -a
    source "$ENV_FILE"
    set +a
else
    warn "未找到 .env 文件，请手动设置环境变量"
fi

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
STORAGE_DIR="${STORAGE_DIR:-/data/sem-platform/storage}"
MODEL_WEIGHTS_DIR="${MODEL_WEIGHTS_DIR:-/data/sem-platform/models}"

# ---------- 检查前置条件 ----------
info "检查前置条件..."

command -v docker >/dev/null 2>&1 || error "Docker 未安装。请先安装 Docker: https://docs.docker.com/engine/install/"
docker compose version >/dev/null 2>&1 || docker-compose version >/dev/null 2>&1 || \
    error "Docker Compose 未安装。请先安装 Docker Compose."

info "Docker 版本: $(docker version --format '{{.Server.Version}}' 2>/dev/null || echo 'unknown')"

# 检查 GPU
if command -v nvidia-smi >/dev/null 2>&1; then
    info "检测到 NVIDIA GPU:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 || true
    nvidia-smi --query-gpu=driver --format=csv,noheader 2>/dev/null || true
    info "nvidia-container-toolkit 检查: $(docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi 2>&1 | head -1 || echo '未配置')"
else
    warn "未检测到 NVIDIA GPU，将以 CPU 模式运行"
fi

# ---------- 创建必要目录 ----------
info "创建必要目录..."
for dir in "$STORAGE_DIR" "$MODEL_WEIGHTS_DIR"; do
    if [[ ! -d "$dir" ]]; then
        info "创建目录: $dir"
        sudo mkdir -p "$dir"
        sudo chown -R $(id -u):$(id -g) "$dir" 2>/dev/null || true
    fi
done

# ---------- 复制权重文件提示 ----------
if [[ -d "$MODEL_WEIGHTS_DIR" ]] && [[ -z "$(ls -A "$MODEL_WEIGHTS_DIR" 2>/dev/null)" ]]; then
    warn "模型权重目录为空: $MODEL_WEIGHTS_DIR"
    warn "请将 best.pt（MBU-Net++）和 sam_vit_h_4ec893100.pth（MatSAM）放入此目录"
fi

# ---------- 构建并启动 ----------
cd "$PROJECT_ROOT"
info "进入项目目录: $PROJECT_ROOT"

info "构建 Docker 镜像（首次约需 15-30 分钟）..."
docker compose -f "$SCRIPT_DIR/docker-compose.yml" build --no-cache

info "启动服务..."
docker compose -f "$SCRIPT_DIR/docker-compose.yml" up -d

# ---------- 健康检查 ----------
info "等待服务就绪（最多 60 秒）..."
sleep 5

BACKEND_URL="http://localhost:8000/api/health"
FRONTEND_URL="http://localhost"

for i in {1..12}; do
    if curl -sf "$BACKEND_URL" > /dev/null 2>&1; then
        info "后端就绪: $BACKEND_URL"
        break
    fi
    if [[ $i -eq 12 ]]; then
        error "后端健康检查超时，请检查日志: docker compose -f $SCRIPT_DIR/docker-compose.yml logs backend"
    fi
    sleep 5
done

info "前端就绪: $FRONTEND_URL"

# ---------- 完成 ----------
info ""
info "========================================"
info "  部署完成！"
info "========================================"
info "  前端地址: http://\$(hostname -I | awk '{print \$1}'):${FRONTEND_PORT:-80}"
info "  后端地址: http://localhost:8000/api/docs （Swagger 文档）"
info "  日志查看: docker compose -f $SCRIPT_DIR/docker-compose.yml logs -f"
info "  停止服务: docker compose -f $SCRIPT_DIR/docker-compose.yml down"
info "========================================"
