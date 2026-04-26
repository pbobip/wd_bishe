# SEM 分割统计平台 — 生产部署指南

> 目标环境：云服务器（有 GPU）+ 域名 + HTTPS

---

## 目录

1. [部署架构](#1-部署架构)
2. [服务器前置准备](#2-服务器前置准备)
3. [上传代码到服务器](#3-上传代码到服务器)
4. [配置模型权重](#4-配置模型权重)
5. [配置 .env 环境变量](#5-配置-env-环境变量)
6. [一键部署](#6-一键部署)
7. [域名 + HTTPS 配置](#7-域名--https-配置)
8. [开机自启配置](#8-开机自启配置)
9. [日常运维](#9-日常运维)
10. [故障排查](#10-故障排查)

---

## 1. 部署架构

```
                        ┌──────────────────────────────────┐
                        │          云服务器 (Ubuntu 22.04)  │
                        │                                    │
  用户浏览器 ──────────►│  Nginx (80/443)                  │
                        │    └─► /api/      ──► 后端容器    │
                        │    └─► /static/   ──► 后端容器    │
                        │    └─► /          ──► 前端容器    │
                        │                        (静态)     │
                        │  ┌──────────────────────────┐     │
                        │  │  后端容器 (FastAPI)     │     │
                        │  │  - GPU: CUDA 12.1      │     │
                        │  │  - 线程池: 2 workers   │     │
                        │  │  - port: 8000 (内部)   │     │
                        │  └──────────┬───────────┘     │
                        │             │                   │
                        │  ┌──────────▼───────────┐     │
                        │  │  前端容器 (Nginx)     │     │
                        │  │  - port: 80 (内部)    │     │
                        │  └──────────────────────┘     │
                        │                                  │
                        │  /data/sem-platform/             │
                        │    ├─ storage/  (用户数据)       │
                        │    └─ models/   (权重文件)       │
                        └──────────────────────────────────┘
                                     │
                        ┌────────────┴────────────┐
                        │      域名解析            │
                        │  sem.yourdomain.com      │
                        │            │            │
                        │    Certbot → HTTPS       │
                        └─────────────────────────┘
```

### 端口说明

| 端口 | 用途 | 外部访问 |
|------|------|---------|
| 80 | HTTP（自动跳转 HTTPS） | 需开放 |
| 443 | HTTPS（正式访问） | 需开放 |
| 8000 | 后端 API（仅本地） | **不开放**，由 Nginx 代理 |

---

## 2. 服务器前置准备

### 2.1 系统要求

| 项目 | 要求 |
|------|------|
| OS | Ubuntu 22.04 LTS（其他发行版请自行调整） |
| CPU | 4 核以上 |
| 内存 | 16 GB 以上（深度学习推理建议 32 GB） |
| GPU | NVIDIA GPU，驱动 >= 525.60 |
| 磁盘 | 100 GB 以上（存储图像数据） |

### 2.2 安装 Docker

```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装必要组件
sudo apt install -y curl ca-certificates gnupg lsb-release

# 添加 Docker GPG key
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
    sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# 添加 Docker 仓库
echo "deb [arch=$(dpkg --print-architecture) \
    signed-by=/etc/apt/keyrings/docker.gpg] \
    https://download.docker.com/linux/ubuntu \
    $(lsb_release -cs) stable" | \
    sudo tee /etc/apt/sources.list.d/docker.list

# 安装 Docker
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# 当前用户加入 docker 组（免 sudo）
sudo usermod -aG docker $USER
newgrp docker
```

### 2.3 安装 NVIDIA Container Toolkit（GPU 支持）

```bash
# 添加 NVIDIA 仓库
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 安装
sudo apt update
sudo apt install -y nvidia-container-toolkit

# 配置 Docker 使用 NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 验证
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### 2.4 安装 Git（如果未安装）

```bash
sudo apt install -y git
```

---

## 3. 上传代码到服务器

### 方式一：Git 克隆（推荐）

```bash
# 在服务器上克隆仓库（假设你已将代码推送到 GitHub/Gitea）
git clone https://github.com/yourusername/wd_bishe.git /root/wd_bishe
cd /root/wd_bishe
```

### 方式二：SCP 上传

```bash
# 在本地终端执行（Windows 用 PowerShell）
scp -r ./wd_bishe root@你的服务器IP:/root/
```

---

## 4. 配置模型权重

MBU-Net++ 推理需要权重文件，请将以下文件放入 `/data/sem-platform/models/` 目录：

```bash
# 创建目录
sudo mkdir -p /data/sem-platform/models
sudo chown -R $(id -u):$(id -g) /data/sem-platform

# 复制权重文件（请将本地路径替换为实际路径）
# MBU-Net++ 最优权重
cp /path/to/your/best.pt /data/sem-platform/models/

# MatSAM 权重（如果需要 MatSAM 分割）
# 从 https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4ec893100.pth 下载
cp sam_vit_h_4ec893100.pth /data/sem-platform/models/

# 验证
ls -lh /data/sem-platform/models/
```

**注意**：MBU-Net++ 权重的 `weight_path` 需要与数据库中 `ModelRunner` 表里注册的路径一致。建议在部署完成后，通过系统前端"模型管理"页面确认路径，或直接修改 `backend/app/db/init_db.py` 中的 `DEFAULT_RUNNERS` 配置。

---

## 5. 配置 .env 环境变量

```bash
cd /root/wd_bishe
cp deployment/.env.example deployment/.env
nano deployment/.env
```

关键配置项说明：

| 变量 | 说明 | 示例值 |
|------|------|--------|
| `PROJECT_ROOT` | 项目在服务器上的绝对路径 | `/root/wd_bishe` |
| `STORAGE_DIR` | 持久化存储目录 | `/data/sem-platform/storage` |
| `MODEL_WEIGHTS_DIR` | 权重文件目录（只读） | `/data/sem-platform/models` |
| `BASE_IMAGE` | Docker 基础镜像（GPU 用 CUDA 镜像） | `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04` |
| `GPU_COUNT` | 使用几块 GPU | `1` |
| `FRONTEND_PORT` | 对外 HTTP 端口 | `80` |

**如果使用 CPU-only 服务器**，修改：

```bash
BASE_IMAGE=ubuntu:22.04
CUDA_ENABLED=false
```

---

## 6. 一键部署

```bash
cd /root/wd_bishe

# 赋予执行权限
chmod +x deployment/deploy.sh

# 运行部署脚本（约需 15-30 分钟，首次构建）
./deployment/deploy.sh
```

部署脚本会自动：
1. 检查 Docker 和 GPU 环境
2. 构建前后端 Docker 镜像
3. 启动容器
4. 执行健康检查

### 手动分步部署（如果脚本失败）

```bash
cd /root/wd_bishe

# 构建
docker compose -f deployment/docker-compose.yml build --no-cache

# 启动
docker compose -f deployment/docker-compose.yml up -d

# 查看日志
docker compose -f deployment/docker-compose.yml logs -f

# 查看状态
docker compose -f deployment/docker-compose.yml ps
```

### 验证服务

```bash
# 后端健康检查
curl http://localhost:8000/api/health

# 前端（服务器 IP 替换实际值）
curl http://你的服务器IP/
```

---

## 7. 域名 + HTTPS 配置

### 7.1 域名解析

在域名服务商（阿里云 / 腾讯云 / Cloudflare 等）的 DNS 管理页面，添加 A 记录：

| 主机记录 | 记录类型 | 记录值 |
|---------|---------|--------|
| `sem` | A | `你的服务器公网IP` |

等待 1-5 分钟生效后验证：

```bash
nslookup sem.yourdomain.com
```

### 7.2 安装 Certbot 并申请 SSL 证书

```bash
# 安装 Certbot
sudo apt install -y certbot python3-certbot-nginx

# 申请证书（将域名替换为实际值）
sudo certbot --nginx -d sem.yourdomain.com
```

> **提示**：Certbot 会自动修改 Nginx 配置并添加 HTTPS 支持。如果使用 Docker 方式，需要先停止容器：
> ```bash
> docker compose -f deployment/docker-compose.yml stop frontend
> sudo certbot --nginx -d sem.yourdomain.com
> # 之后重新启动
> docker compose -f deployment/docker-compose.yml start frontend
> ```

### 7.3 证书自动续期

Certbot 自动安装 cron 任务，每年到期前自动续期。可手动测试：

```bash
sudo certbot renew --dry-run
```

---

## 8. 开机自启配置

### 8.1 安装 systemd 服务

```bash
cd /root/wd_bishe

# 复制服务文件
sudo cp deployment/sem-platform.service /etc/systemd/system/

# 重新加载 systemd
sudo systemctl daemon-reload

# 启用开机自启
sudo systemctl enable sem-platform

# 启动服务
sudo systemctl start sem-platform

# 查看状态
sudo systemctl status sem-platform
```

### 8.2 管理命令

```bash
# 启动
sudo systemctl start sem-platform

# 停止
sudo systemctl stop sem-platform

# 重启
sudo systemctl restart sem-platform

# 查看日志
sudo journalctl -u sem-platform -f
```

---

## 9. 日常运维

### 9.1 更新部署

```bash
cd /root/wd_bishe

# 拉取最新代码
git pull

# 重新构建并启动（不清理卷）
docker compose -f deployment/docker-compose.yml up -d --build

# 查看更新后容器状态
docker compose -f deployment/docker-compose.yml ps
```

### 9.2 查看日志

```bash
# 实时查看所有日志
docker compose -f deployment/docker-compose.yml logs -f

# 只看后端日志
docker compose -f deployment/docker-compose.yml logs -f backend

# 只看前端日志
docker compose -f deployment/docker-compose.yml logs -f frontend

# 查看最近 100 行
docker compose -f deployment/docker-compose.yml logs --tail=100
```

### 9.3 数据备份

```bash
# 备份存储目录（图像和结果）
BACKUP_DATE=$(date +%Y%m%d)
tar -czf /root/sem-platform-backup-$BACKUP_DATE.tar.gz \
    /data/sem-platform/storage \
    /data/sem-platform/models

# 上传到云存储（如 OSS、S3）
# aws s3 cp /root/sem-platform-backup-$BACKUP_DATE.tar.gz s3://your-bucket/
```

### 9.4 清理磁盘空间

```bash
# 清理未使用的 Docker 资源
docker system prune -f

# 清理旧的 Docker 构建缓存
docker builder prune -f
```

---

## 10. 故障排查

### 后端容器启动失败

```bash
# 查看详细日志
docker compose -f deployment/docker-compose.yml logs backend

# 常见问题：
# - "ModuleNotFoundError: No module named 'torch'" → GPU 镜像构建失败，用 CPU 镜像
# - "CUDA out of memory" → 图像太大，增加 GPU 显存或改用 CPU
# - "Permission denied" → 检查存储目录权限
```

### 前端无法访问后端 API

```bash
# 确认后端正在运行
docker compose -f deployment/docker-compose.yml ps

# 测试后端健康端点
curl http://localhost:8000/api/health

# 检查 Nginx 代理配置
docker compose -f deployment/docker-compose.yml logs frontend

# 进入 Nginx 容器检查配置
docker exec -it sem-frontend cat /etc/nginx/conf.d/default.conf
```

### GPU 未被识别

```bash
# 在服务器上验证
nvidia-smi

# 在 Docker 容器内验证
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# 如果容器内无法识别 GPU，检查 nvidia-container-toolkit
sudo systemctl status nvidia-docker
```

### SSL 证书申请失败

```bash
# 确认域名已解析
nslookup sem.yourdomain.com

# 确认 80 和 443 端口已开放
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
# 或在云服务商安全组中开放

# 测试 Let's Encrypt 连接
curl -I http://sem.yourdomain.com/.well-known/acme-challenge/test
```

---

## 快速命令速查

```bash
# 启动
sudo systemctl start sem-platform

# 停止
sudo systemctl stop sem-platform

# 重启
sudo systemctl restart sem-platform

# 查看状态
sudo systemctl status sem-platform

# 实时日志
sudo journalctl -u sem-platform -f

# 查看容器日志
docker compose -f deployment/docker-compose.yml logs -f

# 完全重建
docker compose -f deployment/docker-compose.yml down && \
docker compose -f deployment/docker-compose.yml up -d --build

# 备份数据
tar -czf backup-$(date +%Y%m%d).tar.gz /data/sem-platform/storage
```
