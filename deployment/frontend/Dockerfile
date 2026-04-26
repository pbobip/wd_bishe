# ============================================================
# 前端 Dockerfile — Node 构建 + nginx 静态部署
# 构建:
#   docker build -t sem-frontend -f deployment/frontend/Dockerfile .
#
# 运行:
#   docker run -d -p 80:80 sem-frontend
# ============================================================

# ---------- Stage 1: Builder ----------
FROM node:20-alpine AS builder

WORKDIR /app

# Copy package files first for better layer caching
COPY frontend/package.json frontend/package-lock.json* ./

# Install dependencies
RUN npm ci --legacy-peer-deps 2>/dev/null || npm install --legacy-peer-deps

# Copy source code
COPY frontend/ ./

# Build production bundle
# Vite will proxy /api and /static to the backend at runtime via nginx
RUN npm run build

# ---------- Stage 2: nginx runtime ----------
FROM nginx:1.25-alpine AS runtime

# Remove default nginx config
RUN rm /etc/nginx/conf.d/default.conf

# Copy built assets
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy custom nginx config
COPY deployment/frontend/nginx.conf /etc/nginx/conf.d/default.conf

# Expose port
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD wget -qO- http://localhost/health || exit 1

CMD ["nginx", "-g", "daemon off;"]
