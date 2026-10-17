#!/usr/bin/env bash
set -e

# Render 会自动注入 $PORT；本地测试时默认 10000。
PORT="${PORT:-10000}"

# production.py 的静态资源路径基于 junior-ai-lab 目录计算，故切换到 backend 启动。
cd "$(dirname "$0")/junior-ai-lab/backend"
export APP_ENV=production

exec uvicorn production:app --host 0.0.0.0 --port "$PORT" --workers 1
