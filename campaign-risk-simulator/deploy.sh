#!/bin/bash
set -e

echo ">>> 拉取最新代码"
git pull

echo ">>> 重新构建并启动"
docker compose up --build -d

echo ">>> 完成，服务运行在 http://localhost:6791"
echo ">>> API 文档: http://localhost:6791/docs"
