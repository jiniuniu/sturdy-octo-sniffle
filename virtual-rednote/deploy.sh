#!/bin/bash
set -e

IMAGE_NAME="virtual-rednote"

echo "==> 构建镜像..."
docker build -t $IMAGE_NAME .

echo "==> 启动服务..."
docker compose up -d

echo "==> 完成。"
docker compose ps
