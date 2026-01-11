#!/bin/bash
# Docker 镜像加速器配置脚本

echo "=== Docker 镜像加速器配置 ==="
echo ""

# 检查 Docker 是否运行
if ! docker ps > /dev/null 2>&1; then
    echo "❌ Docker 未运行，请先启动 Docker Desktop"
    exit 1
fi

echo "✅ Docker 正在运行"
echo ""

# 检查 Docker Desktop 配置文件位置
DOCKER_CONFIG_DIR="$HOME/Library/Group Containers/group.com.docker"
SETTINGS_FILE="$DOCKER_CONFIG_DIR/settings.json"

if [ -f "$SETTINGS_FILE" ]; then
    echo "找到 Docker Desktop 配置文件: $SETTINGS_FILE"
    echo ""
    echo "请手动在 Docker Desktop 中配置镜像加速器："
    echo "1. 打开 Docker Desktop"
    echo "2. 点击右上角设置图标 (Settings)"
    echo "3. 选择 'Docker Engine'"
    echo "4. 在 JSON 配置中添加以下内容："
    echo ""
    cat << 'EOF'
{
  "registry-mirrors": [
    "https://docker.mirrors.ustc.edu.cn",
    "https://hub-mirror.c.163.com",
    "https://mirror.baidubce.com"
  ]
}
EOF
    echo ""
    echo "5. 点击 'Apply & Restart'"
    echo ""
else
    echo "未找到 Docker Desktop 配置文件"
    echo "请按照以下步骤手动配置："
    echo ""
    echo "1. 打开 Docker Desktop"
    echo "2. 进入 Settings > Docker Engine"
    echo "3. 添加镜像加速器配置（见上方）"
    echo "4. 点击 Apply & Restart"
    echo ""
fi

echo "配置完成后，运行以下命令构建镜像："
echo "  cd evaluator_agent"
echo "  docker build -f Dockerfile.with_numpy -t code-evaluator:latest ."
echo ""


