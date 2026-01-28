#!/bin/bash

echo "🔍 检查当前目录..."
pwd

if [[ $(basename $(pwd)) != "HydroMTL_CGC" ]]; then
    echo "❌ 错误：不在 HydroMTL_CGC 目录"
    echo "请执行: cd ~/code/HydroMTL_CGC"
    exit 1
fi

echo "🧹 清理已有的 .git..."
if [ -d ".git" ]; then
    backup_name=".git_backup_$(date +%Y%m%d_%H%M%S)"
    mv .git "$backup_name"
    echo "✅ 已备份 .git 到 $backup_name"
fi

echo "🚀 初始化 Git..."
git init
git config user.email "kingbroleo@outlook.com"
git config user.name "kingbroleo"

echo "📄 创建 .gitignore..."
cat > .gitignore << 'GITIGNORE'
# Python
__pycache__/
*.pyc

# 环境
.env
venv/

# IDE
.vscode/
.idea/

# 数据文件（绝对不要推送！）
../../hydro_data/
*.npy
*.pkl
*.h5

# 复现代码（不要推送）
../HydroMTL/

# 日志
*.log
logs/

# 系统文件
.cache/
.vscode-server/
GITIGNORE

echo "📝 添加文件..."
git add .

echo "📋 将要提交的文件："
git status --short

echo "💾 提交更改..."
read -p "请输入提交信息（默认: Initial commit）: " commit_msg
if [ -z "$commit_msg" ]; then
    commit_msg="Initial commit"
fi
git commit -m "$commit_msg"

echo "🔗 添加远程仓库..."
git remote add origin git@github.com:kingbroleo/HydroMTL_CGC.git

echo "🌿 重命名分支..."
git branch -M main 2>/dev/null || true

echo "✅ 本地设置完成！"
echo ""
echo "下一步："
echo "1. 测试 SSH 连接: ssh -T git@github.com"
echo "2. 推送代码: git push -u origin main"
echo "3. 如果推送失败，尝试: git push -f origin main"
