#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v conda >/dev/null 2>&1; then
    echo "错误：没有找到 conda。请先安装 Miniconda/Anaconda/Miniforge。" >&2
    exit 1
fi

if conda env list | awk '{print $1}' | grep -qx 'autoscene-ui'; then
    echo "更新已有 Conda 环境 autoscene-ui ..."
    conda env update --name autoscene-ui --file "${SCRIPT_DIR}/environment.yml" --prune
else
    echo "创建 Conda 环境 autoscene-ui ..."
    conda env create --file "${SCRIPT_DIR}/environment.yml"
fi

echo
echo "环境准备完成。运行："
echo "  conda activate autoscene-ui"
echo "  ${SCRIPT_DIR}/run_conda_frontend.sh"
