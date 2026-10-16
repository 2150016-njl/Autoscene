#!/usr/bin/env bash
set -euo pipefail

SCRIPT_SOURCE="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_SOURCE}")" && pwd)"
BUNDLE_AUTOSCENE="$(cd "${SCRIPT_DIR}/.." 2>/dev/null && pwd)/AutoScene"

if [[ -n "${AUTOSCENE_ROOT:-}" ]]; then
    ROOT="${AUTOSCENE_ROOT}"
elif [[ -d "${BUNDLE_AUTOSCENE}/src/autoscene_bringup" ]]; then
    ROOT="${BUNDLE_AUTOSCENE}"
else
    ROOT="${HOME}/AutoScene"
fi

BRIDGE_URL="${AUTOSCENE_BRIDGE:-ws://127.0.0.1:9090}"
CONFIG_PATH="${AUTOSCENE_CONFIG:-${ROOT}/src/autoscene_bringup/config/hybrid.yaml}"
# 默认不再硬编码旧数据集。为空时让 UI 直接使用 hybrid.yaml 的 scenario.dataset_path。
LOCAL_DATASET_PATH="${AUTOSCENE_DATASET:-}"
ROS_DATASET_PATH="${AUTOSCENE_ROS_DATASET:-}"

if ! command -v conda >/dev/null 2>&1; then
    echo "错误：没有找到 conda。" >&2
    exit 1
fi

if ! conda env list | awk '{print $1}' | grep -qx 'autoscene-ui'; then
    echo "错误：Conda 环境 autoscene-ui 不存在。请先运行：" >&2
    echo "  ${SCRIPT_DIR}/create_conda_env.sh" >&2
    exit 1
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "错误：找不到配置文件 ${CONFIG_PATH}" >&2
    echo "可通过 AUTOSCENE_ROOT 或 AUTOSCENE_CONFIG 指定路径。" >&2
    exit 1
fi

if [[ -n "${LOCAL_DATASET_PATH}" && ! -d "${LOCAL_DATASET_PATH}" ]]; then
    echo "错误：前端本机找不到数据集 ${LOCAL_DATASET_PATH}" >&2
    echo "可通过 AUTOSCENE_DATASET 指定路径；不设置时使用 YAML 中的 dataset_path。" >&2
    exit 1
fi

# 禁止读取 ~/.local，确保测试和交付真正使用独立 Conda 环境。
export PYTHONNOUSERSITE=1

echo "前端数据：${LOCAL_DATASET_PATH:-<使用 YAML scenario.dataset_path>}"
echo "ROS 数据：${ROS_DATASET_PATH:-<同机时与前端一致>}"
echo "ROS Bridge：${BRIDGE_URL}"

ARGS=(
    --backend rosbridge
    --bridge "${BRIDGE_URL}"
    --config "${CONFIG_PATH}"
)
if [[ -n "${LOCAL_DATASET_PATH}" ]]; then
    ARGS+=(--dataset "${LOCAL_DATASET_PATH}")
fi
if [[ -n "${ROS_DATASET_PATH}" ]]; then
    ARGS+=(--ros-dataset-path "${ROS_DATASET_PATH}")
fi

exec conda run --no-capture-output -n autoscene-ui \
    python "${SCRIPT_DIR}/autoscene_ui.py" "${ARGS[@]}" "$@"
