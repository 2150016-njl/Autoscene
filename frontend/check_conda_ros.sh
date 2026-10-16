#!/usr/bin/env bash
set -euo pipefail

BRIDGE_URL="${AUTOSCENE_BRIDGE:-ws://127.0.0.1:9090}"

if ! command -v conda >/dev/null 2>&1; then
    echo "错误：没有找到 conda。" >&2
    exit 1
fi

if ! conda env list | awk '{print $1}' | grep -qx 'autoscene-ui'; then
    echo "错误：Conda 环境 autoscene-ui 不存在，请先运行 ./create_conda_env.sh" >&2
    exit 1
fi

export PYTHONNOUSERSITE=1
export AUTOSCENE_BRIDGE_CHECK="${BRIDGE_URL}"

conda run --no-capture-output -n autoscene-ui python - <<'PY'
import json
import os
import sys

from PyQt5 import QtCore
import numpy
import websocket
import yaml

url = os.environ["AUTOSCENE_BRIDGE_CHECK"]
ws = websocket.create_connection(url, timeout=5)
request_id = "autoscene_conda_check"
ws.send(json.dumps({
    "op": "call_service",
    "service": "/rosapi/get_time",
    "args": {},
    "id": request_id,
}))

while True:
    reply = json.loads(ws.recv())
    if reply.get("id") == request_id:
        break
ws.close()

if not reply.get("result", False):
    raise SystemExit("rosbridge 已连接，但 /rosapi/get_time 调用失败：%r" % reply)

print("Python:", sys.version.split()[0])
print("PyQt5:", QtCore.PYQT_VERSION_STR)
print("NumPy:", numpy.__version__)
print("PyYAML:", yaml.__version__)
print("websocket-client:", websocket.__version__)
print("ROS bridge: OK", url)
PY
