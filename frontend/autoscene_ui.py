#!/usr/bin/env python3
"""AutoScene 上位机入口。

    # 默认连接 rosbridge；离线演示必须显式 --backend mock。
    python3 autoscene_ui.py --config ../AutoScene_0802/AutoScene/src/autoscene_bringup/config/hybrid.yaml

    # 指定 rosbridge 地址
    python3 autoscene_ui.py --backend rosbridge --bridge ws://127.0.0.1:9090 \
        --config .../config/hybrid.yaml
"""

from __future__ import annotations

import argparse
import sys

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication

from autoscene_gui import __version__
from autoscene_gui.main_window import MainWindow
from autoscene_gui.theme import APP_STYLE


def main() -> int:
    ap = argparse.ArgumentParser(description="AutoScene 上位机")
    ap.add_argument("--config", default="", help="bringup 的 hybrid.yaml / simulation.yaml 路径")
    ap.add_argument("--backend", default="rosbridge", choices=["mock", "rosbridge", "rospy"],
                   help="rosbridge=websocket 连 ROS（默认）；mock=离线模拟；rospy=本机 ROS1")
    ap.add_argument("--bridge", default="ws://127.0.0.1:9090", help="rosbridge websocket 地址")
    ap.add_argument("--dataset", default="",
                    help="前端本机读取的数据集目录；用于加载地图与候选车辆")
    ap.add_argument("--ros-dataset-path", default="",
                    help="数据集在远程 ROS 主机中的路径；为空时与前端使用同一路径")
    args = ap.parse_args()

    app = QApplication(sys.argv)
    app.setApplicationName("AutoScene Experiment Manager")
    app.setApplicationVersion(__version__)
    app.setOrganizationName("AutoScene")
    app.setStyle("Fusion")
    app.setStyleSheet(APP_STYLE)
    win = MainWindow(config_path=args.config or None, backend=args.backend,
                     bridge_url=args.bridge, local_dataset_path=args.dataset,
                     remote_dataset_path=args.ros_dataset_path)
    win.show()
    win.raise_()
    win.activateWindow()
    QTimer.singleShot(0, lambda: (win.raise_(), win.activateWindow()))
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())