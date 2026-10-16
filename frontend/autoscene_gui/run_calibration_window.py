#!/usr/bin/env python3
"""Standalone launcher for AutoScene G<->A calibration window.

Place this file in the frontend directory that contains ``autoscene_gui/``.
It deliberately does not import rospy; rosbridge mode only needs websocket-client.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make the sibling autoscene_gui package importable even when invoked through
# ``conda run`` from another shell environment.
FRONTEND_ROOT = Path(__file__).resolve().parent
if str(FRONTEND_ROOT) not in sys.path:
    sys.path.insert(0, str(FRONTEND_ROOT))

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication

from autoscene_gui.calibration_window import CalibrationWindow
from autoscene_gui.config_model import ExperimentConfig
from autoscene_gui.ros_client import RosbridgeClient

try:
    from autoscene_gui.theme import APP_STYLE
except Exception:
    APP_STYLE = ""


def main() -> int:
    ap = argparse.ArgumentParser(description="AutoScene G<->A calibration")
    ap.add_argument("--config", required=True, help="hybrid.yaml path")
    ap.add_argument("--bridge", default="ws://127.0.0.1:9090", help="rosbridge websocket URL")
    ap.add_argument(
        "--no-rosbridge",
        action="store_true",
        help="do not connect rosbridge; use the window's direct UDP listener instead",
    )
    args = ap.parse_args()

    config_path = os.path.abspath(os.path.expanduser(args.config))
    cfg = ExperimentConfig.load(config_path)

    app = QApplication(sys.argv)
    app.setApplicationName("AutoScene G-A Calibration")
    app.setStyle("Fusion")
    if APP_STYLE:
        app.setStyleSheet(APP_STYLE)

    client = None
    if not args.no_rosbridge:
        client = RosbridgeClient(args.bridge)

    win = CalibrationWindow(cfg, bridge_client=client)

    if client is not None:
        client.ads_decoded.connect(win.on_ads_decoded)
        client.connected.connect(
            lambda ok, text: win.lab_rx_mode.setText(text if text else ("rosbridge connected" if ok else "rosbridge disconnected"))
        )
        client.start()
        app.aboutToQuit.connect(client.shutdown)

    win.show()
    win.raise_()
    win.activateWindow()
    QTimer.singleShot(0, lambda: (win.raise_(), win.activateWindow()))
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())