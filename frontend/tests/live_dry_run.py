"""通过新版 UI 状态机执行一次真实 rosbridge dry-run 集成测试。"""

import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["AUTOSCENE_DISABLE_SETTINGS"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication

from autoscene_gui.main_window import MainWindow


def main() -> int:
    app = QApplication([])
    window = MainWindow(
        config_path="/Users/mac/Desktop/UI/AutoScene/src/autoscene_bringup/config/hybrid.yaml",
        backend="rosbridge",
        bridge_url="ws://127.0.0.1:9090",
        local_dataset_path="/Users/mac/Desktop/UI/UI-0819/AutoScene/data/tjwest260725intersection",
        remote_dataset_path="/home/stark/AutoScene/data/tjwest260725intersection",
    )
    result = {"code": 1, "started": False, "seen_running": False,
              "frames": 0, "deadline": time.monotonic() + 70.0}
    window.client.world.connect(
        lambda _msg: result.__setitem__("frames", result["frames"] + 1))

    def finish(code: int, message: str) -> None:
        result["code"] = code
        print(message)
        print("final_state=", window.state)
        print("world_frames=", result["frames"])
        window.state["state"] = "idle"
        window.close()
        app.quit()

    def begin() -> None:
        if not window._connected or window.dataset is None:
            finish(2, "preflight_failed connected=%s dataset=%s" %
                   (window._connected, window.dataset is not None))
            return
        window.set_endpoint("12", "platform_001")
        window.prepare()

    def poll() -> None:
        state = str(window.state.get("state"))
        if state == "error":
            finish(3, "experiment_error")
            return
        if state == "prepared" and not result["started"]:
            result["started"] = True
            window.start_experiment()
        if state == "running":
            result["seen_running"] = True
        if result["seen_running"] and state == "idle":
            finish(0, "dry_run_experiment=OK")
            return
        if time.monotonic() > result["deadline"]:
            window.stop_experiment()
            finish(4, "dry_run_timeout")
            return
        QTimer.singleShot(200, poll)

    QTimer.singleShot(3000, begin)
    QTimer.singleShot(200, poll)
    app.exec_()
    return int(result["code"])


if __name__ == "__main__":
    raise SystemExit(main())
