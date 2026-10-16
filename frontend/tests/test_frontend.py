"""AutoScene 前端关键状态机回归测试。"""

import os
import sys
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["AUTOSCENE_DISABLE_SETTINGS"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication

from autoscene_gui.main_window import MainWindow
from autoscene_gui.ros_client import RosbridgeClient


class _DatasetStub:
    count = 1


class FrontendStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = MainWindow(backend="mock", remote_dataset_path="/vm/data")
        self.app.processEvents()

    def tearDown(self):
        self.window.state["state"] = "idle"
        self.window.close()
        self.app.processEvents()

    def test_remote_path_is_sent_to_ros(self):
        self.assertFalse(self.window.chk_ros_same_host.isChecked())
        payload = self.window._rosparam_payload([])
        self.assertEqual(payload["rosparams"]["/scenario"]["dataset_path"], "/vm/data")

    def test_same_host_uses_one_dataset_path_and_hides_remote_input(self):
        window = MainWindow(backend="mock", local_dataset_path="/shared/data")
        self.app.processEvents()
        try:
            self.assertTrue(window.chk_ros_same_host.isChecked())
            self.assertTrue(window.ed_remote_path.isHidden())
            payload = window._rosparam_payload([])
            self.assertEqual(payload["rosparams"]["/scenario"]["dataset_path"],
                             "/shared/data")
        finally:
            window.close()

    def test_remote_path_is_restored_after_same_host_toggle(self):
        self.window.chk_ros_same_host.setChecked(True)
        self.assertEqual(self.window.remote_dataset_path, "")
        self.window.chk_ros_same_host.setChecked(False)
        self.assertEqual(self.window.remote_dataset_path, "/vm/data")

    def test_config_panel_keeps_a_readable_width(self):
        self.window.resize(1180, 720)
        self.window.show()
        QTest.qWait(50)
        self.assertGreaterEqual(self.window.main_splitter.widget(0).width(), 360)
        self.assertGreaterEqual(self.window.ed_path.width(), 200)
        self.assertIn("/vm/data", self.window.ed_remote_path.toolTip())

    def test_ros_actions_follow_connection_and_runtime_state(self):
        self.assertTrue(self.window._connected)
        self.assertFalse(self.window.btn_prepare.isEnabled())

        self.window.dataset = _DatasetStub()
        self.window.refresh_all()
        self.assertTrue(self.window.btn_prepare.isEnabled())

        self.window._on_status({"state": "running", "prepared": True})
        self.assertFalse(self.window.btn_prepare.isEnabled())
        self.assertTrue(self.window.btn_stop.isEnabled())
        self.assertFalse(self.window.view.editing_enabled)

        self.window._on_connected(False, "test disconnect")
        self.assertFalse(self.window.btn_stop.isEnabled())
        self.assertFalse(self.window.act_apply.isEnabled())

    def test_over_limit_speed_can_be_fixed_in_one_action(self):
        self.window.dataset = _DatasetStub()
        self.window.virtuals = [{
            "key": "V-001", "track_id": "3", "x": 0.0, "y": 0.0,
            "heading": 0.0, "speed": 7.901,
        }]
        self.window.clamp_initial_speeds()
        self.assertAlmostEqual(self.window.virtuals[0]["speed"],
                               self.window.cfg.max_initial_speed)

    def test_initial_speed_is_clamped_when_vehicle_joins_experiment(self):
        self.window.cfg.scenario["history_source"] = "initial"
        value = self.window._speed_for_initial(7.901, "track 3")
        self.assertAlmostEqual(value, self.window.cfg.max_initial_speed)

    def test_selected_virtual_bv_can_be_deleted(self):
        self.window.dataset = _DatasetStub()
        self.window.virtuals = [{
            "key": "V-001", "track_id": "3", "x": 0.0, "y": 0.0,
            "heading": 0.0, "speed": 4.0,
        }]
        self.window.refresh_all()
        self.window.tbl_virtual.selectRow(0)
        self.window.remove_selected_virtual()
        self.assertEqual(self.window.virtuals, [])


class RosbridgeTimeoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_pending_request_times_out(self):
        client = RosbridgeClient("ws://127.0.0.1:1")
        client.REQUEST_TIMEOUT_SEC = 0.05
        results = []
        client._register_pending("test", results.append, "测试请求")
        QTest.qWait(120)
        self.app.processEvents()
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0]["success"])
        self.assertIn("超时", results[0]["message"])


if __name__ == "__main__":
    unittest.main()
