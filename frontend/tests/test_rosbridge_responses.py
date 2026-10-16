"""Real rosbridge wire formats: service errors must not become disconnects."""
import copy
import os
import sys
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["AUTOSCENE_DISABLE_SETTINGS"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5.QtWidgets import QApplication
from autoscene_gui.ros_client import RosbridgeClient, TOPICS
from autoscene_gui.main_window import MainWindow


class RosbridgeResponseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.client = RosbridgeClient()
        self.addCleanup(self.client.shutdown)

    def dispatch(self, values, result=False):
        results = []
        self.client._register_pending("test", results.append, "test")
        frame = {"op": "service_response", "id": "test", "values": values,
                 "result": result, "service": "/autoscene/experiment/start"}
        original = copy.deepcopy(frame)
        self.client._dispatch(frame)
        self.app.processEvents()
        self.assertEqual(frame, original)
        self.assertEqual(len(results), 1)
        self.assertEqual(self.client._pending, {})
        self.assertEqual(self.client._pending_timers, {})
        return results[0]

    def test_missing_service_string_returns_real_reason(self):
        result = self.dispatch("Service /autoscene/experiment/start does not exist")
        self.assertFalse(result["success"])
        self.assertIn("does not exist", result["message"])
        self.assertIn("experiment_manager", result["message"])

    def test_failure_does_not_break_following_status_or_report_disconnect(self):
        connected, statuses = [], []
        self.client.connected.connect(lambda *args: connected.append(args))
        self.client.status.connect(statuses.append)
        self.dispatch("backend raised TypeError")
        self.client._dispatch({"op": "publish", "topic": TOPICS["status"][0],
                               "msg": {"state": "idle", "step": 7}})
        self.assertEqual(connected, [])
        self.assertEqual(statuses[-1]["step"], 7)

    def test_success_dictionary_and_empty_service(self):
        self.assertTrue(self.dispatch({"success": True, "message": "started"}, True)["success"])
        self.assertTrue(self.dispatch({}, True)["success"])
        self.assertTrue(self.dispatch(None, True)["success"])

    def test_transport_failure_cannot_be_overridden_by_nested_success(self):
        self.assertFalse(self.dispatch({"success": True, "message": "transport failed"})["success"])

    def test_application_failure_is_not_transport_success(self):
        result = self.dispatch({"success": False, "message": "not prepared"}, True)
        self.assertFalse(result["success"])
        self.assertEqual(result["message"], "not prepared")

    def test_malformed_values_are_explicit_failures(self):
        for raw in (["unexpected"], 1, True, "", []):
            with self.subTest(raw=raw):
                result = self.dispatch(raw, True)
                self.assertFalse(result["success"])
                self.assertTrue(result["message"])

    def test_status_error_completes_once_even_if_service_response_follows(self):
        results = []
        self.client._register_pending("test", results.append, "start")
        self.client._dispatch({"op": "status", "id": "test", "level": "error", "msg": "missing service"})
        self.client._dispatch({"op": "service_response", "id": "test", "result": False, "values": "missing service"})
        self.app.processEvents()
        self.assertEqual(results, [{"success": False, "message": "missing service"}])
        self.assertEqual(self.client._pending_timers, {})

    def test_unmatched_error_and_warning_do_not_fail_other_requests(self):
        results = []
        self.client._register_pending("test", results.append, "start")
        self.client._dispatch({"op": "status", "id": "test", "level": "warning", "msg": "slow"})
        self.client._dispatch({"op": "status", "id": "other", "level": "error", "msg": "unrelated"})
        self.assertEqual(results, [])
        self.assertIn("test", self.client._pending)

    def test_set_params_preserves_original_error_and_stops_at_first_failure(self):
        sent, results = [], []
        with patch.object(self.client, "_send", side_effect=lambda frame: sent.append(frame) or True):
            self.client.set_params({"rosparams": {"/ego": {}, "/vehicles": {}}}, results.append)
            self.client._dispatch({"op": "service_response", "id": sent[0]["id"],
                                   "result": False, "values": "permission denied"})
            self.app.processEvents()
        self.assertEqual(len(sent), 1)
        self.assertFalse(results[0]["success"])
        self.assertIn("/ego", results[0]["message"])
        self.assertIn("permission denied", results[0]["message"])

    def test_start_error_releases_window_busy_state_and_keeps_connection(self):
        window = MainWindow(backend="mock")
        self.app.processEvents()
        window.client.shutdown()
        window.client = self.client
        window._connected = True
        window.state["prepared"] = True
        try:
            with patch.object(self.client, "_send", return_value=True):
                window.start_experiment()
                self.assertTrue(window._busy_action)
                self.client._dispatch({"op": "service_response", "id": "call_1", "result": False,
                                       "values": "Service /autoscene/experiment/start does not exist"})
                self.app.processEvents()
            self.assertEqual(window._busy_action, "")
            self.assertTrue(window._connected)
            self.assertIn("does not exist", window.ed_log.toPlainText())
        finally:
            window.state["state"] = "idle"
            window.close()


if __name__ == "__main__":
    unittest.main()
