"""Pose editing, placement and readable UI regressions; no ROS required."""
import copy
import math
import os
import sys
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["AUTOSCENE_DISABLE_SETTINGS"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5.QtCore import QEvent, QPointF, Qt
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QGraphicsItem
from autoscene_gui.main_window import MainWindow
from autoscene_gui.scenario_loader import Scene, SceneObject
from autoscene_gui.scene_view import w2s


class DatasetStub:
    count = 1
    path = "/tmp/source-data"


class SceneInteractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = MainWindow(backend="mock")
        self.app.processEvents()
        w = self.window
        w.dataset = DatasetStub()
        w.scene = Scene("test", SceneObject("sdc", 100, 200, 0, 1), [
            SceneObject("real", 105, 200, 0, 2, physical=True),
            SceneObject("near", 110, 200, 0, 2),
            SceneObject("far", 130, 210, 1, 2),
        ], runtime_origin_x=100, runtime_origin_y=200)
        w.refresh_all()
        self.app.processEvents()

    def tearDown(self):
        self.window.state["state"] = "idle"
        self.window.close()
        self.app.processEvents()

    def test_platform_drag_changes_initial_ros_coordinates(self):
        w = self.window
        w.set_endpoint("real", w.cfg.endpoint_ids()[0])
        item = w.view.vehicles["real"]
        self.assertTrue(item.flags() & QGraphicsItem.ItemIsMovable)
        w.view.select_key("real")
        self.assertIs(w.view.handle.target, item)
        item.setPos(w2s(120, 230))
        agent = w.agents(runtime_coordinates=True)[0]
        self.assertEqual((agent.x, agent.y), (20, 30))
        w.cb_edit_vehicle.setCurrentIndex(w.cb_edit_vehicle.findData("real"))
        w.pose_inputs["heading"].setValue(90)
        self.assertAlmostEqual(w.agents()[0].heading, math.pi / 2)
        self.assertFalse(w.state["prepared"])

    def test_free_placement_uses_selected_track_not_nearest_car_or_trail(self):
        w = self.window
        w.cb_edit_vehicle.setCurrentIndex(w.cb_edit_vehicle.findData("far"))
        w.on_map_clicked(110.1, 199.8)  # closest is 'near', chosen source is 'far'
        v = w.virtuals[0]
        self.assertEqual(v["track_id"], "far")
        self.assertEqual((v["x"], v["y"]), (110.1, 199.8))
        self.assertNotIn("far", w.view.vehicles)  # no original duplicate underneath
        self.assertIn(v["key"], w.view.vehicles)

    def test_source_required_and_lane_snap_never_falls_back_to_car(self):
        w = self.window
        w.on_map_clicked(110, 200)
        self.assertEqual(w.virtuals, [])
        w.cb_edit_vehicle.setCurrentIndex(w.cb_edit_vehicle.findData("near"))
        w.chk_snap_lane.setChecked(True)
        w.on_map_clicked(110, 200)  # absent map: reject instead of snapping to track
        self.assertEqual(w.virtuals, [])
        w.scene.lanes = [[(100, 202), (150, 202)]]
        w.on_map_clicked(111, 200)
        self.assertEqual((w.virtuals[0]["x"], w.virtuals[0]["y"]), (111, 202))

    def test_builder_edits_background_but_live_state_cannot_overwrite_it(self):
        w = self.window
        self.assertFalse(w.view.can_move("background"))
        w.chk_builder.setChecked(True)
        item = w.view.vehicles["far"]
        self.assertTrue(item.flags() & QGraphicsItem.ItemIsMovable)
        item.setPos(w2s(140, 220))
        self.assertEqual(w._edited_track_poses["far"][:2], (140, 220))
        w._on_world({"agents": [{"vehicle_id": "far", "pose": {"position": {"x": 0, "y": 0}}}]})
        self.assertEqual(item.world_pose()[:2], (140, 220))
        self.assertFalse(w.btn_start.isEnabled())
        self.assertFalse(w.btn_prepare.isEnabled())

    def test_running_locks_platform_ego_and_builder_editing(self):
        w = self.window
        w._on_status({"state": "running", "prepared": True})
        for key in ("ego", "real"):
            self.assertFalse(w.view.vehicles[key].flags() & QGraphicsItem.ItemIsMovable)
        self.assertFalse(w.chk_builder.isEnabled())
        self.assertFalse(w.btn_export_scene.isEnabled())
        w.on_vehicle_moved("real", 999, 999, 0)
        self.assertEqual(w.scene.objects[0].x, 105)

    def test_rotation_handle_receives_actual_mouse_drag(self):
        w = self.window
        w.show()
        self.app.processEvents()
        w.view.focus_actors()
        w.view.select_key("real")
        w.cb_edit_vehicle.setCurrentIndex(w.cb_edit_vehicle.findData("real"))
        handle = w.view.handle
        start = w.view.mapFromScene(handle.scenePos())
        end = w.view.mapFromScene(w2s(105, 207))
        QTest.mousePress(w.view.viewport(), Qt.LeftButton, Qt.NoModifier, start)
        event = QMouseEvent(QEvent.MouseMove, QPointF(end), Qt.NoButton, Qt.LeftButton, Qt.NoModifier)
        QApplication.sendEvent(w.view.viewport(), event)
        QTest.mouseRelease(w.view.viewport(), Qt.LeftButton, Qt.NoModifier, end)
        self.assertAlmostEqual(w.scene.objects[0].heading, math.pi / 2, delta=0.05)

    def test_font_scale_applies_to_new_table_items_and_survives_resize(self):
        w = self.window
        w.set_ui_scale(175)
        w.refresh_platform_table()
        self.assertEqual(w.tbl_platform.item(0, 1).font().pixelSize(), 21)
        self.assertGreaterEqual(w.tbl_platform.verticalHeader().defaultSectionSize(), 59)
        w.showMaximized()
        self.app.processEvents()
        w.showNormal()
        w.resize(1180, 720)
        self.app.processEvents()
        self.assertEqual(w.ui_scale, 175)
        self.assertEqual(w.sp_ui_scale.value(), 175)

    def test_font_control_is_on_top_and_vbd_cadence_has_no_ui(self):
        w = self.window
        w.show()
        self.app.processEvents()
        self.assertTrue(w.sp_ui_scale_top.isVisible())
        self.assertLess(w.sp_ui_scale_top.mapTo(w, w.sp_ui_scale_top.rect().topLeft()).y(), 200)
        w.sp_ui_scale_top.setValue(175)
        self.assertEqual(w.ui_scale, 175)
        self.assertEqual(w.sp_ui_scale.value(), 175)
        self.assertFalse(hasattr(w, "cb_vbd_cadence"))
        w.btn_vehicle_algorithms.click()
        self.assertIs(w.tabs.currentWidget(), w.algorithm_page)

    def test_each_vehicle_algorithm_reaches_yaml_ros_and_locks_when_running(self):
        w = self.window
        w.set_endpoint("real", w.cfg.endpoint_ids()[0])
        w.cb_edit_vehicle.setCurrentIndex(w.cb_edit_vehicle.findData("far"))
        w.on_map_clicked(130, 210)
        w.refresh_algorithm_table()
        methods = {"sdc": "VBD", "real": "Pluto", "far": "pluto_hybrid"}
        for row in range(w.tbl_algorithms.rowCount()):
            sid = w.tbl_algorithms.item(row, 2).text()
            combo = w.tbl_algorithms.cellWidget(row, 4)
            combo.setCurrentIndex(combo.findData(methods[sid]))
        payload = w._rosparam_payload(w.agents())["rosparams"]
        self.assertEqual(payload["/ego"]["planner_method"], "VBD")
        self.assertEqual({a["scenario_track_id"]: a["planner_method"] for a in payload["/vehicles"]["agents"]},
                         {"real": "Pluto", "far": "pluto_hybrid"})
        self.assertFalse(w.state["prepared"])
        w._on_status({"state": "running", "prepared": True})
        self.assertFalse(w.tbl_algorithms.isEnabled())
        w._set_vehicle_planner("real", "VBD")
        self.assertEqual(w.planner_methods["real"], "Pluto")

    def test_algorithm_inheritance_and_saved_config_round_trip(self):
        import tempfile
        from autoscene_gui.config_model import ExperimentConfig
        w = self.window
        w.set_endpoint("real", w.cfg.endpoint_ids()[0])
        w._set_vehicle_planner("real", "VBD")
        w._set_vehicle_planner("sdc", "Pluto", True)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "experiment.yaml")
            w.cfg.save(path)
            loaded = ExperimentConfig.load(path)
            self.assertEqual(loaded.raw["ego"]["planner_method"], "Pluto")
            self.assertEqual(loaded.raw["vehicles"]["agents"][0]["planner_method"], "VBD")
        w._set_vehicle_planner("real", None)
        w._set_vehicle_planner("sdc", None, True)
        self.assertNotIn("planner_method", w.cfg.raw["ego"])
        self.assertNotIn("planner_method", w.cfg.raw["vehicles"]["agents"][0])

    def test_map_tables_and_selected_vehicle_offer_synchronized_planners(self):
        w = self.window
        w.set_endpoint("real", w.cfg.endpoint_ids()[0])
        w.cb_edit_vehicle.setCurrentIndex(w.cb_edit_vehicle.findData("far"))
        self.assertFalse(w.cb_vehicle_planner.isEnabled())  # background replay, not configured
        w.on_map_clicked(130, 210)
        real = w.tbl_platform.cellWidget(0, 5)
        virtual = w.tbl_virtual.cellWidget(0, 5)
        self.assertEqual(w.tbl_platform.horizontalHeader().visualIndex(5), 1)
        self.assertEqual(w.tbl_virtual.horizontalHeader().visualIndex(5), 1)
        for combo, method in ((real, "Pluto"), (virtual, "VBD")):
            self.assertEqual([combo.itemText(i) for i in range(3)], ["默认", "Pluto", "VBD"])
            combo.setCurrentIndex(combo.findData(method))
        w.cb_edit_vehicle.setCurrentIndex(w.cb_edit_vehicle.findData("real"))
        self.assertTrue(w.cb_vehicle_planner.isEnabled())
        self.assertEqual(w.cb_vehicle_planner.currentData(), "Pluto")
        w.cb_vehicle_planner.setCurrentIndex(w.cb_vehicle_planner.findData("VBD"))
        self.assertEqual(real.currentData(), "VBD")
        real.setCurrentIndex(0)
        self.assertIsNone(w.cb_vehicle_planner.currentData())
        w.cb_edit_vehicle.setCurrentIndex(w.cb_edit_vehicle.findData("ego"))
        w.cb_vehicle_planner.setCurrentIndex(w.cb_vehicle_planner.findData("Pluto"))
        payload = w._rosparam_payload(w.agents())["rosparams"]
        self.assertEqual(payload["/ego"]["planner_method"], "Pluto")
        by_track = {agent["scenario_track_id"]: agent for agent in payload["/vehicles"]["agents"]}
        self.assertNotIn("planner_method", by_track["real"])
        self.assertEqual(by_track["far"]["planner_method"], "VBD")
        for row in range(w.tbl_algorithms.rowCount()):
            tid = w.tbl_algorithms.item(row, 2).text()
            self.assertEqual(w.tbl_algorithms.cellWidget(row, 4).currentData(),
                             {"sdc": "Pluto", "real": None, "far": "VBD"}[tid])

    def test_inline_planners_are_locked_in_builder_and_running_modes(self):
        w = self.window
        w.cb_edit_vehicle.setCurrentIndex(w.cb_edit_vehicle.findData("real"))
        w.chk_builder.setChecked(True)
        self.assertFalse(w.cb_vehicle_planner.isEnabled())
        self.assertFalse(w.tbl_platform.cellWidget(0, 5).isEnabled())
        w.refresh_platform_table()  # rebuilding during a pose edit must not unlock algorithms
        self.assertFalse(w.tbl_platform.cellWidget(0, 5).isEnabled())
        w._set_vehicle_planner("real", "VBD")
        self.assertNotIn("real", w.planner_methods)
        w.chk_builder.setChecked(False)
        self.assertTrue(w.cb_vehicle_planner.isEnabled())
        w._on_status({"state": "starting", "prepared": True})
        self.assertFalse(w.cb_vehicle_planner.isEnabled())
        self.assertFalse(w.tbl_platform.cellWidget(0, 5).isEnabled())

    def test_old_backend_prepared_status_cannot_authorize_unsent_algorithm_change(self):
        w = self.window
        w._set_vehicle_planner("sdc", "VBD", True)
        w._on_status({"state": "stopped", "prepared": True})
        self.assertFalse(w.state["prepared"])
        w._connected = True
        with patch.object(w.client, "call") as call:
            w.start_experiment()
        call.assert_not_called()

    def test_yaml_and_log_keep_light_theme(self):
        w = self.window
        self.assertFalse(w.ed_yaml.property("dark"))
        self.assertFalse(w.ed_log.property("dark"))
        w.ed_log.clear()
        w.log("WARN", "visible warning")
        cursor = w.ed_log.textCursor()
        cursor.setPosition(1)
        self.assertEqual(cursor.charFormat().foreground().color().name(), "#8a4000")

    def test_dataset_history_rejects_pose_edits_before_ros_write(self):
        w = self.window
        w.cfg.scenario["history_source"] = "dataset"
        w._original_scene = copy.deepcopy(w.scene)
        w.on_vehicle_moved("ego", 104, 201, 0.5)
        self.assertEqual(w._dataset_pose_override_ids(), ["sdc"])
        w._connected = True
        with patch("autoscene_gui.main_window.QMessageBox.warning") as warning, \
                patch.object(w.client, "set_params") as send:
            w.prepare()
            w.apply_params()
        self.assertEqual(warning.call_count, 2)
        send.assert_not_called()
        self.assertFalse(w.state["prepared"])

    def test_dataset_history_detects_virtual_pose_but_allows_unchanged_source(self):
        w = self.window
        w.cfg.scenario["history_source"] = "dataset"
        w.virtuals = [{"key": "V-001", "track_id": "far", "x": 130, "y": 210,
                       "heading": 1 + 2 * math.pi, "speed": 2}]
        self.assertEqual(w._dataset_pose_override_ids(), [])
        w.virtuals[0]["x"] = 110
        self.assertEqual(w._dataset_pose_override_ids(), ["far"])
        w.cfg.scenario["history_source"] = "initial"
        self.assertTrue(w._validate_pose_history_mode())

    def test_history_change_requires_prepare_again(self):
        w = self.window
        w.cfg.mode = "simulation"
        w.state["prepared"] = True
        w.on_history_changed("dataset")
        self.assertFalse(w.state["prepared"])

    def test_export_ui_maps_ego_and_virtual_to_original_track_ids(self):
        w = self.window
        w.on_vehicle_moved("ego", 104, 201, 0.5)
        w.virtuals = [{"key": "V-001", "track_id": "far", "x": 112, "y": 212,
                       "heading": 1.5, "speed": 2}]
        with patch("autoscene_gui.main_window.QFileDialog.getSaveFileName", return_value=("/tmp/new-variant", "")), \
                patch("autoscene_gui.main_window.QMessageBox.information"), \
                patch("autoscene_gui.scenario_editor.export_scene_variant", return_value="/tmp/new-variant/sd.pkl") as export:
            w.export_scene()
        export.assert_called_once_with(w.dataset, 0, {"sdc": (104, 201, 0.5),
                                                    "far": (112, 212, 1.5)}, "/tmp/new-variant")


if __name__ == "__main__":
    unittest.main()
