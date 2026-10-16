"""CPU-only loader/export regression tests (no Qt, ROS, renderer or GPU).

Run: python -B -m unittest discover -s frontend/tests -p test_scenario_editor.py -v
Set AUTOSCENE_ORIGINAL_ROOT to the explicitly trusted bundled checkout to also
validate real OSM/nuPlan/tjwest PKLs against that checkout's backend source.
"""

import ast
import copy
import hashlib
import logging
import math
import os
from pathlib import Path
import pickle
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoscene_gui import scenario_editor as editor
from autoscene_gui.scenario_editor import ScenarioExportError, export_scene_variant
from autoscene_gui.scenario_loader import DatasetLoadError, ScenarioDataset, _scene_from_raw


def _dump(path, value):
    with Path(path).open("wb") as stream:
        pickle.dump(value, stream, protocol=4)


def _read(path):
    with Path(path).open("rb") as stream:
        return pickle.load(stream)


def _scenario(dtype=np.float64, columns=3, singleton=False):
    length = 7
    tracks = {}
    for tid in ("ego", "car", "other"):
        t = np.arange(length, dtype=dtype)
        positions = np.column_stack((10 + t, 20 + 0.5 * t, 5 + t))[:, :columns]
        velocities = np.column_stack((1 + 0.1 * t, 2 + 0.2 * t, 9 + t))[:, :columns]
        heading = np.linspace(-3.1, 3.3, length, dtype=dtype)
        valid = np.ones(length, dtype=bool)
        if tid == "car":
            valid[[1, 5]] = False
            positions[1] = np.nan
            positions[5] = -0.0
            heading[1] = np.nan
            heading[5] = 123.0
            velocities[1] = np.nan
            velocities[5] = -987.0
        tracks[tid] = {
            "type": "VEHICLE",
            "metadata": {"type": "VEHICLE", "object_id": tid, "track_length": length},
            "state": {
                "position": positions, "velocity": velocities,
                "heading": heading[:, None] if singleton else heading,
                "valid": valid[:, None] if singleton else valid,
                "length": np.full((length, 1), 5.2, dtype=dtype),
                "width": np.full((length, 1), 2.3, dtype=dtype),
                "action": np.arange(length * 2).reshape(length, 2),
                "custom": np.array(["keep"] * length),
            },
        }
    return {
        "id": "source-id", "version": "test-v1", "length": length, "tracks": tracks,
        "dynamic_map_states": {},
        "map_features": {"lane": {
            "type": "LANE_SURFACE_STREET",
            "polyline": np.array([[0., 0., 2.], [100., 100., 3.]]),
            "entry_lanes": ["unchanged-entry-id"], "custom": "keep",
        }},
        "metadata": {
            "id": "source-id", "scenario_id": "original", "sdc_id": "ego",
            "ts": np.arange(length, dtype=np.float32) * 0.1,
            "coordinate": "metadrive", "metadrive_processed": False,
            "current_time_index": 3, "dataset": "fixture", "custom": {"a": [1, 2]},
        },
    }


class _TreeAssertions:
    def assertTreeEqual(self, left, right):
        if isinstance(left, np.ndarray):
            self.assertIsInstance(right, np.ndarray)
            self.assertEqual(left.shape, right.shape)
            self.assertEqual(left.dtype, right.dtype)
            self.assertEqual(left.tobytes(), right.tobytes())
        elif isinstance(left, dict):
            self.assertEqual(list(left), list(right))
            for key in left:
                self.assertTreeEqual(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(type(left), type(right))
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assertTreeEqual(a, b)
        else:
            self.assertEqual(left, right)

    def assertVariant(self, original, result, poses, frame):
        self.assertEqual(list(original), list(result))
        for key in original:
            if key != "tracks":
                self.assertTreeEqual(original[key], result[key])
        self.assertEqual(list(original["tracks"]), list(result["tracks"]))
        for tid, track in original["tracks"].items():
            modified = result["tracks"][tid]
            if tid not in poses:
                self.assertTreeEqual(track, modified)
                continue
            for key in track:
                if key != "state":
                    self.assertTreeEqual(track[key], modified[key])
            before, after = track["state"], modified["state"]
            valid = np.asarray(before.get("valid", np.ones(original["length"]))).reshape(-1).astype(bool)
            source_heading = np.asarray(before["heading"]).reshape(-1)[frame]
            delta = poses[tid][2] - source_heading
            c, s = math.cos(delta), math.sin(delta)
            rotation = np.array([[c, -s], [s, c]])
            for key, value in before.items():
                if key not in ("position", "heading", "velocity"):
                    self.assertTreeEqual(value, after[key])
                    continue
                self.assertEqual(np.asarray(value).shape, np.asarray(after[key]).shape)
                if isinstance(value, np.ndarray):
                    self.assertEqual(value.dtype, after[key].dtype)
                    self.assertTreeEqual(value[~valid], after[key][~valid])
                else:
                    for i in np.flatnonzero(~valid):
                        self.assertTreeEqual(value[i], after[key][i])
            p0, p1 = np.asarray(before["position"]), np.asarray(after["position"])
            expected = (p0[valid, :2] - p0[frame, :2]) @ rotation.T + poses[tid][:2]
            np.testing.assert_allclose(p1[valid, :2], expected, atol=2e-5, rtol=2e-6)
            np.testing.assert_allclose(p1[frame, :2], poses[tid][:2], atol=2e-5, rtol=2e-6)
            # Every valid segment rotates, not just the displayed frame.
            np.testing.assert_allclose(np.diff(p1[valid, :2], axis=0),
                                       np.diff(p0[valid, :2], axis=0) @ rotation.T, atol=3e-5, rtol=2e-6)
            if p0.shape[1] > 2:
                np.testing.assert_array_equal(p0[:, 2:], p1[:, 2:])
            angles = np.asarray(after["heading"]).reshape(-1)[valid]
            expected_angles = np.asarray(before["heading"]).reshape(-1)[valid] + delta
            np.testing.assert_allclose(np.sin(angles), np.sin(expected_angles), atol=3e-6)
            np.testing.assert_allclose(np.cos(angles), np.cos(expected_angles), atol=3e-6)
            self.assertTrue(np.all(np.abs(angles) <= math.pi + 1e-6))
            if "velocity" in before:
                v0, v1 = np.asarray(before["velocity"]), np.asarray(after["velocity"])
                np.testing.assert_allclose(v1[valid, :2], v0[valid, :2] @ rotation.T, atol=3e-6)
                if v0.shape[1] > 2:
                    np.testing.assert_array_equal(v0[:, 2:], v1[:, 2:])


class ScenarioEditorTests(_TreeAssertions, unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="autoscene-editor-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.raw = _scenario()
        self.file = self.source / "sd_source.pkl"
        _dump(self.file, self.raw)
        # Deliberately stale summary must not leak into the exported sidecar.
        _dump(self.source / "dataset_summary.pkl", {self.file.name: {"stale": True}})
        self.dataset = ScenarioDataset(self.source)
        self.output = self.root / "variant"

    def test_rigid_trajectory_ego_shapes_dtypes_and_immutable_source(self):
        for dtype in (np.float32, np.float64):
            for columns in (2, 3):
                for singleton in (False, True):
                    with self.subTest(dtype=dtype, columns=columns, singleton=singleton):
                        raw = _scenario(dtype, columns, singleton)
                        _dump(self.file, raw)
                        before_bytes = self.file.read_bytes()
                        scene = self.dataset.load_scene(0)
                        poses = {o.track_id: (o.x + 9.5, o.y - 2, o.heading + 2.9)
                                 for o in [scene.ego] + [o for o in scene.objects if o.track_id == "car"]}
                        unchanged_poses = copy.deepcopy(poses)
                        destination = self.root / ("variant_%s_%s_%s" % (np.dtype(dtype), columns, singleton))
                        path = export_scene_variant(self.dataset, 0, poses, destination)
                        self.assertEqual(self.file.read_bytes(), before_bytes)
                        self.assertEqual(poses, unchanged_poses)
                        self.assertVariant(raw, _read(path), poses, scene.current_time_index)
                        self.assertEqual(set(p.name for p in destination.iterdir()),
                                         {Path(path).name, "dataset_summary.pkl", "dataset_mapping.pkl"})
                        self.assertEqual(_read(destination / "dataset_mapping.pkl"), {Path(path).name: ""})
                        summary = _read(destination / "dataset_summary.pkl")
                        self.assertTreeEqual(summary, {Path(path).name: editor._summary_value(raw["metadata"])})
                        self.assertEqual(ScenarioDataset(destination).count, 1)
                        loaded = ScenarioDataset(path).load_scene(0)
                        np.testing.assert_allclose([loaded.ego.x, loaded.ego.y], poses["ego"][:2], atol=2e-5)
                        np.testing.assert_allclose([loaded.runtime_origin_x, loaded.runtime_origin_y],
                                                   _read(path)["tracks"]["ego"]["state"]["position"][0, :2])

    def test_empty_edits_are_an_exact_copy(self):
        path = export_scene_variant(self.dataset, 0, {}, self.output)
        self.assertTreeEqual(self.raw, _read(path))

    def test_list_tuple_states_and_optional_velocity(self):
        raw = _scenario()
        state = raw["tracks"]["ego"]["state"]
        for key in ("position", "heading", "valid"):
            state[key] = tuple(tuple(v) if isinstance(v, list) else v for v in state[key].tolist())
        del state["velocity"]
        _dump(self.file, raw)
        poses = {"ego": (100., -20., -3.0)}
        path = export_scene_variant(self.dataset, 0, poses, self.output)
        self.assertVariant(raw, _read(path), poses, 3)
        self.assertIsInstance(_read(path)["tracks"]["ego"]["state"]["position"], tuple)

    def test_short_recording_uses_the_displayed_anchor(self):
        del self.raw["metadata"]["current_time_index"]
        _dump(self.file, self.raw)
        self.assertEqual(self.dataset.load_scene(0).current_time_index, 6)
        poses = {"ego": (30, 40, 1.1)}
        path = export_scene_variant(self.dataset, 0, poses, self.output)
        self.assertVariant(self.raw, _read(path), poses, 6)

    def test_bad_edits_fail_before_creating_output(self):
        edits = [{"missing": (1, 2, 3)}, {"ego": (1, 2)}, {"ego": (1, 2, float("nan"))},
                 {"ego": (float("inf"), 2, 3)}, {"ego": "bad"}, {0: (1, 2, 3)}, []]
        for poses in edits:
            with self.subTest(poses=poses), self.assertRaises(ScenarioExportError):
                export_scene_variant(self.dataset, 0, poses, self.output)
            self.assertFalse(self.output.exists())

    def test_invalid_anchor_malformed_state_and_integer_arrays_are_refused(self):
        for problem in ("invalid", "nan", "shape", "integer", "valid_shape", "valid_nan", "valid_fraction"):
            with self.subTest(problem=problem):
                raw = _scenario()
                state = raw["tracks"]["ego"]["state"]
                if problem == "invalid":
                    state["valid"][3] = False
                elif problem == "nan":
                    state["heading"][4] = float("nan")
                elif problem == "shape":
                    state["velocity"] = np.ones((7, 1))
                elif problem == "integer":
                    state["position"] = state["position"].astype(np.int32)
                elif problem == "valid_shape":
                    state["valid"] = np.ones((7, 2))
                else:
                    state["valid"] = np.ones(7)
                    state["valid"][0] = np.nan if problem == "valid_nan" else 0.1
                _dump(self.file, raw)
                with self.assertRaises(ScenarioExportError):
                    export_scene_variant(self.dataset, 0, {"ego": (1, 2, 3)}, self.output)
                self.assertFalse(self.output.exists())

    def test_existing_directory_file_symlink_and_missing_parent_are_refused(self):
        directory = self.root / "exists"
        directory.mkdir()
        sentinel = directory / "sentinel"
        sentinel.write_bytes(b"keep")
        link = self.root / "link"
        link.symlink_to(directory, target_is_directory=True)
        dangling = self.root / "dangling"
        dangling.symlink_to(self.root / "missing")
        for target in (directory, self.file, link, dangling, self.root / "missing" / "nested"):
            with self.subTest(target=target), self.assertRaises(ScenarioExportError):
                export_scene_variant(self.dataset, 0, {}, target)
        self.assertEqual(sentinel.read_bytes(), b"keep")
        self.assertTrue(link.is_symlink())
        self.assertTrue(dangling.is_symlink())
        self.assertTreeEqual(_read(self.file), self.raw)

    def test_write_failures_clean_each_own_partial_file(self):
        real_dump = pickle.dump
        for failure_at in (1, 2, 3):
            calls = []

            def failing_dump(payload, stream, **kwargs):
                calls.append(stream.name)
                if len(calls) == failure_at:
                    stream.write(b"partial")
                    raise OSError("injected disk full")
                return real_dump(payload, stream, **kwargs)

            with self.subTest(failure_at=failure_at), patch.object(editor.pickle, "dump", side_effect=failing_dump):
                with self.assertRaisesRegex(ScenarioExportError, "injected disk full"):
                    export_scene_variant(self.dataset, 0, {}, self.output)
            self.assertFalse(self.output.exists())
            self.assertTreeEqual(_read(self.file), self.raw)

    def test_failed_cleanup_preserves_concurrently_added_file(self):
        def fail(payload, stream, **kwargs):
            (self.output / "someone_else").write_bytes(b"keep")
            raise OSError("injected failure")

        with patch.object(editor.pickle, "dump", side_effect=fail), self.assertRaises(ScenarioExportError):
            export_scene_variant(self.dataset, 0, {}, self.output)
        self.assertEqual([p.name for p in self.output.iterdir()], ["someone_else"])
        self.assertEqual((self.output / "someone_else").read_bytes(), b"keep")

    def test_target_creation_race_never_removes_competing_directory(self):
        mkdir = os.mkdir

        def race(path):
            mkdir(path)
            (Path(path) / "someone_else").write_bytes(b"keep")
            raise FileExistsError(path)

        with patch.object(editor.os, "mkdir", side_effect=race), self.assertRaises(ScenarioExportError):
            export_scene_variant(self.dataset, 0, {}, self.output)
        self.assertEqual((self.output / "someone_else").read_bytes(), b"keep")

    def test_fsync_failure_and_interrupt_clean_partial_output(self):
        for error in (OSError("fsync failed"), KeyboardInterrupt()):
            with self.subTest(error=type(error)), patch.object(editor.os, "fsync", side_effect=error):
                with self.assertRaises(ScenarioExportError if isinstance(error, OSError) else KeyboardInterrupt):
                    export_scene_variant(self.dataset, 0, {}, self.output)
            self.assertFalse(self.output.exists())

    def test_cached_source_is_not_mutated_even_when_later_edit_fails(self):
        before = copy.deepcopy(self.raw)
        with patch.object(self.dataset, "load_raw", return_value=self.raw), self.assertRaises(ScenarioExportError):
            export_scene_variant(self.dataset, 0, {"ego": (100, 200, 2), "unknown": (1, 2, 3)}, self.output)
        self.assertTreeEqual(before, self.raw)
        self.assertFalse(self.output.exists())

    def test_shared_state_alias_does_not_edit_another_track(self):
        self.raw["tracks"]["other"]["state"] = self.raw["tracks"]["ego"]["state"]
        _dump(self.file, self.raw)
        poses = {"ego": (50, 70, 0.3)}
        path = export_scene_variant(self.dataset, 0, poses, self.output)
        self.assertVariant(self.raw, _read(path), poses, 3)

    def test_float32_overflow_is_refused_without_output(self):
        _dump(self.file, _scenario(np.float32))
        with self.assertRaises(ScenarioExportError):
            export_scene_variant(self.dataset, 0, {"ego": (1e100, 0, 0)}, self.output)
        self.assertFalse(self.output.exists())


class ScenarioLoaderTests(_TreeAssertions, unittest.TestCase):
    def test_current_validity_no_ghosts_and_no_nan_frame_compaction(self):
        raw = _scenario(singleton=True)
        state = raw["tracks"]["other"]["state"]
        state["valid"][:4] = False  # Future-only track must not appear today.
        raw["tracks"]["past"] = copy.deepcopy(raw["tracks"]["other"])
        raw["tracks"]["past"]["state"]["valid"][:] = False
        raw["tracks"]["past"]["state"]["valid"][0] = True
        scene = _scene_from_raw(raw, "fixture", 2)
        self.assertEqual([obj.track_id for obj in scene.objects], ["car"])
        car = scene.objects[0]
        np.testing.assert_equal([car.x, car.y], raw["tracks"]["car"]["state"]["position"][3, :2])
        self.assertEqual(len(car.trail), 2)  # Do not bridge the invalid frame 1.
        self.assertAlmostEqual(car.length, 5.2)
        self.assertAlmostEqual(car.width, 2.3)

    def test_numpy_size_array_is_not_used_as_a_boolean(self):
        raw = _scenario()
        state = raw["tracks"]["ego"]["state"]
        del state["length"], state["width"]
        state["size"] = np.tile([6., 3., 2.], (7, 1))
        scene = _scene_from_raw(raw, "fixture", 0)
        self.assertEqual((scene.ego.length, scene.ego.width), (6., 3.))

    def test_numeric_zero_ego_id_and_invalid_ego_not_replaced(self):
        raw = _scenario()
        raw["tracks"]["0"] = raw["tracks"].pop("ego")
        raw["metadata"]["sdc_id"] = 0
        scene = _scene_from_raw(raw, "fixture", 0)
        self.assertEqual(scene.ego.track_id, "0")
        raw["tracks"]["0"]["state"]["valid"][3] = False
        with self.assertRaises(DatasetLoadError):
            _scene_from_raw(raw, "fixture", 0)

    def test_single_file_does_not_unpickle_metadata_or_siblings(self):
        with tempfile.TemporaryDirectory(prefix="autoscene-loader-test-") as tmp:
            root = Path(tmp)
            file = root / "chosen.pkl"
            _dump(file, _scenario())
            (root / "dataset_summary.pkl").write_bytes(b"not a pickle")
            (root / "unrelated.pkl").write_bytes(b"not a pickle")
            dataset = ScenarioDataset(file)
            self.assertEqual(dataset.count, 1)
            self.assertEqual(dataset.load_scene(0).ego.track_id, "ego")
            with self.assertRaises(DatasetLoadError):
                ScenarioDataset(root / "dataset_summary.pkl")

    def test_summary_order_mapping_and_missing_entries_do_not_shift_indices(self):
        with tempfile.TemporaryDirectory(prefix="autoscene-loader-test-") as tmp:
            root = Path(tmp)
            (root / "nested").mkdir()
            names = ["sd_z.pkl", "7.pkl", "sd_a.pkl"]
            for name in names:
                _dump(root / "nested" / name, _scenario())
            _dump(root / "dataset_summary.pkl", dict.fromkeys(names, {}))
            _dump(root / "dataset_mapping.pkl", dict.fromkeys(names, "nested"))
            _dump(root / "sd_unlisted.pkl", _scenario())
            dataset = ScenarioDataset(root)
            self.assertEqual([Path(f).name for f in dataset.files], names)
            (root / "nested" / names[0]).unlink()
            with self.assertRaises(DatasetLoadError):
                ScenarioDataset(root)

    def test_mapping_cannot_escape_the_selected_dataset(self):
        with tempfile.TemporaryDirectory(prefix="autoscene-loader-test-") as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            _dump(root / "sd_outside.pkl", _scenario())
            _dump(source / "dataset_summary.pkl", {"sd_outside.pkl": {}})
            _dump(source / "dataset_mapping.pkl", {"sd_outside.pkl": ".."})
            with self.assertRaises(DatasetLoadError):
                ScenarioDataset(source)
            _dump(source / "dataset_mapping.pkl", {"sd_outside.pkl": ""})
            (source / "sd_outside.pkl").symlink_to(root / "sd_outside.pkl")
            with self.assertRaises(DatasetLoadError):
                ScenarioDataset(source)

    def test_summary_symlink_cannot_load_outside_selected_dataset(self):
        with tempfile.TemporaryDirectory(prefix="autoscene-loader-test-") as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            (root / "outside.pkl").write_bytes(b"must not unpickle")
            (source / "dataset_summary.pkl").symlink_to(root / "outside.pkl")
            with patch("autoscene_gui.scenario_loader.pickle.load") as read:
                with self.assertRaises(DatasetLoadError):
                    ScenarioDataset(source)
                read.assert_not_called()

    def test_scan_includes_numeric_names_but_not_unrelated_pickles(self):
        with tempfile.TemporaryDirectory(prefix="autoscene-loader-test-[brackets]-") as tmp:
            root = Path(tmp)
            for name in ("sd_z.pkl", "7.pkl"):
                _dump(root / name, _scenario())
            (root / "unrelated.pkl").write_bytes(b"do not load")
            dataset = ScenarioDataset(root)
            self.assertEqual([Path(f).name for f in dataset.files], ["7.pkl", "sd_z.pkl"])


def _backend_contract(backend_root):
    """Execute the real, local SD module without importing MetaDrive's renderer.

    Package shells bypass only __init__.py (which starts simulator imports).
    Math/type/ScenarioDescription source is unmodified. The two real dataset
    readers are extracted verbatim from utils.py, whose other functions import
    renderer/vehicle classes irrelevant to serialization.
    """
    package_root = backend_root / "metadrive"
    packages = {}
    for name, directory in (("metadrive", package_root), ("metadrive.utils", package_root / "utils"),
                            ("metadrive.scenario", package_root / "scenario")):
        module = types.ModuleType(name)
        module.__path__ = [str(directory)]
        packages[name] = module
    with patch.dict(sys.modules, packages):
        loaded = {}
        for name, path in (("metadrive.type", package_root / "type.py"),
                           ("metadrive.utils.math", package_root / "utils/math.py"),
                           ("metadrive.scenario.scenario_description", package_root / "scenario/scenario_description.py")):
            module = types.ModuleType(name)
            module.__file__ = str(path)
            sys.modules[name] = module
            exec(compile(path.read_text(), str(path), "exec"), module.__dict__)
            loaded[name] = module
        sd = loaded["metadrive.scenario.scenario_description"].ScenarioDescription
        path = package_root / "scenario/utils.py"
        parsed = ast.parse(path.read_text(), filename=str(path))
        keep = {"read_dataset_summary", "read_scenario_data", "dict_recursive_remove_array_and_set"}
        tree = ast.Module(body=[node for node in parsed.body if isinstance(node, ast.FunctionDef)
                                and node.name in keep], type_ignores=[])
        namespace = {"SD": sd, "ScenarioDescription": sd, "os": os, "pathlib": __import__("pathlib"),
                     "pickle": pickle, "np": np, "logger": logging.getLogger("backend-contract")}
        exec(compile(tree, str(path), "exec"), namespace)
        return sd, namespace["read_dataset_summary"], namespace["read_scenario_data"]


@unittest.skipUnless(os.environ.get("AUTOSCENE_ORIGINAL_ROOT"), "set trusted AUTOSCENE_ORIGINAL_ROOT for bundled data")
class BundledScenarioTests(_TreeAssertions, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(os.environ["AUTOSCENE_ORIGINAL_ROOT"])
        cls.backend = cls.root / "AutoScene/src/autoscene_runtime/scripts/Autoscene_generator/metadrive"
        sd, summary_reader, scenario_reader = _backend_contract(cls.backend)
        cls.sd = sd
        cls.summary_reader = staticmethod(summary_reader)
        cls.scenario_reader = staticmethod(scenario_reader)

    def _check_real(self, folder, index):
        dataset = ScenarioDataset(folder)
        original = dataset.load_raw(index)
        scene = dataset.load_scene(index)
        before_hash = hashlib.sha256(Path(dataset.files[index]).read_bytes()).digest()
        objects = [scene.ego] + [o for o in scene.objects if o.obj_type == "VEHICLE"][:2]
        poses = {o.track_id: (o.x + 3.25, o.y - 4.5, o.heading + 0.7) for o in objects}
        with tempfile.TemporaryDirectory(prefix="autoscene-bundled-variant-") as tmp:
            output = Path(tmp) / "standalone"
            path = export_scene_variant(dataset, index, poses, output)
            summary, names, mapping = self.summary_reader(output)
            self.assertEqual(names, [Path(path).name])
            self.assertEqual(mapping, {Path(path).name: ""})
            variant = self.scenario_reader(path)
            self.sd.sanity_check(dict(variant), check_self_type=True)
            self.assertVariant(original, variant, poses, scene.current_time_index)
            self.assertTreeEqual(summary[Path(path).name], editor._summary_value(variant["metadata"]))
            reloaded = ScenarioDataset(output).load_scene(0)
            np.testing.assert_allclose([reloaded.ego.x, reloaded.ego.y], poses[scene.ego.track_id][:2])
            # The real backend's moving-distance metric must be invariant.
            for obj in objects:
                a = self.sd.get_object_summary(original["tracks"][obj.track_id], obj.track_id)
                b = self.sd.get_object_summary(variant["tracks"][obj.track_id], obj.track_id)
                self.assertAlmostEqual(a["moving_distance"], b["moving_distance"], places=6)
                self.assertEqual(a["valid_length"], b["valid_length"])
        self.assertEqual(hashlib.sha256(Path(dataset.files[index]).read_bytes()).digest(), before_hash)

    def test_real_osm(self):
        folder = self.backend / "plugins/examples/scenario_record/ni_76"
        for index in (0, 11, 21):
            with self.subTest(index=index):
                self._check_real(folder, index)

    def test_real_nuplan(self):
        folder = self.root / "AutoScene/data/pittsburgh_convert10000_0"
        for index in (0, 239, 477):
            with self.subTest(index=index):
                self._check_real(folder, index)

    def test_real_tjwest_3d_velocity(self):
        self._check_real(self.root / "AutoScene/data/tjwest260725intersection", 0)

    def test_real_summary_order_and_nuplan_visibility(self):
        folder = self.root / "AutoScene/data/pittsburgh_convert10000_0"
        dataset = ScenarioDataset(folder)
        _, names, _ = self.summary_reader(folder)
        self.assertEqual([Path(f).name for f in dataset.files], names)
        raw = dataset.load_raw(0)
        scene = dataset.load_scene(0)
        expected = {tid for tid, track in raw["tracks"].items()
                    if track["state"]["valid"][scene.current_time_index]}
        self.assertEqual({o.track_id for o in [scene.ego] + scene.objects}, expected)
        self.assertEqual(len(raw["tracks"]), 68)
        self.assertEqual(len(expected), 39)


if __name__ == "__main__":
    unittest.main()
