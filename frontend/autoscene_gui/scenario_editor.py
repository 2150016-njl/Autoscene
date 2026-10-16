"""CPU-only, non-destructive export of trusted ScenarioDescription variants.

No simulator/Qt imports are needed. Poses are absolute dataset coordinates, in
metres/radians, keyed by the original track ID (including the ego's actual ID).
Only position XY, heading and velocity XY at valid samples are changed. Other
state channels, invalid samples, map, IDs, timestamps and metadata stay intact.
"""

from __future__ import annotations

import copy
import math
import os
import pickle
import uuid
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from .scenario_loader import DatasetLoadError, ScenarioDataset, _current_time_index


class ScenarioExportError(DatasetLoadError):
    """The source/edit is unsuitable, or a new output could not be committed."""


def _validate_scenario(raw: Dict[str, Any]) -> int:
    required = {"id", "version", "length", "metadata", "tracks", "map_features", "dynamic_map_states"}
    if not required.issubset(raw):
        raise ScenarioExportError("不是完整的 ScenarioDescription；缺少：%s" % sorted(required - raw.keys()))
    length = raw["length"]
    if isinstance(length, bool) or not isinstance(length, (int, np.integer)) or length < 1:
        raise ScenarioExportError("场景 length 必须是正整数")
    for key in ("metadata", "tracks", "map_features", "dynamic_map_states"):
        if not isinstance(raw[key], dict):
            raise ScenarioExportError("ScenarioDescription.%s 必须是字典" % key)
    metadata = raw["metadata"]
    if not {"ts", "coordinate", "metadrive_processed"}.issubset(metadata):
        raise ScenarioExportError("metadata 缺少 ts / coordinate / metadrive_processed")
    if np.asarray(metadata["ts"]).shape != (length,):
        raise ScenarioExportError("metadata.ts 的维度与 length 不一致")
    for group in ("tracks", "dynamic_map_states"):
        for tid, track in raw[group].items():
            if (not isinstance(track, dict) or not {"state", "type", "metadata"}.issubset(track)
                    or not isinstance(track["state"], dict) or not isinstance(track["metadata"], dict)):
                raise ScenarioExportError("track %s 缺少 state/type/metadata" % tid)
            if track["metadata"].get("object_id") != tid:
                raise ScenarioExportError("track %s 的 object_id 不匹配" % tid)
            for key, values in track["state"].items():
                if not isinstance(values, (np.ndarray, list, tuple)) or np.ndim(values) not in (1, 2):
                    raise ScenarioExportError("track %s.%s 的状态维度无效" % (tid, key))
                if len(values) != length:
                    raise ScenarioExportError("track %s.%s 的帧数与 length 不一致" % (tid, key))
            if group == "tracks" and not {"position", "heading"}.issubset(track["state"]):
                raise ScenarioExportError("track %s 缺少 position/heading" % tid)
    return int(length)


def _array(state: Dict[str, Any], key: str, length: int, tid: str) -> np.ndarray:
    array = np.asarray(state[key])
    if array.dtype.kind not in "fiu":
        raise ScenarioExportError("track %s.%s 必须是实数数组" % (tid, key))
    if key == "heading":
        shape_ok = array.shape in ((length,), (length, 1))
    else:
        shape_ok = array.ndim == 2 and array.shape[0] == length and array.shape[1] in (2, 3)
    if not shape_ok:
        raise ScenarioExportError("track %s.%s 的状态维度不支持：%s" % (tid, key, array.shape))
    # Silently assigning fractional transforms into integer arrays truncates the
    # trajectory. Refuse rather than change the source dtype/schema.
    if isinstance(state[key], np.ndarray) and array.dtype.kind != "f":
        raise ScenarioExportError("track %s.%s 需要浮点数组才能无损编辑" % (tid, key))
    return array


def _like_sample(template: Any, value: Any) -> Any:
    if isinstance(template, (list, tuple)):
        return type(template)(_like_sample(t, v) for t, v in zip(template, value))
    return value


def _replace_valid(original: Any, values: np.ndarray, valid: np.ndarray) -> Any:
    """Copy only valid entries, retaining shape/dtype and invalid sample bits."""
    if isinstance(original, np.ndarray):
        result = original.copy()
        with np.errstate(over="ignore", invalid="ignore"):
            result[valid] = values
        if not np.isfinite(result[valid]).all():
            raise ScenarioExportError("编辑结果超出原始数组精度/范围")
        return result
    result = list(copy.deepcopy(original))
    for index, value in zip(np.flatnonzero(valid), values.tolist()):
        result[index] = _like_sample(original[index], value)
    return tuple(result) if isinstance(original, tuple) else result


def _transform_track(track: Dict[str, Any], tid: str, pose: Any, frame: int, length: int) -> None:
    try:
        target = np.asarray(pose, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ScenarioExportError("track %s 的目标位姿必须是 (x, y, heading)" % tid) from exc
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ScenarioExportError("track %s 的目标位姿必须包含三个有限数值" % tid)
    state = track["state"]
    positions = _array(state, "position", length, tid)
    headings = _array(state, "heading", length, tid)
    flags = np.asarray(state.get("valid", np.ones(length, dtype=bool)))
    if flags.shape not in ((length,), (length, 1)) or flags.dtype.kind not in "bifu":
        raise ScenarioExportError("track %s 的 valid 必须是 (T,) 或 (T, 1) 数组" % tid)
    if not np.isfinite(flags).all() or not np.isin(flags, (0, 1)).all():
        raise ScenarioExportError("track %s 的 valid 必须由 0/1 组成" % tid)
    valid = flags.reshape(length).astype(bool)
    if not valid[frame]:
        raise ScenarioExportError("track %s 在显示帧 %d 无效，不能编辑历史/未来的幽灵位姿" % (tid, frame))
    if not np.isfinite(positions[valid]).all() or not np.isfinite(headings[valid]).all():
        raise ScenarioExportError("track %s 的有效轨迹含有非有限数值" % tid)

    source_heading = float(headings.reshape(length)[frame])
    # Reduce both terms before subtraction so even large finite angle inputs do
    # not overflow. This is a single rigid transform anchored at the UI frame.
    delta = math.remainder(float(target[2]), 2 * math.pi) - math.remainder(source_heading, 2 * math.pi)
    c, s = math.cos(delta), math.sin(delta)
    rotation = np.array([[c, -s], [s, c]])
    with np.errstate(over="ignore", invalid="ignore"):
        moved = positions[valid].astype(np.float64, copy=True)
        moved[:, :2] = (moved[:, :2] - positions[frame, :2]) @ rotation.T + target[:2]
        turned = np.remainder(headings[valid].astype(np.float64), 2 * math.pi) + delta
        turned = (turned + math.pi) % (2 * math.pi) - math.pi
    if not np.isfinite(moved).all() or not np.isfinite(turned).all():
        raise ScenarioExportError("track %s 的编辑结果超出数值范围" % tid)
    state["position"] = _replace_valid(state["position"], moved, valid)
    state["heading"] = _replace_valid(state["heading"], turned, valid)
    if "velocity" in state:
        velocities = _array(state, "velocity", length, tid)
        if not np.isfinite(velocities[valid]).all():
            raise ScenarioExportError("track %s 的有效 velocity 含有非有限数值" % tid)
        rotated = velocities[valid].astype(np.float64, copy=True)
        with np.errstate(over="ignore", invalid="ignore"):
            rotated[:, :2] = rotated[:, :2] @ rotation.T
        if not np.isfinite(rotated).all():
            raise ScenarioExportError("track %s 的 velocity 编辑结果超出数值范围" % tid)
        state["velocity"] = _replace_valid(state["velocity"], rotated, valid)


def _summary_value(value: Any) -> Any:
    """Mirror backend save_dataset's array/set normalization in the sidecar."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _summary_value(item) for key, item in value.items()}
    if isinstance(value, set):
        # Set iteration order can change after pickling; stabilize the sidecar
        # while leaving the original metadata's set type/content untouched.
        return tuple(_summary_value(item) for item in sorted(value, key=repr))
    if isinstance(value, tuple):
        return tuple(_summary_value(item) for item in value)
    if isinstance(value, list):
        return [_summary_value(item) for item in value]
    return copy.deepcopy(value)


def _identity(stat: os.stat_result) -> Tuple[int, int]:
    return stat.st_dev, stat.st_ino


def _write_new_dataset(output: str, files: Dict[str, Any]) -> None:
    # Exclusive directory reservation also catches a target created after the
    # initial validation. No replace(), recursive deletion or exist_ok=True.
    os.mkdir(output)
    directory_id = _identity(os.lstat(output))
    owned = []
    try:
        for filename, payload in files.items():
            path = os.path.join(output, filename)
            with open(path, "xb") as stream:
                owned.append((path, _identity(os.fstat(stream.fileno()))))
                # Protocol 4 remains readable by the Python 3.8 ROS runtime.
                pickle.dump(payload, stream, protocol=4)
                stream.flush()
                os.fsync(stream.fileno())
    except BaseException:
        # Clean only files created by this call, never pre-existing/concurrent
        # contents or a target directory substituted by another process.
        try:
            if _identity(os.lstat(output)) == directory_id:
                for path, file_id in reversed(owned):
                    try:
                        if _identity(os.lstat(path)) == file_id:
                            os.unlink(path)
                    except OSError:
                        pass
                try:
                    os.rmdir(output)  # Fails safely if someone added a file.
                except OSError:
                    pass
        except OSError:
            pass
        raise


def export_scene_variant(dataset: ScenarioDataset, index: int,
                         poses: Mapping[str, Tuple[float, float, float]],
                         output_dir: str) -> str:
    """Export one scene to a NEW standalone dataset directory; return PKL path.

    Example: export_scene_variant(ds, 0, {scene.ego.track_id: (x, y, yaw)}, dest)
    The anchor is the same current_time_index used by ds.load_scene(index).
    Empty poses exports an unchanged standalone copy. Unknown tracks, invalid
    current samples, non-finite poses and existing targets are refused. Parent
    directories must already exist. The source dataset and caller's poses are
    never mutated; each call reloads and deep-copies its source.

    The summary maps the NEW filename to exported metadata, not to the source
    dataset's potentially stale sidecar. Existing object/number summaries are
    rigid-transform invariants and are kept along with all other metadata.
    The mapping uses an empty relative directory, as read_dataset_summary
    expects. Summary is written last; failed writes clean only our own partial
    files. This is exception-safe, not a crash-proof filesystem transaction.

    Raises ScenarioExportError (also a DatasetLoadError) on ordinary failures.
    Only load trusted, user-selected PKLs: Python pickle can execute code.
    """
    try:
        output = os.path.abspath(os.fspath(output_dir))
        if os.path.lexists(output):
            raise ScenarioExportError("导出目录已存在；不会覆盖：%s" % output)
        if not os.path.isdir(os.path.dirname(output)):
            raise ScenarioExportError("导出目录的父目录不存在：%s" % output)
        if not isinstance(poses, Mapping):
            raise ScenarioExportError("poses 必须是 track_id 到 (x, y, heading) 的字典")
        raw = copy.deepcopy(dataset.load_raw(index))
        length = _validate_scenario(raw)
        frame = _current_time_index(raw)
        tracks = {}
        for key in raw["tracks"]:
            if str(key) in tracks:
                raise ScenarioExportError("场景中存在字符串表示重复的 track ID：%s" % key)
            tracks[str(key)] = key
        for tid, pose in poses.items():
            if not isinstance(tid, str) or tid not in tracks:
                raise ScenarioExportError("找不到待编辑 track_id：%s" % tid)
            # Pickle/deepcopy preserve aliases. Detach the edited track so an
            # unusual source sharing state dicts cannot alter an unedited one.
            track = copy.deepcopy(raw["tracks"][tracks[tid]])
            _transform_track(track, tid, pose, frame, length)
            raw["tracks"][tracks[tid]] = track

        filename = "sd_ui_variant_%s.pkl" % uuid.uuid4().hex
        _write_new_dataset(output, {
            filename: raw,
            "dataset_mapping.pkl": {filename: ""},
            "dataset_summary.pkl": {filename: _summary_value(raw["metadata"])},
        })
        return os.path.join(output, filename)
    except ScenarioExportError:
        raise
    except Exception as exc:
        raise ScenarioExportError("场景导出失败：%s" % exc) from exc
