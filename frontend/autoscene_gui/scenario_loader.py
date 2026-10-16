"""数据集加载：把 tjwest 的 sd_*.pkl / dataset_summary.pkl 解析成 UI 需要的场景结构。

未加载数据集时用 DEMO_SCENE 画一张好看的示意路口，加载成功后换成真实地图与真实车辆位姿。
pkl 的字段名在不同导出版本间有差异，这里按 ScenarioNet / UniTraj 的常见键做兜底探测，
解析失败会把原因回报给 UI 而不是抛栈。
"""

from __future__ import annotations

import math
import os
import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class SceneObject:
    track_id: str
    x: float
    y: float
    heading: float
    speed: float
    obj_type: str = "VEHICLE"
    physical: bool = False          # 是否为实物运动平台车位
    length: float = 4.6
    width: float = 2.0
    trail: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class Scene:
    name: str
    ego: SceneObject
    objects: List[SceneObject] = field(default_factory=list)
    lanes: List[List[Tuple[float, float]]] = field(default_factory=list)      # 车道中心线
    boundaries: List[List[Tuple[float, float]]] = field(default_factory=list)  # 路缘 / 边界
    crosswalks: List[List[Tuple[float, float]]] = field(default_factory=list)
    is_demo: bool = False
    note: str = ""
    current_time_index: int = 0
    # ScenarioEnv 会把数据集第 0 帧 ego 的全局坐标平移到 (0, 0)。
    # UI 仍使用原始全局坐标绘图；与 ROS runtime 通信时用这个原点互转。
    runtime_origin_x: float = 0.0
    runtime_origin_y: float = 0.0

    @property
    def physical_slots(self) -> List[SceneObject]:
        return [o for o in self.objects if o.physical]


# --------------------------------------------------------------------------
# 示意场景（未加载数据集时显示）
# --------------------------------------------------------------------------
def _demo_scene() -> Scene:
    ego = SceneObject("ego", -2.0, -16.0, 0.0, 2.5, physical=False,
                      trail=[(-24, -16), (-18, -16), (-12, -16), (-6, -16), (-2, -16)])
    objs = [
        SceneObject("P1", 24.0, -12.0, math.pi, 4.2, physical=True,
                    trail=[(40, -12), (34, -12), (28, -12), (24, -12)]),
        SceneObject("P2", 14.0, -32.0, math.pi / 2, 3.0, physical=True,
                    trail=[(14, -46), (14, -40), (14, -36), (14, -32)]),
        SceneObject("B1", 8.0, 4.0, -math.pi / 2, 5.5, physical=False,
                    trail=[(8, 20), (8, 14), (8, 9), (8, 4)]),
        SceneObject("B2", -20.0, -8.0, 0.0, 6.1, physical=False),
    ]
    lanes: List[List[Tuple[float, float]]] = []
    # 东西向双向四车道 + 南北向双向四车道，中心线用于画虚线
    for y in (-20.0, -12.0, -4.0):
        lanes.append([(-60.0, y), (60.0, y)])
    for x in (2.0, 10.0, 18.0):
        lanes.append([(x, -60.0), (x, 40.0)])
    boundaries = [
        [(-60.0, -24.0), (60.0, -24.0)],
        [(-60.0, 0.0), (60.0, 0.0)],
        [(-2.0, -60.0), (-2.0, 40.0)],
        [(22.0, -60.0), (22.0, 40.0)],
    ]
    crosswalks = [
        [(-6.0, -24.0), (-6.0, 0.0)],
        [(26.0, -24.0), (26.0, 0.0)],
        [(-2.0, -28.0), (22.0, -28.0)],
        [(-2.0, 4.0), (22.0, 4.0)],
    ]
    return Scene("示意路口（未加载数据集）", ego, objs, lanes, boundaries, crosswalks,
                 is_demo=True, note="加载数据集后显示真实地图与真实车辆位姿")


DEMO_SCENE = _demo_scene


# --------------------------------------------------------------------------
# 真实数据集
# --------------------------------------------------------------------------
class DatasetLoadError(RuntimeError):
    pass


class ScenarioDataset:
    """用户明确选择的可信 ScenarioDescription 目录或单个 PKL。

    Pickle 可执行代码；不要用此接口探测不受信任的下载文件。
    """

    def __init__(self, path: str):
        self.path = os.path.abspath(os.fspath(path))
        self.files: List[str] = []
        self.summary: Any = None
        self._scan()

    def _scan(self) -> None:
        if os.path.isfile(self.path):
            if not self.path.endswith(".pkl") or os.path.basename(self.path) in {
                "dataset_summary.pkl", "dataset_mapping.pkl"
            }:
                raise DatasetLoadError("请选择场景 PKL，而不是 summary/mapping：%s" % self.path)
            # A single-file selection must not unpickle unrelated siblings.
            self.files = [self.path]
            return
        if not os.path.isdir(self.path):
            raise DatasetLoadError("目录不存在：%s" % self.path)

        # ScenarioNet 目录除了真正的 sd_*.pkl 场景文件，还可能包含
        # dataset_summary.pkl / dataset_mapping.pkl。旧版 UI 只排除了 summary，
        # 因而会把 dataset_mapping.pkl 当成 scenario_index=0 来解析。
        summary_path = os.path.join(self.path, "dataset_summary.pkl")
        mapping_path = os.path.join(self.path, "dataset_mapping.pkl")
        mapping: Dict[str, Any] = {}

        if os.path.isfile(summary_path):
            _check_dataset_member(self.path, summary_path)
            try:
                with open(summary_path, "rb") as fh:
                    self.summary = pickle.load(fh)
            except Exception as exc:  # noqa: BLE001
                raise DatasetLoadError("dataset_summary.pkl 解析失败：%s" % exc) from exc
            if not isinstance(self.summary, dict):
                raise DatasetLoadError("dataset_summary.pkl 必须是文件名到 metadata 的字典")

        if os.path.isfile(mapping_path):
            _check_dataset_member(self.path, mapping_path)
            try:
                with open(mapping_path, "rb") as fh:
                    loaded_mapping = pickle.load(fh)
                if not isinstance(loaded_mapping, dict):
                    raise ValueError("mapping 必须是字典")
                mapping = loaded_mapping
            except Exception as exc:  # noqa: BLE001
                raise DatasetLoadError("dataset_mapping.pkl 解析失败：%s" % exc) from exc

        # 优先沿用 dataset_summary.pkl 的键顺序。AutoScene/ScenarioNet 也是按
        # 数据集 summary 来确定场景数量/索引，这样 UI 的 scenario_index 才能
        # 与 runtime 的 env.reset(seed=scenario_index) 指向同一个场景。
        ordered: List[str] = []
        if isinstance(self.summary, dict):
            for filename in self.summary.keys():
                if not isinstance(filename, str) or not _is_scenario_filename(filename):
                    raise DatasetLoadError("summary 中的场景文件名无效：%s" % filename)
                if mapping and filename not in mapping:
                    raise DatasetLoadError("mapping 中缺少场景：%s" % filename)
                subdir = mapping.get(filename, "")
                if not isinstance(subdir, str) or os.path.isabs(subdir):
                    raise DatasetLoadError("mapping 必须使用数据集内的相对目录：%s" % filename)
                candidate = os.path.join(self.path, subdir, filename) if subdir else os.path.join(self.path, filename)
                _check_dataset_member(self.path, candidate)
                if not os.path.isfile(candidate):
                    # Never compact indices by silently skipping a missing entry.
                    raise DatasetLoadError("summary 中的场景文件不存在：%s" % candidate)
                ordered.append(candidate)

        # 没有 summary 时，只扫描 backend 认可的 sd_*.pkl / 数字.pkl。
        # 不探测任意其他 pickle；没有规范前缀的文件须由用户明确单独选择。
        if self.summary is None:
            with os.scandir(self.path) as entries:
                ordered = sorted(entry.path for entry in entries
                                 if _is_scenario_filename(entry.name) and entry.is_file())

        for candidate in ordered:
            _check_dataset_member(self.path, candidate)

        self.files = ordered
        if not self.files:
            raise DatasetLoadError("目录下没有场景 pkl：%s" % self.path)

    @property
    def count(self) -> int:
        return len(self.files)

    def scenario_name(self, index: int) -> str:
        return os.path.splitext(os.path.basename(self.files[index]))[0]

    def load_scene(self, index: int, physical_slots: int = 2) -> Scene:
        return _scene_from_raw(self.load_raw(index), self.scenario_name(index), physical_slots)

    def load_raw(self, index: int) -> Dict[str, Any]:
        """Fresh source data on every call; UI edits never mutate the source/cache."""
        if not (0 <= index < self.count):
            raise DatasetLoadError("scenario_index=%d 超出范围 0..%d" % (index, self.count - 1))
        if os.path.isdir(self.path):
            _check_dataset_member(self.path, self.files[index])
        try:
            with open(self.files[index], "rb") as fh:
                raw = pickle.load(fh)
        except Exception as exc:  # noqa: BLE001
            raise DatasetLoadError("场景 PKL 读取失败：%s" % exc) from exc
        if not isinstance(raw, dict):
            raise DatasetLoadError("场景 PKL 必须包含 ScenarioDescription 字典")
        return raw


def _is_scenario_filename(filename: str) -> bool:
    """Match ScenarioDescription.is_scenario_file, but disallow path components."""
    return (os.path.basename(filename) == filename and filename.endswith(".pkl")
            and (filename.startswith("sd_") or filename[:-4].isdigit()))


def _check_dataset_member(root: str, candidate: str) -> None:
    if os.path.commonpath([os.path.realpath(root), os.path.realpath(candidate)]) != os.path.realpath(root):
        raise DatasetLoadError("场景路径超出所选数据集目录：%s" % candidate)


# ---- pkl → Scene ---------------------------------------------------------
def _first(d: Any, *keys: str) -> Any:
    if not isinstance(d, dict):
        return None
    for k in keys:
        if k in d:
            return d[k]
    return None


def _or_default(value: Any, default: Any) -> Any:
    """只把 None 当缺失值，避免对 NumPy 数组做含糊的布尔判断。"""
    return default if value is None else value


def _xy(seq: Any) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    try:
        for p in seq:
            if p is None:
                continue
            if hasattr(p, "__len__") and len(p) >= 2:
                x, y = float(p[0]), float(p[1])
                if math.isfinite(x) and math.isfinite(y):
                    out.append((x, y))
    except TypeError:
        return []
    return out


def _track_state(track: Any) -> Optional[Dict[str, Any]]:
    state = _first(track, "state", "states")
    if not isinstance(state, dict):
        return None
    return state


def _last_valid_index(state: Dict[str, Any]) -> int:
    positions = _or_default(_first(state, "position"), [])
    return next((i for i in range(len(positions) - 1, -1, -1)
                 if _state_valid_at(state, i)), -1)


def _scalar(value: Any, default: float = 0.0) -> float:
    """Accept scalar or singleton-array samples, including NumPy 2.x (T, 1)."""
    try:
        if hasattr(value, "size"):
            value = value.item()
        elif isinstance(value, (list, tuple)) and len(value) == 1:
            value = value[0]
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError, OverflowError):
        return default


def _position_at(state: Dict[str, Any], index: int) -> Optional[Tuple[float, float]]:
    try:
        if index < 0:
            return None
        sample = state["position"][index]
        x, y = float(sample[0]), float(sample[1])
        return (x, y) if math.isfinite(x) and math.isfinite(y) else None
    except (KeyError, IndexError, TypeError, ValueError, OverflowError):
        return None


def _state_valid_at(state: Dict[str, Any], index: int) -> bool:
    if _position_at(state, index) is None:
        return False
    valid = state.get("valid")
    if valid is None:
        return True
    try:
        return _scalar(valid[index]) > 0.5
    except (IndexError, TypeError):
        return False


def _current_time_index(raw: Dict[str, Any]) -> int:
    """One shared display/export anchor; short recordings use their final frame."""
    metadata = _or_default(_first(raw, "metadata"), {})
    value = _first(metadata, "current_time_index")
    try:
        index = int(value) if value is not None else 20
    except (TypeError, ValueError, OverflowError):
        index = 20
    length = _first(raw, "length")
    if length is None:
        tracks = _or_default(_first(raw, "tracks", "agents", "objects"), {})
        length = max((len(state.get("position", [])) for track in tracks.values()
                      for state in [_track_state(track)] if state is not None), default=0)
    return max(0, min(index, int(length) - 1))


def _obj_from_track(track_id: str, track: Any,
                    current_time_index: Optional[int] = None) -> Optional[SceneObject]:
    state = _track_state(track)
    if state is None:
        return None
    i = current_time_index if current_time_index is not None else _last_valid_index(state)
    if not _state_valid_at(state, i):
        return None
    # Keep original frame indices; filtering positions first shifts NaN/gap frames.
    pos = _position_at(state, i)
    heading_seq = _or_default(_first(state, "heading"), [])
    vel_seq = _or_default(_first(state, "velocity"), [])
    try:
        heading = _scalar(heading_seq[i], float("nan"))
    except Exception:  # noqa: BLE001
        heading = 0.0
    if not math.isfinite(heading):
        return None
    speed = 0.0
    try:
        v = vel_seq[i]
        if hasattr(v, "__len__") and len(v) >= 2:
            speed = math.hypot(float(v[0]), float(v[1]))
        else:
            speed = abs(float(v))
    except Exception:  # noqa: BLE001
        speed = 0.0
    obj_type = str(_first(track, "type") or "VEHICLE")
    size = _or_default(_first(state, "size"), _first(track, "size"))
    length, width = 4.6, 2.0
    try:
        if size is not None:
            s = size[i] if hasattr(size[0], "__len__") else size
            length, width = float(s[0]), float(s[1])
    except Exception:  # noqa: BLE001
        pass
    # ScenarioDescription/nuPlan stores dimensions separately, often as (T, 1).
    for key in ("length", "width"):
        seq = state.get(key)
        if seq is not None:
            try:
                if key == "length":
                    length = _scalar(seq[i], length)
                else:
                    width = _scalar(seq[i], width)
            except (IndexError, TypeError):
                pass
    trail = []
    for frame in range(i, max(-1, i - 21), -1):
        if not _state_valid_at(state, frame):
            break
        trail.append(_position_at(state, frame))
    trail.reverse()
    return SceneObject(track_id, pos[0], pos[1], heading, speed if math.isfinite(speed) else 0.0,
                       obj_type=obj_type, length=max(length, 1.0), width=max(width, 0.8),
                       trail=trail)


def nearest_polyline_pose(polylines: List[List[Tuple[float, float]]],
                          x: float, y: float,
                          heading_hint: Optional[float] = None) -> Optional[Tuple[float, float, float, float]]:
    """投影到最近折线段，返回 x/y/切向 heading/距离。

    车道线本身没有行驶方向时，用 heading_hint 在两个相反切向中选更接近的一侧。
    """
    best: Optional[Tuple[float, float, float, float]] = None
    best_d2 = float("inf")
    for line in polylines:
        for a, b in zip(line, line[1:]):
            ax, ay = float(a[0]), float(a[1])
            bx, by = float(b[0]), float(b[1])
            dx, dy = bx - ax, by - ay
            length2 = dx * dx + dy * dy
            if length2 <= 1e-12:
                continue
            t = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / length2))
            px, py = ax + t * dx, ay + t * dy
            d2 = (x - px) ** 2 + (y - py) ** 2
            if d2 >= best_d2:
                continue
            heading = math.atan2(dy, dx)
            if heading_hint is not None and math.cos(heading - heading_hint) < 0.0:
                heading += math.pi
            heading = math.atan2(math.sin(heading), math.cos(heading))
            best_d2 = d2
            best = (px, py, heading, math.sqrt(d2))
    return best


def _map_polylines(raw: Any) -> Tuple[List[List[Tuple[float, float]]],
                                     List[List[Tuple[float, float]]],
                                     List[List[Tuple[float, float]]]]:
    feats = _first(raw, "map_features", "map", "map_feature")
    lanes: List[List[Tuple[float, float]]] = []
    bounds: List[List[Tuple[float, float]]] = []
    walks: List[List[Tuple[float, float]]] = []
    if not isinstance(feats, dict):
        return lanes, bounds, walks
    for _fid, feat in feats.items():
        if not isinstance(feat, dict):
            continue
        ftype = str(feat.get("type", "")).upper()
        line = _xy(_or_default(_first(feat, "polyline", "polygon", "position"), []))
        if len(line) < 2:
            continue
        if "CROSSWALK" in ftype:
            walks.append(line)
        elif "LANE" in ftype and "BOUNDARY" not in ftype:
            lanes.append(line)
        else:
            bounds.append(line)
    return lanes, bounds, walks


def _scene_from_raw(raw: Any, name: str, physical_slots: int) -> Scene:
    tracks = _or_default(_first(raw, "tracks", "agents", "objects"), {})
    metadata = _or_default(_first(raw, "metadata"), {})
    ego_id = str(_or_default(_first(metadata, "sdc_id"),
                            _or_default(_first(raw, "sdc_id", "ego_id", "sdc_track_index"), "ego")))
    current_time_index = _current_time_index(raw)
    ego_obj: Optional[SceneObject] = None
    runtime_origin = (0.0, 0.0)
    others: List[SceneObject] = []

    if isinstance(tracks, dict):
        for tid, track in tracks.items():
            is_ego = str(tid) == ego_id or str(_first(track, "is_sdc")).lower() == "true"
            obj = _obj_from_track(str(tid), track, current_time_index)
            if obj is None:
                if is_ego:
                    raise DatasetLoadError("自车 track %s 在当前帧 %d 无有效位姿" % (tid, current_time_index))
                continue
            if is_ego:
                ego_obj = obj
                state = _track_state(track)
                origin = _position_at(state, 0)
                if origin is not None:
                    runtime_origin = origin
            else:
                others.append(obj)

    if ego_obj is None:
        if _first(metadata, "sdc_id") is not None:
            raise DatasetLoadError("pkl 中找不到 metadata.sdc_id 指定的自车：%s" % ego_id)
        if others:
            ego_obj = others.pop(0)
        else:
            raise DatasetLoadError("pkl 中没能解析出任何车辆轨迹（字段结构与预期不符）")

    # 距自车最近的若干辆车视作可用的实物平台车位
    others.sort(key=lambda o: math.hypot(o.x - ego_obj.x, o.y - ego_obj.y))
    vehicle_slots = [o for o in others
                     if o.obj_type.upper().startswith("VEH") or o.obj_type == ""]
    for o in vehicle_slots[: max(0, physical_slots)]:
        o.physical = True

    lanes, bounds, walks = _map_polylines(raw)
    note = "" if lanes or bounds else "pkl 中未找到 map_features，地图仅显示轨迹走廊"
    return Scene(name, ego_obj, others, lanes, bounds, walks, is_demo=False, note=note,
                 current_time_index=current_time_index,
                 runtime_origin_x=float(runtime_origin[0]),
                 runtime_origin_y=float(runtime_origin[1]))
