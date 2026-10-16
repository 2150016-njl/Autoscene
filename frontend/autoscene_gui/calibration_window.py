"""Single-point physical G ↔ MetaDrive runtime A calibration and frame-0 alignment.

Only G and the *current* ScenarioEnv A coordinate frame are shown to users.
For cross-scenario reuse, YAML records the A transformation at the calibration
scenario plus its frame-0 reference offset. Runtime transparently compensates
for the next scenario's offset. No third editable/displayed coordinate frame.
"""
from __future__ import annotations

import copy
import datetime
import shutil
import importlib.util
import json
import math
import os
import socket
import struct
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PyQt5.QtCore import QPointF, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPathItem,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)


ADS_PAYLOAD_SIZE = 176


def _read_f64(data: bytes, offset: int) -> float:
    return struct.unpack_from("<d", data, offset)[0]


def _read_f32(data: bytes, offset: int) -> float:
    return struct.unpack_from("<f", data, offset)[0]


def _read_i16(data: bytes, offset: int) -> int:
    return struct.unpack_from("<h", data, offset)[0]


def _read_u16(data: bytes, offset: int) -> int:
    return struct.unpack_from("<H", data, offset)[0]


def _read_u8(data: bytes, offset: int) -> int:
    return struct.unpack_from("<B", data, offset)[0]


def _read_i32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<i", data, offset)[0]


def _read_u32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def decode_ads_payload176(data: bytes) -> dict:
    """Decode the fields used by the calibration/alignment UI."""
    if len(data) != ADS_PAYLOAD_SIZE:
        raise ValueError("ADS UDP payload must be exactly 176 bytes, got %d" % len(data))
    return {
        "latitude_deg": _read_f64(data, 0),
        "longitude_deg": _read_f64(data, 8),
        "ads_id": _read_u32(data, 16),
        "gps_time_raw_ms": _read_i32(data, 20),
        "gps_week": _read_i32(data, 24),
        "posi_rms_m": _read_f32(data, 56),
        "lat_dev_m": _read_f32(data, 60),
        "heading_deg": _read_u16(data, 92) * 0.01,
        "speed_kmh": _read_i16(data, 94) * 0.01,
        "speed_e_kmh": _read_i16(data, 96) * 0.01,
        "speed_n_kmh": _read_i16(data, 98) * 0.01,
        "sender": _read_u8(data, 164),
        "version": _read_u8(data, 165),
        "message_id": _read_u8(data, 166),
        "counter": _read_u8(data, 167),
        "gnss_status": _read_u8(data, 168),
        "ins_status": _read_u8(data, 169),
    }


def _validate_calibration_packet(p: dict, expected_message_id: int) -> List[str]:
    errors: List[str] = []
    lat = float(p.get("latitude_deg", float("nan")))
    lon = float(p.get("longitude_deg", float("nan")))
    heading = float(p.get("heading_deg", float("nan")))
    if not math.isfinite(lat) or not -90.0 <= lat <= 90.0:
        errors.append("latitude_invalid")
    if not math.isfinite(lon) or not -180.0 <= lon <= 180.0:
        errors.append("longitude_invalid")
    if not math.isfinite(heading) or not 0.0 <= heading <= 360.0:
        errors.append("heading_invalid")
    if int(p.get("version", -1)) != 0xF0:
        errors.append("version_not_0xF0")
    if int(p.get("message_id", -1)) != int(expected_message_id):
        errors.append("unexpected_message_id:%s" % p.get("message_id"))
    return errors


def _project_root_from_config(config_path: str) -> Optional[Path]:
    if not config_path:
        return None
    p = Path(config_path).expanduser().resolve()
    for parent in [p.parent] + list(p.parents):
        if (parent / "src" / "autoscene_platform" / "scripts" / "geo_projection.py").is_file():
            return parent
    return None


def load_geo_projection(config_path: str = ""):
    """Load project geo_projection.py without importing rospy/catkin packages."""
    candidates: List[Path] = []
    env_root = os.environ.get("AUTOSCENE_ROOT", "").strip()
    if env_root:
        candidates.append(Path(env_root).expanduser() / "src/autoscene_platform/scripts/geo_projection.py")
    root = _project_root_from_config(config_path)
    if root is not None:
        candidates.append(root / "src/autoscene_platform/scripts/geo_projection.py")

    here = Path(__file__).resolve()
    for parent in here.parents:
        candidates.append(parent / "src/autoscene_platform/scripts/geo_projection.py")
    candidates.append(here.with_name("geo_projection.py"))

    seen = set()
    for path in candidates:
        path = path.resolve()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        spec = importlib.util.spec_from_file_location("autoscene_calibration_geo", str(path))
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    try:
        import geo_projection as module  # type: ignore

        return module
    except Exception as exc:
        raise RuntimeError(
            "找不到 geo_projection.py；请设置 AUTOSCENE_ROOT 或从 AutoScene 工程配置启动 UI"
        ) from exc


@dataclass
class OsmWay:
    points_g: List[Tuple[float, float]]
    tags: Dict[str, str]


class OsmMapData:
    """Complete OSM geometry stored in G coordinates for post-calibration verification."""

    def __init__(self, ways: List[OsmWay], source_path: str):
        self.ways = ways
        self.source_path = source_path

    @classmethod
    def load(cls, path: str, geo) -> "OsmMapData":
        path = os.path.abspath(os.path.expanduser(path))
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

        nodes: Dict[str, Tuple[float, float]] = {}
        raw_ways: List[Tuple[List[str], Dict[str, str]]] = []
        for _event, elem in ET.iterparse(path, events=("end",)):
            tag = elem.tag.rsplit("}", 1)[-1]
            if tag == "node":
                node_id = elem.attrib.get("id")
                lat = elem.attrib.get("lat")
                lon = elem.attrib.get("lon")
                if node_id is not None and lat is not None and lon is not None:
                    try:
                        nodes[str(node_id)] = geo.latlon_to_xy(float(lat), float(lon))
                    except Exception:
                        pass
                elem.clear()
            elif tag == "way":
                refs: List[str] = []
                tags: Dict[str, str] = {}
                for child in list(elem):
                    ctag = child.tag.rsplit("}", 1)[-1]
                    if ctag == "nd" and child.attrib.get("ref") is not None:
                        refs.append(str(child.attrib["ref"]))
                    elif ctag == "tag":
                        k = child.attrib.get("k")
                        v = child.attrib.get("v")
                        if k is not None and v is not None:
                            tags[str(k)] = str(v)
                if len(refs) >= 2:
                    raw_ways.append((refs, tags))
                elem.clear()

        ways: List[OsmWay] = []
        for refs, tags in raw_ways:
            pts = [nodes[r] for r in refs if r in nodes]
            if len(pts) >= 2:
                ways.append(OsmWay(points_g=pts, tags=tags))
        if not ways:
            raise RuntimeError("OSM 中没有可绘制 way")
        return cls(ways, path)


class CalibrationMapView(QGraphicsView):
    """A_runtime (ScenarioEnv world/map) view. Caller passes geometry in A."""

    hover_xy = pyqtSignal(float, float)
    target_selected = pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setMouseTracking(True)

        self._reference_items: List[QGraphicsPathItem] = []
        self._osm_items: List[QGraphicsPathItem] = []
        self._markers: Dict[str, Tuple[QGraphicsEllipseItem, QGraphicsLineItem]] = {}
        self._marker_poses: Dict[str, Tuple[float, float, float]] = {}

    @staticmethod
    def _s(x: float, y: float) -> QPointF:
        return QPointF(float(x), -float(y))

    def _remove_items(self, items: List[QGraphicsPathItem]) -> None:
        for item in list(items):
            try:
                self._scene.removeItem(item)
            except Exception:
                pass
        items.clear()

    @staticmethod
    def _build_path(polylines: Sequence[Sequence[Tuple[float, float]]]) -> QPainterPath:
        path = QPainterPath()
        for line in polylines:
            if len(line) < 2:
                continue
            p0 = line[0]
            path.moveTo(QPointF(float(p0[0]), -float(p0[1])))
            for x, y in line[1:]:
                path.lineTo(QPointF(float(x), -float(y)))
        return path

    def set_reference_map(
        self,
        lanes: Sequence[Sequence[Tuple[float, float]]],
        boundaries: Sequence[Sequence[Tuple[float, float]]],
        crosswalks: Sequence[Sequence[Tuple[float, float]]],
        fit: bool = True,
    ) -> None:
        """Draw current PKL map in the A_runtime frame supplied by caller."""
        self._remove_items(self._reference_items)

        for polylines, color, width, z in (
            (boundaries, QColor("#a8adb3"), 0.0, 2),
            (crosswalks, QColor("#8b929a"), 0.0, 3),
            (lanes, QColor("#39424e"), 0.0, 4),
        ):
            path = self._build_path(polylines)
            item = QGraphicsPathItem(path)
            item.setPen(QPen(color, width))
            item.setZValue(z)
            self._scene.addItem(item)
            self._reference_items.append(item)

        self._restore_markers()
        self._update_scene_rect(fit=fit)

    def set_osm_overlay(self, osm: OsmMapData, transform, fit: bool = False) -> None:
        """Transform complete OSM G geometry to the current A_runtime for visual verification."""
        self._remove_items(self._osm_items)
        lane_lines: List[List[Tuple[float, float]]] = []
        boundary_lines: List[List[Tuple[float, float]]] = []
        other_lines: List[List[Tuple[float, float]]] = []

        for way in osm.ways:
            pts = [transform.point_g_to_a(x, y) for x, y in way.points_g]
            if len(pts) < 2:
                continue
            tags = way.tags
            custom_type = str(tags.get("Type", tags.get("type", ""))).strip().lower()
            if custom_type == "lane" or "highway" in tags:
                lane_lines.append(pts)
            elif custom_type == "laneboundary" or "boundary" in custom_type:
                boundary_lines.append(pts)
            else:
                other_lines.append(pts)

        for polylines, color, z in (
            (other_lines, QColor("#d5d9dd"), 1),
            (boundary_lines, QColor("#c4a46a"), 5),
            (lane_lines, QColor("#b7791f"), 6),
        ):
            path = self._build_path(polylines)
            item = QGraphicsPathItem(path)
            item.setPen(QPen(color, 0))
            item.setZValue(z)
            self._scene.addItem(item)
            self._osm_items.append(item)

        self._restore_markers()
        self._update_scene_rect(fit=fit)

    def clear_osm_overlay(self) -> None:
        self._remove_items(self._osm_items)
        self._update_scene_rect(fit=False)

    def _update_scene_rect(self, fit: bool) -> None:
        rect = self._scene.itemsBoundingRect().adjusted(-20, -20, 20, 20)
        if rect.isValid() and rect.width() > 0 and rect.height() > 0:
            self._scene.setSceneRect(rect)
            if fit:
                self.fitInView(rect, Qt.KeepAspectRatio)

    def fit_all(self) -> None:
        self._update_scene_rect(fit=True)

    def _marker_style(self, which: str) -> Tuple[QColor, float]:
        if which == "actual":
            return QColor("#d97706"), 52
        if which == "calibration_target":
            return QColor("#2563eb"), 54
        if which == "start_target":
            return QColor("#059669"), 56
        return QColor("#7c3aed"), 50

    def _set_pose_marker(self, which: str, x: float, y: float, yaw: float) -> None:
        self._marker_poses[which] = (float(x), float(y), float(yaw))
        marker_arrow = self._markers.get(which)
        color, z = self._marker_style(which)
        if marker_arrow is None:
            # Fixed-screen-size marker: keep the calibration point small even when
            # the complete OSM is fitted or heavily zoomed.  Coordinates remain in
            # A; only the glyph size ignores view transforms.
            marker = QGraphicsEllipseItem(-3.0, -3.0, 6.0, 6.0)
            marker.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
            marker.setBrush(QBrush(color))
            marker.setPen(QPen(color, 1.0))
            marker.setZValue(z)
            self._scene.addItem(marker)
            arrow = QGraphicsLineItem()
            arrow.setPen(QPen(color, 0))
            arrow.setZValue(z + 1)
            self._scene.addItem(arrow)
            self._markers[which] = (marker, arrow)
        else:
            marker, arrow = marker_arrow

        p = self._s(x, y)
        marker.setPos(p)
        length = 10.0
        end = self._s(x + length * math.cos(yaw), y + length * math.sin(yaw))
        arrow.setLine(p.x(), p.y(), end.x(), end.y())

    def _restore_markers(self) -> None:
        # Static layers are removed independently, so markers normally survive. This
        # also repairs them if a future caller cleared the scene externally.
        for which, pose in list(self._marker_poses.items()):
            marker_arrow = self._markers.get(which)
            if marker_arrow is None or marker_arrow[0].scene() is None:
                self._markers.pop(which, None)
                self._set_pose_marker(which, *pose)

    def set_actual_pose(self, x: float, y: float, yaw: float) -> None:
        self._set_pose_marker("actual", x, y, yaw)

    def set_calibration_target_pose(self, x: float, y: float, yaw: float) -> None:
        self._set_pose_marker("calibration_target", x, y, yaw)

    def set_start_target_pose(self, x: float, y: float, yaw: float) -> None:
        self._set_pose_marker("start_target", x, y, yaw)

    def clear_marker(self, which: str) -> None:
        pair = self._markers.pop(str(which), None)
        self._marker_poses.pop(str(which), None)
        if pair is not None:
            for item in pair:
                if item.scene() is self._scene:
                    self._scene.removeItem(item)

    def mouseMoveEvent(self, event) -> None:
        p = self.mapToScene(event.pos())
        self.hover_xy.emit(float(p.x()), float(-p.y()))
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event) -> None:
        # No lane snapping and no automatic heading inference. Clicking only picks x/y;
        # the A yaw must come from the virtual vehicle/simulator as requested.
        if event.button() == Qt.LeftButton and not (event.modifiers() & Qt.ControlModifier):
            p = self.mapToScene(event.pos())
            self.target_selected.emit(float(p.x()), float(-p.y()))
        super().mousePressEvent(event)

    def wheelEvent(self, event) -> None:
        factor = 1.18 if event.angleDelta().y() > 0 else 1.0 / 1.18
        self.scale(factor, factor)


class CalibrationWindow(QDialog):
    """Single-point G->A calibration plus pre-start hybrid platform alignment."""

    direct_sample = pyqtSignal(dict)

    def __init__(self, config, bridge_client=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("G ↔ A（MetaDrive / ScenarioEnv）单点标定与起步对齐")
        self.resize(1540, 940)
        self.cfg = config
        self.bridge_client = bridge_client
        self.geo = load_geo_projection(getattr(config, "path", "") or "")

        self.osm: Optional[OsmMapData] = None
        self.latest_sample: Optional[dict] = None
        self.frozen_sample: Optional[dict] = None
        self.start_target_pose: Optional[dict] = None
        self.start_target_diag: Optional[dict] = None
        self._solved_this_session = False
        self._candidate_from_legacy = False
        self._has_target = False
        self._display_origin = None
        self._display_scene_id = None

        self._udp_thread = None
        self._udp_stop = threading.Event()
        self._udp_sock = None

        self.direct_sample.connect(self._accept_sample)
        self._build_ui()
        self._populate_platforms()
        self._load_parent_scene_map(silent=True)
        self._load_candidate_from_config()
        self._refresh_start_target(silent=True)
        # An old UDP reading must not be displayed as a fresh live platform pose.
        self._freshness_timer = QTimer(self)
        self._freshness_timer.setInterval(400)
        self._freshness_timer.timeout.connect(self._expire_live_sample)
        self._freshness_timer.start()

        # Important: do not connect bridge_client.ads_decoded here. MainWindow already
        # forwards that signal, and the standalone launcher explicitly connects it too.
        # Connecting here as well would process every UDP/GNSS sample twice.

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        top = QHBoxLayout()
        self.ed_osm = QLineEdit()
        self.ed_osm.setPlaceholderText("完整 TesT_Field_Tongji_East.osm（求解后仅用于 G→A 视觉核验）")
        btn_browse = QPushButton("选择 OSM…")
        btn_browse.clicked.connect(self._browse_osm)
        btn_load = QPushButton("读取完整 OSM")
        btn_load.clicked.connect(self._load_osm)
        btn_scene = QPushButton("载入当前 PKL A 地图")
        btn_scene.clicked.connect(lambda: self._load_parent_scene_map(silent=False))
        top.addWidget(QLabel("地图："))
        top.addWidget(self.ed_osm, 1)
        top.addWidget(btn_browse)
        top.addWidget(btn_load)
        top.addWidget(btn_scene)
        root.addLayout(top)

        split = QSplitter(Qt.Horizontal)
        self.map_view = CalibrationMapView()
        self.map_view.hover_xy.connect(self._on_hover)
        self.map_view.target_selected.connect(self._on_target_selected)
        split.addWidget(self.map_view)

        controls = QWidget()
        c = QVBoxLayout(controls)

        self.lab_map_source = QLabel(
            "等待真实 PKL 场景。深色=当前 MetaDrive A 地图；棕色=标定后完整 OSM 验证层。"
        )
        self.lab_map_source.setWordWrap(True)
        c.addWidget(self.lab_map_source)

        g_rx = QGroupBox("1. 实车 G 姿态（GNSS/INS）")
        f = QGridLayout(g_rx)
        self.cb_platform = QComboBox()
        self.cb_platform.currentIndexChanged.connect(self._platform_changed)
        self.lab_rx_mode = QLabel("等待 /autoscene/platform/ads_udp_decoded")
        self.lab_rx_mode.setWordWrap(True)
        self.ed_bind_ip = QLineEdit()
        self.sp_bind_port = QSpinBox(); self.sp_bind_port.setRange(1, 65535)
        self.ed_remote_ip = QLineEdit()
        self.sp_remote_port = QSpinBox(); self.sp_remote_port.setRange(-1, 65535)
        self.sp_message_id = QSpinBox(); self.sp_message_id.setRange(0, 255)
        self.btn_udp = QPushButton("启动直接 UDP 监听")
        self.btn_udp.clicked.connect(self._toggle_direct_udp)
        self.btn_freeze = QPushButton("冻结当前 G 姿态")
        self.btn_freeze.clicked.connect(self._freeze_current_g)
        f.addWidget(QLabel("平台"), 0, 0); f.addWidget(self.cb_platform, 0, 1, 1, 3)
        f.addWidget(self.lab_rx_mode, 1, 0, 1, 4)
        f.addWidget(QLabel("bind IP"), 2, 0); f.addWidget(self.ed_bind_ip, 2, 1)
        f.addWidget(QLabel("port"), 2, 2); f.addWidget(self.sp_bind_port, 2, 3)
        f.addWidget(QLabel("remote IP"), 3, 0); f.addWidget(self.ed_remote_ip, 3, 1)
        f.addWidget(QLabel("remote port(-1任意)"), 3, 2); f.addWidget(self.sp_remote_port, 3, 3)
        f.addWidget(QLabel("feedback MsgID"), 4, 0); f.addWidget(self.sp_message_id, 4, 1)
        f.addWidget(self.btn_udp, 4, 2, 1, 2)
        f.addWidget(self.btn_freeze, 5, 0, 1, 4)
        c.addWidget(g_rx)

        g_pose = QGroupBox("当前 / 已冻结 G 姿态")
        form = QFormLayout(g_pose)
        self.lab_g = QLabel("尚无实时数据")
        self.lab_g.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lab_g.setWordWrap(True)
        self.lab_frozen = QLabel("尚未冻结。标定计算只使用冻结样本。")
        self.lab_frozen.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lab_frozen.setWordWrap(True)
        form.addRow(self.lab_g)
        form.addRow(self.lab_frozen)
        c.addWidget(g_pose)

        g_target = QGroupBox("2. 仿真 A 姿态（与当前 MetaDrive/ROS world/map 一致）")
        grid = QGridLayout(g_target)
        self.sp_ax = self._big_spin(6)
        self.sp_ay = self._big_spin(6)
        self.sp_ayaw = QDoubleSpinBox()
        self.sp_ayaw.setRange(-math.pi * 20, math.pi * 20)
        self.sp_ayaw.setDecimals(12)
        self.sp_ayaw.setSingleStep(math.radians(0.1))
        self.sp_ax.valueChanged.connect(self._target_changed)
        self.sp_ay.valueChanged.connect(self._target_changed)
        self.sp_ayaw.valueChanged.connect(self._target_changed)
        btn_parent_pose = QPushButton("读取主界面车辆 A（只读）")
        btn_parent_pose.clicked.connect(self._copy_selected_parent_pose)
        btn_estimate = QPushButton("由冻结 G + 当前 A 计算 G→A")
        btn_estimate.clicked.connect(self._estimate_single)
        grid.addWidget(QLabel("A x [m]"), 0, 0); grid.addWidget(self.sp_ax, 0, 1)
        grid.addWidget(QLabel("A y [m]"), 0, 2); grid.addWidget(self.sp_ay, 0, 3)
        grid.addWidget(QLabel("A yaw [rad] (+x=0, CCW+)"), 1, 0); grid.addWidget(self.sp_ayaw, 1, 1, 1, 3)
        grid.addWidget(btn_parent_pose, 2, 0, 1, 2)
        grid.addWidget(btn_estimate, 2, 2, 1, 2)
        self.lab_calib_error = QLabel(
            "左键只选择中心 A x/y；航向由人工输入或读取虚拟车；不吸附、不猜测方向。"
        )
        self.lab_calib_error.setWordWrap(True)
        self.lab_origin_and_a = QLabel("请在主 GUI 加载真实 PKL；全部显示值即当前运行时 A。")
        self.lab_origin_and_a.setWordWrap(True)
        grid.addWidget(self.lab_origin_and_a, 3, 0, 1, 4)
        grid.addWidget(self.lab_calib_error, 4, 0, 1, 4)
        self.chk_precision = QCheckBox("主界面精确对齐：车身改为中心十字+方向线（不遮地图）")
        self.chk_precision.setChecked(True)
        self.chk_precision.toggled.connect(self._toggle_parent_precision_mode)
        grid.addWidget(self.chk_precision, 5, 0, 1, 4)
        c.addWidget(g_target)

        g_tf = QGroupBox("3. 实车 G → 当前场景 MetaDrive A（保存时记录参考场景）")
        grid = QGridLayout(g_tf)
        self.sp_tx = self._big_spin(9)
        self.sp_ty = self._big_spin(9)
        self.sp_yaw = QDoubleSpinBox()
        self.sp_yaw.setRange(-100.0, 100.0)
        self.sp_yaw.setDecimals(12)
        self.sp_yaw.setSingleStep(0.0001)
        for spin in (self.sp_tx, self.sp_ty, self.sp_yaw):
            spin.valueChanged.connect(self._candidate_changed)
        self.lab_yaw_deg = QLabel()
        btn_cfg = QPushButton("从 YAML 恢复")
        btn_cfg.clicked.connect(self._load_candidate_from_config)
        btn_osm = QPushButton("按 G→A 候选绘制完整 OSM（仅验证）")
        btn_osm.clicked.connect(self._use_candidate_for_map)
        btn_save = QPushButton("写入当前 YAML")
        btn_save.clicked.connect(self._save_yaml)
        grid.addWidget(QLabel("当前 tx_A [m]"), 0, 0); grid.addWidget(self.sp_tx, 0, 1, 1, 3)
        grid.addWidget(QLabel("当前 ty_A [m]"), 1, 0); grid.addWidget(self.sp_ty, 1, 1, 1, 3)
        grid.addWidget(QLabel("yaw [rad]"), 2, 0); grid.addWidget(self.sp_yaw, 2, 1, 1, 2); grid.addWidget(self.lab_yaw_deg, 2, 3)
        self.lab_runtime_tf = QLabel("候选参数即当前场景 G→A；换场景时自动调整平移量。")
        self.lab_runtime_tf.setWordWrap(True)
        grid.addWidget(self.lab_runtime_tf, 3, 0, 1, 4)
        grid.addWidget(btn_cfg, 4, 0); grid.addWidget(btn_osm, 4, 1, 1, 2); grid.addWidget(btn_save, 4, 3)
        c.addWidget(g_tf)

        g_start = QGroupBox("4. 混合模式上车对齐：initial(第21帧) → 自动回退第0帧")
        grid = QGridLayout(g_start)
        self.lab_initial21 = QLabel("等待平台 initial 配置")
        self.lab_initial21.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lab_initial21.setWordWrap(True)
        self.lab_start0 = QLabel("—")
        self.lab_start0.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lab_start0.setWordWrap(True)
        self.lab_align_error = QLabel("收到实时平台 G 后，将与当前场景 A0 目标比较。")
        self.lab_align_error.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lab_align_error.setWordWrap(True)
        btn_start = QPushButton("刷新并显示第0帧目标")
        btn_start.clicked.connect(lambda: self._refresh_start_target(silent=False))
        grid.addWidget(QLabel("initial / 第21帧"), 0, 0); grid.addWidget(self.lab_initial21, 0, 1, 1, 3)
        grid.addWidget(QLabel("自动回退 / 第0帧"), 1, 0); grid.addWidget(self.lab_start0, 1, 1, 1, 3)
        grid.addWidget(btn_start, 2, 0, 1, 4)
        grid.addWidget(QLabel("实车 vs 第0帧"), 3, 0); grid.addWidget(self.lab_align_error, 3, 1, 1, 3)
        c.addWidget(g_start)

        g_hover = QGroupBox("A（当前 ScenarioEnv world/map）坐标探针")
        v = QVBoxLayout(g_hover)
        self.lab_hover = QLabel("地图光标只显示运行时 A x/y；不做吸附。")
        self.lab_hover.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lab_hover.setWordWrap(True)
        v.addWidget(self.lab_hover)
        c.addWidget(g_hover)

        row = QHBoxLayout()
        btn_fit2 = QPushButton("全图")
        btn_fit2.clicked.connect(self.map_view.fit_all)
        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(self.close)
        row.addWidget(btn_fit2)
        row.addStretch(1)
        row.addWidget(btn_close)
        c.addLayout(row)
        c.addStretch(1)

        split.addWidget(controls)
        split.setSizes([1030, 500])
        root.addWidget(split, 1)

    def map_view_fit(self) -> None:
        if hasattr(self, "map_view"):
            self.map_view.fit_all()

    @staticmethod
    def _big_spin(decimals: int) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(-1.0e9, 1.0e9)
        spin.setDecimals(decimals)
        spin.setSingleStep(0.1)
        return spin

    # -------------------------------------------------------------- map/A frame
    def _origin(self) -> Optional[Tuple[float, float]]:
        parent = self.parent()
        scene = getattr(parent, "scene", None) if parent is not None else None
        if scene is None or bool(getattr(scene, "is_demo", True)):
            return None
        try:
            ox = float(scene.runtime_origin_x)
            oy = float(scene.runtime_origin_y)
        except (AttributeError, ValueError, TypeError):
            return None
        if not math.isfinite(ox) or not math.isfinite(oy):
            return None
        return ox, oy

    def _runtime_transform(self):
        """All UI calibration fields are the effective G→A for THIS scenario."""
        if self._origin() is None:
            raise RuntimeError('请先加载真实 PKL 场景')
        return self._candidate()

    def _toggle_parent_precision_mode(self, enabled: bool) -> None:
        parent = self.parent()
        view = getattr(parent, "view", None) if parent is not None else None
        if view is not None and hasattr(view, "set_precision_pose_mode"):
            view.set_precision_pose_mode(bool(enabled))

    def _load_parent_scene_map(self, silent: bool = False) -> None:
        parent = self.parent()
        scene = getattr(parent, "scene", None) if parent is not None else None
        origin = self._origin()
        if scene is None or origin is None:
            self._display_origin = None
            self._display_scene_id = None
            self.map_view.set_reference_map([], [], [], fit=False)
            self.map_view.clear_osm_overlay()
            self.lab_map_source.setText("没有真实 PKL：请先在主界面加载场景。")
            if not silent:
                QMessageBox.warning(self, "缺少运行时 A 坐标", "请先在主界面加载一个真实 PKL。")
            return
        if self._display_origin != origin or self._display_scene_id != id(scene):
            self.map_view.clear_osm_overlay()  # previous scenario no longer valid
        # MainWindow loaded every geometry and vehicle into actual runtime A.
        lanes = [line for line in scene.lanes if len(line)>=2]
        bounds = [line for line in scene.boundaries if len(line)>=2]
        walks = [line for line in scene.crosswalks if len(line)>=2]
        self.map_view.set_reference_map(lanes,bounds,walks,fit=True)
        scene_id = id(scene)
        if self._display_scene_id is not None and self._display_scene_id != scene_id:
            self._has_target = False
            self.latest_sample = None
            self.map_view.clear_marker('actual')
            self._solved_this_session = False
            self.frozen_sample = None
            self.lab_frozen.setText("场景已切换，原有冻结点无效；请重新冻结后校验当前场景的标定。")
        self._display_scene_id = scene_id
        self._display_origin = origin
        self.lab_map_source.setText(
            "当前真实 PKL：%s；本窗口与主 GUI 均使用同一当前 MetaDrive A 坐标。"
            "完整 OSM 仅用于标定后的辅助验证。"
            % getattr(scene,'name','scenario')
        )
        self._target_changed()
        self._candidate_changed()
        self._toggle_parent_precision_mode(self.chk_precision.isChecked())

    def _browse_osm(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择完整 OSM XML",
            self.ed_osm.text(),
            "OSM (*.osm *.xml);;All files (*)",
        )
        if path:
            self.ed_osm.setText(path)

    def _load_osm(self) -> None:
        try:
            self.osm = OsmMapData.load(self.ed_osm.text().strip(), self.geo)
        except Exception as exc:
            QMessageBox.critical(self, "读取 OSM 失败", str(exc))
            return
        QMessageBox.information(
            self,
            "OSM 已读取",
            "已读取 %d 条 way。标定 A 姿态仍应来自仿真车辆；OSM 只在算出候选后用于全图验证。"
            % len(self.osm.ways),
        )

    def _use_candidate_for_map(self) -> None:
        if self._origin() is None:
            QMessageBox.warning(self, "缺少当前 PKL", "请先加载真实 PKL 获取 A 坐标和 runtime_origin")
            return
        if self.osm is None:
            if not self.ed_osm.text().strip():
                QMessageBox.warning(self, "未选择 OSM", "请先选择完整 TesT_Field_Tongji_East.osm。")
                return
            self._load_osm()
            if self.osm is None:
                return
        self.map_view.set_osm_overlay(self.osm, self._runtime_transform(), fit=False)
        self.map_view.fit_all()

    def _on_hover(self, x: float, y: float) -> None:
        try:
            self.lab_hover.setText("A x=%.6f m, A y=%.6f m（当前 ScenarioEnv world）" % (x,y))
        except RuntimeError:
            self.lab_hover.setText("请先加载真实 PKL 地图")

    def _on_target_selected(self, x: float, y: float) -> None:
        if self._origin() is None:
            QMessageBox.warning(self, '缺少场景', '请先加载真实 PKL')
            return
        # Click directly specifies A. No snap; it never moves a vehicle.
        self._set_target_pose_atomic(x, y, self.sp_ayaw.value())

    def _copy_selected_parent_pose(self) -> None:
        parent = self.parent()
        if parent is None or not hasattr(parent, "scene"):
            QMessageBox.warning(self, "无法读取", "该功能需要从 AutoScene 主界面打开标定窗口。")
            return
        key = None
        combo = getattr(parent, "cb_edit_vehicle", None)
        if combo is not None:
            try:
                key = combo.currentData()
            except Exception:
                key = None
        if key is None:
            QMessageBox.warning(self, "没有选择车辆", "请先在主界面选择/编辑一台虚拟车辆或 Ego。")
            return
        key = str(key)
        obj = None
        if key == "ego":
            obj = getattr(parent.scene, "ego", None)
        if obj is None:
            for item in getattr(parent, "virtuals", []) or []:
                if str(item.get("key", "")) == key:
                    self._set_target_pose_atomic(
                        float(item["x"]), float(item["y"]), float(item["heading"])
                    )
                    return
        if obj is None:
            obj = next(
                (o for o in getattr(parent.scene, "objects", []) if str(getattr(o, "track_id", "")) == key),
                None,
            )
        if obj is None:
            QMessageBox.warning(self, "无法读取", "找不到主界面当前车辆：%s" % key)
            return
        self._set_target_pose_atomic(float(obj.x), float(obj.y), float(obj.heading))

    def _set_target_pose_atomic(self, x: float, y: float, yaw: float) -> None:
        """Read-only/atomic update of the current runtime A target fields.

        The previous implementation changed x, y and yaw one by one with signals
        enabled.  Each intermediate value refreshed the blue overlay, which looked
        like the simulated vehicle was jumping even though no vehicle state was
        actually modified.  Block signals and refresh exactly once.
        """
        values = (float(x), float(y), self.geo.wrap_rad(float(yaw)))
        for spin, value in zip((self.sp_ax, self.sp_ay, self.sp_ayaw), values):
            spin.blockSignals(True)
            try:
                spin.setValue(value)
            finally:
                spin.blockSignals(False)
        self._has_target = True
        # A different target invalidates the previous G→A candidate's provenance.
        self._solved_this_session = False
        self._target_changed()

    # -------------------------------------------------------------- platform/G
    def _populate_platforms(self) -> None:
        current_pid = None
        old = self.cb_platform.currentData() if self.cb_platform.count() else None
        if isinstance(old, dict):
            current_pid = str(old.get("platform_id", ""))
        self.cb_platform.blockSignals(True)
        self.cb_platform.clear()
        eps = self.cfg.raw.get("platform", {}).get("platforms", [])
        selected_index = 0
        for ep in eps:
            if not isinstance(ep, dict):
                continue
            pid = str(ep.get("platform_id", ""))
            vid = str(ep.get("vehicle_id", ""))
            idx = self.cb_platform.count()
            self.cb_platform.addItem("%s / %s" % (pid, vid), ep)
            if current_pid and pid == current_pid:
                selected_index = idx
        if self.cb_platform.count():
            self.cb_platform.setCurrentIndex(selected_index)
        self.cb_platform.blockSignals(False)
        self._refresh_direct_fields()

    def _platform_changed(self, _index=None) -> None:
        # Never reuse a frozen GNSS sample from a previously selected platform.
        self._stop_direct_udp()
        self.latest_sample = None
        self.frozen_sample = None
        self.lab_g.setText("等待当前平台有效 GNSS/INS 数据。")
        self.lab_frozen.setText("平台已切换；请重新冻结当前 G 姿态。")
        self._solved_this_session = False
        self._refresh_direct_fields()
        self._refresh_start_target(silent=True)

    def _refresh_direct_fields(self) -> None:
        p = self.cfg.raw.get("platform", {})
        ep = self.cb_platform.currentData() or {}
        self.ed_bind_ip.setText(str(p.get("local_ip", "0.0.0.0")))
        self.sp_bind_port.setValue(int(p.get("local_port", 31100)))
        self.ed_remote_ip.setText(str(ep.get("expected_remote_ip", ep.get("ip", ""))))
        self.sp_remote_port.setValue(int(ep.get("expected_remote_port", -1)))
        self.sp_message_id.setValue(int(p.get("expected_feedback_message_id", 1)))

    def on_ads_decoded(self, msg: dict) -> None:
        """Consume std_msgs/String payload from platform_comm_node via rosbridge/rospy client."""
        raw = msg.get("data") if isinstance(msg, dict) else None
        if not raw:
            return
        try:
            payload = json.loads(raw)
        except Exception:
            return
        selected = self.cb_platform.currentData() or {}
        pid = str(selected.get("platform_id", ""))
        if pid and str(payload.get("platform_id", "")) != pid:
            return
        if payload.get("protocol_valid") is False:
            self.lab_rx_mode.setText("ROS 解码流收到无效包：%s" % payload.get("protocol_errors"))
            return
        earth = payload.get("earth_pose") or {}
        decoded = payload.get("decoded") or {}
        try:
            sample = {
                "source": "rosbridge decoded",
                "platform_id": payload.get("platform_id", ""),
                "vehicle_id": payload.get("vehicle_id", ""),
                "remote_ip": payload.get("remote_ip", ""),
                "earth": {
                    "x": float(earth["x"]),
                    "y": float(earth["y"]),
                    "yaw": float(earth["yaw"]),
                },
                "decoded": decoded,
            }
        except Exception:
            return
        self._accept_sample(sample)

    def _accept_sample(self, sample: dict) -> None:
        if sample.get("error"):
            self.lab_rx_mode.setText("UDP 监听失败：%s" % sample.get("error"))
            self.btn_udp.setText("启动直接 UDP 监听")
            return
        # Drop delayed queued packets from the previously selected platform.
        chosen = self.cb_platform.currentData() or {}
        current_pid = str(chosen.get("platform_id", ""))
        if current_pid and str(sample.get("platform_id", "")) != current_pid:
            return
        sample["received_monotonic"] = time.monotonic()
        self.latest_sample = sample
        e = sample["earth"]
        d = sample.get("decoded") or {}
        heading_deg = d.get("heading_deg")
        self.lab_g.setText(
            "source=%s  platform=%s\n"
            "G x=%.6f m\nG y=%.6f m\nG yaw=%.9f rad\n"
            "INS heading=%s°（北=0°, 顺时针+）\nlat=%s lon=%s"
            % (
                sample.get("source", ""),
                sample.get("platform_id", ""),
                e["x"],
                e["y"],
                e["yaw"],
                "%.6f" % float(heading_deg) if heading_deg is not None else "—",
                d.get("latitude_deg", "—"),
                d.get("longitude_deg", "—"),
            )
        )
        self.lab_rx_mode.setText("已收到平台状态：%s" % sample.get("source", ""))
        self._candidate_changed()

    def _expire_live_sample(self) -> None:
        sample = self.latest_sample
        if sample is None:
            return
        if time.monotonic() - float(sample.get('received_monotonic', 0.0)) <= 1.5:
            return
        self.latest_sample = None
        self.map_view.clear_marker('actual')
        parent_view = getattr(self.parent(), 'view', None)
        if parent_view is not None and hasattr(parent_view, 'clear_pose_overlay'):
            parent_view.clear_pose_overlay('alignment:platform_actual')
        self.lab_rx_mode.setText('平台实时数据已超时；橙色实时姿态已隐藏。')
        self.lab_align_error.setText('等待最新平台 G 姿态；禁止据旧数据摆车。')
        self.lab_calib_error.setText('平台 G 实时数据已超时；请等待新反馈。')

    def _freeze_current_g(self) -> None:
        if self.latest_sample is None:
            QMessageBox.warning(self, "没有 G 数据", "尚未收到有效平台 GNSS/INS 姿态。")
            return
        age = time.monotonic() - float(self.latest_sample.get("received_monotonic", 0.0))
        if age > 1.5:
            QMessageBox.warning(self, "GNSS 数据过期", "最新平台包距今 %.2fs，请先恢复实时通信。" % age)
            return
        self.frozen_sample = copy.deepcopy(self.latest_sample)
        # Never save an old solution against a newly frozen physical pose.
        self._solved_this_session = False
        e = self.frozen_sample["earth"]
        d = self.frozen_sample.get("decoded") or {}
        self.lab_frozen.setText(
            "已冻结：G=(%.6f, %.6f), G yaw=%.9f rad；INS heading=%s°。\n"
            "移动平台或继续收到 UDP 不会改变本次标定使用的 G。"
            % (
                e["x"],
                e["y"],
                e["yaw"],
                "%.6f" % float(d.get("heading_deg")) if d.get("heading_deg") is not None else "—",
            )
        )

    # -------------------------------------------------------------- direct UDP
    def _toggle_direct_udp(self) -> None:
        if self._udp_thread is not None and self._udp_thread.is_alive():
            self._stop_direct_udp()
        else:
            self._start_direct_udp()

    def _start_direct_udp(self) -> None:
        self._udp_stop.clear()
        bind_ip = self.ed_bind_ip.text().strip() or "0.0.0.0"
        bind_port = int(self.sp_bind_port.value())
        remote_ip = self.ed_remote_ip.text().strip()
        remote_port = int(self.sp_remote_port.value())
        expected_mid = int(self.sp_message_id.value())
        selected = dict(self.cb_platform.currentData() or {})

        def worker():
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(0.3)
            try:
                sock.bind((bind_ip, bind_port))
                self._udp_sock = sock
                while not self._udp_stop.is_set():
                    try:
                        data, addr = sock.recvfrom(4096)
                    except socket.timeout:
                        continue
                    except OSError:
                        if self._udp_stop.is_set():
                            break
                        raise
                    if remote_ip and addr[0] != remote_ip:
                        continue
                    if remote_port >= 0 and int(addr[1]) != remote_port:
                        continue
                    try:
                        decoded = decode_ads_payload176(data)
                        errors = _validate_calibration_packet(decoded, expected_mid)
                        if errors:
                            continue
                        gx, gy = self.geo.latlon_to_xy(decoded["latitude_deg"], decoded["longitude_deg"])
                        gyaw = self.geo.ins_heading_deg_to_earth_yaw(decoded["heading_deg"])
                        self.direct_sample.emit(
                            {
                                "source": "direct UDP %s:%d" % addr,
                                "platform_id": selected.get("platform_id", ""),
                                "vehicle_id": selected.get("vehicle_id", ""),
                                "remote_ip": addr[0],
                                "earth": {"x": gx, "y": gy, "yaw": gyaw},
                                "decoded": decoded,
                            }
                        )
                    except Exception:
                        continue
            except Exception as exc:
                self.direct_sample.emit(
                    {
                        "error": str(exc),
                        "earth": {"x": 0.0, "y": 0.0, "yaw": 0.0},
                        "source": "UDP bind failed",
                    }
                )
            finally:
                try:
                    sock.close()
                except Exception:
                    pass
                self._udp_sock = None

        self._udp_thread = threading.Thread(target=worker, daemon=True)
        self._udp_thread.start()
        self.btn_udp.setText("停止直接 UDP 监听")
        self.lab_rx_mode.setText(
            "直接监听 %s:%d；若 platform_comm_node 已占用此端口会绑定失败" % (bind_ip, bind_port)
        )

    def _stop_direct_udp(self) -> None:
        self._udp_stop.set()
        sock = self._udp_sock
        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass
        self.btn_udp.setText("启动直接 UDP 监听")

    # -------------------------------------------------------------- parent SceneView overlay
    def _set_parent_overlay(self, key: str, x: float, y: float, yaw: float,
                            label: str, kind: str) -> None:
        parent = self.parent()
        view = getattr(parent, "view", None) if parent is not None else None
        if view is None or not hasattr(view, "set_pose_overlay"):
            return
        try:
            view.set_pose_overlay(
                "alignment:" + str(key), float(x), float(y), float(yaw),
                label=str(label), kind=str(kind),
            )
        except Exception:
            pass

    def _clear_parent_overlays(self) -> None:
        parent = self.parent()
        view = getattr(parent, "view", None) if parent is not None else None
        if view is not None and hasattr(view, "clear_pose_overlays"):
            try:
                view.clear_pose_overlays(prefix="alignment:")
            except Exception:
                pass

    def refresh_from_parent(self) -> None:
        """Update map origin and overlays after scene edits, without modifying vehicles."""
        origin = self._origin()
        parent = self.parent()
        scene = getattr(parent, "scene", None) if parent is not None else None
        if origin != self._display_origin or (scene is not None and id(scene) != self._display_scene_id):
            self._load_parent_scene_map(silent=True)
            if not self._solved_this_session:
                self._load_candidate_from_config()
        self._refresh_start_target(silent=True)
        self._target_changed()
        self._candidate_changed()

    def _candidate(self):
        return self.geo.EarthToAutoSceneTransform(
            yaw=self.sp_yaw.value(),
            tx=self.sp_tx.value(),
            ty=self.sp_ty.value(),
        )

    def _load_candidate_from_config(self) -> None:
        platform = self.cfg.raw.get("platform", {}) or {}
        cfg = platform.get("earth_to_autoscene", {}) or {}
        semantics = str(platform.get("earth_to_autoscene_frame", "runtime")).strip().lower()
        self._candidate_from_legacy = semantics != 'reference_scenario'
        if semantics == 'reference_scenario':
            ref_dataset = str(platform.get('calibration_reference_dataset', '')).strip()
            current_dataset = str(self.cfg.scenario.get('dataset_path', '')).strip()
            if ref_dataset and current_dataset and ref_dataset != current_dataset:
                self._solved_this_session = False
                self.lab_runtime_tf.setText('标定来自另一个数据集；不同地图的 A 不能直接复用，请重新标定。')
                return
        tx, ty, yaw = (float(cfg.get(k,0.0)) for k in ("tx","ty","yaw"))
        origin = self._origin()
        if semantics == 'reference_scenario':
            ref = platform.get('calibration_reference_origin') or {}
            if origin is None or 'x' not in ref or 'y' not in ref:
                self.lab_runtime_tf.setText('参考场景元数据缺失；禁止计算或保存。')
                return
            tx += float(ref['x']) - origin[0]
            ty += float(ref['y']) - origin[1]
        elif semantics == 'dataset_global':
            # Backward compatibility for previously issued G/U/A patch.
            if origin is None:
                self.lab_runtime_tf.setText('旧场地配置需要当前 PKL 原点。')
                return
            tx -= origin[0]
            ty -= origin[1]
        elif semantics != 'runtime':
            self.lab_runtime_tf.setText('未知标定类型：%s' % semantics)
            return
        for spin, value in ((self.sp_tx, tx),(self.sp_ty,ty),(self.sp_yaw,yaw)):
            spin.blockSignals(True)
            try: spin.setValue(value)
            finally: spin.blockSignals(False)
        self._solved_this_session = False
        self._candidate_changed()
        if semantics != 'reference_scenario':
            self.lab_runtime_tf.setText(
                '兼容读取旧标定类型 %s；如需跨 PKL 复用，请重新冻结 G 并执行单点标定保存。' % semantics
            )

    def _set_candidate(self, tf) -> None:
        for spin, value in ((self.sp_tx, tf.tx), (self.sp_ty, tf.ty), (self.sp_yaw, tf.yaw)):
            spin.blockSignals(True)
            spin.setValue(float(value))
            spin.blockSignals(False)
        self._candidate_changed()

    def _candidate_changed(self, _value=None) -> None:
        # A manual edit is not a new verified G/A calibration.
        if _value is not None:
            self._solved_this_session = False
        self.lab_yaw_deg.setText("%.6f°" % math.degrees(self.sp_yaw.value()))
        origin = self._origin()
        if origin is None:
            self.lab_runtime_tf.setText("未加载真实 PKL：不可算 G→A，也不可保存本场景标定。")
            return
        tf_a = self._runtime_transform()
        warning = ('【旧配置兼容预览，重新标定后才能保存】 ' if self._candidate_from_legacy else '')
        self.lab_runtime_tf.setText(
            warning + '当前场景 G→A: tx=%.9f, ty=%.9f, yaw=%.12f rad。'
            '更换 PKL 后，参考场景元数据会自动修正平移。'
            % (tf_a.tx,tf_a.ty,tf_a.yaw)
        )
        if self.latest_sample is not None:
            e=self.latest_sample['earth']
            ax, ay = tf_a.point_g_to_a(e['x'],e['y'])
            ayaw = tf_a.yaw_g_to_a(e['yaw'])
            self.map_view.set_actual_pose(ax,ay,ayaw)
            self._set_parent_overlay('platform_actual',ax,ay,ayaw,
                                     '实车 G→A','platform_actual')
            if self._has_target:
                self._update_calibration_error(ax,ay,ayaw)
            else:
                self.lab_calib_error.setText('请先设置 A 标定点，才可显示目标误差。')
        self._update_alignment_error()

    def _target_changed(self, _value=None) -> None:
        if _value is not None:
            self._has_target = True
            self._solved_this_session = False
        origin = self._origin()
        if origin is None:
            self.lab_origin_and_a.setText("未加载真实 PKL；禁止标定。")
            return
        if not self._has_target:
            self.lab_origin_and_a.setText(
                '尚未选择 A 标定目标；请点击 A 地图或只读获取仿真车辆 A。' )
            self.map_view.clear_marker('calibration_target')
            parent=self.parent()
            view=getattr(parent,'view',None)
            if view is not None and hasattr(view,'clear_pose_overlay'):
                view.clear_pose_overlay('alignment:calibration_target')
            return
        ax, ay = self.sp_ax.value(), self.sp_ay.value()
        yaw = self.geo.wrap_rad(self.sp_ayaw.value())
        self.lab_origin_and_a.setText(
            '标定目标 A=(%.6f, %.6f)，yaw=%.9f rad。' % (ax, ay, yaw)
        )
        self.map_view.set_calibration_target_pose(ax,ay,yaw)
        self._set_parent_overlay('calibration_target',ax,ay,yaw,
                                 '标定 A 目标','calibration_target')
        if self.latest_sample is not None:
            e=self.latest_sample['earth']
            t=self._runtime_transform()
            rx,ry=t.point_g_to_a(e['x'],e['y'])
            self._update_calibration_error(rx,ry,t.yaw_g_to_a(e['yaw']))

    def _update_calibration_error(self, ax: float, ay: float, ayaw: float) -> None:
        target_x, target_y = self.sp_ax.value(), self.sp_ay.value()
        dx, dy = ax-target_x, ay-target_y
        dyaw = self.geo.wrap_rad(ayaw-self.sp_ayaw.value())
        self.lab_calib_error.setText(
            "实时平台 G→A=(%.4f,%.4f), yaw=%.6f rad；"
            "相对 A 目标: Δx=%+.4f m, Δy=%+.4f m, 距离=%.4f m, Δyaw=%+.3f°"
            % (ax,ay,ayaw,dx,dy,math.hypot(dx,dy),math.degrees(dyaw))
        )

    def _estimate_single(self) -> None:
        if self.frozen_sample is None:
            QMessageBox.warning(self,'请先冻结 G','先将实车中心对准标定点并冻结 GNSS/INS。')
            return
        origin = self._origin()
        if origin is None:
            QMessageBox.critical(self,'无运行时坐标','没有真实 PKL 的 runtime_origin，无法形成 G→A。')
            return
        if not self._has_target:
            QMessageBox.warning(self, '没有标定点', '请先在 A 地图左键点击交接点，或从主界面只读载入一个 A 姿态。')
            return
        ax,ay = self.sp_ax.value(),self.sp_ay.value()
        yaw = self.geo.wrap_rad(self.sp_ayaw.value())
        pair={'earth':dict(self.frozen_sample['earth']),
              'autoscene':{'x':ax,'y':ay,'yaw':yaw}}
        try:
            tf_a,diag=self.geo.estimate_earth_to_autoscene([pair])
        except Exception as exc:
            QMessageBox.critical(self,'标定失败',str(exc)); return
        # Store the same G→A transform for the calibration reference scenario.
        # A future scenario derives its own G→A using the two PKL origins.
        self._candidate_from_legacy = False
        self._set_candidate(tf_a)
        self._solved_this_session=True
        if self.osm is not None:
            self.map_view.set_osm_overlay(self.osm,tf_a,fit=False)
        QMessageBox.information(
            self,'完成当前场景 G→A 单点标定',
            '当前场景 G→A: tx=%.12f, ty=%.12f, yaw=%.12f rad\n'
            '保存时附带标定场景原点信息；换 PKL 自动计算该场景 G→A。\n\n'
            '单点残差必然接近零；请在另一物理位置独立验证。'
            % (tf_a.tx,tf_a.ty,tf_a.yaw)
        )

    def _selected_initial_pose(self) -> Optional[dict]:
        ep = self.cb_platform.currentData() or {}
        pid = str(ep.get("platform_id", ""))
        vid = str(ep.get("vehicle_id", ""))

        parent = self.parent()
        if parent is not None and hasattr(parent, "agents"):
            try:
                for agent in parent.agents(runtime_coordinates=False):
                    if ((pid and str(getattr(agent, "platform_id", "")) == pid)
                            or (vid and str(getattr(agent, "agent_id", "")) == vid)):
                        return {
                            "vehicle_id": str(getattr(agent, "agent_id", vid)),
                            "platform_id": pid,
                            "x": float(agent.x),
                            "y": float(agent.y),
                            "yaw": float(agent.heading),
                            "speed": float(agent.speed),
                            "source": "主界面 A 编辑值",
                        }
            except Exception:
                pass

        for agent in self.cfg.raw.get("vehicles", {}).get("agents", []) or []:
            if not isinstance(agent, dict):
                continue
            if not ((pid and str(agent.get("platform_id", "")) == pid)
                    or (vid and str(agent.get("id", "")) == vid)):
                continue
            initial = agent.get("initial") or {}
            if all(k in initial for k in ("x", "y", "heading", "speed")):
                if self._origin() is None:
                    return None
                gx, gy = float(initial['x']), float(initial['y'])
                return {
                    "vehicle_id": str(agent.get("id", vid)),
                    "platform_id": str(agent.get("platform_id", pid)),
                    "x": gx,
                    "y": gy,
                    "yaw": float(initial["heading"]),
                    "speed": float(initial["speed"]),
                    "source": "YAML 中的运行时 A",
                }
        return None

    def _history_params(self) -> Tuple[int, float, float]:
        algorithm = self.cfg.raw.get("algorithm", {}) or {}
        scenario = self.cfg.raw.get("scenario", {}) or {}
        steps = int(algorithm.get("pluto_history_steps", scenario.get("pluto_history_steps", 21)))
        dt = float(algorithm.get("pluto_history_dt_sec", scenario.get("pluto_history_dt_sec", 0.1)))
        accel = float(
            algorithm.get("initial_history_accel_mps2", scenario.get("initial_history_accel_mps2", 2.0))
        )
        return steps, dt, accel

    def _refresh_start_target(self, silent: bool = False) -> None:
        initial = self._selected_initial_pose()
        if initial is None:
            self.start_target_pose=None
            self.start_target_diag=None
            self.lab_initial21.setText('当前平台未绑定/配置 initial(21帧)。')
            self.lab_start0.setText('—')
            self._clear_start_overlays()
            if not silent:
                QMessageBox.warning(self,'没有 initial','请先在主界面绑定平台 BV 并配置 initial。')
            return
        origin=self._origin()
        if origin is None:
            self.lab_start0.setText('缺少真实 PKL runtime_origin，无法求 A0。')
            return
        steps, dt, accel=self._history_params()
        try:
            start_a,diag=self.geo.autoscene_start_pose_from_initial(
                initial['x'],initial['y'],initial['yaw'],initial['speed'],
                history_steps=steps,history_dt_sec=dt,accel_mps2=accel)
        except Exception as exc:
            self.start_target_pose=None
            self.start_target_diag=None
            self._clear_start_overlays()
            self.lab_start0.setText('第0帧回退失败：%s'%exc)
            if not silent: QMessageBox.critical(self,'回退失败',str(exc))
            return
        ax,ay=start_a['x'], start_a['y']
        self.start_target_pose={'x':ax,'y':ay,'yaw':start_a['yaw'],'speed':0.0}
        self.start_target_diag=diag
        ix,iy=initial['x'], initial['y']
        self.lab_initial21.setText(
            '%s (%s)\nA21=(%.6f,%.6f)；yaw=%.9f rad；v21=%.4f m/s'
            % (initial['vehicle_id'],initial['source'],ix,iy,initial['yaw'],initial['speed'])
        )
        self.lab_start0.setText(
            'A0=(%.6f,%.6f)；yaw0=%.9f rad；v0=0；'
            '回退距离=%.4f m（%d点、dt=%.3f、a=%.3f）'
            % (ax,ay,start_a['yaw'],diag['backtrack_distance_m'],
               diag['history_steps'],diag['history_dt_sec'],diag['accel_mps2'])
        )
        # Both views are now in A; no extra translation at display time.
        self.map_view.set_start_target_pose(ax,ay,start_a['yaw'])
        self._set_parent_overlay('start_target',ax,ay,start_a['yaw'],
                                 '第0帧 A 目标','start_target')
        self._update_alignment_error()

    def _clear_start_overlays(self):
        parent = self.parent()
        view = getattr(parent,'view',None)
        if view is not None and hasattr(view,'clear_pose_overlay'):
            view.clear_pose_overlay('alignment:start_target')
        if hasattr(self.map_view,'clear_marker'):
            self.map_view.clear_marker('start_target')

    def _update_alignment_error(self) -> None:
        target=self.start_target_pose
        if target is None:
            self.lab_align_error.setText('尚无第0帧 A 目标。')
            return
        if self.latest_sample is None:
            self.lab_align_error.setText('等待 GNSS/INS 实时平台状态；绿色=A0 目标。')
            return
        if self._origin() is None:
            return
        e=self.latest_sample['earth']
        t=self._runtime_transform()
        ax,ay=t.point_g_to_a(e['x'],e['y'])
        yaw=t.yaw_g_to_a(e['yaw'])
        dx,dy=ax-target['x'],ay-target['y']
        dyaw=self.geo.wrap_rad(yaw-target['yaw'])
        self.lab_align_error.setText(
            '实车 A=(%.6f,%.6f),yaw=%.9f rad；目标 A0=(%.6f,%.6f)\n'
            'A误差 Δx=%+.4f m, Δy=%+.4f m, 距离=%.4f m, Δyaw=%+.3f°。'
            '摆车达到现场误差要求并停止独立 UDP 后，再启动 hybrid。'
            % (ax,ay,yaw,target['x'],target['y'],dx,dy,math.hypot(dx,dy),math.degrees(dyaw))
        )

    def _save_yaml(self) -> None:
        if not self._solved_this_session:
            QMessageBox.warning(self,'禁止误保存',
                '尚未在当前窗口执行“冻结 G + A 标定”或者候选值已被手工修改。'
                '请重新求解后保存；旧 YAML 的 frame 可能与新配置不同。')
            return
        origin=self._origin()
        path=getattr(self.cfg,'path',None)
        if origin is None or not path:
            QMessageBox.warning(self,'不能保存','请在主 GUI 加载真实 PKL，并确定 YAML 路径。')
            return
        tf=self._candidate()
        runtime=self._runtime_transform()
        answer=QMessageBox.question(
            self,'保存新的场地标定',
            '保存：%s\n\n当前场景 G→A: tx=%.12f,ty=%.12f,yaw=%.12f\n'
            '自动记录参考场景信息，切换 PKL 时修正平移量。\n'
            '保存后请重新准备/重启 worker，再允许运动平台执行。'
            % (path,tf.tx,tf.ty,tf.yaw),
            QMessageBox.Yes|QMessageBox.No,QMessageBox.No)
        if answer!=QMessageBox.Yes: return
        platform=self.cfg.raw.setdefault('platform',{})
        before=copy.deepcopy(platform)
        platform['earth_to_autoscene']={'tx':float(tf.tx),'ty':float(tf.ty),'yaw':float(tf.yaw)}
        platform['earth_to_autoscene_frame']='reference_scenario'
        platform['calibration_reference_origin']={'x':float(origin[0]), 'y':float(origin[1])}
        platform['calibration_reference_dataset'] = str(self.cfg.scenario.get('dataset_path', ''))
        try:
            backup = None
            if os.path.isfile(path):
                backup = path + '.before_GA_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.bak'
                shutil.copy2(path, backup)
            self.cfg.save(path)
        except Exception as exc:
            platform.clear(); platform.update(before)
            QMessageBox.critical(self,'保存失败',str(exc)); return
        parent=self.parent()
        if parent is not None and hasattr(parent,'refresh_all'):
            parent.refresh_all()
        QMessageBox.information(self,'保存成功',
            '已保存本场景 G→A 和参考场景元数据；其它 PKL 将自动获得其对应 G→A。'
            '请重新执行“准备 BV”并重启 worker。\n备份：%s' % (backup or '原文件不存在'))

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._freshness_timer.start()
        self._populate_platforms()
        self._load_parent_scene_map(silent=True)
        self._refresh_start_target(silent=True)
        self._toggle_parent_precision_mode(self.chk_precision.isChecked())

    def closeEvent(self, event) -> None:
        self._freshness_timer.stop()
        self._toggle_parent_precision_mode(False)
        self._stop_direct_udp()
        self._clear_parent_overlays()
        super().closeEvent(event)