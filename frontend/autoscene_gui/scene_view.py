"""场景 2D 俯视图：QGraphicsView 画地图与车辆，virtual BV 可选中、拖拽移动、拖手柄旋转。

坐标：world (m, y 向上) → scene (px, y 向下)，1 m = PX_PER_M px。
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

from PyQt5.QtCore import QEvent, QPoint, QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen, QPolygonF
from PyQt5.QtWidgets import (
    QGraphicsEllipseItem, QGraphicsItem, QGraphicsLineItem, QGraphicsScene,
    QGraphicsSimpleTextItem, QGraphicsView,
)

from .scenario_loader import Scene, SceneObject

PX_PER_M = 12.0

# v2: 对比度提升 — 背景车辆加深、道路与背景区分更明显、平台车橙色更饱和
C_BG = QColor("#f5f5f2")
C_BG_DEMO = QColor("#eceee8")
C_ROAD = QColor("#cccec8")          # was #d9dad6 — 加深与背景的区分
C_EDGE = QColor("#a8aaa4")          # was #bfc1bb — 路缘更清晰
C_DASH = QColor("#f4f4f0")
C_GRID = QColor(0, 0, 0, 14)
C_EGO = QColor("#2f6fbf")
C_EGO_D = QColor("#1b4c85")
C_SLOT = QColor("#f2d4a8")          # was #f0d7b4 — 稍暖，与平台橙区分更明显
C_SLOT_D = QColor("#a3853f")
C_PLAT = QColor("#e07818")          # was #e88a2c — 更饱和的橙色
C_PLAT_D = QColor("#8b4a0e")        # was #a35c11
C_VIRT = QColor("#5aa06a")
C_VIRT_D = QColor("#33714a")
C_BG_CAR = QColor("#acada5")        # was #c9c9c3 — 深 15%，在浅色背景上可见
C_BG_CAR_D = QColor("#6e6e68")      # was #8a8a84

# 预启动对齐 overlay：只用于可视化，不参与车辆状态、不写回配置。
C_ALIGN_START = QColor("#0a9b68")    # initial(21帧) 自动回退得到的 0 帧目标
C_ALIGN_ACTUAL = QColor("#f05a28")   # 实物平台 GNSS/INS 经 G→A 后再转换到主界面 U 的实时姿态
C_ALIGN_CALIB = QColor("#315efb")    # 单点标定 A 目标


def w2s(x: float, y: float) -> QPointF:
    return QPointF(x * PX_PER_M, -y * PX_PER_M)


def s2w(p: QPointF) -> tuple:
    return (p.x() / PX_PER_M, -p.y() / PX_PER_M)


class VehicleItem(QGraphicsItem):
    """一台车。kind: ego / slot / platform / virtual / background。"""

    def __init__(self, kind: str, key: str, length: float, width: float, view: "SceneView"):
        super().__init__()
        self.kind = kind
        self.key = key
        self.length = max(length, 1.2)
        self.width = max(width, 0.8)
        self.view = view
        self.heading = 0.0
        self.label = key
        self.setZValue(20 if kind == "virtual" else 15)
        self.setAcceptHoverEvents(True)
        if view.can_move(kind):
            self.setFlag(QGraphicsItem.ItemIsMovable, True)
            self.setCursor(Qt.SizeAllCursor)
        else:
            self.setCursor(Qt.PointingHandCursor)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)

    # 几何 -------------------------------------------------------------
    def boundingRect(self) -> QRectF:
        if self.view.precision_pose_mode:
            a = 1.8 * PX_PER_M
            return QRectF(-0.7 * PX_PER_M, -0.7 * PX_PER_M,
                          a + 0.7 * PX_PER_M, 1.4 * PX_PER_M)
        w = self.length * PX_PER_M
        h = self.width * PX_PER_M
        return QRectF(-w / 2 - 3, -h / 2 - 3, w + 6, h + 6)

    def set_world_pose(self, x: float, y: float, heading: float) -> None:
        self.heading = heading
        self.setRotation(-math.degrees(heading))
        self.setPos(w2s(x, y))

    def world_pose(self) -> tuple:
        x, y = s2w(self.pos())
        return (x, y, self.heading)

    def colors(self) -> tuple:
        return {
            "ego": (C_EGO, C_EGO_D),
            "slot": (C_SLOT, C_SLOT_D),
            "platform": (C_PLAT, C_PLAT_D),
            "virtual": (C_VIRT, C_VIRT_D),
        }.get(self.kind, (C_BG_CAR, C_BG_CAR_D))

    def paint(self, painter: QPainter, option, widget=None) -> None:  # noqa: ARG002
        fill, line = self.colors()
        w = self.length * PX_PER_M
        h = self.width * PX_PER_M
        selected = self.isSelected()
        painter.setRenderHint(QPainter.Antialiasing, True)
        if self.view.precision_pose_mode:
            # Render the same QGraphicsItem as a thin center cross + nose axis.
            # Its world pose, dragging, rotation handle and signals are unchanged.
            pen = QPen(line, 0)  # cosmetic 1-pixel stroke at every zoom
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            a = 0.55 * PX_PER_M
            painter.drawLine(QPointF(-a, 0), QPointF(a, 0))
            painter.drawLine(QPointF(0, -a), QPointF(0, a))
            nose = 1.4 * PX_PER_M
            painter.drawLine(QPointF(0, 0), QPointF(nose, 0))
            painter.drawLine(QPointF(nose, 0), QPointF(nose - 0.3*PX_PER_M, -0.23*PX_PER_M))
            painter.drawLine(QPointF(nose, 0), QPointF(nose - 0.3*PX_PER_M, 0.23*PX_PER_M))
            return
        painter.setPen(QPen(line, 2.4 if selected else 1.3))
        painter.setBrush(QBrush(fill))
        painter.drawRoundedRect(QRectF(-w / 2, -h / 2, w, h), 3, 3)
        # 车头指示
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(line, 1.2))
        nose = QPolygonF([
            QPointF(w / 2 - h * 0.35, -h * 0.3),
            QPointF(w / 2 - 2, 0.0),
            QPointF(w / 2 - h * 0.35, h * 0.3),
        ])
        painter.drawPolyline(nose)

    # 交互 -------------------------------------------------------------
    def itemChange(self, change, value):
        if (change == QGraphicsItem.ItemPositionHasChanged and self.view.can_move(self.kind)
                and self.view.vehicles.get(self.key) is self
                and not self.view._applying_live_state):
            self.view.vehicle_moved.emit(self.key, *s2w(self.pos()), self.heading)
            self.view.update_label(self)
            self.view.refresh_handle()
        if change == QGraphicsItem.ItemSelectedHasChanged:
            self.view.refresh_handle()
        return super().itemChange(change, value)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        self.view.vehicle_clicked.emit(self.kind, self.key)


class RotateHandle(QGraphicsEllipseItem):
    """选中可编辑车辆后出现的旋转手柄。"""

    R = 7.0

    def __init__(self, view: "SceneView"):
        super().__init__(-self.R, -self.R, self.R * 2, self.R * 2)
        self.view = view
        self.target: Optional[VehicleItem] = None
        self.setBrush(QBrush(QColor("#ffffff")))
        self.setPen(QPen(C_EGO, 2))
        self.setZValue(40)
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self.setCursor(Qt.CrossCursor)
        self.setVisible(False)
        self.line = QGraphicsLineItem()
        self.line.setPen(QPen(C_EGO, 1.4, Qt.DashLine))
        self.line.setZValue(39)
        self.line.setVisible(False)

    def attach(self, item: Optional[VehicleItem]) -> None:
        self.target = item
        vis = item is not None
        self.setVisible(vis)
        self.line.setVisible(vis)
        if item is None:
            return
        x, y, heading = item.world_pose()
        d = max(item.length * 0.75 + 1.4, 3.2)
        hx, hy = x + math.cos(heading) * d, y + math.sin(heading) * d
        self.setPos(w2s(hx, hy))
        self.line.setLine(w2s(x, y).x(), w2s(x, y).y(), w2s(hx, hy).x(), w2s(hx, hy).y())

    def mouseMoveEvent(self, event):
        if self.target is None or not self.view.editing_enabled:
            return
        cx, cy, _ = self.target.world_pose()
        wx, wy = s2w(event.scenePos())
        heading = math.atan2(wy - cy, wx - cx)
        self.target.heading = heading
        self.target.setRotation(-math.degrees(heading))
        self.attach(self.target)
        self.view.vehicle_moved.emit(self.target.key, cx, cy, heading)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.target is not None and self.view.editing_enabled:
            event.accept()
        else:
            event.ignore()

    def mouseReleaseEvent(self, event):
        event.accept()


class SceneView(QGraphicsView):
    vehicle_clicked = pyqtSignal(str, str)               # kind, key
    vehicle_moved = pyqtSignal(str, float, float, float)  # key, x, y, heading
    edit_finished = pyqtSignal()
    map_clicked = pyqtSignal(float, float)               # world x, y（放置模式）
    zoom_changed = pyqtSignal(int)                       # 相对 fit 视图百分比

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(QBrush(C_BG))
        self.editing_enabled = True
        self.builder_mode = False
        self.precision_pose_mode = False
        self.label_scale = 1.0
        self.place_mode = False
        self.show_trails = True
        self.show_ids = True
        self.show_grid = False
        self.scene_data: Optional[Scene] = None
        self.vehicles: Dict[str, VehicleItem] = {}
        self.vehicle_labels: Dict[str, QGraphicsSimpleTextItem] = {}
        # 运行期轨迹 overlay。planner=runtime 当前规划；command=platform_comm 实际下发轨迹。
        # 轨迹只用于显示，绝不驱动车身位置；车身始终由 /world/state 更新。
        self.planned_paths: Dict[str, Dict[str, QGraphicsItem]] = {
            "planner": {},
            "command": {},
        }
        # 独立的姿态标记层。pose_overlays 保存 world/数据集全局坐标，
        # pose_overlay_items 仅保存当前 QGraphicsScene 中的图元。
        # set_scene_data() 重建场景时会自动恢复，不会修改 SceneObject/live pose。
        self.pose_overlays: Dict[str, tuple] = {}
        self.pose_overlay_items: Dict[str, List[QGraphicsItem]] = {}
        self.handle = RotateHandle(self)
        self._scene.addItem(self.handle.line)
        self._scene.addItem(self.handle)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        # 10 Hz world/state 下只重绘发生变化的区域，避免每辆车移动都刷新整张地图。
        self.setViewportUpdateMode(QGraphicsView.BoundingRectViewportUpdate)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        # macOS 触控板通常发送 NativeGesture；部分平台/驱动发送 PinchGesture。
        # 两种都注册/处理，保证双指捏合不会退化成普通滚动。
        self.grabGesture(Qt.PinchGesture)
        self.viewport().grabGesture(Qt.PinchGesture)
        self._fit_scale = 1.0
        self._applying_live_state = False
        self._clearing_scene = False
        # Runtime poses are cached independently from editable initial Scene data.
        # Any UI redraw/rebuild re-applies these poses while live mode is active.
        self.live_mode = False
        self.live_poses: Dict[str, tuple] = {}
        self._is_panning = False
        self._space_down = False
        self._pan_start = QPoint()

    # ---- 绘制 --------------------------------------------------------
    def set_scene_data(self, scene: Scene, platform_bind: Dict[str, str],
                       virtuals: List[dict], preserve_view: bool = False) -> None:
        had_view = bool(preserve_view and self.scene_data is not None
                        and self._scene.sceneRect().isValid())
        old_transform = self.transform()
        old_center = self.mapToScene(self.viewport().rect().center()) if had_view else QPointF()
        self.scene_data = scene
        self.setBackgroundBrush(QBrush(C_BG_DEMO if scene.is_demo else C_BG))
        self._clearing_scene = True
        self._scene.clear()
        self.vehicles.clear()
        self.vehicle_labels.clear()
        self.planned_paths = {"planner": {}, "command": {}}
        # QGraphicsScene.clear() 已销毁旧 overlay 图元；保留 pose_overlays 数据，
        # 稍后在新场景上恢复。
        self.pose_overlay_items.clear()
        self.handle = RotateHandle(self)
        self._scene.addItem(self.handle.line)
        self._scene.addItem(self.handle)
        self._clearing_scene = False

        self._draw_map(scene)
        if self.show_grid:
            self._draw_grid(scene)
        if self.show_trails:
            self._draw_trails(scene, platform_bind)

        self._add_vehicle("ego", "ego", scene.ego, "EGO")
        replaced_tracks = {str(v.get("track_id")) for v in virtuals}
        for obj in scene.objects:
            if str(obj.track_id) in replaced_tracks:
                continue
            if obj.physical:
                ep = platform_bind.get(obj.track_id)
                kind = "platform" if ep else "slot"
                label = ("%s · %s" % (ep.replace("platform_", "P-"), obj.track_id)) if ep \
                    else ("实物车位 · %s" % obj.track_id)
                self._add_vehicle(kind, obj.track_id, obj, label)
            else:
                self._add_vehicle("background", obj.track_id, obj, obj.track_id)
        for v in virtuals:
            item = VehicleItem("virtual", v["key"], 4.6, 2.0, self)
            item.set_world_pose(v["x"], v["y"], v["heading"])
            item.label = v["key"]
            self._scene.addItem(item)
            self.vehicles[v["key"]] = item
            if self.show_ids:
                self.vehicle_labels[v["key"]] = self._add_label(
                    v["key"], v["x"], v["y"] + item.width / 2 + 1.1, C_VIRT_D)

        # set_scene_data() rebuilds every graphics item.  During an experiment,
        # restore the latest runtime pose immediately instead of snapping vehicles
        # back to their editable initial SceneObject coordinates.
        if self.live_mode and self.live_poses:
            for key, live in list(self.live_poses.items()):
                item = self.vehicles.get(str(key))
                if item is None:
                    continue
                x, y, heading, valid = live
                item.set_world_pose(float(x), float(y), float(heading))
                opacity = 1.0 if valid else 0.22
                item.setOpacity(opacity)
                label_item = self.vehicle_labels.get(str(key))
                if label_item is not None:
                    p = w2s(float(x), float(y) + item.width / 2 + 1.1)
                    label_item.setPos(
                        p.x() - label_item.boundingRect().width() / 2,
                        p.y() - 18,
                    )
                    label_item.setOpacity(opacity)

        # 标定/起步对齐标记与车辆 live 状态完全独立。
        self._restore_pose_overlays()
        self._apply_precision_labels()

        rect = self._scene.itemsBoundingRect().adjusted(-60, -60, 60, 60)
        self._scene.setSceneRect(rect)
        if had_view:
            self.setTransform(old_transform)
            self.centerOn(old_center)
            self._emit_zoom()
        else:
            # 构建窗口时 viewport 尚未得到最终尺寸；延后一轮可避免首次加载空白。
            QTimer.singleShot(0, self.fit)

    def _add_vehicle(self, kind: str, key: str, obj: SceneObject, label: str) -> None:
        item = VehicleItem(kind, key, obj.length, obj.width, self)
        item.set_world_pose(obj.x, obj.y, obj.heading)
        item.label = label
        self._scene.addItem(item)
        self.vehicles[key] = item
        if self.show_ids:
            color = item.colors()[1]
            self.vehicle_labels[key] = self._add_label(
                label, obj.x, obj.y + obj.width / 2 + 1.1, color)

    def _add_label(self, text: str, x: float, y: float,
                   color: QColor) -> QGraphicsSimpleTextItem:
        t = QGraphicsSimpleTextItem(text)
        f = QFont("Monospace")
        f.setPixelSize(round(12 * self.label_scale))
        f.setStyleHint(QFont.Monospace)
        t.setFont(f)
        t.setBrush(QBrush(color))
        p = w2s(x, y)
        t.setPos(p.x() - t.boundingRect().width() / 2, p.y() - 18)
        t.setZValue(30)
        t.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        t.setAcceptedMouseButtons(Qt.NoButton)
        self._scene.addItem(t)
        return t

    def _draw_map(self, scene: Scene) -> None:
        if scene.lanes or scene.boundaries:
            for line in scene.boundaries:
                self._polyline(line, QPen(C_EDGE, 3.0), z=2)
            for line in scene.lanes:
                pen = QPen(C_DASH, 2.2, Qt.DashLine)
                self._polyline(line, pen, z=4)
            for line in scene.crosswalks:
                self._polyline(line, QPen(C_DASH, 6.0, Qt.DotLine), z=3)
            # 车道底色：按中心线画粗灰带
            for line in scene.lanes:
                self._polyline(line, QPen(C_ROAD, 3.6 * PX_PER_M, Qt.SolidLine,
                                          Qt.FlatCap, Qt.RoundJoin), z=1)
        else:
            # 没有 map_features：按各车轨迹方向画走廊
            for obj in [scene.ego] + scene.objects:
                if len(obj.trail) >= 2:
                    self._polyline(obj.trail, QPen(C_ROAD, 3.4 * PX_PER_M, Qt.SolidLine,
                                                   Qt.FlatCap, Qt.RoundJoin), z=1)

    def _draw_grid(self, scene: Scene) -> None:
        rect = self._scene.itemsBoundingRect().adjusted(-40, -40, 40, 40)
        step = 5 * PX_PER_M
        pen = QPen(C_GRID, 1)
        x = math.floor(rect.left() / step) * step
        while x < rect.right():
            self._scene.addLine(x, rect.top(), x, rect.bottom(), pen).setZValue(0.5)
            x += step
        y = math.floor(rect.top() / step) * step
        while y < rect.bottom():
            self._scene.addLine(rect.left(), y, rect.right(), y, pen).setZValue(0.5)
            y += step

    def _draw_trails(self, scene: Scene, platform_bind: Dict[str, str]) -> None:
        if len(scene.ego.trail) >= 2:
            self._polyline(scene.ego.trail, QPen(C_EGO, 1.8), z=6)
        for obj in scene.objects:
            if len(obj.trail) < 2:
                continue
            color = C_PLAT if platform_bind.get(obj.track_id) else C_BG_CAR_D
            pen = QPen(color, 1.6, Qt.DashLine)
            self._polyline(obj.trail, pen, z=6)

    def _polyline(self, pts, pen: QPen, z: float = 5) -> None:
        # A nuPlan map can contain thousands of segments. Keep one graphics
        # item per polyline instead of per segment to make dragging responsive.
        path = QPainterPath()
        if not pts:
            return
        path.moveTo(w2s(pts[0][0], pts[0][1]))
        for a, b in zip(pts, pts[1:]):
            path.lineTo(w2s(b[0], b[1]))
        item = self._scene.addPath(path, pen)
        item.setAcceptedMouseButtons(Qt.NoButton)
        item.setZValue(z)

    # ---- 精确中心对齐显示（纯视图模式，不修改 SceneObject/车辆位置） ----
    def _apply_precision_labels(self) -> None:
        for label in self.vehicle_labels.values():
            label.setVisible(not self.precision_pose_mode)
        for item in self.vehicles.values():
            item.update()
        self.viewport().update()

    def set_precision_pose_mode(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled == self.precision_pose_mode:
            return
        for item in self.vehicles.values():
            item.prepareGeometryChange()
        self.precision_pose_mode = enabled
        self._apply_precision_labels()

    # ---- 标定 / 起步对齐 overlay -------------------------------------
    @staticmethod
    def _pose_overlay_color(kind: str) -> QColor:
        return {
            "start_target": C_ALIGN_START,
            "platform_actual": C_ALIGN_ACTUAL,
            "calibration_target": C_ALIGN_CALIB,
        }.get(str(kind), C_ALIGN_CALIB)

    def _remove_pose_overlay_items(self, key: str) -> None:
        items = self.pose_overlay_items.pop(str(key), [])
        for item in items:
            try:
                if item is not None and item.scene() is self._scene:
                    self._scene.removeItem(item)
            except RuntimeError:
                # 场景重建后 Python wrapper 可能已对应被 Qt 删除的 C++ 对象。
                pass

    def _draw_pose_overlay(self, key: str) -> None:
        key = str(key)
        self._remove_pose_overlay_items(key)
        data = self.pose_overlays.get(key)
        if data is None:
            return
        x, y, heading, label, kind = data
        color = self._pose_overlay_color(kind)
        center = w2s(float(x), float(y))
        # Small fixed-screen glyph.  Do not let the pose marker grow/shrink with map zoom.
        radius_px = 4.0
        pen_style = Qt.DashLine if kind == "start_target" else Qt.SolidLine

        # Fixed-screen marker: use a local rectangle centred on the item origin.
        # Its scene position is the exact pose centre; map zoom must not move it.
        ring = QGraphicsEllipseItem(
            -radius_px, -radius_px, 2.0 * radius_px, 2.0 * radius_px,
        )
        ring.setPen(QPen(color, 2.6, pen_style))
        ring.setBrush(QBrush(Qt.NoBrush))
        ring.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        ring.setPos(center)
        ring.setZValue(70)
        ring.setAcceptedMouseButtons(Qt.NoButton)
        self._scene.addItem(ring)

        arrow_len_m = 4.5
        end = w2s(
            float(x) + arrow_len_m * math.cos(float(heading)),
            float(y) + arrow_len_m * math.sin(float(heading)),
        )
        arrow = self._scene.addLine(
            center.x(), center.y(), end.x(), end.y(),
            QPen(color, 2.8, pen_style),
        )
        arrow.setZValue(71)
        arrow.setAcceptedMouseButtons(Qt.NoButton)

        # 车头方向再画一个小横杠，避免仅靠长线难以辨认箭头端。
        side = 0.55
        hx = float(x) + arrow_len_m * math.cos(float(heading))
        hy = float(y) + arrow_len_m * math.sin(float(heading))
        left = w2s(
            hx + side * math.cos(float(heading) + math.pi / 2.0),
            hy + side * math.sin(float(heading) + math.pi / 2.0),
        )
        right = w2s(
            hx + side * math.cos(float(heading) - math.pi / 2.0),
            hy + side * math.sin(float(heading) - math.pi / 2.0),
        )
        head = self._scene.addLine(left.x(), left.y(), right.x(), right.y(), QPen(color, 2.8))
        head.setZValue(71)
        head.setAcceptedMouseButtons(Qt.NoButton)

        text_item = QGraphicsSimpleTextItem(str(label or key))
        font = QFont("Monospace")
        font.setPixelSize(round(12 * self.label_scale))
        font.setStyleHint(QFont.Monospace)
        text_item.setFont(font)
        text_item.setBrush(QBrush(color))
        text_item.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        text_item.setAcceptedMouseButtons(Qt.NoButton)
        text_item.setZValue(72)
        text_item.setPos(center.x() + radius_px + 4, center.y() - radius_px - 14)
        self._scene.addItem(text_item)

        self.pose_overlay_items[key] = [ring, arrow, head, text_item]

    def _restore_pose_overlays(self) -> None:
        for key in list(self.pose_overlays):
            self._draw_pose_overlay(key)

    def set_pose_overlay(self, key: str, x: float, y: float, heading: float,
                         label: str = "", kind: str = "calibration_target") -> None:
        """在主地图画一个独立姿态标记。

        坐标必须是 UI 使用的 U/原始数据集全局坐标。该函数只画图，
        不修改车辆、initial、live pose 或 ROS 状态。
        """
        values = (float(x), float(y), float(heading))
        if not all(math.isfinite(v) for v in values):
            return
        self.pose_overlays[str(key)] = (values[0], values[1], values[2], str(label), str(kind))
        self._draw_pose_overlay(str(key))
        self.viewport().update()

    def clear_pose_overlay(self, key: str) -> None:
        self.pose_overlays.pop(str(key), None)
        self._remove_pose_overlay_items(str(key))
        self.viewport().update()

    def clear_pose_overlays(self, prefix: Optional[str] = None) -> None:
        """清除标定/对齐图层；prefix 只清除指定命名空间。"""
        keys = list(self.pose_overlays)
        if prefix is not None:
            keys = [key for key in keys if str(key).startswith(str(prefix))]
        for key in keys:
            self.pose_overlays.pop(key, None)
            self._remove_pose_overlay_items(key)
        self.viewport().update()

    # ---- 交互 --------------------------------------------------------
    def refresh_handle(self) -> None:
        if self._clearing_scene:
            return
        if not self.editing_enabled:
            self.handle.attach(None)
            return
        sel = [i for i in self._scene.selectedItems()
               if isinstance(i, VehicleItem) and self.can_move(i.kind)]
        self.handle.attach(sel[0] if sel else None)

    def select_key(self, key: str) -> None:
        for k, item in self.vehicles.items():
            item.setSelected(k == key)
        self.refresh_handle()

    def set_place_mode(self, on: bool) -> None:
        self.place_mode = bool(on and self.editing_enabled)
        self.setDragMode(QGraphicsView.NoDrag if self.place_mode else QGraphicsView.RubberBandDrag)
        self.setCursor(Qt.CrossCursor if self.place_mode else Qt.ArrowCursor)

    def set_editing_enabled(self, enabled: bool) -> None:
        """运行期间锁定车辆编辑，避免界面配置与 ROS 执行参数分叉。"""
        self.editing_enabled = bool(enabled)
        if not self.editing_enabled:
            self.set_place_mode(False)
        for item in self.vehicles.values():
            movable = self.can_move(item.kind)
            item.setFlag(QGraphicsItem.ItemIsMovable, movable)
            item.setCursor(Qt.SizeAllCursor if movable else Qt.PointingHandCursor)
        self.refresh_handle()

    def can_move(self, kind: str) -> bool:
        return self.editing_enabled and (kind in ("ego", "slot", "platform", "virtual")
                                         or self.builder_mode)

    def set_builder_mode(self, enabled: bool) -> None:
        self.builder_mode = bool(enabled)
        self.set_editing_enabled(self.editing_enabled)

    def update_label(self, item: VehicleItem) -> None:
        label = self.vehicle_labels.get(item.key)
        if label is not None:
            x, y, _ = item.world_pose()
            p = w2s(x, y + item.width / 2 + 1.1)
            label.setPos(p.x() - label.boundingRect().width() / 2, p.y() - 18)

    def _planned_path_color(self, key: str) -> QColor:
        item = self.vehicles.get(str(key))
        if item is None:
            return C_BG_CAR_D
        return item.colors()[1]

    def clear_planned_trajectory(self, key: str, layer: Optional[str] = None) -> None:
        """清除一辆车的规划轨迹显示，不改变车辆状态。"""
        layers = (layer,) if layer is not None else tuple(self.planned_paths.keys())
        for layer_name in layers:
            bucket = self.planned_paths.get(layer_name)
            if bucket is None:
                continue
            item = bucket.pop(str(key), None)
            if item is not None and item.scene() is self._scene:
                self._scene.removeItem(item)

    def clear_planned_trajectories(self, layer: Optional[str] = None) -> None:
        """清除 planner/command overlay；地图和车辆图元保持不变。"""
        layers = (layer,) if layer is not None else tuple(self.planned_paths.keys())
        for layer_name in layers:
            bucket = self.planned_paths.get(layer_name)
            if bucket is None:
                continue
            for item in list(bucket.values()):
                if item is not None and item.scene() is self._scene:
                    self._scene.removeItem(item)
            bucket.clear()

    def update_planned_trajectory(self, key: str, points, layer: str = "planner") -> bool:
        """更新一辆车的未来轨迹 overlay。

        points 使用 UI/world 全局坐标 [(x, y), ...]。本函数只画线，
        不调用 set_world_pose，因此不会把规划轨迹冒充 actual state。
        """
        if layer not in self.planned_paths:
            raise ValueError("trajectory layer must be planner or command")
        key = str(key)
        xy = []
        for point in points or []:
            try:
                x, y = float(point[0]), float(point[1])
            except (TypeError, ValueError, IndexError):
                return False
            if not math.isfinite(x) or not math.isfinite(y):
                return False
            xy.append((x, y))
        if len(xy) < 2:
            self.clear_planned_trajectory(key, layer=layer)
            return False

        self.clear_planned_trajectory(key, layer=layer)
        path = QPainterPath()
        path.moveTo(w2s(xy[0][0], xy[0][1]))
        for x, y in xy[1:]:
            path.lineTo(w2s(x, y))

        color = self._planned_path_color(key)
        pen = QPen(color, 2.6 if layer == "command" else 2.0,
                   Qt.SolidLine if layer == "command" else Qt.DashLine,
                   Qt.RoundCap, Qt.RoundJoin)
        item = self._scene.addPath(path, pen)
        item.setAcceptedMouseButtons(Qt.NoButton)
        item.setZValue(11.0 if layer == "command" else 10.0)
        item.setOpacity(0.95 if layer == "command" else 0.72)
        self.planned_paths[layer][key] = item
        return True

    def fit(self) -> None:
        if self._scene.sceneRect().isValid():
            self.resetTransform()
            self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)
            self._fit_scale = max(self.transform().m11(), 1e-9)
            self.zoom_changed.emit(100)

    def _emit_zoom(self) -> None:
        relative = self.transform().m11() / max(self._fit_scale, 1e-9)
        self.zoom_changed.emit(int(round(relative * 100)))

    def zoom_by(self, factor: float) -> None:
        """以鼠标所在位置为锚点缩放，并限制 CAD 视图的极端倍率。"""
        current = max(self.transform().m11(), 1e-9)
        # 不再以“全图 fit 倍率”设 40x 硬上限。数据里有远处背景对象时，
        # fit_scale 很小，40x 仍不足以检查车身/车道细节。改用稳定的绝对上限。
        minimum = max(self._fit_scale * 0.04, 1e-5)
        maximum = 128.0
        target = max(minimum, min(maximum, current * max(float(factor), 1e-6)))
        factor = target / current
        if abs(factor - 1.0) <= 1e-9:
            return
        self.scale(factor, factor)
        self._emit_zoom()

    def zoom_in(self) -> None:
        self.zoom_by(1.25)

    def zoom_out(self) -> None:
        self.zoom_by(0.8)

    def focus_actors(self) -> None:
        """缩放到车辆工作区，忽略数据集中远处的零散地图要素。"""
        items = [item for item in self.vehicles.values()
                 if item.isVisible() and item.kind != "background"]
        if not items:
            self.fit()
            return
        rect = items[0].sceneBoundingRect()
        for item in items[1:]:
            rect = rect.united(item.sceneBoundingRect())
        rect = rect.adjusted(-120, -120, 120, 120)
        self.fitInView(rect, Qt.KeepAspectRatio)
        relative = self.transform().m11() / max(self._fit_scale, 1e-9)
        self.zoom_changed.emit(int(round(relative * 100)))

    def set_live_mode(self, enabled: bool, clear: bool = False) -> None:
        """Enable/disable runtime pose ownership without changing initial Scene data."""
        self.live_mode = bool(enabled)
        if clear:
            self.live_poses.clear()

    def clear_live_state(self) -> None:
        """Forget cached runtime poses when loading/resetting a scenario."""
        self.live_mode = False
        self.live_poses.clear()

    def update_live_vehicle(self, key: str, x: float, y: float, heading: float,
                            valid: bool = True, refresh_handle: bool = True) -> bool:
        """Update and cache a runtime vehicle pose without mutating initial config."""
        key = str(key)
        x = float(x)
        y = float(y)
        heading = float(heading)
        valid = bool(valid)
        item = self.vehicles.get(key)
        if item is None:
            return False

        self.live_poses[key] = (x, y, heading, valid)
        self._applying_live_state = True
        try:
            item.set_world_pose(x, y, heading)
            opacity = 1.0 if valid else 0.22
            item.setOpacity(opacity)
            label = self.vehicle_labels.get(key)
            if label is not None:
                p = w2s(x, y + item.width / 2 + 1.1)
                label.setPos(p.x() - label.boundingRect().width() / 2, p.y() - 18)
                label.setOpacity(opacity)
            # Explicitly invalidate both item and viewport.  Qt normally tracks moved
            # QGraphicsItems automatically, but forcing the repaint removes backend/
            # platform-specific viewport-update ambiguity on the Ubuntu delivery host.
            item.update()
            self._scene.update(item.sceneBoundingRect())
            self.viewport().update()
        finally:
            self._applying_live_state = False
        if refresh_handle:
            self.refresh_handle()
        return True

    def _handle_zoom_gesture(self, event) -> bool:
        """处理 view 或 viewport 收到的原生/Qt 捏合手势。"""
        # Ubuntu 20.04 的 Qt 5.14 没有 QEvent.NativeGesture；macOS 使用的
        # Qt 5.15 才暴露该枚举。用 getattr 保持 Linux 滚轮缩放可用，
        # 同时不影响支持原生触控板手势的平台。
        native_gesture = getattr(QEvent, "NativeGesture", None)
        if native_gesture is not None and event.type() == native_gesture:
            if event.gestureType() == Qt.ZoomNativeGesture:
                # NativeGesture.value() 是本次手势的增量；指数映射在放大、
                # 缩小时对称，且比 1 + value 在负值时更稳定。
                delta = max(-0.7, min(0.7, float(event.value())))
                if abs(delta) > 1e-6:
                    self.zoom_by(math.exp(delta))
                event.accept()
                return True
        elif event.type() == QEvent.Gesture:
            pinch = event.gesture(Qt.PinchGesture)
            if pinch is not None:
                last = float(pinch.lastScaleFactor())
                current = float(pinch.scaleFactor())
                factor = current / last if last > 1e-9 else current
                if factor > 0.0 and abs(factor - 1.0) > 1e-6:
                    self.zoom_by(factor)
                event.accept()
                return True
        return False

    def event(self, event):
        if self._handle_zoom_gesture(event):
            return True
        return super().event(event)

    def viewportEvent(self, event):
        # macOS 通常把触控板手势发给 QGraphicsView 的 viewport 子控件。
        if self._handle_zoom_gesture(event):
            return True
        return super().viewportEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton or (
                event.button() == Qt.LeftButton and self._space_down):
            self._is_panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        if self.place_mode and event.button() == Qt.LeftButton:
            item = self.itemAt(event.pos())
            if not isinstance(item, (VehicleItem, RotateHandle)):
                wx, wy = s2w(self.mapToScene(event.pos()))
                self.map_clicked.emit(wx, wy)
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._is_panning:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._is_panning and event.button() in (Qt.MiddleButton, Qt.LeftButton):
            self._is_panning = False
            self.setCursor(Qt.CrossCursor if self.place_mode else Qt.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)
        if event.button() == Qt.LeftButton and self.editing_enabled:
            self.edit_finished.emit()

    def wheelEvent(self, event):
        # 与 CAD 一致：光标位置保持不动，滚轮连续缩放。
        delta = event.angleDelta().y()
        if delta:
            self.zoom_by(math.pow(1.0016, delta))
        event.accept()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton and self.itemAt(event.pos()) is None:
            self.fit()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._space_down = True
            self.setCursor(Qt.OpenHandCursor)
            event.accept()
            return
        if event.key() in (Qt.Key_F, Qt.Key_0):
            self.fit()
            event.accept()
            return
        if event.key() == Qt.Key_A:
            self.focus_actors()
            event.accept()
            return
        if event.key() in (Qt.Key_Plus, Qt.Key_Equal):
            self.zoom_in()
            event.accept()
            return
        if event.key() == Qt.Key_Minus:
            self.zoom_out()
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._space_down = False
            if not self._is_panning:
                self.setCursor(Qt.CrossCursor if self.place_mode else Qt.ArrowCursor)
            event.accept()
            return
        super().keyReleaseEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # CAD 视图调整窗口时保留中心和倍率；全图复位只由 F/0/按钮触发。