"""AutoScene 上位机主窗口：左配置树 + 中场景可视化 / 页签 + 右运行状态。"""

from __future__ import annotations

import copy
import glob
import html
import json
import math
import os
import re
from datetime import datetime
from typing import Dict, List, Optional

from PyQt5.QtCore import QSettings, Qt, QTimer
from PyQt5.QtGui import QColor, QFont, QTextCursor
from PyQt5.QtWidgets import (
    QAbstractItemView, QAction, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QFrame, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit,
    QApplication, QPushButton, QScrollArea, QSpinBox, QSplitter, QTabWidget, QTableWidget,
    QTableWidgetItem, QToolBar, QVBoxLayout, QWidget,
)

from .config_model import AgentSpec, ExperimentConfig
from . import __version__
from .ros_client import make_client
from .scenario_loader import (DEMO_SCENE, DatasetLoadError, Scene, ScenarioDataset,
                              nearest_polyline_pose)
from .scene_view import SceneView
from .calibration_window import CalibrationWindow
from .theme import APP_STYLE

MONO = QFont("Monospace", 9)
MONO.setStyleHint(QFont.Monospace)


def mono(text: str = "") -> QLabel:
    lab = QLabel(text)
    lab.setFont(MONO)
    return lab


class MainWindow(QMainWindow):
    def __init__(self, config_path: Optional[str] = None, backend: str = "rosbridge",
                 bridge_url: str = "ws://127.0.0.1:9090",
                 local_dataset_path: str = "",
                 remote_dataset_path: str = ""):
        super().__init__()
        self.setWindowTitle("AutoScene Experiment Manager %s" % __version__)
        self.resize(1600, 1000)
        self.setMinimumSize(1000, 650)

        self.cfg = ExperimentConfig.load(config_path) if config_path and os.path.isfile(config_path) \
            else ExperimentConfig()
        if local_dataset_path.strip():
            self.cfg.scenario["dataset_path"] = local_dataset_path.strip()
        supplied_remote_path = remote_dataset_path.strip()
        local_path = str(self.cfg.scenario.get("dataset_path") or "").strip()
        self.ros_same_host = not supplied_remote_path or supplied_remote_path == local_path
        self.remote_dataset_path = "" if self.ros_same_host else supplied_remote_path
        self._last_remote_dataset_path = "" if self.ros_same_host else supplied_remote_path
        self.backend = backend
        self._connected = False
        self._busy_action = ""
        self._persist_ui_state = os.environ.get("AUTOSCENE_DISABLE_SETTINGS") != "1"
        self.dataset: Optional[ScenarioDataset] = None
        self.scene_index = 0
        self.scene: Scene = DEMO_SCENE()
        self.platform_bind: Dict[str, str] = {}     # track_id -> platform_id
        algorithm_cfg = self.cfg.raw.get("algorithm", {})
        self.ego_method = algorithm_cfg.get("ego_method")
        self.adv_method = algorithm_cfg.get("adv_method")
        self._algorithm_selection_dirty = False
        self.virtuals: List[dict] = []              # {key,x,y,heading,speed}
        self._next_virtual = 1
        self.speeds: Dict[str, float] = {}          # ego / t<track> 的初始速度覆盖
        # YAML 中的 initial.speed 只在同一 dataset_path/scenario_index 上回填，
        # 避免切换数据集或场景后把旧 track_id 的速度误套到新场景。
        self._speed_config_context = self._current_speed_context()
        self._initial_speed_config_errors: List[str] = []
        self._edited_track_poses = {}
        self._original_scene = None
        self.ui_scale = 125
        self._mono_font = QFont(MONO)
        self._edit_refresh_timer = QTimer(self)
        self._edit_refresh_timer.setSingleShot(True)
        self._edit_refresh_timer.timeout.connect(self._refresh_pose_tables)
        self.state = {"state": "idle", "step": 0, "sim_time": 0.0,
                      "real_time_factor": 0.0, "prepared": False}
        # Runtime live-state ownership is explicit. Vehicle bodies follow the fused
        # /autoscene/world/state stream during an experiment. Raw MetaDrive topics are
        # retained only as a startup/diagnostic fallback until the first usable world
        # packet is applied; they must never permanently steal ownership from world/state.
        self._runtime_live_enabled = False
        self._world_state_seen = False
        self._raw_sim_ego_seen = False       # compatibility/diagnostic state only
        self._raw_sim_agent_ids = set()      # compatibility/diagnostic state only
        self._live_source_logged = set()
        self._live_vehicle_states: Dict[str, dict] = {}
        self._calibration_window = None
        # scenario_runtime_node is launched only after the UI has already connected to
        # rosbridge. Some rosbridge deployments do not retain custom-message
        # subscriptions created before the publisher exists, so retry until the first
        # world/state frame is actually received.
        self._runtime_stream_retry_count = 0
        self._runtime_stream_retry_timer = QTimer(self)
        self._runtime_stream_retry_timer.setInterval(2000)
        self._runtime_stream_retry_timer.timeout.connect(self._ensure_runtime_stream)

        self._build_toolbar()
        self._build_body()
        self._build_status_bar()
        self._restore_ui_state()
        self.set_ui_scale(self.ui_scale)

        self.client = make_client(backend, bridge_url)
        self.client.connected.connect(self._on_connected)
        self.client.status.connect(self._on_status)
        # world/state is the authoritative vehicle-body stream. It already contains
        # MetaDrive states in simulation and fused actual-platform feedback in hybrid.
        self.client.world.connect(self._on_world)
        # Keep raw topics connected only as a startup/diagnostic fallback. Once one
        # usable world/state packet has been applied, these callbacks no longer move cars.
        self.client.sim_ego.connect(self._on_sim_ego)
        self.client.sim_agents.connect(self._on_sim_agents)
        self.client.planning.connect(self._on_planning_trajectories)
        self.client.platform_command.connect(self._on_platform_command_trajectories)
        self.client.metrics.connect(self._on_metrics)
        self.client.comm.connect(self._on_comm)
        if hasattr(self.client, "ads_decoded"):
            self.client.ads_decoded.connect(self._on_ads_decoded)
        self.client.log.connect(self.log)
        self.client.start()

        self.refresh_all(preserve_view=False)
        self.log("INFO", "当前为示意场景，请选择 dataset_path 后点击「① 加载数据集场景」")
        QTimer.singleShot(0, self._autoload_dataset)

    def _autoload_dataset(self) -> None:
        """启动配置已指向有效本机目录时，自动进入真实场景。"""
        path = self.ed_path.text().strip()
        if path and os.path.exists(path) and self.dataset is None:
            self.load_scenario()

    # ==================================================================
    # 顶部工具栏
    # ==================================================================
    def _build_toolbar(self) -> None:
        tb = QToolBar("主操作")
        tb.setMovable(False)
        tb.setObjectName("mainToolbar")
        self.addToolBar(tb)

        self.act_save = QAction("保存 YAML", self)
        self.act_save.triggered.connect(self.save_yaml)
        self.act_apply = QAction("应用到 rosparam", self)
        self.act_apply.triggered.connect(self.apply_params)
        tb.addAction(self.act_save)
        tb.addAction(self.act_apply)
        tb.addSeparator()

        self.btn_load = QPushButton("① 加载数据集场景")
        self.btn_load.setObjectName("loadButton")
        self.btn_load.clicked.connect(self.load_scenario)
        self.btn_prepare = QPushButton("② 准备 BV")
        self.btn_prepare.setObjectName("prepareButton")
        self.btn_prepare.clicked.connect(self.prepare)
        self.btn_start = QPushButton("▶ 启动实验")
        self.btn_start.setObjectName("primaryButton")
        self.btn_start.clicked.connect(self.start_experiment)
        self.btn_stop = QPushButton("■ 停止")
        self.btn_stop.setObjectName("dangerButton")
        self.btn_stop.clicked.connect(self.stop_experiment)
        self.btn_reset = QPushButton("↺ Reset")
        self.btn_reset.setObjectName("quietButton")
        self.btn_reset.setToolTip("重置 ROS 运行状态，并清除本轮端点绑定和 virtual BV")
        self.btn_reset.clicked.connect(self.reset_experiment)
        for b in (self.btn_load, self.btn_prepare, self.btn_start, self.btn_stop, self.btn_reset):
            tb.addWidget(b)
        tb.addSeparator()

        self.lab_mode = QLabel()
        self.lab_mode.setObjectName("modeBadge")
        self.lab_mode.setFont(MONO)
        self.lab_mode.setMargin(6)
        tb.addWidget(self.lab_mode)
        self.lab_loaded = QLabel()
        self.lab_loaded.setObjectName("dataBadge")
        self.lab_loaded.setFont(MONO)
        self.lab_loaded.setMargin(6)
        tb.addWidget(self.lab_loaded)

        spacer = QWidget()
        spacer.setSizePolicy(spacer.sizePolicy().Expanding, spacer.sizePolicy().Preferred)
        tb.addWidget(spacer)
        self.lab_link = mono("后端未连接")
        self.lab_link.setObjectName("linkBadge")
        tb.addWidget(self.lab_link)

        # A short, dedicated row stays visible even when the main toolbar overflows.
        self.addToolBarBreak()
        self.display_toolbar = QToolBar("字号与规划算法")
        self.display_toolbar.setMovable(False)
        self.display_toolbar.setObjectName("displayToolbar")
        self.addToolBar(self.display_toolbar)
        self.display_toolbar.addWidget(QLabel("  界面字号："))
        self.sp_ui_scale_top = QSpinBox()
        self.sp_ui_scale_top.setRange(100, 200)
        self.sp_ui_scale_top.setSingleStep(25)
        self.sp_ui_scale_top.setSuffix("%")
        self.sp_ui_scale_top.setValue(self.ui_scale)
        self.sp_ui_scale_top.valueChanged.connect(self.set_ui_scale)
        self.display_toolbar.addWidget(self.sp_ui_scale_top)
        for label, delta in (("A− 缩小", -25), ("A+ 放大", 25)):
            button = QPushButton(label)
            button.clicked.connect(lambda _=False, d=delta: self.set_ui_scale(self.ui_scale + d))
            self.display_toolbar.addWidget(button)
        self.display_toolbar.addSeparator()
        self.btn_vehicle_algorithms = QPushButton("选择 Ego / BV 算法…")
        self.btn_vehicle_algorithms.clicked.connect(lambda: self.tabs.setCurrentWidget(self.algorithm_page))
        self.display_toolbar.addWidget(self.btn_vehicle_algorithms)
        self.display_toolbar.addSeparator()
        self.btn_calibration = QPushButton("G↔A 坐标标定…")
        self.btn_calibration.setToolTip("主界面和标定窗口都显示当前 MetaDrive A 坐标；实车 G→A 单点标定，跨PKL自动补偿原点")
        self.btn_calibration.clicked.connect(self.open_calibration_window)
        self.display_toolbar.addWidget(self.btn_calibration)

    # ==================================================================
    # 主体三栏
    # ==================================================================
    def _build_body(self) -> None:
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.addWidget(self._build_config_panel())
        self.main_splitter.addWidget(self._build_center())
        self.main_splitter.addWidget(self._build_status_panel())
        self.main_splitter.setCollapsible(0, False)
        self.main_splitter.setSizes([380, 850, 360])
        self.setCentralWidget(self.main_splitter)

    def _restore_ui_state(self) -> None:
        if not self._persist_ui_state:
            return
        settings = QSettings("AutoScene", "ExperimentManager")
        self.ui_scale = max(100, min(200, int(settings.value("window/ui_scale", 125))))
        geometry = settings.value("window/geometry")
        splitter = settings.value("window/main_splitter")
        if geometry is not None:
            self.restoreGeometry(geometry)
            QTimer.singleShot(0, self._ensure_window_visible)
        if splitter is not None:
            self.main_splitter.restoreState(splitter)
            QTimer.singleShot(0, self._validate_splitter_sizes)

    def _validate_splitter_sizes(self) -> None:
        sizes = self.main_splitter.sizes()
        if len(sizes) != 3 or sizes[0] < 360 or sizes[1] < 480 or sizes[2] < 260:
            self.main_splitter.setSizes([380, 850, 360])

    def _ensure_window_visible(self) -> None:
        """显示器变化后把保存在屏幕外的窗口移回当前主屏。"""
        app = QApplication.instance()
        screens = app.screens() if app is not None else []
        frame = self.frameGeometry()
        visible = any(
            frame.intersected(screen.availableGeometry()).width() >= 160
            and frame.intersected(screen.availableGeometry()).height() >= 120
            for screen in screens
        )
        if visible or not screens:
            return
        area = app.primaryScreen().availableGeometry()
        width = min(1600, area.width())
        height = min(1000, area.height())
        self.resize(width, height)
        self.move(area.center() - self.rect().center())

    # ---- 左：配置 ----------------------------------------------------
    def _build_config_panel(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        # v2: 包入 QScrollArea，小屏幕下可滚动全部配置
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(360)

        g_rt = QGroupBox("runtime")
        f = QGridLayout(g_rt)
        self.cb_mode = QComboBox()
        self.cb_mode.addItems(["simulation", "hybrid"])
        self.cb_mode.setCurrentText(self.cfg.mode)
        self.cb_mode.currentTextChanged.connect(self.on_mode_changed)
        f.addWidget(QLabel("mode"), 0, 0)
        f.addWidget(self.cb_mode, 0, 1)
        f.addWidget(QLabel("world_state_hz"), 1, 0)
        f.addWidget(mono(str(self.cfg.raw["runtime"]["world_state_hz"])), 1, 1)
        lay.addWidget(g_rt)

        g_sc = QGroupBox("scenario")
        f = QGridLayout(g_sc)
        self.cb_dataset_type = QComboBox()
        self.cb_dataset_type.addItems(["tjwest", "nuplan"])
        self.cb_dataset_type.setCurrentText(self.cfg.scenario["dataset_type"])
        self.cb_dataset_type.currentTextChanged.connect(self.on_dataset_type_changed)
        f.addWidget(QLabel("dataset_type"), 0, 0)
        f.addWidget(self.cb_dataset_type, 0, 1, 1, 2)

        self.ed_path = QLineEdit(str(self.cfg.scenario.get("dataset_path") or ""))
        self.ed_path.setFont(MONO)
        self.ed_path.editingFinished.connect(self.on_path_edited)
        self.btn_browse = QPushButton("浏览…")
        self.btn_browse.clicked.connect(self.browse_dataset)
        self.ed_path.setToolTip("前端电脑用于绘制地图和读取候选车辆的数据目录")
        f.addWidget(QLabel("场景数据目录"), 1, 0, 1, 3)
        f.addWidget(self.ed_path, 2, 0, 1, 2)
        f.addWidget(self.btn_browse, 2, 2)

        self.chk_ros_same_host = QCheckBox("ROS 与前端在同一台主机（推荐）")
        self.chk_ros_same_host.setChecked(self.ros_same_host)
        self.chk_ros_same_host.setToolTip(
            "同机时 ROS 自动使用上面的场景数据目录；只有虚拟机或远程 ROS 才需取消勾选")
        f.addWidget(self.chk_ros_same_host, 3, 0, 1, 3)

        self.ed_remote_path = QLineEdit(self.remote_dataset_path)
        self.ed_remote_path.setFont(MONO)
        self.ed_remote_path.setPlaceholderText("例如 /home/user/AutoScene/data/tjwest…")
        self.ed_remote_path.setToolTip("ROS 所在主机/虚拟机能够访问的数据集路径")
        self.ed_remote_path.editingFinished.connect(self.on_remote_path_edited)
        self.lab_remote_path = QLabel("远程 ROS 数据目录")
        f.addWidget(self.lab_remote_path, 4, 0, 1, 3)
        f.addWidget(self.ed_remote_path, 5, 0, 1, 3)
        self.chk_ros_same_host.toggled.connect(self.on_ros_same_host_toggled)
        self._update_ros_path_visibility()
        self._refresh_path_tooltips()

        self.cb_history = QComboBox()
        self.cb_history.addItems(["dataset", "initial"])
        self.cb_history.setCurrentText(self.cfg.scenario["history_source"])
        self.cb_history.currentTextChanged.connect(self.on_history_changed)
        f.addWidget(QLabel("history_source"), 6, 0)
        f.addWidget(self.cb_history, 6, 1, 1, 2)

        self.sp_index = QSpinBox()
        self.sp_index.setRange(0, 9999)
        self.sp_index.setValue(int(self.cfg.scenario["scenario_index"]))
        self.sp_index.valueChanged.connect(lambda v: self._set_scenario_index(v))
        f.addWidget(QLabel("scenario_index"), 7, 0)
        f.addWidget(self.sp_index, 7, 1)
        self.lab_index_range = mono("0..?")
        f.addWidget(self.lab_index_range, 7, 2)

        self.lab_constraint = QLabel()
        self.lab_constraint.setWordWrap(True)
        self.lab_constraint.setStyleSheet("color:#8a4000")
        f.addWidget(self.lab_constraint, 8, 0, 1, 3)
        f.setColumnStretch(1, 1)
        lay.addWidget(g_sc)

        g_bv = QGroupBox("本轮 BV（由地图操作决定）")
        f = QGridLayout(g_bv)
        self.lab_n_platform = mono("0")
        self.lab_n_virtual = mono("0")
        f.addWidget(QLabel("platform BV"), 0, 0)
        f.addWidget(self.lab_n_platform, 0, 1)
        f.addWidget(QLabel("virtual BV"), 1, 0)
        f.addWidget(self.lab_n_virtual, 1, 1)
        lay.addWidget(g_bv)

        g_pf = QGroupBox("platform 通信（只读）")
        f = QVBoxLayout(g_pf)
        self.lab_udp = QLabel()
        self.lab_udp.setFont(MONO)
        self.lab_udp.setWordWrap(True)
        f.addWidget(self.lab_udp)
        self.list_endpoints = QListWidget()
        self.list_endpoints.setFont(MONO)
        self.list_endpoints.setMaximumHeight(120)
        f.addWidget(self.list_endpoints)
        lay.addWidget(g_pf)

        lay.addStretch(1)

        # v2: 将 page 设为 scroll 的内容，返回 scroll 而非 page
        scroll.setWidget(page)
        return scroll

    # ---- 中：可视化 + 页签 ------------------------------------------
    def _build_center(self) -> QWidget:
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_viz_tab(), "场景可视化")
        self.tabs.addTab(self._build_initial_tab(), "初始状态录入")
        self.tabs.addTab(self._build_yaml_tab(), "YAML 源码")
        self.tabs.addTab(self._build_history_tab(), "历史结果")
        self.algorithm_page = self._build_algorithm_tab()
        self.tabs.addTab(self.algorithm_page, "规划算法")
        self.tabs.currentChanged.connect(lambda _i: self.refresh_tabs())
        return self.tabs

    def _build_algorithm_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        hint = QLabel(
            "只区分两个全局规划角色：Ego 使用 algorithm.ego_method；所有需要模型规划的 BV "
            "统一使用 algorithm.adv_method。initial_once 下绑定平台的 BV 不进入 BV planner；"
            "pluto_replan 下平台 BV 与其他受规划 BV 共用 adv_method。"
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        group = QGroupBox("规划算法")
        grid = QGridLayout(group)
        grid.addWidget(QLabel("Ego 算法"), 0, 0)
        self.cb_ego_method = self._make_global_planner_combo("ego")
        grid.addWidget(self.cb_ego_method, 0, 1)
        grid.addWidget(QLabel("BV 算法"), 1, 0)
        self.cb_adv_method = self._make_global_planner_combo("adv")
        grid.addWidget(self.cb_adv_method, 1, 1)
        layout.addWidget(group)

        note = QLabel(
            "“沿用 UniTraj config.yaml”表示不向后端覆盖该角色；显式选择后，值写入 "
            "algorithm.ego_method / algorithm.adv_method，并在重新准备、启动后生效。"
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#666")
        layout.addWidget(note)
        layout.addStretch(1)
        return page

    def _make_global_planner_combo(self, role: str) -> QComboBox:
        if role not in ("ego", "adv"):
            raise ValueError("planner role must be ego or adv")
        combo = QComboBox()
        combo.setProperty("plannerRole", role)
        combo.addItem("沿用 UniTraj config.yaml", None)
        for label, method in (("Pluto", "Pluto"), ("VBD", "VBD"),
                              ("Pluto Hybrid", "pluto_hybrid"),
                              ("Pluto Hybrid Blend", "pluto_hybrid_blend")):
            combo.addItem(label, method)
        combo.setMinimumContentsLength(18)
        combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        combo.currentIndexChanged.connect(
            lambda _i, cb=combo: self._set_global_planner(
                str(cb.property("plannerRole")), cb.currentData()
            )
        )
        self._sync_global_planner_combo(combo)
        return combo

    def _sync_global_planner_combo(self, combo: QComboBox) -> None:
        role = str(combo.property("plannerRole"))
        current = self.ego_method if role == "ego" else self.adv_method
        blocked = combo.blockSignals(True)
        index = combo.findData(current)
        if index < 0:
            combo.addItem("不支持：%s" % current, current)
            index = combo.count() - 1
        combo.setCurrentIndex(index)
        combo.blockSignals(blocked)

    def _set_global_planner(self, role: str, method) -> None:
        if role not in ("ego", "adv"):
            raise ValueError("planner role must be ego or adv")
        if method not in (None, "Pluto", "VBD", "pluto_hybrid", "pluto_hybrid_blend"):
            self.log("FAIL", "不支持的算法：%s" % method)
            return

        key = "ego_method" if role == "ego" else "adv_method"
        algorithm_cfg = self.cfg.raw.setdefault("algorithm", {})
        if method is None:
            algorithm_cfg.pop(key, None)
        else:
            algorithm_cfg[key] = method

        if role == "ego":
            self.ego_method = method
        else:
            self.adv_method = method

        self._algorithm_selection_dirty = True
        self.state["prepared"] = False
        self.log(
            "OK",
            "%s 算法 = %s；重新准备并启动后生效"
            % ("Ego" if role == "ego" else "BV", method or "沿用 UniTraj config.yaml"),
        )
        self.refresh_yaml()
        self.refresh_algorithm_table()
        self._update_action_states()

    def refresh_algorithm_table(self) -> None:
        # 保留原函数名，避免 refresh_all() 等现有调用点发生额外结构变化。
        if hasattr(self, "cb_ego_method"):
            self._sync_global_planner_combo(self.cb_ego_method)
        if hasattr(self, "cb_adv_method"):
            self._sync_global_planner_combo(self.cb_adv_method)

    def _build_viz_tab(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        bar = QToolBar("地图工具")
        bar.setObjectName("mapToolbar")
        bar.setMovable(False)
        h = bar
        self.btn_place = QPushButton("✛ 放置 virtual BV")
        self.btn_place.setObjectName("placeButton")
        self.btn_place.setCheckable(True)
        self.btn_place.toggled.connect(self.on_place_toggled)
        self.btn_delete_virtual = QPushButton("删除所选 BV")
        self.btn_delete_virtual.setToolTip("在下方统一初始状态表或地图中选择 virtual BV 后删除")
        self.btn_delete_virtual.clicked.connect(self.remove_selected_virtual)
        self.chk_trail = QCheckBox("轨迹")
        self.chk_trail.setChecked(True)
        self.chk_id = QCheckBox("ID")
        self.chk_id.setChecked(True)
        self.chk_grid = QCheckBox("网格")
        for w in (self.btn_place, self.btn_delete_virtual,
                  self.chk_trail, self.chk_id, self.chk_grid):
            h.addWidget(w)
        for c in (self.chk_trail, self.chk_id, self.chk_grid):
            c.toggled.connect(lambda _v: self.redraw_scene())
        h.addSeparator()
        btn_zoom_out = QPushButton("−")
        btn_zoom_out.setObjectName("mapToolButton")
        btn_zoom_out.setToolTip("缩小（-）")
        btn_zoom_out.clicked.connect(lambda: self.view.zoom_out())
        h.addWidget(btn_zoom_out)
        self.lab_zoom = mono("100%")
        self.lab_zoom.setObjectName("zoomReadout")
        self.lab_zoom.setAlignment(Qt.AlignCenter)
        self.lab_zoom.setMinimumWidth(58)
        h.addWidget(self.lab_zoom)
        btn_zoom_in = QPushButton("+")
        btn_zoom_in.setObjectName("mapToolButton")
        btn_zoom_in.setToolTip("放大（+）")
        btn_zoom_in.clicked.connect(lambda: self.view.zoom_in())
        h.addWidget(btn_zoom_in)
        btn_fit = QPushButton("适应全部  F")
        btn_fit.setObjectName("mapToolButton")
        btn_fit.setToolTip("显示全部对象（F / 双击空白处）")
        btn_fit.clicked.connect(lambda: self.view.fit())
        h.addWidget(btn_fit)
        btn_focus = QPushButton("聚焦车辆  A")
        btn_focus.setObjectName("mapToolButton")
        btn_focus.setToolTip("聚焦车辆工作区，忽略远处零散地图要素（A）")
        btn_focus.clicked.connect(lambda: self.view.focus_actors())
        h.addWidget(btn_focus)
        self.lab_hint = QLabel("双指捏合 / 滚轮缩放 · 中键平移 · F 全图 · A 车辆")
        self.lab_hint.setObjectName("mapHint")
        h.addWidget(self.lab_hint)
        btn_shortcuts = QPushButton("?")
        btn_shortcuts.setObjectName("mapToolButton")
        btn_shortcuts.setToolTip("快捷键速查")
        btn_shortcuts.clicked.connect(self._show_shortcuts)
        h.addWidget(btn_shortcuts)
        lay.addWidget(bar)

        editor_bar = QToolBar("场景编辑")
        self.btn_open_pkl = QPushButton("打开 PKL…")
        self.btn_open_pkl.clicked.connect(self.open_scene_pkl)
        editor_bar.addWidget(self.btn_open_pkl)
        self.chk_builder = QCheckBox("场景构建（含背景车）")
        self.chk_builder.setToolTip("调整已有车辆的位姿，另存完整轨迹为新 PKL 数据集")
        self.chk_builder.toggled.connect(self.on_builder_toggled)
        editor_bar.addWidget(self.chk_builder)
        self.chk_snap_lane = QCheckBox("放置时吸附车道")
        self.chk_snap_lane.setChecked(False)
        self.chk_snap_lane.setToolTip("仅吸附真实车道中心线；关闭时严格使用鼠标点击位置")
        editor_bar.addWidget(self.chk_snap_lane)
        self.btn_export_scene = QPushButton("另存新 PKL 场景…")
        self.btn_export_scene.clicked.connect(self.export_scene)
        editor_bar.addWidget(self.btn_export_scene)
        self.btn_restore_scene = QPushButton("还原场景编辑")
        self.btn_restore_scene.clicked.connect(self.restore_scene_edits)
        editor_bar.addWidget(self.btn_restore_scene)
        lay.addWidget(editor_bar)

        # 当前选中车辆的快捷编辑栏：位置 / 航向 / 初始速度全部在地图上方直接改。
        pose_bar = QToolBar("车辆初始状态快捷编辑")
        pose_bar.addWidget(QLabel("选择车辆："))
        self.cb_edit_vehicle = QComboBox()
        self.cb_edit_vehicle.setMinimumContentsLength(12)
        self.cb_edit_vehicle.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.cb_edit_vehicle.currentIndexChanged.connect(self._load_pose_controls)
        pose_bar.addWidget(self.cb_edit_vehicle)
        self.pose_inputs = {}
        for field, label, minimum, maximum, decimals, step in (
            ("x", "x (m)", -1e9, 1e9, 3, 0.1),
            ("y", "y (m)", -1e9, 1e9, 3, 0.1),
            ("heading", "heading (°)", -36000.0, 36000.0, 2, 1.0),
            ("speed", "speed (m/s)", 0.0, 1000000.0, 3, 0.1),
        ):
            pose_bar.addWidget(QLabel(label))
            spin = QDoubleSpinBox()
            spin.setRange(minimum, maximum)
            spin.setDecimals(decimals)
            spin.setSingleStep(step)
            spin.setKeyboardTracking(False)
            spin.setMinimumWidth(95)
            if field == "speed":
                spin.valueChanged.connect(self._apply_speed_control)
            else:
                spin.valueChanged.connect(self._apply_pose_controls)
            self.pose_inputs[field] = spin
            pose_bar.addWidget(spin)
        lay.addWidget(pose_bar)

        self.view = SceneView()
        self.view.setObjectName("sceneView")
        self.view.map_clicked.connect(self.on_map_clicked)
        self.view.vehicle_clicked.connect(self.on_vehicle_clicked)
        self.view.vehicle_moved.connect(self.on_vehicle_moved)
        self.view.edit_finished.connect(lambda: QTimer.singleShot(0, self._redraw_after_edit))
        self.view.zoom_changed.connect(lambda value: self.lab_zoom.setText("%d%%" % value))
        lay.addWidget(self.view, 1)

        # 统一编辑 Ego / platform 候选或已绑定 BV / virtual BV。
        # 不再把 x/y/heading/speed 分散在多个表格和不同 tab。
        state_box = QGroupBox("实验车辆初始状态（直接编辑）")
        state_lay = QVBoxLayout(state_box)
        state_lay.setContentsMargins(6, 8, 6, 6)
        self.lab_state_editor_hint = QLabel()
        self.lab_state_editor_hint.setWordWrap(True)
        state_lay.addWidget(self.lab_state_editor_hint)

        self.tbl_vehicle_initial = QTableWidget(0, 9)
        self.tbl_vehicle_initial.setHorizontalHeaderLabels(
            ["类型", "vehicle_id", "track_id", "platform",
             "x (m)", "y (m)", "heading (rad)", "speed (m/s)", "操作"])
        self.tbl_vehicle_initial.verticalHeader().setVisible(False)
        self.tbl_vehicle_initial.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl_vehicle_initial.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tbl_vehicle_initial.setFont(MONO)
        self.tbl_vehicle_initial.itemSelectionChanged.connect(self.on_initial_state_row_selected)
        header = self.tbl_vehicle_initial.horizontalHeader()
        for column in (0, 1, 2, 3, 8):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        for column in (4, 5, 6, 7):
            header.setSectionResizeMode(column, QHeaderView.Stretch)
        state_lay.addWidget(self.tbl_vehicle_initial)
        state_box.setMaximumHeight(300)
        lay.addWidget(state_box)
        return page

    def _build_initial_tab(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        self.lab_init_head = QLabel()
        self.lab_init_head.setWordWrap(True)
        lay.addWidget(self.lab_init_head)
        self.tbl_initial = QTableWidget(0, 6)
        self.tbl_initial.setHorizontalHeaderLabels(
            ["车辆", "角色", "initial.x", "initial.y", "heading (rad)", "speed (m/s)"])
        self.tbl_initial.verticalHeader().setVisible(False)
        self.tbl_initial.setFont(MONO)
        self.tbl_initial.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        lay.addWidget(self.tbl_initial, 1)
        self.lab_init_hint = QLabel()
        self.lab_init_hint.setFont(MONO)
        row = QHBoxLayout()
        row.addWidget(self.lab_init_hint, 1)
        self.btn_fix_speeds = QPushButton("将超限速度修正到上限")
        self.btn_fix_speeds.clicked.connect(self.clamp_initial_speeds)
        row.addWidget(self.btn_fix_speeds)
        lay.addLayout(row)
        return page

    def _build_yaml_tab(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        self.ed_yaml = QPlainTextEdit()
        self.ed_yaml.setFont(MONO)
        self.ed_yaml.setReadOnly(True)
        self.ed_yaml.setStyleSheet("")  # 继承浅色主题和可调整字号
        self.ed_yaml.setProperty("dark", False)
        lay.addWidget(self.ed_yaml)
        row = QHBoxLayout()
        b1 = QPushButton("保存到当前 YAML")
        b1.clicked.connect(self.save_yaml)
        b2 = QPushButton("另存为…")
        b2.clicked.connect(lambda: self.save_yaml(as_new=True))
        row.addWidget(b1)
        row.addWidget(b2)
        row.addStretch(1)
        lay.addLayout(row)
        return page

    def _build_history_tab(self) -> QWidget:
        page = QWidget()
        lay = QHBoxLayout(page)
        self.list_history = QListWidget()
        self.list_history.setFont(MONO)
        self.list_history.currentRowChanged.connect(self.show_history_detail)
        lay.addWidget(self.list_history, 1)
        right = QWidget()
        v = QVBoxLayout(right)
        self.lab_hist_title = QLabel("选择左侧结果文件")
        self.lab_hist_title.setWordWrap(True)
        v.addWidget(self.lab_hist_title)
        self.tbl_hist = QTableWidget(0, 2)
        self.tbl_hist.setHorizontalHeaderLabels(["指标", "值"])
        self.tbl_hist.verticalHeader().setVisible(False)
        self.tbl_hist.setFont(MONO)
        self.tbl_hist.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        v.addWidget(self.tbl_hist, 1)
        btn = QPushButton("刷新 outputs 目录")
        btn.clicked.connect(self.reload_history)
        v.addWidget(btn)
        lay.addWidget(right, 1)
        return page

    # ---- 右：状态 ----------------------------------------------------
    def _build_status_panel(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(8, 8, 8, 8)

        g1 = QGroupBox("运行状态")
        f = QGridLayout(g1)
        self.lab_state = mono("idle")
        self.lab_state.setObjectName("stateLabel")
        self.lab_step = mono("0 · 0.0s")
        self.lab_rtf = mono("—")
        self.lab_elapsed = mono("00:0.0")
        for i, (name, lab) in enumerate([("state", self.lab_state), ("step / sim_time", self.lab_step),
                                         ("real_time_factor", self.lab_rtf), ("elapsed", self.lab_elapsed)]):
            f.addWidget(QLabel(name), i, 0)
            f.addWidget(lab, i, 1)
        lay.addWidget(g1)

        g2 = QGroupBox("评价指标（10 Hz）")
        f = QGridLayout(g2)
        self.metric_labels: Dict[str, QLabel] = {}
        for i, key in enumerate(["collision", "min_distance", "time_to_collision",
                                 "ego_speed_mean", "ego_speed_max", "platform_tracking_error"]):
            lab = mono("—")
            lab.setStyleSheet("color:#b0b0b0")  # v2: 无数据时灰色
            self.metric_labels[key] = lab
            f.addWidget(QLabel(key), i, 0)
            f.addWidget(lab, i, 1)
        lay.addWidget(g2)

        g_live = QGroupBox("实时车辆状态（world/state）")
        live_lay = QVBoxLayout(g_live)
        self.lab_live_stream = mono("等待 /autoscene/world/state")
        self.lab_live_stream.setStyleSheet("color:#8a4000")
        live_lay.addWidget(self.lab_live_stream)
        self.tbl_live = QTableWidget(0, 6)
        self.tbl_live.setHorizontalHeaderLabels(["vehicle", "x(A)", "y(A)", "yaw(rad)", "speed", "source"] )
        self.tbl_live.verticalHeader().setVisible(False)
        self.tbl_live.setFont(MONO)
        self.tbl_live.setMaximumHeight(170)
        self.tbl_live.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tbl_live.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
        live_lay.addWidget(self.tbl_live)
        lay.addWidget(g_live)

        g3 = QGroupBox("平台通信 comm_status")
        f = QVBoxLayout(g3)
        self.tbl_comm = QTableWidget(0, 4)
        self.tbl_comm.setHorizontalHeaderLabels(["platform", "tx", "rx", "rx_age"])
        self.tbl_comm.verticalHeader().setVisible(False)
        self.tbl_comm.setFont(MONO)
        self.tbl_comm.setMaximumHeight(110)
        self.tbl_comm.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        f.addWidget(self.tbl_comm)
        self.lab_comm_note = QLabel()
        self.lab_comm_note.setWordWrap(True)
        self.lab_comm_note.setStyleSheet("color:#8a4000")
        f.addWidget(self.lab_comm_note)
        lay.addWidget(g3)

        g4 = QGroupBox("节点 / 日志")
        f = QVBoxLayout(g4)
        self.ed_log = QPlainTextEdit()
        self.ed_log.setReadOnly(True)
        self.ed_log.setFont(MONO)
        self.ed_log.document().setMaximumBlockCount(2000)
        self.ed_log.setStyleSheet("")  # 继承浅色主题和可调整字号
        self.ed_log.setProperty("dark", False)
        f.addWidget(self.ed_log)
        lay.addWidget(g4, 1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(page)
        scroll.setMinimumWidth(280)
        scroll.setFrameShape(QFrame.NoFrame)
        return scroll

    def _build_status_bar(self) -> None:
        self.lab_bar = mono()
        self.statusBar().addWidget(self.lab_bar)
        self.statusBar().addPermanentWidget(QLabel("界面字号"))
        self.sp_ui_scale = QSpinBox()
        self.sp_ui_scale.setRange(100, 200)
        self.sp_ui_scale.setSingleStep(25)
        self.sp_ui_scale.setSuffix("%")
        self.sp_ui_scale.setValue(self.ui_scale)
        self.sp_ui_scale.valueChanged.connect(self.set_ui_scale)
        self.statusBar().addPermanentWidget(self.sp_ui_scale)
        menu = self.menuBar().addMenu("显示")
        for title, index, shortcut in (("显示左侧配置", 0, "F8"), ("显示右侧状态", 2, "F9")):
            action = menu.addAction(title)
            action.setCheckable(True)
            action.setChecked(True)
            action.setShortcut(shortcut)
            action.toggled.connect(lambda shown, i=index: self.main_splitter.widget(i).setVisible(shown))
        for title, shortcut, delta in (("放大界面字号", "Ctrl+=", 25),
                                       ("缩小界面字号", "Ctrl+-", -25),
                                       ("恢复默认字号", "Ctrl+0", 0)):
            action = menu.addAction(title)
            action.setShortcut(shortcut)
            action.triggered.connect(lambda _=False, d=delta:
                                     self.set_ui_scale(self.ui_scale + d if d else 125))

    def set_ui_scale(self, percent: int) -> None:
        self.ui_scale = max(100, min(200, int(percent)))
        scale = self.ui_scale / 100.0
        self._mono_font.setPixelSize(round(12 * scale))
        style = re.sub(r"(\d+(?:\.\d+)?)px", lambda m: "%dpx" %
                       max(1, round(float(m.group(1)) * scale)), APP_STYLE)
        self.setStyleSheet(style)
        for table in self.findChildren(QTableWidget):
            table.setFont(self._mono_font)
            table.verticalHeader().setDefaultSectionSize(round(34 * scale))
            for row in range(table.rowCount()):
                for col in range(table.columnCount()):
                    if table.item(row, col):
                        table.item(row, col).setFont(self._mono_font)
        self.view.label_scale = scale
        for label in self.view.vehicle_labels.values():
            label.setFont(self._mono_font)
        self.sp_ui_scale.blockSignals(True)
        self.sp_ui_scale.setValue(self.ui_scale)
        self.sp_ui_scale.blockSignals(False)
        self.sp_ui_scale_top.blockSignals(True)
        self.sp_ui_scale_top.setValue(self.ui_scale)
        self.sp_ui_scale_top.blockSignals(False)
        if self._persist_ui_state:
            QSettings("AutoScene", "ExperimentManager").setValue("window/ui_scale", self.ui_scale)

    # ==================================================================
    # 事件
    # ==================================================================
    def log(self, level: str, text: str) -> None:
        color = {"FAIL": "#b42318", "WARN": "#8a4000", "OK": "#187345"}.get(level, "#31536f")
        self.ed_log.appendHtml('<span style="color:%s">[%s]</span> %s' %
                               (color, html.escape(str(level)), html.escape(str(text))))
        # v2: 自动滚动到最新日志
        cursor = self.ed_log.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.ed_log.setTextCursor(cursor)
        self.ed_log.ensureCursorVisible()

    def on_mode_changed(self, mode: str) -> None:
        self.cfg.mode = mode
        if mode == "hybrid":
            self.cb_dataset_type.setCurrentText("tjwest")
            self.cb_history.setCurrentText("initial")
        else:
            self.platform_bind.clear()
        self.state["prepared"] = False
        self.log("INFO", "runtime.mode = %s%s" % (mode, "（强制 tjwest + initial）" if mode == "hybrid" else "（清除端点绑定）"))
        self.refresh_all(preserve_view=False)

    def on_dataset_type_changed(self, value: str) -> None:
        if self.cfg.mode == "hybrid" and value != "tjwest":
            self.log("FAIL", "hybrid 模式只支持 dataset_type=tjwest")
            self.cb_dataset_type.setCurrentText("tjwest")
            return
        self.cfg.scenario["dataset_type"] = value
        self._invalidate_dataset()

    def on_history_changed(self, value: str) -> None:
        if self.cfg.mode == "hybrid" and value != "initial":
            self.log("FAIL", "hybrid 模式要求 history_source=initial")
            self.cb_history.setCurrentText("initial")
            return
        self.cfg.scenario["history_source"] = value
        self.state["prepared"] = False
        self.refresh_all()

    def on_path_edited(self) -> None:
        self.cfg.scenario["dataset_path"] = self.ed_path.text().strip()
        self._refresh_path_tooltips()
        self._invalidate_dataset()

    def on_remote_path_edited(self) -> None:
        self.remote_dataset_path = self.ed_remote_path.text().strip()
        self._last_remote_dataset_path = self.remote_dataset_path
        self._refresh_path_tooltips()
        self.state["prepared"] = False
        self.log("INFO", "ROS 数据目录 = %s" %
                 (self.remote_dataset_path or "与本机数据目录相同"))
        self.refresh_all()

    def on_ros_same_host_toggled(self, same_host: bool) -> None:
        """同机只用一个目录；远程模式恢复上次填写的 ROS 路径。"""
        self.ros_same_host = bool(same_host)
        if self.ros_same_host:
            current = self.ed_remote_path.text().strip()
            if current:
                self._last_remote_dataset_path = current
            self.remote_dataset_path = ""
        else:
            self.ed_remote_path.setText(self._last_remote_dataset_path)
            self.remote_dataset_path = self.ed_remote_path.text().strip()
        self._update_ros_path_visibility()
        self._refresh_path_tooltips()
        self.state["prepared"] = False
        self.log("INFO", "ROS 数据目录：%s" %
                 ("自动使用场景数据目录" if self.ros_same_host else
                  (self.remote_dataset_path or "请填写远程主机路径")))
        self.refresh_all()

    def _update_ros_path_visibility(self) -> None:
        visible = not self.ros_same_host
        self.lab_remote_path.setVisible(visible)
        self.ed_remote_path.setVisible(visible)

    def _refresh_path_tooltips(self) -> None:
        local_path = self.ed_path.text().strip() or "（未设置）"
        remote_path = (local_path if self.ros_same_host else
                       (self.ed_remote_path.text().strip() or "（未设置）"))
        self.ed_path.setToolTip("前端本机完整数据路径：\n%s" % local_path)
        self.ed_remote_path.setToolTip("远程 ROS 主机/虚拟机完整数据路径：\n%s" % remote_path)
        self.chk_ros_same_host.setToolTip(
            "ROS 实际使用：%s\n同机时无需维护第二条路径。" % remote_path)

    def browse_dataset(self) -> None:
        start = self.ed_path.text().strip() or os.path.expanduser("~")
        path = QFileDialog.getExistingDirectory(self, "选择数据集目录", start)
        if not path:
            return
        self.ed_path.setText(path)
        self.cfg.scenario["dataset_path"] = path
        self._refresh_path_tooltips()
        self._invalidate_dataset()
        self.log("OK", "dataset_path = %s（点击「① 加载数据集场景」解析）" % path)

    def _current_speed_context(self):
        return (
            str(self.cfg.scenario.get("dataset_path") or ""),
            int(self.cfg.scenario.get("scenario_index") or 0),
        )

    def _restore_initial_speeds_from_config(self) -> None:
        """回填同一场景 YAML 中的 ego/platform initial.speed，不跨场景套用。"""
        self.speeds.clear()
        self._initial_speed_config_errors = []
        if self._current_speed_context() != self._speed_config_context:
            return

        def read_speed(initial, label):
            if not isinstance(initial, dict) or "speed" not in initial:
                return None
            try:
                value = float(initial["speed"])
            except (TypeError, ValueError):
                self._initial_speed_config_errors.append(
                    "%s initial.speed 不是数值" % label
                )
                return None
            if not math.isfinite(value) or value < 0.0:
                self._initial_speed_config_errors.append(
                    "%s initial.speed 非法：%r" % (label, initial.get("speed"))
                )
                return None
            return value

        ego_cfg = self.cfg.raw.get("ego", {})
        ego_speed = read_speed(ego_cfg.get("initial", {}), str(ego_cfg.get("id", "ego_001")))
        if ego_speed is not None:
            self.speeds["ego"] = ego_speed

        for index, agent in enumerate(self.cfg.raw.get("vehicles", {}).get("agents", [])):
            if not isinstance(agent, dict):
                self._initial_speed_config_errors.append(
                    "vehicles.agents[%d] 不是映射" % index
                )
                continue
            track_id = agent.get("scenario_track_id")
            if track_id is None:
                continue
            speed = read_speed(
                agent.get("initial", {}),
                str(agent.get("id", "agent_%d" % index)),
            )
            if speed is not None:
                self.speeds["t" + str(track_id)] = speed

    def _invalidate_dataset(self) -> None:
        self.dataset = None
        self.scene = DEMO_SCENE()
        self.platform_bind.clear()
        self.virtuals.clear()
        self.speeds.clear()
        self._edited_track_poses.clear()
        self._original_scene = None
        self.state["prepared"] = False
        self.refresh_all(preserve_view=False)

    def _set_scenario_index(self, value: int) -> None:
        self.cfg.scenario["scenario_index"] = value
        if self.dataset:
            self.load_scenario()

    @staticmethod
    def _scene_in_runtime_a(source_scene):
        """Adapt the raw ScenarioDescription scene to actual ScenarioEnv A coordinates.

        MetaDrive ScenarioDataManager loads PKL with centralize=True.  Keep the
        original ego frame-0 offset as *metadata only* for cross-PKL calibration;
        all editable/rendered geometry in this MainWindow is A afterwards.
        This converts a NEWLY loaded scene once. Never call on an edited scene.
        """
        if source_scene.is_demo:
            return source_scene
        ox = float(source_scene.runtime_origin_x)
        oy = float(source_scene.runtime_origin_y)
        if not (math.isfinite(ox) and math.isfinite(oy)):
            raise ValueError('PKL frame-0 origin is invalid')
        for obj in [source_scene.ego] + list(source_scene.objects):
            obj.x -= ox
            obj.y -= oy
            obj.trail = [(float(x) - ox, float(y) - oy) for x, y in obj.trail]
        for field in ('lanes', 'boundaries', 'crosswalks'):
            polylines = getattr(source_scene, field)
            setattr(source_scene, field, [
                [(float(x) - ox, float(y) - oy) for x, y in line]
                for line in polylines
            ])
        return source_scene

    def load_scenario(self) -> None:
        # 新场景不能继承上一轮 runtime pose / 预启动对齐标记。
        if hasattr(self, "view"):
            self.view.clear_live_state()
            if hasattr(self.view, "clear_pose_overlays"):
                self.view.clear_pose_overlays(prefix="alignment:")
        self._runtime_live_enabled = False
        self._runtime_stream_retry_timer.stop()
        self._world_state_seen = False
        self._live_vehicle_states.clear()
        self._raw_sim_ego_seen = False
        self._raw_sim_agent_ids.clear()
        path = self.ed_path.text().strip()
        try:
            self.dataset = ScenarioDataset(path)
            idx = int(self.sp_index.value())
            slots = max(1, len(self.cfg.endpoints)) if self.cfg.mode == "hybrid" else 0
            self.scene = self._scene_in_runtime_a(self.dataset.load_scene(idx, physical_slots=slots))
            self.scene_index = idx
        except DatasetLoadError as exc:
            self.dataset = None
            self.scene = DEMO_SCENE()
            self.log("FAIL", "加载失败：%s" % exc)
            QMessageBox.warning(self, "加载数据集", str(exc))
        except Exception as exc:  # noqa: BLE001
            self.dataset = None
            self.scene = DEMO_SCENE()
            self.log("FAIL", "加载异常：%s" % exc)
        else:
            self.platform_bind.clear()
            self.virtuals.clear()
            self._restore_initial_speeds_from_config()
            self._edited_track_poses.clear()
            self._original_scene = copy.deepcopy(self.scene)
            self.state["prepared"] = False
            self.log("OK", "已加载 %s：%d 个对象，%d 个实物车位%s"
                     % (self.scene.name, len(self.scene.objects), len(self.scene.physical_slots),
                        "（%s）" % self.scene.note if self.scene.note else ""))
            for error in self._initial_speed_config_errors:
                self.log("FAIL", error)
        self.refresh_all(preserve_view=False)
        if self.dataset is not None:
            # 数据中可能包含远处孤立地图点；默认先展示车辆工作区，F 可查看全图。
            QTimer.singleShot(0, self.view.focus_actors)

    def on_place_toggled(self, on: bool) -> None:
        self.view.set_place_mode(on)
        self.lab_hint.setText("先选择放置源车辆，再点击空白位置" if on
                              else "双指捏合 / 滚轮缩放；橙色=实物平台车，绿色=virtual BV；虚线=planner future，实线=platform实际下发")

    def on_map_clicked(self, x: float, y: float) -> None:
        if not self.view.editing_enabled:
            return
        if self.dataset is None:
            self.log("FAIL", "请先加载数据集；virtual BV 必须绑定真实 scenario_track_id")
            return
        used_tracks = set(self.platform_bind) | {
            str(item.get("track_id")) for item in self.virtuals if item.get("track_id") is not None
        }
        candidates = [obj for obj in self.scene.objects if obj.track_id not in used_tracks
                      and str(obj.obj_type).upper() in ("VEHICLE", "CAR", "TRUCK", "BUS")]
        if not candidates:
            self.log("FAIL", "当前场景没有可绑定的剩余车辆轨迹")
            return
        selected = self.cb_edit_vehicle.currentData()
        source = next((obj for obj in candidates if obj.track_id == selected), None)
        if source is None:
            self.log("WARN", "请先在“选择车辆 / 放置源”中选择一台未绑定的周车，再点击放置位置")
            return
        if self.chk_snap_lane.isChecked():
            snap = nearest_polyline_pose(self.scene.lanes, x, y, source.heading)
            if snap is None or snap[3] > 8.0:
                distance = snap[3] if snap is not None else float("inf")
                self.log("FAIL", "点击位置距车道中心线 %.2f m；请在车道附近放置" % distance)
                return
        else:
            snap = (x, y, source.heading, 0.0)
        snap_x, snap_y, snap_heading, snap_distance = snap
        key = "V-%03d" % self._next_virtual
        self._next_virtual += 1
        speed = self._speed_for_initial(source.speed, "%s（track %s）" % (key, source.track_id))
        self.virtuals.append({"key": key, "track_id": source.track_id,
                              "x": snap_x, "y": snap_y,
                              "heading": snap_heading, "speed": speed})
        self.state["prepared"] = False
        self.log("OK", "放置 %s @ x=%.3f y=%.3f heading=%.3f；吸附 %.2f m，绑定 track=%s"
                 % (key, snap_x, snap_y, snap_heading, snap_distance, source.track_id))
        self.refresh_all()
        self.view.select_key(key)
        self.cb_edit_vehicle.setCurrentIndex(self.cb_edit_vehicle.findData(key))
        self.btn_place.setChecked(False)

    def on_vehicle_clicked(self, kind: str, key: str) -> None:
        self.cb_edit_vehicle.setCurrentIndex(self.cb_edit_vehicle.findData(key))
        self._select_initial_state_row(key)

    def on_vehicle_moved(self, key: str, x: float, y: float, heading: float) -> None:
        if not self.view.editing_enabled:
            return
        for v in self.virtuals:
            if v["key"] == key:
                v["x"], v["y"], v["heading"] = x, y, heading
                break
        else:
            obj = self.scene.ego if key == "ego" else next(
                (obj for obj in self.scene.objects if obj.track_id == key), None)
            if obj is None:
                return
            angle = heading - obj.heading
            c, s = math.cos(angle), math.sin(angle)
            obj.trail = [(x + c * (px - obj.x) - s * (py - obj.y),
                          y + s * (px - obj.x) + c * (py - obj.y)) for px, py in obj.trail]
            obj.x, obj.y, obj.heading = x, y, heading
            self._edited_track_poses[str(obj.track_id)] = (x, y, heading)
        self.state["prepared"] = False
        self._load_pose_controls()
        # 拖拽只更新图元；表格/参数合并刷新，避免每个鼠标事件重建控件。
        if not self._edit_refresh_timer.isActive():
            self._edit_refresh_timer.start(100)

    def _refresh_pose_tables(self) -> None:
        self.refresh_vehicle_initial_table()
        if self.tabs.currentIndex() == 1:
            self.refresh_initial_table()
        self.refresh_yaml()
        self._update_action_states()

    def _refresh_vehicle_choices(self) -> None:
        selected = self.cb_edit_vehicle.currentData()
        self.cb_edit_vehicle.blockSignals(True)
        self.cb_edit_vehicle.clear()
        self.cb_edit_vehicle.addItem("请选择车辆", "")
        self.cb_edit_vehicle.addItem("EGO · 主车", "ego")
        replaced = {str(v.get("track_id")) for v in self.virtuals}
        for obj in self.scene.objects:
            if obj.track_id not in replaced:
                role = "实车候选" if obj.physical else "背景车"
                self.cb_edit_vehicle.addItem("%s · %s" % (obj.track_id, role), obj.track_id)
        for v in self.virtuals:
            self.cb_edit_vehicle.addItem("%s · track %s" % (v["key"], v["track_id"]), v["key"])
        self.cb_edit_vehicle.setCurrentIndex(max(0, self.cb_edit_vehicle.findData(selected)))
        self.cb_edit_vehicle.blockSignals(False)
        self._load_pose_controls()

    def _load_pose_controls(self, _index=None) -> None:
        key = self.cb_edit_vehicle.currentData()
        item = self.view.vehicles.get(key) if hasattr(self, "view") else None
        pose_enabled = item is not None and self.view.can_move(item.kind)
        speed_value = self._vehicle_initial_speed(key)

        if item is not None:
            self.view.select_key(key)
            x, y, heading = item.world_pose()
            values = {
                "x": x,
                "y": y,
                "heading": math.degrees(heading),
            }
            for field in ("x", "y", "heading"):
                spin = self.pose_inputs[field]
                spin.blockSignals(True)
                spin.setValue(values[field])
                spin.blockSignals(False)

        for field in ("x", "y", "heading"):
            self.pose_inputs[field].setEnabled(pose_enabled)

        speed_spin = self.pose_inputs.get("speed")
        if speed_spin is not None:
            speed_spin.blockSignals(True)
            if speed_value is not None:
                speed_spin.setValue(float(speed_value))
                vmax = self.cfg.max_initial_speed
                speed_spin.setStyleSheet(
                    "border:1px solid #c62828" if float(speed_value) > vmax + 1e-9 else "")
                speed_spin.setToolTip(
                    "超过当前 initial history 上限 %.3f m/s；不会自动截断，准备实验时会拒绝"
                    % vmax if float(speed_value) > vmax + 1e-9 else
                    "该值写入实验 initial.speed")
            else:
                speed_spin.setValue(0.0)
                speed_spin.setStyleSheet("")
                speed_spin.setToolTip("背景车不属于当前实验 initial 配置")
            speed_spin.blockSignals(False)
            speed_spin.setEnabled(pose_enabled and speed_value is not None)

    def _apply_pose_controls(self, _value=None) -> None:
        key = self.cb_edit_vehicle.currentData()
        item = self.view.vehicles.get(key)
        if item is None or not self.view.can_move(item.kind):
            return

        x = self.pose_inputs["x"].value()
        y = self.pose_inputs["y"].value()
        heading = math.radians(self.pose_inputs["heading"].value())
        self.view.update_live_vehicle(key, x, y, heading)
        self.on_vehicle_moved(key, x, y, heading)
        QTimer.singleShot(0, self._redraw_after_edit)

    def _apply_speed_control(self, value: float) -> None:
        key = self.cb_edit_vehicle.currentData()
        if not key or self._vehicle_initial_speed(key) is None:
            return
        self._set_vehicle_initial_speed(str(key), float(value), refresh=True)

    def _redraw_after_edit(self) -> None:
        if not self.view.editing_enabled or self.dataset is None:
            return
        key = self.cb_edit_vehicle.currentData()
        self.redraw_scene()
        if key:
            self.view.select_key(key)

    def on_builder_toggled(self, enabled: bool) -> None:
        if not hasattr(self, "view"):
            return
        self.btn_place.setChecked(False)
        self.view.set_builder_mode(enabled)
        self._load_pose_controls()
        self._update_action_states()
        if enabled:
            self.log("INFO", "场景构建：可拖动/旋转主车和背景车；另存时对整段轨迹做刚体变换。加载导出数据集后再运行实验。")

    def open_scene_pkl(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "打开场景 PKL", self.ed_path.text(), "场景文件 (*.pkl)")
        if path:
            self.ed_path.setText(path)
            self.on_path_edited()
            self.sp_index.setValue(0)
            self.load_scenario()

    def restore_scene_edits(self) -> None:
        if self._original_scene is None or not self.view.editing_enabled:
            return
        answer = QMessageBox.question(self, "还原场景编辑", "还原所有车辆位姿并移除本轮 virtual BV？源 PKL 不受影响。",
                                      QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer != QMessageBox.Yes:
            return
        self.scene = copy.deepcopy(self._original_scene)
        self.virtuals.clear()
        self._edited_track_poses.clear()
        self.speeds.clear()
        self.state["prepared"] = False
        self.refresh_all()

    def export_scene(self) -> None:
        from .scenario_editor import export_scene_variant
        if self.dataset is None or not self.view.editing_enabled:
            return
        default = os.path.join(os.path.dirname(os.path.abspath(self.dataset.path)),
                               "scene_variant_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        target, _ = QFileDialog.getSaveFileName(
            self, "新数据集目录名（将创建文件夹，含 PKL 和索引）", default, "场景数据集目录 (*)")
        if not target:
            return
        # Scene editor writes raw PKL tracks (pre-centralization).  Restore the
        # original file coordinates ONLY at this export boundary, never in GUI.
        poses = dict(self._edited_track_poses)
        poses.update({str(v["track_id"]): (v["x"], v["y"], v["heading"]) for v in self.virtuals})
        if not self.scene.is_demo:
            ox = float(self.scene.runtime_origin_x)
            oy = float(self.scene.runtime_origin_y)
            poses = {tid: (float(x) + ox, float(y) + oy, float(h))
                     for tid, (x, y, h) in poses.items()}
        try:
            result = export_scene_variant(self.dataset, self.scene_index,
                                          poses, target)
        except Exception as exc:
            QMessageBox.warning(self, "场景导出失败", str(exc))
            return
        self.log("OK", "新场景已另存：%s" % result)
        QMessageBox.information(self, "场景已保存", "新数据集：%s\n\n源文件保留不变。可从左侧加载新目录查看效果。\n编辑会平移/旋转该车整段轨迹，不会自动生成新的道路路线。" % target)

    def on_initial_state_row_selected(self) -> None:
        row = self.tbl_vehicle_initial.currentRow()
        item = self.tbl_vehicle_initial.item(row, 0) if row >= 0 else None
        if item is None:
            return
        key = item.data(Qt.UserRole)
        if not key:
            return
        self.view.select_key(str(key))
        index = self.cb_edit_vehicle.findData(str(key))
        if index >= 0 and self.cb_edit_vehicle.currentIndex() != index:
            self.cb_edit_vehicle.setCurrentIndex(index)

    def _select_initial_state_row(self, key: str) -> None:
        if not hasattr(self, "tbl_vehicle_initial"):
            return
        for row in range(self.tbl_vehicle_initial.rowCount()):
            item = self.tbl_vehicle_initial.item(row, 0)
            if item is not None and str(item.data(Qt.UserRole) or "") == str(key):
                self.tbl_vehicle_initial.selectRow(row)
                return

    def on_virtual_row_selected(self) -> None:
        """旧接口保留；统一初始状态表使用 on_initial_state_row_selected。"""
        self.on_initial_state_row_selected()

    def set_endpoint(self, track_id: str, endpoint: str) -> None:
        if self.cfg.mode != "hybrid":
            self.log("FAIL", "simulation 模式不使用运动平台端点")
            return
        if endpoint:
            for virtual in self.virtuals:
                if str(virtual.get("track_id")) == str(track_id):
                    self.log("FAIL", "track %s 已绑定给 %s，请先删除该 virtual BV" %
                             (track_id, virtual.get("key")))
                    self.refresh_vehicle_initial_table()
                    return
            for k, v in self.platform_bind.items():
                if k != track_id and v == endpoint:
                    self.log("FAIL", "端点 %s 已绑定给 track %s" % (endpoint, k))
                    self.refresh_vehicle_initial_table()
                    return
            self.platform_bind[track_id] = endpoint
            obj = next((o for o in self.scene.objects if o.track_id == track_id), None)
            if obj is not None:
                self.speeds["t" + track_id] = self._speed_for_initial(
                    self.speeds.get("t" + track_id, obj.speed), "track %s" % track_id)
            self.log("OK", "track %s → %s（%s）" %
                     (track_id, endpoint, self.cfg.endpoint_window(endpoint)))
        else:
            self.platform_bind.pop(track_id, None)
            self.log("INFO", "track %s 解除端点绑定" % track_id)
        self.state["prepared"] = False
        self.refresh_all()

    def remove_virtual(self, key: str) -> None:
        self.virtuals = [v for v in self.virtuals if v["key"] != key]
        self.state["prepared"] = False
        self.log("INFO", "删除 %s" % key)
        self.refresh_all()

    def remove_selected_virtual(self) -> None:
        """删除统一初始状态表中当前选中的 virtual BV。"""
        row = self.tbl_vehicle_initial.currentRow()
        item = self.tbl_vehicle_initial.item(row, 0) if row >= 0 else None
        if item is None or item.data(Qt.UserRole + 1) != "virtual":
            self.log("WARN", "请先在下方“实验车辆初始状态”表中选中一台 Virtual BV")
            return
        self.remove_virtual(str(item.data(Qt.UserRole)))

    def _speed_for_initial(self, speed: float, label: str) -> float:
        """保留用户/数据原速度；initial 超限只提示并由 prepare() 严格校验。"""
        value = float(speed)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("%s 初始速度非法：%r" % (label, speed))
        if self.cfg.scenario.get("history_source") == "initial":
            vmax = self.cfg.max_initial_speed
            if value > vmax + 1e-9:
                self.log(
                    "WARN",
                    "%s 初始速度 %.3f m/s 超过当前历史上限 %.3f m/s；不会自动修改，准备实验时将拒绝启动"
                    % (label, value, vmax),
                )
        return value

    def _set_buttons_loading(self, loading: bool) -> None:
        """兼容原调用点；按钮状态统一由状态机刷新。"""
        self._busy_action = "正在准备" if loading else ""
        self.btn_prepare.setText("… 准备中" if loading else "② 准备 BV")
        self._update_action_states()

    @staticmethod
    def _state_color(state: str) -> str:
        """运行状态 → 颜色映射。"""
        return {
            "idle": "#718096",
            "preparing": "#2563eb",
            "prepared": "#1e7d32",
            "running": "#1e7d32",
            "stopping": "#b26a00",
            "error": "#c62828",
        }.get(state, "#718096")

    def _show_shortcuts(self) -> None:
        """v2: 弹出快捷键速查对话框。"""
        QMessageBox.information(self, "地图快捷键",
            "滚轮 / 双指捏合    以光标为中心缩放\n"
            "中键拖拽 / Space+左键    平移画布\n"
            "F / 0 / 双击空白    适应全部\n"
            "A    聚焦车辆工作区\n"
            "+ / -    分级放大、缩小\n"
            "左键拖过空白    框选车辆\n"
            "✛ 放置模式    点击地图放置 virtual BV")

    # ==================================================================
    # 实验控制
    # ==================================================================
    def _runtime_xy(self, x: float, y: float) -> tuple:
        """UI is already in current ScenarioEnv A coordinates."""
        return float(x), float(y)

    def _scene_xy(self, x: float, y: float) -> tuple:
        """World/state and UI now share current ScenarioEnv A coordinates."""
        return float(x), float(y)

    def agents(self, runtime_coordinates: bool = False) -> List[AgentSpec]:
        out: List[AgentSpec] = []
        for track_id, endpoint in self.platform_bind.items():
            obj = next((o for o in self.scene.objects if o.track_id == track_id), None)
            if obj is None:
                continue
            endpoint_cfg = self.cfg.endpoint(endpoint) or {}
            x, y = self._runtime_xy(obj.x, obj.y) if runtime_coordinates else (obj.x, obj.y)
            out.append(AgentSpec(
                agent_id=str(endpoint_cfg.get("vehicle_id") or endpoint.replace("platform", "agent")),
                role="platform",
                platform_id=endpoint, scenario_track_id=track_id,
                x=x, y=y, heading=obj.heading,
                speed=self.speeds.get("t" + track_id, obj.speed)))
        for i, v in enumerate(self.virtuals, start=1):
            x, y = self._runtime_xy(v["x"], v["y"]) if runtime_coordinates else (v["x"], v["y"])
            out.append(AgentSpec(agent_id="agent_1%02d" % i, role="virtual",
                                 scenario_track_id=str(v.get("track_id", "")) or None,
                                 x=x, y=y, heading=v["heading"], speed=v["speed"]))
        return out

    def ego_spec(self, runtime_coordinates: bool = False) -> AgentSpec:
        e = self.scene.ego
        x, y = self._runtime_xy(e.x, e.y) if runtime_coordinates else (e.x, e.y)
        return AgentSpec("ego_001", "ego", x=x, y=y, heading=e.heading,
                         speed=self.speeds.get("ego", e.speed))

    def _dataset_pose_override_ids(self) -> List[str]:
        """dataset history ignores initial poses; expose preview/runtime mismatches."""
        if self.cfg.scenario.get("history_source") != "dataset":
            return []
        original = self._original_scene or self.scene
        tracks = {str(obj.track_id): obj for obj in [original.ego] + list(original.objects)}
        poses = dict(self._edited_track_poses)
        for vehicle in self.virtuals:
            poses[str(vehicle.get("track_id", ""))] = (
                vehicle["x"], vehicle["y"], vehicle["heading"])
        changed = []
        for track_id, (x, y, heading) in poses.items():
            source = tracks.get(str(track_id))
            if source is None:
                changed.append(str(track_id))
                continue
            angle = math.atan2(math.sin(heading - source.heading), math.cos(heading - source.heading))
            if (abs(x - source.x) > 1e-5 or abs(y - source.y) > 1e-5 or abs(angle) > 1e-5):
                changed.append(str(track_id))
        return sorted(set(changed))

    def _validate_pose_history_mode(self) -> bool:
        changed = self._dataset_pose_override_ids()
        if not changed:
            return True
        message = (
            "history_source=dataset 会使用 PKL 原始轨迹，不会采用拖拽后的初始位置。\n"
            "已修改车辆：%s\n\n"
            "请将 history_source 改为 initial；或另存新 PKL 并加载导出目录。\n"
            "本次未向 ROS 提交配置或启动实验。" % ", ".join(changed[:8])
        )
        self.state["prepared"] = False
        self.log("FAIL", message)
        self._update_action_states()
        QMessageBox.warning(self, "场景预览与运行配置不一致", message)
        return False

    def _validate_reference_calibration(self) -> bool:
        """Fail closed on unsupported dataset/automatic scenario switching."""
        platform = self.cfg.raw.get('platform', {}) or {}
        if str(platform.get('earth_to_autoscene_frame', 'runtime')).lower() != 'reference_scenario':
            return True
        reference_dataset = str(platform.get('calibration_reference_dataset', '')).strip()
        current_dataset = str(self.cfg.scenario.get('dataset_path', '')).strip()
        if reference_dataset and current_dataset != reference_dataset:
            msg = '当前数据集与 G→A 标定参考数据集不同；请先在此地图上重新标定，避免坐标错误。'
        elif bool(self.cfg.scenario.get('auto_advance_scenario', False)):
            msg = ('使用跨场景 G→A 校正时，不支持 worker 内自动切换 PKL：'
                   '请先将 scenario.auto_advance_scenario 设置为 false；'
                   '每次切换场景须在主 GUI 重新加载并“准备 BV”，更新坐标原点。')
        else:
            return True
        self.state['prepared'] = False
        self.log('FAIL', msg)
        QMessageBox.warning(self, 'G→A 场景安全检查未通过', msg)
        return False

    def prepare(self) -> None:
        if not self._connected:
            self.log("FAIL", "ROS 后端未连接，无法准备实验")
            return
        if self.dataset is None:
            self.log("FAIL", "请先加载数据集场景；示意场景不能用于实验")
            return
        if not self._validate_pose_history_mode() or not self._validate_reference_calibration():
            return
        if self._initial_speed_config_errors:
            for error in self._initial_speed_config_errors:
                self.log("FAIL", error)
            self.log("FAIL", "YAML initial.speed 配置无效；请修正后再准备实验")
            return
        agents = self.agents(runtime_coordinates=True)
        ego = self.ego_spec(runtime_coordinates=True)
        self.cfg.apply_agents(agents, ego)
        issues = self.cfg.validate(agents, ego,
                                   self.dataset.count if self.dataset else 0)
        errors = [i for i in issues if i.level == "error"]
        for i in issues:
            self.log("FAIL" if i.level == "error" else "WARN", i.text)
        if errors:
            return
        self.state.update({"state": "preparing", "prepared": False})
        # v2: 异步操作期间禁用操作按钮，防止重复点击
        self._set_buttons_loading(True)
        self.refresh_all()

        def after_params(result: dict) -> None:
            if not result.get("success"):
                self.state.update({"state": "error", "prepared": False})
                self.log("FAIL", "ROS 参数写入失败：%s" % result.get("message", ""))
                self._set_buttons_loading(False)
                self.refresh_all()
                return

            # 新 manager 没有 /experiment/prepare 服务。UI 的“准备 BV”只负责：
            # 1) 完成本地选择/校验；2) 把完整 /scenario /vehicles /ego 等写入 rosparam。
            # 真正的 manager.prepare_experiment_configuration() 会在点击“启动实验”时，
            # 由 interactive_console=false 的 manager.start() 自动执行。
            self.state.update({"state": "prepared", "prepared": True})
            self._algorithm_selection_dirty = False
            self.log(
                "OK",
                "UI 配置已准备：platform %d · virtual %d；点击“启动实验”后 manager 将非交互准备并启动"
                % (sum(1 for a in agents if a.role == "platform"),
                   sum(1 for a in agents if a.role == "virtual")),
            )
            self._set_buttons_loading(False)
            self.refresh_all()

        self.client.set_params(self._rosparam_payload(agents), done=after_params)

    def start_experiment(self) -> None:
        if not self._connected:
            self.log("FAIL", "ROS 后端未连接，无法启动实验")
            return
        if self._algorithm_selection_dirty or not self.state.get("prepared"):
            self.log("FAIL", "experiment is not prepared；先执行 ② 准备 BV")
            return
        if not self._validate_pose_history_mode() or not self._validate_reference_calibration():
            return

        # 从点击启动开始，地图进入 runtime live 模式。不要依赖 manager 状态消息
        # 晚一拍到达后才允许车辆更新，也不要让 builder checkbox 冻结 live pose。
        self._runtime_live_enabled = True
        self._world_state_seen = False
        self._raw_sim_ego_seen = False
        self._raw_sim_agent_ids.clear()
        self._live_source_logged.clear()
        self._live_vehicle_states.clear()
        self._refresh_live_vehicle_table()
        self._runtime_stream_retry_count = 0
        self._runtime_stream_retry_timer.start()
        if hasattr(self.client, "resubscribe_runtime_topics"):
            self.client.resubscribe_runtime_topics()
        self.view.set_live_mode(True)
        if self.chk_builder.isChecked():
            self.chk_builder.setChecked(False)

        self._busy_action = "正在启动"
        self._update_action_states()
        self.client.call("start", done=lambda result: self._finish_remote_action("启动", result))

    def stop_experiment(self) -> None:
        if not self._connected:
            self.log("FAIL", "ROS 后端未连接，无法停止实验")
            return
        self._busy_action = "正在停止"
        self._update_action_states()
        self.client.call("stop", done=lambda result: self._finish_remote_action("停止", result))

    def reset_experiment(self) -> None:
        if not self._connected:
            self.log("FAIL", "ROS 后端未连接，无法 Reset")
            return
        if self.platform_bind or self.virtuals or self.state.get("state") == "running":
            answer = QMessageBox.question(
                self, "确认 Reset",
                "Reset 会停止当前实验，并清除本轮端点绑定和所有 virtual BV。\n\n是否继续？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
        self._busy_action = "正在 Reset"
        self._update_action_states()
        self.client.call("reset", {"hard": False}, done=self._after_reset)

    def _after_reset(self, result: dict) -> None:
        self._busy_action = ""
        if result.get("success"):
            self._runtime_live_enabled = False
            self._world_state_seen = False
            self._raw_sim_ego_seen = False
            self._raw_sim_agent_ids.clear()
            self._runtime_stream_retry_timer.stop()
            self._live_vehicle_states.clear()
            self._refresh_live_vehicle_table()
            self.view.set_live_mode(False, clear=True)
            if hasattr(self.view, "clear_pose_overlays"):
                self.view.clear_pose_overlays(prefix="alignment:")
            self.virtuals.clear()
            self._next_virtual = 1
            self.platform_bind.clear()
            self.speeds.clear()
            self.state.update({"state": "idle", "prepared": False,
                               "step": 0, "sim_time": 0.0})
            self.log("INFO", "Reset 完成：已清除端点绑定与放置车辆")
        else:
            self.log("FAIL", "Reset 失败，本地配置未清除：%s" % result.get("message", ""))
        self.refresh_all()

    def _finish_remote_action(self, name: str, result: dict) -> None:
        self._busy_action = ""
        if not result.get("success"):
            if name == "启动":
                self._runtime_live_enabled = False
                self._runtime_stream_retry_timer.stop()
                self.view.set_live_mode(False)
            self.log("FAIL", "%s失败：%s" % (name, result.get("message", "无返回信息")))
        self._update_action_states()

    def save_yaml(self, as_new: bool = False) -> None:
        self.cfg.apply_agents(self.agents(runtime_coordinates=True),
                              self.ego_spec(runtime_coordinates=True))
        target = self.cfg.path
        if as_new or not target:
            target, _ = QFileDialog.getSaveFileName(self, "保存配置", target or "hybrid.yaml",
                                                    "YAML (*.yaml *.yml)")
            if not target:
                return
        self.cfg.save(target)
        self.log("OK", "已写入 %s" % target)

    def apply_params(self) -> None:
        if not self._connected:
            self.log("FAIL", "ROS 后端未连接，无法应用参数")
            return
        if not self._validate_pose_history_mode() or not self._validate_reference_calibration():
            return
        agents = self.agents(runtime_coordinates=True)
        self.cfg.apply_agents(agents, self.ego_spec(runtime_coordinates=True))
        self._busy_action = "正在应用参数"
        self._update_action_states()
        self.client.set_params(self._rosparam_payload(agents), done=self._after_apply_params)

    def _rosparam_payload(self, agents: Optional[List[AgentSpec]] = None) -> dict:
        """manager/runtime 按根参数读取配置，而不是读取 /autoscene/config。"""
        selected = agents if agents is not None else self.agents()
        remote_config = copy.deepcopy(self.cfg.raw)
        if not self.ros_same_host and self.remote_dataset_path:
            remote_config["scenario"]["dataset_path"] = self.remote_dataset_path

        # Keep the hidden PKL center offset only as calibration metadata.
        # GUI and ROS vehicle poses are already identical in current A.
        # Tagged reference_scenario calibration adjusts G→A when PKL changes.
        remote_config.setdefault("platform", {})["runtime_origin"] = {
            "x": float(getattr(self.scene, "runtime_origin_x", 0.0)),
            "y": float(getattr(self.scene, "runtime_origin_y", 0.0)),
        }

        # UI 已经把具体 scenario_track_id 和 initial 写入 /vehicles。
        # 使用 manager 的非交互准备分支，但禁止 runtime 再自动重选 BV：
        # - agent_assignment_mode=auto：避免 manager 进入终端 manual_assign_agents()
        # - auto_assign_agents=false：保留 UI 选择的 scenario_track_id
        remote_config["scenario"]["agent_assignment_mode"] = "auto"
        remote_config["scenario"]["auto_assign_agents"] = False
        return {
            "endpoints": [a.platform_id for a in selected if a.platform_id],
            "rosparams": {"/" + key: value for key, value in remote_config.items()},
        }

    def _after_apply_params(self, result: dict) -> None:
        self._busy_action = ""
        self.log("OK" if result.get("success") else "FAIL", result.get("message", "ROS 参数更新完成"))
        self._update_action_states()

    # ==================================================================
    # ROS 回调
    # ==================================================================
    def _on_connected(self, ok: bool, msg: str) -> None:
        self._connected = bool(ok)
        self.lab_link.setText(("● " if ok else "○ ") + msg)
        color = "#b26a00" if ok and self.backend == "mock" else ("#1e7d32" if ok else "#a03030")
        self.lab_link.setStyleSheet("color:%s" % color)
        if not ok:
            self._busy_action = ""
        self._update_action_states()

    def _on_status(self, msg: dict) -> None:
        # ExperimentStatus 本身没有 prepared 字段；ros_client 会根据 manager.state 推导。
        # 新的 UI 流程中，“准备 BV”先完成 rosparam 写入，此时 manager 仍是 idle，
        # 所以 idle 状态不能把 UI 已经准备好的配置标志清掉。
        local_prepared = bool(self.state.get("prepared"))
        self.state.update({k: msg.get(k, self.state.get(k)) for k in
                           ("state", "step", "sim_time", "real_time_factor", "elapsed")})
        incoming_state = str(self.state.get("state", "idle"))
        if incoming_state in ("preparing", "starting", "running", "stopping"):
            self._runtime_live_enabled = True
            self.view.set_live_mode(True)
            if not self._world_state_seen and not self._runtime_stream_retry_timer.isActive():
                self._runtime_stream_retry_timer.start()
        elif incoming_state in ("idle", "error"):
            self._runtime_live_enabled = False
            self._runtime_stream_retry_timer.stop()
            self.view.set_live_mode(False)

        backend_prepared = bool(msg.get("prepared")) or incoming_state in (
            "prepared", "starting", "running", "stopped"
        )
        if incoming_state == "error" or self._algorithm_selection_dirty:
            self.state["prepared"] = False
        else:
            self.state["prepared"] = local_prepared or backend_prepared
        self.lab_state.setText(str(self.state["state"]))
        # v2: 状态颜色映射
        state_color = self._state_color(str(self.state["state"]))
        self.lab_state.setStyleSheet("color:%s" % state_color)
        self.lab_step.setText("%s · %.1fs" % (self.state["step"], float(self.state["sim_time"] or 0)))
        rtf = float(self.state.get("real_time_factor") or 0)
        self.lab_rtf.setText("%.2f" % rtf if rtf else "—")
        self.lab_elapsed.setText("00:%04.1f" % float(self.state.get("elapsed") or 0))
        running = self.state["state"] == "running"
        self.lab_bar.setText("state=%s  prepared=%s  experiment=%s  scenario=%s  %s"
                             % (self.state["state"], self.state.get("prepared"),
                                self.cfg.raw["project"]["experiment_id"],
                                self.cfg.scenario["scenario_index"],
                                "● REC" if running else "○"))
        self._update_action_states()

    def _update_action_states(self) -> None:
        """统一维护按钮和编辑状态，防止断线、运行中或请求中重复提交。"""
        if not hasattr(self, "btn_start"):
            return
        busy = bool(self._busy_action)
        running = self.state.get("state") in ("preparing", "starting", "running", "stopping")
        loaded = self.dataset is not None
        remote_ready = self._connected and not busy
        building = self.chk_builder.isChecked()
        self.act_apply.setEnabled(remote_ready and not running and not building)
        self.btn_prepare.setEnabled(remote_ready and loaded and not running and not building)
        self.btn_start.setEnabled(remote_ready and bool(self.state.get("prepared")) and not running and not building)
        self.btn_stop.setEnabled(remote_ready and running)
        self.btn_reset.setEnabled(remote_ready)
        self.btn_load.setEnabled(not running and not busy)
        self.btn_place.setEnabled(loaded and not running and not busy)
        self.btn_delete_virtual.setEnabled(bool(self.virtuals) and not running and not busy)
        self.chk_ros_same_host.setEnabled(not running and not busy)
        for widget in (self.cb_mode, self.cb_dataset_type, self.cb_history,
                       self.ed_path, self.ed_remote_path, self.btn_browse,
                       self.sp_index, self.tbl_vehicle_initial,
                       self.tbl_initial, self.chk_builder, self.chk_snap_lane,
                       self.cb_edit_vehicle):
            widget.setEnabled(not running and not busy)
        self.btn_export_scene.setEnabled(loaded and not running and not busy)
        self.btn_open_pkl.setEnabled(not running and not busy)
        self.btn_restore_scene.setEnabled(loaded and not running and not busy)
        algorithm_enabled = not running and not busy and not building
        if hasattr(self, "cb_ego_method"):
            self.cb_ego_method.setEnabled(algorithm_enabled)
        if hasattr(self, "cb_adv_method"):
            self.cb_adv_method.setEnabled(algorithm_enabled)
        self.view.set_editing_enabled(not running and not busy)
        self._load_pose_controls()
        if busy:
            self.lab_bar.setText(self._busy_action + "…")

    def _world_vehicle_key_map(self) -> Dict[str, str]:
        """ROS vehicle_id → 当前 QGraphicsView 图元 key。"""
        ego_id = str(self.cfg.raw.get("ego", {}).get("id", "ego_001"))
        mapping = {"ego": "ego", ego_id: "ego"}
        for track_id, endpoint in self.platform_bind.items():
            endpoint_cfg = self.cfg.endpoint(endpoint) or {}
            vehicle_id = str(endpoint_cfg.get("vehicle_id") or endpoint.replace("platform", "agent"))
            mapping[vehicle_id] = str(track_id)
        for index, virtual in enumerate(self.virtuals, start=1):
            mapping["agent_1%02d" % index] = str(virtual["key"])
        for obj in self.scene.objects:
            mapping[str(obj.track_id)] = str(obj.track_id)
            mapping["background_%s" % obj.track_id] = str(obj.track_id)
        return mapping

    def _live_scene_update_blocked(self) -> bool:
        """Runtime live state always wins while an experiment is active.

        Before runtime starts, builder/edit mode may legitimately own vehicle poses.
        Once start has been requested or manager reports running, ROS/MetaDrive state
        must never be blocked by a stale builder checkbox.
        """
        if self._runtime_live_enabled:
            return False
        return bool(self.chk_builder.isChecked() and self.view.editing_enabled)

    def _apply_sim_vehicle_state(self, key: str, state: dict, source: str = "runtime") -> bool:
        """Apply one runtime VehicleState and persist it across any UI redraw.

        This updates only the live QGraphics overlay/cache.  It intentionally does not
        write the changing runtime pose back into initial-condition YAML/state.
        """
        if not bool(state.get("valid", True)):
            # Invalid placeholders may change opacity but must not invent a pose.
            return False
        pose = state.get("pose") or {}
        position = pose.get("position") or {}
        try:
            runtime_x = float(position.get("x"))
            runtime_y = float(position.get("y"))
            heading = float(state.get("heading", 0.0))
        except (TypeError, ValueError):
            return False
        if not (math.isfinite(runtime_x) and math.isfinite(runtime_y)
                and math.isfinite(heading)):
            return False
        twist = state.get("twist") or {}
        linear = twist.get("linear") or {}
        try:
            vx = float(linear.get("x", 0.0) or 0.0)
            vy = float(linear.get("y", 0.0) or 0.0)
            speed = math.hypot(vx, vy)
        except (TypeError, ValueError):
            speed = float("nan")
        self._live_vehicle_states[str(key)] = {
            "x": runtime_x, "y": runtime_y, "heading": heading,
            "speed": speed, "source": source,
            "vehicle_id": str(state.get("vehicle_id", key) or key),
        }
        x, y = self._scene_xy(runtime_x, runtime_y)
        applied = self.view.update_live_vehicle(
            str(key), x, y, heading, True, refresh_handle=False)
        if applied and source not in self._live_source_logged:
            self._live_source_logged.add(source)
            self.log("INFO", "UI live pose active: source=%s key=%s x=%.3f y=%.3f heading=%.3f"
                     % (source, key, x, y, heading))
        return applied

    def _on_sim_ego(self, msg: dict) -> None:
        """Startup fallback for raw MetaDrive ego state.

        /autoscene/world/state is authoritative because it is the runtime's fused
        state stream. The raw topic is allowed to move the ego only before the first
        usable world packet arrives, which preserves startup diagnostics without
        allowing a one-off raw message to permanently suppress world updates.
        """
        if self._world_state_seen or self._live_scene_update_blocked():
            return
        if self._apply_sim_vehicle_state("ego", msg, source="ego/state-fallback"):
            self._raw_sim_ego_seen = True
            self.lab_live_stream.setText("△ world/state 未到；暂用 ego/state 启动兜底")
            self._refresh_live_vehicle_table()
            self.view.refresh_handle()

    def _on_sim_agents(self, msg: dict) -> None:
        """Startup fallback for raw MetaDrive agent states.

        This path is disabled as soon as a usable world/state packet is applied.
        In hybrid mode that is essential: raw /agents/state is simulated state while
        world/state may contain actual platform feedback.
        """
        if self._world_state_seen or self._live_scene_update_blocked():
            return
        mapping = self._world_vehicle_key_map()
        updated = False
        for state in msg.get("vehicles") or msg.get("agents") or []:
            vehicle_id = str(state.get("vehicle_id", ""))
            key = mapping.get(vehicle_id)
            if key is None:
                continue
            if self._apply_sim_vehicle_state(key, state, source="agents/state-fallback"):
                self._raw_sim_agent_ids.add(vehicle_id)
                updated = True
        if updated:
            self.lab_live_stream.setText("△ world/state 未到；暂用 agents/state 启动兜底")
            self._refresh_live_vehicle_table()
            self.view.refresh_handle()

    def _on_world(self, msg: dict) -> None:
        """Drive all live vehicle bodies from /autoscene/world/state.

        The runtime publishes world/state after each successful MetaDrive step and
        merges actual feedback into it for hybrid operation. Therefore this stream is
        the single authoritative UI pose source for ego, configured BVs and background
        vehicles. Initial-condition objects/YAML are intentionally not mutated.
        """
        if self._live_scene_update_blocked():
            return

        mapping = self._world_vehicle_key_map()
        updated = False

        ego = msg.get("ego") or {}
        if ego:
            updated = self._apply_sim_vehicle_state(
                "ego", ego, source="world/state-ego") or updated

        for state in msg.get("agents") or []:
            vehicle_id = str(state.get("vehicle_id", "")).strip()
            if not vehicle_id:
                continue

            key = mapping.get(vehicle_id)
            # Defensive compatibility for background_<track_id>. The normal mapping
            # already contains this alias; the direct lookup prevents a harmless ID
            # formatting difference from freezing background traffic.
            if key is None and vehicle_id.startswith("background_"):
                track_id = vehicle_id[len("background_"):]
                if track_id in self.view.vehicles:
                    key = track_id
            if key is None:
                continue

            updated = self._apply_sim_vehicle_state(
                key, state, source="world/state") or updated

        if updated:
            # Commit ownership only after at least one pose was actually mapped and
            # applied. A malformed/empty world packet must not disable raw fallback.
            first_world = not self._world_state_seen
            self._world_state_seen = True
            self._runtime_stream_retry_timer.stop()
            self.lab_live_stream.setText("● world/state 实时更新")
            self.lab_live_stream.setStyleSheet("color:#187345")
            self._refresh_live_vehicle_table()
            self.view.refresh_handle()
            if first_world:
                self.log("OK", "UI 已接管 /autoscene/world/state；车辆图元与实时状态表开始跟随仿真")

        if "sim_time" in msg:
            self.state["sim_time"] = float(msg.get("sim_time") or 0.0)
            self.lab_step.setText("%s · %.1fs" %
                                  (self.state.get("step", 0), self.state["sim_time"]))

    def _ensure_runtime_stream(self) -> None:
        """Retry rosbridge runtime subscriptions until a real world/state frame arrives."""
        if self._world_state_seen:
            self._runtime_stream_retry_timer.stop()
            return
        if not self._connected or not self._runtime_live_enabled:
            return
        self._runtime_stream_retry_count += 1
        self.lab_live_stream.setText(
            "等待 world/state · 已重订阅 %d 次" % self._runtime_stream_retry_count)
        if hasattr(self.client, "resubscribe_runtime_topics"):
            self.client.resubscribe_runtime_topics()
        if self._runtime_stream_retry_count in (1, 3, 5):
            self.log("WARN", "尚未收到 /autoscene/world/state，正在刷新 rosbridge runtime 订阅（%d）"
                     % self._runtime_stream_retry_count)

    def _refresh_live_vehicle_table(self) -> None:
        if not hasattr(self, "tbl_live"):
            return
        rows = sorted(self._live_vehicle_states.values(), key=lambda row: str(row.get("vehicle_id", "")))
        self.tbl_live.setRowCount(len(rows))
        for r, row in enumerate(rows):
            values = [
                str(row.get("vehicle_id", "")),
                "%.3f" % float(row.get("x", 0.0)),
                "%.3f" % float(row.get("y", 0.0)),
                "%.5f" % float(row.get("heading", 0.0)),
                ("%.3f" % float(row.get("speed"))) if math.isfinite(float(row.get("speed", float("nan")))) else "—",
                str(row.get("source", "")),
            ]
            for c, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setFont(self._mono_font)
                self.tbl_live.setItem(r, c, item)

    def open_calibration_window(self) -> None:
        if self._calibration_window is None:
            self._calibration_window = CalibrationWindow(self.cfg, bridge_client=self.client, parent=self)
        if hasattr(self._calibration_window, "refresh_from_parent"):
            self._calibration_window.refresh_from_parent()
        self._calibration_window.show()
        self._calibration_window.raise_()
        self._calibration_window.activateWindow()

    def _on_ads_decoded(self, msg: dict) -> None:
        win = self._calibration_window
        # 关闭标定窗口后不再让后台 rosbridge 样本重新画回已清除的对齐标记。
        if win is not None and win.isVisible():
            win.on_ads_decoded(msg)

    def _trajectory_vehicle_key(self, traj: dict, mapping: Dict[str, str]) -> Optional[str]:
        """把 Trajectory 的 vehicle/platform 标识映射到当前地图图元 key。"""
        vehicle_id = str(traj.get("vehicle_id", "") or "")
        if vehicle_id and vehicle_id in mapping:
            return mapping[vehicle_id]
        platform_id = str(traj.get("platform_id", "") or "")
        if platform_id:
            for track_id, endpoint in self.platform_bind.items():
                if str(endpoint) == platform_id:
                    return str(track_id)
        return None

    def _trajectory_scene_points(self, traj: dict) -> List[tuple]:
        """Trajectory ROS A 坐标 → UI A 坐标；非法点严格丢弃整条轨迹。"""
        points = []
        for point in traj.get("points") or []:
            pose = point.get("pose") or {}
            position = pose.get("position") or {}
            try:
                runtime_x = float(position.get("x"))
                runtime_y = float(position.get("y"))
            except (TypeError, ValueError):
                return []
            if not math.isfinite(runtime_x) or not math.isfinite(runtime_y):
                return []
            points.append(self._scene_xy(runtime_x, runtime_y))
        return points

    def _on_planning_trajectories(self, msg: dict) -> None:
        """显示 runtime 当前 Ego/BV 规划轨迹；轨迹只画线，不驱动车身。"""
        if self._live_scene_update_blocked():
            return
        if not self.chk_trail.isChecked():
            self.view.clear_planned_trajectories(layer="planner")
            return
        mapping = self._world_vehicle_key_map()
        self.view.clear_planned_trajectories(layer="planner")
        for traj in msg.get("trajectories") or []:
            key = self._trajectory_vehicle_key(traj, mapping)
            if key is None or not bool(traj.get("valid", True)):
                continue
            points = self._trajectory_scene_points(traj)
            if len(points) >= 2:
                self.view.update_planned_trajectory(key, points, layer="planner")

    def _on_platform_command_trajectories(self, msg: dict) -> None:
        """显示 platform_comm 本周期真正编码/下发的轨迹。

        initial_once 下这是基于最新实际反馈每 0.1 s 重锚定的 50 点；
        10 s 完成后 platform_comm 发布空数组，本层立即清除。
        """
        if self.chk_builder.isChecked():
            return
        self.view.clear_planned_trajectories(layer="command")
        if not self.chk_trail.isChecked():
            return
        mapping = self._world_vehicle_key_map()
        for traj in msg.get("trajectories") or []:
            key = self._trajectory_vehicle_key(traj, mapping)
            if key is None or not bool(traj.get("valid", True)):
                continue
            points = self._trajectory_scene_points(traj)
            if len(points) >= 2:
                self.view.update_planned_trajectory(key, points, layer="command")

    def _on_metrics(self, msg: dict) -> None:
        for key, lab in self.metric_labels.items():
            val = msg.get(key)
            if val is None:
                continue
            lab.setText(str(val) if isinstance(val, bool) else "%.3f" % float(val))
            # v2: 默认灰色（无数据），收到数据后恢复正常样式
            lab.setStyleSheet("")
        ttc = msg.get("time_to_collision")
        if ttc is not None:
            ttc_val = float(ttc)
            if ttc_val < 1.5:
                color = "#c62828"       # 危险
            elif ttc_val < 2.5:
                color = "#b26a00"       # 警告
            else:
                color = "#1e7d32"       # 安全
            self.metric_labels["time_to_collision"].setStyleSheet(
                "color:%s;font-weight:600" % color)

    def _on_comm(self, msg: dict) -> None:
        rows = msg.get("platforms") or []
        self.tbl_comm.setRowCount(len(rows))
        for r, p in enumerate(rows):
            rx_age = p.get("last_rx_age", p.get("rx_age", -1))
            vals = [str(p.get("platform_id", "")), str(p.get("tx_count", 0)),
                    str(p.get("rx_count", 0)), "%.2f" % float(rx_age if rx_age is not None else -1)]
            for c, v in enumerate(vals):
                item = QTableWidgetItem(v)
                item.setFont(self._mono_font)
                if not p.get("connected", False):
                    item.setForeground(QColor("#999999"))
                self.tbl_comm.setItem(r, c, item)

    # ==================================================================
    # 刷新
    # ==================================================================
    def refresh_all(self, preserve_view: bool = True) -> None:
        hybrid = self.cfg.mode == "hybrid"
        self.lab_mode.setText("HYBRID · %s" % ("LIVE" if self.cfg.hardware_live else "DRY-RUN")
                              if hybrid else "SIMULATION")
        self.lab_mode.setStyleSheet(
            "background:#fdf0e0;border:1px solid #e0a869;color:#8a4000;padding:3px 8px" if hybrid
            else "background:#e6f2ea;border:1px solid #7fb391;color:#1e6b3a;padding:3px 8px")
        loaded = self.dataset is not None
        self.lab_loaded.setText("数据集已加载" if loaded else "示意场景")
        self.lab_loaded.setStyleSheet(
            "background:#e6f2ea;border:1px solid #7fb391;color:#1e6b3a;padding:3px 8px" if loaded
            else "background:#f0f0f0;border:1px dashed #b0b0b0;color:#777;padding:3px 8px")
        self.lab_constraint.setText(
            "hybrid 只允许 tjwest + initial（manager 会校验）" if hybrid
            else "simulation 允许 dataset 或 initial · 不使用运动平台端点")
        self.lab_index_range.setText("0..%d" % (self.dataset.count - 1) if self.dataset else "0..?")
        if self.dataset:
            self.sp_index.setMaximum(max(0, self.dataset.count - 1))

        p = self.cfg.platform
        self.lab_udp.setText("UDP 接收：%s    UDP 发送：%s\n模拟反馈：%s\n本机：%s:%s\n下发：%d Hz · %d 点 × %.1fs"
                             % ("开启" if p.get("receive_udp") else "关闭",
                                "开启" if p.get("send_udp") else "关闭",
                                "开启" if p.get("emulate_controlled_feedback") else "关闭",
                                p.get("local_ip"), p.get("local_port"), int(p.get("send_rate_hz", 10)),
                                int(p.get("packet_point_num", 50)), float(p.get("trajectory_dt_sec", 0.1))))
        self.list_endpoints.clear()
        used = set(self.platform_bind.values())
        for e in self.cfg.endpoints:
            pid = str(e.get("platform_id"))
            short_id = pid.replace("platform_", "P-")
            item = QListWidgetItem("%s · %s:%s · %s" %
                                   (short_id, e.get("ip"), e.get("send_port", e.get("port")),
                                    "已绑定" if pid in used else "空闲"))
            if pid not in used:
                item.setForeground(QColor("#888888"))
            self.list_endpoints.addItem(item)

        self.lab_n_platform.setText("%d / %d 车位" % (len(self.platform_bind), len(self.scene.physical_slots))
                                    if hybrid else "0（simulation 不适用）")
        self.lab_n_virtual.setText("%d 台（地图放置）" % len(self.virtuals))
        self.lab_comm_note.setText(
            "dry-run：不收发真实 UDP，由 platform_comm_node 模拟反馈。接实车需 send_udp/receive_udp=true 且 emulate=false"
            if hybrid and not self.cfg.hardware_live else
            ("实车模式：正在向端点下发 MsgID 12 轨迹" if hybrid else "simulation 模式不启用 platform_comm_node"))

        self.redraw_scene(preserve_view=preserve_view)
        self._refresh_vehicle_choices()
        self.refresh_vehicle_initial_table()
        self.refresh_algorithm_table()
        self.refresh_tabs()
        self._on_status(dict(self.state))
        # initial 位置/速度/航向一旦在主界面修改，已打开的标定窗口同步重算
        # 第0帧目标，并把绿色目标标记恢复到 SceneView。
        win = getattr(self, "_calibration_window", None)
        if win is not None and win.isVisible() and hasattr(win, "refresh_from_parent"):
            try:
                win.refresh_from_parent()
            except Exception as exc:
                self.log("WARN", "刷新标定/对齐可视化失败：%s" % exc)

    def redraw_scene(self, preserve_view: bool = True) -> None:
        self.view.show_trails = self.chk_trail.isChecked()
        self.view.show_ids = self.chk_id.isChecked()
        self.view.show_grid = self.chk_grid.isChecked()
        self.view.set_scene_data(self.scene, self.platform_bind, self.virtuals,
                                 preserve_view=preserve_view)

    def _initial_state_rows(self, include_unbound_platform: bool = True) -> List[dict]:
        """返回初始状态行；所有 x/y/heading 均是当前 ScenarioEnv A。"""
        rows = [{
            "key": "ego",
            "kind": "ego",
            "role": "Ego",
            "vehicle_id": str(self.cfg.raw.get("ego", {}).get("id", "ego_001")),
            "track_id": str(self.scene.ego.track_id),
            "platform_id": "",
            "x": float(self.scene.ego.x),
            "y": float(self.scene.ego.y),
            "heading": float(self.scene.ego.heading),
            "speed": float(self.speeds.get("ego", self.scene.ego.speed)),
        }]

        virtual_track_ids = {str(v.get("track_id")) for v in self.virtuals if v.get("track_id") is not None}
        for obj in self.scene.physical_slots:
            track_id = str(obj.track_id)
            if track_id in virtual_track_ids:
                # 该 dataset track 已由 Virtual BV 占用，不能同时绑定 platform。
                continue
            endpoint = str(self.platform_bind.get(track_id, ""))
            if not include_unbound_platform and not endpoint:
                continue
            endpoint_cfg = self.cfg.endpoint(endpoint) or {}
            rows.append({
                "key": track_id,
                "kind": "platform",
                "role": "Platform BV" if endpoint else "Platform 候选",
                "vehicle_id": str(endpoint_cfg.get("vehicle_id") or "—"),
                "track_id": track_id,
                "platform_id": endpoint,
                "x": float(obj.x),
                "y": float(obj.y),
                "heading": float(obj.heading),
                "speed": float(self.speeds.get("t" + track_id, obj.speed)),
            })

        for index, virtual in enumerate(self.virtuals, start=1):
            rows.append({
                "key": str(virtual["key"]),
                "kind": "virtual",
                "role": "Virtual BV",
                "vehicle_id": "agent_1%02d" % index,
                "track_id": str(virtual.get("track_id", "")),
                "platform_id": "",
                "x": float(virtual["x"]),
                "y": float(virtual["y"]),
                "heading": float(virtual["heading"]),
                "speed": float(virtual["speed"]),
            })
        return rows

    def _vehicle_initial_speed(self, key: str):
        key = str(key or "")
        if key == "ego":
            return float(self.speeds.get("ego", self.scene.ego.speed))
        for virtual in self.virtuals:
            if str(virtual.get("key")) == key:
                return float(virtual.get("speed", 0.0))
        obj = next((o for o in self.scene.physical_slots if str(o.track_id) == key), None)
        if obj is not None:
            return float(self.speeds.get("t" + key, obj.speed))
        return None

    def _set_vehicle_initial_speed(self, key: str, value: float, refresh: bool = True) -> None:
        value = self._speed_for_initial(float(value), str(key))
        key = str(key)
        if key == "ego":
            self.speeds["ego"] = value
        else:
            for virtual in self.virtuals:
                if str(virtual.get("key")) == key:
                    virtual["speed"] = value
                    break
            else:
                obj = next((o for o in self.scene.physical_slots if str(o.track_id) == key), None)
                if obj is None:
                    raise ValueError("找不到可配置 initial.speed 的车辆：%s" % key)
                self.speeds["t" + key] = value
        self.state["prepared"] = False
        if refresh:
            # 延后到当前 spinbox 信号返回后再重建表格，避免删除正在发信号的 cell widget。
            QTimer.singleShot(0, self._refresh_pose_tables)

    def _set_vehicle_initial_field(self, key: str, field: str, value: float) -> None:
        """统一表/初始状态 tab 修改 x/y/heading/speed，并同步地图与 YAML。"""
        key = str(key)
        if field == "speed":
            self._set_vehicle_initial_speed(key, value, refresh=True)
            return
        if field not in ("x", "y", "heading"):
            raise ValueError("unsupported initial field: %s" % field)

        virtual = next((v for v in self.virtuals if str(v.get("key")) == key), None)
        if virtual is not None:
            x = float(virtual["x"])
            y = float(virtual["y"])
            heading = float(virtual["heading"])
        else:
            obj = self.scene.ego if key == "ego" else next(
                (o for o in self.scene.objects if str(o.track_id) == key), None)
            if obj is None:
                raise ValueError("找不到可编辑车辆：%s" % key)
            x, y, heading = float(obj.x), float(obj.y), float(obj.heading)

        if field == "x":
            x = float(value)
        elif field == "y":
            y = float(value)
        else:
            heading = float(value)

        self.view.update_live_vehicle(key, x, y, heading)
        self.on_vehicle_moved(key, x, y, heading)
        self._select_initial_state_row(key)

    def _make_initial_spin(self, key: str, field: str, value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        if field in ("x", "y"):
            spin.setRange(-1e9, 1e9)
            spin.setDecimals(3)
            spin.setSingleStep(0.1)
        elif field == "heading":
            spin.setRange(-1000000.0, 1000000.0)
            spin.setDecimals(5)
            spin.setSingleStep(math.radians(1.0))
        elif field == "speed":
            vmax = self.cfg.max_initial_speed
            spin.setRange(0.0, 1000000.0)
            spin.setDecimals(3)
            spin.setSingleStep(0.1)
            if float(value) > vmax + 1e-9:
                spin.setStyleSheet("border:1px solid #c62828")
                spin.setToolTip(
                    "超过当前 initial history 上限 %.3f m/s；不会自动截断，准备实验时会拒绝" % vmax)
        else:
            raise ValueError("unsupported initial field: %s" % field)
        spin.setKeyboardTracking(False)
        spin.setValue(float(value))
        spin.valueChanged.connect(
            lambda val, k=str(key), f=field: self._set_vehicle_initial_field(k, f, val))
        return spin

    def refresh_vehicle_initial_table(self) -> None:
        selected_key = ""
        current_row = self.tbl_vehicle_initial.currentRow()
        if current_row >= 0:
            current_item = self.tbl_vehicle_initial.item(current_row, 0)
            if current_item is not None:
                selected_key = str(current_item.data(Qt.UserRole) or "")

        rows = self._initial_state_rows(include_unbound_platform=True)
        vmax = self.cfg.max_initial_speed
        active_rows = self._initial_state_rows(include_unbound_platform=False)
        over = [row["vehicle_id"] for row in active_rows if row["speed"] > vmax + 1e-9]
        self.lab_state_editor_hint.setText(
            "直接编辑 Ego / Platform / Virtual BV 的 x、y、heading、speed；地图拖拽/旋转与本表双向同步。"
            " Platform 候选可在同一行绑定端点。当前 initial.speed 上限 %.3f m/s。%s"
            % (vmax, (" 超限：" + ", ".join(over)) if over else "")
        )
        self.lab_state_editor_hint.setStyleSheet("color:#c62828" if over else "color:#526173")

        self.tbl_vehicle_initial.blockSignals(True)
        self.tbl_vehicle_initial.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            role_item = QTableWidgetItem(row["role"])
            role_item.setData(Qt.UserRole, row["key"])
            role_item.setData(Qt.UserRole + 1, row["kind"])
            role_item.setFlags(role_item.flags() & ~Qt.ItemIsEditable)
            self.tbl_vehicle_initial.setItem(row_index, 0, role_item)

            for column, value in ((1, row["vehicle_id"]), (2, row["track_id"])):
                item = QTableWidgetItem(str(value))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.tbl_vehicle_initial.setItem(row_index, column, item)

            if row["kind"] == "platform":
                combo = QComboBox()
                combo.addItem("未绑定", "")
                for pid in self.cfg.endpoint_ids():
                    combo.addItem(pid.replace("platform_", "P-"), pid)
                index = combo.findData(row["platform_id"])
                combo.setCurrentIndex(index if index >= 0 else 0)
                combo.setEnabled(self.cfg.mode == "hybrid")
                combo.currentIndexChanged.connect(
                    lambda _i, tid=row["track_id"], cb=combo: self.set_endpoint(tid, cb.currentData()))
                combo.setToolTip(
                    self.cfg.endpoint_window(row["platform_id"]) if row["platform_id"] else "选择运动平台端点")
                self.tbl_vehicle_initial.setCellWidget(row_index, 3, combo)
            else:
                item = QTableWidgetItem("—")
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.tbl_vehicle_initial.setItem(row_index, 3, item)

            for column, field in ((4, "x"), (5, "y"), (6, "heading"), (7, "speed")):
                self.tbl_vehicle_initial.setCellWidget(
                    row_index, column, self._make_initial_spin(row["key"], field, row[field]))

            if row["kind"] == "virtual":
                button = QPushButton("删除")
                button.setToolTip("删除 %s" % row["key"])
                button.clicked.connect(lambda _checked=False, key=row["key"]: self.remove_virtual(key))
                self.tbl_vehicle_initial.setCellWidget(row_index, 8, button)
            else:
                item = QTableWidgetItem("")
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.tbl_vehicle_initial.setItem(row_index, 8, item)

        self.tbl_vehicle_initial.blockSignals(False)
        if selected_key:
            self._select_initial_state_row(selected_key)

    # 旧函数名保留为轻量兼容入口，避免外部扩展调用失效。
    def refresh_platform_table(self) -> None:
        self.refresh_vehicle_initial_table()

    def refresh_virtual_table(self) -> None:
        self.refresh_vehicle_initial_table()

    def _set_virtual(self, key: str, field: str, value: float) -> None:
        self._set_vehicle_initial_field(key, field, value)

    def refresh_tabs(self) -> None:
        idx = self.tabs.currentIndex()
        if idx == 1:
            self.refresh_initial_table()
        elif idx == 2:
            self.refresh_yaml()
        elif idx == 3 and self.list_history.count() == 0:
            self.reload_history()

    def refresh_initial_table(self) -> None:
        vmax = self.cfg.max_initial_speed
        self.lab_init_head.setText(
            "与“场景可视化”页下方统一初始状态表使用同一份数据。"
            "速度上限 = initial_history_accel_mps2 × pluto_history_dt_sec × "
            "(pluto_history_steps − 1) = %.3f m/s；地图、表格、ROS 均使用当前 MetaDrive A 坐标。" % vmax)

        rows = self._initial_state_rows(include_unbound_platform=False)
        self.tbl_initial.blockSignals(True)
        self.tbl_initial.setRowCount(len(rows))
        over = []
        for row_index, row in enumerate(rows):
            for column, value in ((0, row["vehicle_id"]), (1, row["role"])):
                item = QTableWidgetItem(str(value))
                item.setFont(self._mono_font)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.tbl_initial.setItem(row_index, column, item)

            for column, field in ((2, "x"), (3, "y"), (4, "heading"), (5, "speed")):
                self.tbl_initial.setCellWidget(
                    row_index, column, self._make_initial_spin(row["key"], field, row[field]))
            if row["speed"] > vmax + 1e-9:
                over.append(row["vehicle_id"])

        self.tbl_initial.blockSignals(False)
        if over:
            self.lab_init_hint.setText("✗ 初始速度超限：%s" % ", ".join(over))
            self.lab_init_hint.setStyleSheet("color:#c62828")
        else:
            self.lab_init_hint.setText("✓ 所有参与实验车辆的初始速度均在允许范围内")
            self.lab_init_hint.setStyleSheet("color:#1e7d32")
        self.btn_fix_speeds.setEnabled(bool(over) and self.state.get("state") != "running")

    def clamp_initial_speeds(self) -> None:
        """将 initial 历史无法达到的速度修正到当前算法上限。"""
        changed = self._clamp_initial_speeds(refresh=True)
        self.log("OK", "已将 %d 台车辆的初始速度修正到 %.3f m/s" %
                 (changed, self.cfg.max_initial_speed))

    def _clamp_initial_speeds(self, refresh: bool) -> int:
        """修正所有实际参与实验的车辆；未绑定候选车不属于实验车辆。"""
        vmax = self.cfg.max_initial_speed
        changed = 0
        if self.ego_spec().speed > vmax:
            self.speeds["ego"] = vmax
            changed += 1
        for track_id in self.platform_bind:
            obj = next((o for o in self.scene.objects if o.track_id == track_id), None)
            if obj and self.speeds.get("t" + track_id, obj.speed) > vmax:
                self.speeds["t" + track_id] = vmax
                changed += 1
        for virtual in self.virtuals:
            if float(virtual.get("speed", 0.0)) > vmax:
                virtual["speed"] = vmax
                changed += 1
        if changed:
            self.state["prepared"] = False
            if refresh:
                self.refresh_all()
        return changed

    def _set_initial_speed(self, key: str, role: str, value: float) -> None:
        """旧 initial tab 回调兼容入口；新 UI 统一走 key → initial state。"""
        if role == "virtual":
            for index, virtual in enumerate(self.virtuals, start=1):
                if "agent_1%02d" % index == key:
                    self._set_vehicle_initial_speed(str(virtual["key"]), value)
                    return
            return
        self._set_vehicle_initial_speed(key, value)

    def refresh_yaml(self) -> None:
        self.cfg.apply_agents(self.agents(runtime_coordinates=True),
                              self.ego_spec(runtime_coordinates=True))
        self._speed_config_context = self._current_speed_context()
        self.ed_yaml.setPlainText(self.cfg.dump())

    # ---- 历史结果 ----------------------------------------------------
    def outputs_dir(self) -> str:
        base = self.cfg.raw["evaluation"].get("output_dir", "./outputs")
        if os.path.isabs(base):
            return base
        root = os.path.dirname(os.path.abspath(self.cfg.path)) if self.cfg.path else os.getcwd()
        return os.path.normpath(os.path.join(root, "..", "..", base))

    def reload_history(self) -> None:
        self.list_history.clear()
        self._history_files: List[str] = sorted(
            glob.glob(os.path.join(self.outputs_dir(), "*_evaluation_summary.json")), reverse=True)
        for f in self._history_files:
            self.list_history.addItem(os.path.basename(f))
        if not self._history_files:
            self.lab_hist_title.setText("outputs 目录下没有 *_evaluation_summary.json：%s" % self.outputs_dir())

    def show_history_detail(self, row: int) -> None:
        if row < 0 or row >= len(getattr(self, "_history_files", [])):
            return
        path = self._history_files[row]
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:  # noqa: BLE001
            self.lab_hist_title.setText("读取失败：%s" % exc)
            return
        self.lab_hist_title.setText(os.path.basename(path))
        flat = _flatten(data)
        self.tbl_hist.setRowCount(len(flat))
        for r, (k, v) in enumerate(flat):
            for c, text in enumerate([k, str(v)]):
                item = QTableWidgetItem(text)
                item.setFont(self._mono_font)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.tbl_hist.setItem(r, c, item)

    def closeEvent(self, event):
        if self.state.get("state") in ("starting", "running", "stopping"):
            answer = QMessageBox.question(
                self, "实验仍在运行",
                "关闭界面不会自动停止 ROS 后端实验。\n\n仍要关闭界面吗？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                event.ignore()
                return
        if self._persist_ui_state:
            settings = QSettings("AutoScene", "ExperimentManager")
            settings.setValue("window/geometry", self.saveGeometry())
            settings.setValue("window/main_splitter", self.main_splitter.saveState())
        self.client.shutdown()
        super().closeEvent(event)


def _flatten(data, prefix: str = "") -> List[tuple]:
    out: List[tuple] = []
    if isinstance(data, dict):
        for k, v in data.items():
            out.extend(_flatten(v, "%s.%s" % (prefix, k) if prefix else str(k)))
    elif isinstance(data, list):
        out.append((prefix, json.dumps(data, ensure_ascii=False)))
    else:
        out.append((prefix, data))
    return out