"""与 AutoScene 的 ROS 侧通信。

你的机器是 Ubuntu 22.04 且没有 ROS1，所以这里提供三种后端，界面代码完全一致：

  MockClient      —— 不连任何东西，本地模拟 status / metrics / comm，用来做 UI 演示与调试
  RosbridgeClient —— 通过 rosbridge_suite 的 websocket 连师兄那台 20.04（推荐）
                     对端执行： roslaunch rosbridge_server rosbridge_websocket.launch
                     本机只需 pip install websocket-client，不需要装 ROS
  RospyClient     —— 只有在同机有 ROS1 环境时才可用（import rospy 成功时启用）

订阅：experiment/status、scenario/status、simulation/status、world/state、
      evaluation/metrics、platform/comm_status
调用：experiment/start、experiment/stop、experiment/reset；参数通过 rosapi/set_param 写入
"""

from __future__ import annotations

import json
import math
import threading
import time
from typing import Any, Callable, Dict, Optional

from PyQt5.QtCore import QObject, QTimer, pyqtSignal

TOPICS = {
    "status": ("/autoscene/experiment/status", "autoscene_msgs/ExperimentStatus"),
    "scenario": ("/autoscene/scenario/status", "autoscene_msgs/ScenarioStatus"),
    "simulation": ("/autoscene/simulation/status", "autoscene_msgs/SimulationStatus"),
    "world": ("/autoscene/world/state", "autoscene_msgs/WorldState"),
    # Raw MetaDrive simulation state.  UI vehicle bodies use these topics directly;
    # world/state remains the fused/evaluation stream and may contain actual platform feedback.
    "sim_ego": ("/autoscene/ego/state", "autoscene_msgs/VehicleState"),
    "sim_agents": ("/autoscene/agents/state", "autoscene_msgs/VehicleStateArray"),
    "planning": ("/autoscene/planning/trajectories", "autoscene_msgs/TrajectoryArray"),
    "platform_command": ("/autoscene/platform/command_trajectory", "autoscene_msgs/TrajectoryArray"),
    "metrics": ("/autoscene/evaluation/metrics", "autoscene_msgs/EvaluationMetrics"),
    "comm": ("/autoscene/platform/comm_status", "autoscene_msgs/PlatformCommStatus"),
    # Raw ADS/GNSS decode JSON from platform_comm_node; used by calibration UI.
    "ads_decoded": ("/autoscene/platform/ads_udp_decoded", "std_msgs/String"),
}
SERVICES = {
    "load": "/autoscene/experiment/load",
    "start": "/autoscene/experiment/start",
    "stop": "/autoscene/experiment/stop",
    "reset": "/autoscene/experiment/reset",
}


class BaseClient(QObject):
    connected = pyqtSignal(bool, str)
    status = pyqtSignal(dict)
    world = pyqtSignal(dict)
    sim_ego = pyqtSignal(dict)
    sim_agents = pyqtSignal(dict)
    planning = pyqtSignal(dict)
    platform_command = pyqtSignal(dict)
    metrics = pyqtSignal(dict)
    comm = pyqtSignal(dict)
    ads_decoded = pyqtSignal(dict)
    log = pyqtSignal(str, str)   # level, text
    _done = pyqtSignal(object, dict)

    name = "base"

    def __init__(self) -> None:
        super().__init__()
        # 后端回调可能来自 websocket/rospy 线程，统一排队回到 Qt 主线程。
        self._done.connect(lambda callback, result: callback(result))

    def start(self) -> None: ...
    def shutdown(self) -> None: ...
    def call(self, key: str, payload: Optional[Dict[str, Any]] = None,
             done: Optional[Callable[[dict], None]] = None) -> None: ...
    def set_params(self, params: Dict[str, Any],
                   done: Optional[Callable[[dict], None]] = None) -> None: ...

    def resubscribe_runtime_topics(self) -> None:
        """Best-effort refresh for topics whose publishers start after the UI."""
        return

    def _finish(self, done: Optional[Callable[[dict], None]], result: dict) -> None:
        if done:
            self._done.emit(done, result)


# --------------------------------------------------------------------------
class MockClient(BaseClient):
    """离线演示：本地生成与真实话题同字段的数据。"""

    name = "mock"

    def __init__(self) -> None:
        super().__init__()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.step = 0
        self.running = False
        self.prepared = False
        self.bound_endpoints: list = []

    def start(self) -> None:
        self.connected.emit(True, "离线模拟（未连接 ROS）")
        self.timer.start(100)

    def shutdown(self) -> None:
        self.timer.stop()

    def set_params(self, params: Dict[str, Any], done=None) -> None:
        self.bound_endpoints = list(params.get("endpoints") or [])
        self._finish(done, {"success": True, "message": "mock params"})

    def call(self, key: str, payload=None, done=None) -> None:
        if key == "load":
            self.prepared = True
            self.log.emit("OK", "mock: 场景已加载并 prepare")
        elif key == "start":
            if not self.prepared:
                self.log.emit("FAIL", "mock: experiment is not prepared")
                self._finish(done, {"success": False, "message": "not prepared"})
                return
            self.running, self.step = True, 0
            self.log.emit("OK", "mock: experiment started")
        elif key == "prepare":
            self.prepared = True
            self.log.emit("OK", "mock: 配置已准备")
        elif key == "stop":
            self.running = False
            self.log.emit("INFO", "mock: experiment stopped")
        elif key == "reset":
            self.running, self.prepared, self.step = False, False, 0
            self.log.emit("INFO", "mock: reset")
        self._finish(done, {"success": True, "message": "mock"})

    def _tick(self) -> None:
        if self.running:
            self.step += 1
        t = self.step * 0.1
        self.status.emit({
            "state": "running" if self.running else ("prepared" if self.prepared else "idle"),
            "step": self.step, "sim_time": round(t, 2),
            "real_time_factor": 0.98 if self.running else 0.0,
            "prepared": self.prepared,
        })
        if self.running:
            self.metrics.emit({
                "collision": False,
                "min_distance": max(3.4, 12.6 - t * 0.62 + math.sin(t) * 0.5),
                "time_to_collision": max(1.7, 9.4 - t * 0.42 + math.cos(t * 0.8) * 0.35),
                "ego_speed_mean": min(2.61, 0.3 + t * 0.22),
                "ego_speed_max": min(4.02, 0.5 + t * 0.4),
                "platform_tracking_error": min(0.184, t * 0.02),
            })
            self.comm.emit({"platforms": [
                {"platform_id": pid, "tx_count": self.step + 3, "rx_count": self.step,
                 "rx_age": 0.03, "connected": True}
                for pid in self.bound_endpoints
            ]})


# --------------------------------------------------------------------------
class RosbridgeClient(BaseClient):
    """rosbridge websocket 客户端；本机无需安装 ROS。"""

    name = "rosbridge"
    REQUEST_TIMEOUT_SEC = 10.0

    def __init__(self, url: str = "ws://127.0.0.1:9090") -> None:
        super().__init__()
        self.url = url
        self._ws = None
        self._thread: Optional[threading.Thread] = None
        self._alive = False
        self._pending: Dict[str, Callable[[dict], None]] = {}
        self._pending_timers: Dict[str, threading.Timer] = {}
        self._pending_lock = threading.Lock()
        self._seq = 0
        self._send_lock = threading.Lock()
        self._status_cache: Dict[str, Any] = {}
        self._comm_cache: Dict[str, dict] = {}
        self._rx_counts: Dict[str, int] = {}
        self._first_rx_logged = set()

    def start(self) -> None:
        try:
            import websocket  # noqa: F401  # websocket-client
        except ImportError:
            self.connected.emit(False, "缺少 websocket-client：pip install websocket-client")
            return
        self._alive = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        self._alive = False
        self._fail_pending("前端正在关闭")
        try:
            if self._ws:
                self._ws.close()
        except Exception:  # noqa: BLE001
            pass

    def _run(self) -> None:
        import websocket

        while self._alive:
            try:
                self._ws = websocket.create_connection(self.url, timeout=5)
                self.connected.emit(True, "rosbridge %s" % self.url)
                for key in TOPICS:
                    self._subscribe_key(key)
                while self._alive:
                    raw = self._ws.recv()
                    if not raw:
                        break
                    self._dispatch(json.loads(raw))
                if self._alive:
                    self.connected.emit(False, "rosbridge 连接已断开，正在重连")
                    self._fail_pending("rosbridge 连接中断")
                    self._ws = None
                    time.sleep(2.0)
            except Exception as exc:  # noqa: BLE001
                self.connected.emit(False, "rosbridge 连接失败：%s" % exc)
                self._fail_pending("rosbridge 连接中断")
                self._ws = None
                time.sleep(2.0)

    def _subscribe_key(self, key: str) -> None:
        topic, mtype = TOPICS[key]
        self._send({
            "op": "subscribe",
            "topic": topic,
            "type": mtype,
            "throttle_rate": 0,
            "queue_length": 1,
            "id": "sub_%s" % key,
        })

    def resubscribe_runtime_topics(self) -> None:
        """Refresh transient runtime subscriptions after worker roslaunch starts.

        The GUI connects to rosbridge before scenario_runtime_node exists. Most
        rosbridge versions keep such subscriptions correctly, but some deployments
        drop custom-message subscriptions created before the publisher is advertised.
        Re-subscribing is idempotent and avoids a MetaDrive-running/UI-static state.
        """
        if not self._ws:
            return
        keys = (
            # Manager/status topics are included too: the UI may have connected before
            # a latched/custom-message subscription was fully established.
            "status", "scenario", "simulation",
            "world", "sim_ego", "sim_agents", "planning",
            "platform_command", "metrics", "comm", "ads_decoded",
        )
        for key in keys:
            topic, _mtype = TOPICS[key]
            self._send({"op": "unsubscribe", "topic": topic, "id": "sub_%s" % key})
            self._subscribe_key(key)
        self.log.emit("INFO", "已刷新 runtime rosbridge 订阅（等待 world/state）")

    def _register_pending(self, request_id: str, callback: Callable[[dict], None],
                          description: str) -> None:
        """登记服务请求并设置超时，避免 UI 永久停在处理中。"""
        timer = threading.Timer(
            self.REQUEST_TIMEOUT_SEC,
            lambda: self._expire_pending(request_id, description),
        )
        timer.daemon = True
        with self._pending_lock:
            self._pending[request_id] = callback
            self._pending_timers[request_id] = timer
        timer.start()

    def _pop_pending(self, request_id: str):
        with self._pending_lock:
            callback = self._pending.pop(request_id, None)
            timer = self._pending_timers.pop(request_id, None)
        if timer:
            timer.cancel()
        return callback

    def _expire_pending(self, request_id: str, description: str) -> None:
        callback = self._pop_pending(request_id)
        if callback:
            message = "%s 超时（%.0f 秒）" % (description, self.REQUEST_TIMEOUT_SEC)
            self.log.emit("FAIL", message)
            self._finish(callback, {"success": False, "message": message})

    def _fail_pending(self, message: str) -> None:
        with self._pending_lock:
            request_ids = list(self._pending)
        for request_id in request_ids:
            callback = self._pop_pending(request_id)
            if callback:
                self._finish(callback, {"success": False, "message": message})

    def _send(self, msg: dict) -> bool:
        if not self._ws:
            return False
        try:
            with self._send_lock:
                self._ws.send(json.dumps(msg))
            return True
        except Exception as exc:  # noqa: BLE001
            self.log.emit("FAIL", "rosbridge 发送失败：%s" % exc)
            return False

    def _dispatch(self, msg: dict) -> None:
        op = msg.get("op")
        if op == "publish":
            topic = msg.get("topic", "")
            payload = msg.get("msg", {})
            for key, (t, _m) in TOPICS.items():
                if topic == t:
                    self._rx_counts[key] = int(self._rx_counts.get(key, 0)) + 1
                    if key in ("status", "scenario", "simulation", "world", "sim_ego", "sim_agents", "ads_decoded") and key not in self._first_rx_logged:
                        self._first_rx_logged.add(key)
                        self.log.emit("INFO", "rosbridge RX active: %s" % topic)
                    if key == "status":
                        self._status_cache.update(payload)
                        self._status_cache["prepared"] = payload.get("state") in (
                            "prepared", "starting", "running", "stopped")
                        self.status.emit(dict(self._status_cache))
                    elif key == "scenario":
                        self._status_cache.update({
                            "step": payload.get("step", 0),
                            "scenario_loaded": payload.get("loaded", False),
                            "scenario_running": payload.get("running", False),
                        })
                        self.status.emit(dict(self._status_cache))
                    elif key == "simulation":
                        self._status_cache.update({
                            "sim_time": payload.get("sim_time", 0.0),
                            "real_time_factor": payload.get("real_time_factor", 0.0),
                        })
                        self.status.emit(dict(self._status_cache))
                    elif key == "comm":
                        pid = str(payload.get("platform_id", ""))
                        if pid:
                            row = dict(payload)
                            row["rx_age"] = row.get("last_rx_age", -1.0)
                            self._comm_cache[pid] = row
                        self.comm.emit({"platforms": list(self._comm_cache.values())})
                    else:
                        getattr(self, key).emit(payload)
                    return
        elif op == "service_response":
            cb = self._pop_pending(msg.get("id", ""))
            values = self._service_result(msg)
            if cb:
                self._finish(cb, values)
            level = "OK" if values["success"] else "FAIL"
            self.log.emit(level, "%s → %s" % (msg.get("service", ""), values.get("message", "")))
        elif op == "status":
            # Protocol failures may arrive without a service_response. Complete the
            # matching request instead of leaving the UI busy until a timeout.
            level = str(msg.get("level", "info")).lower()
            message = str(msg.get("msg", ""))
            self.log.emit({"error": "FAIL", "warning": "WARN"}.get(level, "INFO"), message)
            if level == "error":
                cb = self._pop_pending(msg.get("id", ""))
                if cb:
                    self._finish(cb, {"success": False, "message": message})

    @staticmethod
    def _service_result(msg: dict) -> dict:
        """rosbridge failures use a string in values, not a ROS response object."""
        raw = msg.get("values")
        if isinstance(raw, dict):
            values = dict(raw)  # never mutate the received frame
            values["success"] = (bool(values.get("success", msg.get("result", False)))
                                 and msg.get("result") is not False)
        elif raw is None:
            values = {"success": bool(msg.get("result", False))}
        else:
            values = {"success": False, "message": raw if isinstance(raw, str) else
                      "ROS 服务返回格式异常（%s）：%s" % (type(raw).__name__, raw)}
        if not values["success"]:
            message = str(values.get("message") or "ROS 服务调用失败，未返回错误详情")
            service = str(msg.get("service", ""))
            if service.startswith("/autoscene/") and any(
                    phrase in message.lower() for phrase in ("does not exist", "not available", "unknown service")):
                message += ("；rosbridge 已连接，但 AutoScene 服务 %s 不可用。"
                            "请在 ROS 主机启动 AutoScene experiment_manager，"
                            "并确认前端与它连接同一个 ROS master。" % service)
            values["message"] = message
        return values

    def call(self, key: str, payload=None, done=None) -> None:
        service = SERVICES.get(key)
        if not service:
            return
        self._seq += 1
        cid = "call_%d" % self._seq
        if done:
            self._register_pending(cid, done, service)
        if not self._send({"op": "call_service", "service": service,
                           "args": payload or {}, "id": cid}):
            self._pop_pending(cid)
            self._finish(done, {"success": False, "message": "rosbridge 未连接"})

    def set_params(self, params: Dict[str, Any], done=None) -> None:
        """通过 rosapi 顺序写入参数，全部成功后才允许调用 prepare。"""
        self._comm_cache.clear()
        rosparams = list((params.get("rosparams") or {}).items())
        if not rosparams:
            self._finish(done, {"success": True, "message": "没有待更新参数"})
            return

        def write(index: int) -> None:
            if index >= len(rosparams):
                self._finish(done, {"success": True,
                                    "message": "已更新 %d 个 ROS 参数" % len(rosparams)})
                return
            name, value = rosparams[index]
            self._seq += 1
            cid = "param_%d" % self._seq

            def after(result: dict) -> None:
                if not result.get("success"):
                    self._finish(done, {"success": False,
                                        "message": "写入 %s 失败：%s" % (name, result.get("message", "无错误详情"))})
                    return
                write(index + 1)

            self._register_pending(cid, after, "写入 ROS 参数 %s" % name)
            sent = self._send({"op": "call_service", "service": "/rosapi/set_param",
                               "args": {"name": name, "value": json.dumps(value)},
                               "id": cid})
            if not sent:
                self._pop_pending(cid)
                self._finish(done, {"success": False, "message": "rosbridge 未连接"})

        write(0)


# --------------------------------------------------------------------------
class RospyClient(BaseClient):
    """同机有 ROS1 时使用；22.04 上通常不可用。"""

    name = "rospy"

    def __init__(self) -> None:
        super().__init__()
        self._ok = False
        try:
            import rospy  # noqa: F401
            self._ok = True
        except Exception:  # noqa: BLE001
            self._ok = False
        self._subscribers = []

    def start(self) -> None:
        if not self._ok:
            self.connected.emit(False, "本机没有可用的 rospy（22.04 无 ROS1）")
            return
        try:
            import rospy
            from std_msgs.msg import String
            from autoscene_msgs.msg import (EvaluationMetrics, ExperimentStatus,
                                             PlatformCommStatus, ScenarioStatus,
                                             SimulationStatus, TrajectoryArray, VehicleState,
                                             VehicleStateArray, WorldState)
            if not rospy.core.is_initialized():
                rospy.init_node("autoscene_gui", anonymous=True, disable_signals=True)
            message_types = {
                "status": ExperimentStatus, "scenario": ScenarioStatus,
                "simulation": SimulationStatus, "world": WorldState,
                "sim_ego": VehicleState, "sim_agents": VehicleStateArray,
                "planning": TrajectoryArray, "platform_command": TrajectoryArray,
                "metrics": EvaluationMetrics, "comm": PlatformCommStatus,
                "ads_decoded": String,
            }
            self._status_cache = {}
            self._comm_cache = {}
            for key, (topic, _type_name) in TOPICS.items():
                self._subscribers.append(rospy.Subscriber(
                    topic, message_types[key], lambda msg, k=key: self._receive(k, msg),
                    queue_size=1))
            self.connected.emit(True, "rospy 本机连接")
        except Exception as exc:  # noqa: BLE001
            self.connected.emit(False, "rospy 初始化失败：%s" % exc)

    @staticmethod
    def _to_dict(msg: Any) -> Any:
        if hasattr(msg, "__slots__"):
            return {name: RospyClient._to_dict(getattr(msg, name)) for name in msg.__slots__}
        if isinstance(msg, (list, tuple)):
            return [RospyClient._to_dict(item) for item in msg]
        return msg

    def _receive(self, key: str, ros_msg: Any) -> None:
        payload = self._to_dict(ros_msg)
        if key == "status":
            self._status_cache.update(payload)
            self._status_cache["prepared"] = payload.get("state") in (
                "prepared", "starting", "running", "stopped")
            self.status.emit(dict(self._status_cache))
        elif key == "scenario":
            self._status_cache.update({"step": payload.get("step", 0)})
            self.status.emit(dict(self._status_cache))
        elif key == "simulation":
            self._status_cache.update({"sim_time": payload.get("sim_time", 0.0),
                                       "real_time_factor": payload.get("real_time_factor", 0.0)})
            self.status.emit(dict(self._status_cache))
        elif key == "comm":
            pid = str(payload.get("platform_id", ""))
            payload["rx_age"] = payload.get("last_rx_age", -1.0)
            self._comm_cache[pid] = payload
            self.comm.emit({"platforms": list(self._comm_cache.values())})
        else:
            getattr(self, key).emit(payload)

    def shutdown(self) -> None:
        for subscriber in self._subscribers:
            subscriber.unregister()
        self._subscribers = []

    def call(self, key: str, payload=None, done=None) -> None:
        def worker() -> None:
            try:
                import rospy
                from std_srvs.srv import Trigger
                from autoscene_msgs.srv import LoadExperiment, ResetRuntime
                service_types = {"load": LoadExperiment,
                                 "start": Trigger, "stop": Trigger, "reset": ResetRuntime}
                proxy = rospy.ServiceProxy(SERVICES[key], service_types[key])
                response = proxy(**(payload or {}))
                result = self._to_dict(response)
                self.log.emit("OK" if result.get("success") else "FAIL",
                              "%s → %s" % (SERVICES[key], result.get("message", "")))
                self._finish(done, result)
            except Exception as exc:  # noqa: BLE001
                self._finish(done, {"success": False, "message": str(exc)})
                self.log.emit("FAIL", "%s：%s" % (SERVICES.get(key, key), exc))
        threading.Thread(target=worker, daemon=True).start()

    def set_params(self, params: Dict[str, Any], done=None) -> None:
        try:
            import rospy
            for name, value in (params.get("rosparams") or {}).items():
                rospy.set_param(name, value)
            self._finish(done, {"success": True, "message": "ROS 参数已更新"})
        except Exception as exc:  # noqa: BLE001
            self._finish(done, {"success": False, "message": str(exc)})


def make_client(kind: str = "rosbridge", url: str = "ws://127.0.0.1:9090") -> BaseClient:
    if kind == "rosbridge":
        return RosbridgeClient(url)
    if kind == "rospy":
        return RospyClient()
    return MockClient()