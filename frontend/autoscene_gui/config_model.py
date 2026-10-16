"""AutoScene 实验配置模型：读写 bringup 的 hybrid.yaml / simulation.yaml，并做启动前校验。

字段与约束全部对齐 experiment_manager_node.py 的行为：
  * launch 的 expected_mode 必须与 runtime.mode 一致
  * hybrid 只允许 scenario.dataset_type=tjwest 且 history_source=initial
  * history_source=initial 时，初始速度上限
        v_max = initial_history_accel_mps2 * pluto_history_dt_sec * (pluto_history_steps - 1)
  * platform BV 数量不得超过 platform.platforms 的条目数
"""

from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("需要 PyYAML：pip install pyyaml") from exc


DEFAULT_CONFIG: Dict[str, Any] = {
    "project": {"experiment_id": "exp_hybrid_001"},
    "runtime": {"mode": "hybrid", "world_state_hz": 10, "frame_id": "map"},
    "scenario": {
        "source": "unitraj_pluto",
        "dataset_type": "tjwest",
        "dataset_path": "",
        "history_source": "initial",
        "scenario_index": 0,
        "eval_eps": 1,
        "auto_advance_scenario": False,
        "agent_assignment_mode": "manual",
        "manual_platform_bv_count": 1,
        "manual_virtual_bv_count": 1,
        "initial_history_accel_mps2": 2.0,
        "pluto_history_dt_sec": 0.1,
        "pluto_history_steps": 21,
    },
    "ego": {
        "id": "ego_001",
        "execution_mode": "simulation",
        "initial": {"x": 0.0, "y": 0.0, "heading": 0.0, "speed": 0.0},
    },
    "platform": {
        "receive_udp": False,
        "send_udp": False,
        "emulate_controlled_feedback": True,
        "strict_actual_feedback": True,
        "local_ip": "192.168.88.3",
        "local_port": 31100,
        "send_rate_hz": 10,
        "packet_point_num": 50,
        "trajectory_dt_sec": 0.1,
        "earth_to_autoscene": {"tx": 0.0, "ty": 0.0, "yaw": 0.0},
        "platforms": [
            {"platform_id": "platform_001", "vehicle_id": "agent_001",
             "ip": "192.168.88.100", "send_port": 31000,
             "expected_remote_ip": "192.168.88.100", "expected_remote_port": -1},
            {"platform_id": "platform_002", "vehicle_id": "agent_002",
             "ip": "192.168.88.101", "send_port": 31000,
             "expected_remote_ip": "192.168.88.101", "expected_remote_port": -1},
        ],
    },
    "algorithm": {
        "initial_history_accel_mps2": 2.0,
        "pluto_history_dt_sec": 0.1,
        "pluto_history_steps": 21,
    },
    "vehicles": {"agents": []},
    "evaluation": {
        "include_background_vehicles": True,
        "rate_hz": 10,
        "auto_export": True,
        "output_dir": "./outputs",
    },
    "record": {"enable_rosbag": True, "output_dir": "./outputs"},
}


@dataclass
class AgentSpec:
    """一台 BV。platform_id 为空表示纯虚拟车。"""

    agent_id: str
    role: str  # "platform" | "virtual"
    platform_id: Optional[str] = None
    scenario_track_id: Optional[str] = None
    planner_mode: str = "max"
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    speed: float = 0.0

    def to_yaml(self) -> Dict[str, Any]:
        node: Dict[str, Any] = {
            "id": self.agent_id,
            "platform_id": self.platform_id if self.platform_id else -1,
            "planner_mode": self.planner_mode,
            "initial": {
                "x": round(self.x, 4),
                "y": round(self.y, 4),
                "heading": round(self.heading, 6),
                "speed": round(self.speed, 4),
            },
        }
        if self.scenario_track_id is not None:
            node["scenario_track_id"] = str(self.scenario_track_id)
        return node


@dataclass
class Issue:
    level: str  # "error" | "warn"
    text: str


@dataclass
class ExperimentConfig:
    raw: Dict[str, Any] = field(default_factory=lambda: copy.deepcopy(DEFAULT_CONFIG))
    path: Optional[str] = None

    # ---- 载入 / 保存 -------------------------------------------------
    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        merged = copy.deepcopy(DEFAULT_CONFIG)
        _deep_update(merged, data)
        # 兼容早期 UI 写出的 endpoints/port，但以 ROS 节点实际读取的
        # platform.platforms/send_port 为准。
        source_platform = data.get("platform", {}) if isinstance(data, dict) else {}
        if "platforms" not in source_platform and "endpoints" in source_platform:
            merged["platform"]["platforms"] = copy.deepcopy(source_platform["endpoints"])
        for endpoint in merged["platform"].get("platforms", []):
            if "send_port" not in endpoint and "port" in endpoint:
                endpoint["send_port"] = endpoint["port"]
        return cls(raw=merged, path=path)

    def save(self, path: Optional[str] = None) -> str:
        target = path or self.path
        if not target:
            raise ValueError("未指定保存路径")
        os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
        with open(target, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.raw, fh, allow_unicode=True, sort_keys=False, default_flow_style=False)
        self.path = target
        return target

    def dump(self) -> str:
        return yaml.safe_dump(self.raw, allow_unicode=True, sort_keys=False, default_flow_style=False)

    # ---- 便捷访问 ----------------------------------------------------
    @property
    def mode(self) -> str:
        return self.raw["runtime"]["mode"]

    @mode.setter
    def mode(self, value: str) -> None:
        self.raw["runtime"]["mode"] = value
        self.raw["project"]["experiment_id"] = "exp_hybrid_001" if value == "hybrid" else "exp_sim_001"
        if value == "hybrid":
            # manager 会强制这两项，UI 先行对齐避免启动被拒
            self.raw["scenario"]["dataset_type"] = "tjwest"
            self.raw["scenario"]["history_source"] = "initial"

    @property
    def scenario(self) -> Dict[str, Any]:
        return self.raw["scenario"]

    @property
    def platform(self) -> Dict[str, Any]:
        return self.raw["platform"]

    @property
    def endpoints(self) -> List[Dict[str, Any]]:
        return list(self.platform.get("platforms") or self.platform.get("endpoints") or [])

    def endpoint(self, platform_id: str) -> Optional[Dict[str, Any]]:
        return next((e for e in self.endpoints
                     if str(e.get("platform_id")) == str(platform_id)), None)

    def endpoint_ids(self) -> List[str]:
        return [str(e.get("platform_id")) for e in self.endpoints]

    def endpoint_window(self, platform_id: str) -> str:
        for e in self.endpoints:
            if str(e.get("platform_id")) == str(platform_id):
                return "%s:%s ← %s:%s" % (
                    e.get("ip"), e.get("send_port", e.get("port")),
                    self.platform.get("local_ip"), self.platform.get("local_port"),
                )
        return "—"

    @property
    def max_initial_speed(self) -> float:
        algorithm = self.raw.get("algorithm", {})
        sc = self.scenario  # fallback for YAML files produced by the early UI
        return float(algorithm.get("initial_history_accel_mps2", sc.get("initial_history_accel_mps2", 2.0))) * float(
            algorithm.get("pluto_history_dt_sec", sc.get("pluto_history_dt_sec", 0.1))) * (
            int(algorithm.get("pluto_history_steps", sc.get("pluto_history_steps", 21))) - 1
        )

    @property
    def hardware_live(self) -> bool:
        """是否真正收发 UDP（非 dry-run）。"""
        p = self.platform
        return (bool(p.get("receive_udp")) and bool(p.get("send_udp"))
                and not bool(p.get("emulate_controlled_feedback")))

    # ---- 写入 BV 列表 ------------------------------------------------
    def apply_agents(self, agents: List[AgentSpec], ego: Optional[AgentSpec] = None) -> None:
        ordered = [a for a in agents if a.role == "platform"] + [a for a in agents if a.role == "virtual"]
        # 当前 scenario_runtime 对 Pluto / pluto_hybrid 只接受 planner_mode=max。
        # UI 只负责选择具体 BV，不再沿用旧版“第一辆 adversarial”的语义。
        for a in ordered:
            a.planner_mode = "max"
        self.raw["vehicles"]["agents"] = [a.to_yaml() for a in ordered]
        self.scenario["manual_platform_bv_count"] = sum(1 for a in ordered if a.role == "platform")
        self.scenario["manual_virtual_bv_count"] = sum(1 for a in ordered if a.role == "virtual")
        # Global planner selection lives only in algorithm.ego_method / adv_method.
        # Remove legacy per-vehicle planner_method fields whenever UI materializes the
        # current experiment configuration, so the backend has one unambiguous source.
        self.raw["ego"].pop("planner_method", None)
        if ego is not None:
            self.raw["ego"]["initial"] = {
                "x": round(ego.x, 4), "y": round(ego.y, 4),
                "heading": round(ego.heading, 6), "speed": round(ego.speed, 4),
            }

    # ---- 校验 --------------------------------------------------------
    def validate(self, agents: List[AgentSpec], ego: Optional[AgentSpec], scenario_count: int) -> List[Issue]:
        out: List[Issue] = []
        sc = self.scenario
        mode = self.mode

        algorithm = self.raw.get("algorithm", {})
        allowed_methods = ("Pluto", "VBD", "pluto_hybrid")
        for key, label in (("ego_method", "Ego"), ("adv_method", "BV")):
            method = algorithm.get(key)
            if method in (None, ""):
                continue
            if method not in allowed_methods:
                out.append(Issue("error", "%s 算法不支持：%s" % (label, method)))

        if mode not in ("simulation", "hybrid"):
            out.append(Issue("error", "runtime.mode 只能是 simulation 或 hybrid"))
        if mode == "hybrid":
            if sc["dataset_type"] != "tjwest":
                out.append(Issue("error", "hybrid 模式只支持 dataset_type=tjwest"))
            if sc["history_source"] != "initial":
                out.append(Issue("error", "hybrid 模式要求 history_source=initial"))

        path = str(sc.get("dataset_path") or "")
        if not path or not os.path.isdir(path):
            out.append(Issue("error", "dataset_path 不存在：%s" % (path or "(空)")))

        idx = int(sc.get("scenario_index") or 0)
        if scenario_count and not (0 <= idx < scenario_count):
            out.append(Issue("error", "scenario_index=%d 超出范围 0..%d" % (idx, scenario_count - 1)))

        platform_agents = [a for a in agents if a.role == "platform"]
        virtual_agents = [a for a in agents if a.role == "virtual"]
        if mode == "hybrid":
            if not platform_agents:
                out.append(Issue("error", "hybrid 至少需要 1 台实物平台车绑定端点"))
            if len(platform_agents) > len(self.endpoints):
                out.append(Issue("error", "platform BV 数量 %d 超过已配置端点数 %d"
                                 % (len(platform_agents), len(self.endpoints))))
            used = [a.platform_id for a in platform_agents if a.platform_id]
            if len(set(used)) != len(used):
                out.append(Issue("error", "同一个运动平台端点被绑定给了多台车"))
            known = set(self.endpoint_ids())
            for platform_id in used:
                if platform_id not in known:
                    out.append(Issue("error", "找不到运动平台端点：%s" % platform_id))
        else:
            if platform_agents:
                out.append(Issue("error", "simulation 模式不应存在 platform BV"))
            if not virtual_agents:
                out.append(Issue("error", "simulation 至少需要 1 台 virtual BV"))

        if sc["history_source"] == "initial":
            vmax = self.max_initial_speed
            for a in agents + ([ego] if ego else []):
                if a is None:
                    continue
                speed = float(a.speed)
                if not math.isfinite(speed):
                    out.append(Issue("error", "%s 初始速度必须为有限数值" % a.agent_id))
                elif speed < 0.0:
                    out.append(Issue("error", "%s 初始速度 %.3f 不能小于 0 m/s"
                                     % (a.agent_id, speed)))
                elif speed > vmax + 1e-9:
                    out.append(Issue("error", "%s 初始速度 %.3f 超过上限 %.3f m/s"
                                     % (a.agent_id, speed, vmax)))

            # Also validate raw YAML values so malformed configured speeds cannot be
            # silently replaced by dataset speeds before prepare().
            raw_initials = [(str(self.raw.get("ego", {}).get("id", "ego_001")),
                             self.raw.get("ego", {}).get("initial", {}))]
            for index, node in enumerate(self.raw.get("vehicles", {}).get("agents", [])):
                if not isinstance(node, dict):
                    out.append(Issue("error", "vehicles.agents[%d] 必须为映射" % index))
                    continue
                raw_initials.append((str(node.get("id", "agent_%d" % index)),
                                     node.get("initial", {})))
            for label, initial in raw_initials:
                if not isinstance(initial, dict) or "speed" not in initial:
                    continue
                try:
                    raw_speed = float(initial["speed"])
                except (TypeError, ValueError):
                    out.append(Issue("error", "%s initial.speed 不是数值" % label))
                    continue
                if not math.isfinite(raw_speed):
                    out.append(Issue("error", "%s initial.speed 必须为有限数值" % label))
                elif raw_speed < 0.0:
                    out.append(Issue("error", "%s initial.speed %.3f 不能小于 0 m/s"
                                     % (label, raw_speed)))
                elif raw_speed > vmax + 1e-9:
                    out.append(Issue("error", "%s initial.speed %.3f 超过上限 %.3f m/s"
                                     % (label, raw_speed, vmax)))

        p = self.platform
        if bool(p.get("receive_udp")) and bool(p.get("emulate_controlled_feedback")):
            out.append(Issue("error", "receive_udp 与 emulate_controlled_feedback 不能同时为 true"))
        if bool(p.get("send_udp")) and not bool(p.get("receive_udp")):
            out.append(Issue("error", "实车模式下 send_udp=true 要求 receive_udp=true"))
        if self.hardware_live and not bool(p.get("strict_actual_feedback", True)):
            out.append(Issue("error", "实车模式要求 strict_actual_feedback=true"))
        if mode == "hybrid" and not self.hardware_live:
            out.append(Issue("warn", "platform 为 dry-run，不会向实车发包"))
        if mode == "hybrid" and self.hardware_live:
            transform = p.get("earth_to_autoscene")
            if not isinstance(transform, dict) or any(k not in transform for k in ("tx", "ty", "yaw")):
                out.append(Issue("error", "实车模式缺少 platform.earth_to_autoscene 的 tx/ty/yaw 标定"))
            for e in self.endpoints:
                for key in ("vehicle_id", "ip", "send_port", "expected_remote_ip", "expected_remote_port"):
                    if key not in e:
                        out.append(Issue("error", "端点 %s 缺少 %s" % (e.get("platform_id"), key)))
        return out


def _deep_update(base: Dict[str, Any], patch: Dict[str, Any]) -> None:
    for key, val in (patch or {}).items():
        if isinstance(val, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], val)
        else:
            base[key] = val
