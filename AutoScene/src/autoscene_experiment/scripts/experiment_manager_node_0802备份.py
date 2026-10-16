#!/usr/bin/env python3
"""AutoScene 常驻实验管理节点。

职责：
1. 常驻 ROS master 中，提供 start/stop/load/reset 服务和交互式终端菜单；
2. 在实验启动前选择单个场景，并按候选 rank/track_id 手动绑定 platform/virtual BV；
3. 为每次实验新建一个工作终端，按模式启动 runtime、必要的适配节点和 evaluator；
4. 监听 scenario/status，在单次实验结束后保留 manager，继续配置下一次实验；
5. 按配置启动/停止 rosbag，并周期性发布实验状态。

该节点不直接推进 MetaDrive，也不执行 Pluto 控制；它只管理实验配置和工作节点生命周期。
"""
import copy
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time

import rospy
from std_srvs.srv import Trigger, TriggerResponse

from autoscene_msgs.msg import ExperimentStatus, ScenarioStatus, SystemControl
from autoscene_msgs.srv import LoadExperiment, LoadExperimentResponse, ResetRuntime, ResetRuntimeResponse


class ExperimentManagerNode:
    def __init__(self):
        rospy.init_node("experiment_manager_node")

        # project/runtime/record 来自 bringup YAML，可通过 experiment/load 动态覆盖。
        self.project = rospy.get_param("/project", {})
        self.runtime = rospy.get_param("/runtime", {})
        self.record = rospy.get_param("/record", {})
        self.experiment_id = self.project.get("experiment_id", "exp_001")
        self.mode = str(self.runtime.get("mode", "simulation")).strip().lower()
        self.expected_mode = str(
            rospy.get_param("~expected_mode", "")
        ).strip().lower()
        if self.expected_mode and self.mode != self.expected_mode:
            raise ValueError(
                "launch/config mode mismatch: launch expects %s, config runtime.mode=%s"
                % (self.expected_mode, self.mode)
            )

        self.state = "idle"
        self.message = "ready"
        self.start_time = None
        self.config_path = rospy.get_param("~config_path", "")

        self.manage_nodes = bool(rospy.get_param("~manage_nodes", True))
        self.auto_start_after_prepare = bool(
            rospy.get_param("~auto_start_after_prepare", False)
        )
        self.interactive_console = bool(
            rospy.get_param("~interactive_console", True)
        )
        self.launch_in_terminal = bool(
            rospy.get_param("~launch_in_terminal", True)
        )
        self.worker_launch = str(
            rospy.get_param(
                "~worker_launch",
                "autoscene_experiment_nodes.launch",
            )
        )
        self.terminal_emulator = str(
            rospy.get_param("~terminal_emulator", "auto")
        ).strip()

        self.rosbag_proc = None

        # 每次实验只启动一个 worker roslaunch。
        # node_procs 保留列表形式，便于沿用统一的 poll/stop 逻辑。
        self.node_procs = []
        self.prepared = False
        self.runtime_reported_terminal = False
        self.lifecycle_lock = threading.RLock()
        self.console_thread = None

        self.status_pub = rospy.Publisher(
            "experiment/status",
            ExperimentStatus,
            queue_size=1,
            latch=True,
        )
        rospy.Subscriber(
            "system/control",
            SystemControl,
            self.on_control,
            queue_size=10,
        )
        rospy.Subscriber(
            "scenario/status",
            ScenarioStatus,
            self.on_scenario_status,
            queue_size=1,
        )
        rospy.Service("experiment/start", Trigger, self.on_start)
        rospy.Service("experiment/stop", Trigger, self.on_stop)
        rospy.Service("experiment/load", LoadExperiment, self.on_load)
        rospy.Service("experiment/reset", ResetRuntime, self.on_reset)
        rospy.Timer(rospy.Duration(0.5), self.on_timer)
        rospy.on_shutdown(self.shutdown_children)

        rospy.loginfo(
            "AutoScene experiment manager started, experiment=%s, interactive=%s",
            self.experiment_id,
            self.interactive_console,
        )

        if self.manage_nodes and self.auto_start_after_prepare:
            try:
                self.start("startup")
            except Exception as exc:
                self.state = "error"
                self.message = "startup failed: %s" % exc
                rospy.logerr("%s", self.message)
                self.publish_status()

        # 交互菜单不能在后台线程启动。MetaDrive/Panda3D 初始化过程中
        # 会注册 signal handler，而 Python 只允许主线程调用 signal.signal()。
        # 菜单由 spin() 在主线程运行；rospy subscriber/service/timer 回调
        # 仍由 rospy 的内部线程处理。

    def on_scenario_status(self, msg):
        """Handle the terminal runtime state without stopping the manager."""
        message = str(msg.message).strip()
        terminal = bool(msg.loaded and not msg.running)
        finished = terminal and message.lower() == "finished"
        failed = terminal and message.lower().startswith("failed:")

        if not (finished or failed):
            return

        with self.lifecycle_lock:
            self.runtime_reported_terminal = True
            self.state = "finished" if finished else "error"
            self.message = message
            self.stop_rosbag()
            if failed:
                rospy.logerr("scenario runtime failed: %s", message)
            else:
                rospy.loginfo("scenario finished")
            self.publish_status()

    def on_control(self, msg):
        """处理 /system/control 话题命令，便于脚本或上位机控制实验。"""
        command = msg.command.strip().lower()
        if command == "start":
            self.start("control topic: %s" % msg.reason)
        elif command == "stop":
            self.stop("control topic: %s" % msg.reason)
        elif command == "reset":
            self.reset(False)
        else:
            self.message = "unknown command: %s" % msg.command

    def on_start(self, _req):
        ok, message = self.start("service")
        return TriggerResponse(ok, message)

    def on_stop(self, _req):
        ok, message = self.stop("service")
        return TriggerResponse(ok, message)

    def on_load(self, req):
        """加载新的实验 YAML，并写入 ROS 参数服务器。

        注意：这里只更新参数，不重启各节点。需要节点读取新参数时，通常还要 reset
        或重新 roslaunch，后续可扩展成真正的 runtime 热重载。
        """
        try:
            import yaml
        except Exception as exc:
            return LoadExperimentResponse(False, "PyYAML is required to load config: %s" % exc)
        if not req.config_path:
            return LoadExperimentResponse(False, "config_path is empty")
        path = req.config_path if os.path.isabs(req.config_path) else os.path.abspath(req.config_path)
        if not os.path.exists(path):
            return LoadExperimentResponse(False, "config not found: %s" % path)
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        loaded_mode = str(
            data.get("runtime", {}).get("mode", self.mode)
        ).strip().lower()
        if self.expected_mode and loaded_mode != self.expected_mode:
            return LoadExperimentResponse(
                False,
                "launch/config mode mismatch: launch expects %s, config runtime.mode=%s"
                % (self.expected_mode, loaded_mode),
            )
        for key, value in data.items():
            rospy.set_param("/" + key, value)
        self.project = rospy.get_param("/project", {})
        self.runtime = rospy.get_param("/runtime", {})
        self.record = rospy.get_param("/record", {})
        self.experiment_id = self.project.get("experiment_id", self.experiment_id)
        self.mode = str(self.runtime.get("mode", self.mode)).strip().lower()
        self.config_path = path
        self.prepared = False
        self.runtime_reported_terminal = False
        self.message = "loaded config: %s" % path
        return LoadExperimentResponse(True, self.message)

    def on_reset(self, req):
        ok, message = self.reset(req.hard)
        return ResetRuntimeResponse(ok, message)

    def start(self, reason):
        """启动一次实验；manager 本身始终保持运行。"""
        with self.lifecycle_lock:
            self.reap_worker_processes()

            if self.state == "running":
                return True, "experiment already running"

            if self.node_procs:
                return (
                    False,
                    "previous worker terminal is still open; close it before starting a new experiment",
                )

        if self.manage_nodes and not self.prepared:
            if self.interactive_console:
                return (
                    False,
                    "experiment is not prepared; "
                    "use menu 1 then menu 2 before start",
                )

            # 非交互自动化模式仍允许自动准备。
            try:
                self.prepare_experiment_configuration()
            except Exception as exc:
                with self.lifecycle_lock:
                    self.state = "error"
                    self.message = "prepare failed: %s" % exc
                    self.publish_status()
                return False, self.message

        with self.lifecycle_lock:
            self.runtime_reported_terminal = False
            self.state = "starting"
            self.start_time = time.time()
            self.message = "starting by %s" % reason
            self.publish_status()

        try:
            self.start_rosbag()
            if self.manage_nodes:
                self.start_managed_nodes()
        except Exception as exc:
            self.stop_rosbag()
            self.stop_managed_nodes()
            with self.lifecycle_lock:
                self.state = "error"
                self.message = "start managed nodes failed: %s" % exc
                self.publish_status()
            return False, self.message

        with self.lifecycle_lock:
            self.state = "running"
            self.message = "started by %s" % reason
            self.publish_status()
            return True, self.message

    def stop(self, reason):
        """停止当前实验的工作节点；manager 继续等待下一次配置。"""
        self.stop_rosbag()

        if self.manage_nodes:
            self.stop_managed_nodes()

        with self.lifecycle_lock:
            self.state = "stopped"
            self.start_time = None
            self.runtime_reported_terminal = False
            self.message = "stopped by %s" % reason
            self.publish_status()
            return True, self.message

    def reset(self, hard):
        """清理当前实验。

soft reset：
- 停止 rosbag 和工作节点；
- 清除场景/BV 准备标志；
- 返回 idle。

hard reset：
- 交互模式下返回 idle，要求重新选择场景和 BV；
- 非交互模式下执行自动准备并重新启动。
"""
        self.stop_rosbag()

        if self.manage_nodes:
            self.stop_managed_nodes()

        with self.lifecycle_lock:
            self.state = "idle"
            self.start_time = None
            self.prepared = False
            self.runtime_reported_terminal = False

        if hard:
            if self.interactive_console:
                # 交互模式不能在 service/timer 回调中与 console 线程争抢终端输入。
                with self.lifecycle_lock:
                    self.message = (
                        "hard reset completed; "
                        "use menu 1 then menu 2 before start"
                    )
                    self.publish_status()
                    return True, self.message

            # 非交互自动化模式仍可自动准备并重新启动。
            return self.start("hard reset")

        with self.lifecycle_lock:
            self.message = "soft reset; select scenario and BV before start"
            self.publish_status()
            return True, self.message

    def prepare_experiment_configuration(self):
        """启动 runtime 前准备实验配置。

        manual 模式会打开当前 MetaDrive 场景、打印排序后的候选 CBV，并在终端
        为配置中的每个 agent 选择 rank/track_id。选择结果写入 ROS 参数服务器，
        runtime 启动后直接读取，不再被自动选车覆盖。
        """
        if self.prepared:
            return

        scenario = rospy.get_param("/scenario", {})
        history_source = str(
            scenario.get("history_source", "dataset")
        ).strip().lower()
        if history_source not in ("dataset", "initial"):
            raise ValueError(
                "scenario.history_source must be dataset or initial, got %s"
                % history_source
            )
        self.validate_mode_dataset_history()

        mode = str(
            scenario.get("agent_assignment_mode", "auto")
        ).strip().lower()

        if mode not in ("auto", "manual"):
            raise ValueError(
                "scenario.agent_assignment_mode must be auto or manual, got %s"
                % mode
            )

        if mode == "manual":
            self.manual_assign_agents()
            scenario = rospy.get_param("/scenario", {})
            scenario["auto_assign_agents"] = False
            rospy.set_param("/scenario", scenario)
        else:
            scenario["auto_assign_agents"] = bool(
                scenario.get("auto_assign_agents", True)
            )
            rospy.set_param("/scenario", scenario)

        self.prepared = True

    def prompt_integer(self, prompt, minimum=0, maximum=None, default=None):
        """从终端读取整数，并执行上下界校验；空输入可沿用默认值。"""
        while not rospy.is_shutdown():
            suffix = ""
            if default is not None:
                suffix = " [%s]" % default

            value = self.read_terminal_line(prompt + suffix + ": ").strip()

            if not value and default is not None:
                number = int(default)
            else:
                try:
                    number = int(value)
                except (TypeError, ValueError):
                    print("please input an integer")
                    continue

            if number < int(minimum):
                print("value must be >= %d" % int(minimum))
                continue

            if maximum is not None and number > int(maximum):
                print("value must be <= %d" % int(maximum))
                continue

            return number

        raise RuntimeError("ROS shutdown while waiting for integer input")

    def prompt_choice(self, prompt, choices, default=None):
        """读取枚举值；空输入使用默认值。"""
        choices = [str(item).strip().lower() for item in choices]
        default = str(default).strip().lower() if default is not None else None
        while not rospy.is_shutdown():
            suffix = " (%s)" % "/".join(choices)
            if default:
                suffix += " [%s]" % default
            value = self.read_terminal_line(prompt + suffix + ": ").strip().lower()
            if not value and default:
                value = default
            if value in choices:
                return value
            print("value must be one of: %s" % ", ".join(choices))
        raise RuntimeError("ROS shutdown while waiting for choice input")

    def prompt_float(self, prompt, default=None):
        """读取浮点数；空输入使用默认值。"""
        while not rospy.is_shutdown():
            suffix = " [%.6f]" % float(default) if default is not None else ""
            value = self.read_terminal_line(prompt + suffix + ": ").strip()
            if not value and default is not None:
                return float(default)
            try:
                return float(value)
            except (TypeError, ValueError):
                print("please input a number")
        raise RuntimeError("ROS shutdown while waiting for float input")
    def initial_history_max_speed(self):
        """与 runtime 使用同一固定历史窗口速度上限。"""
        algorithm = rospy.get_param("/algorithm", {})
        steps = int(algorithm.get("pluto_history_steps", 21))
        dt = float(algorithm.get("pluto_history_dt_sec", 0.1))
        accel = float(algorithm.get("initial_history_accel_mps2", 2.0))
        if steps < 2 or dt <= 0.0 or accel <= 0.0:
            raise RuntimeError(
                "invalid initial history settings: steps=%s dt=%s accel=%s"
                % (steps, dt, accel)
            )
        return accel * dt * float(steps - 1)


    def validate_mode_dataset_history(self):
        """校验 manager 本轮写入参数的组合约束。"""
        runtime = rospy.get_param("/runtime", {})
        scenario = rospy.get_param("/scenario", {})
        mode = str(runtime.get("mode", self.mode)).strip().lower()
        dataset_type = str(scenario.get("dataset_type", "tjwest")).strip().lower()
        history_source = str(scenario.get("history_source", "dataset")).strip().lower()
        dataset_path = str(scenario.get("dataset_path", "")).strip()

        if mode not in ("simulation", "hybrid"):
            raise RuntimeError("runtime.mode must be simulation or hybrid")
        if dataset_type not in ("nuplan", "tjwest"):
            raise RuntimeError("scenario.dataset_type must be nuplan or tjwest")
        if history_source not in ("dataset", "initial"):
            raise RuntimeError("scenario.history_source must be dataset or initial")
        if mode == "hybrid" and dataset_type != "tjwest":
            raise RuntimeError("hybrid mode only supports tjwest dataset")
        if mode == "hybrid" and history_source != "initial":
            raise RuntimeError("hybrid mode requires history_source=initial")
        if history_source == "dataset" and mode != "simulation":
            raise RuntimeError("history_source=dataset is only allowed in simulation mode")
        if not dataset_path:
            raise RuntimeError("scenario.dataset_path is required")

    def configure_runtime_dataset_from_terminal(self):
        """把未来 UI 中的模式/数据集/历史来源选择先集成到终端菜单。"""
        runtime = rospy.get_param("/runtime", {})
        scenario = rospy.get_param("/scenario", {})

        mode_default = str(runtime.get("mode", self.mode)).strip().lower()
        mode = self.prompt_choice("Select runtime mode", ("simulation", "hybrid"), default=mode_default)
        if self.expected_mode and mode != self.expected_mode:
            raise RuntimeError(
                "this launch expects mode=%s, got %s" % (self.expected_mode, mode)
            )

        dataset_default = str(scenario.get("dataset_type", "tjwest")).strip().lower()
        dataset_type = self.prompt_choice("Select dataset type", ("nuplan", "tjwest"), default=dataset_default)
        if mode == "hybrid" and dataset_type != "tjwest":
            raise RuntimeError("hybrid mode only supports tjwest dataset")

        history_default = str(scenario.get("history_source", "initial" if mode == "hybrid" else "dataset")).strip().lower()
        history_source = self.prompt_choice("Select Pluto history source", ("dataset", "initial"), default=history_default)
        if history_source == "dataset" and mode != "simulation":
            raise RuntimeError("history_source=dataset is only allowed in simulation mode")
        if mode == "hybrid" and history_source != "initial":
            raise RuntimeError("hybrid mode requires history_source=initial")

        default_path = str(scenario.get("dataset_path", "")).strip()
        dataset_path = self.read_terminal_line(
            "Input dataset path [%s]: " % default_path
        ).strip() or default_path
        if not dataset_path:
            raise RuntimeError("scenario.dataset_path is required")

        runtime["mode"] = mode
        scenario["dataset_type"] = dataset_type
        scenario["dataset_path"] = dataset_path
        scenario["history_source"] = history_source
        rospy.set_param("/runtime", runtime)
        rospy.set_param("/scenario", scenario)
        self.mode = mode
        self.validate_mode_dataset_history()

    def select_platform_endpoints_from_terminal(self, endpoints, count):
        """人工选择本轮实际使用的运动平台 endpoint。

        可输入 endpoint rank、platform_id 或 vehicle_id；
        同一个 endpoint 不能重复选择。
        """
        count = int(count)

        if count < 1:
            return []

        if count > len(endpoints):
            raise RuntimeError(
                "requested %d platform endpoints, but only %d are configured"
                % (count, len(endpoints))
            )

        selected = []
        used_indexes = set()

        for slot in range(count):
            while not rospy.is_shutdown():
                value = self.read_terminal_line(
                    "Select platform endpoint %d/%d "
                    "(rank/platform_id/vehicle_id): "
                    % (slot + 1, count)
                ).strip()

                if not value:
                    print("empty input is not valid")
                    continue

                matches = []

                # 输入数字时按列表 rank 查找。
                if value.isdigit():
                    index = int(value) - 1

                    if 0 <= index < len(endpoints):
                        matches = [index]

                # 输入字符串时匹配 platform_id 或 vehicle_id。
                else:
                    for index, endpoint in enumerate(endpoints):
                        if (
                                value
                                == str(
                            endpoint.get(
                                "platform_id",
                                "",
                            )
                        ).strip()
                                or value
                                == str(
                            endpoint.get(
                                "vehicle_id",
                                "",
                            )
                        ).strip()
                        ):
                            matches.append(index)

                if len(matches) == 0:
                    print(
                        "cannot find platform endpoint: %s"
                        % value
                    )
                    continue

                if len(matches) > 1:
                    print(
                        "endpoint identifier is ambiguous: %s; "
                        "input the numeric rank"
                        % value
                    )
                    continue

                index = matches[0]

                if index in used_indexes:
                    print(
                        "platform endpoint already selected: %s"
                        % value
                    )
                    continue

                endpoint = endpoints[index]
                selected.append(endpoint)
                used_indexes.add(index)

                print(
                    "Selected endpoint: rank=%d "
                    "platform_id=%s vehicle_id=%s "
                    "ip=%s send_port=%s"
                    % (
                        index + 1,
                        endpoint.get("platform_id"),
                        endpoint.get("vehicle_id"),
                        endpoint.get("ip", ""),
                        endpoint.get("send_port", ""),
                    )
                )

                break

        return selected

    @staticmethod
    def clear_agent_assignment(agent):
        """删除上一轮场景绑定，保留车辆尺寸等模板字段。"""
        for key in (
                "scenario_track_id",
                "metadrive_object_id",
                "track_id",
                "initial",
        ):
            agent.pop(key, None)
        return agent

    @staticmethod
    def next_free_agent_id(used_ids):
        """生成不与 platform endpoint vehicle_id 冲突的 agent_NNN。"""
        index = 1
        while True:
            vehicle_id = "agent_%03d" % index
            if vehicle_id not in used_ids:
                return vehicle_id
            index += 1

    def build_manual_agent_configs(
            self,
            platform_count,
            virtual_count,
            platform_endpoints,
    ):
        """按本轮输入数量动态生成 /vehicles/agents。

        hybrid 的 platform agent 必须复用 endpoint.vehicle_id/platform_id，保证
        platform_comm_node 能找到对应通信端点；virtual agent 使用未占用的
        agent_NNN 逻辑 ID。旧 YAML agent 只作为长度、宽度等字段的模板。
        """
        vehicles = rospy.get_param("/vehicles", {})
        old_agents = vehicles.get("agents", [])

        platform_templates = [
            agent
            for agent in old_agents
            if self.valid_platform_id(agent.get("platform_id"))
        ]
        virtual_templates = [
            agent
            for agent in old_agents
            if not self.valid_platform_id(agent.get("platform_id"))
        ]

        agents = []
        used_ids = set()
        used_platform_ids = set()

        if len(platform_endpoints) != int(platform_count):
            raise RuntimeError(
                "selected platform endpoint count mismatch: "
                "expected=%d actual=%d"
                % (
                    int(platform_count),
                    len(platform_endpoints),
                )
            )

        for endpoint_index, endpoint in enumerate(
                platform_endpoints
        ):
            platform_id = str(endpoint.get("platform_id", "")).strip()
            vehicle_id = str(endpoint.get("vehicle_id", "")).strip()

            if not platform_id or not vehicle_id:
                raise RuntimeError(
                    "platform endpoint %d requires non-empty platform_id and vehicle_id"
                    % (endpoint_index + 1)
                )

            if vehicle_id in used_ids:
                raise RuntimeError(
                    "duplicate platform endpoint vehicle_id: %s" % vehicle_id
                )
            if platform_id in used_platform_ids:
                raise RuntimeError(
                    "duplicate selected platform_id: %s" % platform_id
                )

            template = next(
                (
                    agent
                    for agent in platform_templates
                    if str(agent.get("id", "")) == vehicle_id
                    or str(agent.get("platform_id", "")) == platform_id
                ),
                platform_templates[0] if platform_templates else {},
            )
            agent = self.clear_agent_assignment(copy.deepcopy(template))
            agent["id"] = vehicle_id
            agent["platform_id"] = platform_id
            agents.append(agent)
            used_ids.add(vehicle_id)
            used_platform_ids.add(platform_id)

        for virtual_index in range(virtual_count):
            template = (
                virtual_templates[min(virtual_index, len(virtual_templates) - 1)]
                if virtual_templates
                else {}
            )
            agent = self.clear_agent_assignment(copy.deepcopy(template))
            vehicle_id = self.next_free_agent_id(used_ids)
            agent["id"] = vehicle_id
            agent["platform_id"] = -1
            agents.append(agent)
            used_ids.add(vehicle_id)

        if not agents:
            raise RuntimeError("at least one Pluto-controlled BV is required")

        # The first configured BV is the adversarial actor. Platform agents are
        # inserted first, so hybrid mode defaults to an adversarial platform BV.
        for index, agent in enumerate(agents):
            agent["planner_mode"] = "adversarial" if index == 0 else "max"

        vehicles["agents"] = agents
        rospy.set_param("/vehicles", vehicles)
        return agents

    def configure_agent_counts_from_terminal(self):
        """按运行模式读取本轮 platform/virtual BV 数量并重建 agent 列表。"""
        runtime = rospy.get_param("/runtime", {})
        scenario = rospy.get_param("/scenario", {})
        platform_cfg = rospy.get_param("/platform", {})
        mode = str(runtime.get("mode", self.mode)).strip().lower()
        self.mode = mode

        default_platform = int(
            scenario.get("manual_platform_bv_count", 0 if mode == "simulation" else 1)
        )
        default_virtual = int(
            scenario.get("manual_virtual_bv_count", 2 if mode == "simulation" else 1)
        )

        if mode == "simulation":
            platform_count = 0
            virtual_count = self.prompt_integer(
                "Input Pluto-controlled virtual BV count",
                minimum=1,
                default=max(default_virtual, 1),
            )
            platform_endpoints = []

        elif mode == "hybrid":
            platform_endpoints = self.validate_platform_endpoint_configs(
                platform_cfg.get("platforms", []),
                platform_cfg,
            )

            if not platform_endpoints:
                raise RuntimeError(
                    "hybrid mode requires at least one valid platform.platforms endpoint"
                )

            print("")
            print("=== Configured platform endpoints ===")
            for rank, endpoint in enumerate(platform_endpoints, start=1):
                print(
                    "%d: platform_id=%s vehicle_id=%s ip=%s send_port=%s"
                    % (
                        rank,
                        endpoint.get("platform_id"),
                        endpoint.get("vehicle_id"),
                        endpoint.get("ip", ""),
                        endpoint.get("send_port", ""),
                    )
                )
            print(
                "Choose the exact endpoint(s) used by this "
                "experiment after entering the count."
            )

            platform_count = self.prompt_integer(
                "Input Pluto-controlled platform BV count",
                minimum=1,
                maximum=len(platform_endpoints),
                default=min(max(default_platform, 1), len(platform_endpoints)),
            )
            platform_endpoints = (
                self.select_platform_endpoints_from_terminal(
                    platform_endpoints,
                    platform_count,
                )
            )
            virtual_count = self.prompt_integer(
                "Input Pluto-controlled virtual BV count",
                minimum=0,
                default=max(default_virtual, 0),
            )

        else:
            raise ValueError(
                "runtime.mode must be simulation or hybrid, got %s" % mode
            )

        agents = self.build_manual_agent_configs(
            platform_count,
            virtual_count,
            platform_endpoints,
        )

        scenario["manual_platform_bv_count"] = int(platform_count)
        scenario["manual_virtual_bv_count"] = int(virtual_count)

        scenario["manual_platform_ids"] = [
            str(endpoint.get("platform_id", ""))
            for endpoint in platform_endpoints
        ]

        scenario["agent_assignment_mode"] = "manual"
        scenario["auto_assign_agents"] = False
        rospy.set_param("/scenario", scenario)

        print(
            "[OK] configured Pluto BV counts: platform=%d virtual=%d total=%d"
            % (platform_count, virtual_count, len(agents))
        )
        return agents

    def manual_assign_agents(self):
        """按本轮动态 agent 数量，逐辆人工选择 platform/virtual BV。"""
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError(
                "manual BV preparation must run in the Python main thread"
            )

        helper = self.load_runtime_helper()
        config = self.current_config_dict()
        scenario = config.get("scenario", {})
        scenario_index = int(scenario.get("scenario_index", 0))
        autoscene_root = self.autoscene_root()
        env = helper.load_closeloop_env(
            config,
            autoscene_root,
            scenario_index,
        )

        try:
            engine = env.engine
            ego_pose = helper.ego_pose_from_engine(engine)
            candidates = helper.sorted_bv_candidates(engine)

            if not candidates:
                raise RuntimeError("no valid BV candidates in current scenario")

            self.print_candidates(candidates, ego_pose)
            vehicles = rospy.get_param("/vehicles", {})
            agents = vehicles.get("agents", [])

            if not agents:
                raise RuntimeError("vehicles.agents is empty")

            if len(candidates) < len(agents):
                raise RuntimeError(
                    "requested %d Pluto-controlled BVs, but scenario %d only has "
                    "%d selectable candidates; reduce the BV counts"
                    % (
                        len(agents),
                        scenario_index,
                        len(candidates),
                    )
                )

            platform_count = sum(
                1
                for agent in agents
                if self.valid_platform_id(agent.get("platform_id"))
            )
            virtual_count = len(agents) - platform_count
            print(
                "Selecting %d platform BV(s) and %d virtual BV(s)."
                % (platform_count, virtual_count)
            )

            used_tracks = set()
            selected_by_vehicle = {}

            for agent in agents:
                role = (
                    "platform"
                    if self.valid_platform_id(agent.get("platform_id"))
                    else "virtual"
                )
                selected = self.prompt_candidate_for_agent(
                    agent,
                    role,
                    candidates,
                    used_tracks,
                )
                agent["scenario_track_id"] = selected["track_id"]
                agent.pop("metadrive_object_id", None)
                agent.pop("track_id", None)
                used_tracks.add(selected["track_id"])
                selected_by_vehicle[str(agent.get("id", ""))] = selected

            if str(scenario.get("history_source", "dataset")).strip().lower() == "initial":
                self.prompt_initial_states(
                    ego_pose,
                    agents,
                    selected_by_vehicle,
                )

            rospy.set_param("/vehicles", vehicles)
            self.message = "manual BV assignment completed"
            rospy.loginfo("%s", self.message)
        finally:
            try:
                env.close()
            except Exception:
                pass

    def print_candidates(self, candidates, ego_pose):
        """打印排序后的候选 CBV，便于现场选择 platform/virtual。"""
        print("")
        print("=== AutoScene candidate CBVs ===")
        print(
            "ego AutoScene pose: x={x:.3f}, y={y:.3f}, heading={heading:.3f}, speed={speed:.3f}".format(
                **ego_pose
            )
        )
        print(
            "rank | track_id | type | x y heading speed | "
            "rel_long rel_lat rel_heading | distance | valid[first,last,count]"
        )

        for rank, item in enumerate(candidates, start=1):
            rel_heading = item.get("relative_heading", 0.0)
            print(
                "%4d | %s | %s | %.3f %.3f %.3f %.3f | "
                "%.3f %.3f %.3f | %.3f | [%s,%s,%s]"
                % (
                    rank,
                    item["track_id"],
                    item.get("type", ""),
                    item["x"],
                    item["y"],
                    item["heading"],
                    item["speed"],
                    item["relative_longitudinal"],
                    item["relative_lateral"],
                    rel_heading,
                    item["distance_to_ego"],
                    item.get("first_valid", "-"),
                    item.get("last_valid", "-"),
                    item.get("valid_steps", "-"),
                )
            )
        print("")
        sys.stdout.flush()

    def prompt_candidate_for_agent(self, agent, role, candidates, used_tracks):
        """为一个配置 agent 读取人工选择，可输入 rank 或 track_id。"""
        prompt = (
            "Select %s BV for %s (platform_id=%s), input rank or track_id: "
            % (
                role,
                agent.get("id", "agent"),
                agent.get("platform_id", -1),
            )
        )

        by_track = {
            str(item["track_id"]): item
            for item in candidates
        }

        while not rospy.is_shutdown():
            value = self.read_terminal_line(prompt).strip()

            if not value:
                print("empty input is not valid")
                continue

            selected = None

            if value.isdigit():
                index = int(value) - 1
                if 0 <= index < len(candidates):
                    selected = candidates[index]
            else:
                selected = by_track.get(value)

            if selected is None:
                print("cannot find candidate: %s" % value)
                continue

            if selected["track_id"] in used_tracks:
                print("track_id already selected: %s" % selected["track_id"])
                continue

            print(
                "Selected %s for %s: rank/track_id=%s x=%.3f y=%.3f heading=%.3f speed=%.3f"
                % (
                    role,
                    agent.get("id", "agent"),
                    selected["track_id"],
                    selected["x"],
                    selected["y"],
                    selected["heading"],
                    selected["speed"],
                )
            )
            return selected

        raise RuntimeError("ROS shutdown while waiting for manual BV selection")

    def prompt_initial_pose(self, title, default_pose):
        """输入/确认 AutoScene world 初始状态；只新增固定历史速度上限校验。"""
        print("")
        print(
            "%s default: x=%.3f y=%.3f heading=%.6f speed=%.3f"
            % (
                title,
                float(default_pose.get("x", 0.0)),
                float(default_pose.get("y", 0.0)),
                float(default_pose.get("heading", 0.0)),
                float(default_pose.get("speed", 0.0)),
            )
        )
        x = self.prompt_float("%s initial.x" % title, default_pose.get("x", 0.0))
        y = self.prompt_float("%s initial.y" % title, default_pose.get("y", 0.0))
        heading = self.prompt_float(
            "%s initial.heading(rad)" % title,
            default_pose.get("heading", 0.0),
        )
        max_speed = self.initial_history_max_speed()
        while not rospy.is_shutdown():
            speed = self.prompt_float(
                "%s initial.speed(m/s, max=%.6f)" % (title, max_speed),
                default_pose.get("speed", 0.0),
            )
            if 0.0 <= speed <= max_speed + 1e-9:
                break
            print(
                "initial speed must be in [0, %.6f] m/s for the fixed history window"
                % max_speed
            )
        return {
            "x": x,
            "y": y,
            "heading": heading,
            "speed": speed,
        }

    def prompt_initial_states(self, ego_pose, agents, selected_by_vehicle):
        """history_source=initial 时，为 ego 和已选 BV 写入启动初始状态。"""
        ego_cfg = rospy.get_param("/ego", {})
        ego_cfg["initial"] = self.prompt_initial_pose(
            "ego",
            ego_pose,
        )
        rospy.set_param("/ego", ego_cfg)

        for agent in agents:
            vehicle_id = str(agent.get("id", "agent"))
            selected = selected_by_vehicle.get(vehicle_id)
            if selected is None:
                raise RuntimeError("missing selected candidate for %s" % vehicle_id)
            agent["initial"] = self.prompt_initial_pose(
                vehicle_id,
                {
                    "x": selected["x"],
                    "y": selected["y"],
                    "heading": selected["heading"],
                    "speed": selected["speed"],
                },
            )

    @staticmethod
    def read_terminal_line(prompt):
        """从启动终端读取一行；roslaunch 未连接 stdin 时尝试 /dev/tty。"""
        try:
            return input(prompt)
        except EOFError:
            if os.path.exists("/dev/tty"):
                with open("/dev/tty", "r", encoding="utf-8") as tty_in:
                    with open("/dev/tty", "w", encoding="utf-8") as tty_out:
                        tty_out.write(prompt)
                        tty_out.flush()
                        line = tty_in.readline()
                        if line:
                            return line
            raise RuntimeError(
                "manual BV assignment requires an interactive terminal"
            )

    def spin(self):
        """运行 manager。

        交互模式必须让 console_loop() 占用 Python 主线程，因为菜单 2 会初始化
        MetaDrive/Panda3D；其 signal handler 只能在主线程注册。rospy 的订阅、
        service 和 timer 回调不依赖 rospy.spin()，仍会由内部线程正常执行。
        """
        if self.interactive_console:
            self.console_loop()
        else:
            rospy.spin()

    def console_loop(self):
        """在 Python 主线程中运行常驻键盘菜单。"""
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError(
                "interactive console must run in the Python main thread"
            )
        while not rospy.is_shutdown():
            try:
                self.print_console_menu()
                command = self.read_terminal_line("AutoScene manager> ").strip().lower()

                if command in ("1", "scene", "scenario"):
                    self.select_scenario_from_terminal()
                elif command in ("2", "bv", "prepare", "count"):
                    self.prepare_from_terminal()
                elif command in ("3", "show", "selection"):
                    self.print_current_selection()
                elif command in ("4", "start", "run"):
                    ok, message = self.start("interactive console")
                    print(("[OK] " if ok else "[FAIL] ") + message)
                elif command in ("5", "stop"):
                    ok, message = self.stop("interactive console")
                    print(("[OK] " if ok else "[FAIL] ") + message)
                elif command in ("6", "status"):
                    self.print_manager_status()
                elif command in ("r", "reset"):
                    ok, message = self.reset(False)
                    print(("[OK] " if ok else "[FAIL] ") + message)
                elif command in ("q", "quit", "exit"):
                    rospy.signal_shutdown("interactive console requested shutdown")
                    return
                elif command:
                    print("unknown command: %s" % command)
            except (EOFError, KeyboardInterrupt):
                rospy.signal_shutdown("interactive console closed")
                return
            except Exception as exc:
                rospy.logerr("interactive command failed: %s", exc)
                print("[FAIL] %s" % exc)

    def print_console_menu(self):
        """打印一次常驻实验控制菜单。"""
        print("")
        print("========== AutoScene Experiment Manager ==========")
        print("1 / scene    选择模式、数据集、历史来源和单个 scenario_index")
        print("2 / prepare  设置 platform/virtual 数量，打印候选并逐辆选车")
        print("3 / show     显示当前场景和 BV 绑定")
        print("4 / start    在新终端启动本次实验")
        print("5 / stop     停止当前实验工作节点")
        print("6 / status   显示 manager 状态")
        print("r / reset    清除准备标志，重新选择")
        print("q / quit     退出常驻 manager")
        print("==================================================")
        sys.stdout.flush()

    def select_scenario_from_terminal(self):
        """读取运行配置和 scenario_index，并把本次实验固定为单场景运行。"""
        self.reap_worker_processes()

        if self.node_procs:
            raise RuntimeError(
                "worker terminal is still open; close it before changing scenario"
            )

        self.configure_runtime_dataset_from_terminal()

        scenario_count = self.dataset_scenario_count()
        print(
            "Dataset contains %d scenario(s); valid scenario_index range: 0..%d"
            % (scenario_count, scenario_count - 1)
        )
        value = self.read_terminal_line(
            "Input scenario_index (non-negative integer): "
        ).strip()

        if not value.isdigit():
            raise ValueError("scenario_index must be a non-negative integer")

        scenario_index = int(value)
        if scenario_index >= scenario_count:
            raise ValueError(
                "scenario_index=%d is out of range; dataset has %d scenario(s), "
                "valid range is 0..%d"
                % (scenario_index, scenario_count, scenario_count - 1)
            )

        scenario = rospy.get_param("/scenario", {})
        scenario["scenario_index"] = scenario_index
        scenario["eval_eps"] = 1
        scenario["auto_advance_scenario"] = False
        scenario["agent_assignment_mode"] = "manual"
        scenario["auto_assign_agents"] = False
        rospy.set_param("/scenario", scenario)

        # 场景变化后旧 track 绑定不能复用。
        vehicles = rospy.get_param("/vehicles", {})
        for agent in vehicles.get("agents", []):
            agent.pop("scenario_track_id", None)
            agent.pop("metadrive_object_id", None)
            agent.pop("track_id", None)
            agent.pop("initial", None)
        rospy.set_param("/vehicles", vehicles)

        with self.lifecycle_lock:
            self.prepared = False
            self.runtime_reported_terminal = False
            self.state = "idle"
            self.message = "selected scenario_index=%d; BV assignment required" % scenario_index
            self.publish_status()

        print("[OK] %s" % self.message)

    def prepare_from_terminal(self):
        """先设置本轮 BV 数量，再打开场景并逐辆人工选择。"""
        self.reap_worker_processes()

        if self.node_procs:
            raise RuntimeError(
                "worker terminal is still open; close it before preparing another experiment"
            )

        with self.lifecycle_lock:
            self.state = "preparing"
            self.message = "loading scenario and waiting for manual BV selection"
            self.publish_status()

        try:
            self.prepared = False
            self.configure_agent_counts_from_terminal()
            self.prepare_experiment_configuration()
        except Exception:
            with self.lifecycle_lock:
                self.state = "error"
                self.message = "manual BV preparation failed"
                self.publish_status()
            raise

        with self.lifecycle_lock:
            self.state = "prepared"
            self.message = "scenario and BV assignment prepared"
            self.publish_status()

        self.print_current_selection()

    def print_current_selection(self):
        """显示将被下一次 worker 读取的 ROS 参数。"""
        scenario = rospy.get_param("/scenario", {})
        vehicles = rospy.get_param("/vehicles", {})

        print("")
        print("=== Current experiment selection ===")
        agents = vehicles.get("agents", [])
        platform_count = sum(
            1
            for agent in agents
            if self.valid_platform_id(agent.get("platform_id"))
        )
        virtual_count = len(agents) - platform_count

        print(
            "mode=%s dataset=%s history=%s scenario_index=%s eval_eps=%s auto_advance=%s "
            "platform_bv=%d virtual_bv=%d prepared=%s"
            % (
                self.mode,
                scenario.get("dataset_type"),
                scenario.get("history_source"),
                scenario.get("scenario_index"),
                scenario.get("eval_eps"),
                scenario.get("auto_advance_scenario"),
                platform_count,
                virtual_count,
                self.prepared,
            )
        )

        for agent in vehicles.get("agents", []):
            role = (
                "platform"
                if self.valid_platform_id(agent.get("platform_id"))
                else "virtual"
            )
            print(
                "%s role=%s planner_mode=%s platform_id=%s track_id=%s initial=%s"
                % (
                    agent.get("id"),
                    role,
                    agent.get("planner_mode", "max"),
                    agent.get("platform_id", -1),
                    agent.get("scenario_track_id", "<not selected>"),
                    agent.get("initial", "<dataset>"),
                )
            )
        print("")
        sys.stdout.flush()

    def print_manager_status(self):
        """打印 manager、worker 和最近一次完成状态。"""
        self.reap_worker_processes()
        worker_open = any(
            proc.poll() is None
            for _name, proc in self.node_procs
        )
        print(
            "state=%s message=%s prepared=%s worker_terminal_open=%s "
            "runtime_reported_terminal=%s"
            % (
                self.state,
                self.message,
                self.prepared,
                worker_open,
                self.runtime_reported_terminal,
            )
        )
        sys.stdout.flush()

    def load_runtime_helper(self):
        """加载 runtime 侧离线工具函数，复用同一套候选排序逻辑。"""
        import importlib.util

        helper_path = os.path.join(
            self.autoscene_root(),
            "src",
            "autoscene_runtime",
            "scripts",
            "calibrate_origins.py",
        )
        spec = importlib.util.spec_from_file_location(
            "autoscene_calibrate_origins",
            helper_path,
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def current_config_dict(self):
        """从参数服务器拼出离线场景解析需要的配置字典。"""
        return {
            "project": rospy.get_param("/project", {}),
            "runtime": rospy.get_param("/runtime", {}),
            "scenario": rospy.get_param("/scenario", {}),
            "algorithm": rospy.get_param("/algorithm", {}),
            "ego": rospy.get_param("/ego", {}),
            "vehicles": rospy.get_param("/vehicles", {}),
            "platform": rospy.get_param("/platform", {}),
        }

    def autoscene_root(self):
        return os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "../../..",
            )
        )

    def resolve_project_path(self, path):
        """相对路径固定解析到 AutoScene 工程根目录。"""
        path = os.path.expanduser(str(path))

        if os.path.isabs(path):
            return path

        return os.path.abspath(
            os.path.join(
                self.autoscene_root(),
                path,
            )
        )

    def dataset_scenario_count(self):
        """读取 ScenarioNet summary，返回数据集真实场景数。

        不按目录文件数猜测，也不对越界 index 做夹取或自动改写。
        summary 缺失、数据集为空或读取失败时直接暴露错误。
        """
        scenario = rospy.get_param("/scenario", {})
        dataset_path = self.resolve_project_path(
            str(scenario.get("dataset_path", "")).strip()
        )
        if not dataset_path:
            raise RuntimeError("scenario.dataset_path is required")
        if not os.path.isdir(dataset_path):
            raise RuntimeError(
                "scenario.dataset_path is not a directory: %s" % dataset_path
            )

        from scenarionet.common_utils import read_dataset_summary

        _summary, summary_list, _mapping = read_dataset_summary(dataset_path)
        scenario_count = len(summary_list)
        if scenario_count <= 0:
            raise RuntimeError(
                "ScenarioNet dataset contains no scenarios: %s" % dataset_path
            )
        return scenario_count

    @staticmethod
    def valid_platform_id(value):
        if value is None:
            return False
        return str(value).strip().lower() not in ("", "-1", "none", "null")

    def validate_platform_endpoint_configs(self, endpoints, platform_cfg):
        """Validate selected remote endpoints and the shared local UDP socket."""
        seen_platform_ids = set()
        seen_vehicle_ids = set()
        seen_remote_filters = set()

        receive_udp = bool(platform_cfg.get("receive_udp", False))
        send_udp = bool(platform_cfg.get("send_udp", False))
        emulate = bool(platform_cfg.get("emulate_controlled_feedback", False))
        if receive_udp and emulate:
            raise RuntimeError(
                "receive_udp and emulate_controlled_feedback cannot both be true"
            )
        if send_udp and emulate:
            raise RuntimeError(
                "send_udp and emulate_controlled_feedback cannot both be true"
            )
        if send_udp and not receive_udp:
            raise RuntimeError("send_udp requires receive_udp in hardware hybrid mode")
        if receive_udp and not emulate and not bool(
                platform_cfg.get("strict_actual_feedback", True)
        ):
            raise RuntimeError(
                "hardware hybrid requires platform.strict_actual_feedback=true"
            )
        if not receive_udp and not emulate:
            raise RuntimeError(
                "hybrid platform requires receive_udp or emulate_controlled_feedback"
            )

        algorithm_cfg = rospy.get_param("/algorithm", {})
        history_steps = int(algorithm_cfg.get("pluto_history_steps", 21))
        history_dt = float(algorithm_cfg.get("pluto_history_dt_sec", 0.1))
        startup_accel = float(algorithm_cfg.get("initial_history_accel_mps2", 2.0))
        if history_steps < 21:
            raise RuntimeError("algorithm.pluto_history_steps must be >= 21")
        if history_dt <= 0.0:
            raise RuntimeError("algorithm.pluto_history_dt_sec must be > 0")
        if startup_accel <= 0.0:
            raise RuntimeError("algorithm.initial_history_accel_mps2 must be > 0")

        point_num = int(platform_cfg.get("packet_point_num", 50))
        trajectory_dt = float(platform_cfg.get("trajectory_dt_sec", 0.1))
        if point_num != 50:
            raise RuntimeError("platform.packet_point_num must be 50")
        if abs(trajectory_dt - 0.1) > 1e-9:
            raise RuntimeError("platform.trajectory_dt_sec must be 0.1")

        use_udp = receive_udp or send_udp
        if use_udp:
            local_ip = str(platform_cfg.get("local_ip", "")).strip()
            local_port = int(platform_cfg.get("local_port", 0))
            if not local_ip:
                raise RuntimeError("platform.local_ip is required when UDP is enabled")
            if not 0 < local_port <= 65535:
                raise RuntimeError("invalid platform.local_port=%r" % local_port)

        earth_to_autoscene = platform_cfg.get("earth_to_autoscene", {})
        if not isinstance(earth_to_autoscene, dict):
            raise RuntimeError("platform.earth_to_autoscene must be a mapping")
        for key in ("tx", "ty", "yaw"):
            try:
                float(earth_to_autoscene.get(key, 0.0))
            except (TypeError, ValueError):
                raise RuntimeError("platform.earth_to_autoscene.%s must be numeric" % key)

        validated = []
        for index, endpoint in enumerate(endpoints, start=1):
            if not isinstance(endpoint, dict):
                raise RuntimeError("platform endpoint %d must be a mapping" % index)

            platform_id = str(endpoint.get("platform_id", "")).strip()
            vehicle_id = str(endpoint.get("vehicle_id", "")).strip()
            remote_ip = str(endpoint.get("ip", "")).strip()
            send_port = int(endpoint.get("send_port", 0))

            if not self.valid_platform_id(platform_id):
                raise RuntimeError(
                    "platform endpoint %d has invalid platform_id=%r"
                    % (index, platform_id)
                )
            if not vehicle_id:
                raise RuntimeError("platform endpoint %d has empty vehicle_id" % index)
            if platform_id in seen_platform_ids:
                raise RuntimeError("duplicate platform_id=%s" % platform_id)
            if vehicle_id in seen_vehicle_ids:
                raise RuntimeError("duplicate vehicle_id=%s" % vehicle_id)
            if not remote_ip:
                raise RuntimeError("platform endpoint %d has empty ip" % index)
            if not 0 < send_port <= 65535:
                raise RuntimeError(
                    "platform endpoint %d has invalid send_port=%r"
                    % (index, send_port)
                )

            if receive_udp:
                expected_ip = str(endpoint.get("expected_remote_ip", "")).strip()
                expected_port = int(endpoint.get("expected_remote_port", -1))
                if not expected_ip:
                    raise RuntimeError(
                        "platform endpoint %d expected_remote_ip is required" % index
                    )
                remote_filter = (expected_ip, expected_port)
                if remote_filter in seen_remote_filters:
                    raise RuntimeError(
                        "duplicate platform feedback filter %s:%s"
                        % remote_filter
                    )
                seen_remote_filters.add(remote_filter)

            seen_platform_ids.add(platform_id)
            seen_vehicle_ids.add(vehicle_id)
            validated.append(endpoint)

        return validated

    def resolve_terminal_emulator(self):
        """解析用于显示单次实验日志的终端程序。"""
        requested = self.terminal_emulator

        if requested and requested.lower() != "auto":
            path = shutil.which(requested)
            if path is None:
                raise RuntimeError(
                    "terminal emulator not found: %s" % requested
                )
            return path

        for candidate in ("gnome-terminal", "xterm", "konsole"):
            path = shutil.which(candidate)
            if path is not None:
                return path

        raise RuntimeError(
            "no supported terminal emulator found; "
            "install gnome-terminal/xterm/konsole or set launch_in_terminal=false"
        )

    def worker_roslaunch_command(self):
        """只连接当前 manager 的 ROS master，不允许创建空 master。"""
        ego_mode = str(
            rospy.get_param("/ego/execution_mode", "simulation")
        ).strip().lower()
        return [
            "roslaunch",
            "--wait",
            "autoscene_bringup",
            self.worker_launch,
            "mode:=%s" % self.mode,
            "ego_execution_mode:=%s" % ego_mode,
        ]

    @staticmethod
    def shell_join(argv):
        """兼容 Python 3.8 的 shell 安全拼接。"""
        return " ".join(shlex.quote(str(item)) for item in argv)

    def build_terminal_command(self, launch_cmd):
        """构造工作终端命令。

        正常退出时自动关闭终端；异常退出时保留日志，按 Enter 后关闭。
        manager 通过终端进程判断本轮 worker 是否已经结束。
        """
        terminal = self.resolve_terminal_emulator()
        terminal_name = os.path.basename(terminal)
        title = "AutoScene %s scenario %s" % (
            self.mode,
            rospy.get_param("/scenario/scenario_index", 0),
        )
        launch_line = self.shell_join(launch_cmd)
        script = (
                "set +e; "
                + launch_line
                + "; rc=$?; echo; "
                + 'echo "AutoScene worker finished (exit code=${rc})."; '
                + "echo 'Press Enter to close this log window.'; "
                + "read -r _; "
                + "exit ${rc}"
        )

        if terminal_name == "gnome-terminal":
            return [
                terminal,
                "--wait",
                "--title=%s" % title,
                "--",
                "bash",
                "-c",
                script,
            ]

        if terminal_name == "xterm":
            return [
                terminal,
                "-T",
                title,
                "-e",
                "bash",
                "-c",
                script,
            ]

        if terminal_name == "konsole":
            return [
                terminal,
                "-p",
                "tabtitle=%s" % title,
                "-e",
                "bash",
                "-c",
                script,
            ]

        raise RuntimeError(
            "unsupported terminal emulator: %s" % terminal
        )

    def start_managed_nodes(self):
        """为一次实验启动独立 worker roslaunch。

        worker launch 将 scenario_runtime_node 标记为 required；runtime 退出后，
        roslaunch 会关闭本轮可选适配节点和 evaluator。manager 不在 worker launch
        中，因此始终保持运行。
        """
        self.reap_worker_processes()

        if self.node_procs:
            raise RuntimeError(
                "previous worker terminal is still open"
            )

        launch_cmd = self.worker_roslaunch_command()
        cmd = (
            self.build_terminal_command(launch_cmd)
            if self.launch_in_terminal
            else launch_cmd
        )

        popen_kwargs = (
            {"preexec_fn": os.setsid}
            if hasattr(os, "setsid")
            else {}
        )
        worker_env = os.environ.copy()

        # manager 已位于 /autoscene，而 worker launch 自身已有
        # <group ns="autoscene">，必须清除继承 namespace。
        worker_env.pop("ROS_NAMESPACE", None)

        proc = subprocess.Popen(
            cmd,
            cwd=self.autoscene_root(),
            env=worker_env,
            **popen_kwargs
        )
        process_name = (
            "experiment_terminal"
            if self.launch_in_terminal
            else "experiment_worker_launch"
        )
        time.sleep(0.1)
        if proc.poll() is not None:
            raise RuntimeError(
                "experiment worker exited immediately with code %s"
                % proc.returncode
            )
        self.node_procs = [(process_name, proc)]
        rospy.loginfo(
            "started experiment worker: %s",
            self.shell_join(cmd),
        )

    def request_worker_ros_shutdown(self):
        """按后缀查找并关闭 worker，兼容单层和旧双层 namespace。"""
        suffixes = (
            "/scenario_runtime_node",
            "/ego_adapter_node",
            "/platform_comm_node",
            "/evaluator_node",
        )

        try:
            output = subprocess.check_output(
                ["rosnode", "list"],
                text=True,
                timeout=2.0,
            )

            node_names = [
                name.strip()
                for name in output.splitlines()
                if name.strip().endswith(suffixes)
            ]

            if node_names:
                subprocess.call(
                    ["rosnode", "kill"] + node_names,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5.0,
                )
        except Exception:
            pass

    def stop_managed_nodes(self):
        """停止本次 worker；不影响常驻 experiment_manager_node。"""
        if not self.node_procs:
            return

        self.request_worker_ros_shutdown()
        time.sleep(0.3)

        for _name, proc in reversed(self.node_procs):
            try:
                if proc.poll() is not None:
                    continue

                if hasattr(os, "killpg") and hasattr(os, "getpgid"):
                    os.killpg(os.getpgid(proc.pid), signal.SIGINT)
                else:
                    proc.send_signal(signal.SIGINT)

                proc.wait(timeout=3.0)
            except Exception:
                try:
                    proc.terminate()
                except Exception:
                    pass

        self.node_procs = []

    def reap_worker_processes(self):
        """清理已经关闭的工作终端/worker roslaunch 进程引用。"""
        if not self.node_procs:
            return []

        alive = []
        exited = []

        for name, proc in self.node_procs:
            return_code = proc.poll()

            if return_code is None:
                alive.append((name, proc))
            else:
                exited.append((name, return_code))

        self.node_procs = alive
        return exited

    def start_rosbag(self):
        """按当前参数服务器中的 record 配置启动 rosbag。"""
        self.record = rospy.get_param("/record", {})

        if not bool(self.record.get("enable_rosbag", False)):
            return

        if self.rosbag_proc is not None:
            if self.rosbag_proc.poll() is None:
                raise RuntimeError("rosbag is already running")
            self.rosbag_proc = None

        topics = list(self.record.get("topics", []))

        if not topics:
            raise RuntimeError(
                "record.enable_rosbag=true but record.topics is empty"
            )

        output_dir = self.resolve_project_path(
            self.record.get("output_dir", "./outputs")
        )
        os.makedirs(output_dir, exist_ok=True)

        scenario_index = int(
            rospy.get_param("/scenario/scenario_index", 0)
        )
        stamp = time.strftime("%Y%m%d_%H%M%S")

        bag_path = os.path.join(
            output_dir,
            "%s_scenario_%d_%s.bag"
            % (
                self.experiment_id,
                scenario_index,
                stamp,
            ),
        )

        cmd = [
                  "rosbag",
                  "record",
                  "-O",
                  bag_path,
              ] + topics

        try:
            popen_kwargs = (
                {"preexec_fn": os.setsid}
                if hasattr(os, "setsid")
                else {}
            )

            self.rosbag_proc = subprocess.Popen(
                cmd,
                cwd=self.autoscene_root(),
                **popen_kwargs
            )
            time.sleep(0.1)
            if self.rosbag_proc.poll() is not None:
                raise RuntimeError(
                    "rosbag exited immediately with code %s"
                    % self.rosbag_proc.returncode
                )

            self.message += (
                    "; rosbag recording: %s" % bag_path
            )
            rospy.loginfo(
                "rosbag recording started: %s",
                bag_path,
            )

        except Exception as exc:
            self.rosbag_proc = None
            raise RuntimeError("failed to start rosbag: %s" % exc) from exc

    def stop_rosbag(self):
        """Stop rosbag and wait for the index to be finalized."""
        proc = self.rosbag_proc
        if proc is None:
            return True
        if proc.poll() is not None:
            self.rosbag_proc = None
            return True

        def signal_process(sig):
            if hasattr(os, "killpg") and hasattr(os, "getpgid"):
                os.killpg(os.getpgid(proc.pid), sig)
            else:
                proc.send_signal(sig)

        try:
            rospy.loginfo("stopping rosbag pid=%d", proc.pid)
            signal_process(signal.SIGINT)
            try:
                proc.wait(timeout=10.0)
                rospy.loginfo("rosbag stopped cleanly")
                self.rosbag_proc = None
                return True
            except subprocess.TimeoutExpired:
                rospy.logwarn("rosbag did not exit after SIGINT; sending SIGTERM")

            signal_process(signal.SIGTERM)
            try:
                proc.wait(timeout=3.0)
                self.rosbag_proc = None
                return True
            except subprocess.TimeoutExpired:
                rospy.logerr("rosbag did not exit after SIGTERM; sending SIGKILL")

            signal_process(signal.SIGKILL)
            proc.wait(timeout=3.0)
            self.rosbag_proc = None
            return False
        except ProcessLookupError:
            self.rosbag_proc = None
            return True
        except Exception as exc:
            rospy.logerr("failed to stop rosbag: %s", exc)
            if proc.poll() is not None:
                self.rosbag_proc = None
            return False

    def shutdown_children(self):
        """manager 退出时停止 rosbag 和当前 worker。"""
        self.stop_rosbag()

        if self.manage_nodes:
            self.stop_managed_nodes()

    def publish_status(self):
        """发布当前实验状态，latch 使后启动的观察者也能拿到最近状态。"""
        msg = ExperimentStatus()
        msg.header.stamp = rospy.Time.now()
        msg.experiment_id = self.experiment_id
        msg.mode = self.mode
        msg.state = self.state
        msg.elapsed = time.time() - self.start_time if self.start_time else 0.0
        msg.message = self.message
        self.status_pub.publish(msg)

    def on_timer(self, _event):
        """发布状态，并检测用户是否已经关闭上一轮工作终端。"""
        with self.lifecycle_lock:
            exited = self.reap_worker_processes()

            if exited:
                exit_text = ", ".join(
                    "%s=%s" % (name, code)
                    for name, code in exited
                )

                if self.state in ("finished", "stopped", "idle", "error"):
                    if self.state == "finished":
                        self.state = "idle"
                        self.start_time = None
                        self.prepared = False
                        self.message = (
                            "scenario finished; worker terminal closed; "
                            "select scenario and BV for the next experiment"
                        )
                elif self.runtime_reported_terminal:
                    self.state = "finished"
                    self.message = (
                        "scenario finished; worker exited: %s" % exit_text
                    )
                    self.stop_rosbag()
                else:
                    self.state = "error"
                    self.message = (
                        "worker exited before runtime reported finished: %s"
                        % exit_text
                    )
                    self.stop_rosbag()
                    rospy.logerr("%s", self.message)

            self.publish_status()


if __name__ == "__main__":
    try:
        node = ExperimentManagerNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass