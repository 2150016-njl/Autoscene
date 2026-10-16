#!/usr/bin/env python3
"""AutoScene 场景运行节点。

本节点是整个系统的“世界状态生产者”：
1. 优先启动 UniTraj/MetaDrive/Pluto 闭环仿真，生成自车和周车的 world 坐标状态。
2. 在虚实结合模式下，把需要真实运动平台执行的周车轨迹发布给 platform_comm_node。
3. 接收平台反馈后，将真实平台状态回灌到 MetaDrive 对象，保持仿真镜像和实物同步。

坐标约定：
- 本文件所有对外 ROS 消息均使用 world/map 坐标，heading 为 yaw 弧度，yaw=0 指向 world +x。
- 平台大地投影坐标和 UDP 协议字段只在 platform_comm_node 中处理。
"""
import copy
import re
import csv
import json
import math
import os
import sys
import time
import traceback

import rospy
from std_msgs.msg import String

from autoscene_msgs.msg import (
    ScenarioStatus,
    SimulationStatus,
    Trajectory,
    TrajectoryArray,
    TrajectoryPoint,
    VehicleState,
    VehicleStateArray,
    WorldState,
)

import numpy as np


def get_param(name, default=None):
    """同时兼容全局参数和私有参数，方便 launch 中按 namespace 注入配置。"""
    return rospy.get_param(name, rospy.get_param("~" + name.strip("/"), default))


def wrap_rad(angle):
    """把弧度角归一化到 [-pi, pi)。"""
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def yaw_to_quat(yaw):
    """把平面 yaw 弧度转换成 ROS Quaternion；本项目不使用 roll/pitch。"""
    from geometry_msgs.msg import Quaternion

    q = Quaternion()
    q.z = math.sin(yaw * 0.5)
    q.w = math.cos(yaw * 0.5)
    return q


def apply_pose(pose, x, y, yaw):
    """统一写入 2D pose，避免各处重复写 quaternion 逻辑。"""
    pose.position.x = x
    pose.position.y = y
    pose.position.z = 0.0
    pose.orientation = yaw_to_quat(yaw)


def valid_platform_id(value):
    """判断 platform_id 是否表示真实/模拟底盘。"""
    if value is None:
        return False
    return str(value).strip().lower() not in ("", "-1", "none", "null")


def is_platform_controlled_agent(cfg):
    """判断一个 BV 是否绑定真实/模拟运动平台。"""
    if not isinstance(cfg, dict):
        return False

    return valid_platform_id(
        cfg.get("platform_id")
    )


class ScenarioRuntimeNode:
    def __init__(self):
        rospy.init_node("scenario_runtime_node")
        # 这些参数由 autoscene_bringup 的 YAML 统一加载到 /autoscene namespace 下。
        self.project = get_param("/project", {})
        self.runtime = get_param("/runtime", {})
        self.scenario = get_param("/scenario", {})
        self.algorithm = get_param("/algorithm", {})

        self.ego_cfg = get_param("/ego", {})
        self.vehicles = get_param("/vehicles", {})
        self.platform_cfg = get_param("/platform", {})

        self.evaluation_cfg = get_param(
            "/evaluation",
            {},
        )

        self.include_background_vehicles = bool(
            self.evaluation_cfg.get(
                "include_background_vehicles",
                True,
            )
        )

        self.evaluation_finalize_delay_sec = max(
            float(
                self.evaluation_cfg.get(
                    "finalize_delay_sec",
                    0.5,
                )
            ),
            0.0,
        )
        self.mode = self.runtime.get(
            "mode",
            "simulation",
        )
        self.frame_id = self.runtime.get("frame_id", "map")
        self.scenario_id = self.scenario.get("scenario_id", "case_001")
        self.loop_hz = float(self.runtime.get("world_state_hz", 10.0))
        self.dataset_type = str(
            self.scenario.get("dataset_type", "tjwest")
        ).strip().lower()
        self.dataset_path = str(
            self.scenario.get("dataset_path", "")
        ).strip()
        self.history_source = str(
            self.scenario.get("history_source", "dataset")
        ).strip().lower()

        self.validate_runtime_dataset_config()

        # AutoScene 对外实际使用的轨迹窗口固定由平台协议决定：
        # 50 个 10Hz 点，即 5 秒。即便 Pluto 原始输出更多点，这里也只取前 50 点。
        self.output_dt = float(self.platform_cfg.get("trajectory_dt_sec", 0.1))
        self.point_num = int(self.platform_cfg.get("packet_point_num", 50))
        # 50 个 0.1s 采样点的首末点时间跨度是 4.9s。
        self.output_horizon_sec = self.output_dt * max(self.point_num - 1, 0)

        self.ego_execution_mode = str(
            self.ego_cfg.get(
                "execution_mode",
                "simulation",
            )
        ).strip().lower()

        if self.ego_execution_mode not in (
                "simulation",
                "external",
        ):
            raise ValueError(
                "ego.execution_mode must be "
                "simulation or external"
            )

        if self.mode not in ("simulation", "hybrid"):
            raise ValueError(
                "runtime.mode must be simulation or hybrid, got %s"
                % self.mode
            )

        if self.mode == "simulation" and self.ego_execution_mode != "simulation":
            raise ValueError(
                "simulation mode requires ego.execution_mode=simulation"
            )

        self.start_wall = time.time()

        # 仿真状态直接由 runtime 使用。只有 external ego 和真实平台状态
        # 通过适配节点反馈，且各自使用独立的保鲜时间。
        self.ego_feedback_timeout_sec = max(
            float(self.ego_cfg.get("external_timeout_ms", 300)) * 1e-3,
            1e-3,
        )
        self.platform_feedback_timeout_sec = max(
            float(self.platform_cfg.get("feedback_timeout_sec", 0.5)),
            1e-3,
        )

        # 真实硬件闭环中，缺少任一受控平台 actual feedback 时暂停 env.step()，
        # 不允许 MetaDrive/Pluto 继续以模拟状态推进。
        self.strict_actual_feedback = bool(
            self.platform_cfg.get(
                "strict_actual_feedback",
                True,
            )
        )
        self.emulated_platform_feedback = bool(
            self.mode == "hybrid"
            and self.platform_cfg.get("emulate_controlled_feedback", False)
        )
        self.hardware_platform_feedback = bool(
            self.mode == "hybrid"
            and self.platform_cfg.get("receive_udp", False)
            and not self.emulated_platform_feedback
        )
        if self.hardware_platform_feedback and not self.strict_actual_feedback:
            raise ValueError(
                "hardware hybrid requires platform.strict_actual_feedback=true"
            )

        self.strict_external_ego_feedback = bool(
            self.ego_cfg.get("strict_external_feedback", True)
        )
        self.platform_startup_timeout_sec = float(
            self.platform_cfg.get("startup_timeout_sec", 10.0)
        )
        self.platform_dropout_timeout_sec = float(
            self.platform_cfg.get("dropout_timeout_sec", 1.0)
        )
        self.ego_startup_timeout_sec = float(
            self.ego_cfg.get("startup_timeout_sec", 10.0)
        )
        self.ego_dropout_timeout_sec = float(
            self.ego_cfg.get("dropout_timeout_sec", 1.0)
        )
        self.platform_feedback_seen = False
        self.ego_feedback_seen = False
        self.last_platform_valid_wall = None
        self.last_ego_valid_wall = None
        self.runtime_failure_reason = ""

        self.latest_ego_state_feedback = None
        self.latest_ego_state_feedback_stamp = None

        self.latest_agent_state_feedback = {}
        # 每条真实平台反馈只能推进一次闭环；禁止重复消费同一 UDP 样本。
        self.last_consumed_platform_feedback_stamp_ns = {}

        self.controlled_vehicle_ids = {
            str(agent.get("id"))
            for agent in self.vehicles.get("agents", [])
            if is_platform_controlled_agent(agent)
        }

        if self.mode == "simulation" and self.controlled_vehicle_ids:
            raise ValueError(
                "simulation mode requires every BV platform_id=-1; "
                "controlled vehicles=%s"
                % sorted(self.controlled_vehicle_ids)
            )

        if self.mode == "hybrid" and not self.controlled_vehicle_ids:
            raise ValueError(
                "hybrid mode requires at least one BV with a valid platform_id"
            )

        # 场景数据路径由 Autoscene_generator 的 config.yaml 决定。
        self.unitraj = UniTrajPlutoBackend(
            self.scenario,
            self.vehicles,
            self.frame_id,
            self.algorithm,
            self.platform_cfg,
            runtime_cfg=self.runtime,
            ego_cfg=self.ego_cfg,
            evaluation_cfg=self.evaluation_cfg,
        )
        self.unitraj.initialize()
        if self.unitraj.active_scenario_id:
            self.scenario_id = self.unitraj.active_scenario_id
        self.feedback_wait_start_wall = time.time()

        self.world_pub = rospy.Publisher("world/state", WorldState, queue_size=1) # 评估用
        self.ego_state_pub = rospy.Publisher("ego/state", VehicleState, queue_size=1) # ego adapter不再调用
        self.agents_pub = rospy.Publisher("agents/state", VehicleStateArray, queue_size=1) # 有实物下，平台运行节点不再调用
        self.map_pub = rospy.Publisher("map/info", String, queue_size=1, latch=True)
        self.target_pub = rospy.Publisher("platform/target_trajectory", TrajectoryArray, queue_size=1)
        # 状态 topic 使用 latch，确保稍晚启动的 manager/evaluator 也能拿到最近状态。
        self.scenario_status_pub = rospy.Publisher(
            "scenario/status",
            ScenarioStatus,
            queue_size=1,
            latch=True,
        ) # 评估节点和实验管理节点使用
        self.sim_status_pub = rospy.Publisher(
            "simulation/status",
            SimulationStatus,
            queue_size=1,
            latch=True,
        ) # 没谁订阅

        if self.ego_execution_mode == "external":
            rospy.Subscriber(
                "ego/state_feedback",
                VehicleState,
                self.on_ego_state_feedback,
                queue_size=1,
            )
        if self.mode == "hybrid" and not self.emulated_platform_feedback:
            rospy.Subscriber(
                "agents/state_feedback",
                VehicleStateArray,
                self.on_agents_state_feedback,
                queue_size=1,
            )

        # 先发布一次启动上下文；拿到真实 ScenarioNet scenario_id 后会再次更新。
        self.publish_map_info()
        rospy.loginfo("AutoScene runtime started in %s mode, scenario=%s", self.mode, self.scenario_id)

    def validate_runtime_dataset_config(self):
        """校验运行模式、数据集类型和历史来源的允许组合。"""
        if self.dataset_type not in ("nuplan", "tjwest"):
            raise ValueError(
                "scenario.dataset_type must be nuplan or tjwest, got %s"
                % self.dataset_type
            )
        if self.history_source not in ("dataset", "initial"):
            raise ValueError(
                "scenario.history_source must be dataset or initial, got %s"
                % self.history_source
            )
        if self.mode == "hybrid" and self.dataset_type != "tjwest":
            raise ValueError("hybrid mode only supports scenario.dataset_type=tjwest")
        if self.mode == "hybrid" and self.history_source != "initial":
            raise ValueError("hybrid mode requires scenario.history_source=initial")
        if self.history_source == "dataset" and self.mode != "simulation":
            raise ValueError("scenario.history_source=dataset is only allowed in simulation mode")
        if not self.dataset_path:
            raise ValueError("scenario.dataset_path is required")

    def spin(self):
        """手动主循环。

        runtime.step() 可能包含模型推理和渲染；手动循环能保证本帧发布完成后
        再按 world_state_hz 睡眠，便于观察真实闭环延迟。
        """
        rate = rospy.Rate(max(self.loop_hz, 1.0))
        while not rospy.is_shutdown():
            try:
                self.on_timer(None)
            except Exception as exc:
                stamp = rospy.Time.now()
                sim_time = max(time.time() - self.start_wall, 0.0)
                rospy.logerr(
                    "unhandled runtime error: %s\n%s",
                    exc,
                    traceback.format_exc(),
                )
                self.fail_runtime(
                    "unhandled_exception:%s" % exc,
                    stamp,
                    sim_time,
                )
                raise
            rate.sleep()

    def publish_map_info(self):
        """发布当前场景上下文；latch topic 会保留最近一次真实场景信息。"""
        self.map_pub.publish(String(data=json.dumps({
            "scenario_id": self.scenario_id,
            "map_path": self.scenario.get("map_path", ""),
            "source": self.scenario.get("source", "synthetic"),
            "coordinate": "world_xy_yaw_rad",
            "unitraj": self.unitraj.status,
            "platform_output": {
                "point_num": self.point_num,
                "dt_sec": self.output_dt,
                "horizon_sec": self.output_horizon_sec,
            },
        })))

    def on_ego_state_feedback(self, msg):
        """缓存 ego feedback；invalid 也必须保留，禁止回退到 MetaDrive pose。"""
        self.latest_ego_state_feedback = copy.deepcopy(msg)
        self.latest_ego_state_feedback_stamp = rospy.Time.now()

    def on_agents_state_feedback(
            self,
            msg: VehicleStateArray,
    ):
        """缓存 platform_comm 输出的受控平台实际状态。"""
        states = {}

        for state in msg.vehicles:
            if not state.vehicle_id:
                continue

            states[state.vehicle_id] = copy.deepcopy(
                state
            )

        self.latest_agent_state_feedback = states

    def get_fresh_agent_feedback(
            self,
            stamp,
            valid_only=False,
    ):
        """筛选仍在超时窗口内的 BV feedback。"""
        fresh = {}

        for vehicle_id, state in (
                self.latest_agent_state_feedback.items()
        ):
            state_stamp = state.header.stamp

            if state_stamp.to_sec() <= 0.0:
                continue

            age = max(
                (stamp - state_stamp).to_sec(),
                0.0,
            )

            if age > self.platform_feedback_timeout_sec:
                continue

            if valid_only and not state.valid:
                continue

            fresh[vehicle_id] = state

        return fresh

    def get_unconsumed_platform_feedback(self, feedback):
        """筛选本轮尚未被 env.step() 消费的真实平台反馈。

        只比较消息时间戳，不对重复样本做位置外推，也不使用 MetaDrive 状态补齐。
        """
        unconsumed = {}
        repeated = []
        for vehicle_id in sorted(self.controlled_vehicle_ids):
            state = feedback.get(vehicle_id)
            if state is None:
                continue
            stamp_ns = int(state.header.stamp.to_nsec())
            previous_ns = int(
                self.last_consumed_platform_feedback_stamp_ns.get(vehicle_id, -1)
            )
            if stamp_ns <= previous_ns:
                repeated.append(vehicle_id)
                continue
            unconsumed[vehicle_id] = state
        return unconsumed, repeated

    def commit_platform_feedback_consumption(self, feedback):
        """仅在 backend step 成功后提交已消费反馈时间戳。"""
        for vehicle_id, state in feedback.items():
            self.last_consumed_platform_feedback_stamp_ns[vehicle_id] = int(
                state.header.stamp.to_nsec()
            )

    def make_invalid_vehicle_state(
            self,
            template,
            entity_type,
    ):
        """保留 ID/尺寸，清空动态字段，避免 invalid 状态携带 MetaDrive 兜底。"""
        state = copy.deepcopy(template)
        state.header.stamp = rospy.Time.now()
        state.header.frame_id = self.frame_id
        state.entity_type = entity_type
        state.valid = False

        state.pose.position.x = 0.0
        state.pose.position.y = 0.0
        state.pose.position.z = 0.0
        state.pose.orientation.x = 0.0
        state.pose.orientation.y = 0.0
        state.pose.orientation.z = 0.0
        state.pose.orientation.w = 1.0

        state.twist.linear.x = 0.0
        state.twist.linear.y = 0.0
        state.twist.linear.z = 0.0
        state.twist.angular.x = 0.0
        state.twist.angular.y = 0.0
        state.twist.angular.z = 0.0

        state.accel.linear.x = 0.0
        state.accel.linear.y = 0.0
        state.accel.linear.z = 0.0
        state.accel.angular.x = 0.0
        state.accel.angular.y = 0.0
        state.accel.angular.z = 0.0
        state.heading = 0.0
        return state

    def merged_ego_state(self, sim_state, feedback_state):
        """Return simulation ego directly or external actual ego feedback."""
        if self.ego_execution_mode == "simulation":
            state = copy.deepcopy(sim_state)
            state.entity_type = "virtual"
            return state
        if feedback_state is not None:
            return copy.deepcopy(feedback_state)
        return self.make_invalid_vehicle_state(sim_state, "real")

    def merged_agent_state(self, sim_state, feedback_state):
        """Use MetaDrive for virtual BVs and actual feedback for platforms."""
        if sim_state.vehicle_id not in self.controlled_vehicle_ids:
            state = copy.deepcopy(sim_state)
            state.entity_type = "virtual"
            return state
        if self.emulated_platform_feedback:
            state = copy.deepcopy(sim_state)
            state.entity_type = "physical_platform"
            return state
        if feedback_state is not None:
            return copy.deepcopy(feedback_state)
        return self.make_invalid_vehicle_state(
            sim_state,
            "physical_platform",
        )

    def publish_status(self, stamp, sim_time, message_override=None):
        """Publish runtime health and terminal state."""
        scenario = ScenarioStatus()
        scenario.header.stamp = stamp
        scenario.scenario_id = self.scenario_id
        scenario.loaded = True

        if self.runtime_failure_reason:
            scenario.running = False
            scenario.message = "failed:%s" % self.runtime_failure_reason
        elif self.unitraj.finished_all_scenarios:
            scenario.running = False
            scenario.message = "finished"
        else:
            scenario.running = True
            scenario.message = message_override or "running"
        self.scenario_status_pub.publish(scenario)

        sim = SimulationStatus()
        sim.header.stamp = stamp
        sim.backend = self.unitraj.status.get(
            "backend", "unitraj_pluto_metadrive"
        )
        sim.running = scenario.running
        sim.sim_time = sim_time
        sim.real_time_factor = 1.0
        sim.message = scenario.message
        self.sim_status_pub.publish(sim)

    def fail_runtime(self, reason, stamp, sim_time):
        """Publish a terminal failure before shutting down the worker."""
        if self.runtime_failure_reason:
            return
        self.runtime_failure_reason = str(reason)
        rospy.logerr("runtime failed: %s", self.runtime_failure_reason)
        self.publish_status(stamp, sim_time)
        rospy.sleep(max(self.evaluation_finalize_delay_sec, 0.2))
        rospy.signal_shutdown("runtime failed: %s" % self.runtime_failure_reason)

    def wait_for_required_feedback(
            self,
            kind,
            missing,
            seen,
            last_valid_wall,
            startup_timeout,
            dropout_timeout,
            stamp,
            sim_time,
    ):
        """Freeze the model briefly, then fail instead of waiting forever."""
        now_wall = time.time()
        timeout = dropout_timeout if seen else startup_timeout
        reference = (
            last_valid_wall
            if seen and last_valid_wall is not None
            else self.feedback_wait_start_wall
        )
        elapsed = max(now_wall - reference, 0.0)
        detail = ",".join(missing) if isinstance(missing, (list, tuple)) else str(missing)
        message = "waiting_%s_feedback:%s" % (kind, detail)
        self.publish_status(stamp, sim_time, message_override=message)
        rospy.logwarn_throttle(1.0, "%s (%.2f/%.2fs)", message, elapsed, timeout)
        if elapsed > timeout:
            self.fail_runtime("%s_timeout:%s" % (kind, detail), stamp, sim_time)
        return True

    def on_timer(self, _event):
        """Advance one closed-loop step and publish raw and fused state."""
        stamp = rospy.Time.now()
        sim_time = max(time.time() - self.start_wall, 0.0)

        fresh_ego_feedback = None
        if self.ego_execution_mode == "external":
            if (
                self.latest_ego_state_feedback_stamp is not None
                and (
                    stamp - self.latest_ego_state_feedback_stamp
                ).to_sec() <= self.ego_feedback_timeout_sec
            ):
                fresh_ego_feedback = self.latest_ego_state_feedback
        valid_ego_feedback = (
            fresh_ego_feedback
            if fresh_ego_feedback is not None and fresh_ego_feedback.valid
            else None
        )

        use_actual_platform_feedback = (
            self.mode == "hybrid" and not self.emulated_platform_feedback
        )
        all_agent_feedback = (
            self.get_fresh_agent_feedback(stamp, valid_only=False)
            if use_actual_platform_feedback
            else {}
        )
        '''每一帧先检查反馈是否新鲜'''
        valid_agent_feedback = (
            self.get_fresh_agent_feedback(stamp, valid_only=True)
            if use_actual_platform_feedback
            else {}
        )

        now_wall = time.time()
        if self.hardware_platform_feedback:
            # 缺少有效反馈或仍是上一仿真帧已经消费过的样本时，均不推进 env.step()。
            missing = sorted(
                self.controlled_vehicle_ids
                - set(valid_agent_feedback.keys())
            )
            if missing and self.strict_actual_feedback:
                self.wait_for_required_feedback(
                    "platform",
                    missing,
                    self.platform_feedback_seen,
                    self.last_platform_valid_wall,
                    self.platform_startup_timeout_sec,
                    self.platform_dropout_timeout_sec,
                    stamp,
                    sim_time,
                )
                return

            valid_agent_feedback, repeated = (
                self.get_unconsumed_platform_feedback(valid_agent_feedback)
            )
            if repeated and self.strict_actual_feedback:
                self.wait_for_required_feedback(
                    "new_platform",
                    repeated,
                    self.platform_feedback_seen,
                    self.last_platform_valid_wall,
                    self.platform_startup_timeout_sec,
                    self.platform_dropout_timeout_sec,
                    stamp,
                    sim_time,
                )
                return

            if set(valid_agent_feedback.keys()) == self.controlled_vehicle_ids:
                self.platform_feedback_seen = True
                self.last_platform_valid_wall = now_wall

        if self.ego_execution_mode == "external":
            if valid_ego_feedback is not None:
                self.ego_feedback_seen = True
                self.last_ego_valid_wall = now_wall
            elif self.strict_external_ego_feedback:
                self.wait_for_required_feedback(
                    "ego",
                    [self.ego_cfg.get("id", "ego_001")],
                    self.ego_feedback_seen,
                    self.last_ego_valid_wall,
                    self.ego_startup_timeout_sec,
                    self.ego_dropout_timeout_sec,
                    stamp,
                    sim_time,
                )
                return

        previous_scenario_id = self.scenario_id
        snapshot = self.unitraj.step(
            valid_ego_feedback,
            valid_agent_feedback,
        ) # 来自更新后的 MetaDrive
        if snapshot is None:
            self.fail_runtime("backend_returned_no_snapshot", stamp, sim_time)
            return

        if self.hardware_platform_feedback:
            self.commit_platform_feedback_consumption(valid_agent_feedback)

        if snapshot.get("finished", False):
            self.publish_status(stamp, sim_time)
            if self.evaluation_finalize_delay_sec > 0.0:
                rospy.sleep(self.evaluation_finalize_delay_sec)
            rospy.signal_shutdown("all configured scenarios finished")
            return

        new_scenario_id = snapshot["scenario_id"]
        if str(new_scenario_id) != str(previous_scenario_id):
            fresh_ego_feedback = None
            valid_ego_feedback = None
            all_agent_feedback = {}
            valid_agent_feedback = {}
            self.latest_ego_state_feedback = None
            self.latest_ego_state_feedback_stamp = None
            self.latest_agent_state_feedback = {}
            self.last_consumed_platform_feedback_stamp_ns = {}
            self.platform_feedback_seen = False
            self.ego_feedback_seen = False
            self.last_platform_valid_wall = None
            self.last_ego_valid_wall = None
            self.feedback_wait_start_wall = time.time()
            rospy.loginfo(
                "Scenario changed: %s -> %s; cleared actual feedback",
                previous_scenario_id,
                new_scenario_id,
            )

        self.scenario_id = new_scenario_id
        if str(new_scenario_id) != str(previous_scenario_id):
            self.publish_map_info()

        sim_time = snapshot["sim_time"]
        sim_ego = snapshot["ego"]
        sim_agents = snapshot["agents"]
        background_agents = snapshot.get("background_agents", [])
        target_trajectories = snapshot["platform_targets"]

        self.ego_state_pub.publish(sim_ego)
        agents_msg = VehicleStateArray()
        agents_msg.header.stamp = stamp
        agents_msg.header.frame_id = self.frame_id
        agents_msg.vehicles = sim_agents
        self.agents_pub.publish(agents_msg)

        target_msg = TrajectoryArray()
        target_msg.header = agents_msg.header
        target_msg.trajectories = target_trajectories
        self.target_pub.publish(target_msg)

        world = WorldState()
        world.header = agents_msg.header
        world.scenario_id = self.scenario_id
        world.sim_time = sim_time
        world.ego = self.merged_ego_state(sim_ego, fresh_ego_feedback)
        world.agents = [
            self.merged_agent_state(
                sim_state,
                all_agent_feedback.get(sim_state.vehicle_id),
            )
            for sim_state in sim_agents
        ] + [copy.deepcopy(state) for state in background_agents]

        self.world_pub.publish(world)
        self.publish_status(stamp, sim_time)


class UniTrajPlutoBackend:
    """UniTraj closeloop_sim.py + PlutoPolicy 的 ROS 适配层。

    对外保持 world 坐标：
    - x/y 是 MetaDrive 世界坐标米；
    - heading 是 yaw 弧度；
    - 只有 platform_comm_node 会在 UDP 边界做 AutoScene A 坐标与平台大地投影
      G 坐标之间的固定标定变换。
    """

    def __init__(
            self,
            scenario_cfg,
            vehicles_cfg,
            frame_id,
            algorithm_cfg=None,
            platform_cfg=None,
            runtime_cfg=None,
            ego_cfg=None,
            evaluation_cfg=None,
    ):
        self.scenario_cfg = scenario_cfg
        self.vehicles_cfg = vehicles_cfg

        self.controlled_vehicle_ids = {
            str(agent.get("id"))
            for agent in self.vehicles_cfg.get(
                "agents",
                [],
            )
            if is_platform_controlled_agent(agent)
        }

        self.algorithm_cfg = algorithm_cfg or {}
        self.platform_cfg = platform_cfg or {}
        self.frame_id = frame_id
        self.env = None

        self.closeloop_module = None
        self.start_scenario_index = int(self.scenario_cfg.get("scenario_index", 0))
        self.end_scenario_index = self.start_scenario_index + int(self.scenario_cfg.get("eval_eps", 1))

        self.step_index = 0
        self.object_map = {}
        self.output_point_num = int(self.platform_cfg.get("packet_point_num", 50))
        self.output_dt = float(self.platform_cfg.get("trajectory_dt_sec", 0.1))

        self.history_steps = int(self.algorithm_cfg.get("pluto_history_steps", 21))
        self.history_dt = float(self.algorithm_cfg.get("pluto_history_dt_sec", 0.1))
        self.plot_history_trajectories_enabled = bool(
            self.algorithm_cfg.get("plot_history_trajectories", True)
        )
        self.history_plot_include_background = bool(
            self.algorithm_cfg.get("history_plot_include_background", False)
        )
        configured_history_plot_dir = str(
            self.algorithm_cfg.get("history_plot_dir", "")
        ).strip()
        self.history_plot_dir = (
            (
                configured_history_plot_dir
                if os.path.isabs(configured_history_plot_dir)
                else os.path.join(self.autoscene_root(), configured_history_plot_dir)
            )
            if configured_history_plot_dir
            else os.path.join(self.autoscene_root(), "outputs", "history_plots")
        )
        self.history_plot_dir = os.path.abspath(self.history_plot_dir)
        self.dataset_history_for_plot = {}

        self.history_source = str(
            self.scenario_cfg.get("history_source", "dataset")
        ).strip().lower()

        if self.history_source not in ("dataset", "initial"):
            raise ValueError(
                "scenario.history_source must be dataset or initial, got %s"
                % self.history_source
            )
        self.dataset_type = str(
            self.scenario_cfg.get("dataset_type", "tjwest")
        ).strip().lower()
        self.dataset_path = str(
            self.scenario_cfg.get("dataset_path", "")
        ).strip()

        self.status = {
            "backend": "synthetic_metadrive_adapter",
            "active": False,
            "message": "UniTraj/Pluto backend not requested",
        }

        self.terminated = False
        self.termination_message = ""
        self.last_snapshot = None
        # scenario track id -> 完整启动加速轨迹；末点是用户 initial，末 21 点供 Pluto 输入。
        self.startup_trajectories_by_scenario_id = {}
        self.render_mode = str(self.scenario_cfg.get("render", os.environ.get("AUTOSCENE_RENDER", "none")))
        self.render_every_n_steps = int(self.scenario_cfg.get("render_every_n_steps", 1))
        self.save_render_png = bool(self.scenario_cfg.get("save_render_png", False))
        self.render_dir = os.path.abspath(
            self.scenario_cfg.get(
                "render_dir",
                "./outputs/render"
            )
        )

        self.render_save_every_n_steps = max(
            int(
                self.scenario_cfg.get(
                    "render_save_every_n_steps",
                    self.scenario_cfg.get("render_every_n_steps", 1),
                )
            ),
            1,
        )
        self._render_debug_logged = False
        self.scenario_index = int(self.scenario_cfg.get("scenario_index", 0))
        self.eval_eps = int(self.scenario_cfg.get("eval_eps", 1))
        self.auto_advance_scenario = bool(self.scenario_cfg.get("auto_advance_scenario", False))
        self.finished_all_scenarios = False
        self.active_scenario_id = None

        # 场景终止帧先完整发布旧场景 snapshot；
        # 下一次 step() 开始时再 reset 到下一个场景。
        self.pending_scenario_index = None

        self.runtime_cfg = runtime_cfg or {}
        self.mode = self.runtime_cfg.get("mode", "simulation")
        self.platform_feedback_timeout_sec = max(
            float(self.platform_cfg.get("feedback_timeout_sec", 0.5)),
            1e-3,
        )

        self.ego_cfg = ego_cfg or {}
        self.evaluation_cfg = evaluation_cfg or {}
        self.include_background_vehicles = bool(
            self.evaluation_cfg.get(
                "include_background_vehicles",
                True,
            )
        )

    def pluto_history_time_index(self):
        """Return the current dataset-aligned Pluto history index."""
        engine = getattr(self.env, "engine", None)
        return max(
            int(getattr(engine, "episode_step", self.step_index)) - 1,
            0,
        )

    def configure_pluto_cbvs(self):
        """把 YAML 中配置的全部 BV 固定为 Pluto 的 CBV。

        platform_id 只决定执行方式：
        - platform_id=-1：MetaDrive 执行；
        - 有效 platform_id：运动平台执行。

        两类 BV 都仍然是 Pluto 需要生成轨迹的 CBV。
        """
        policy = self.default_policy()

        scenario = self.env.engine.data_manager.current_scenario
        tracks = scenario.get("tracks", {})

        cbv_ids = []

        for cfg in self.vehicles_cfg.get("agents", []):
            vehicle_id = str(cfg.get("id", ""))
            scenario_id = cfg.get("scenario_track_id")

            if not scenario_id:
                raise RuntimeError(
                    "vehicle %s has no scenario_track_id"
                    % vehicle_id
                )

            scenario_id = str(scenario_id)

            if scenario_id not in tracks:
                raise RuntimeError(
                    "vehicle %s scenario_track_id=%s is not in current scenario tracks"
                    % (vehicle_id, scenario_id)
                )

            cbv_ids.append(scenario_id)

        if len(cbv_ids) < 1:
            raise RuntimeError(
                "at least one configured CBV is required"
            )

        planner_modes = {}
        external_history_ids = set()
        for cfg, scenario_id in zip(self.vehicles_cfg.get("agents", []), cbv_ids):
            planner_mode = str(cfg.get("planner_mode", "max")).strip().lower()
            if planner_mode not in ("adversarial", "max"):
                raise ValueError(
                    "vehicle %s planner_mode must be adversarial or max"
                    % cfg.get("id", scenario_id)
                )
            planner_modes[str(scenario_id)] = planner_mode
            if (
                is_platform_controlled_agent(cfg)
                and self.platform_cfg.get("receive_udp", False)
                and not self.platform_cfg.get("emulate_controlled_feedback", False)
            ):
                external_history_ids.add(str(scenario_id))

        policy.current_cbv_ids = list(cbv_ids)
        policy.cbv_selection_locked = True
        policy.max_cbv_num = len(cbv_ids)
        policy.planner_mode_by_scenario_id = planner_modes
        policy.external_history_owned_ids = external_history_ids
        # 真实硬件平台只由 runtime 写入的 actual feedback 执行；Pluto 只负责规划。
        policy.external_execution_owned_ids = set(external_history_ids)
        if hasattr(policy, "install_external_feedback_policies"):
            policy.install_external_feedback_policies()
        policy.external_ego_control = (
            str(self.ego_cfg.get("execution_mode", "simulation")).strip().lower()
            == "external"
        )
        policy.platform_output_point_num = self.output_point_num

        if hasattr(policy, "_controlled_cbv_scenario_ids"):
            policy._controlled_cbv_scenario_ids.update(cbv_ids)

        rospy.loginfo(
            "Configured Pluto CBVs: ids=%s planner_modes=%s external_history=%s external_execution=%s",
            cbv_ids,
            planner_modes,
            sorted(external_history_ids),
            sorted(external_history_ids),
        )

    def render_frame(self, snapshot=None):
        """按 closeloop_sim.py standalone 路径执行 2D/3D 渲染。

        这里不做 dataset/schema 适配，不改 renderer 逻辑。
        2D 时只使用 reset_scenario() 挂上的 env.head_renderer。
        """
        render_mode = str(getattr(self, "render_mode", "none")).lower()

        if render_mode in ("none", "false", "0", ""):
            if not getattr(self, "_render_debug_logged", False):
                rospy.logwarn("Render disabled: render_mode=%s", render_mode)
                self._render_debug_logged = True
            return

        if self.render_every_n_steps > 1 and self.step_index % self.render_every_n_steps != 0:
            return

        if not getattr(self, "_render_debug_logged", False):
            engine = getattr(self.env, "engine", None)
            rospy.loginfo(
                "Render active: mode=%s, every_n=%d, save_png=%s, render_dir=%s, "
                "save_every_n=%d, DISPLAY=%s, "
                "env.head_renderer=%s, engine.head_renderer=%s, engine=%s",
                render_mode,
                self.render_every_n_steps,
                self.save_render_png,
                self.render_dir if self.save_render_png else "",
                self.render_save_every_n_steps,
                os.environ.get("DISPLAY", ""),
                type(getattr(self.env, "head_renderer", None)).__name__
                if getattr(self.env, "head_renderer", None) is not None else None,
                type(getattr(engine, "head_renderer", None)).__name__
                if engine is not None and getattr(engine, "head_renderer", None) is not None else None,
                type(engine).__name__ if engine is not None else None,
            )
            self._render_debug_logged = True

        if render_mode in ("2d", "topdown"):
            try:
                engine = getattr(self.env, "engine", None)
                renderer = getattr(self.env, "head_renderer", None)
                engine_renderer = getattr(engine, "head_renderer", None) if engine is not None else None

                if renderer is None and engine_renderer is not None:
                    renderer = engine_renderer
                    self.env.head_renderer = renderer

                if renderer is None:
                    rospy.logwarn_throttle(
                        2.0,
                        "Topdown renderer is None: env.head_renderer=%s, engine.head_renderer=%s, engine=%s",
                        getattr(self.env, "head_renderer", None),
                        engine_renderer,
                        type(engine).__name__ if engine is not None else None,
                    )
                    return

                text = {
                    "step": self.step_index,
                    "mode": "AutoScene-Pluto",
                }
                try:
                    scenario = self.env.engine.data_manager.current_scenario
                    text["id"] = scenario.get(
                        "id",
                        scenario.get("metadata", {}).get("scenario_id", "unknown"),
                    )
                except Exception:
                    pass

                if snapshot is not None:
                    text["sim_time"] = round(float(snapshot.get("sim_time", 0.0)), 2)

                # 对齐 closeloop_sim.py standalone：
                # head_renderer.render(screen_record=False, show_plan_traj=True,
                #                      scaling=6, film_size=(6000, 4000), mode="topdown", text=...)
                renderer.render(
                    screen_record=False,
                    show_plan_traj=True,
                    scaling=6,
                    film_size=(6000, 4000),
                    mode="topdown",
                    text=text,
                )

                try:
                    import pygame

                    # 有真实显示窗口时刷新窗口；无窗口或 SDL dummy 模式下不影响 PNG 保存。
                    display_surface = pygame.display.get_surface()
                    if display_surface is not None:
                        pygame.event.pump()
                        pygame.display.flip()

                    if (
                            self.save_render_png
                            and self.step_index % self.render_save_every_n_steps == 0
                    ):
                        os.makedirs(
                            self.render_dir,
                            exist_ok=True,
                        )

                        # 不要使用 pygame.display.get_surface() 保存。
                        # 真正完成 MetaDrive 地图、车辆、轨迹和参考线绘制的是
                        # HeadTopDownRenderer 内部 TopDownRenderer 的 screen_canvas。
                        topdown_renderer = getattr(
                            renderer,
                            "head_top_down_renderer",
                            None,
                        )

                        if topdown_renderer is None:
                            raise RuntimeError(
                                "head_top_down_renderer is None after renderer.render()"
                            )

                        surface = getattr(
                            topdown_renderer,
                            "screen_canvas",
                            None,
                        )

                        if surface is None:
                            surface = getattr(
                                topdown_renderer,
                                "_screen_canvas",
                                None,
                            )

                        if surface is None:
                            raise RuntimeError(
                                "TopDownRenderer screen canvas is None"
                            )

                        frame_path = os.path.join(
                            self.render_dir,
                            "render_%06d.png" % int(self.step_index),
                        )

                        pygame.image.save(
                            surface,
                            frame_path,
                        )

                        rospy.loginfo(
                            "Saved topdown render frame: %s",
                            frame_path,
                        )

                except Exception:
                    rospy.logwarn_throttle(
                        2.0,
                        "Failed to refresh pygame display or save render PNG:\n%s",
                        traceback.format_exc(),
                    )

            except Exception:
                rospy.logwarn_throttle(
                    2.0,
                    "Topdown render failed:\n%s",
                    traceback.format_exc(),
                )

    def get_render_camera_position(self, snapshot=None):
        """返回 2D topdown renderer 的相机中心。

        优先使用 MetaDrive default_agent 的当前位置。
        如果取不到，再退回 snapshot ego pose。
        """
        try:
            engine = getattr(self.env, "engine", None)
            agents = getattr(engine, "agents", {}) if engine is not None else {}
            default_agent = agents.get("default_agent", None)
            if default_agent is not None and hasattr(default_agent, "position"):
                pos = default_agent.position
                return (float(pos[0]), float(pos[1]))
        except Exception:
            pass

        try:
            if snapshot is not None:
                ego = snapshot.get("ego", None)
                if ego is not None:
                    return (
                        float(ego.pose.position.x),
                        float(ego.pose.position.y),
                    )
        except Exception:
            pass

        return None

    def initialize(self):
        """按配置启动 UniTraj/MetaDrive/Pluto 后端。

        closeloop_sim.py is imported without argparse or cwd side effects.
        Runtime passes the generator config path directly to create_env().
        """
        backend = self.scenario_cfg.get("backend", self.scenario_cfg.get("source", "synthetic"))
        if str(backend).lower() not in (
                "unitraj",
                "unitraj_pluto",
                "pluto",
                "metadrive",
        ):
            raise RuntimeError(
                "unsupported scenario backend: %s"
                % backend
            )

        unitraj_root = self.resolve_path(self.scenario_cfg.get("unitraj_root") or self.default_unitraj_root())
        config_path = self.scenario_cfg.get("unitraj_config") or os.path.join(
            unitraj_root, "unitraj", "configs", "config.yaml"
        )
        config_path = self.resolve_path(config_path)
        if not os.path.exists(os.path.join(unitraj_root, "unitraj", "closeloop_sim.py")):
            self.status = {
                "backend": "unitraj_pluto",
                "active": False,
                "message": "UniTraj closeloop_sim.py not found: %s" % unitraj_root,
            }
            raise RuntimeError(
                self.status["message"]
            )

        try:
            os.environ["AUTOSCENE_DATASET_TYPE"] = self.dataset_type
            if self.dataset_path:
                os.environ["AUTOSCENE_DATASET_PATH"] = self.resolve_path(self.dataset_path)

            if unitraj_root not in sys.path:
                sys.path.insert(0, unitraj_root)

            module = self.import_closeloop(unitraj_root, config_path)
            self.closeloop_module = module

            start_index = int(self.scenario_cfg.get("scenario_index", 0))
            eval_eps = int(self.scenario_cfg.get("eval_eps", 1))
            self.start_scenario_index = start_index
            self.end_scenario_index = start_index + eval_eps

            # ScenarioEnv needs all seeds up to the requested range.
            num_scenarios = max(eval_eps, start_index + eval_eps)
            self.env = module.create_env(num_scenarios, config_path)
            self.scenario_index = start_index
            self.eval_eps = eval_eps
            self.reset_scenario(self.scenario_index)

            self.status = {
                "backend": "unitraj_pluto_metadrive",
                "active": True,
                "message": "UniTraj Pluto backend active, config=%s" % config_path,
            }
            rospy.loginfo(self.status["message"])
            return True

        except Exception as exc:
            self.status = {
                "backend": "unitraj_pluto_metadrive",
                "active": False,
                "message": (
                        "UniTraj Pluto backend failed: %s"
                        % exc
                ),
            }

            rospy.logerr(
                "%s\n%s",
                self.status["message"],
                traceback.format_exc(),
            )

            raise RuntimeError(
                self.status["message"]
            ) from exc

    @staticmethod
    def _history_tracks_by_id(scenario):
        """Normalize ScenarioNet track keys so int/string IDs both work."""
        tracks = scenario.get("tracks", {}) if isinstance(scenario, dict) else {}
        if not isinstance(tracks, dict):
            return {}
        return {str(track_id): track for track_id, track in tracks.items()}

    def _history_actor_specs(self, scenario):
        tracks = self._history_tracks_by_id(scenario)
        metadata = scenario.get("metadata", {}) if isinstance(scenario, dict) else {}
        sdc_id = str(metadata.get("sdc_id", "")).strip()
        specs = []
        seen = set()
        if sdc_id and sdc_id in tracks:
            specs.append(("ego", sdc_id))
            seen.add(sdc_id)

        for cfg in self.vehicles_cfg.get("agents", []):
            label = str(cfg.get("id", "agent"))
            scenario_id = str(cfg.get("scenario_track_id", "")).strip()
            if scenario_id and scenario_id in tracks and scenario_id not in seen:
                specs.append((label, scenario_id))
                seen.add(scenario_id)

        if self.history_plot_include_background:
            for scenario_id in sorted(tracks):
                if scenario_id not in seen:
                    specs.append(("background_%s" % scenario_id, scenario_id))
        return specs

    def _track_history_entries(self, track, present_idx, count):
        import numpy as np

        state = track.get("state", {}) if isinstance(track, dict) else {}
        position = np.asarray(state.get("position", []), dtype=np.float64)
        heading = np.asarray(state.get("heading", []), dtype=np.float64).reshape(-1)
        velocity = np.asarray(state.get("velocity", []), dtype=np.float64)
        acceleration = np.asarray(
            state.get("acceleration", np.zeros_like(velocity)),
            dtype=np.float64,
        )
        valid = np.asarray(state.get("valid", []), dtype=bool).reshape(-1)

        if position.ndim != 2 or position.shape[1] < 2 or len(position) == 0:
            return []
        present_idx = int(np.clip(int(present_idx), 0, len(position) - 1))
        start_idx = max(0, present_idx - int(count) + 1)
        entries = []
        for index in range(start_idx, present_idx + 1):
            if index >= len(heading) or index >= len(velocity):
                break
            entries.append(
                {
                    "track_index": int(index),
                    "position": np.asarray(position[index, :2], dtype=np.float64),
                    "heading": float(heading[index]),
                    "velocity": np.asarray(velocity[index, :2], dtype=np.float64),
                    "acceleration": (
                        np.asarray(acceleration[index, :2], dtype=np.float64)
                        if acceleration.ndim == 2 and index < len(acceleration)
                        else np.zeros(2, dtype=np.float64)
                    ),
                    "valid": bool(valid[index]) if index < len(valid) else True,
                }
            )
        return entries

    def capture_dataset_history_for_plot(self):
        """Capture the untouched dataset window before initial-mode overwrites."""
        engine = getattr(self.env, "engine", None)
        scenario = (
            getattr(getattr(engine, "data_manager", None), "current_scenario", {})
            if engine else {}
        )
        tracks = self._history_tracks_by_id(scenario)
        policy = self.default_policy()
        present_idx = int(
            getattr(policy, "WARMUP_STEPS", max(self.history_steps - 1, 20))
        )
        captured = {}
        for label, scenario_id in self._history_actor_specs(scenario):
            track = tracks.get(str(scenario_id))
            if track is None:
                continue
            entries = self._track_history_entries(
                track,
                present_idx=present_idx,
                count=self.history_steps,
            )
            if entries:
                captured[str(scenario_id)] = {
                    "label": label,
                    "entries": entries,
                }
        self.dataset_history_for_plot = captured
        rospy.loginfo(
            "Captured dataset history: source=%s actors=%d samples=%s",
            self.history_source,
            len(captured),
            {
                scenario_id: len(payload["entries"])
                for scenario_id, payload in captured.items()
            },
        )

    def _serializable_history_entry(self, entry):
        return {
            "track_index": int(entry.get("track_index", -1)),
            "x": float(entry["position"][0]),
            "y": float(entry["position"][1]),
            "heading": float(entry["heading"]),
            "vx": float(entry["velocity"][0]),
            "vy": float(entry["velocity"][1]),
            "ax": float(entry["acceleration"][0]),
            "ay": float(entry["acceleration"][1]),
            "valid": bool(entry.get("valid", True)),
        }

    def plot_history_trajectories(self, scenario_index):
        """Write XY/speed/acceleration plots plus JSON/CSV for dataset and initial."""
        if not self.plot_history_trajectories_enabled:
            return

        try:
            import matplotlib
            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
        except Exception as exc:
            raise RuntimeError(
                "algorithm.plot_history_trajectories requires matplotlib"
            ) from exc

        os.makedirs(self.history_plot_dir, exist_ok=True)
        engine = getattr(self.env, "engine", None)
        scenario = getattr(getattr(engine, "data_manager", None), "current_scenario", {}) if engine else {}
        scenario_name = str(
            scenario.get("id", scenario.get("metadata", {}).get("scenario_id", scenario_index))
        )
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", scenario_name)
        prefix = os.path.join(
            self.history_plot_dir,
            "scenario_%04d_%s_%s"
            % (int(scenario_index), safe_name, self.history_source),
        )

        series = []
        for scenario_id, payload in self.dataset_history_for_plot.items():
            series.append(
                {
                    "source": "dataset",
                    "label": payload["label"],
                    "scenario_id": scenario_id,
                    "entries": copy.deepcopy(payload["entries"]),
                    "window_start": 0,
                }
            )

        if self.history_source == "initial":
            label_by_id = {
                scenario_id: label
                for label, scenario_id in self._history_actor_specs(scenario)
            }
            for scenario_id, startup in self.startup_trajectories_by_scenario_id.items():
                entries = []
                for index, entry in enumerate(startup):
                    item = copy.deepcopy(entry)
                    item["track_index"] = int(index)
                    entries.append(item)
                series.append(
                    {
                        "source": "initial",
                        "label": label_by_id.get(str(scenario_id), str(scenario_id)),
                        "scenario_id": str(scenario_id),
                        "entries": entries,
                        "window_start": max(0, len(entries) - int(self.history_steps)),
                    }
                )

        if not series:
            rospy.logwarn(
                "No history samples available for plotting: "
                "scenario_index=%d source=%s output_dir=%s",
                int(scenario_index),
                self.history_source,
                self.history_plot_dir,
            )

        payload = {
            "scenario_index": int(scenario_index),
            "scenario_id": scenario_name,
            "history_source": self.history_source,
            "history_steps": int(self.history_steps),
            "history_dt_sec": float(self.history_dt),
            "series": [
                {
                    "source": item["source"],
                    "label": item["label"],
                    "scenario_id": item["scenario_id"],
                    "window_start": int(item["window_start"]),
                    "entries": [
                        self._serializable_history_entry(entry)
                        for entry in item["entries"]
                    ],
                }
                for item in series
            ],
        }
        with open(prefix + "_history.json", "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)

        with open(prefix + "_history.csv", "w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "source", "label", "scenario_id", "sample_index", "track_index",
                    "is_pluto_window", "time_to_present_sec",
                    "x", "y", "heading", "vx", "vy", "speed", "ax", "ay", "accel", "valid",
                ]
            )
            for item in series:
                entries = item["entries"]
                window_start = int(item["window_start"])
                for sample_index, entry in enumerate(entries):
                    speed = float(np.linalg.norm(entry["velocity"]))
                    accel = float(np.linalg.norm(entry["acceleration"]))
                    writer.writerow(
                        [
                            item["source"],
                            item["label"],
                            item["scenario_id"],
                            sample_index,
                            int(entry.get("track_index", sample_index)),
                            sample_index >= window_start,
                            (sample_index - len(entries) + 1) * float(self.history_dt),
                            float(entry["position"][0]),
                            float(entry["position"][1]),
                            float(entry["heading"]),
                            float(entry["velocity"][0]),
                            float(entry["velocity"][1]),
                            speed,
                            float(entry["acceleration"][0]),
                            float(entry["acceleration"][1]),
                            accel,
                            bool(entry.get("valid", True)),
                        ]
                    )

        def valid_arrays(item):
            entries = [entry for entry in item["entries"] if bool(entry.get("valid", True))]
            if not entries:
                return None
            xy = np.asarray([entry["position"] for entry in entries], dtype=np.float64)
            speed = np.asarray(
                [np.linalg.norm(entry["velocity"]) for entry in entries],
                dtype=np.float64,
            )
            accel = np.asarray(
                [np.linalg.norm(entry["acceleration"]) for entry in entries],
                dtype=np.float64,
            )
            time_axis = (
                np.arange(len(entries), dtype=np.float64) - len(entries) + 1
            ) * float(self.history_dt)
            return entries, xy, speed, accel, time_axis

        plt.figure(figsize=(10, 8))
        for item in series:
            arrays = valid_arrays(item)
            if arrays is None:
                continue
            _entries, xy, _speed, _accel, _time_axis = arrays
            legend = "%s:%s" % (item["label"], item["source"])
            plt.plot(xy[:, 0], xy[:, 1], marker="o", markersize=2, label=legend)
            plt.scatter([xy[-1, 0]], [xy[-1, 1]], marker="x", s=50)
        plt.axis("equal")
        plt.xlabel("AutoScene world x [m]")
        plt.ylabel("AutoScene world y [m]")
        plt.title("Pluto history XY: %s" % scenario_name)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(prefix + "_history_xy.png", dpi=180)
        plt.close()

        plt.figure(figsize=(10, 6))
        for item in series:
            arrays = valid_arrays(item)
            if arrays is None:
                continue
            _entries, _xy, speed, _accel, time_axis = arrays
            plt.plot(time_axis, speed, marker="o", markersize=2,
                     label="%s:%s" % (item["label"], item["source"]))
        plt.xlabel("Time to present [s]")
        plt.ylabel("Speed [m/s]")
        plt.title("Pluto history speed: %s" % scenario_name)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(prefix + "_history_speed.png", dpi=180)
        plt.close()

        plt.figure(figsize=(10, 6))
        for item in series:
            arrays = valid_arrays(item)
            if arrays is None:
                continue
            _entries, _xy, _speed, accel, time_axis = arrays
            plt.plot(time_axis, accel, marker="o", markersize=2,
                     label="%s:%s" % (item["label"], item["source"]))
        plt.xlabel("Time to present [s]")
        plt.ylabel("Acceleration magnitude [m/s²]")
        plt.title("Pluto history acceleration: %s" % scenario_name)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(prefix + "_history_acceleration.png", dpi=180)
        plt.close()

        rospy.loginfo(
            "History plots written: prefix=%s sources=%s actors=%d",
            prefix,
            sorted(set(item["source"] for item in series)),
            len(series),
        )

    def reset_scenario(self, scenario_index):
        """Reset MetaDrive/UniTraj 到指定 scenario index。

        严格对齐 closeloop_sim.py standalone 路径：
        clear_render_overlays(env)
        env.reset(seed=i_ep)
        clear_render_overlays(env)
        env.head_renderer = HeadTopDownRenderer(env)
        env.engine.head_renderer = env.head_renderer
        clear_render_overlays(env)
        """
        self.terminated = False
        self.termination_message = ""
        self.last_snapshot = None
        self.step_index = 0
        self.object_map = {}
        module = getattr(self, "closeloop_module", None)
        if module is None:
            raise RuntimeError("closeloop_module is None; initialize() did not import closeloop_sim.py")

        if hasattr(module, "clear_render_overlays"):
            module.clear_render_overlays(self.env)

        self.env.reset(seed=int(scenario_index))

        if hasattr(module, "clear_render_overlays"):
            module.clear_render_overlays(self.env)

        if str(getattr(self, "render_mode", "none")).lower() in ("2d", "topdown"):
            if not hasattr(module, "HeadTopDownRenderer"):
                raise RuntimeError("closeloop_sim.py does not expose HeadTopDownRenderer")

            self.env.head_renderer = module.HeadTopDownRenderer(self.env)
            self.env.engine.head_renderer = self.env.head_renderer

            if hasattr(module, "clear_render_overlays"):
                module.clear_render_overlays(self.env)

            rospy.logwarn(
                "HeadTopDownRenderer attached: env.head_renderer=%s, engine.head_renderer=%s",
                type(getattr(self.env, "head_renderer", None)).__name__
                if getattr(self.env, "head_renderer", None) is not None else None,
                type(getattr(getattr(self.env, "engine", None), "head_renderer", None)).__name__
                if getattr(self.env, "engine", None) is not None
                   and getattr(getattr(self.env, "engine", None), "head_renderer", None) is not None else None,
            )

        try:
            scenario = self.env.engine.data_manager.current_scenario
            sdc_id = scenario["metadata"]["sdc_id"]
            self.env.engine.agents["default_agent"].expert_traj = (
                scenario["tracks"][sdc_id]["state"]["position"]
            )
            scenario_id = str(
                scenario.get(
                    "id",
                    scenario.get("metadata", {}).get("scenario_id", "unknown"),
                )
            )
            self.active_scenario_id = scenario_id
            rospy.loginfo(
                "UniTraj scenario reset: index=%d, scenario_id=%s, sdc_id=%s",
                int(scenario_index),
                str(scenario_id),
                str(sdc_id),
            )
        except Exception:
            self.active_scenario_id = "scenario_%d" % int(scenario_index)
            rospy.logwarn("Failed to attach expert_traj after reset:\n%s", traceback.format_exc())
            rospy.loginfo("UniTraj scenario reset: index=%d", int(scenario_index))

        self.auto_assign_agents_for_current_scenario()
        self.object_map = self.build_object_map()
        self.startup_trajectories_by_scenario_id = {}
        rospy.loginfo(
            "History plotting configuration: enabled=%s source=%s dir=%s "
            "include_background=%s",
            self.plot_history_trajectories_enabled,
            self.history_source,
            self.history_plot_dir,
            self.history_plot_include_background,
        )
        self.capture_dataset_history_for_plot()
        if self.history_source == "initial":
            self.apply_initial_history_for_current_scenario()
        self.configure_pluto_cbvs()
        self.plot_history_trajectories(scenario_index)

    def default_unitraj_root(self):
        """默认 UniTraj 根目录。

        当前仓库把生成器放在 autoscene_runtime/scripts/Autoscene_generator。
        """
        candidates = [
            os.path.abspath(os.path.join(os.path.dirname(__file__), "Autoscene_generator")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..", "Autoscene_generator")),
        ]
        for candidate in candidates:
            if os.path.exists(os.path.join(candidate, "unitraj", "closeloop_sim.py")):
                return candidate
        return candidates[0]

    def autoscene_root(self):
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

    def resolve_path(self, path):
        """解析用户配置路径，兼容从项目根、AutoScene 根或当前目录启动 roslaunch。"""
        if not path or os.path.isabs(path):
            return path
        candidates = [
            os.path.abspath(path),
            os.path.abspath(os.path.join(self.autoscene_root(), path)),
            os.path.abspath(os.path.join(self.autoscene_root(), "..", path)),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return candidates[0]

    def import_closeloop(self, unitraj_root, config_path):
        """Import the generator module without changing argv or cwd."""
        import importlib.util

        module_path = os.path.join(unitraj_root, "unitraj", "closeloop_sim.py")
        spec = importlib.util.spec_from_file_location(
            "autoscene_unitraj_closeloop_sim",
            module_path,
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot import closeloop_sim.py: %s" % module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["autoscene_unitraj_closeloop_sim"] = module
        spec.loader.exec_module(module)
        return module

    def auto_assign_agents_for_current_scenario(self):
        """按当前 MetaDrive 场景自动选择 platform/virtual 周车。

        该逻辑等价于原“查看真实track_id.txt”的排序规则，但结果只写入
        runtime 内存配置：
        - 有 platform_id 的 agent 优先拿排序第 1 的候选；
        - 纯虚拟 agent 依次拿后续候选；
        自动模式只写 scenario_track_id；initial 初始状态由 experiment_manager
        的人工确认流程写入。
        """
        if not bool(self.scenario_cfg.get("auto_assign_agents", True)):
            return

        agents_cfg = self.vehicles_cfg.get("agents", [])

        if not agents_cfg:
            return

        candidates = self.sorted_bv_candidates_for_current_scenario()

        if len(candidates) < len(agents_cfg):
            raise RuntimeError(
                "auto_assign_agents requires %d BV candidates, got %d"
                % (len(agents_cfg), len(candidates))
            )

        platform_agents = [
            cfg
            for cfg in agents_cfg
            if is_platform_controlled_agent(cfg)
        ]

        virtual_agents = [
            cfg
            for cfg in agents_cfg
            if not is_platform_controlled_agent(cfg)
        ]

        assignment_order = platform_agents + virtual_agents

        for index, cfg in enumerate(assignment_order):
            candidate = candidates[index]
            cfg["scenario_track_id"] = candidate["track_id"]
            cfg["planner_mode"] = "adversarial" if index == 0 else "max"
            cfg.pop("metadrive_object_id", None)
            cfg.pop("track_id", None)
            rospy.loginfo(
                "Auto-assigned %s as %s: rank=%d track_id=%s "
                "x=%.3f y=%.3f heading=%.3f speed=%.3f "
                "rel_long=%.3f rel_lat=%.3f",
                cfg.get("id", "agent"),
                "platform" if is_platform_controlled_agent(cfg) else "virtual",
                index + 1,
                candidate["track_id"],
                candidate["x"],
                candidate["y"],
                candidate["heading"],
                candidate["speed"],
                candidate["relative_longitudinal"],
                candidate["relative_lateral"],
            )

    def sorted_bv_candidates_for_current_scenario(self):
        """解析并排序当前场景可用 BV 候选。

        排序规则与“查看真实track_id.txt”一致：
        ego 前方优先，横向越接近 ego 越优先，纵向越近越优先，
        总距离越近越优先，速度越高越优先。
        """
        import numpy as np

        engine = getattr(self.env, "engine", None)

        if engine is None:
            return []

        scenario = getattr(
            getattr(engine, "data_manager", None),
            "current_scenario",
            {},
        ) or {}

        if not isinstance(scenario, dict):
            return []

        tracks = scenario.get("tracks", {})
        metadata = scenario.get("metadata", {})
        sdc_id = str(metadata.get("sdc_id", ""))
        ego_obj = getattr(engine, "agents", {}).get("default_agent")

        if ego_obj is None:
            raise RuntimeError("cannot auto-assign BV: MetaDrive ego object is missing")

        ego_xy = np.asarray(getattr(ego_obj, "position", [0.0, 0.0]), dtype=float)[:2]
        ego_heading = float(getattr(ego_obj, "heading_theta", 0.0))
        ego_forward = np.array([math.cos(ego_heading), math.sin(ego_heading)], dtype=float)
        ego_left = np.array([-math.sin(ego_heading), math.cos(ego_heading)], dtype=float)

        raw_mapping = getattr(
            getattr(engine, "traffic_manager", None),
            "obj_id_to_scenario_id",
            {},
        ) or {}

        track_to_object = {
            str(track_id): str(object_id)
            for object_id, track_id in raw_mapping.items()
        }

        engine_objects = {
            str(object_id): obj
            for object_id, obj in self.engine_objects(engine).items()
        }

        candidates = []

        for track_id, track in tracks.items():
            track_id = str(track_id)

            if track_id == sdc_id:
                continue

            state = track.get("state", {}) if isinstance(track, dict) else {}
            position = np.asarray(state.get("position", []), dtype=float)

            if position.ndim < 2 or len(position) == 0:
                continue

            valid = np.asarray(
                state.get(
                    "valid",
                    np.ones(len(position), dtype=bool),
                )
            ).astype(bool).reshape(-1)

            indexes = np.flatnonzero(valid)

            if len(indexes) == 0 or int(indexes[0]) != 0:
                continue

            object_id = track_to_object.get(track_id)

            if object_id is None or engine_objects.get(object_id) is None:
                continue

            heading_array = np.asarray(
                state.get(
                    "heading",
                    np.zeros(len(position)),
                ),
                dtype=float,
            ).reshape(-1)

            velocity = np.asarray(
                state.get(
                    "velocity",
                    np.zeros((len(position), 2)),
                ),
                dtype=float,
            )

            x = float(position[0, 0])
            y = float(position[0, 1])
            heading = float(heading_array[0]) if len(heading_array) else 0.0

            if velocity.ndim >= 2 and len(velocity) > 0:
                vx = float(velocity[0, 0])
                vy = float(velocity[0, 1])
                speed = float(math.hypot(vx, vy))
            else:
                speed = 0.0

            relative_world = np.array([x, y], dtype=float) - ego_xy
            relative_longitudinal = float(np.dot(relative_world, ego_forward))
            relative_lateral = float(np.dot(relative_world, ego_left))
            relative_heading = wrap_rad(heading - ego_heading)
            distance_to_ego = float(np.linalg.norm(relative_world))

            if relative_longitudinal >= 0.0:
                position_group = 0
            elif relative_longitudinal >= -5.0:
                position_group = 1
            else:
                position_group = 2

            candidates.append(
                {
                    "track_id": track_id,
                    "type": str(track.get("type", "")) if isinstance(track, dict) else "",
                    "x": x,
                    "y": y,
                    "heading": heading,
                    "speed": speed,
                    "relative_longitudinal": relative_longitudinal,
                    "relative_lateral": relative_lateral,
                    "relative_heading": relative_heading,
                    "distance_to_ego": distance_to_ego,
                    "sort_key": (
                        position_group,
                        abs(relative_lateral),
                        abs(relative_longitudinal),
                        distance_to_ego,
                        -speed,
                    ),
                }
            )

        candidates.sort(key=lambda item: item["sort_key"])
        return candidates

    def parse_initial_pose(self, cfg, label):
        """读取用户确认的 initial pose，并校验固定历史窗口的速度上限。"""
        if not isinstance(cfg, dict):
            raise RuntimeError("%s initial config is required" % label)
        initial = cfg.get("initial")
        if not isinstance(initial, dict):
            raise RuntimeError("%s.initial is required when history_source=initial" % label)

        required = ("x", "y", "heading", "speed")
        missing = [key for key in required if key not in initial]
        if missing:
            raise RuntimeError("%s.initial missing fields: %s" % (label, ",".join(missing)))

        pose = {
            "x": float(initial["x"]),
            "y": float(initial["y"]),
            "heading": wrap_rad(float(initial["heading"])),
            "speed": float(initial["speed"]),
        }
        if pose["speed"] < 0.0:
            raise RuntimeError("%s.initial.speed must be non-negative" % label)
        if not all(math.isfinite(value) for value in pose.values()):
            raise RuntimeError("%s.initial contains non-finite value" % label)

        max_speed = self.initial_history_max_speed()
        if pose["speed"] > max_speed + 1e-9:
            raise RuntimeError(
                "%s.initial.speed=%.6f exceeds fixed %d-point history limit %.6f m/s "
                "(accel=%.6f m/s^2, dt=%.6f s)"
                % (
                    label,
                    pose["speed"],
                    int(self.history_steps),
                    max_speed,
                    float(self.algorithm_cfg.get("initial_history_accel_mps2", 2.0)),
                    float(self.history_dt),
                )
            )
        return pose

    def required_startup_steps(self, initial_pose):
        """固定历史窗口下，从 0 加速到 initial.speed 所需的采样点数。"""
        accel = float(self.algorithm_cfg.get("initial_history_accel_mps2", 2.0))
        dt = float(self.history_dt)
        if accel <= 0.0:
            raise RuntimeError("algorithm.initial_history_accel_mps2 must be > 0")
        if dt <= 0.0:
            raise RuntimeError("algorithm.pluto_history_dt_sec must be > 0")
        speed = float(initial_pose["speed"])
        accel_intervals = int(math.ceil(max(speed, 0.0) / (accel * dt) - 1e-12))
        return accel_intervals + 1

    def initial_history_max_speed(self):
        """返回固定历史窗口从 0 起步、按给定加速度可达到的最大速度。"""
        steps = int(self.history_steps)
        dt = float(self.history_dt)
        accel = float(self.algorithm_cfg.get("initial_history_accel_mps2", 2.0))
        if steps < 2:
            raise RuntimeError("algorithm.pluto_history_steps must be >= 2")
        if dt <= 0.0:
            raise RuntimeError("algorithm.pluto_history_dt_sec must be > 0")
        if accel <= 0.0:
            raise RuntimeError("algorithm.initial_history_accel_mps2 must be > 0")
        return accel * dt * float(steps - 1)

    def build_accel_trajectory(self, initial_pose, total_steps=None):
        """构造固定长度 initial 历史：从 0 加速，达到目标速度后匀速。

        历史点数严格等于 algorithm.pluto_history_steps。第 0 点速度为 0，
        最后一点的位置、航向和速度严格等于用户确认的 initial 状态。
        """
        import numpy as np

        steps = int(self.history_steps)
        if steps < 2:
            raise RuntimeError(
                "algorithm.pluto_history_steps must be >= 2, got %d" % steps
            )
        if total_steps is not None and int(total_steps) != steps:
            raise RuntimeError(
                "initial history length is fixed at %d points; requested %d"
                % (steps, int(total_steps))
            )

        dt = float(self.history_dt)
        accel = float(self.algorithm_cfg.get("initial_history_accel_mps2", 2.0))
        if dt <= 0.0:
            raise RuntimeError("algorithm.pluto_history_dt_sec must be > 0")
        if accel <= 0.0:
            raise RuntimeError("algorithm.initial_history_accel_mps2 must be > 0")

        x_final = float(initial_pose["x"])
        y_final = float(initial_pose["y"])
        heading = float(initial_pose["heading"])
        target_speed = float(initial_pose["speed"])
        max_speed = accel * dt * float(steps - 1)
        if target_speed < 0.0:
            raise RuntimeError("initial speed must be non-negative")
        if target_speed > max_speed + 1e-9:
            raise RuntimeError(
                "initial speed %.6f exceeds fixed %d-point history limit %.6f m/s"
                % (target_speed, steps, max_speed)
            )

        direction = np.array(
            [math.cos(heading), math.sin(heading)],
            dtype=np.float64,
        )
        accel_time = target_speed / accel if target_speed > 0.0 else 0.0
        total_time = dt * float(steps - 1)

        def traveled_distance(t):
            t = float(max(t, 0.0))
            if target_speed <= 0.0:
                return 0.0
            if t <= accel_time:
                return 0.5 * accel * t * t
            return (
                0.5 * accel * accel_time * accel_time
                + target_speed * (t - accel_time)
            )

        total_distance = traveled_distance(total_time)
        start_position = (
            np.array([x_final, y_final], dtype=np.float64)
            - total_distance * direction
        )

        entries = []
        for index in range(steps):
            elapsed = float(index) * dt
            speed = min(accel * elapsed, target_speed)
            distance = traveled_distance(elapsed)
            accelerating = target_speed > 0.0 and elapsed < accel_time - 1e-12
            entries.append(
                {
                    "position": start_position + distance * direction,
                    "heading": heading,
                    "velocity": speed * direction,
                    "acceleration": (
                        accel * direction
                        if accelerating
                        else np.zeros(2, dtype=np.float64)
                    ),
                    "valid": True,
                }
            )

        entries[0]["velocity"] = np.zeros(2, dtype=np.float64)
        entries[-1]["position"] = np.array(
            [x_final, y_final],
            dtype=np.float64,
        )
        entries[-1]["velocity"] = target_speed * direction

        if len(entries) != steps:
            raise RuntimeError("constructed initial history length mismatch")
        if np.linalg.norm(entries[0]["velocity"]) > 1e-12:
            raise RuntimeError("constructed initial history does not start at zero speed")
        if (
            np.linalg.norm(
                entries[-1]["position"]
                - np.array([x_final, y_final], dtype=np.float64)
            ) > 1e-9
            or abs(float(np.linalg.norm(entries[-1]["velocity"])) - target_speed) > 1e-9
        ):
            raise RuntimeError("constructed initial history does not end at initial state")
        return entries

    def build_accel_history(self, initial_pose):
        """返回固定长度的 initial 历史窗口。"""
        return self.build_accel_trajectory(initial_pose)

    def patch_track_history(self, track, history, label):
        """把 track 前 N 帧严格替换为启动历史，包括 acceleration。"""
        import numpy as np

        if not isinstance(track, dict) or "state" not in track:
            raise RuntimeError("%s track has no state" % label)
        state = track["state"]
        length = len(history)

        position = np.asarray(state.get("position", []), dtype=np.float64)
        heading = np.asarray(state.get("heading", []), dtype=np.float64).reshape(-1)
        velocity = np.asarray(state.get("velocity", []), dtype=np.float64)
        valid = np.asarray(state.get("valid", []), dtype=bool).reshape(-1)

        if position.ndim != 2 or position.shape[0] < length or position.shape[1] < 2:
            raise RuntimeError("%s track position length is shorter than initial history" % label)
        if heading.shape[0] < length:
            raise RuntimeError("%s track heading length is shorter than initial history" % label)
        if velocity.ndim != 2 or velocity.shape[0] < length or velocity.shape[1] < 2:
            raise RuntimeError("%s track velocity length is shorter than initial history" % label)
        if valid.shape[0] < length:
            valid = np.pad(valid, (0, position.shape[0] - valid.shape[0]), constant_values=False)

        raw_acceleration = state.get("acceleration")
        if raw_acceleration is None:
            acceleration = np.zeros_like(velocity, dtype=np.float64)
        else:
            acceleration = np.asarray(raw_acceleration, dtype=np.float64)
            if acceleration.ndim != 2 or acceleration.shape[0] < length or acceleration.shape[1] < 2:
                raise RuntimeError(
                    "%s track acceleration shape is incompatible with initial history" % label
                )

        position = np.array(position, copy=True)
        heading = np.array(heading, copy=True)
        velocity = np.array(velocity, copy=True)
        acceleration = np.array(acceleration, copy=True)
        valid = np.array(valid, copy=True)

        for index, entry in enumerate(history):
            position[index, :2] = entry["position"][:2]
            heading[index] = entry["heading"]
            velocity[index, :2] = entry["velocity"][:2]
            acceleration[index, :2] = entry["acceleration"][:2]
            valid[index] = True

        state["position"] = position
        state["heading"] = heading
        state["velocity"] = velocity
        state["acceleration"] = acceleration
        state["valid"] = valid

    def set_object_to_history_entry(self, obj, entry):
        """把已创建的 MetaDrive 对象摆到启动历史第一帧。"""
        if obj is None:
            return
        try:
            obj.set_position([
                float(entry["position"][0]),
                float(entry["position"][1]),
            ])
            obj.set_velocity([
                float(entry["velocity"][0]),
                float(entry["velocity"][1]),
            ])
            obj.set_heading_theta(float(entry["heading"]))
        except Exception as exc:
            raise RuntimeError("failed to set initial object state") from exc

    def apply_initial_history_for_current_scenario(self):
        """只覆盖 ego/选定 BV 的固定长度 initial 历史，背景车保持数据集。"""
        engine = getattr(self.env, "engine", None)
        scenario = (
            getattr(getattr(engine, "data_manager", None), "current_scenario", {})
            if engine else {}
        )
        if not isinstance(scenario, dict):
            raise RuntimeError("current scenario is missing")

        tracks = scenario.get("tracks", {})
        metadata = scenario.get("metadata", {})
        sdc_id = str(metadata.get("sdc_id", ""))
        if not sdc_id or sdc_id not in tracks:
            raise RuntimeError("cannot find ego/sdc track for initial history")

        policy = self.default_policy()
        dataset_history_steps = int(getattr(policy, "WARMUP_STEPS", 20)) + 1
        if int(self.history_steps) != dataset_history_steps:
            raise RuntimeError(
                "initial history length %d must equal dataset/Pluto history length %d"
                % (int(self.history_steps), dataset_history_steps)
            )
        policy.WARMUP_STEPS = int(self.history_steps) - 1

        base_cache = getattr(policy, "_base_scenario_cache", None)
        base_tracks = base_cache.get("tracks", {}) if isinstance(base_cache, dict) else {}

        configured = []
        ego_initial = self.parse_initial_pose(self.ego_cfg, "ego")
        configured.append(("ego", sdc_id, ego_initial, None))
        for agent in self.vehicles_cfg.get("agents", []):
            vehicle_id = str(agent.get("id", "agent"))
            scenario_id = str(agent.get("scenario_track_id", "")).strip()
            if not scenario_id:
                raise RuntimeError(
                    "%s scenario_track_id is required before initial history" % vehicle_id
                )
            if scenario_id not in tracks:
                raise RuntimeError(
                    "%s scenario_track_id=%s is not in current scenario"
                    % (vehicle_id, scenario_id)
                )
            configured.append(
                (vehicle_id, scenario_id, self.parse_initial_pose(agent, vehicle_id), agent)
            )

        startup_steps = int(self.history_steps)
        objects = self.engine_objects(engine)
        scenario_to_object = self.scenario_to_object_id_map(engine)
        seeded_agent_histories = {}
        self.startup_trajectories_by_scenario_id = {}

        for label, scenario_id, initial, agent_cfg in configured:
            startup = self.build_accel_trajectory(
                initial,
                total_steps=startup_steps,
            )
            self.patch_track_history(tracks[scenario_id], startup, label)
            if scenario_id in base_tracks:
                self.patch_track_history(
                    base_tracks[scenario_id],
                    startup,
                    "%s_base" % label,
                )
            self.startup_trajectories_by_scenario_id[scenario_id] = copy.deepcopy(startup)

            if scenario_id == sdc_id:
                obj = getattr(engine, "agents", {}).get("default_agent") if engine else None
                self.set_object_to_history_entry(obj, startup[0])
                if obj is not None:
                    obj.expert_traj = tracks[scenario_id]["state"]["position"]
                if hasattr(policy, "hist_buffer"):
                    policy.hist_buffer = {
                        index: copy.deepcopy(entry)
                        for index, entry in enumerate(startup)
                    }
            else:
                object_id = scenario_to_object.get(scenario_id, scenario_id)
                self.set_object_to_history_entry(objects.get(str(object_id)), startup[0])
                seeded_agent_histories[scenario_id] = {
                    index: copy.deepcopy(entry)
                    for index, entry in enumerate(startup)
                }

        if hasattr(policy, "agent_hist_buffer"):
            policy.agent_hist_buffer.update(seeded_agent_histories)
        if hasattr(policy, "current_state"):
            policy.current_state = scenario

        rospy.loginfo(
            "Applied fixed initial histories: ego_track=%s selected_bv=%d "
            "history_steps=%d max_initial_speed=%.6f",
            sdc_id,
            len(self.vehicles_cfg.get("agents", [])),
            startup_steps,
            self.initial_history_max_speed(),
        )

    def build_object_map(self):
        """建立 AutoScene vehicle_id 与 UniTraj/MetaDrive track id 的映射。

        auto_assign_agents=true 时，scenario_track_id 已在 reset_scenario()
        内按当前场景自动写入内存配置。
        """
        mapping = {}
        engine = getattr(self.env, "engine", None)
        scenario = getattr(getattr(engine, "data_manager", None), "current_scenario", None)
        if isinstance(scenario, dict):
            for track_id in scenario.get("tracks", {}).keys():
                mapping[str(track_id)] = str(track_id)
        for idx, agent in enumerate(self.vehicles_cfg.get("agents", []), start=1):
            scenario_id = agent.get("scenario_track_id") or agent.get("metadrive_object_id")
            if scenario_id:
                mapping[agent.get("id", "agent_%03d" % idx)] = str(scenario_id)
        return mapping

    def step(
            self,
            latest_ego_state_feedback,
            agent_state_feedback,
    ):
        """推进一次 MetaDrive 环境，并返回 ROS 可发布的 snapshot。

        顺序很关键：
        0. 如果上一帧结束了场景，先切换到下一个场景并丢弃旧场景 feedback；
        1. 把上一周期 actual feedback 覆盖到 MetaDrive mirror；
        2. 把真实平台 feedback 注入 Pluto 历史；
        3. 调用 env.step() 生成新规划和新状态；
        4. 抽取 snapshot 发布给 ROS。
        """
        if self.env is None:
            return None

        # 延迟到下一帧开始时切换场景，避免返回旧 snapshot 时
        # MetaDrive 内部环境已经被 reset 成新场景。
        if self.pending_scenario_index is not None:
            next_index = self.pending_scenario_index
            self.pending_scenario_index = None
            self.scenario_index = next_index

            self.reset_scenario(
                self.scenario_index
            )

            # runtime 此时传入的 feedback 来自上一场景，
            # 新场景第一帧不能把旧场景车辆状态回灌进来。
            latest_ego_state_feedback = None
            agent_state_feedback = {}

        # 已终止后不要继续 env.step()，否则会重复 terminated at step 101/102...
        if self.finished_all_scenarios:
            return {
                "finished": True,
            }

        if self.terminated:
            raise RuntimeError(
                "scenario terminated without "
                "successful auto advance"
            )

        try:
            self.apply_ego_feedback(
                latest_ego_state_feedback
            )

            self.apply_controlled_bv_feedback(
                agent_state_feedback
            )

            self.inject_pluto_histories(agent_state_feedback)

            _obs, _reward, term, trunc, _info = (
                self.env.step([0, 0])
            )

            # 必须先提取 MetaDrive 原始状态。
            snapshot = self.make_snapshot()

            snapshot["terminated"] = bool(term or trunc)
            snapshot["terminate_info"] = dict(_info or {})
            self.last_snapshot = snapshot

            self.render_frame(snapshot)

            self.step_index += 1

            if term or trunc:
                reason = ""
                if isinstance(_info, dict):
                    reason = str(_info.get("terminate_reason", _info.get("terminate_scene", "")))
                self.termination_message = reason or ("term=%s trunc=%s" % (term, trunc))

                rospy.loginfo(
                    "UniTraj scenario %d terminated at step %d: %s",
                    self.scenario_index,
                    self.step_index,
                    self.termination_message,
                )

                if self.auto_advance_scenario:
                    next_index = self.scenario_index + 1

                    if next_index < self.end_scenario_index:
                        # 不在终止帧立即 reset。
                        # 终止帧的 snapshot 仍属于当前场景，
                        # 下一次 step() 开始时再切换环境。
                        self.pending_scenario_index = next_index
                    else:
                        self.terminated = True
                        self.finished_all_scenarios = True
                        rospy.loginfo(
                            "UniTraj finished all scenarios: "
                            "start=%d, end=%d, total=%d",
                            self.start_scenario_index,
                            self.end_scenario_index,
                            self.eval_eps,
                        )
                else:
                    self.terminated = True
                    self.finished_all_scenarios = True

            return snapshot


        except Exception as exc:
            self.status["active"] = False
            self.status["message"] = (
                    "UniTraj step failed: %s" % exc
            )
            rospy.logerr(
                "%s\n%s",
                self.status["message"],
                traceback.format_exc(),
            )
            raise

    def apply_ego_feedback(
            self,
            latest_ego_state_feedback,
    ):
        """Apply external actual ego feedback to the MetaDrive mirror."""
        engine = getattr(self.env, "engine", None)
        ego_obj = getattr(engine, "agents", {}).get("default_agent", None) if engine else None
        if ego_obj is None:
            return
        if latest_ego_state_feedback is not None and latest_ego_state_feedback.valid:
            try:
                ego_obj.set_position([
                    latest_ego_state_feedback.pose.position.x,
                    latest_ego_state_feedback.pose.position.y,
                ])
                ego_obj.set_velocity([
                    latest_ego_state_feedback.twist.linear.x,
                    latest_ego_state_feedback.twist.linear.y,
                ])
                ego_obj.set_heading_theta(float(latest_ego_state_feedback.heading))
                return
            except Exception as exc:
                raise RuntimeError(
                    "failed to apply ego feedback"
                ) from exc

    def apply_controlled_bv_feedback(
            self,
            agent_state_feedback,
    ):
        """用真实平台反馈覆盖对应 MetaDrive 周车对象。

        platform_comm_node 已经把 UDP 中的经纬度状态解算为 G 坐标，
        再转换为 AutoScene world 坐标。
        这里还要把 scenario_track_id 翻译为 MetaDrive object_id，才能改到正确对象。
        """
        engine = getattr(self.env, "engine", None)
        if engine is None:
            return
        objects = self.engine_objects(engine)
        scenario_to_obj = self.scenario_to_object_id_map(engine)
        for vehicle_id, state in (
                agent_state_feedback.items()
        ):
            # platform_id=-1 的虚拟 BV 不需要 set_position，
            # 它本身就在 MetaDrive 中按仿真结果运行。
            if vehicle_id not in self.controlled_vehicle_ids:
                continue

            if not state.valid:
                continue

            object_id = self.object_map.get(vehicle_id, vehicle_id)
            object_id = scenario_to_obj.get(str(object_id), str(object_id))
            obj = objects.get(object_id)
            if obj is None:
                obj = objects.get(vehicle_id)
            if obj is None:
                raise RuntimeError(
                    "cannot find MetaDrive mirror object for controlled BV %s"
                    % vehicle_id
                )
            try:
                obj.set_position([state.pose.position.x, state.pose.position.y])
                obj.set_velocity([state.twist.linear.x, state.twist.linear.y])
                obj.set_heading_theta(float(state.heading))
            except Exception as exc:
                raise RuntimeError(
                    "failed to apply BV feedback: %s"
                    % vehicle_id
                ) from exc

    def inject_pluto_histories(self, agent_state_feedback):
        """把 ego 和周车反馈写入 PlutoPolicy 的历史轨迹缓存。

        Physical-platform history is owned by runtime. Ego history is always
        written once inside InferencePolicy after the mirror has been updated.
        """
        policy = self.default_policy()
        if policy is None:
            return

        engine = getattr(self.env, "engine", None)
        time_index = self.pluto_history_time_index()

        agent_hist_buffer = getattr(policy, "agent_hist_buffer", None)
        if agent_hist_buffer is None:
            return
        for vehicle_id, state in (
                agent_state_feedback.items()
        ):
            if not getattr(state, "valid", False):
                continue

            scenario_id = (
                self.resolve_vehicle_to_scenario_id(
                    engine,
                    vehicle_id,
                )
            )

            if scenario_id is None:
                raise RuntimeError(
                    "cannot resolve Pluto history track for vehicle %s"
                    % vehicle_id
                )

            scenario_id = str(scenario_id)

            # 所有 BV 都更新 history；
            # 只有平台控制 BV 才加入 controlled set。
            if (
                    vehicle_id
                    in self.controlled_vehicle_ids
                    and hasattr(
                policy,
                "_controlled_cbv_scenario_ids",
            )
            ):
                policy._controlled_cbv_scenario_ids.add(
                    scenario_id
                )

            entry = self.state_to_history_entry(state)
            # 真实平台历史只写当前实际样本；禁止外推或补造缺失帧。
            agent_hist_buffer.setdefault(scenario_id, {})[int(time_index)] = entry

    def state_to_history_entry(self, state):
        """把一条真实反馈原样转换为 Pluto 历史样本，不做时间外推。"""
        import numpy as np

        values = (
            float(state.pose.position.x),
            float(state.pose.position.y),
            float(state.heading),
            float(state.twist.linear.x),
            float(state.twist.linear.y),
            float(state.accel.linear.x),
            float(state.accel.linear.y),
        )
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError("actual platform feedback contains non-finite state")
        return {
            "position": np.array(values[0:2], dtype=np.float64),
            "heading": values[2],
            "velocity": np.array(values[3:5], dtype=np.float64),
            "acceleration": np.array(values[5:7], dtype=np.float64),
            "valid": True,
        }

    def resolve_vehicle_to_scenario_id(self, engine, vehicle_id):
        """把 AutoScene vehicle_id 解析成 UniTraj scenario track id。"""
        if engine is None:
            return None
        scenario = getattr(getattr(engine, "data_manager", None), "current_scenario", {}) or {}
        tracks = scenario.get("tracks", {}) if isinstance(scenario, dict) else {}
        raw_object_to_scenario = getattr(getattr(engine, "traffic_manager", None), "obj_id_to_scenario_id", {}) or {}
        object_to_scenario = {
            str(object_id): str(scenario_id)
            for object_id, scenario_id in raw_object_to_scenario.items()
        }

        candidate = self.object_map.get(vehicle_id, vehicle_id)
        if str(candidate) in tracks:
            return str(candidate)
        if str(candidate) in object_to_scenario:
            return str(object_to_scenario[str(candidate)])

        for cfg in self.vehicles_cfg.get("agents", []):
            if cfg.get("id") != vehicle_id:
                continue
            scenario_id = cfg.get("scenario_track_id") or cfg.get("track_id")
            if scenario_id is not None and str(scenario_id) in tracks:
                return str(scenario_id)
            object_id = cfg.get("metadrive_object_id")
            if object_id is not None and str(object_id) in object_to_scenario:
                return str(object_to_scenario[str(object_id)])

        # 最后再做一次反向查找，兼容 object_map 中保存的是 MetaDrive object_id 的情况。
        for object_id, scenario_id in object_to_scenario.items():
            if str(object_id) == str(candidate) or str(scenario_id) == str(candidate):
                return str(scenario_id)
        return None

    def background_vehicle_states(
            self,
            engine,
            objects,
            scenario,
            skip_ids,
            stamp,
    ):
        """抽取未被人工配置的当前活动背景车辆。

        背景车辆只加入 world/state，用于全场景评估；
        不进入 /agents/state、platform feedback 或 Pluto history。
        """
        if (
                not self.include_background_vehicles
                or engine is None
        ):
            return []

        traffic_manager = getattr(
            engine,
            "traffic_manager",
            None,
        )

        raw_mapping = (
                          getattr(
                              traffic_manager,
                              "obj_id_to_scenario_id",
                              {},
                          )
                          if traffic_manager is not None
                          else {}
                      ) or {}

        tracks = (
            scenario.get("tracks", {})
            if isinstance(scenario, dict)
            else {}
        )

        tracks_by_id = {
            str(track_id): track
            for track_id, track in tracks.items()
        }

        background = []
        seen_scenario_ids = set()

        for (
                raw_object_id,
                raw_scenario_id,
        ) in raw_mapping.items():
            object_id = str(raw_object_id)
            scenario_id = str(raw_scenario_id)

            # 排除 ego、人工配置的 Pluto BV 和重复映射。
            if (
                    object_id in skip_ids
                    or scenario_id in skip_ids
                    or scenario_id in seen_scenario_ids
            ):
                continue

            obj = objects.get(object_id)

            if obj is None:
                continue

            track = tracks_by_id.get(
                scenario_id,
                {},
            )

            track_type = str(
                track.get("type", "")
            ).upper()

            # 当前只加入机动车背景车辆。
            if (
                    track_type
                    and "VEHICLE" not in track_type
            ):
                continue

            state = self.object_to_state(
                obj,
                "background_%s" % scenario_id,
                "virtual_background",
                stamp,
            )

            if not state.valid:
                continue

            background.append(state)
            seen_scenario_ids.add(scenario_id)

        background.sort(
            key=lambda state: state.vehicle_id
        )

        return background

    def make_snapshot(self):
        """从 MetaDrive 当前状态抽取 ROS world/ego/agent/platform target。"""
        stamp = rospy.Time.now()
        engine = getattr(self.env, "engine", None)
        objects = self.engine_objects(engine)
        agents = getattr(engine, "agents", {}) if engine else {}
        ego_obj = agents.get("default_agent") or next(iter(agents.values()), None)
        scenario = getattr(getattr(engine, "data_manager", None), "current_scenario", {}) if engine else {}
        sdc_id = str(scenario.get("metadata", {}).get("sdc_id", "ego_001")) if isinstance(scenario, dict) else "ego_001"
        scenario_to_obj = self.scenario_to_object_id_map(engine)
        skip_ids = {sdc_id, "default_agent"}
        if sdc_id in scenario_to_obj:
            skip_ids.add(str(scenario_to_obj[sdc_id]))
        ego = self.object_to_state(
            ego_obj,
            str(self.ego_cfg.get("id", "ego_001")),
            "virtual",
            stamp,
        )
        agent_states = []
        targets = []
        configured = self.vehicles_cfg.get("agents", [])
        for idx, cfg in enumerate(configured, start=1):
            vehicle_id = cfg.get("id", "agent_%03d" % idx)
            obj, scenario_id = self.resolve_configured_object(engine, objects, cfg, idx - 1, skip_ids=skip_ids)

            if scenario_id is not None:
                skip_ids.add(str(scenario_id))

            if obj is not None:
                object_id = getattr(obj, "id", None)

                if object_id is not None:
                    skip_ids.add(str(object_id))

                for object_key, candidate in objects.items():
                    if candidate is obj:
                        skip_ids.add(str(object_key))
                        break

            entity_type = (
                "physical_platform"
                if is_platform_controlled_agent(cfg)
                else "virtual"
            )

            state = self.object_to_state(
                obj,
                vehicle_id,
                entity_type,
                stamp,
            )
            state.length = float(cfg.get("length", state.length or 4.5))
            state.width = float(cfg.get("width", state.width or 2.0))
            agent_states.append(state)
            if is_platform_controlled_agent(cfg):
                targets.append(self.object_plan_to_trajectory(obj, cfg, stamp, scenario_id))

        background_states = (
            self.background_vehicle_states(
                engine,
                objects,
                scenario,
                skip_ids,
                stamp,
            )
        )

        current_scenario_id = str(
            scenario.get(
                "id",
                scenario.get(
                    "metadata",
                    {},
                ).get(
                    "scenario_id",
                    "unknown",
                ),
            )
        )

        return {
            "scenario_id": current_scenario_id,
            "sim_time": (
                    self.step_index * self.history_dt
            ),
            "ego": ego,
            "agents": agent_states,
            "background_agents": background_states,
            "platform_targets": [
                target
                for target in targets
                if target is not None
            ],
        }

    def default_policy(self):
        """获取 default_agent 实际车辆对象绑定的 PlutoPolicy。"""
        engine = getattr(self.env, "engine", None)

        if engine is None or not hasattr(engine, "get_policy"):
            raise RuntimeError(
                "MetaDrive engine does not provide get_policy()"
            )

        agents = getattr(engine, "agents", {})
        ego_obj = agents.get("default_agent")

        if ego_obj is None:
            raise RuntimeError(
                "MetaDrive default_agent object is missing"
            )

        # MetaDrive policy 使用车辆对象 ID 注册，
        # 不是使用逻辑 agent 名称 default_agent。
        object_id = getattr(ego_obj, "id", None)

        if object_id is None:
            raise RuntimeError(
                "MetaDrive default_agent object has no id"
            )

        policy = engine.get_policy(object_id)

        if policy is None:
            available_policy_ids = []

            if hasattr(engine, "get_policies"):
                available_policy_ids = list(
                    engine.get_policies().keys()
                )
            else:
                available_policy_ids = list(
                    getattr(
                        engine,
                        "_object_policies",
                        {},
                    ).keys()
                )

            raise RuntimeError(
                "Pluto policy is missing for "
                "default_agent object_id=%s; "
                "available_policy_ids=%s"
                % (
                    object_id,
                    available_policy_ids,
                )
            )

        return policy

    def scenario_to_object_id_map(self, engine):
        """读取 MetaDrive traffic_manager 中 scenario_id -> object_id 的映射。"""
        mapping = {}
        traffic_manager = getattr(engine, "traffic_manager", None)
        raw = getattr(traffic_manager, "obj_id_to_scenario_id", {}) if traffic_manager else {}
        try:
            for obj_id, scenario_id in raw.items():
                mapping[str(scenario_id)] = str(obj_id)
        except Exception:
            pass
        return mapping

    def object_to_scenario_id(self, engine, objects, obj):
        """反查某个 MetaDrive 对象对应的 scenario track id。"""
        if obj is None:
            return None
        reverse = self.scenario_to_object_id_map(engine)
        for scenario_id, obj_id in reverse.items():
            if objects.get(str(obj_id)) is obj:
                return scenario_id
        for key, candidate in objects.items():
            if candidate is obj:
                return str(key)
        return None

    def resolve_configured_object(self, engine, objects, cfg, ordinal, skip_ids):
        """按明确的 scenario_track_id 解析配置 BV。

        不按位置猜测对象，也不回退到 Pluto 动态选择结果。track 尚未激活时返回
        invalid 占位状态；配置错误则立即报错，避免把轨迹下发给错误车辆。
        """
        del ordinal, skip_ids
        scenario_id = cfg.get("scenario_track_id")
        if not scenario_id:
            raise RuntimeError(
                "vehicle %s has no scenario_track_id"
                % cfg.get("id", "<unknown>")
            )

        scenario_id = str(scenario_id)
        scenario = getattr(
            getattr(engine, "data_manager", None),
            "current_scenario",
            {},
        ) or {}
        tracks = scenario.get("tracks", {}) if isinstance(scenario, dict) else {}
        if scenario_id not in tracks:
            raise RuntimeError(
                "scenario_track_id=%s for vehicle %s is not in current scenario"
                % (scenario_id, cfg.get("id", "<unknown>"))
            )

        object_id = self.scenario_to_object_id_map(engine).get(
            scenario_id,
            scenario_id,
        )
        obj = objects.get(str(object_id))
        if obj is None:
            rospy.logwarn_throttle(
                2.0,
                "configured track is not active yet: vehicle=%s "
                "scenario_track_id=%s episode_step=%s",
                cfg.get("id"),
                scenario_id,
                getattr(engine, "episode_step", -1),
            )
        return obj, scenario_id

    def engine_objects(self, engine):
        objects = {}
        if engine is None:
            return objects
        if hasattr(engine, "get_objects"):
            try:
                objects.update({str(k): v for k, v in engine.get_objects().items()})
            except Exception:
                pass
        if hasattr(engine, "agents"):
            objects.update({str(k): v for k, v in getattr(engine, "agents", {}).items()})
        return objects

    def object_to_state(self, obj, vehicle_id, entity_type, stamp):
        """把 MetaDrive 对象转换为 autoscene_msgs/VehicleState。

        obj 为 None 时返回 valid=false 占位状态，明确表示该配置车辆当前未激活。
        """
        state = VehicleState()
        state.header.stamp = stamp
        state.header.frame_id = self.frame_id
        state.vehicle_id = vehicle_id
        state.entity_type = entity_type
        state.length = float(getattr(obj, "LENGTH", getattr(obj, "top_down_length", 4.5)) or 4.5) if obj else 4.5
        state.width = float(getattr(obj, "WIDTH", getattr(obj, "top_down_width", 2.0)) or 2.0) if obj else 2.0
        state.valid = obj is not None
        if not obj:
            return state
        pos = getattr(obj, "position", [0.0, 0.0])
        yaw = float(getattr(obj, "heading_theta", 0.0))
        speed = float(getattr(obj, "speed", 0.0))
        apply_pose(state.pose, float(pos[0]), float(pos[1]), yaw)
        state.twist.linear.x = math.cos(yaw) * speed
        state.twist.linear.y = math.sin(yaw) * speed
        state.heading = yaw
        return state

    def object_plan_to_trajectory(self, obj, cfg, stamp, scenario_id):
        """为配置中的周车生成 world-frame ROS Trajectory。

        runtime 只负责提取/生成 world 轨迹；大地坐标转换和 V2 UDP
        编码全部由 platform_comm_node 完成。
        """
        traj = Trajectory()
        traj.header.stamp = stamp
        traj.header.frame_id = self.frame_id
        traj.vehicle_id = cfg.get("id", "")
        traj.platform_id = cfg.get("platform_id", "")
        traj.source = "unitraj_pluto_agent"
        traj.frame_id = self.frame_id
        traj.valid = obj is not None
        traj.confidence = 1.0
        if obj is None:
            return traj
        raw = self.startup_agent_plan(scenario_id)
        if raw is not None:
            traj.source = "initial_accel_history"
        else:
            raw, source = self.pluto_agent_plan(obj, cfg, scenario_id)
            traj.source = source
        if raw is None:
            traj.valid = False
            traj.source = "unitraj_pluto_missing_agent_plan"
            return traj
        self.fill_trajectory_from_rows(traj, raw)
        traj.valid = bool(traj.points)
        return traj

    def startup_agent_plan(self, scenario_id):
        """生成 warmup 阶段滚动的固定 50 点启动轨迹。

        前段严格来自完整启动轨迹；末点是用户 initial，末 21 点供 Pluto 输入。
        为了满足固定 50 点 UDP 协议，initial 之后的包尾按 initial 的已知速度
        做匀速延伸。该尾段是
        启动协议轨迹的明确组成，不用于补造实际反馈历史。
        """
        if self.history_source != "initial" or scenario_id is None:
            return None
        startup = self.startup_trajectories_by_scenario_id.get(str(scenario_id))
        if not startup:
            return None
        time_index = self.pluto_history_time_index()
        if time_index >= len(startup) - 1:
            return None

        import numpy as np

        terminal = startup[-1]
        terminal_pos = np.asarray(terminal["position"], dtype=np.float64)[:2]
        terminal_vel = np.asarray(terminal["velocity"], dtype=np.float64)[:2]
        rows = []
        for offset in range(self.output_point_num):
            desired_index = time_index + offset + 1
            if desired_index < len(startup):
                entry = startup[desired_index]
                position = np.asarray(entry["position"], dtype=np.float64)[:2]
                velocity = np.asarray(entry["velocity"], dtype=np.float64)[:2]
                acceleration = np.asarray(entry["acceleration"], dtype=np.float64)[:2]
                heading = float(entry["heading"])
            else:
                extra_time = (desired_index - (len(startup) - 1)) * self.history_dt
                position = terminal_pos + terminal_vel * extra_time
                velocity = terminal_vel
                acceleration = np.zeros(2, dtype=np.float64)
                heading = float(terminal["heading"])
            rows.append([
                float(position[0]),
                float(position[1]),
                math.cos(heading),
                math.sin(heading),
                float(velocity[0]),
                float(velocity[1]),
                float(acceleration[0]),
                float(acceleration[1]),
            ])
        return rows

    def pluto_agent_plan(self, obj, cfg, scenario_id):
        """Return the exact CBV trajectory already aligned by the policy."""
        policy = self.default_policy()
        if policy is None:
            return None, "unitraj_pluto_missing_policy"

        adv_ids = [str(x) for x in getattr(policy, "adv_agent_ids", [])]
        resolved_id = self.resolve_track_id(policy, scenario_id, adv_ids)
        aligned = getattr(policy, "current_aligned_adv_trajs", {})
        raw = aligned.get(str(resolved_id)) if resolved_id is not None else None
        if raw is None or len(raw) < 2:
            return None, "unitraj_pluto_missing_aligned_agent_plan"

        converted = self.convert_pluto_control_traj(policy, obj, raw)
        patch_info = getattr(policy, "current_adv_patch_info", {}).get(
            str(resolved_id), {}
        )
        rospy.logdebug(
            "Pluto ROS trajectory: vehicle=%s track=%s patch_offset=%s points=%d",
            cfg.get("id"),
            resolved_id,
            patch_info.get("patch_offset"),
            len(converted),
        )
        source_name = (
            "unitraj_pluto_raw_external_platform_plan"
            if patch_info.get("source") == "pluto_raw_external_platform"
            else "unitraj_pluto_aligned_agent_plan"
        )
        return converted, source_name

    def resolve_track_id(self, policy, track_id, available_ids):
        """解析 UniTraj track id，支持精确匹配和唯一后缀匹配。"""
        if track_id is None:
            return None
        if hasattr(policy, "_resolve_track_id"):
            return policy._resolve_track_id(
                track_id,
                available_ids,
            )
        track_id = str(track_id)
        available = [str(x) for x in available_ids]
        if track_id in available:
            return track_id
        suffix = track_id[-5:]
        matches = [x for x in available if x.endswith(suffix) or track_id.endswith(x[-5:])]
        return matches[0] if len(matches) == 1 else None

    def convert_pluto_control_traj(self, policy, obj, raw):
        """严格转换 Pluto 轨迹；模型输出至少 50 点且每点包含 xy/cos/sin/vx/vy。"""
        import numpy as np

        arr = np.asarray(raw, dtype=np.float64)
        if arr.ndim > 2:
            if arr.shape[0] != 1:
                raise RuntimeError("Pluto platform trajectory has unexpected batch dimension")
            arr = arr[0]
        if arr.ndim != 2:
            raise RuntimeError("Pluto platform trajectory must be a 2D array")
        if len(arr) < self.output_point_num:
            raise RuntimeError(
                "Pluto platform trajectory requires at least %d points, got %d"
                % (self.output_point_num, len(arr))
            )
        if arr.shape[1] < 6:
            raise RuntimeError(
                "Pluto platform trajectory requires [x,y,cos,sin,vx,vy], got width=%d"
                % arr.shape[1]
            )
        if not np.isfinite(arr[:, :6]).all():
            raise RuntimeError("Pluto platform trajectory contains non-finite values")

        if hasattr(policy, "_convert_rear_axle_traj_xy_to_center"):
            xy = policy._convert_rear_axle_traj_xy_to_center(
                obj,
                arr,
                default_heading=float(getattr(obj, "heading_theta", 0.0)),
            )
        else:
            raise RuntimeError("Pluto policy does not expose rear-axle to center conversion")

        heading = np.arctan2(arr[:, 3], arr[:, 2])
        velocity = arr[:, 4:6]
        # V2 协议需要加速度；只对 Pluto 已提供的速度做确定性差分，不补造缺失速度。
        acceleration = np.gradient(velocity, self.output_dt, axis=0)

        rows = []
        for i in range(len(arr)):
            yaw = float(heading[i])
            rows.append([
                float(xy[i, 0]),
                float(xy[i, 1]),
                math.cos(yaw),
                math.sin(yaw),
                float(velocity[i, 0]),
                float(velocity[i, 1]),
                float(acceleration[i, 0]),
                float(acceleration[i, 1]),
            ])
        return rows

    def fill_trajectory_from_rows(self, traj, raw):
        """严格取轨迹前 50 点填入 ROS Trajectory，不推导缺失 heading/velocity。"""
        points = list(raw)
        if len(points) < self.output_point_num:
            raise RuntimeError(
                "platform trajectory requires %d points, got %d"
                % (self.output_point_num, len(points))
            )
        points = points[: self.output_point_num]

        for i, row in enumerate(points):
            try:
                row_values = [float(value) for value in row]
            except (TypeError, ValueError) as exc:
                raise RuntimeError("trajectory row %d is not numeric" % i) from exc
            if len(row_values) < 8:
                raise RuntimeError(
                    "trajectory row %d requires [x,y,cos,sin,vx,vy,ax,ay], got %d values"
                    % (i, len(row_values))
                )
            if not all(math.isfinite(value) for value in row_values[:8]):
                raise RuntimeError("trajectory row %d contains non-finite values" % i)

            x, y, cos_yaw, sin_yaw, vx, vy, ax, ay = row_values[:8]
            norm = math.hypot(cos_yaw, sin_yaw)
            if norm < 1e-6:
                raise RuntimeError("trajectory row %d has invalid heading vector" % i)
            yaw = math.atan2(sin_yaw, cos_yaw)

            point = TrajectoryPoint()
            point.time_from_start = (i + 1) * self.output_dt
            apply_pose(point.pose, x, y, yaw)
            point.velocity.linear.x = vx
            point.velocity.linear.y = vy
            point.acceleration.linear.x = ax
            point.acceleration.linear.y = ay
            point.heading = yaw
            traj.points.append(point)


if __name__ == "__main__":
    try:
        node = ScenarioRuntimeNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass