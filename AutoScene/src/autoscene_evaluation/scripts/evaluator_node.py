#!/usr/bin/env python3
"""Evaluate the fused ``world/state`` produced by scenario_runtime_node.

The evaluator is source-agnostic: ego and BV state may come from simulation or
external feedback. Collision and clearance use oriented vehicle rectangles.
"""
import json
import math
import os
import re
import threading
import time
import traceback

import rospy

from autoscene_msgs.msg import (
    EvaluationMetrics,
    PlatformStateArray,
    ScenarioStatus,
    WorldState,
)
from autoscene_msgs.srv import ExportEvaluation, ExportEvaluationResponse


EPS = 1e-9


def merge_runtime_collision(metric, failure_reasons):
    """Preserve swept contacts even when endpoint boxes no longer overlap.

    Runtime checks between frames. Its terminal contact is authoritative; merge
    it without counting a second world sample when status arrives after metrics.
    """
    pairs = set()
    for reason in failure_reasons:
        if not str(reason).startswith("failed:collision:"):
            continue
        try:
            raw = json.loads(str(reason).split("collision:", 1)[1])
            if isinstance(raw, list):
                for pair in raw:
                    if isinstance(pair, list) and len(pair) == 2 and all(isinstance(v, str) for v in pair):
                        pairs.add(tuple(sorted(pair)))
        except (TypeError, ValueError):
            continue
    if not pairs:
        return False
    try:
        detail = json.loads(metric.detail_json)
        if not isinstance(detail, dict):
            detail = {}
    except (TypeError, ValueError):
        detail = {}
    pairs.update(tuple(pair) for pair in detail.get("collision_pairs", []))
    detail["collision_pairs"] = [list(pair) for pair in sorted(pairs)]
    detail["runtime_collision_terminal"] = True
    metric.collision = True
    metric.scenario_success = False
    metric.detail_json = json.dumps(detail)
    return True


def dist_xy(a, b):
    return math.hypot(a.position.x - b.position.x, a.position.y - b.position.y)


def point_xy(pose):
    return float(pose.position.x), float(pose.position.y)


def vehicle_dimensions(state):
    """读取车辆尺寸；消息未填时使用常见乘用车默认尺寸。"""
    length = float(state.length) if state.length > 0.0 else 4.5
    width = float(state.width) if state.width > 0.0 else 2.0
    return length, width


def vehicle_polygon(state):
    """返回车辆有向矩形四角，顺序为前左、前右、后右、后左。"""
    x, y = point_xy(state.pose)
    length, width = vehicle_dimensions(state)
    half_l = 0.5 * length
    half_w = 0.5 * width
    yaw = float(state.heading)

    fx = math.cos(yaw)
    fy = math.sin(yaw)
    lx = -math.sin(yaw)
    ly = math.cos(yaw)

    return [
        (x + fx * half_l + lx * half_w, y + fy * half_l + ly * half_w),
        (x + fx * half_l - lx * half_w, y + fy * half_l - ly * half_w),
        (x - fx * half_l - lx * half_w, y - fy * half_l - ly * half_w),
        (x - fx * half_l + lx * half_w, y - fy * half_l + ly * half_w),
    ]


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def polygon_axes(poly):
    """生成 SAT 分离轴，用于判断两个有向矩形是否重叠。"""
    axes = []
    for i, p0 in enumerate(poly):
        p1 = poly[(i + 1) % len(poly)]
        ex = p1[0] - p0[0]
        ey = p1[1] - p0[1]
        length = math.hypot(ex, ey)
        if length <= EPS:
            continue
        axes.append((-ey / length, ex / length))
    return axes


def project_polygon(poly, axis):
    values = [dot(point, axis) for point in poly]
    return min(values), max(values)


def polygons_overlap(poly_a, poly_b):
    for axis in polygon_axes(poly_a) + polygon_axes(poly_b):
        min_a, max_a = project_polygon(poly_a, axis)
        min_b, max_b = project_polygon(poly_b, axis)
        if max_a < min_b or max_b < min_a:
            return False
    return True


def point_segment_distance(p, a, b):
    ab = (b[0] - a[0], b[1] - a[1])
    ap = (p[0] - a[0], p[1] - a[1])
    denom = dot(ab, ab)
    if denom <= EPS:
        return math.hypot(p[0] - a[0], p[1] - a[1])
    t = max(0.0, min(1.0, dot(ap, ab) / denom))
    closest = (a[0] + ab[0] * t, a[1] + ab[1] * t)
    return math.hypot(p[0] - closest[0], p[1] - closest[1])


def polygon_distance(poly_a, poly_b):
    """计算两个 OBB 多边形的最小轮廓距离；重叠时返回 0。"""
    if polygons_overlap(poly_a, poly_b):
        return 0.0

    best = float("inf")
    edges_a = list(zip(poly_a, poly_a[1:] + poly_a[:1]))
    edges_b = list(zip(poly_b, poly_b[1:] + poly_b[:1]))

    for point in poly_a:
        for edge_start, edge_end in edges_b:
            best = min(best, point_segment_distance(point, edge_start, edge_end))

    for point in poly_b:
        for edge_start, edge_end in edges_a:
            best = min(best, point_segment_distance(point, edge_start, edge_end))

    return best


def closing_time_to_collision(ego, agent, clearance):
    """基于轮廓间距与中心连线闭合速度估计 TTC。"""
    if clearance <= 0.0:
        return 0.0

    ex, ey = point_xy(ego.pose)
    ax, ay = point_xy(agent.pose)
    dx = ax - ex
    dy = ay - ey
    center_distance = math.hypot(dx, dy)

    if center_distance <= EPS:
        return 0.0

    rvx = agent.twist.linear.x - ego.twist.linear.x
    rvy = agent.twist.linear.y - ego.twist.linear.y
    range_rate = (dx * rvx + dy * rvy) / center_distance
    closing_speed = -range_rate

    if closing_speed <= 0.1:
        return -1.0

    return clearance / closing_speed


class EvaluatorNode:
    def __init__(self):
        rospy.init_node("evaluator_node")
        # collision_distance_m 是 OBB 轮廓间距缓冲；0.0 表示矩形接触/重叠才判碰撞。
        self.project = rospy.get_param("/project", {})
        self.evaluation = rospy.get_param("/evaluation", {})
        self.rate_hz = float(self.evaluation.get("rate_hz", 10.0))
        self.collision_distance = float(self.evaluation.get("collision_distance_m", 0.0))

        self.auto_export = bool(
            self.evaluation.get(
                "auto_export",
                True,
            )
        )

        self.export_output_dir = str(
            self.evaluation.get(
                "output_dir",
                rospy.get_param(
                    "/record/output_dir",
                    "./outputs",
                ),
            )
        )

        self.fail_on_invalid_platform = bool(
            self.evaluation.get(
                "fail_on_invalid_platform",
                True,
            )
        )

        self.platform_startup_grace_frames = max(
            int(
                self.evaluation.get(
                    "platform_startup_grace_frames",
                    10,
                )
            ),
            0,
        )

        self.max_consecutive_invalid_platform_frames = max(
            int(
                self.evaluation.get(
                    "max_consecutive_invalid_platform_frames",
                    3,
                )
            ),
            1,
        )

        self.scenario_finished = False
        self.evaluated_frame_count = 0
        self.invalid_platform_frames = {}
        self.failure_reasons = set()
        self.auto_export_path = None
        self.state_lock = threading.RLock()

        self.latest_world = None
        self.latest_platform_states = None

        # 当前正在评估的场景 ID。
        # scenario_id 变化时，清空上一场景的统计和失败状态。
        self.current_scenario_id = None

        # 场景内一旦发生碰撞即保持失败，直到收到新的 scenario_id。
        self.scenario_failed = False

        # 只保存当前场景的评估历史和 ego 速度历史。
        self.history = []
        self.ego_speed_history = []

        # 每个 world/state 最多评估一次，避免 runtime 停更后重复统计最后一帧。
        self.world_generation = 0
        self.last_evaluated_world_generation = -1

        self.metrics_pub = rospy.Publisher("evaluation/metrics", EvaluationMetrics, queue_size=1)
        self.summary_pub = rospy.Publisher("evaluation/summary", EvaluationMetrics, queue_size=1, latch=True)
        rospy.Subscriber("world/state", WorldState, self.on_world, queue_size=1)
        rospy.Subscriber("platform/state", PlatformStateArray, self.on_platform_states, queue_size=1)

        rospy.Subscriber(
            "scenario/status",
            ScenarioStatus,
            self.on_scenario_status,
            queue_size=1,
        )

        rospy.on_shutdown(
            self.on_shutdown
        )

        rospy.Service("evaluation/export", ExportEvaluation, self.on_export)
        rospy.Timer(rospy.Duration(1.0 / max(self.rate_hz, 1.0)), self.on_timer)
        rospy.loginfo("AutoScene evaluator started")

    @staticmethod
    def scenario_terminal_message(msg):
        if not (msg.loaded and not msg.running):
            return None
        message = str(msg.message).strip()
        lower = message.lower()
        if lower == "finished" or lower.startswith("failed:"):
            return message
        return None

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

    def on_scenario_status(self, msg):
        terminal_message = self.scenario_terminal_message(msg)
        if terminal_message is None:
            return

        with self.state_lock:
            if not self.current_scenario_id and msg.scenario_id:
                self.current_scenario_id = str(msg.scenario_id)
            self.scenario_finished = True
            if terminal_message.lower().startswith("failed:"):
                self.scenario_failed = True
                self.failure_reasons.add(terminal_message)
            self.flush_pending_world_locked()

            if self.history and merge_runtime_collision(self.history[-1], self.failure_reasons):
                self.metrics_pub.publish(self.history[-1])
                self.summary_pub.publish(self.history[-1])

            if self.auto_export:
                try:
                    path = self.export_summary_locked(
                        auto=True
                    )

                    rospy.loginfo(
                        "Auto-exported evaluation: %s",
                        path,
                    )
                except Exception:
                    rospy.logerr(
                        "Automatic evaluation export "
                        "failed:\n%s",
                        traceback.format_exc(),
                    )

        # 再延迟覆写一次，避免最后一个 world/state
        # 比 scenario/status 稍晚到达。
        if self.auto_export:
            rospy.Timer(
                rospy.Duration(0.15),
                self.on_delayed_auto_export,
                oneshot=True,
            )

    def on_delayed_auto_export(self, _event):
        with self.state_lock:
            if (
                    not self.scenario_finished
                    or not self.auto_export
            ):
                return

            try:
                self.flush_pending_world_locked()

                path = self.export_summary_locked(
                    auto=True
                )

                rospy.loginfo(
                    "Finalized evaluation export: %s",
                    path,
                )
            except Exception:
                rospy.logerr(
                    "Delayed evaluation export "
                    "failed:\n%s",
                    traceback.format_exc(),
                )

    def reset_scenario_state(self, scenario_id):
        self.current_scenario_id = str(
            scenario_id
        )
        self.latest_platform_states = None
        self.scenario_failed = False
        self.scenario_finished = False
        self.history = []
        self.ego_speed_history = []
        self.evaluated_frame_count = 0
        self.invalid_platform_frames = {}
        self.failure_reasons = set()
        self.auto_export_path = None

    def on_world(self, msg):
        with self.state_lock:
            scenario_id = str(msg.scenario_id)

            if scenario_id != self.current_scenario_id:
                rospy.loginfo(
                    "Evaluator switched scenario: "
                    "%s -> %s",
                    self.current_scenario_id,
                    scenario_id,
                )

                self.reset_scenario_state(
                    scenario_id
                )

            self.latest_world = msg
            self.world_generation += 1

    def on_platform_states(self, msg):
        with self.state_lock:
            self.latest_platform_states = msg

    def compute(self):
        """计算一帧评估指标。

        指标说明：
        - min_distance：自车有向矩形到所有有效 agent 有向矩形的最小轮廓间距；
        - collision：有向矩形重叠，或轮廓间距低于 collision_distance_m；
        - time_to_collision：基于轮廓间距和中心连线闭合速度的基础 TTC；
        - platform_tracking_error：平台反馈中的 tracking_error 均值。
        """
        metrics = EvaluationMetrics()
        metrics.header.stamp = rospy.Time.now()
        metrics.experiment_id = self.project.get("experiment_id", "")
        metrics.min_distance = float("inf")
        metrics.time_to_collision = float("inf")

        if self.latest_world is None:
            metrics.scenario_success = False
            metrics.min_distance = -1.0
            metrics.time_to_collision = -1.0
            metrics.detail_json = json.dumps({
                "reason": "waiting_world_state",
            })
            return metrics

        self.evaluated_frame_count += 1

        if not self.latest_world.ego.valid:
            self.scenario_failed = True
            self.failure_reasons.add("invalid_ego_state")
            metrics.scenario_success = False
            metrics.min_distance = -1.0
            metrics.time_to_collision = -1.0
            metrics.detail_json = json.dumps({
                "reason": "invalid_ego_state",
                "evaluated_frame_count": (
                    self.evaluated_frame_count
                ),
            })
            return metrics

        ego = self.latest_world.ego
        ego_speed = math.hypot(ego.twist.linear.x, ego.twist.linear.y)
        self.ego_speed_history.append(ego_speed)
        if len(self.ego_speed_history) > 5000:
            self.ego_speed_history = self.ego_speed_history[-5000:]

        metrics.ego_speed_mean = sum(self.ego_speed_history) / len(self.ego_speed_history)
        metrics.ego_speed_max = max(self.ego_speed_history)

        ego_poly = vehicle_polygon(ego)
        nearest_agent_id = ""
        invalid_agent_count = 0
        collision_agents = []
        min_center_distance = float("inf")

        invalid_platform_ids = []

        for agent in self.latest_world.agents:
            is_platform = (
                    str(
                        agent.entity_type
                    ).strip().lower()
                    == "physical_platform"
            )

            if not agent.valid:
                invalid_agent_count += 1

                if is_platform:
                    vehicle_id = str(
                        agent.vehicle_id
                    )

                    invalid_platform_ids.append(
                        vehicle_id
                    )

                    # 启动宽限期内只记录，不累计失败。
                    if (
                            self.evaluated_frame_count
                            > self.platform_startup_grace_frames
                    ):
                        count = (
                                self.invalid_platform_frames.get(
                                    vehicle_id,
                                    0,
                                )
                                + 1
                        )

                        self.invalid_platform_frames[
                            vehicle_id
                        ] = count

                        if (
                                self.fail_on_invalid_platform
                                and count
                                >= self.max_consecutive_invalid_platform_frames
                        ):
                            self.scenario_failed = True
                            self.failure_reasons.add(
                                "platform_invalid:%s"
                                % vehicle_id
                            )

                continue

            if is_platform:
                # 恢复有效后，连续 invalid 计数清零。
                self.invalid_platform_frames[
                    str(agent.vehicle_id)
                ] = 0

            agent_poly = vehicle_polygon(agent)
            clearance = polygon_distance(ego_poly, agent_poly)
            center_distance = dist_xy(ego.pose, agent.pose)
            min_center_distance = min(min_center_distance, center_distance)

            if clearance < metrics.min_distance:
                metrics.min_distance = clearance
                nearest_agent_id = agent.vehicle_id

            if clearance <= self.collision_distance:
                metrics.collision = True
                self.scenario_failed = True
                collision_agents.append(agent.vehicle_id)
                self.failure_reasons.add(
                    "collision:%s" % agent.vehicle_id
                )

            ttc = closing_time_to_collision(ego, agent, clearance)
            if ttc >= 0.0:
                metrics.time_to_collision = min(metrics.time_to_collision, ttc)

        # scenario_success 是场景级结果：
        # 本场景一旦发生过碰撞，后续帧不再恢复为成功。
        # The previous loop deliberately keeps ego-centric clearance/TTC. Crash
        # reporting, however, must also cover BV-BV and background-BV pairs.
        collision_pairs = []
        vehicles = [ego] + [a for a in self.latest_world.agents if a.valid]
        polygons = [vehicle_polygon(v) for v in vehicles]
        bounds = [(min(p[0] for p in poly), max(p[0] for p in poly),
                   min(p[1] for p in poly), max(p[1] for p in poly)) for poly in polygons]
        for i, left in enumerate(vehicles):
            for j in range(i + 1, len(vehicles)):
                right = vehicles[j]
                if left.vehicle_id == right.vehicle_id:
                    continue
                a, b = bounds[i], bounds[j]
                margin = self.collision_distance
                if a[1]+margin < b[0] or b[1]+margin < a[0] or a[3]+margin < b[2] or b[3]+margin < a[2]:
                    continue
                if polygon_distance(polygons[i], polygons[j]) <= self.collision_distance:
                    pair = sorted([str(left.vehicle_id), str(right.vehicle_id)])
                    collision_pairs.append(pair)
                    metrics.collision = True
                    self.scenario_failed = True
                    self.failure_reasons.add("collision_pair:%s:%s" % tuple(pair))
                    collision_agents.extend(v for v in pair if v != str(ego.vehicle_id))
        collision_agents = sorted(set(collision_agents))
        metrics.scenario_success = not self.scenario_failed

        if metrics.min_distance == float("inf"):
            metrics.min_distance = -1.0
        if metrics.time_to_collision == float("inf"):
            metrics.time_to_collision = -1.0

        if self.latest_platform_states and self.latest_platform_states.states:
            errors = [s.tracking_error for s in self.latest_platform_states.states if s.valid]
            metrics.platform_tracking_error = sum(errors) / len(errors) if errors else 0.0

        metrics.detail_json = json.dumps({
            "agent_count": len(self.latest_world.agents),
            "valid_agent_count": len(self.latest_world.agents) - invalid_agent_count,
            "invalid_agent_count": invalid_agent_count,
            "nearest_agent_id": nearest_agent_id,
            "min_center_distance": -1.0 if min_center_distance == float("inf") else min_center_distance,
            "collision_agents": collision_agents,
            "collision_pairs": collision_pairs,
            "collision_model": "oriented_box",
            "ttc_model": "box_clearance_centerline_closing_speed",
            "trajectory_metrics": "not_evaluated",
            "metrics": self.evaluation.get("metrics", []),
            "invalid_platform_ids": (
                invalid_platform_ids
            ),

            "invalid_platform_consecutive_frames": dict(
                self.invalid_platform_frames
            ),

            "failure_reasons": sorted(
                self.failure_reasons
            ),

            "evaluated_frame_count": (
                self.evaluated_frame_count
            ),

            "evaluation_scope": (
                "configured_and_all_active_"
                "background_vehicles"
            ),
        })
        merge_runtime_collision(metrics, self.failure_reasons)
        return metrics

    def flush_pending_world_locked(self):
        """最多评估一次最新 world/state。

        调用方必须持有 state_lock。
        """
        if (
                self.world_generation
                == self.last_evaluated_world_generation
        ):
            return None

        msg = self.compute()

        self.last_evaluated_world_generation = (
            self.world_generation
        )

        if self.latest_world is not None:
            self.history.append(msg)

            if len(self.history) > 5000:
                self.history = self.history[-5000:]

        self.metrics_pub.publish(msg)
        self.summary_pub.publish(msg)

        return msg

    def on_timer(self, _event):
        with self.state_lock:
            self.flush_pending_world_locked()

    @staticmethod
    def safe_filename_component(value):
        value = re.sub(
            r"[^A-Za-z0-9_.-]+",
            "_",
            str(value),
        )

        return value.strip("._") or "unknown"

    def build_auto_export_path_locked(self):
        """为当前场景生成稳定的自动导出路径。

        第一次调用时创建文件名；延迟收尾和 shutdown 兜底继续覆写同一个文件，
        避免一次实验生成多个内容近似的 summary。
        """
        if self.auto_export_path:
            return self.auto_export_path

        output_dir = self.resolve_project_path(
            self.export_output_dir
        )

        os.makedirs(output_dir, exist_ok=True)

        experiment_id = self.safe_filename_component(
            self.project.get("experiment_id", "exp")
        )
        scenario_id = self.safe_filename_component(
            self.current_scenario_id or "unknown"
        )
        stamp = time.strftime("%Y%m%d_%H%M%S")

        self.auto_export_path = os.path.join(
            output_dir,
            "%s_%s_%s_evaluation_summary.json"
            % (experiment_id, scenario_id, stamp),
        )
        return self.auto_export_path

    @staticmethod
    def _metric_detail_dict(metric):
        """安全解析 EvaluationMetrics.detail_json。"""
        if metric is None:
            return {}
        raw = getattr(metric, "detail_json", "")
        if not raw:
            return {}
        try:
            value = json.loads(raw)
            return value if isinstance(value, dict) else {"value": value}
        except Exception:
            return {"raw": str(raw)}

    def build_summary_data_locked(self, experiment_id):
        """根据当前场景累计状态生成可落盘的 JSON 字典。

        调用方必须持有 state_lock。该函数不再次消费 world/state，避免导出动作
        让 evaluated_frame_count 或连续 invalid 计数额外增长。
        """
        latest = self.history[-1] if self.history else None

        valid_distances = [
            item.min_distance
            for item in self.history
            if item.min_distance >= 0.0
        ]
        valid_ttc_values = [
            item.time_to_collision
            for item in self.history
            if item.time_to_collision >= 0.0
        ]

        collision = any(item.collision for item in self.history)
        scenario_success = bool(self.history) and not self.scenario_failed

        return {
            "experiment_id": str(experiment_id),
            "scenario_id": self.current_scenario_id or "",
            "samples": len(self.history),
            "evaluated_frame_count": self.evaluated_frame_count,
            "scenario_finished": bool(self.scenario_finished),
            "scenario_success": scenario_success,
            "collision": collision,
            "failure_reasons": sorted(self.failure_reasons),
            "min_distance": min(valid_distances) if valid_distances else -1.0,
            "min_time_to_collision": min(valid_ttc_values) if valid_ttc_values else -1.0,
            "ego_speed_mean": (
                sum(self.ego_speed_history) / len(self.ego_speed_history)
                if self.ego_speed_history
                else 0.0
            ),
            "ego_speed_max": max(self.ego_speed_history) if self.ego_speed_history else 0.0,
            "latest_platform_tracking_error": (
                float(latest.platform_tracking_error)
                if latest is not None
                else 0.0
            ),
            "invalid_platform_consecutive_frames": dict(
                self.invalid_platform_frames
            ),
            "latest_metrics": {
                "scenario_success": bool(latest.scenario_success) if latest is not None else False,
                "collision": bool(latest.collision) if latest is not None else False,
                "min_distance": float(latest.min_distance) if latest is not None else -1.0,
                "time_to_collision": float(latest.time_to_collision) if latest is not None else -1.0,
                "ego_speed_mean": float(latest.ego_speed_mean) if latest is not None else 0.0,
                "ego_speed_max": float(latest.ego_speed_max) if latest is not None else 0.0,
                "platform_tracking_error": (
                    float(latest.platform_tracking_error)
                    if latest is not None
                    else 0.0
                ),
                "detail": self._metric_detail_dict(latest),
            },
            "evaluation_config": {
                "collision_distance_m": self.collision_distance,
                "fail_on_invalid_platform": self.fail_on_invalid_platform,
                "platform_startup_grace_frames": self.platform_startup_grace_frames,
                "max_consecutive_invalid_platform_frames": (
                    self.max_consecutive_invalid_platform_frames
                ),
                "include_background_vehicles": bool(
                    self.evaluation.get("include_background_vehicles", True)
                ),
            },
        }

    @staticmethod
    def write_json_atomic(path, data):
        """原子写 JSON，避免 worker 被 roslaunch 关闭时留下半截文件。"""
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        temp_path = "%s.tmp.%d" % (path, os.getpid())
        try:
            with open(temp_path, "w", encoding="utf-8") as stream:
                json.dump(data, stream, ensure_ascii=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_path, path)
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    def export_summary_locked(
            self,
            auto,
            experiment_id=None,
            output_path=None,
    ):
        """导出当前场景评估摘要并返回最终文件路径。

        调用方必须持有 state_lock。
        - auto=True：使用带 scenario_id 和时间戳的稳定自动导出路径；
        - auto=False：兼容原 service 语义，把 output_path 当作目录。
        """
        resolved_experiment_id = (
            experiment_id
            or self.project.get("experiment_id", "exp")
        )

        if auto:
            path = self.build_auto_export_path_locked()
        else:
            output_dir = (
                output_path
                or rospy.get_param("/record/output_dir", "./outputs")
            )
            output_dir = self.resolve_project_path(
                output_dir
            )
            os.makedirs(output_dir, exist_ok=True)
            filename = "%s_evaluation_summary.json" % (
                self.safe_filename_component(resolved_experiment_id)
            )
            path = os.path.join(output_dir, filename)

        data = self.build_summary_data_locked(resolved_experiment_id)
        self.write_json_atomic(path, data)
        return path

    def on_shutdown(self):
        if not self.auto_export:
            return

        with self.state_lock:
            if not self.scenario_finished:
                return

            try:
                self.flush_pending_world_locked()
                self.export_summary_locked(auto=True)
            except Exception:
                rospy.logerr(
                    "Evaluation export during shutdown failed:\n%s",
                    traceback.format_exc(),
                )

    def on_export(self, req):
        """把当前场景评估摘要导出为 JSON 文件。"""
        with self.state_lock:
            try:
                self.flush_pending_world_locked()
                path = self.export_summary_locked(
                    auto=False,
                    experiment_id=req.experiment_id,
                    output_path=req.output_path,
                )
                return ExportEvaluationResponse(True, path, "exported")
            except Exception as exc:
                rospy.logerr(
                    "Manual evaluation export failed:\n%s",
                    traceback.format_exc(),
                )
                return ExportEvaluationResponse(False, "", str(exc))


if __name__ == "__main__":
    try:
        EvaluatorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
