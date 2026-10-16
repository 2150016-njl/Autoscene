#!/usr/bin/env python3
import ast
import copy
import json
import math
import py_compile
import threading
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
RESULTS = {}


def extract_methods(path, class_name, names):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    methods = {
        node.name: node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    }
    missing = set(names) - set(methods)
    if missing:
        raise AssertionError("missing methods: %s" % sorted(missing))
    module = ast.Module(body=[methods[name] for name in names], type_ignores=[])
    ast.fix_missing_locations(module)
    return module


for name in (
    "scenario_runtime_node.py",
    "inference_policy_tjwest1.py",
    "platform_comm_node.py",
):
    py_compile.compile(str(ROOT / name), doraise=True)
RESULTS["python_compile"] = "passed"

cfg = yaml.safe_load((ROOT / "hybrid.yaml").read_text(encoding="utf-8"))
assert cfg["platform"]["trajectory_mode"] == "initial_once"
assert cfg["platform"]["packet_point_num"] == 50
assert abs(float(cfg["platform"]["trajectory_dt_sec"]) - 0.1) < 1e-12
RESULTS["yaml"] = "passed"

# Test the exact fixed_initial_packet_50 method extracted from the patched source.
module = extract_methods(
    ROOT / "scenario_runtime_node.py",
    "UniTrajPlutoBackend",
    ["fixed_initial_packet_50"],
)
ns = {"np": np, "copy": copy, "math": math, "RuntimeError": RuntimeError}
exec(compile(module, "<fixed_initial_packet_50>", "exec"), ns)

class DummyBackend:
    pass

backend = DummyBackend()
backend.history_source = "initial"
backend.scenario_index = 3
backend.history_steps = 21
backend.history_dt = 0.1
backend.output_point_num = 50
backend._fixed_initial_packet_cache = {}
heading = -1.0
accel = 2.0
target_speed = 2.0
direction = np.array([math.cos(heading), math.sin(heading)])


def distance(t):
    accel_time = target_speed / accel
    if t <= accel_time:
        return 0.5 * accel * t * t
    return 0.5 * accel * accel_time * accel_time + target_speed * (t - accel_time)

final_xy = np.array([40.0, 20.0])
start_xy = final_xy - distance(2.0) * direction
startup = []
for index in range(21):
    elapsed = index * 0.1
    speed = min(accel * elapsed, target_speed)
    startup.append({
        "position": start_xy + distance(elapsed) * direction,
        "velocity": speed * direction,
        "acceleration": accel * direction if elapsed < 1.0 - 1e-12 else np.zeros(2),
        "heading": heading,
        "valid": True,
    })
startup[-1]["position"] = final_xy
startup[-1]["velocity"] = target_speed * direction
backend.startup_trajectories_by_scenario_id = {"adv_2": startup}
rows = ns["fixed_initial_packet_50"](backend, "adv_2")
arr = np.asarray(rows, dtype=np.float64)
field_speed = np.linalg.norm(arr[:, 4:6], axis=1)
interval_speed = np.linalg.norm(np.diff(arr[:, :2], axis=0), axis=1) / 0.1
assert arr.shape == (50, 8)
assert np.allclose(field_speed[:10], np.arange(1, 11) * 0.2)
assert np.allclose(field_speed[9:], 2.0)
assert np.allclose(interval_speed, (field_speed[:-1] + field_speed[1:]) * 0.5)
RESULTS["fixed_packet"] = {
    "status": "passed",
    "shape": list(arr.shape),
    "first_12_speed_mps": np.round(field_speed[:12], 6).tolist(),
    "last_speed_mps": float(field_speed[-1]),
}

# Test frame-0-only generation and confirm the Pluto path is never called.
module = extract_methods(
    ROOT / "scenario_runtime_node.py",
    "UniTrajPlutoBackend",
    ["object_plan_to_trajectory"],
)

class Header:
    pass

class Trajectory:
    def __init__(self):
        self.header = Header()
        self.vehicle_id = ""
        self.platform_id = ""
        self.source = ""
        self.frame_id = ""
        self.valid = False
        self.confidence = 0.0
        self.points = []

class ObjectPlanDummy:
    def pluto_history_time_index(self):
        return self.time_index
    def fixed_initial_packet_50(self, _scenario_id):
        self.fixed_calls += 1
        return [[0.0] * 8 for _ in range(50)]
    def save_initial_plan_csv(self, *_args):
        self.save_calls += 1
    def startup_agent_plan(self, *_args):
        raise AssertionError("rolling startup path called in initial_once")
    def pluto_agent_plan(self, *_args):
        raise AssertionError("Pluto path called in initial_once")
    def fill_trajectory_from_rows(self, traj, raw):
        traj.points = list(raw)
    def save_pluto_plan_csv(self, *_args):
        raise AssertionError("Pluto CSV path called in initial_once")

ns = {"Trajectory": Trajectory, "RuntimeError": RuntimeError}
exec(compile(module, "<object_plan_to_trajectory>", "exec"), ns)
obj = ObjectPlanDummy()
obj.frame_id = "map"
obj.trajectory_mode = "initial_once"
obj.time_index = 0
obj.fixed_calls = 0
obj.save_calls = 0
traj0 = ns["object_plan_to_trajectory"](
    obj, object(), {"id": "agent_001", "platform_id": "platform_001"}, Header(), "adv_2"
)
assert traj0.valid and len(traj0.points) == 50
assert traj0.source == "initial_fixed_frame0_50"
obj.time_index = 1
traj1 = ns["object_plan_to_trajectory"](
    obj, object(), {"id": "agent_001", "platform_id": "platform_001"}, Header(), "adv_2"
)
assert not traj1.valid and not traj1.points
assert obj.fixed_calls == 1 and obj.save_calls == 1
RESULTS["frame_zero_only_generation"] = "passed"

# Test ROS publication exactly once.
module = extract_methods(
    ROOT / "scenario_runtime_node.py",
    "ScenarioRuntimeNode",
    ["publish_platform_target_message"],
)

class RosStub:
    @staticmethod
    def loginfo(*_args, **_kwargs):
        pass

class Publisher:
    def __init__(self):
        self.count = 0
    def publish(self, _msg):
        self.count += 1

class Target:
    def __init__(self, valid=True):
        self.valid = valid
        self.points = [object()] * 50 if valid else []
        self.source = "initial_fixed_frame0_50"
        self.vehicle_id = "agent_001"

class TargetArray:
    def __init__(self, targets):
        self.trajectories = targets

class NodeDummy:
    pass

ns = {"rospy": RosStub(), "RuntimeError": RuntimeError}
exec(compile(module, "<publish_platform_target_message>", "exec"), ns)
node = NodeDummy()
node.trajectory_mode = "initial_once"
node.target_pub = Publisher()
node.scenario_id = "scenario_3"
node._initial_target_published_scenarios = set()
node.point_num = 50
ns["publish_platform_target_message"](node, TargetArray([Target()]))
ns["publish_platform_target_message"](node, TargetArray([Target(False)]))
assert node.target_pub.count == 1
try:
    ns["publish_platform_target_message"](node, TargetArray([Target()]))
except RuntimeError:
    pass
else:
    raise AssertionError("duplicate ROS target was accepted")
RESULTS["ros_publish_once"] = "passed"

# Test repeated and concurrent communication timers: only one encode/send attempt succeeds.
module = extract_methods(
    ROOT / "platform_comm_node.py",
    "PlatformCommNode",
    ["on_targets", "on_timer"],
)

class Duration:
    def __init__(self, value):
        self.value = value
    def to_sec(self):
        return self.value

class Stamp:
    def __init__(self, value=100.0):
        self.value = value
    def __sub__(self, other):
        return Duration(self.value - other.value)

class RosTime:
    @staticmethod
    def now():
        return Stamp(100.0)

class CommRosStub:
    Time = RosTime
    @staticmethod
    def logerr_throttle(*_args, **_kwargs):
        pass

class UInt8MultiArray:
    def __init__(self):
        self.data = []

class ListPublisher:
    def __init__(self):
        self.items = []
        self.lock = threading.Lock()
    def publish(self, item):
        with self.lock:
            self.items.append(item)

class CommTrajectory:
    platform_id = "platform_001"
    vehicle_id = "agent_001"
    valid = True
    points = [object()] * 50
    source = "initial_fixed_frame0_50"

class CommArray:
    trajectories = [CommTrajectory()]

class Endpoint:
    def __init__(self):
        self.platform_id = "platform_001"
        self.vehicle_id = "agent_001"
        self.detail = ""
        self.packet_counter = 0
        self.tx_count = 0
        self.last_tx = 0.0
        self.ip = "127.0.0.1"
        self.send_port = 1

class CommDummy:
    def publish_agent_feedback(self):
        pass
    def publish_states_and_status(self):
        pass
    def encode_packet(self, _traj, _endpoint):
        with self.count_lock:
            self.encode_calls += 1
        time.sleep(0.02)
        return b"abc"

ns = {
    "copy": copy,
    "rospy": CommRosStub(),
    "time": time,
    "UInt8MultiArray": UInt8MultiArray,
    "RuntimeError": RuntimeError,
}
exec(compile(module, "<platform_comm>", "exec"), ns)
comm = CommDummy()
comm.trajectory_mode = "initial_once"
comm.point_num = 50
comm.controlled_platform_by_vehicle = {"agent_001": "platform_001"}
comm.initial_once_sent_endpoints = set()
comm.initial_once_inflight_endpoints = set()
comm.latest_targets = {}
comm.latest_targets_stamp = None
comm.lock = threading.RLock()
comm.endpoints = [Endpoint()]
comm.target_timeout_sec = 1.0
comm.send_udp = False
comm.udp_sock = None
comm.payload_pub = ListPublisher()
comm.encode_calls = 0
comm.count_lock = threading.Lock()
ns["on_targets"](comm, CommArray())
threads = [threading.Thread(target=ns["on_timer"], args=(comm, None)) for _ in range(2)]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
ns["on_timer"](comm, None)
assert comm.encode_calls == 1
assert len(comm.payload_pub.items) == 1
assert comm.endpoints[0].packet_counter == 1
assert comm.initial_once_sent_endpoints == {"platform_001"}
assert not comm.initial_once_inflight_endpoints
assert comm.latest_targets == {} and comm.latest_targets_stamp is None
try:
    ns["on_targets"](comm, CommArray())
except RuntimeError:
    pass
else:
    raise AssertionError("duplicate UDP target was accepted")
RESULTS["udp_send_once"] = {
    "status": "passed",
    "encode_calls": comm.encode_calls,
    "payload_publications": len(comm.payload_pub.items),
    "concurrent_timer_threads": 2,
}

# Static contract checks for the replanning path.
policy_source = (ROOT / "inference_policy_tjwest1.py").read_text(encoding="utf-8")
assert "planning_phase = int(time_index) - int(self.WARMUP_STEPS)" in policy_source
assert "score_selections=score_selections" in policy_source
assert "self.cbv_replanning_enabled" in policy_source
RESULTS["replanning_contract"] = "passed"

(ROOT / "validation.json").write_text(
    json.dumps(RESULTS, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(json.dumps(RESULTS, ensure_ascii=False, indent=2))
