#!/usr/bin/env python3
"""不依赖 ROS/MetaDrive 的补丁纯逻辑测试。"""
import ast
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def extract_backend_methods():
    tree = ast.parse((ROOT / "scenario_runtime_node.py").read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "UniTrajPlutoBackend"
    )
    methods = {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)}
    selected = [
        methods[name]
        for name in (
            "required_startup_steps",
            "build_accel_trajectory",
            "build_accel_history",
            "startup_agent_plan",
            "pluto_history_time_index",
        )
    ]
    module = ast.Module(
        body=[ast.ClassDef(name="Backend", bases=[], keywords=[], body=selected, decorator_list=[])],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {"math": math}
    exec(compile(module, "<backend-tests>", "exec"), namespace)
    return namespace["Backend"]


def extract_quantize():
    tree = ast.parse((ROOT / "platform_comm_node.py").read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "quantize"
    )
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"math": math}
    exec(compile(module, "<quantize-tests>", "exec"), namespace)
    return namespace["quantize"]


def main():
    Backend = extract_backend_methods()
    backend = Backend()
    backend.history_steps = 21
    backend.history_dt = 0.1
    backend.algorithm_cfg = {"initial_history_accel_mps2": 2.0}

    low = backend.build_accel_trajectory(
        {"x": 10.0, "y": 5.0, "heading": 0.0, "speed": 2.4}
    )
    assert len(low) == 21
    assert all(np.linalg.norm(entry["velocity"]) < 1e-12 for entry in low[:9])
    assert abs(np.linalg.norm(low[9]["velocity"]) - 0.2) < 1e-9
    assert np.allclose(low[-1]["position"], [10.0, 5.0])

    high = backend.build_accel_trajectory(
        {"x": 0.0, "y": 0.0, "heading": 0.0, "speed": 5.0}
    )
    assert len(high) == 26
    assert np.linalg.norm(high[0]["velocity"]) < 1e-12
    assert len(backend.build_accel_history(
        {"x": 0.0, "y": 0.0, "heading": 0.0, "speed": 5.0}
    )) == 21

    synchronized = backend.build_accel_trajectory(
        {"x": 10.0, "y": 5.0, "heading": 0.0, "speed": 2.4},
        total_steps=26,
    )
    assert all(np.linalg.norm(entry["velocity"]) < 1e-12 for entry in synchronized[:14])

    backend.history_source = "initial"
    backend.output_point_num = 50
    backend.startup_trajectories_by_scenario_id = {"track": high}
    backend.env = None
    backend.step_index = 0
    rows = backend.startup_agent_plan("track")
    assert len(rows) == 50
    assert np.allclose(rows[25][:2], high[-1]["position"])

    quantize = extract_quantize()
    assert quantize(1.23, 0.01, -32768, 32767) == 123
    try:
        quantize(1000.0, 0.01, -32768, 32767)
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-range quantize must fail")

    print("all patch logic tests passed")


if __name__ == "__main__":
    main()
