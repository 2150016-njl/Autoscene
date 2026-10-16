#!/usr/bin/env python3
"""Verify fixed G<->A SE(2) coordinate transforms.

G is the projected-earth coordinate used by platform UDP. A is the
AutoScene/MetaDrive world frame. This script is offline and does not start ROS.
"""
import math
import os
import random
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AUTOSCENE_SRC = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
PLATFORM_SCRIPTS = os.path.join(AUTOSCENE_SRC, "autoscene_platform", "scripts")
if PLATFORM_SCRIPTS not in sys.path:
    sys.path.insert(0, PLATFORM_SCRIPTS)

from geo_projection import (  # noqa: E402
    EarthToAutoSceneTransform,
    earth_yaw_to_ins_heading_deg,
    estimate_earth_to_autoscene,
    ins_heading_deg_to_earth_yaw,
    wrap_rad,
)


EPS_POS = 1e-8
EPS_YAW = 1e-9


def yaw_err(a, b):
    return abs(wrap_rad(a - b))


def assert_pose(a, b, label):
    assert abs(a["x"] - b["x"]) < EPS_POS, (label, "x", a, b)
    assert abs(a["y"] - b["y"]) < EPS_POS, (label, "y", a, b)
    assert yaw_err(a["yaw"], b["yaw"]) < EPS_YAW, (label, "yaw", a, b)


def main():
    transforms = [
        EarthToAutoSceneTransform(yaw=math.radians(12.0), tx=-520000.0, ty=-3420000.0),
        EarthToAutoSceneTransform(yaw=math.radians(-37.0), tx=103.5, ty=-77.2),
    ]

    random.seed(20260728)
    for _ in range(100):
        transforms.append(
            EarthToAutoSceneTransform(
                yaw=random.uniform(-math.pi, math.pi),
                tx=random.uniform(-1.0e6, 1.0e6),
                ty=random.uniform(-5.0e6, 5.0e6),
            )
        )

    for ti, transform in enumerate(transforms):
        for pi in range(20):
            earth = {
                "x": random.uniform(450000.0, 550000.0),
                "y": random.uniform(3.0e6, 4.0e6),
                "yaw": random.uniform(-math.pi, math.pi),
            }
            ax, ay = transform.point_g_to_a(earth["x"], earth["y"])
            ayaw = transform.yaw_g_to_a(earth["yaw"])
            gx, gy = transform.point_a_to_g(ax, ay)
            gyaw = transform.yaw_a_to_g(ayaw)
            assert_pose(
                {"x": gx, "y": gy, "yaw": gyaw},
                earth,
                "roundtrip transform=%d pose=%d" % (ti, pi),
            )

    heading_cases = [
        (0.0, math.pi / 2.0),
        (90.0, 0.0),
        (180.0, -math.pi / 2.0),
        (270.0, -math.pi),
    ]
    for heading_deg, expected_yaw in heading_cases:
        yaw = ins_heading_deg_to_earth_yaw(heading_deg)
        assert yaw_err(yaw, expected_yaw) < EPS_YAW, (heading_deg, yaw)
        recovered = earth_yaw_to_ins_heading_deg(yaw)
        assert abs((recovered - heading_deg) % 360.0) < 1e-9, (heading_deg, recovered)

    base = EarthToAutoSceneTransform(yaw=math.radians(8.0), tx=-500123.0, ty=-3500456.0)
    pairs = []
    for earth_x, earth_y, earth_yaw in [
        (500000.0, 3500000.0, math.radians(0.0)),
        (500050.0, 3500010.0, math.radians(0.1)),
        (499980.0, 3500060.0, math.radians(-0.2)),
    ]:
        ax, ay = base.point_g_to_a(earth_x, earth_y)
        pairs.append(
            {
                "earth": {"x": earth_x, "y": earth_y, "yaw": earth_yaw},
                "autoscene": {"x": ax, "y": ay, "yaw": base.yaw_g_to_a(earth_yaw)},
            }
        )
    estimated, diag = estimate_earth_to_autoscene(pairs)
    assert yaw_err(estimated.yaw, base.yaw) < 1e-10, (estimated.yaw, base.yaw)
    assert abs(estimated.tx - base.tx) < 1e-4, (estimated.tx, base.tx)
    assert abs(estimated.ty - base.ty) < 1e-4, (estimated.ty, base.ty)
    assert diag["position_rms_m"] < 1e-8, diag
    assert diag["yaw_max_abs_rad"] < 1e-10, diag

    print("PASS: G<->A transform, INS heading conversion, and calibration estimate")


if __name__ == "__main__":
    main()
