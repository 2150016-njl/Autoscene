#!/usr/bin/env python3
"""平台大地投影坐标与 AutoScene 坐标的基础工具。

约定：
- G 坐标系：由平台上游经纬度按 `LatLon2XY.m` 解算得到的高斯投影坐标，
  x 为 easting，y 为 northing，单位 m。
- A 坐标系：AutoScene/MetaDrive world/map 坐标，单位 m，yaw=0 指向 +x，
  逆时针为正。
- 上游 INS/GNSS 航向：北向 0 deg，顺时针为正。
"""
import math


WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563


def wrap_rad(angle):
    """把弧度角归一化到 [-pi, pi)。"""
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def normalize_deg(angle):
    """把角度归一化到 [0, 360)。"""
    value = math.fmod(float(angle), 360.0)
    return value + 360.0 if value < 0.0 else value


def lat_to_meridian_arc(latitude_rad, semi_major, eccentricity):
    """按 `LatLon2XY.m::Lat2Mer` 计算子午线弧长。"""
    m0 = semi_major * (1.0 - eccentricity ** 2)
    m2 = 1.5 * eccentricity ** 2 * m0
    m4 = 1.25 * eccentricity ** 2 * m2
    m6 = (7.0 / 6.0) * eccentricity ** 2 * m4
    m8 = 1.125 * eccentricity ** 2 * m6

    a0 = m0 + m2 / 2.0 + 3.0 * m4 / 8.0 + 5.0 * m6 / 16.0 + 35.0 * m8 / 128.0
    a2 = m2 / 2.0 + m4 / 2.0 + 15.0 * m6 / 32.0 + 7.0 * m8 / 16.0
    a4 = m4 / 8.0 + 3.0 * m6 / 16.0 + 7.0 * m8 / 32.0
    a6 = m6 / 32.0 + m8 / 16.0
    a8 = m8 / 128.0

    b = latitude_rad
    return (
        a0 * b
        - a2 * math.sin(2.0 * b) / 2.0
        + a4 * math.sin(4.0 * b) / 4.0
        - a6 * math.sin(6.0 * b) / 6.0
        + a8 * math.sin(8.0 * b) / 8.0
    )


def latlon_to_xy(latitude_deg, longitude_deg):
    """复刻 `LatLon2XY.m` 的 3 度带高斯正算。

    返回 `(earth_x, earth_y)`：
    - `earth_x = easting + 500000`
    - `earth_y = northing / meridian arc`

    本函数只接受十进制度经纬度，不处理度分格式。
    """
    lat = math.radians(float(latitude_deg))
    lon = math.radians(float(longitude_deg))

    semi_major = WGS84_A
    flattening = WGS84_F
    semi_minor = semi_major * (1.0 - flattening)
    eccentricity = math.sqrt((semi_major ** 2 - semi_minor ** 2) / semi_major ** 2)
    second_eccentricity = math.sqrt((semi_major ** 2 - semi_minor ** 2) / semi_minor ** 2)

    zone = math.trunc(math.degrees(lon) / 3.0)
    central_lon = math.radians(3.0 * zone)
    lon_delta = lon - central_lon

    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    tan_lat = math.tan(lat)
    prime_vertical_radius = semi_major / math.sqrt(1.0 - eccentricity ** 2 * sin_lat ** 2)
    eta = second_eccentricity * cos_lat
    meridian_arc = lat_to_meridian_arc(lat, semi_major, eccentricity)

    earth_y = (
        meridian_arc
        + prime_vertical_radius / 2.0 * sin_lat * cos_lat * lon_delta ** 2
        + prime_vertical_radius / 24.0 * sin_lat * cos_lat ** 3
        * (5.0 - tan_lat ** 2 + 9.0 * eta ** 2 + 4.0 * eta ** 4)
        * lon_delta ** 4
        + prime_vertical_radius / 720.0 * sin_lat * cos_lat ** 5
        * (61.0 - 58.0 * tan_lat ** 2 + tan_lat ** 4 + 270.0 * eta ** 2 - 330.0 * eta ** 2 * tan_lat ** 2)
        * lon_delta ** 6
    )
    earth_x = (
        prime_vertical_radius * cos_lat * lon_delta
        + prime_vertical_radius / 6.0 * cos_lat ** 3 * (1.0 - tan_lat ** 2 + eta ** 2) * lon_delta ** 3
        + prime_vertical_radius / 120.0 * cos_lat ** 5
        * (5.0 - 18.0 * tan_lat ** 2 + tan_lat ** 4 + 14.0 * eta ** 2 - 58.0 * eta ** 2 * tan_lat ** 2)
        * lon_delta ** 5
    )
    return earth_x + 500000.0, earth_y


def ins_heading_deg_to_earth_yaw(heading_deg):
    """INS/GNSS 航向转 G 坐标数学 yaw。"""
    return wrap_rad(math.radians(90.0 - float(heading_deg)))


def earth_yaw_to_ins_heading_deg(earth_yaw):
    """G 坐标数学 yaw 转 INS/GNSS 航向。"""
    return normalize_deg(90.0 - math.degrees(float(earth_yaw)))


class EarthToAutoSceneTransform:
    """固定 SE(2) 变换：`p_A = R(yaw) * p_G + [tx, ty]`。"""

    def __init__(self, yaw=0.0, tx=0.0, ty=0.0):
        self.yaw = float(yaw)
        self.tx = float(tx)
        self.ty = float(ty)

    @classmethod
    def from_config(cls, cfg):
        cfg = cfg or {}
        return cls(
            yaw=float(cfg.get("yaw", cfg.get("heading", 0.0))),
            tx=float(cfg.get("tx", cfg.get("x", 0.0))),
            ty=float(cfg.get("ty", cfg.get("y", 0.0))),
        )

    def rotate_g_to_a(self, x, y):
        c = math.cos(self.yaw)
        s = math.sin(self.yaw)
        return c * x - s * y, s * x + c * y

    def rotate_a_to_g(self, x, y):
        c = math.cos(self.yaw)
        s = math.sin(self.yaw)
        return c * x + s * y, -s * x + c * y

    def point_g_to_a(self, x, y):
        rx, ry = self.rotate_g_to_a(float(x), float(y))
        return self.tx + rx, self.ty + ry

    def point_a_to_g(self, x, y):
        return self.rotate_a_to_g(float(x) - self.tx, float(y) - self.ty)

    def yaw_g_to_a(self, yaw):
        return wrap_rad(self.yaw + float(yaw))

    def yaw_a_to_g(self, yaw):
        return wrap_rad(float(yaw) - self.yaw)


def estimate_earth_to_autoscene(pairs):
    """由多组同名姿态点估计 G->A 变换。

    `pairs` 元素格式：
    `{ "earth": {"x": .., "y": .., "yaw": ..},
       "autoscene": {"x": .., "y": .., "yaw": ..} }`

    1 个点时使用航向解析解；多个点时用位置 SVD 最小二乘估计旋转和平移，
    再用航向残差评估质量。返回 `(transform, diagnostics)`。
    """
    if not pairs:
        raise ValueError("at least one calibration pair is required")

    if len(pairs) == 1:
        earth = pairs[0]["earth"]
        autoscene = pairs[0]["autoscene"]
        yaw = wrap_rad(float(autoscene["yaw"]) - float(earth["yaw"]))
        c = math.cos(yaw)
        s = math.sin(yaw)
        tx = float(autoscene["x"]) - (c * float(earth["x"]) - s * float(earth["y"]))
        ty = float(autoscene["y"]) - (s * float(earth["x"]) + c * float(earth["y"]))
        transform = EarthToAutoSceneTransform(yaw=yaw, tx=tx, ty=ty)
    else:
        import numpy as np

        g = np.asarray([[p["earth"]["x"], p["earth"]["y"]] for p in pairs], dtype=np.float64)
        a = np.asarray([[p["autoscene"]["x"], p["autoscene"]["y"]] for p in pairs], dtype=np.float64)
        g_mean = g.mean(axis=0)
        a_mean = a.mean(axis=0)
        h = (g - g_mean).T @ (a - a_mean)
        u, _s, vt = np.linalg.svd(h)
        r = vt.T @ u.T
        if np.linalg.det(r) < 0.0:
            vt[-1, :] *= -1.0
            r = vt.T @ u.T
        yaw = math.atan2(float(r[1, 0]), float(r[0, 0]))
        t = a_mean - r @ g_mean
        transform = EarthToAutoSceneTransform(yaw=yaw, tx=float(t[0]), ty=float(t[1]))

    pos_errors = []
    yaw_errors = []
    for pair in pairs:
        ex, ey = transform.point_g_to_a(pair["earth"]["x"], pair["earth"]["y"])
        pos_errors.append(math.hypot(ex - float(pair["autoscene"]["x"]), ey - float(pair["autoscene"]["y"])))
        yaw_errors.append(wrap_rad(float(pair["autoscene"]["yaw"]) - transform.yaw_g_to_a(pair["earth"]["yaw"])))

    diagnostics = {
        "point_count": len(pairs),
        "position_rms_m": math.sqrt(sum(e * e for e in pos_errors) / len(pos_errors)),
        "position_max_m": max(pos_errors),
        "yaw_max_abs_rad": max(abs(e) for e in yaw_errors),
    }
    return transform, diagnostics

def autoscene_start_pose_from_initial(
        initial_x,
        initial_y,
        initial_yaw,
        initial_speed,
        history_steps=21,
        history_dt_sec=0.1,
        accel_mps2=2.0,
):
    """由 runtime 的 initial 历史模型倒推出直线启动起点 A 姿态。

    与 scenario_runtime_node.py::build_accel_trajectory() 保持同一运动学语义：
    - 共 history_steps 个采样点；
    - 相邻点间隔 history_dt_sec；
    - 第 0 点速度为 0；
    - 先以 accel_mps2 匀加速到 initial_speed，再匀速；
    - 最后一点严格等于用户配置的 initial x/y/yaw/speed。

    返回 `(start_pose, diagnostics)`，其中 start_pose 含 x/y/yaw/speed。
    """
    steps = int(history_steps)
    dt = float(history_dt_sec)
    accel = float(accel_mps2)
    speed = float(initial_speed)
    yaw = wrap_rad(float(initial_yaw))

    if steps < 2:
        raise ValueError("history_steps must be >= 2")
    if dt <= 0.0:
        raise ValueError("history_dt_sec must be > 0")
    if accel <= 0.0:
        raise ValueError("accel_mps2 must be > 0")
    if speed < 0.0:
        raise ValueError("initial_speed must be non-negative")

    total_time = dt * float(steps - 1)
    max_speed = accel * total_time
    if speed > max_speed + 1e-9:
        raise ValueError(
            "initial_speed %.6f exceeds startup history limit %.6f m/s "
            "(steps=%d dt=%.6f accel=%.6f)"
            % (speed, max_speed, steps, dt, accel)
        )

    if speed <= 0.0:
        accel_time = 0.0
        total_distance = 0.0
    else:
        accel_time = speed / accel
        if total_time <= accel_time:
            total_distance = 0.5 * accel * total_time * total_time
        else:
            total_distance = (
                0.5 * accel * accel_time * accel_time
                + speed * (total_time - accel_time)
            )

    start_x = float(initial_x) - total_distance * math.cos(yaw)
    start_y = float(initial_y) - total_distance * math.sin(yaw)

    return (
        {
            "x": start_x,
            "y": start_y,
            "yaw": yaw,
            "speed": 0.0,
        },
        {
            "history_steps": steps,
            "history_dt_sec": dt,
            "accel_mps2": accel,
            "initial_speed_mps": speed,
            "total_time_sec": total_time,
            "accel_time_sec": accel_time,
            "backtrack_distance_m": total_distance,
        },
    )


def main():
    """单姿态 G->A 标定命令行；只打印结果，不修改配置文件。"""
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Estimate platform.earth_to_autoscene from one physical platform "
            "start pose in earth G and its corresponding AutoScene A pose."
        )
    )
    earth_group = parser.add_mutually_exclusive_group(required=True)
    earth_group.add_argument(
        "--earth-pose",
        nargs=3,
        type=float,
        metavar=("EARTH_X_M", "EARTH_Y_M", "INS_HEADING_DEG"),
        help="Physical platform start pose in projected G coordinates.",
    )
    earth_group.add_argument(
        "--latlon-pose",
        nargs=3,
        type=float,
        metavar=("LAT_DEG", "LON_DEG", "INS_HEADING_DEG"),
        help="Physical platform start pose in latitude/longitude; converted by LatLon2XY logic.",
    )

    auto_group = parser.add_mutually_exclusive_group(required=True)
    auto_group.add_argument(
        "--autoscene-pose",
        nargs=3,
        type=float,
        metavar=("A_X_M", "A_Y_M", "A_YAW_RAD"),
        help=(
            "AutoScene pose of the SAME physical point as --earth-pose/--latlon-pose. "
            "Use this only when no startup backtracking is needed."
        ),
    )
    auto_group.add_argument(
        "--autoscene-initial-pose",
        nargs=4,
        type=float,
        metavar=("INITIAL_A_X_M", "INITIAL_A_Y_M", "INITIAL_A_YAW_RAD", "INITIAL_SPEED_MPS"),
        help=(
            "Configured AutoScene initial endpoint. The tool backtracks the runtime "
            "startup line to obtain the corresponding physical platform start pose "
            "before estimating earth_to_autoscene."
        ),
    )
    parser.add_argument(
        "--history-steps",
        type=int,
        default=21,
        help="Startup/history sample count; must match algorithm.pluto_history_steps (default: 21).",
    )
    parser.add_argument(
        "--history-dt-sec",
        type=float,
        default=0.1,
        help="Startup/history sample interval; must match algorithm.pluto_history_dt_sec (default: 0.1).",
    )
    parser.add_argument(
        "--initial-history-accel-mps2",
        type=float,
        default=2.0,
        help="Startup acceleration; must match algorithm.initial_history_accel_mps2 (default: 2.0).",
    )
    args = parser.parse_args()

    if args.earth_pose is not None:
        earth_x, earth_y, ins_heading_deg = args.earth_pose
    else:
        latitude_deg, longitude_deg, ins_heading_deg = args.latlon_pose
        earth_x, earth_y = latlon_to_xy(latitude_deg, longitude_deg)

    startup_diagnostics = None
    if args.autoscene_initial_pose is not None:
        initial_x, initial_y, initial_yaw, initial_speed = args.autoscene_initial_pose
        start_pose, startup_diagnostics = autoscene_start_pose_from_initial(
            initial_x,
            initial_y,
            initial_yaw,
            initial_speed,
            history_steps=args.history_steps,
            history_dt_sec=args.history_dt_sec,
            accel_mps2=args.initial_history_accel_mps2,
        )
        auto_x = start_pose["x"]
        auto_y = start_pose["y"]
        auto_yaw = start_pose["yaw"]
    else:
        auto_x, auto_y, auto_yaw = args.autoscene_pose

    earth_yaw = ins_heading_deg_to_earth_yaw(ins_heading_deg)
    transform, diagnostics = estimate_earth_to_autoscene(
        [
            {
                "earth": {"x": earth_x, "y": earth_y, "yaw": earth_yaw},
                "autoscene": {"x": auto_x, "y": auto_y, "yaw": auto_yaw},
            }
        ]
    )
    check_x, check_y = transform.point_g_to_a(earth_x, earth_y)
    check_yaw = transform.yaw_g_to_a(earth_yaw)

    print("platform:")
    print("  earth_to_autoscene:")
    print("    tx: %.15f" % transform.tx)
    print("    ty: %.15f" % transform.ty)
    print("    yaw: %.15f" % transform.yaw)
    print("# yaw unit: rad")
    print("# earth physical start: x=%.12f y=%.12f heading_deg=%.9f" % (
        earth_x, earth_y, ins_heading_deg
    ))
    if startup_diagnostics is not None:
        print(
            "# AutoScene configured initial: x=%.12f y=%.12f yaw_rad=%.12f speed_mps=%.12f"
            % (initial_x, initial_y, wrap_rad(initial_yaw), initial_speed)
        )
        print(
            "# startup backtrack: steps=%d dt=%.12f accel=%.12f total_time=%.12f "
            "accel_time=%.12f distance=%.12f"
            % (
                startup_diagnostics["history_steps"],
                startup_diagnostics["history_dt_sec"],
                startup_diagnostics["accel_mps2"],
                startup_diagnostics["total_time_sec"],
                startup_diagnostics["accel_time_sec"],
                startup_diagnostics["backtrack_distance_m"],
            )
        )
        print(
            "# derived AutoScene physical start: x=%.12f y=%.12f yaw_rad=%.12f speed_mps=0"
            % (auto_x, auto_y, auto_yaw)
        )
    else:
        print("# AutoScene physical start supplied directly.")
    print("# check A start: x=%.12f y=%.12f yaw_rad=%.12f" % (check_x, check_y, check_yaw))
    print(
        "# residual: position_rms_m=%.12g yaw_max_abs_rad=%.12g"
        % (diagnostics["position_rms_m"], diagnostics["yaw_max_abs_rad"])
    )


if __name__ == "__main__":
    main()