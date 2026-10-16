#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离线计算平台 G->A 标定和 ego external origin。

用法示例：
python3 calibrate_origins.py \
  --config ../../autoscene_bringup/config/hybrid.yaml \
  --platform-pose 512345.0 4312345.0 90.0 \
  --platform-heading-convention ins_deg \
  --ego-pose 0.0 0.0 0.0 \
  --ego-heading-convention yaw_rad

输入 pose 均为 x y heading

平台 x/y 是经纬度解算后的高斯投影坐标，heading 默认使用 ins_deg；
ego heading 默认使用 yaw_rad；
也可以分别通过 heading-convention 参数指定。

脚本只打印 YAML 片段，不直接改配置文件，避免破坏手写注释。
"""
import argparse
import importlib.util
import math
import os
import sys


PLATFORM_SCRIPT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "autoscene_platform", "scripts")
)
if PLATFORM_SCRIPT_DIR not in sys.path:
    sys.path.insert(0, PLATFORM_SCRIPT_DIR)

from geo_projection import estimate_earth_to_autoscene, ins_heading_deg_to_earth_yaw, latlon_to_xy


def wrap_rad(angle):
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def compute_origin(autoscene_pose, feedback_pose):
    """由同一车辆在 AutoScene 坐标和反馈坐标中的 pose 反推 origin。"""
    origin_yaw = wrap_rad(autoscene_pose["heading"] - feedback_pose["heading"])
    c = math.cos(origin_yaw)
    s = math.sin(origin_yaw)
    rotated_x = c * feedback_pose["x"] - s * feedback_pose["y"]
    rotated_y = s * feedback_pose["x"] + c * feedback_pose["y"]
    return {
        "x": autoscene_pose["x"] - rotated_x,
        "y": autoscene_pose["y"] - rotated_y,
        "yaw": origin_yaw,
    }


def make_platform_calibration_pair(earth_pose, autoscene_pose):
    """生成 geo_projection.estimate_earth_to_autoscene() 使用的姿态点。"""
    return {
        "earth": {
            "x": float(earth_pose["x"]),
            "y": float(earth_pose["y"]),
            "yaw": float(earth_pose["heading"]),
        },
        "autoscene": {
            "x": float(autoscene_pose["x"]),
            "y": float(autoscene_pose["y"]),
            "yaw": float(autoscene_pose["heading"]),
        },
    }


def relative_heading(heading, ego_heading):
    return wrap_rad(float(heading) - float(ego_heading))


def parse_pose(
        values,
        heading_convention,
):
    x = float(values[0])
    y = float(values[1])
    raw_heading = float(values[2])

    if heading_convention == "yaw_rad":
        heading = raw_heading

    elif heading_convention == "yaw_deg":
        heading = math.radians(
            raw_heading
        )

    elif heading_convention == "ins_deg":
        heading = wrap_rad(
            math.radians(
                90.0 - raw_heading
            )
        )

    else:
        raise ValueError(
            "unsupported heading convention: %s"
            % heading_convention
        )

    return {
        "x": x,
        "y": y,
        "heading": wrap_rad(heading),
    }


def resolve_path(path, base_dir):
    if not path or os.path.isabs(path):
        return path

    candidates = [
        os.path.abspath(path),
        os.path.abspath(os.path.join(base_dir, path)),
        os.path.abspath(os.path.join(base_dir, "..", path)),
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    return candidates[0]


def load_yaml(path):
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("PyYAML is required: %s" % exc)

    with open(path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def load_closeloop_env(config, autoscene_root, scenario_index):
    """Load the import-safe Autoscene_generator environment factory."""
    scenario_cfg = config.get("scenario", {})
    unitraj_root = resolve_path(
        scenario_cfg.get(
            "unitraj_root",
            "./src/autoscene_runtime/scripts/Autoscene_generator",
        ),
        autoscene_root,
    )
    unitraj_config = resolve_path(
        scenario_cfg.get(
            "unitraj_config",
            os.path.join(unitraj_root, "unitraj", "configs", "config.yaml"),
        ),
        autoscene_root,
    )
    os.environ["AUTOSCENE_DATASET_TYPE"] = str(
        scenario_cfg.get("dataset_type", "tjwest")
    ).strip().lower()
    dataset_path = str(scenario_cfg.get("dataset_path", "")).strip()
    if dataset_path:
        os.environ["AUTOSCENE_DATASET_PATH"] = resolve_path(dataset_path, autoscene_root)

    if unitraj_root not in sys.path:
        sys.path.insert(0, unitraj_root)
    for module_name in (
        "unitraj.closeloop.pluto.pluto_policy",
        "unitraj.datasets.Pluto_dataset.Pluto_test_dataset",
    ):
        sys.modules.pop(module_name, None)

    module_path = os.path.join(unitraj_root, "unitraj", "closeloop_sim.py")
    spec = importlib.util.spec_from_file_location(
        "autoscene_calibration_closeloop",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load closeloop_sim.py: %s" % module_path)

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    env = None
    try:
        env = module.create_env(scenario_index + 1, unitraj_config)
        env.reset(seed=scenario_index)
        return env
    except Exception:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        raise


def ego_pose_from_engine(engine, frame_index=None):
    """读取固定 AutoScene A 坐标系中的 ego 姿态。

    frame_index=None 保留旧行为：读取 MetaDrive 当前对象状态。
    显式传入 frame_index 时，从 current_scenario 的 SDC track 读取该帧；
    manager 用它读取第 21 帧，从而统一候选显示、排序和 initial 默认值。

    本函数只做离线场景状态读取，不参与经纬度投影或 G<->A 转换；
    正式坐标转换仍由 geo_projection.py 负责。
    """
    if frame_index is not None:
        import numpy as np

        scenario = getattr(
            getattr(engine, "data_manager", None),
            "current_scenario",
            {},
        ) or {}
        tracks = scenario.get("tracks", {}) if isinstance(scenario, dict) else {}
        sdc_id = str(scenario.get("metadata", {}).get("sdc_id", ""))
        track = tracks.get(sdc_id)
        if not isinstance(track, dict):
            raise RuntimeError("AutoScene SDC track is missing")

        state = track.get("state", {})
        position = np.asarray(state.get("position", []), dtype=float)
        heading = np.asarray(state.get("heading", []), dtype=float).reshape(-1)
        velocity = np.asarray(state.get("velocity", []), dtype=float)
        valid = np.asarray(state.get("valid", []), dtype=bool).reshape(-1)
        index = int(frame_index)

        if (
            index < 0
            or position.ndim != 2
            or position.shape[1] < 2
            or index >= len(position)
            or index >= len(heading)
            or index >= len(valid)
            or not bool(valid[index])
        ):
            raise RuntimeError(
                "SDC track has no valid state at frame_index=%d" % index
            )

        speed = 0.0
        if velocity.ndim >= 2 and index < len(velocity):
            speed = float(
                math.hypot(
                    float(velocity[index, 0]),
                    float(velocity[index, 1]),
                )
            )

        return {
            "x": float(position[index, 0]),
            "y": float(position[index, 1]),
            "heading": float(heading[index]),
            "speed": speed,
        }

    ego_obj = getattr(engine, "agents", {}).get("default_agent")

    if ego_obj is None:
        raise RuntimeError("MetaDrive default_agent is missing")

    position = getattr(ego_obj, "position", None)

    if position is None:
        raise RuntimeError("MetaDrive default_agent has no position")

    return {
        "x": float(position[0]),
        "y": float(position[1]),
        "heading": float(getattr(ego_obj, "heading_theta", 0.0)),
        "speed": float(getattr(ego_obj, "speed", 0.0)),
    }


def sorted_bv_candidates(engine, frame_index=0):
    import numpy as np

    scenario = engine.data_manager.current_scenario
    tracks = scenario.get("tracks", {})
    sdc_id = str(scenario.get("metadata", {}).get("sdc_id", ""))
    frame_index = int(frame_index)
    ego_pose = ego_pose_from_engine(engine, frame_index=frame_index)
    ego_xy = np.array([ego_pose["x"], ego_pose["y"]], dtype=float)
    ego_heading = ego_pose["heading"]
    ego_forward = np.array([math.cos(ego_heading), math.sin(ego_heading)], dtype=float)
    ego_left = np.array([-math.sin(ego_heading), math.cos(ego_heading)], dtype=float)
    raw_mapping = getattr(engine.traffic_manager, "obj_id_to_scenario_id", {}) or {}
    track_to_object = {
        str(track_id): str(object_id)
        for object_id, track_id in raw_mapping.items()
    }
    engine_objects = {
        str(object_id): obj
        for object_id, obj in engine.get_objects().items()
    }
    candidates = []

    for track_id, track in tracks.items():
        track_id = str(track_id)

        if track_id == sdc_id:
            continue

        state = track.get("state", {})
        position = np.asarray(state.get("position", []), dtype=float)

        if position.ndim < 2 or len(position) == 0:
            continue

        valid = np.asarray(
            state.get(
                "valid",
                np.ones(len(position), dtype=bool),
            )
        ).astype(bool).reshape(-1)
        valid_indexes = np.flatnonzero(valid)

        if (
            len(valid_indexes) == 0
            or int(valid_indexes[0]) != 0
            or frame_index < 0
            or frame_index >= len(valid)
            or not bool(valid[frame_index])
        ):
            continue

        object_id = track_to_object.get(track_id)

        if object_id is None or engine_objects.get(object_id) is None:
            continue

        heading_arr = np.asarray(
            state.get("heading", np.zeros(len(position))),
            dtype=float,
        ).reshape(-1)
        velocity = np.asarray(
            state.get("velocity", np.zeros((len(position), 2))),
            dtype=float,
        )
        if (
            frame_index >= len(position)
            or frame_index >= len(heading_arr)
        ):
            continue

        x = float(position[frame_index, 0])
        y = float(position[frame_index, 1])
        heading = float(heading_arr[frame_index])
        speed = 0.0

        if velocity.ndim >= 2 and frame_index < len(velocity):
            speed = float(
                math.hypot(
                    float(velocity[frame_index, 0]),
                    float(velocity[frame_index, 1]),
                )
            )

        relative_world = np.array([x, y], dtype=float) - ego_xy
        relative_longitudinal = float(np.dot(relative_world, ego_forward))
        relative_lateral = float(np.dot(relative_world, ego_left))
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
                # 这些字段只用于 manager 人工查看，不改变原候选过滤和排序规则。
                "type": str(track.get("type", "")),
                "first_valid": int(valid_indexes[0]),
                "last_valid": int(valid_indexes[-1]),
                "valid_steps": int(len(valid_indexes)),
                "x": x,
                "y": y,
                "heading": heading,
                "speed": speed,
                "relative_longitudinal": relative_longitudinal,
                "relative_lateral": relative_lateral,
                "relative_heading": relative_heading(heading, ego_heading),
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


def print_origin_block(origin, indent="  "):
    print("%sorigin:" % indent)
    print("%s  x: %.12f" % (indent, origin["x"]))
    print("%s  y: %.12f" % (indent, origin["y"]))
    print("%s  yaw: %.12f" % (indent, origin["yaw"]))


def print_earth_to_autoscene_block(transform, indent="  "):
    print("%searth_to_autoscene:" % indent)
    print("%s  tx: %.12f" % (indent, transform.tx))
    print("%s  ty: %.12f" % (indent, transform.ty))
    print("%s  yaw: %.12f" % (indent, transform.yaw))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute platform.earth_to_autoscene for projected-earth "
            "platform feedback and ego.origin for external ego feedback."
        )
    )
    parser.add_argument("--config", required=True, help="AutoScene YAML config path.")
    parser.add_argument("--scenario-index", type=int, default=None)
    parser.add_argument("--platform-pose", nargs=3, type=float, required=False, metavar=("EARTH_X", "EARTH_Y", "HEADING"))
    parser.add_argument("--platform-lla", nargs=3, type=float, required=False, metavar=("LAT", "LON", "HEADING"))
    parser.add_argument(
        "--platform-pair",
        nargs=6,
        type=float,
        action="append",
        metavar=("EARTH_X", "EARTH_Y", "HEADING_DEG", "AUTO_X", "AUTO_Y", "AUTO_HEADING_RAD"),
        help="Repeatable multi-point pair for least-squares G->A calibration.",
    )
    parser.add_argument("--ego-pose", nargs=3, type=float, required=True, metavar=("X", "Y", "HEADING"))
    parser.add_argument("--platform-rank", type=int, default=1, help="Sorted BV rank used as platform reference.")

    parser.add_argument(
        "--platform-heading-convention",
        choices=(
            "ins_deg",
            "yaw_deg",
            "yaw_rad",
        ),
        default="ins_deg",
    )

    parser.add_argument(
        "--ego-heading-convention",
        choices=(
            "ins_deg",
            "yaw_deg",
            "yaw_rad",
        ),
        default="yaw_rad",
    )

    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    autoscene_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    config = load_yaml(config_path)
    scenario_index = (
        int(args.scenario_index)
        if args.scenario_index is not None
        else int(config.get("scenario", {}).get("scenario_index", 0))
    )

    if args.platform_pose is None and args.platform_lla is None and not args.platform_pair:
        raise RuntimeError("one of --platform-pose, --platform-lla or --platform-pair is required")

    if args.platform_pose is not None:
        platform_pose_feedback = parse_pose(
            args.platform_pose,
            args.platform_heading_convention,
        )
    elif args.platform_lla is not None:
        earth_x, earth_y = latlon_to_xy(args.platform_lla[0], args.platform_lla[1])
        platform_pose_feedback = {
            "x": earth_x,
            "y": earth_y,
            "heading": ins_heading_deg_to_earth_yaw(args.platform_lla[2]),
        }
    else:
        platform_pose_feedback = None

    ego_pose_feedback = parse_pose(
        args.ego_pose,
        args.ego_heading_convention,
    )

    env = load_closeloop_env(config, autoscene_root, scenario_index)

    try:
        engine = env.engine
        ego_pose_autoscene = ego_pose_from_engine(engine)
        candidates = sorted_bv_candidates(engine)

        if args.platform_rank < 1 or args.platform_rank > len(candidates):
            raise RuntimeError(
                "platform rank %d is out of range, candidates=%d"
                % (args.platform_rank, len(candidates))
            )

        platform_pose_autoscene = candidates[args.platform_rank - 1]
        if args.platform_pair:
            pairs = [
                {
                    "earth": {
                        "x": item[0],
                        "y": item[1],
                        "yaw": ins_heading_deg_to_earth_yaw(item[2]),
                    },
                    "autoscene": {
                        "x": item[3],
                        "y": item[4],
                        "yaw": item[5],
                    },
                }
                for item in args.platform_pair
            ]
        else:
            pairs = [
                make_platform_calibration_pair(
                    platform_pose_feedback,
                    platform_pose_autoscene,
                )
            ]
        platform_transform, platform_diag = estimate_earth_to_autoscene(pairs)
        ego_origin = compute_origin(ego_pose_autoscene, ego_pose_feedback)

        print("scenario_index: %d" % scenario_index)
        print(
            "platform_reference: rank=%d track_id=%s x=%.6f y=%.6f heading=%.6f speed=%.6f"
            % (
                args.platform_rank,
                platform_pose_autoscene["track_id"],
                platform_pose_autoscene["x"],
                platform_pose_autoscene["y"],
                platform_pose_autoscene["heading"],
                platform_pose_autoscene["speed"],
            )
        )
        print(
            "ego_reference: x=%.6f y=%.6f heading=%.6f speed=%.6f"
            % (
                ego_pose_autoscene["x"],
                ego_pose_autoscene["y"],
                ego_pose_autoscene["heading"],
                ego_pose_autoscene["speed"],
            )
        )
        print("")
        print("# Fill these values into the runtime YAML manually:")
        print("platform:")
        print_earth_to_autoscene_block(platform_transform, indent="  ")
        print(
            "  # calibration_residual: point_count=%d position_rms_m=%.6f "
            "position_max_m=%.6f yaw_max_abs_rad=%.6f"
            % (
                platform_diag["point_count"],
                platform_diag["position_rms_m"],
                platform_diag["position_max_m"],
                platform_diag["yaw_max_abs_rad"],
            )
        )
        print("")
        print("ego:")
        print_origin_block(ego_origin, indent="  ")
    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()