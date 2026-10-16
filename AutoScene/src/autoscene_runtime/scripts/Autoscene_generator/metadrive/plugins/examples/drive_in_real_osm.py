#!/usr/bin/env python
"""
This script demonstrates how to use the environment where traffic and road map are loaded from Waymo dataset.
先在终端中export ROOT_PATH="/home/tian/metadrive_original/metadrive"，改为你自己的路径
"""
import argparse
import copy
import math
import pickle
from pathlib import Path
import sys

import numpy as np

sys.path.append("./")  # add project root dir to path
from metadrive.component.sensors.depth_camera import DepthCamera
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.component.sensors.semantic_camera import SemanticCamera
from metadrive.constants import DEFAULT_AGENT, HELP_MESSAGE
from metadrive.engine.asset_loader import AssetLoader
from metadrive.plugins.envs.osm_env import OSMEnv
from metadrive.plugins.manager.osm_map_manager import OSMMapIO
from metadrive.plugins.component.navigation_module.trajectory_navigation_in_osm import OsmTrajectoryNavigation
from metadrive.plugins.utils.osm_scenario_variant import (
    build_variant_config,
    get_available_lane_ids,
    iter_lane_destination_values,
)
from metadrive.scenario.scenario_description import ScenarioDescription as SD
from metadrive.scenario.utils import convert_recorded_scenario_exported
import metadrive.engine.top_down_renderer as td_mod
from metadrive.plugins.engine.top_down_renderer import TopDownRenderer as CustomTopDownRenderer
td_mod.TopDownRenderer = CustomTopDownRenderer

RENDER_MESSAGE = {
    "Quit": "ESC",
    "Switch perspective": "Q or B",
    "Reset Episode": "R",
    "Keyboard Control": "W,A,S,D",
}


RELAXED_IDM_CONFIG = {
    "idm_distance_wanted_m": 4.0,
    "idm_time_wanted_min_s": 0.4,
    "idm_time_wanted_max_s": 0.8,
    "idm_safe_distance_m": 4.0,
    "idm_emergency_distance_m": 1.8,
    "idm_max_long_dist_m": 18.0,
    "idm_forward_lookahead_base_m": 8.0,
    "idm_forward_lookahead_gain_s": 1.0,
    "idm_forward_lookahead_max_m": 20.0,
    "idm_intersection_lookahead_m": 8.0,
    "idm_intersection_deadlock_wait_steps": 10,
    "idm_intersection_release_ttc_min_s": 0.5,
    "idm_intersection_creep_speed_km_h": 10.0,
}


DEFAULT_EGO_VEHICLE_PARAMETERS = {
    "width": 3.07,
    "wheel_base": 3.089,
    "length": 6.076,
    "cog_position_from_rear_axle": 1.67,
    "height": 1.777,
}


def parse_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in ("1", "true", "t", "yes", "y", "on"):
        return True
    if value in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {value!r}")


def get_map_seed(cfg, variant_index: int) -> int:
    start = int(cfg.get("start_scenario_index", 0))
    num_scenarios = max(int(cfg.get("num_scenarios", 1)), 100)
    return start + int(variant_index) % num_scenarios


def _safe_file_token(value) -> str:
    token = str(value)
    safe_chars = []
    for char in token:
        if char.isalnum() or char in ("-", "_"):
            safe_chars.append(char)
        elif char == ".":
            safe_chars.append("p")
        else:
            safe_chars.append("-")
    return "".join(safe_chars).strip("-_") or "unknown"


def _meter_token(value) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "na"
    token = f"{value:.1f}".replace("-", "m").replace(".", "p")
    return token


def _longitudinal_relation_token(relative_s) -> str:
    try:
        relative_s = float(relative_s)
    except (TypeError, ValueError):
        return "sna"
    if relative_s > 1e-3:
        return f"ahead{_meter_token(relative_s)}m"
    if relative_s < -1e-3:
        return f"behind{_meter_token(abs(relative_s))}m"
    return "along0p0m"


def _lane_offset_token(lane_offset) -> str:
    try:
        lane_offset = int(lane_offset)
    except (TypeError, ValueError):
        return "lane_abs"
    if lane_offset < 0:
        return f"right{abs(lane_offset)}"
    if lane_offset > 0:
        return f"left{lane_offset}"
    return "same"


def build_relative_position_filename_token(cfg, variant_index: int, variant_seed: int) -> str:
    agent_cfg = cfg.get("agent_configs", {}).get(DEFAULT_AGENT, {})
    ego_lane = agent_cfg.get("spawn_lane_index")
    ego_s = agent_cfg.get("spawn_longitude")
    parts = [
        f"case_{int(variant_index):04d}",
        f"seed_{int(variant_seed):04d}",
        f"ego_L{_safe_file_token(ego_lane)}_s{_meter_token(ego_s)}",
    ]

    for adv_index, adv_cfg in enumerate(cfg.get("custom_adversaries", []), start=1):
        metadata = adv_cfg.get("metadata", {}) or {}
        name = _safe_file_token(adv_cfg.get("name") or metadata.get("vehicle_id") or f"adv{adv_index}")
        lane_offset = metadata.get("lane_offset")
        relative_s = metadata.get("relative_s")
        relative_to = metadata.get("relative_to")

        if lane_offset is not None and relative_s is not None:
            adv_parts = [
                name,
                _lane_offset_token(lane_offset),
                _longitudinal_relation_token(relative_s),
            ]
            if relative_to and str(relative_to) != "ego":
                adv_parts.append(f"to_{_safe_file_token(relative_to)}")
            parts.append("_".join(adv_parts))
            continue

        adv_lane = adv_cfg.get("spawn_lane_index", metadata.get("lane_id", metadata.get("final_lane_id")))
        adv_s = adv_cfg.get("spawn_longitude", metadata.get("s"))
        requested_base_s = metadata.get("requested_base_s")
        if ego_lane is not None and adv_lane is not None and str(adv_lane) == str(ego_lane):
            try:
                relation = _longitudinal_relation_token(float(adv_s) - float(ego_s))
                parts.append(f"{name}_same_{relation}")
            except (TypeError, ValueError):
                parts.append(f"{name}_same_sna")
        elif str(metadata.get("placement_mode", "")).lower() == "absolute" and requested_base_s is not None:
            parts.append(f"{name}_abs_L{_safe_file_token(adv_lane)}_b{_meter_token(requested_base_s)}")
        else:
            parts.append(f"{name}_abs_L{_safe_file_token(adv_lane)}_s{_meter_token(adv_s)}")

    return _safe_file_token("_".join(parts))


def save_episode_gif(env, gif_dir: Path, filename_token: str, duration: int) -> None:
    if env.top_down_renderer is None or not env.top_down_renderer.screen_frames:
        return
    gif_dir.mkdir(parents=True, exist_ok=True)
    gif_path = gif_dir / f"osm_episode_{filename_token}.gif"
    env.top_down_renderer.generate_gif(str(gif_path), duration=duration)
    env.top_down_renderer._screen_frames = []
    env.top_down_renderer._success_screen_frames = []
    print(f"Saved GIF to {gif_path}")


def rename_sdc_track(scenario, ego_vehicle_id) -> None:
    if not ego_vehicle_id:
        return
    ego_vehicle_id = str(ego_vehicle_id)
    metadata = scenario.setdefault("metadata", {})
    tracks = scenario.get("tracks", {})
    old_sdc_id = metadata.get("sdc_id")
    if old_sdc_id is None:
        return
    old_sdc_id = str(old_sdc_id)
    if old_sdc_id == ego_vehicle_id:
        return
    if old_sdc_id not in tracks:
        return
    if ego_vehicle_id in tracks:
        raise ValueError(f"Can not rename SDC track {old_sdc_id!r} to existing track id {ego_vehicle_id!r}")

    tracks[ego_vehicle_id] = tracks.pop(old_sdc_id)
    track_metadata = tracks[ego_vehicle_id].setdefault("metadata", {})
    track_metadata["object_id"] = ego_vehicle_id
    track_metadata["original_id"] = ego_vehicle_id
    metadata["sdc_id"] = ego_vehicle_id


def enrich_scenario_metadata(scenario, variant_seed: int) -> None:
    metadata = scenario.setdefault("metadata", {})
    metadata["variant_seed"] = int(variant_seed)
    metadata["ego_vehicle_parameters"] = copy.deepcopy(DEFAULT_EGO_VEHICLE_PARAMETERS)
    metadata["track_length"] = int(scenario.get("length", 0))
    metadata.setdefault("dataset", "osm")
    metadata.setdefault("scenario_id", metadata.get("id", scenario.get("id", f"osm_seed_{variant_seed:04d}")))
    SD.update_summaries(scenario)


def _get_export_dt(scenario) -> float:
    ts = np.asarray(scenario.get("metadata", {}).get("ts", []), dtype=np.float32)
    if ts.shape[0] >= 2:
        dt = float(np.median(np.diff(ts[: min(ts.shape[0], 20)])))
        if dt > 1e-4:
            return dt
    return 0.1


def _get_vehicle_export_speed(vehicle, default_speed_mps: float) -> float:
    config = getattr(vehicle, "config", {}) or {}
    spawn_velocity = config.get("spawn_velocity")
    if spawn_velocity is not None and len(spawn_velocity) > 0:
        try:
            speed = abs(float(spawn_velocity[0]))
            if speed > 1e-3:
                return speed
        except (TypeError, ValueError):
            pass

    for key in ("idm_target_speed_km_h", "target_speed_km_h"):
        if key in config:
            try:
                speed = float(config[key]) / 3.6
                if speed > 1e-3:
                    return speed
            except (TypeError, ValueError):
                pass

    try:
        speed = abs(float(getattr(vehicle, "speed", 0.0)))
        if speed > 1e-3:
            return speed
    except (TypeError, ValueError):
        pass
    return float(default_speed_mps)


def _candidate_track_ids(env, vehicle, scenario):
    candidates = []
    if vehicle is getattr(env, "agent", None):
        sdc_id = scenario.get("metadata", {}).get("sdc_id")
        export_ego_id = env.engine.global_config.get("export_ego_vehicle_id", None)
        candidates.extend([sdc_id, export_ego_id, DEFAULT_AGENT])
    candidates.extend([
        getattr(vehicle, "name", None),
        getattr(vehicle, "id", None),
    ])
    seen = set()
    for candidate in candidates:
        if candidate is None:
            continue
        candidate = str(candidate)
        if candidate and candidate not in seen:
            seen.add(candidate)
            yield candidate


def _iter_export_vehicles(env):
    seen = set()
    agent = getattr(env, "agent", None)
    if agent is not None:
        seen.add(id(agent))
        yield agent

    for collection in (
        getattr(env.engine, "agents", {}),
        getattr(env.engine, "get_objects", lambda: {})(),
    ):
        values = collection.values() if isinstance(collection, dict) else collection
        for vehicle in values:
            if id(vehicle) in seen:
                continue
            seen.add(id(vehicle))
            yield vehicle


def _resize_state_like(template, new_values):
    template = np.asarray(template)
    dtype = template.dtype if template.size > 0 else new_values.dtype
    result = np.zeros((new_values.shape[0],) + template.shape[1:], dtype=dtype)
    if template.ndim == 1:
        result[...] = new_values.astype(dtype, copy=False)
    else:
        width = min(result.shape[-1], new_values.shape[-1])
        result[..., :width] = new_values[..., :width].astype(dtype, copy=False)
        if template.shape[0] > 0 and result.shape[-1] > width:
            result[..., width:] = template[0, width:]
    return result


def _sample_reference_track(vehicle, track, horizon: int, dt: float, default_speed_mps: float) -> bool:
    navigation = getattr(vehicle, "navigation", None)
    reference_trajectory = getattr(navigation, "_initial_reference_trajectory", None)
    if reference_trajectory is None:
        reference_trajectory = getattr(navigation, "reference_trajectory", None)
    if reference_trajectory is None or _lane_length(reference_trajectory) <= 1e-3:
        return False

    state = track.get("state", {})
    if "position" not in state or "heading" not in state or "velocity" not in state:
        return False

    start_position = np.asarray(state["position"][0], dtype=np.float64)[:2]
    try:
        start_s, start_l = reference_trajectory.local_coordinates(start_position)
    except Exception:
        start_s, start_l = 0.0, 0.0
    start_s = float(np.clip(start_s, 0.0, reference_trajectory.length))
    start_l = float(np.clip(start_l, -2.0, 2.0))

    speed = _get_vehicle_export_speed(vehicle, default_speed_mps)
    arc_lengths = start_s + speed * dt * np.arange(horizon, dtype=np.float64)
    arc_lengths = np.clip(arc_lengths, 0.0, float(reference_trajectory.length))

    xy = np.asarray(
        [reference_trajectory.position(float(s), start_l) for s in arc_lengths],
        dtype=np.float32,
    )[..., :2]
    heading = np.asarray(
        [reference_trajectory.heading_theta_at(float(s)) for s in arc_lengths],
        dtype=np.float32,
    )
    velocity_xy = np.zeros((horizon, 2), dtype=np.float32)
    if horizon > 1:
        velocity_xy[:-1] = (xy[1:] - xy[:-1]) / max(dt, 1e-4)
        velocity_xy[-1] = velocity_xy[-2]
    reached_end = arc_lengths >= float(reference_trajectory.length) - 1e-3
    velocity_xy[reached_end] = 0.0

    position_values = np.zeros((horizon, max(3, np.asarray(state["position"]).shape[-1])), dtype=np.float32)
    position_values[:, :2] = xy
    old_position = np.asarray(state["position"])
    if old_position.ndim == 2 and old_position.shape[0] > 0 and old_position.shape[1] > 2:
        position_values[:, 2 : old_position.shape[1]] = old_position[0, 2 : old_position.shape[1]]

    velocity_values = np.zeros((horizon, max(2, np.asarray(state["velocity"]).shape[-1])), dtype=np.float32)
    velocity_values[:, :2] = velocity_xy

    state["position"] = _resize_state_like(state["position"], position_values)
    state["heading"] = _resize_state_like(state["heading"], heading)
    state["velocity"] = _resize_state_like(state["velocity"], velocity_values)
    if "valid" in state:
        state["valid"] = np.ones(horizon, dtype=np.asarray(state["valid"]).dtype)
    for key in ("length", "width", "height", "radius"):
        if key in state:
            values = np.asarray(state[key])
            if values.shape[0] != horizon and values.shape[0] > 0:
                state[key] = np.repeat(values[:1], horizon, axis=0)
    metadata = track.setdefault("metadata", {})
    metadata["track_source"] = "planned_global_path"
    metadata["track_length"] = int(horizon)
    return True


def _pad_track_to_horizon(track, horizon: int) -> bool:
    resized = _pad_state_dict_to_horizon(track.get("state", {}), horizon)
    metadata = track.setdefault("metadata", {})
    if resized and metadata.get("track_source") != "planned_global_path":
        metadata["track_source"] = "padded_last_valid_state"
        metadata["track_length"] = int(horizon)
    return resized


def _pad_state_dict_to_horizon(state, horizon: int) -> bool:
    if not state:
        return False
    resized = False
    valid = np.asarray(state.get("valid", []))
    valid_indices = np.where(valid.astype(bool))[0] if valid.size else np.asarray([], dtype=int)
    fallback_index = int(valid_indices[-1]) if len(valid_indices) else 0

    for key, value in list(state.items()):
        original_is_list = isinstance(value, list)
        original_is_tuple = isinstance(value, tuple)
        values = np.asarray(value)
        if values.ndim == 0 or values.shape[0] == int(horizon):
            continue
        old_len = values.shape[0]
        if old_len <= 0:
            continue
        new_values = np.zeros((int(horizon),) + values.shape[1:], dtype=values.dtype)
        copy_len = min(old_len, int(horizon))
        new_values[:copy_len] = values[:copy_len]
        fill_source = values[min(fallback_index, old_len - 1)]
        if copy_len < int(horizon):
            new_values[copy_len:] = fill_source
        if original_is_list:
            state[key] = new_values.tolist()
        elif original_is_tuple:
            state[key] = tuple(new_values.tolist())
        else:
            state[key] = new_values
        resized = True

    if "valid" in state:
        state["valid"] = np.ones(int(horizon), dtype=np.asarray(state["valid"]).dtype)
        resized = True
    return resized


def pad_dynamic_map_states_to_horizon(scenario, horizon: int) -> int:
    padded = 0
    for light in scenario.get(SD.DYNAMIC_MAP_STATES, {}).values():
        padded += int(_pad_state_dict_to_horizon(light.get(SD.STATE, {}), horizon))
    return padded


def overwrite_tracks_with_planned_global_paths(env, scenario, default_speed_mps: float = 8.0, target_horizon=None) -> int:
    if default_speed_mps <= 0.0:
        return 0
    tracks = scenario.get("tracks", {})
    if not tracks:
        return 0

    horizon = int(max(1, target_horizon or scenario.get("length", 0)))
    dt = _get_export_dt(scenario)
    default_speed = float(default_speed_mps)
    overwritten = 0
    used_track_ids = set()
    for vehicle in _iter_export_vehicles(env):
        track_id = next((tid for tid in _candidate_track_ids(env, vehicle, scenario) if tid in tracks), None)
        if track_id is None or track_id in used_track_ids:
            continue
        if _sample_reference_track(vehicle, tracks[track_id], horizon, dt, default_speed):
            used_track_ids.add(track_id)
            overwritten += 1
    padded = 0
    for track_id, track in tracks.items():
        padded += int(_pad_track_to_horizon(track, horizon))
    if overwritten or padded:
        scenario["length"] = horizon
        print(
            f"[export planned paths] overwritten_tracks={overwritten}, padded_tracks={padded}, "
            f"horizon={horizon}, dt={dt:.3f}"
        )
    return overwritten


def update_dataset_index(scenario_dir: Path, scenario_path: Path, scenario) -> None:
    summary_path = scenario_dir / SD.DATASET.SUMMARY_FILE
    mapping_path = scenario_dir / SD.DATASET.MAPPING_FILE
    file_name = scenario_path.name

    if summary_path.exists():
        with open(summary_path, "rb") as file:
            summary = pickle.load(file)
    else:
        summary = {}

    if mapping_path.exists():
        with open(mapping_path, "rb") as file:
            mapping = pickle.load(file)
    else:
        mapping = {}

    summary[file_name] = copy.deepcopy(scenario.get("metadata", {}))
    mapping[file_name] = ""

    with open(summary_path, "wb") as file:
        pickle.dump(summary, file)
    with open(mapping_path, "wb") as file:
        pickle.dump(mapping, file)


def assert_track_state_horizon(scenario) -> None:
    horizon = int(scenario.get("length", 0))
    mismatches = []
    for track_id, track in scenario.get("tracks", {}).items():
        for key, value in track.get("state", {}).items():
            values = np.asarray(value)
            if values.ndim > 0 and values.shape[0] != horizon:
                mismatches.append(f"{track_id}.{key}: {values.shape[0]} != {horizon}")
    for light_id, light in scenario.get(SD.DYNAMIC_MAP_STATES, {}).items():
        for key, value in light.get(SD.STATE, {}).items():
            values = np.asarray(value)
            if values.ndim > 0 and values.shape[0] != horizon:
                mismatches.append(f"{light_id}.{key}: {values.shape[0]} != {horizon}")
    if mismatches:
        raise ValueError("Scenario state horizon mismatch: " + "; ".join(mismatches[:20]))


def save_scenario_description(
    env,
    scenario_dir: Path,
    variant_seed: int,
    filename_token: str,
    export_planned_global_paths: bool = True,
    export_planned_path_speed_mps: float = 8.0,
    target_horizon=None,
) -> None:
    scenario_dir.mkdir(parents=True, exist_ok=True)
    episode = env.engine.dump_episode()
    scenario = convert_recorded_scenario_exported(episode, to_dict=True)
    rename_sdc_track(scenario, env.engine.global_config.get("export_ego_vehicle_id", None))
    if export_planned_global_paths:
        overwrite_tracks_with_planned_global_paths(
            env,
            scenario,
            export_planned_path_speed_mps,
            target_horizon=target_horizon,
        )
        if target_horizon is not None:
            horizon = int(max(1, target_horizon))
            padded_lights = pad_dynamic_map_states_to_horizon(scenario, horizon)
            if padded_lights:
                scenario["length"] = horizon
                print(f"[export padded traffic lights] padded_lights={padded_lights}, horizon={horizon}")
    elif target_horizon is not None:
        horizon = int(max(1, target_horizon))
        padded = sum(int(_pad_track_to_horizon(track, horizon)) for track in scenario.get("tracks", {}).values())
        padded_lights = pad_dynamic_map_states_to_horizon(scenario, horizon)
        if padded or padded_lights:
            scenario["length"] = horizon
            print(f"[export padded paths] padded_tracks={padded}, padded_lights={padded_lights}, horizon={horizon}")
    enrich_scenario_metadata(scenario, variant_seed)
    scenario.setdefault("metadata", {})["filename_token"] = str(filename_token)
    assert_track_state_horizon(scenario)
    scenario_path = scenario_dir / f"sd_osm_v1_{filename_token}.pkl"
    with open(scenario_path, "wb") as file:
        pickle.dump(scenario, file)
    update_dataset_index(scenario_dir, scenario_path, scenario)
    print(f"Saved ScenarioDescription to {scenario_path}")


def get_osm_feature_ids(cfg):
    """Read OSM map feature ids without creating an OSMEnv."""
    parser = OSMMapIO(cfg["osm_file_path"], use_projection=False)
    return sorted(str(feature_id) for feature_id in parser.parse().keys())


def validate_ids_before_probe(cfg, ego_spec, adversary_specs):
    available_ids = set(get_osm_feature_ids(cfg))
    missing = []
    specs = [("ego", ego_spec)] + [(f"adversary[{i}]", spec) for i, spec in enumerate(adversary_specs)]
    for label, spec in specs:
        for field, value in iter_lane_destination_values(spec):
            if str(value) not in available_ids:
                missing.append(f"{label}.{field}={value}")
    if missing:
        preview = " ".join(sorted(available_ids)[:80])
        raise ValueError(
            "Configured ids are not in the parsed OSM map features:\n  "
            + "\n  ".join(missing)
            + "\nFirst available ids: "
            + preview
        )


def make_map_probe_cfg(base_cfg, ego_spec):
    """Create a config that loads the OSM map using a valid ego lane and keeps OSM navigation enabled."""
    def first_value(value):
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                raise ValueError("Probe config can not use an empty lane/destination list")
            return value[0]
        return value

    fallback_destination = (
        base_cfg.get("agent_configs", {}).get("default_agent", {}).get("destination")
        or base_cfg.get("vehicle_config", {}).get("destination")
        or ego_spec["lane_id"]
    )
    probe_destination = str(first_value(ego_spec.get("destination", fallback_destination)))

    probe_cfg = copy.deepcopy(base_cfg)
    probe_cfg["no_traffic"] = True
    probe_cfg["no_light"] = True

    probe_vehicle_config = probe_cfg.setdefault("vehicle_config", {})
    probe_vehicle_config["navigation_module"] = OsmTrajectoryNavigation
    probe_vehicle_config["spawn_lane_index"] = str(first_value(ego_spec["lane_id"]))
    probe_vehicle_config["destination"] = probe_destination
    probe_vehicle_config["spawn_position_heading"] = None

    probe_default_agent = probe_cfg["agent_configs"]["default_agent"]
    probe_default_agent["spawn_lane_index"] = str(first_value(ego_spec["lane_id"]))
    probe_default_agent["destination"] = probe_destination
    probe_default_agent["navigation_module"] = OsmTrajectoryNavigation
    probe_default_agent["spawn_position_heading"] = None
    return probe_cfg


def _as_pair(value, default=None):
    if value is None:
        value = default
    if isinstance(value, (int, float)):
        return float(value), float(value)
    if len(value) != 2:
        raise ValueError(f"Expected a scalar or [low, high], got {value}")
    return float(value[0]), float(value[1])


def _sample_template_value(value, rng, as_int=False):
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            raise ValueError("Can not sample from an empty template value")
        if len(value) == 2 and all(isinstance(item, (int, float)) for item in value) and not as_int:
            return float(rng.uniform(float(value[0]), float(value[1])))
        return value[int(rng.integers(0, len(value)))]
    return value


def _normalize_lane_id(road_network, lane_id):
    if lane_id in getattr(road_network, "graph", {}):
        return lane_id
    lane_id = str(lane_id)
    graph = getattr(road_network, "graph", {}) or {}
    if lane_id in graph:
        return lane_id
    if hasattr(road_network, "_resolve_graph_key"):
        resolved = road_network._resolve_graph_key(lane_id)
        if resolved in graph:
            return resolved
    raise ValueError(f"lane_id {lane_id!r} is not in road network")


def _normalize_neighbor_id(road_network, neighbor):
    if neighbor is None:
        return None
    if hasattr(road_network, "_normalize_neighbor_ref"):
        lane_id = road_network._normalize_neighbor_ref(neighbor)
    elif isinstance(neighbor, dict):
        lane_id = neighbor.get("feature_id", neighbor.get("id"))
    elif hasattr(neighbor, "feature_id"):
        lane_id = neighbor.feature_id
    elif hasattr(neighbor, "index") and not isinstance(neighbor, str):
        lane_id = neighbor.index
    else:
        lane_id = neighbor
    if lane_id is None:
        return None
    graph = getattr(road_network, "graph", {}) or {}
    lane_id = str(lane_id)
    return lane_id if lane_id in graph else None


def _finite_float(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def _lane_points(lane):
    for attr in ("center_line_points", "_raw_center_line_points", "polyline", "points"):
        points = getattr(lane, attr, None)
        if points is not None:
            points = np.asarray(points, dtype=float)[..., :2]
            if len(points) >= 2:
                return points
    start_points = getattr(lane, "_start_points", None)
    end_points = getattr(lane, "_end_points", None)
    if start_points is not None and end_points is not None and len(start_points) > 0:
        return np.concatenate([np.asarray(start_points, dtype=float)[..., :2], np.asarray(end_points, dtype=float)[-1:, :2]])
    return None


def _polyline_lengths(points):
    points = np.asarray(points, dtype=float)[..., :2]
    if len(points) < 2:
        return np.asarray([0.0])
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(segment_lengths)])


def _lane_length(lane):
    raw_length = _finite_float(getattr(lane, "length", 0.0), 0.0)
    points = _lane_points(lane)
    if points is None:
        return raw_length
    point_length = float(_polyline_lengths(points)[-1])
    return max(raw_length, point_length)


def _lane_position(lane, longitudinal, lateral=0.0):
    points = _lane_points(lane)
    if points is None:
        return np.asarray(lane.position(float(longitudinal), float(lateral)), dtype=float)
    cumulative = _polyline_lengths(points)
    longitudinal = float(np.clip(float(longitudinal), 0.0, max(0.0, cumulative[-1])))
    idx = int(np.searchsorted(cumulative, longitudinal, side="right") - 1)
    idx = int(np.clip(idx, 0, len(points) - 2))
    seg_len = max(float(cumulative[idx + 1] - cumulative[idx]), 1e-6)
    ratio = (longitudinal - float(cumulative[idx])) / seg_len
    base = points[idx] + (points[idx + 1] - points[idx]) * ratio
    direction = (points[idx + 1] - points[idx]) / seg_len
    lateral_direction = np.asarray([-direction[1], direction[0]])
    return base + float(lateral) * lateral_direction


def _lane_heading(lane, longitudinal):
    points = _lane_points(lane)
    if points is None:
        return float(lane.heading_theta_at(float(longitudinal)))
    cumulative = _polyline_lengths(points)
    longitudinal = float(np.clip(float(longitudinal), 0.0, max(0.0, cumulative[-1])))
    idx = int(np.searchsorted(cumulative, longitudinal, side="right") - 1)
    idx = int(np.clip(idx, 0, len(points) - 2))
    delta = points[idx + 1] - points[idx]
    if np.linalg.norm(delta) < 1e-6:
        return 0.0
    return float(math.atan2(delta[1], delta[0]))


def _lane_local_coordinates(lane, position):
    points = _lane_points(lane)
    if points is None:
        return lane.local_coordinates(position)
    position = np.asarray(position, dtype=float)[..., :2]
    cumulative = _polyline_lengths(points)
    best = None
    for idx in range(len(points) - 1):
        start = points[idx]
        end = points[idx + 1]
        delta = end - start
        seg_len = float(np.linalg.norm(delta))
        if seg_len < 1e-6:
            continue
        direction = delta / seg_len
        projection = float(np.dot(position - start, direction))
        clipped_projection = float(np.clip(projection, 0.0, seg_len))
        closest = start + direction * clipped_projection
        lateral_direction = np.asarray([-direction[1], direction[0]])
        lateral = float(np.dot(position - closest, lateral_direction))
        distance = float(np.linalg.norm(position - closest))
        candidate = (distance, float(cumulative[idx] + clipped_projection), lateral)
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        return 0.0, 0.0
    return best[1], best[2]


def _lane_is_passable(road_network, lane_id):
    graph = getattr(road_network, "graph", {}) or {}
    if lane_id not in graph:
        return False
    lane = graph[lane_id].lane
    if bool(getattr(lane, "forbidden", False)):
        return False
    if hasattr(road_network, "is_bike_lane") and road_network.is_bike_lane(lane_id):
        return False
    return _lane_length(lane) > 2.0


def _angle_diff(a, b):
    return (float(a) - float(b) + math.pi) % (2.0 * math.pi) - math.pi


def _neighbor_side_by_geometry(road_network, base_lane_id, neighbor_lane_id):
    base_lane = road_network.get_lane(base_lane_id)
    neighbor_lane = road_network.get_lane(neighbor_lane_id)
    base_length = _lane_length(base_lane)
    neighbor_length = _lane_length(neighbor_lane)
    signed_distances = []
    for s in np.linspace(1.0, max(1.0, base_length - 1.0), 9):
        point = np.asarray(_lane_position(base_lane, float(s), 0.0), dtype=float)
        heading = float(_lane_heading(base_lane, float(s)))
        try:
            neighbor_s, _ = _lane_local_coordinates(neighbor_lane, point)
        except Exception:
            continue
        if not 0.0 <= neighbor_s <= neighbor_length:
            continue
        neighbor_point = np.asarray(_lane_position(neighbor_lane, neighbor_s, 0.0), dtype=float)
        left_normal = np.array([-math.sin(heading), math.cos(heading)])
        signed_distances.append(float(np.dot(neighbor_point[:2] - point[:2], left_normal)))
    if not signed_distances:
        return None
    median_signed_distance = float(np.median(signed_distances))
    if abs(median_signed_distance) < 1e-3:
        return None
    return "left" if median_signed_distance > 0.0 else "right"


def _validate_neighbor_pair(
    road_network,
    base_lane_id,
    neighbor_lane_id,
    expected_side=None,
    min_overlap_m=12.0,
    min_center_gap_m=1.5,
    max_center_gap_m=8.0,
    max_heading_diff_deg=35.0,
):
    if not _lane_is_passable(road_network, neighbor_lane_id):
        return False

    base_lane = road_network.get_lane(base_lane_id)
    neighbor_lane = road_network.get_lane(neighbor_lane_id)
    base_length = _lane_length(base_lane)
    neighbor_length = _lane_length(neighbor_lane)
    if min(base_length, neighbor_length) < min_overlap_m:
        return False
    if expected_side is not None and _neighbor_side_by_geometry(road_network, base_lane_id, neighbor_lane_id) != expected_side:
        return False

    sample_count = 9
    overlaps = 0
    heading_diffs = []
    center_gaps = []
    for s in np.linspace(1.0, max(1.0, base_length - 1.0), sample_count):
        point = np.asarray(_lane_position(base_lane, float(s), 0.0), dtype=float)
        try:
            neighbor_s, _ = _lane_local_coordinates(neighbor_lane, point)
        except Exception:
            continue
        if 0.0 <= neighbor_s <= neighbor_length:
            neighbor_point = np.asarray(_lane_position(neighbor_lane, neighbor_s, 0.0), dtype=float)
            overlaps += 1
            center_gaps.append(float(np.linalg.norm(neighbor_point[:2] - point[:2])))
            heading_diffs.append(abs(_angle_diff(_lane_heading(base_lane, float(s)), _lane_heading(neighbor_lane, neighbor_s))))

    overlap_m = base_length * overlaps / float(sample_count)
    if overlap_m < float(min_overlap_m) or not heading_diffs:
        return False
    median_gap = float(np.median(center_gaps))
    if median_gap < float(min_center_gap_m) or median_gap > float(max_center_gap_m):
        return False
    return float(np.median(heading_diffs)) <= math.radians(float(max_heading_diff_deg))


def _neighbor_pair_score(
    road_network,
    base_lane_id,
    neighbor_lane_id,
    expected_side=None,
    min_overlap_m=2.0,
    min_center_gap_m=1.5,
    max_center_gap_m=5.5,
    max_heading_diff_deg=35.0,
):
    if not _validate_neighbor_pair(
        road_network,
        base_lane_id,
        neighbor_lane_id,
        expected_side=expected_side,
        min_overlap_m=min_overlap_m,
        min_center_gap_m=min_center_gap_m,
        max_center_gap_m=max_center_gap_m,
        max_heading_diff_deg=max_heading_diff_deg,
    ):
        return None

    base_lane = road_network.get_lane(base_lane_id)
    neighbor_lane = road_network.get_lane(neighbor_lane_id)
    base_length = _lane_length(base_lane)
    neighbor_length = _lane_length(neighbor_lane)
    center_gaps = []
    heading_diffs = []
    overlaps = 0
    for s in np.linspace(1.0, max(1.0, base_length - 1.0), 9):
        point = np.asarray(_lane_position(base_lane, float(s), 0.0), dtype=float)
        try:
            neighbor_s, _ = _lane_local_coordinates(neighbor_lane, point)
        except Exception:
            continue
        if 0.0 <= neighbor_s <= neighbor_length:
            neighbor_point = np.asarray(_lane_position(neighbor_lane, neighbor_s, 0.0), dtype=float)
            center_gaps.append(float(np.linalg.norm(neighbor_point[:2] - point[:2])))
            heading_diffs.append(abs(_angle_diff(_lane_heading(base_lane, float(s)), _lane_heading(neighbor_lane, neighbor_s))))
            overlaps += 1
    if not center_gaps:
        return None
    overlap_m = base_length * overlaps / 9.0
    return float(np.median(center_gaps)) + float(np.median(heading_diffs)) * 4.0 - overlap_m * 0.01


def _topology_neighbor_refs(road_network, lane_id):
    info = getattr(road_network, "graph", {}).get(lane_id)
    if info is None:
        return []
    refs = []
    refs.extend(getattr(info, "left_lanes", None) or [])
    refs.extend(getattr(info, "right_lanes", None) or [])
    neighbor_ids = []
    seen = {str(lane_id)}
    for ref in refs:
        neighbor_id = _normalize_neighbor_id(road_network, ref)
        if neighbor_id is None:
            continue
        neighbor_key = str(neighbor_id)
        if neighbor_key in seen:
            continue
        seen.add(neighbor_key)
        if _lane_is_passable(road_network, neighbor_id):
            neighbor_ids.append(neighbor_id)
    return neighbor_ids


def _topology_neighbor(road_network, lane_id, side):
    candidates = []
    for neighbor_id in _topology_neighbor_refs(road_network, lane_id):
        score = _neighbor_pair_score(
            road_network,
            lane_id,
            neighbor_id,
            expected_side=side,
            min_overlap_m=2.0,
            max_center_gap_m=6.0,
        )
        if score is not None:
            candidates.append((float(score), neighbor_id))
    if candidates:
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]
    return None


def _lane_relation_debug(road_network, lane_id):
    return {
        "left": _raw_lane_refs(road_network, lane_id, "left_lanes"),
        "right": _raw_lane_refs(road_network, lane_id, "right_lanes"),
        "entry": _raw_lane_refs(road_network, lane_id, "entry_lanes"),
        "exit": _raw_lane_refs(road_network, lane_id, "exit_lanes"),
    }


def _geometric_neighbor(
    road_network,
    lane_id,
    side,
    min_overlap_m=12.0,
    min_center_gap_m=1.5,
    max_center_gap_m=8.0,
):
    base_lane = road_network.get_lane(lane_id)
    base_length = _lane_length(base_lane)
    side_sign = 1.0 if side == "left" else -1.0
    best = None
    sample_s = np.linspace(1.0, max(1.0, base_length - 1.0), 9)

    for candidate_id, candidate_info in (getattr(road_network, "graph", {}) or {}).items():
        candidate_id = str(candidate_id)
        if candidate_id == lane_id or not _lane_is_passable(road_network, candidate_id):
            continue
        if _is_longitudinally_related(road_network, lane_id, candidate_id):
            continue
        candidate_lane = candidate_info.lane
        candidate_length = _lane_length(candidate_lane)
        if min(base_length, candidate_length) < float(min_overlap_m):
            continue
        lateral_distances = []
        center_gaps = []
        heading_diffs = []
        overlaps = 0
        for s in sample_s:
            point = np.asarray(_lane_position(base_lane, float(s), 0.0), dtype=float)
            heading = float(_lane_heading(base_lane, float(s)))
            try:
                candidate_s, _ = _lane_local_coordinates(candidate_lane, point)
            except Exception:
                continue
            if not 0.0 <= candidate_s <= candidate_length:
                continue
            candidate_point = np.asarray(_lane_position(candidate_lane, candidate_s, 0.0), dtype=float)
            left_normal = np.array([-math.sin(heading), math.cos(heading)])
            signed_distance = float(np.dot(candidate_point[:2] - point[:2], left_normal))
            if signed_distance * side_sign <= 0.0:
                continue
            center_gap = float(np.linalg.norm(candidate_point[:2] - point[:2]))
            if center_gap > float(max_center_gap_m):
                continue
            lateral_distances.append(abs(signed_distance))
            center_gaps.append(center_gap)
            heading_diffs.append(abs(_angle_diff(heading, _lane_heading(candidate_lane, candidate_s))))
            overlaps += 1

        if not center_gaps:
            continue
        overlap_m = base_length * overlaps / float(len(sample_s))
        median_lateral_gap = float(np.median(lateral_distances))
        median_center_gap = float(np.median(center_gaps))
        median_heading_diff = float(np.median(heading_diffs))
        if overlap_m < float(min_overlap_m):
            continue
        if median_center_gap < float(min_center_gap_m) or median_center_gap > float(max_center_gap_m):
            continue
        if median_heading_diff > math.radians(35.0):
            continue
        score = median_center_gap + median_lateral_gap * 0.25 + median_heading_diff * 4.0 - overlap_m * 0.01
        if best is None or score < best[0]:
            best = (score, candidate_id)

    return None if best is None else best[1]


def _longitudinal_chain_neighbor(road_network, lane_id, side, max_hops=4, max_center_gap_m=5.5):
    """Infer a lateral neighbor across segmented OSM lane chains.

    Example: current lane A2 has no right_lanes, but its entry A1 has right
    neighbor B1 and B1 exits to B2. In that case B2 is the right neighbor of A2.
    """
    candidates = []
    for relation_key, continue_key in (("entry_lanes", "exit_lanes"), ("exit_lanes", "entry_lanes")):
        for anchor_id in _ordered_lane_refs(road_network, lane_id, relation_key):
            side_anchor_id = _topology_neighbor(road_network, anchor_id, side)
            if side_anchor_id is None:
                continue

            frontier = [(side_anchor_id, 0)]
            visited = {str(side_anchor_id)}
            while frontier:
                candidate_id, depth = frontier.pop(0)
                score = _neighbor_pair_score(
                    road_network,
                    lane_id,
                    candidate_id,
                    expected_side=None,
                    min_overlap_m=2.0,
                    max_center_gap_m=max_center_gap_m,
                )
                if score is not None:
                    candidates.append((float(score) + depth * 0.25, candidate_id))
                    break
                if depth >= int(max_hops):
                    continue
                for next_id in _ordered_lane_refs(road_network, candidate_id, continue_key):
                    next_key = str(next_id)
                    if next_key in visited:
                        continue
                    visited.add(next_key)
                    frontier.append((next_id, depth + 1))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def resolve_relative_lane(road_network, ego_lane_id, lane_offset):
    lane_offset = int(lane_offset)
    current_id = _normalize_lane_id(road_network, ego_lane_id)
    if lane_offset == 0:
        if not _lane_is_passable(road_network, current_id):
            raise ValueError(f"ego lane {current_id} is not passable")
        return current_id

    side = "right" if lane_offset < 0 else "left"
    for _ in range(abs(lane_offset)):
        next_id = _topology_neighbor(road_network, current_id, side)
        source = "topology"
        if next_id is None:
            next_id = _longitudinal_chain_neighbor(road_network, current_id, side)
            source = "longitudinal_chain"
        if next_id is None:
            next_id = _geometric_neighbor(
                road_network,
                current_id,
                side,
                min_overlap_m=2.0,
                max_center_gap_m=6.0,
            )
            source = "geometry"
        if next_id is None:
            raise ValueError(
                f"No {side} neighbor found from lane {current_id} while resolving offset {lane_offset}; "
                f"relations={_lane_relation_debug(road_network, current_id)}"
            )
        min_overlap_m = 0.0 if source == "topology" else 2.0 if source == "longitudinal_chain" else 12.0
        expected_side = None if source in ("topology", "longitudinal_chain") else side
        max_center_gap_m = 5.5 if source == "longitudinal_chain" else 8.0
        if source != "topology" and not _validate_neighbor_pair(
            road_network,
            current_id,
            next_id,
            expected_side=expected_side,
            min_overlap_m=min_overlap_m,
            max_center_gap_m=max_center_gap_m,
        ):
            raise ValueError(
                f"{source} neighbor {next_id} of {current_id} failed same-direction/overlap/passable validation; "
                f"base_relations={_lane_relation_debug(road_network, current_id)}, "
                f"neighbor_relations={_lane_relation_debug(road_network, next_id)}"
            )
        current_id = next_id
    return current_id


def _ordered_lane_refs(road_network, lane_id, relation_key):
    info = getattr(road_network, "graph", {}).get(lane_id)
    if info is None:
        return []
    refs = getattr(info, relation_key, None) or []
    lane_ids = [_normalize_neighbor_id(road_network, ref) for ref in refs]
    return [item for item in lane_ids if item is not None and _lane_is_passable(road_network, item)]


def _raw_lane_refs(road_network, lane_id, relation_key):
    info = getattr(road_network, "graph", {}).get(lane_id)
    if info is None:
        return []
    refs = getattr(info, relation_key, None) or []
    lane_ids = [_normalize_neighbor_id(road_network, ref) for ref in refs]
    return [item for item in lane_ids if item is not None]


def _is_longitudinally_related(road_network, lane_id, candidate_id, max_hops=4):
    if lane_id is None or candidate_id is None:
        return False
    lane_key = str(lane_id)
    candidate_key = str(candidate_id)
    if lane_key == candidate_key:
        return True
    for relation_key in ("entry_lanes", "exit_lanes"):
        frontier = [(lane_key, 0)]
        visited = {lane_key}
        while frontier:
            current_id, depth = frontier.pop(0)
            if depth >= int(max_hops):
                continue
            for next_id in _raw_lane_refs(road_network, current_id, relation_key):
                next_key = str(next_id)
                if next_key == candidate_key:
                    return True
                if next_key in visited:
                    continue
                visited.add(next_key)
                frontier.append((next_key, depth + 1))
    return False


def _best_longitudinal_continuation(road_network, lane_id, relation_key):
    current_lane = road_network.get_lane(lane_id)
    current_length = _lane_length(current_lane)
    current_s = current_length if relation_key == "exit_lanes" else 0.0
    current_point = np.asarray(_lane_position(current_lane, current_s, 0.0), dtype=float)
    current_heading = float(_lane_heading(current_lane, max(0.0, min(current_s, current_length))))
    best = None
    for candidate_id in _ordered_lane_refs(road_network, lane_id, relation_key):
        candidate_lane = road_network.get_lane(candidate_id)
        candidate_length = _lane_length(candidate_lane)
        candidate_s = 0.0 if relation_key == "exit_lanes" else candidate_length
        candidate_point = np.asarray(_lane_position(candidate_lane, candidate_s, 0.0), dtype=float)
        candidate_heading = float(_lane_heading(candidate_lane, max(0.0, min(candidate_s, candidate_length))))
        score = (
            float(np.linalg.norm(candidate_point[:2] - current_point[:2]))
            + abs(_angle_diff(current_heading, candidate_heading)) * 6.0
        )
        if best is None or score < best[0]:
            best = (score, candidate_id)
    if best is not None:
        return best[1]

    for candidate_id, candidate_info in (getattr(road_network, "graph", {}) or {}).items():
        candidate_id = str(candidate_id)
        if candidate_id == lane_id or not _lane_is_passable(road_network, candidate_id):
            continue
        candidate_lane = candidate_info.lane
        candidate_length = _lane_length(candidate_lane)
        candidate_s = 0.0 if relation_key == "exit_lanes" else candidate_length
        candidate_point = np.asarray(_lane_position(candidate_lane, candidate_s, 0.0), dtype=float)
        candidate_heading = float(_lane_heading(candidate_lane, max(0.0, min(candidate_s, candidate_length))))
        endpoint_gap = float(np.linalg.norm(candidate_point[:2] - current_point[:2]))
        heading_gap = abs(_angle_diff(current_heading, candidate_heading))
        if endpoint_gap > 8.0 or heading_gap > math.radians(35.0):
            continue
        score = endpoint_gap + heading_gap * 6.0
        if best is None or score < best[0]:
            best = (score, candidate_id)

    return None if best is None else best[1]


def _fallback_longitudinal_projection(road_network, lane_id, target_s, s_margin=1.0, max_lateral_m=8.0):
    """Optional geometric fallback: straight-line extrapolate from lane endpoint.

    This is intentionally disabled by default for relative templates because it
    can jump across curved OSM roads to an unrelated nearby lane.
    """
    lane_id = _normalize_lane_id(road_network, lane_id)
    lane = road_network.get_lane(lane_id)
    lane_length = _lane_length(lane)
    if lane_length <= 1e-3:
        return None

    if target_s > lane_length - float(s_margin):
        anchor_s = lane_length
        remaining = float(target_s) - lane_length
    elif target_s < float(s_margin):
        anchor_s = 0.0
        remaining = float(target_s)
    else:
        return lane_id, float(target_s), "in_lane"

    anchor = np.asarray(_lane_position(lane, anchor_s, 0.0), dtype=float)
    heading = float(_lane_heading(lane, anchor_s))
    desired = anchor[:2] + np.array([math.cos(heading), math.sin(heading)]) * remaining
    best = None

    for candidate_id, candidate_info in (getattr(road_network, "graph", {}) or {}).items():
        candidate_id = str(candidate_id)
        if not _lane_is_passable(road_network, candidate_id):
            continue
        candidate_lane = candidate_info.lane
        candidate_length = _lane_length(candidate_lane)
        try:
            candidate_s, candidate_l = _lane_local_coordinates(candidate_lane, desired)
        except Exception:
            continue
        if not float(s_margin) <= candidate_s <= max(float(s_margin), candidate_length - float(s_margin)):
            continue
        candidate_heading = float(_lane_heading(candidate_lane, candidate_s))
        heading_gap = abs(_angle_diff(heading, candidate_heading))
        if heading_gap > math.radians(45.0):
            continue
        lateral_gap = abs(float(candidate_l))
        if lateral_gap > float(max_lateral_m):
            continue
        candidate_point = np.asarray(_lane_position(candidate_lane, candidate_s, 0.0), dtype=float)
        distance = float(np.linalg.norm(candidate_point[:2] - desired[:2]))
        score = distance + lateral_gap * 0.5 + heading_gap * 5.0
        if best is None or score < best[0]:
            best = (score, candidate_id, float(candidate_s))

    if best is not None:
        return best[1], best[2], "geometric_endpoint_projection"

    # Last-resort OSM fallback: some parsed maps have missing/fragmented topology and
    # local projection can reject the target point. Densely search center lines and
    # choose the nearest same-ish direction drivable lane point.
    dense_best = None
    for candidate_id, candidate_info in (getattr(road_network, "graph", {}) or {}).items():
        candidate_id = str(candidate_id)
        if not _lane_is_passable(road_network, candidate_id):
            continue
        candidate_lane = candidate_info.lane
        candidate_length = _lane_length(candidate_lane)
        if candidate_length <= 2.0 * float(s_margin):
            continue
        sample_count = int(np.clip(candidate_length / 2.0, 6, 80))
        for candidate_s in np.linspace(float(s_margin), candidate_length - float(s_margin), sample_count):
            candidate_point = np.asarray(_lane_position(candidate_lane, float(candidate_s), 0.0), dtype=float)
            distance = float(np.linalg.norm(candidate_point[:2] - desired[:2]))
            if distance > 60.0:
                continue
            candidate_heading = float(_lane_heading(candidate_lane, float(candidate_s)))
            heading_gap = abs(_angle_diff(heading, candidate_heading))
            if heading_gap > math.radians(90.0):
                continue
            score = distance + heading_gap * 3.0
            if dense_best is None or score < dense_best[0]:
                dense_best = (score, candidate_id, float(candidate_s))

    if dense_best is None:
        # Keep the pipeline moving for short terminal fragments: place at the
        # nearest legal boundary on the current lane instead of failing the variant.
        clipped_s = float(np.clip(target_s, float(s_margin), max(float(s_margin), lane_length - float(s_margin))))
        return lane_id, clipped_s, "clipped_current_lane"
    return dense_best[1], dense_best[2], "geometric_dense_centerline"


def resolve_longitudinal_relative_lane(
    road_network,
    start_lane_id,
    start_s,
    relative_s,
    s_margin=1.0,
    max_hops=12,
    allow_geometric_projection_fallback=False,
):
    lane_id = _normalize_lane_id(road_network, start_lane_id)
    target_s = float(start_s) + float(relative_s)
    s_margin = float(s_margin)
    visited = set()

    for _ in range(int(max_hops) + 1):
        if lane_id in visited:
            break
        visited.add(lane_id)
        lane = road_network.get_lane(lane_id)
        lane_length = _lane_length(lane)

        if s_margin <= target_s <= max(s_margin, lane_length - s_margin):
            return lane_id, float(target_s), "topology"

        if target_s > lane_length - s_margin:
            next_lane_id = _best_longitudinal_continuation(road_network, lane_id, "exit_lanes")
            if next_lane_id is None:
                if allow_geometric_projection_fallback:
                    fallback = _fallback_longitudinal_projection(road_network, lane_id, target_s, s_margin=s_margin)
                    if fallback is not None:
                        return fallback
                return lane_id, _clip_s_to_lane(lane, target_s, s_margin=s_margin), "clipped_no_exit_projection_disabled"
                break
            target_s -= lane_length
            lane_id = next_lane_id
            continue

        if target_s < s_margin:
            prev_lane_id = _best_longitudinal_continuation(road_network, lane_id, "entry_lanes")
            if prev_lane_id is None:
                if allow_geometric_projection_fallback:
                    fallback = _fallback_longitudinal_projection(road_network, lane_id, target_s, s_margin=s_margin)
                    if fallback is not None:
                        return fallback
                return lane_id, _clip_s_to_lane(lane, target_s, s_margin=s_margin), "clipped_no_entry_projection_disabled"
                break
            prev_lane = road_network.get_lane(prev_lane_id)
            target_s += _lane_length(prev_lane)
            lane_id = prev_lane_id
            continue

    if allow_geometric_projection_fallback:
        fallback = _fallback_longitudinal_projection(road_network, lane_id, target_s, s_margin=s_margin)
        if fallback is not None:
            return fallback
    lane = road_network.get_lane(lane_id)
    return lane_id, _clip_s_to_lane(lane, target_s, s_margin=s_margin), "clipped_projection_disabled"

    raise ValueError(
        f"relative_s={float(relative_s):.2f} from lane {start_lane_id} s={float(start_s):.2f} "
        f"can not be placed within adjacent/previous/next lane chain"
    )


def _clip_s_to_lane(lane, target_s, s_margin=1.0):
    lane_length = _lane_length(lane)
    return float(np.clip(float(target_s), float(s_margin), max(float(s_margin), lane_length - float(s_margin))))


def _choose_downstream_destination(road_network, lane_id, hops=8):
    lane_id = _normalize_lane_id(road_network, lane_id)
    current_id = lane_id
    for _ in range(int(hops)):
        info = getattr(road_network, "graph", {}).get(current_id)
        exits = [_normalize_neighbor_id(road_network, ref) for ref in (getattr(info, "exit_lanes", None) or [])]
        exits = [item for item in exits if item is not None and _lane_is_passable(road_network, item)]
        if not exits:
            break
        current_id = exits[0]
    return current_id


def _uses_absolute_adversary_init(spec):
    mode = spec.get("init_mode", spec.get("placement_mode", None))
    if mode is not None and str(mode).lower() in ("absolute", "lane_s", "direct"):
        return True
    return "lane_id" in spec


def _absolute_specs_from_relative_template(road_network, template, seed):
    rng = np.random.default_rng(seed)
    ego_template = template["ego"]
    ego_lane_id = _normalize_lane_id(road_network, _sample_template_value(ego_template["lane_id"], rng, as_int=True))
    ego_lane = road_network.get_lane(ego_lane_id)
    ego_s_margin = float(ego_template.get("s_margin", 1.0))
    ego_configured_s = ego_template.get("s", ego_template.get("base_s"))
    ego_s_low_raw, ego_s_high_raw = _as_pair(ego_configured_s, (ego_s_margin, ego_s_margin))
    ego_lane_length = _lane_length(ego_lane)
    ego_legal_low = ego_s_margin
    ego_legal_high = ego_lane_length - ego_s_margin
    if ego_s_low_raw == ego_s_high_raw:
        ego_s_low = ego_s_high = float(np.clip(ego_s_low_raw, ego_legal_low, max(ego_legal_low, ego_legal_high)))
    else:
        ego_s_low = max(ego_s_low_raw, ego_legal_low)
        ego_s_high = min(ego_s_high_raw, ego_legal_high)
    if ego_s_low > ego_s_high:
        raw_lane_length = _finite_float(getattr(ego_lane, "length", 0.0), 0.0)
        raise ValueError(
            f"ego s range [{ego_s_low:.2f}, {ego_s_high:.2f}] is outside lane {ego_lane_id}; "
            f"computed_lane_length={ego_lane_length:.2f}, raw_lane_length={raw_lane_length:.2f}, "
            f"configured_s={ego_configured_s!r}, s_margin={ego_s_margin:.2f}"
        )
    ego_s = float(rng.uniform(ego_s_low, ego_s_high))

    ego_spec = copy.deepcopy(ego_template)
    ego_spec["lane_id"] = ego_lane_id
    ego_spec["base_s"] = ego_s
    ego_spec["base_l"] = float(ego_template.get("l", ego_template.get("base_l", 0.0)))
    ego_spec["destination"] = str(ego_template.get("destination") or _choose_downstream_destination(road_network, ego_lane_id))

    resolved_refs = {
        "ego": {
            "lane_id": ego_lane_id,
            "s": ego_s,
            "root_ego_lane_id": ego_lane_id,
            "label": "ego",
            "placement_mode": "ego",
        }
    }
    adversary_specs = []
    for adv_template in template.get("adversaries", []):
        adv_spec = copy.deepcopy(adv_template)
        adv_label = adv_template.get("name", adv_template.get("vehicle_id", "adversary"))

        if _uses_absolute_adversary_init(adv_template):
            if "lane_id" not in adv_template:
                raise ValueError(f"{adv_label} uses absolute init but does not define lane_id")
            adv_lane_id = _normalize_lane_id(
                road_network,
                _sample_template_value(adv_template["lane_id"], rng, as_int=True),
            )
            adv_lane = road_network.get_lane(adv_lane_id)
            adv_s_margin = float(adv_template.get("s_margin", 1.0))
            adv_configured_s = adv_template.get("s", adv_template.get("base_s", adv_s_margin))
            adv_s_low_raw, adv_s_high_raw = _as_pair(adv_configured_s, (adv_s_margin, adv_s_margin))
            adv_lane_length = _lane_length(adv_lane)
            adv_legal_low = adv_s_margin
            adv_legal_high = adv_lane_length - adv_s_margin
            if adv_s_low_raw == adv_s_high_raw:
                adv_s = float(np.clip(adv_s_low_raw, adv_legal_low, max(adv_legal_low, adv_legal_high)))
            else:
                adv_s_low = max(adv_s_low_raw, adv_legal_low)
                adv_s_high = min(adv_s_high_raw, adv_legal_high)
                if adv_s_low > adv_s_high:
                    raw_lane_length = _finite_float(getattr(adv_lane, "length", 0.0), 0.0)
                    raise ValueError(
                        f"{adv_label} s range [{adv_s_low:.2f}, {adv_s_high:.2f}] is outside lane {adv_lane_id}; "
                        f"computed_lane_length={adv_lane_length:.2f}, raw_lane_length={raw_lane_length:.2f}, "
                        f"configured_s={adv_configured_s!r}, s_margin={adv_s_margin:.2f}"
                    )
                adv_s = float(rng.uniform(adv_s_low, adv_s_high))

            print(
                "[absolute init]",
                adv_label,
                "lane=",
                adv_lane_id,
                "configured_s=",
                adv_configured_s,
                "base_s=",
                f"{adv_s:.2f}",
                "s_noise=",
                adv_template.get("s_noise", [0.0, 0.0]),
                flush=True,
            )

            adv_spec["lane_id"] = adv_lane_id
            adv_spec["base_s"] = float(adv_s)
            adv_spec["base_l"] = float(adv_template.get("l", adv_template.get("base_l", 0.0)))
            adv_spec["destination"] = str(
                adv_template.get("destination")
                or _choose_downstream_destination(road_network, adv_lane_id)
            )
            adv_spec["metadata"] = {
                **adv_spec.get("metadata", {}),
                "relative_to": None,
                "relative_to_ego_lane_id": ego_lane_id,
                "relative_to_lane_id": None,
                "relative_to_s": None,
                "adjacent_lane_id": None,
                "final_lane_id": adv_lane_id,
                "lane_offset": None,
                "relative_s": None,
                "placement_mode": "absolute",
                "requested_base_s": float(adv_configured_s)
                if isinstance(adv_configured_s, (int, float))
                else None,
                "requested_s_bounds": [adv_s_low_raw, adv_s_high_raw],
                "s_noise": copy.deepcopy(adv_template.get("s_noise", [0.0, 0.0])),
            }
            adversary_specs.append(adv_spec)
            ref_keys = [
                adv_spec.get("vehicle_id"),
                adv_spec.get("name"),
            ]
            for key in ref_keys:
                if key is not None:
                    resolved_refs[str(key)] = {
                        "lane_id": adv_lane_id,
                        "s": float(adv_s),
                        "root_ego_lane_id": ego_lane_id,
                        "label": str(key),
                        "placement_mode": "absolute",
                    }
            continue

        relative_to = str(adv_template.get("relative_to", "ego"))
        if relative_to not in resolved_refs:
            raise ValueError(
                f"{adv_label} references "
                f"{relative_to!r}, but only previous ego/adversary refs are available: {sorted(resolved_refs)}"
            )
        ref = resolved_refs[relative_to]
        ref_lane_id = ref["lane_id"]
        ref_lane = road_network.get_lane(ref_lane_id)
        ref_s = float(ref["s"])

        lane_offset = int(_sample_template_value(adv_template.get("lane_offset", 0), rng, as_int=True))
        try:
            adjacent_lane_id = resolve_relative_lane(road_network, ref_lane_id, lane_offset)
        except ValueError as exc:
            raise ValueError(
                f"{adv_label} failed to resolve "
                f"lane_offset={lane_offset} from {relative_to} lane {ref_lane_id} "
                f"(ref placement_mode={ref.get('placement_mode')}); {exc}"
            ) from exc
        if lane_offset != 0 and adjacent_lane_id == ref_lane_id:
            raise ValueError(
                f"{adv_label} lane_offset={lane_offset} "
                f"resolved to the same lane {ref_lane_id}. Check OSM left/right neighbor refs for this lane."
            )
        adjacent_lane = road_network.get_lane(adjacent_lane_id)
        projected_ref_s, _ = _lane_local_coordinates(adjacent_lane, np.asarray(_lane_position(ref_lane, ref_s, 0.0), dtype=float))

        rel_low, rel_high = _as_pair(adv_template.get("relative_s", 0.0), (0.0, 0.0))
        adv_s_margin = float(adv_template.get("s_margin", 1.0))
        relative_s = float(rng.uniform(rel_low, rel_high))
        allow_geometric_projection_fallback = bool(
            adv_template.get(
                "allow_geometric_projection_fallback",
                adv_template.get("allow_projection_fallback", False),
            )
        )
        adv_lane_id, adv_s, placement_mode = resolve_longitudinal_relative_lane(
            road_network,
            adjacent_lane_id,
            projected_ref_s,
            relative_s,
            s_margin=adv_s_margin,
            max_hops=int(adv_template.get("longitudinal_search_hops", 12)),
            allow_geometric_projection_fallback=allow_geometric_projection_fallback,
        )
        preserve_lane_offset = bool(adv_template.get("preserve_lane_offset", True))
        if preserve_lane_offset and lane_offset != 0 and adv_lane_id == ref_lane_id:
            adv_lane_id = adjacent_lane_id
            adv_s = _clip_s_to_lane(adjacent_lane, projected_ref_s + relative_s, s_margin=adv_s_margin)
            placement_mode = "clipped_adjacent_lane_offset_preserved"
        if lane_offset != 0 and adv_lane_id == ref_lane_id:
            raise ValueError(
                f"{adv_label} lane_offset={lane_offset} "
                f"resolved back to reference lane {ref_lane_id}; adjacent_lane={adjacent_lane_id}, "
                f"placement_mode={placement_mode}. Set preserve_lane_offset=True or choose another ego/relative_s."
            )

        print(
            "[relative init]",
            adv_label,
            "relative_to=",
            relative_to,
            "ref_lane=",
            ref_lane_id,
            "lane_offset=",
            lane_offset,
            "side=",
            "right" if lane_offset < 0 else "left" if lane_offset > 0 else "same",
            "adjacent_lane=",
            adjacent_lane_id,
            "adjacent_exits=",
            _ordered_lane_refs(road_network, adjacent_lane_id, "exit_lanes"),
            "adjacent_raw_exits=",
            _raw_lane_refs(road_network, adjacent_lane_id, "exit_lanes"),
            "adjacent_entries=",
            _ordered_lane_refs(road_network, adjacent_lane_id, "entry_lanes"),
            "adjacent_raw_entries=",
            _raw_lane_refs(road_network, adjacent_lane_id, "entry_lanes"),
            "final_lane=",
            adv_lane_id,
            "ref_s=",
            f"{ref_s:.2f}",
            "projected_s=",
            f"{projected_ref_s:.2f}",
            "relative_s=",
            f"{relative_s:.2f}",
            "final_s=",
            f"{adv_s:.2f}",
            "placement_mode=",
            placement_mode,
            "allow_geometric_projection_fallback=",
            allow_geometric_projection_fallback,
            flush=True,
        )

        adv_spec["lane_id"] = adv_lane_id
        adv_spec["base_s"] = float(adv_s)
        adv_spec["base_l"] = float(adv_template.get("l", adv_template.get("base_l", 0.0)))
        adv_spec["destination"] = str(adv_template.get("destination") or _choose_downstream_destination(road_network, adv_lane_id))
        adv_spec["metadata"] = {
            **adv_spec.get("metadata", {}),
            "relative_to": relative_to,
            "relative_to_ego_lane_id": ego_lane_id,
            "relative_to_lane_id": ref_lane_id,
            "relative_to_s": ref_s,
            "adjacent_lane_id": adjacent_lane_id,
            "final_lane_id": adv_lane_id,
            "projected_ref_s_on_adjacent_lane": float(projected_ref_s),
            "lane_offset": lane_offset,
            "relative_s": relative_s,
            "placement_mode": placement_mode,
            "allow_geometric_projection_fallback": allow_geometric_projection_fallback,
            "requested_relative_s_bounds": [rel_low, rel_high],
        }
        adversary_specs.append(adv_spec)
        ref_keys = [
            adv_spec.get("vehicle_id"),
            adv_spec.get("name"),
        ]
        for key in ref_keys:
            if key is not None:
                resolved_refs[str(key)] = {
                    "lane_id": adv_lane_id,
                    "s": float(adv_s),
                    "root_ego_lane_id": ego_lane_id,
                    "label": str(key),
                    "placement_mode": placement_mode,
                }

    return ego_spec, adversary_specs


def print_custom_init_summary(cfg):
    print("[custom init] default_agent:", cfg["agent_configs"]["default_agent"])
    print("[custom init] use_custom_adversaries:", cfg.get("use_custom_adversaries"))
    for adv_cfg in cfg.get("custom_adversaries", []):
        metadata = adv_cfg.get("metadata", {})
        print(
            "[custom init] adversary config:",
            adv_cfg.get("name"),
            "lane=",
            adv_cfg.get("spawn_lane_index"),
            "s=",
            adv_cfg.get("spawn_longitude"),
            "pos_heading=",
            adv_cfg.get("spawn_position_heading"),
            "destination=",
            adv_cfg.get("destination"),
            "relative_to=",
            metadata.get("relative_to"),
            "lane_offset=",
            metadata.get("lane_offset"),
            "adjacent_lane=",
            metadata.get("adjacent_lane_id"),
            "placement_mode=",
            metadata.get("placement_mode"),
        )


def print_spawned_traffic(env):
    if not hasattr(env.engine, "traffic_manager"):
        return
    print("[custom init] spawned traffic vehicles:")
    for vehicle in env.engine.traffic_manager.traffic_vehicles:
        print(
            "  ",
            vehicle.name,
            "pos=",
            vehicle.position,
            "lane=",
            getattr(vehicle, "lane_index", None),
            "speed=",
            vehicle.speed,
        )


def run_one_variant(base_cfg, args, variant_index: int, extra_render_args, custom_params=None) -> None:
    custom_params = copy.deepcopy(custom_params or {})
    custom_params.setdefault("adv2_mode", args.adv2_init_mode)
    variant_seed = int(args.variant_seed) + int(variant_index)
    map_seed = get_map_seed(base_cfg, variant_index)
    cfg = copy.deepcopy(base_cfg)

    if args.use_custom_init:
        try:
            cfg = build_custom_init_cfg(cfg, seed=variant_seed, map_seed=map_seed, custom_params=custom_params)
        except Exception as exc:
            print(
                f"[variant {variant_index}] skip: variant_seed={variant_seed}, "
                f"map_seed={map_seed}, reason={type(exc).__name__}: {exc}"
            )
            return
    else:
        print(f"[variant {variant_index}] map_seed={map_seed}")

    if args.save_scenarios:
        cfg["record_episode"] = True
    filename_token = build_relative_position_filename_token(cfg, variant_index, variant_seed)
    if custom_params:
        represented_custom_keys = {"a", "b", "c", "adv2_mode"} if args.use_custom_init else set()
        extra_custom_params = {
            key: value for key, value in custom_params.items()
            if key not in represented_custom_keys
        }
        combo_token = "_".join(
            f"{_safe_file_token(key)}{_meter_token(value) if isinstance(value, (int, float)) else _safe_file_token(value)}"
            for key, value in sorted(extra_custom_params.items())
        )
        if combo_token:
            filename_token = f"{filename_token}_{combo_token}"
        print(f"[variant {variant_index}] custom_params={custom_params}")
    print(f"[variant {variant_index}] filename_token={filename_token}")

    env = None
    try:
        env = OSMEnv(cfg)
        try:
            env.reset(seed=map_seed)
        except Exception as exc:
            print(
                f"[variant {variant_index}] skip: variant_seed={variant_seed}, "
                f"map_seed={map_seed}, reason={type(exc).__name__}: {exc}"
            )
            return
        # if args.use_custom_init:
        #     print_spawned_traffic(env)
        track_agent_name = next(
            (name for name, vehicle in getattr(env.engine, "agents", {}).items() if vehicle is env.agent),
            "default_agent"
        )

        step = -1
        try:
            for step in range(int(args.max_steps)):
                _, _, terminated, truncated, info = env.step([1.0, 0.])
                text = {"Frame": f"{step}", "Seed": variant_seed} if args.top_down else RENDER_MESSAGE
                env.render(
                    mode="top_down" if args.top_down else None,
                    scaling=3,
                    show_agent_name=True,
                    show_navigation_path=True,
                    show_traffic_navigation_paths=True,
                    current_track_agent=track_agent_name,
                    window=False,
                    screen_record=args.save_gif,
                    text=text,
                    **extra_render_args
                )
                if terminated or truncated:
                    break
        except Exception as exc:
            print(
                f"[variant {variant_index}] skip: variant_seed={variant_seed}, "
                f"map_seed={map_seed}, step={step}, reason={type(exc).__name__}: {exc}"
            )
            return

        try:
            if args.save_gif:
                save_episode_gif(env, args.gif_dir, filename_token, args.gif_duration)
            if args.save_scenarios:
                save_scenario_description(
                    env,
                    args.scenario_dir,
                    variant_seed,
                    filename_token,
                    export_planned_global_paths=args.export_planned_global_paths,
                    export_planned_path_speed_mps=args.export_planned_path_speed_mps,
                    target_horizon=args.max_steps,
                )
        except Exception as exc:
            print(
                f"[variant {variant_index}] skip save: variant_seed={variant_seed}, "
                f"map_seed={map_seed}, reason={type(exc).__name__}: {exc}"
            )
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                print(
                    f"[variant {variant_index}] skip close error: variant_seed={variant_seed}, "
                    f"map_seed={map_seed}, reason={type(exc).__name__}: {exc}"
                )

def build_custom_init_cfg(base_cfg, seed, map_seed=None, custom_params=None):
    """
    Build one randomized init config for ego + adversarial vehicles.

    Fill the template below with the ego lane/initial s and adversary offsets relative to ego.
    The resolver first uses road_network left/right topology. If topology is missing, it infers
    neighbors from center-line geometry, then validates same direction, distance, overlap and passability.

    Units:
        ego.s: scalar or [low, high] meters along ego lane.
        adversary.relative_to: "ego" or a previous adversary vehicle_id/name. Default is "ego".
        adversary.relative_s: scalar or [low, high] meters relative to relative_to projected on target lane.
            Positive is ahead of relative_to; negative is behind relative_to. The search follows the
            lane centerline and entry/exit topology chain.
        adversary.lane_offset: 0 same lane as relative_to, -N N lanes right, +N N lanes left.
        adversary.allow_geometric_projection_fallback: false by default; enable only when you explicitly
            want a straight endpoint projection to geometrically nearby lanes.
        adversary.lane_id + base_s/s: absolute lane init. If lane_id is set, relative_to/lane_offset/relative_s
            are ignored for that adversary; s_noise still samples around base_s in build_variant_config.
        base_l/l_noise: meters lateral to target lane center.
        base_speed: m/s.
    """
    custom_params = custom_params or {}
    a = float(custom_params.get("a", 15.0))
    b = float(custom_params.get("b", 12.0))
    c = float(custom_params.get("c", 20.0))
    adv2_mode = str(custom_params.get("adv2_mode", "absolute")).lower()
    if adv2_mode not in ("absolute", "relative"):
        raise ValueError(f"Unsupported adv2_mode={adv2_mode!r}; expected 'absolute' or 'relative'")

    adv2_spec = {
        "vehicle_id": "2",
        "name": "adv_2",
        "l": 0.0,
        "base_speed": 10.0,
        "s_margin": 0.0,
        "s_noise": [0.0, 0.0],
        "l_noise": [0.0, 0.0],
        "speed_scale": [0.99, 1.01],
        "vehicle_config": {
            **RELAXED_IDM_CONFIG,
            "idm_target_speed_km_h": 36,
            "idm_target_speed_min_km_h": 20,
            "idm_target_speed_max_km_h": 45,
        },
    }
    if adv2_mode == "absolute":
        adv2_spec.update({
            "init_mode": "absolute",
            "lane_id": ["-101869", "-101868"],
            "base_s": b,
        })
    else:
        adv2_spec.update({
            "relative_to": "adv_1",
            "lane_offset": -1,
            "relative_s": [5.0, 15.0],
        })

    relative_template = {
        # Minimum same-lane longitudinal gap used by build_variant_config.
        # Lower this when you intentionally explore close adversarial placements.
        "min_same_lane_gap_m": 4.0,
        "ego": {
            "vehicle_id": "ego",
            "lane_id": ["-101886"],  # you can specify multiple ego lane ids for sampling
            "s": [c, c],
            "l": 0.0,
            "base_speed": 12.0,
            # Omit destination to automatically pick a downstream lane.
            # "destination": "63103",
            "s_noise": [0.0, 0.0],
            "l_noise": [0.0, 0.0],
            "speed_scale": [0.99, 1.01],
            "vehicle_config": {
                **RELAXED_IDM_CONFIG,
                "idm_target_speed_km_h": 42,
                "idm_target_speed_min_km_h": 30,
                "idm_target_speed_max_km_h": 50,
            },
        },
        "adversaries": [
            {
                "vehicle_id": "1",
                "name": "adv_1",
                "lane_offset": -1, # -右+左
                "relative_s": [a, a],
                # "relative_s": [10, 20],
                "l": 0.0,
                "base_speed": 10.0,
                "s_noise": [0.0, 0.0],
                "l_noise": [0.0, 0.0],
                "speed_scale": [0.99, 1.01],
                "vehicle_config": {
                    **RELAXED_IDM_CONFIG,
                    "idm_target_speed_km_h": 36,
                    "idm_target_speed_min_km_h": 20,
                    "idm_target_speed_max_km_h": 45,
                },
            },
            adv2_spec,
        ],
    }
    print(
        f"[custom init template] c={c:.2f}, a={a:.2f}, b={b:.2f}, "
        f"adv2_mode={adv2_mode}, seed={seed}, map_seed={map_seed}"
    )

    validate_ids_before_probe(base_cfg, relative_template["ego"], [])

    probe_env = None
    try:
        probe_env = OSMEnv(make_map_probe_cfg(base_cfg, relative_template["ego"]))
        probe_env.reset(seed=get_map_seed(base_cfg, 0) if map_seed is None else map_seed)
        road_network = probe_env.engine.current_map.road_network
        ego_spec, adversary_specs = _absolute_specs_from_relative_template(road_network, relative_template, seed)

        cfg = build_variant_config(
            base_config=base_cfg,
            road_network=road_network,
            ego_spec=ego_spec,
            adversary_specs=adversary_specs,
            seed=seed,
            min_same_lane_gap_m=float(relative_template.get("min_same_lane_gap_m", 7.0)),
        )
        for adv_cfg, adv_spec in zip(cfg.get("custom_adversaries", []), adversary_specs):
            adv_cfg.setdefault("metadata", {}).update(copy.deepcopy(adv_spec.get("metadata", {})))
            print(
                "[custom init resolved]",
                adv_spec.get("name", adv_spec.get("vehicle_id", "adversary")),
                "requested_lane=",
                adv_spec.get("lane_id"),
                "spawn_lane=",
                adv_cfg.get("spawn_lane_index"),
                "requested_s=",
                f"{float(adv_spec.get('base_s', 0.0)):.2f}",
                "spawn_s=",
                adv_cfg.get("spawn_longitude"),
                "relative_to=",
                adv_spec.get("metadata", {}).get("relative_to"),
                "lane_offset=",
                adv_spec.get("metadata", {}).get("lane_offset"),
                "adjacent_lane=",
                adv_spec.get("metadata", {}).get("adjacent_lane_id"),
                "final_lane=",
                adv_spec.get("metadata", {}).get("final_lane_id"),
                "placement_mode=",
                adv_spec.get("metadata", {}).get("placement_mode"),
                flush=True,
            )
        cfg["export_ego_vehicle_id"] = str(ego_spec["vehicle_id"])
        return cfg
    finally:
        if probe_env is not None:
            probe_env.close()


def _validate_sample_range(start, stop, name):
    start = float(start)
    stop = float(stop)
    if stop <= start:
        raise ValueError(f"{name} range should satisfy stop > start, got [{start}, {stop})")
    return start, stop


def _sample_uniform_range(rng, start, stop, name):
    start, stop = _validate_sample_range(start, stop, name)
    return float(rng.uniform(start, stop))


def _make_grid_values(start, stop, step, name):
    start, stop = _validate_sample_range(start, stop, name)
    step = float(step)
    if step <= 0:
        raise ValueError(f"{name}_step must be positive, got {step}")
    values = []
    current = start
    while current < stop - 1e-9:
        values.append(float(current))
        current += step
    if not values:
        raise ValueError(f"{name} grid is empty: start={start}, stop={stop}, step={step}")
    return values


def _iter_grid_combos(args):
    ego_s_values = _make_grid_values(args.ego_s_start, args.ego_s_stop, args.ego_s_step, "ego_s")
    a_values = _make_grid_values(args.adv1_a_start, args.adv1_a_stop, args.adv1_a_step, "adv1_a")
    b_values = _make_grid_values(args.adv2_b_start, args.adv2_b_stop, args.adv2_b_step, "adv2_b")
    for c in ego_s_values:
        for a in a_values:
            for b in b_values:
                for _ in range(int(args.seeds_per_grid)):
                    yield c, a, b


def _arrival_time(distance_m, speed_mps):
    if distance_m < 0.0 or speed_mps <= 1e-6:
        return None
    return distance_m / speed_mps


def build_timing_filtered_custom_combo(
    ego_s,
    a,
    b,
    *,
    ego_speed_mps=12.0,
    black_speed_mps=10.0,
    red_speed_mps=10.0,
    ego_lane_end_s=90.0,
    black_lane_end_s=90.0,
    red_lane_end_s=50.0,
    red_before_black_min_s=0.0,
    red_before_black_max_s=2.0,
    ego_after_black_min_s=0.3,
    ego_after_black_max_s=2.0,
):
    black_base_s = float(ego_s) + float(a)
    ego_dist = float(ego_lane_end_s) - float(ego_s)
    black_dist = float(black_lane_end_s) - black_base_s
    red_dist = float(red_lane_end_s) - float(b)
    t_ego = _arrival_time(ego_dist, ego_speed_mps)
    t_black = _arrival_time(black_dist, black_speed_mps)
    t_red = _arrival_time(red_dist, red_speed_mps)
    if t_ego is None or t_black is None or t_red is None:
        return None

    red_before_black = t_black - t_red
    ego_after_black = t_ego - t_black
    if not (red_before_black_min_s <= red_before_black <= red_before_black_max_s):
        return None
    if not (ego_after_black_min_s <= ego_after_black <= ego_after_black_max_s):
        return None

    return {
        "c": float(ego_s),
        "a": float(a),
        "b": float(b),
        "black_base_s": black_base_s,
        "t_red": t_red,
        "t_black": t_black,
        "t_ego": t_ego,
        "red_before_black": red_before_black,
        "ego_after_black": ego_after_black,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reactive_traffic", action="store_true")
    parser.add_argument("--top_down", nargs="?", const=True, default=True, type=parse_bool)
    parser.add_argument("--waymo", action="store_true")
    parser.add_argument("--add_sensor", action="store_true")
    parser.add_argument("--save_gif", nargs="?", const=True, default=True, type=parse_bool)
    parser.add_argument("--gif_dir", type=Path, default=Path("gif_record1"))
    parser.add_argument("--gif_duration", type=int, default=100)
    parser.add_argument("--use_custom_init", nargs="?", const=True, default=True, type=parse_bool)
    parser.add_argument(
        "--sample_mode",
        choices=["random", "grid"],
        default="random",
        help="random: continuous uniform c/a/b samples; grid: enumerate start/stop/step values for heatmaps.",
    )
    parser.add_argument(
        "--adv2_init_mode",
        choices=["absolute", "relative"],
        default="absolute",
        help="adv_2 init mode: absolute uses lane_id/base_s grid; relative uses relative_to/lane_offset/relative_s.",
    )
    parser.add_argument("--variant_seed", type=int, default=0)
    parser.add_argument("--num_variants", type=int, default=100)
    parser.add_argument(
        "--num_variants_mode",
        choices=["total", "per_combo"],
        default="total",
        help="Compatibility option. Timing sampler now always stops after num_variants accepted samples.",
    )
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--save_scenarios", nargs="?", const=True, default=False, type=parse_bool)
    parser.add_argument("--scenario_dir", type=Path, default=Path("scenario_record1"))
    parser.add_argument("--export_planned_global_paths", nargs="?", const=True, default=True, type=parse_bool)
    parser.add_argument("--export_planned_path_speed_mps", type=float, default=8.0)
    parser.add_argument("--list_lanes", action="store_true")
    parser.add_argument("--ego_s_start", type=int, default=20)
    parser.add_argument("--ego_s_stop", type=int, default=30)
    parser.add_argument("--ego_s_step", type=int, default=5)
    parser.add_argument("--adv1_a_start", type=int, default=-5)
    parser.add_argument("--adv1_a_stop", type=int, default=5)
    parser.add_argument("--adv1_a_step", type=int, default=5)
    parser.add_argument("--adv2_b_start", type=int, default=0)
    parser.add_argument("--adv2_b_stop", type=int, default=15)
    parser.add_argument("--adv2_b_step", type=int, default=5)
    parser.add_argument("--seeds_per_grid", type=int, default=1)
    parser.add_argument(
        "--timing_sample_attempts",
        type=int,
        default=0,
        help="Maximum random timing candidates to try. <=0 means auto: max(10000, num_variants * 1000).",
    )
    parser.add_argument("--red_black_arrival_min", type=float, default=-1.0)
    parser.add_argument("--red_black_arrival_max", type=float, default=2.0)
    parser.add_argument("--ego_black_arrival_min", type=float, default=0.3)
    parser.add_argument("--ego_black_arrival_max", type=float, default=2.0)
    parser.add_argument(
        "--use_timing_filter",
        nargs="?",
        const=True,
        default=True,
        type=parse_bool,
        help="Whether to reject sampled c/a/b by arrival-time constraints. Set false for full-space heatmap data.",
    )
    args = parser.parse_args() 
    extra_args = dict(film_size=(2000, 2000)) if args.top_down else {}
    asset_path = AssetLoader.asset_path
    use_waymo = args.waymo
    print(HELP_MESSAGE)

    cfg = {
        "osm_file_path": "TesT_Field_Tongji_East_intersection_name_topology_fixed.osm",
        "manual_control": False,
        "map_region_size": 512,  # use a large number if your map is toooooo big
        "sequential_seed": True,
        "reactive_traffic": True if args.reactive_traffic else False,
        "use_render": False,
        "total_vehicle_count": 5,
        "traffic_initial_min_spawn_distance_to_ego": 35.0,
        "max_lateral_dist": 20,
        "out_of_route_done": False,
        "enable_idm_lane_change": False,
        "debug_osm_route": False,
        "debug_osm_route_all_vehicles": True,
        "data_directory": '/data_set/nuplan/nuplan', # 没用
        "num_scenarios":100,
        "agent_configs": {
        "default_agent": dict(
            use_special_color=True,
            navigation_module=OsmTrajectoryNavigation,
            spawn_lane_index="-3990",
            destination="-4351",
        ),
    },
        "vehicle_config": dict(
            **RELAXED_IDM_CONFIG,
            idm_target_speed_km_h=20,
            idm_target_speed_min_km_h=20,
            idm_target_speed_max_km_h=20,
            idm_respect_lane_speed_limit=True,
        ),
    }
    if args.add_sensor:
        additional_cfg = {
            "interface_panel": ["rgb_camera", "depth_camera", "semantic"],
            "sensors": {
                "rgb_camera": (DepthCamera, 256, 256),
                "depth_camera": (RGBCamera, 256, 256),
                "semantic": (SemanticCamera, 256, 256)
            }
        }
        cfg.update(additional_cfg)

    if args.list_lanes:
        lane_ids = get_osm_feature_ids(cfg)
        print(f"[osm lanes] count={len(lane_ids)}")
        print("[osm lanes]", " ".join(lane_ids))
        sys.exit(0)

    total_variants = int(args.num_variants)
    max_timing_attempts = int(args.timing_sample_attempts)
    if args.sample_mode == "grid":
        max_timing_attempts = total_variants
    elif not args.use_timing_filter:
        max_timing_attempts = total_variants
    elif max_timing_attempts <= 0:
        max_timing_attempts = max(10000, total_variants * 1000)

    ego_s_range = _validate_sample_range(args.ego_s_start, args.ego_s_stop, "ego_s")
    a_range = _validate_sample_range(args.adv1_a_start, args.adv1_a_stop, "adv1_a")
    b_range = _validate_sample_range(args.adv2_b_start, args.adv2_b_stop, "adv2_b")
    grid_total = None
    if args.sample_mode == "grid":
        grid_total = (
            len(_make_grid_values(args.ego_s_start, args.ego_s_stop, args.ego_s_step, "ego_s"))
            * len(_make_grid_values(args.adv1_a_start, args.adv1_a_stop, args.adv1_a_step, "adv1_a"))
            * len(_make_grid_values(args.adv2_b_start, args.adv2_b_stop, args.adv2_b_step, "adv2_b"))
            * int(args.seeds_per_grid)
        )
    print(
        f"[custom sampler] ego_s=[{ego_s_range[0]:.1f},{ego_s_range[1]:.1f}), "
        f"a=[{a_range[0]:.1f},{a_range[1]:.1f}), b=[{b_range[0]:.1f},{b_range[1]:.1f}), "
        f"sample_mode={args.sample_mode}, seeds_per_grid={int(args.seeds_per_grid)}, "
        f"use_timing_filter={args.use_timing_filter}, "
        f"rule=red_before_black[{args.red_black_arrival_min:.2f},{args.red_black_arrival_max:.2f}]s "
        f"and ego_after_black[{args.ego_black_arrival_min:.2f},{args.ego_black_arrival_max:.2f}]s, "
        f"max_attempts={max_timing_attempts}"
    )
    if grid_total is not None:
        print(f"[custom sampler] grid_total={grid_total}, num_variants={total_variants}")
    if args.num_variants_mode != "total":
        print("[timing sampler] num_variants_mode is kept for compatibility and is ignored in random timing sampling.")

    variant_index = 0
    skipped_timing = 0
    candidate_rng = np.random.default_rng(int(args.variant_seed))
    if args.sample_mode == "grid":
        candidates = _iter_grid_combos(args)
    else:
        candidates = (
            (
                _sample_uniform_range(candidate_rng, args.ego_s_start, args.ego_s_stop, "ego_s"),
                _sample_uniform_range(candidate_rng, args.adv1_a_start, args.adv1_a_stop, "adv1_a"),
                _sample_uniform_range(candidate_rng, args.adv2_b_start, args.adv2_b_stop, "adv2_b"),
            )
            for _ in range(max_timing_attempts)
        )

    for attempt, (c, a, b) in enumerate(candidates):
        if variant_index >= total_variants:
            print(f"[custom sampler] reached num_variants={total_variants}, stop.")
            sys.exit(0)
        if attempt >= max_timing_attempts:
            break
        if args.use_timing_filter:
            combo = build_timing_filtered_custom_combo(
                c,
                a,
                b,
                red_before_black_min_s=float(args.red_black_arrival_min),
                red_before_black_max_s=float(args.red_black_arrival_max),
                ego_after_black_min_s=float(args.ego_black_arrival_min),
                ego_after_black_max_s=float(args.ego_black_arrival_max),
            )
            if combo is None:
                skipped_timing += 1
                continue
            timing_text = (
                f"black_base_s={combo['black_base_s']:.2f}, "
                f"t_red={combo['t_red']:.2f}, t_black={combo['t_black']:.2f}, t_ego={combo['t_ego']:.2f}, "
                f"red_before_black={combo['red_before_black']:.2f}, "
                f"ego_after_black={combo['ego_after_black']:.2f}, "
            )
            log_prefix = "[timing accepted]"
        else:
            combo = {"c": float(c), "a": float(a), "b": float(b)}
            timing_text = ""
            log_prefix = "[sample accepted]"

        print(
            f"{log_prefix} attempt={attempt}, variant={variant_index}, "
            f"c={combo['c']:.2f}, a={combo['a']:.2f}, b={combo['b']:.2f}, "
            f"{timing_text}"
            f"variant_seed={int(args.variant_seed) + variant_index}, skipped_timing={skipped_timing}"
        )
        run_one_variant(
            cfg,
            args,
            variant_index,
            extra_args,
            custom_params={
                "a": combo["a"],
                "b": combo["b"],
                "c": combo["c"],
            },
        )
        variant_index += 1

    print(
        f"[custom sampler] stop: accepted={variant_index}, requested={total_variants}, "
        f"skipped_timing={skipped_timing}, attempts={max_timing_attempts}. "
        "Increase --timing_sample_attempts or loosen timing/range constraints if accepted < requested."
    )
