"""
Utilities for building OSM scenario variants from lane-relative initial states.

The caller provides vehicle ids, lane ids, reference longitudinal positions, and speed ranges.
These helpers sample deterministic variants and produce MetaDrive config fragments that OSMEnv
and TrafficManagerInOsm can consume.
"""
import copy
from typing import Iterable, Mapping, Optional

import numpy as np

from metadrive.constants import DEFAULT_AGENT


def _rng(seed=None):
    return np.random.default_rng(seed)


def _sample_choice(value, rng):
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            raise ValueError("Can not sample from an empty list")
        return value[int(rng.integers(0, len(value)))]
    return value


def _range_pair(spec: Mapping, key: str, default):
    value = spec.get(key, default)
    if isinstance(value, (int, float)):
        if key == "speed_scale":
            value = (1.0 - float(value), 1.0 + float(value))
        else:
            value = (-float(value), float(value))
    if len(value) != 2:
        raise ValueError(f"{key} should be a scalar or a pair, got {value}")
    return float(value[0]), float(value[1])


def _clip_lateral(lane, lateral, margin):
    lane_width = float(getattr(lane, "width", 3.5) or 3.5)
    half_width = max(lane_width * 0.5 - margin, 0.0)
    return float(np.clip(lateral, -half_width, half_width))


def sample_lane_relative_init(road_network, spec: Mapping, rng=None):
    """
    Sample one vehicle initial state near a lane-relative reference.

    Required spec keys:
        lane_id: lane id in the parsed OSM map.

    Common optional keys:
        vehicle_id/name: kept in metadata/name only.
        base_s: reference longitudinal coordinate on lane, meters.
        base_l: reference lateral coordinate on lane, meters.
        base_speed: reference speed, m/s.
        destination: navigation destination lane id.
        s_noise/l_noise/speed_noise/heading_noise: scalar range or [low, high].
        speed_scale: [low, high] multiplicative speed range.
    """
    rng = rng or _rng()
    lane_id = str(_sample_choice(spec["lane_id"], rng))
    lane = road_network.get_lane(lane_id)
    destination = spec.get("destination")
    if destination is not None:
        destination = str(_sample_choice(destination, rng))

    base_s = float(spec.get("base_s", spec.get("s", 1.0)))
    base_l = float(spec.get("base_l", spec.get("l", 0.0)))
    base_speed = float(spec.get("base_speed", spec.get("speed", 0.0)))

    s_low, s_high = _range_pair(spec, "s_noise", (-3.0, 3.0))
    l_low, l_high = _range_pair(spec, "l_noise", (-0.3, 0.3))
    speed_low, speed_high = _range_pair(spec, "speed_noise", (0.0, 0.0))
    scale_low, scale_high = _range_pair(spec, "speed_scale", (1.0, 1.0))
    heading_low, heading_high = _range_pair(spec, "heading_noise", (0.0, 0.0))

    lane_length = float(getattr(lane, "length", 1.0) or 1.0)
    s_margin = float(spec.get("s_margin", 1.0))
    l_margin = float(spec.get("l_margin", 0.2))

    s = base_s + rng.uniform(s_low, s_high)
    s = float(np.clip(s, s_margin, max(s_margin, lane_length - s_margin)))
    lateral = _clip_lateral(lane, base_l + rng.uniform(l_low, l_high), l_margin)

    speed = (base_speed + rng.uniform(speed_low, speed_high)) * rng.uniform(scale_low, scale_high)
    speed = max(0.0, float(speed))

    position = np.asarray(lane.position(s, lateral), dtype=float)
    heading = float(lane.heading_theta_at(s) + rng.uniform(heading_low, heading_high))

    return {
        "vehicle_id": spec.get("vehicle_id", spec.get("track_id", spec.get("name"))),
        "name": spec.get("name"),
        "lane_id": lane_id,
        "s": s,
        "l": lateral,
        "position": position,
        "heading": heading,
        "speed": speed,
        "destination": destination,
    }


def iter_lane_destination_values(spec: Mapping):
    for field in ("lane_id", "destination"):
        values = spec.get(field)
        if values is None:
            continue
        if not isinstance(values, (list, tuple)):
            values = [values]
        for value in values:
            yield field, value


def get_available_lane_ids(road_network):
    """Return sorted lane ids from the parsed OSM road network."""
    graph = getattr(road_network, "graph", {}) or {}
    return sorted(str(lane_id) for lane_id in graph.keys())


def validate_variant_specs(road_network, ego_spec: Mapping, adversary_specs: Iterable[Mapping]):
    """Fail early with a readable message when a configured lane/destination id is not in the OSM graph."""
    lane_ids = set(get_available_lane_ids(road_network))
    specs = [("ego", ego_spec)] + [(f"adversary[{i}]", spec) for i, spec in enumerate(adversary_specs)]
    missing = []
    for label, spec in specs:
        for field, value in iter_lane_destination_values(spec):
            if str(value) not in lane_ids:
                missing.append((label, field, str(value)))

    if missing:
        preview = ", ".join(get_available_lane_ids(road_network)[:80])
        detail = "\n".join(f"  {label}.{field}={value}" for label, field, value in missing)
        raise ValueError(
            "Configured OSM lane ids are not in the current road network:\n"
            f"{detail}\n"
            "Use lane ids from the current parsed OSM center lanes. "
            f"First available ids: {preview}"
        )


def build_vehicle_config(sample: Mapping, extra_config: Optional[Mapping] = None):
    """Convert a sampled state into a MetaDrive vehicle_config fragment."""
    config = {
        "spawn_position_heading": (sample["position"], float(sample["heading"])),
        "spawn_lane_index": str(sample["lane_id"]),
        "spawn_longitude": float(sample["s"]),
        "spawn_lateral": float(sample["l"]),
        "spawn_velocity": [float(sample["speed"]), 0.0],
        "spawn_velocity_car_frame": True,
    }
    destination = sample.get("destination")
    if destination is not None:
        config["destination"] = str(destination)
        config["dest_mode"] = 1
    if extra_config:
        config.update(extra_config)
    return config


def _find_same_lane_longitudinal_conflict(samples, min_gap_m):
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            first = samples[i]
            second = samples[j]
            if str(first["lane_id"]) != str(second["lane_id"]):
                continue
            gap = abs(float(first["s"]) - float(second["s"]))
            if gap < float(min_gap_m):
                return first, second, gap
    return None


def build_variant_config(base_config: Mapping, road_network, ego_spec: Mapping, adversary_specs: Iterable[Mapping],
                         seed=None, min_same_lane_gap_m=7.0, max_resample_attempts=100):
    """
    Return a deep-copied OSMEnv config for one sampled scenario variant.

    Fill ego_spec/adversary_specs in your pipeline; this function handles sampling and config layout.
    """
    config = copy.deepcopy(dict(base_config))
    adversary_specs = list(adversary_specs)
    validate_variant_specs(road_network, ego_spec, adversary_specs)

    ego_sample = None
    adv_samples = None
    last_conflict = None
    for attempt in range(int(max_resample_attempts)):
        attempt_seed = None if seed is None else int(seed) + attempt * 9973
        rng = _rng(attempt_seed)
        ego_sample = sample_lane_relative_init(road_network, ego_spec, rng)
        adv_samples = [sample_lane_relative_init(road_network, adv_spec, rng) for adv_spec in adversary_specs]
        samples = [ego_sample] + adv_samples
        last_conflict = _find_same_lane_longitudinal_conflict(samples, min_same_lane_gap_m)
        if last_conflict is None:
            break
    else:
        first, second, gap = last_conflict
        raise ValueError(
            "Failed to sample a non-overlapping custom init after "
            f"{max_resample_attempts} attempts. Last conflict: "
            f"{first.get('vehicle_id') or first.get('name')} and "
            f"{second.get('vehicle_id') or second.get('name')} on lane {first['lane_id']} "
            f"with longitudinal gap {gap:.2f}m < {float(min_same_lane_gap_m):.2f}m."
        )

    agent_configs = copy.deepcopy(config.get("agent_configs", {}))
    default_agent = copy.deepcopy(agent_configs.get(DEFAULT_AGENT, {}))
    default_agent.update(build_vehicle_config(ego_sample, ego_spec.get("vehicle_config")))
    if ego_sample.get("destination") is not None:
        default_agent["destination"] = str(ego_sample["destination"])
    agent_configs[DEFAULT_AGENT] = default_agent
    config["agent_configs"] = agent_configs

    custom_adversaries = []
    for index, (adv_spec, adv_sample) in enumerate(zip(adversary_specs, adv_samples)):
        adv_config = build_vehicle_config(adv_sample, adv_spec.get("vehicle_config"))
        adv_config["name"] = str(adv_spec.get("name") or adv_sample.get("vehicle_id") or f"adv_{index}")
        adv_config["metadata"] = {
            "vehicle_id": adv_sample.get("vehicle_id"),
            "lane_id": adv_sample["lane_id"],
            "s": adv_sample["s"],
            "l": adv_sample["l"],
            "speed": adv_sample["speed"],
        }
        custom_adversaries.append(adv_config)

    config["use_custom_adversaries"] = True
    config["custom_adversaries"] = custom_adversaries
    total_vehicle_count = config.get("total_vehicle_count")
    if total_vehicle_count is not None:
        min_required_vehicle_count = 1 + len(custom_adversaries)
        if int(total_vehicle_count) < min_required_vehicle_count:
            raise ValueError(
                f"total_vehicle_count={total_vehicle_count} is smaller than the required "
                f"ego + custom adversaries count ({min_required_vehicle_count})."
            )
    config["disable_random_traffic_when_custom_adversaries"] = config.get("total_vehicle_count") is None
    return config


def build_variant_configs(base_config: Mapping, road_network, ego_spec: Mapping, adversary_specs: Iterable[Mapping],
                          num_variants: int, base_seed: int = 0):
    """Build a list of sampled OSMEnv configs for batch rollout/export."""
    adversary_specs = list(adversary_specs)
    return [
        build_variant_config(
            base_config=base_config,
            road_network=road_network,
            ego_spec=ego_spec,
            adversary_specs=adversary_specs,
            seed=base_seed + variant_index,
        )
        for variant_index in range(int(num_variants))
    ]
