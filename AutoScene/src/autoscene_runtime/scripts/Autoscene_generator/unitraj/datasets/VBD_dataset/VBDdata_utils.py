import numpy as np
import torch
from metadrive.type import MetaDriveType


SUPPORTED_AGENT_TYPES = {
    "VEHICLE": 1,
    "PEDESTRIAN": 2,
    "CYCLIST": 3,
}

# Keep the same semantic ids used by the existing AutoScene VBD training adapter.
# These ids are consumed by VBD MapEncoder.type_embed (0..20).
SCENARIONET_MAP_TYPE_TO_VBD = {
    "LANE_UNKNOWN": 0,
    "LANE_FREEWAY": 1,
    "LANE_SURFACE_STREET": 2,
    "LANE_BIKE_LANE": 3,
    "LANE_SURFACE_UNSTRUCTURE": 4,
    "DRIVEWAY": 5,
    "ROAD_LINE_BROKEN_SINGLE_WHITE": 6,
    "ROAD_LINE_SOLID_SINGLE_WHITE": 7,
    "ROAD_LINE_SOLID_DOUBLE_WHITE": 8,
    "ROAD_LINE_BROKEN_SINGLE_YELLOW": 9,
    "ROAD_LINE_BROKEN_DOUBLE_YELLOW": 10,
    "ROAD_LINE_SOLID_SINGLE_YELLOW": 11,
    "ROAD_LINE_SOLID_DOUBLE_YELLOW": 12,
    "ROAD_LINE_PASSING_DOUBLE_YELLOW": 13,
    "ROAD_EDGE_SIDEWALK": 14,
    "ROAD_EDGE_BOUNDARY": 15,
    "ROAD_EDGE_MEDIAN": 16,
    "STOP_SIGN": 17,
    "CROSSWALK": 18,
    "SPEED_BUMP": 19,
}

TRAFFIC_LIGHT_STATE_TO_VBD = {
    "TRAFFIC_LIGHT_UNKNOWN": 0,
    "TRAFFIC_LIGHT_GREEN": 1,
    "TRAFFIC_LIGHT_YELLOW": 2,
    "TRAFFIC_LIGHT_RED": 3,
}


def _resolve_scenarionet_map_type(scenario, map_id, data):
    """Resolve one ScenarioNet map feature to the VBD training semantic id.

    TJ-West conversion used by AutoScene contains explicit ``type=None`` polylines
    for road-boundary geometry.  They are not generic unknown map elements:
    the converted feature keeps zero speed limits, has polyline geometry, and has
    no polygon surface.  Accept only that observed converter signature and map it
    to ROAD_EDGE_BOUNDARY.  Any other missing/unknown type remains a hard error.
    """
    if "type" not in data:
        raise KeyError(f"map feature {map_id} is missing required type field")

    raw_type = data["type"]
    if raw_type is not None:
        map_type = str(raw_type)
        if map_type not in SCENARIONET_MAP_TYPE_TO_VBD:
            raise ValueError(
                f"map feature {map_id} has unsupported type={map_type!r}; "
                "add an explicit training-consistent mapping before using VBD"
            )
        return map_type

    metadata = scenario.get("metadata", {}) if isinstance(scenario, dict) else {}
    dataset_name = str(metadata.get("dataset", "")).strip().lower()
    if dataset_name != "tjwest":
        raise ValueError(
            f"map feature {map_id} has type=None for dataset={dataset_name!r}; "
            "only the verified TJ-West untyped-boundary converter signature is supported"
        )

    speed_mph = data.get("speed_limit_mph", None)
    speed_kph = data.get("speed_limit_kph", None)
    try:
        speed_mph = float(speed_mph)
        speed_kph = float(speed_kph)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"TJ-West map feature {map_id} type=None has invalid speed limits "
            f"mph={speed_mph!r}, kph={speed_kph!r}"
        ) from exc
    if not np.isfinite([speed_mph, speed_kph]).all() or abs(speed_mph) > 1e-6 or abs(speed_kph) > 1e-6:
        raise ValueError(
            f"TJ-West map feature {map_id} type=None does not match the verified "
            f"road-boundary signature: speed limits are mph={speed_mph}, kph={speed_kph}"
        )

    polyline = np.asarray(data.get("polyline", []), dtype=np.float64)
    if polyline.ndim != 2 or len(polyline) < 2 or polyline.shape[1] < 2:
        raise ValueError(
            f"TJ-West map feature {map_id} type=None does not contain the required boundary polyline"
        )
    polygon = np.asarray(data.get("polygon", []))
    if polygon.size != 0:
        raise ValueError(
            f"TJ-West map feature {map_id} type=None unexpectedly contains polygon geometry; "
            "do not infer its VBD semantic type"
        )

    return "ROAD_EDGE_BOUNDARY"


def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _as_numeric_array(value, name, min_ndim=1):
    arr = np.asarray(value)
    if arr.ndim < min_ndim:
        raise ValueError(f"{name} must have ndim >= {min_ndim}, got shape={arr.shape}")
    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError(f"{name} must be numeric, got dtype={arr.dtype}")
    return arr


def _require_track_state(track_id, track):
    if not isinstance(track, dict):
        raise TypeError(f"track {track_id} must be dict")
    state = track.get("state")
    if not isinstance(state, dict):
        raise KeyError(f"track {track_id} has no state dict")
    required = ("position", "heading", "velocity", "length", "width", "height", "valid")
    missing = [key for key in required if key not in state]
    if missing:
        raise KeyError(f"track {track_id} missing state fields: {missing}")

    position = _as_numeric_array(state["position"], f"track[{track_id}].position", 2)
    heading = _as_numeric_array(state["heading"], f"track[{track_id}].heading", 1).reshape(-1)
    velocity = _as_numeric_array(state["velocity"], f"track[{track_id}].velocity", 2)
    length = _as_numeric_array(state["length"], f"track[{track_id}].length", 1).reshape(-1)
    width = _as_numeric_array(state["width"], f"track[{track_id}].width", 1).reshape(-1)
    height = _as_numeric_array(state["height"], f"track[{track_id}].height", 1).reshape(-1)
    valid = np.asarray(state["valid"]).reshape(-1).astype(bool)

    frame_count = len(heading)
    arrays = {
        "position": position,
        "heading": heading,
        "velocity": velocity,
        "length": length,
        "width": width,
        "height": height,
        "valid": valid,
    }
    for key, arr in arrays.items():
        if len(arr) != frame_count:
            raise ValueError(
                f"track {track_id} field {key} length={len(arr)} != heading length={frame_count}"
            )
    if position.shape[1] < 2:
        raise ValueError(f"track {track_id} position must contain x/y")
    if velocity.shape[1] < 2:
        raise ValueError(f"track {track_id} velocity must contain vx/vy")
    return arrays


def _scenario_id(scenario):
    metadata = scenario.get("metadata", {}) if isinstance(scenario, dict) else {}
    return str(
        scenario.get("id", metadata.get("scenario_id", metadata.get("id", "unknown")))
    )


def _normalize_required_ids(required_agent_ids):
    if required_agent_ids is None:
        return []
    out = []
    seen = set()
    for agent_id in required_agent_ids:
        sid = str(agent_id)
        if sid not in seen:
            out.append(sid)
            seen.add(sid)
    return out


def _select_agent_ids(
    scenario,
    current_index,
    max_num_objects,
    selected_agents=None,
    required_agent_ids=None,
):
    tracks = scenario.get("tracks")
    if not isinstance(tracks, dict) or not tracks:
        raise ValueError("scenario.tracks must be a non-empty dict")
    metadata = scenario.get("metadata", {})
    sdc_id = str(metadata.get("sdc_id", ""))
    if not sdc_id or sdc_id not in tracks:
        raise KeyError(f"scenario metadata.sdc_id={sdc_id!r} is not present in tracks")

    track_state_cache = {}
    eligible_ids = []
    for track_id, track in tracks.items():
        sid = str(track_id)
        track_type = str(track.get("type", track.get("metadata", {}).get("type", "")))
        if track_type not in SUPPORTED_AGENT_TYPES:
            continue
        state = _require_track_state(sid, track)
        track_state_cache[sid] = state
        if current_index >= len(state["valid"]):
            continue
        if not bool(state["valid"][current_index]):
            continue
        eligible_ids.append(sid)

    if sdc_id not in eligible_ids:
        raise RuntimeError(
            f"SDC {sdc_id} is not a valid supported VBD agent at current_index={current_index}"
        )

    required = _normalize_required_ids(required_agent_ids)
    if sdc_id not in required:
        required.insert(0, sdc_id)
    if len(required) > int(max_num_objects):
        raise ValueError(
            f"required VBD agents={len(required)} exceed max_num_objects={max_num_objects}"
        )
    for sid in required:
        if sid not in tracks:
            raise KeyError(f"required VBD agent {sid} is absent from scenario tracks")
        if sid not in eligible_ids:
            raise RuntimeError(
                f"required VBD agent {sid} is not valid/supported at current_index={current_index}"
            )

    if selected_agents is not None:
        ordered = [str(agent_id) for agent_id in selected_agents]
        if len(ordered) > int(max_num_objects):
            raise ValueError("selected_agents exceeds max_num_objects")
        if len(set(ordered)) != len(ordered):
            raise ValueError("selected_agents contains duplicate ids")
        for sid in ordered:
            if sid not in eligible_ids:
                raise RuntimeError(
                    f"selected VBD agent {sid} is not valid/supported at current_index={current_index}"
                )
        for sid in required:
            if sid not in ordered:
                raise RuntimeError(f"required VBD agent {sid} is missing from selected_agents")
        return ordered, track_state_cache, sdc_id

    sdc_state = track_state_cache[sdc_id]
    sdc_position = np.asarray(sdc_state["position"][current_index, :2], dtype=np.float64)
    if not np.isfinite(sdc_position).all():
        raise RuntimeError(f"SDC {sdc_id} position is non-finite at current_index={current_index}")

    distance_entries = []
    for order, sid in enumerate(eligible_ids):
        state = track_state_cache[sid]
        pos = np.asarray(state["position"][current_index, :2], dtype=np.float64)
        if not np.isfinite(pos).all():
            raise RuntimeError(f"agent {sid} position is non-finite at current_index={current_index}")
        distance_entries.append((float(np.linalg.norm(pos - sdc_position)), order, sid))
    distance_entries.sort(key=lambda item: (item[0], item[1]))

    chosen = [sid for _, _, sid in distance_entries[: int(max_num_objects)]]
    # Required agents must be part of the same joint diffusion sample. If one falls
    # outside the nearest set, replace the farthest non-required slot explicitly.
    required_set = set(required)
    for sid in required:
        if sid in chosen:
            continue
        replace_idx = None
        for idx in range(len(chosen) - 1, -1, -1):
            if chosen[idx] not in required_set:
                replace_idx = idx
                break
        if replace_idx is None:
            raise RuntimeError(f"cannot place required VBD agent {sid} into joint sample")
        chosen[replace_idx] = sid

    distance_by_id = {sid: (dist, order) for dist, order, sid in distance_entries}
    chosen.sort(key=lambda sid: distance_by_id[sid])
    if chosen[0] != sdc_id:
        # SDC distance is exactly zero; this indicates malformed duplicated positions/order.
        sdc_idx = chosen.index(sdc_id)
        chosen[0], chosen[sdc_idx] = chosen[sdc_idx], chosen[0]

    return chosen, track_state_cache, sdc_id


def data_process_agent(
    scenario,
    max_num_objects=64,
    current_index=10,
    use_log=True,
    selected_agents=None,
    remove_history=False,
    history_length=11,
    future_steps=80,
    required_agent_ids=None,
):
    """Convert ScenarioNet tracks into the VBD agent tensors.

    The VBD online input is a sliding ``history_length`` window ending at
    ``current_index``. For the current integration history_length must be 11
    (10 historical states + current state). ``agents_future`` is retained only
    for model/checkpoint compatibility and is never used as an inference input.
    """
    if not use_log:
        raise ValueError("AutoScene VBD online inference only supports ScenarioNet tracks (use_log=True)")
    current_index = int(current_index)
    history_length = int(history_length)
    future_steps = int(future_steps)
    max_num_objects = int(max_num_objects)
    if history_length <= 0:
        raise ValueError("history_length must be > 0")
    if future_steps <= 0:
        raise ValueError("future_steps must be > 0")
    history_start = current_index - history_length + 1
    if history_start < 0:
        raise RuntimeError(
            f"VBD needs {history_length} states ending at frame {current_index}; "
            f"history would start at {history_start}"
        )

    agent_ids, state_cache, sdc_id = _select_agent_ids(
        scenario=scenario,
        current_index=current_index,
        max_num_objects=max_num_objects,
        selected_agents=selected_agents,
        required_agent_ids=required_agent_ids,
    )

    agents_history = np.zeros((max_num_objects, history_length, 8), dtype=np.float32)
    agents_future = np.zeros((max_num_objects, future_steps + 1, 5), dtype=np.float32)
    agents_interested = np.zeros((max_num_objects,), dtype=np.int32)
    agents_type = np.zeros((max_num_objects,), dtype=np.int32)

    tracks = scenario["tracks"]
    for slot, sid in enumerate(agent_ids):
        track = tracks[sid] if sid in tracks else next(
            track for key, track in tracks.items() if str(key) == sid
        )
        state = state_cache.get(sid) or _require_track_state(sid, track)
        track_type = str(track.get("type", track.get("metadata", {}).get("type", "")))
        if track_type not in SUPPORTED_AGENT_TYPES:
            raise RuntimeError(f"selected VBD agent {sid} has unsupported type={track_type}")

        history_slice = slice(history_start, current_index + 1)
        if current_index >= len(state["valid"]):
            raise RuntimeError(f"selected VBD agent {sid} has no frame {current_index}")
        history_valid = np.asarray(state["valid"][history_slice], dtype=bool)
        if len(history_valid) != history_length:
            raise RuntimeError(
                f"selected VBD agent {sid} history length={len(history_valid)} != {history_length}"
            )

        hist = np.column_stack([
            state["position"][history_slice, 0],
            state["position"][history_slice, 1],
            state["heading"][history_slice],
            state["velocity"][history_slice, 0],
            state["velocity"][history_slice, 1],
            state["length"][history_slice],
            state["width"][history_slice],
            state["height"][history_slice],
        ]).astype(np.float32, copy=False)
        if not np.isfinite(hist[history_valid]).all():
            raise RuntimeError(f"selected VBD agent {sid} contains non-finite valid history")
        hist[~history_valid] = 0.0
        if remove_history:
            hist[:-1] = 0.0
        agents_history[slot] = hist

        # VBD training utilities express future GT as current + N future states.
        # Keep that exact shape, but do not fabricate unavailable dataset future.
        future_end = min(current_index + future_steps + 1, len(state["valid"]))
        available = max(future_end - current_index, 0)
        if available > 0:
            future_slice = slice(current_index, future_end)
            fut = np.column_stack([
                state["position"][future_slice, 0],
                state["position"][future_slice, 1],
                state["heading"][future_slice],
                state["velocity"][future_slice, 0],
                state["velocity"][future_slice, 1],
            ]).astype(np.float32, copy=False)
            future_valid = np.asarray(state["valid"][future_slice], dtype=bool)
            if not np.isfinite(fut[future_valid]).all():
                raise RuntimeError(f"selected VBD agent {sid} contains non-finite valid future")
            fut[~future_valid] = 0.0
            agents_future[slot, :available] = fut

        agents_interested[slot] = 1
        agents_type[slot] = SUPPORTED_AGENT_TYPES[track_type]

    sdc_slot = agent_ids.index(sdc_id)
    sdc_position = agents_history[sdc_slot, -1, :2].astype(np.float64)
    return (
        agents_history,
        agents_future,
        agents_interested,
        agents_type,
        list(agent_ids),
        sdc_position,
        history_start,
    )


def data_process_traffic_light(scenario, current_index=10, max_traffic_lights=16):
    traffic_lights = scenario.get("dynamic_map_states", {})
    max_traffic_lights = int(max_traffic_lights)
    points = np.zeros((max_traffic_lights, 3), dtype=np.float32)
    if not traffic_lights:
        return points, np.asarray([], dtype=object), np.asarray([], dtype=np.int32)

    lane_ids = []
    states = []
    stop_points = []
    for light_id, data in traffic_lights.items():
        if not isinstance(data, dict):
            raise TypeError(f"dynamic_map_states[{light_id}] must be dict")
        object_states = data.get("state", {}).get("object_state")
        if object_states is None or current_index >= len(object_states):
            raise RuntimeError(
                f"traffic light {light_id} has no state at current_index={current_index}"
            )
        stop_point = np.asarray(data.get("stop_point"), dtype=np.float64).reshape(-1)
        if len(stop_point) < 2 or not np.isfinite(stop_point[:2]).all():
            raise RuntimeError(f"traffic light {light_id} has invalid stop_point")
        raw_state = object_states[current_index]
        simplified = MetaDriveType.simplify_light_status(raw_state)
        if simplified not in TRAFFIC_LIGHT_STATE_TO_VBD:
            raise ValueError(
                f"traffic light {light_id} state {raw_state!r} simplified to unsupported {simplified!r}"
            )
        lane_ids.append(str(data.get("lane") or light_id))
        states.append(TRAFFIC_LIGHT_STATE_TO_VBD[simplified])
        stop_points.append(stop_point[:2])

    if len(stop_points) > max_traffic_lights:
        raise RuntimeError(
            f"scenario has {len(stop_points)} traffic lights but VBD max_traffic_lights={max_traffic_lights}"
        )
    if stop_points:
        stop_points_arr = np.asarray(stop_points, dtype=np.float32)
        states_arr = np.asarray(states, dtype=np.int32)
        points[: len(stop_points)] = np.concatenate(
            [stop_points_arr, states_arr[:, None].astype(np.float32)], axis=1
        )
    return points, np.asarray(lane_ids, dtype=object), np.asarray(states, dtype=np.int32)


def _polyline_heading(polyline_xy):
    xy = np.asarray(polyline_xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] < 2 or len(xy) < 2:
        raise ValueError(f"polyline must have at least two xy points, got shape={xy.shape}")
    delta = np.diff(xy[:, :2], axis=0)
    norm = np.linalg.norm(delta, axis=1)
    valid_seg = norm > 1e-9
    if not valid_seg.any():
        raise ValueError("polyline contains no non-zero segment")
    seg_heading = np.full((len(delta),), np.nan, dtype=np.float64)
    seg_heading[valid_seg] = np.arctan2(delta[valid_seg, 1], delta[valid_seg, 0])
    # Duplicate consecutive points have no intrinsic tangent. Propagate the nearest
    # actually observed segment heading instead of inventing a small epsilon vector.
    last = np.nan
    for i in range(len(seg_heading)):
        if np.isfinite(seg_heading[i]):
            last = seg_heading[i]
        elif np.isfinite(last):
            seg_heading[i] = last
    last = np.nan
    for i in range(len(seg_heading) - 1, -1, -1):
        if np.isfinite(seg_heading[i]):
            last = seg_heading[i]
        elif np.isfinite(last):
            seg_heading[i] = last
    if not np.isfinite(seg_heading).all():
        raise ValueError("polyline tangent cannot be determined")
    return np.concatenate([seg_heading, seg_heading[-1:]])


def calculate_relations(agents, polylines, traffic_lights):
    n_agents = agents.shape[0]
    n_polylines = polylines.shape[0]
    n_traffic_lights = traffic_lights.shape[0]
    n = n_agents + n_polylines + n_traffic_lights

    all_elements = np.concatenate([
        agents[:, -1, :3],
        polylines[:, 0, :3],
        np.concatenate(
            [traffic_lights[:, :2], np.zeros((n_traffic_lights, 1), dtype=traffic_lights.dtype)],
            axis=1,
        ),
    ], axis=0)

    pos_diff = all_elements[:, :2][:, None, :] - all_elements[:, :2][None, :, :]
    cos_theta = np.cos(all_elements[:, 2])[:, None]
    sin_theta = np.sin(all_elements[:, 2])[:, None]
    local_pos_x = pos_diff[..., 0] * cos_theta + pos_diff[..., 1] * sin_theta
    local_pos_y = -pos_diff[..., 0] * sin_theta + pos_diff[..., 1] * cos_theta
    theta_diff = wrap_to_pi(all_elements[:, 2][:, None] - all_elements[:, 2][None, :])

    start_idx = n_agents + n_polylines
    theta_diff = np.where(
        (np.arange(n) >= start_idx)[:, None] | (np.arange(n) >= start_idx)[None, :],
        0,
        theta_diff,
    )

    diag_mask = np.eye(n, dtype=bool)
    epsilon = 0.01
    local_pos_x = np.where(diag_mask, epsilon, local_pos_x)
    local_pos_y = np.where(diag_mask, epsilon, local_pos_y)
    theta_diff = np.where(diag_mask, epsilon, theta_diff)

    # Preserve the VBD training-time zero-sentinel convention for padded elements.
    zero_mask = np.logical_or(
        all_elements[:, 0][:, None] == 0,
        all_elements[:, 0][None, :] == 0,
    )
    relations = np.stack([local_pos_x, local_pos_y, theta_diff], axis=-1)
    relations = np.where(zero_mask[..., None], 0.0, relations)
    return relations.astype(np.float32, copy=False)


def data_process_scenario(
    scenario,
    max_num_objects=64,
    max_polylines=256,
    current_index=10,
    num_points_polyline=30,
    use_log=True,
    selected_agents=None,
    remove_history=False,
    history_length=11,
    future_steps=80,
    required_agent_ids=None,
    max_traffic_lights=16,
    map_radius_m=150.0,
):
    if not isinstance(scenario, dict):
        raise TypeError("AutoScene VBD adapter expects ScenarioNet scenario dict")
    if "map_features" not in scenario or not isinstance(scenario["map_features"], dict):
        raise KeyError("scenario.map_features must be a dict")

    (
        agents_history,
        agents_future,
        agents_interested,
        agents_type,
        agent_tokens,
        sdc_position,
        history_start,
    ) = data_process_agent(
        scenario,
        max_num_objects=max_num_objects,
        current_index=current_index,
        use_log=use_log,
        selected_agents=selected_agents,
        remove_history=remove_history,
        history_length=history_length,
        future_steps=future_steps,
        required_agent_ids=required_agent_ids,
    )

    traffic_light_points, traffic_lane_ids, traffic_light_states = data_process_traffic_light(
        scenario,
        current_index=current_index,
        max_traffic_lights=max_traffic_lights,
    )
    traffic_by_lane = {
        str(lane): int(state)
        for lane, state in zip(traffic_lane_ids.tolist(), traffic_light_states.tolist())
    }

    map_radius_m = float(map_radius_m)
    if map_radius_m <= 0:
        raise ValueError("map_radius_m must be > 0")
    max_polylines = int(max_polylines)
    num_points_polyline = int(num_points_polyline)

    polyline_entries = []
    for map_order, (map_id, data) in enumerate(scenario["map_features"].items()):
        if not isinstance(data, dict):
            raise TypeError(f"map_features[{map_id}] must be dict")

        raw = np.asarray(data.get("polyline", []), dtype=np.float64)
        if raw.ndim != 2 or len(raw) < 2 or raw.shape[1] < 2:
            raw = np.asarray(data.get("polygon", []), dtype=np.float64)
        if raw.ndim != 2 or len(raw) < 2 or raw.shape[1] < 2:
            # Geometry-less semantic entries cannot be represented by VBD's polyline encoder.
            continue
        if not np.isfinite(raw[:, :2]).all():
            raise RuntimeError(f"map feature {map_id} contains non-finite geometry")

        map_type = _resolve_scenarionet_map_type(scenario, map_id, data)

        min_distance = float(np.min(np.linalg.norm(raw[:, :2] - sdc_position[None, :], axis=1)))
        if min_distance > map_radius_m:
            continue

        if raw.shape[1] >= 3 and np.isfinite(raw[:, 2]).all():
            heading = wrap_to_pi(raw[:, 2])
        else:
            heading = _polyline_heading(raw[:, :2])

        lane_type = int(SCENARIONET_MAP_TYPE_TO_VBD[map_type])
        traffic_state = int(traffic_by_lane.get(str(map_id), 0))
        polyline = np.stack([
            raw[:, 0],
            raw[:, 1],
            heading,
            np.full((len(raw),), traffic_state, dtype=np.float64),
            np.full((len(raw),), lane_type, dtype=np.float64),
        ], axis=1)

        sampled = np.linspace(0, len(polyline) - 1, num_points_polyline, dtype=np.int32)
        cur_polyline = polyline[sampled].astype(np.float32, copy=False)
        polyline_entries.append((min_distance, map_order, cur_polyline))

    polyline_entries.sort(key=lambda item: (item[0], item[1]))
    polyline_entries = polyline_entries[:max_polylines]
    valid_count = len(polyline_entries)

    polylines = np.zeros((max_polylines, num_points_polyline, 5), dtype=np.float32)
    polylines_valid = np.zeros((max_polylines,), dtype=np.int32)
    for idx, (_, _, polyline) in enumerate(polyline_entries):
        polylines[idx] = polyline
        polylines_valid[idx] = 1

    relations = calculate_relations(agents_history, polylines, traffic_light_points)
    data_dict = {
        "agents_history": np.asarray(agents_history, dtype=np.float32),
        "agents_interested": np.asarray(agents_interested, dtype=np.int32),
        "agents_type": np.asarray(agents_type, dtype=np.int32),
        "agents_future": np.asarray(agents_future, dtype=np.float32),
        "traffic_light_points": np.asarray(traffic_light_points, dtype=np.float32),
        "polylines": np.asarray(polylines, dtype=np.float32),
        "polylines_valid": np.asarray(polylines_valid, dtype=np.int32),
        "relations": np.asarray(relations, dtype=np.float32),
        "agent_tokens": list(agent_tokens),
        "scenario_id": _scenario_id(scenario),
        "history_start_index": int(history_start),
        "history_end_index": int(current_index),
    }
    return data_dict


def data_collate_fn(batch_list):
    if not batch_list:
        raise ValueError("batch_list must be non-empty")
    keys = batch_list[0].keys()
    input_batch = {}
    for key in keys:
        values = [item[key] for item in batch_list]
        first = values[0]
        if isinstance(first, np.ndarray):
            input_batch[key] = torch.from_numpy(np.stack(values, axis=0))
        else:
            # Metadata (agent tokens, scenario id, history indices) must stay out of
            # the tensor model input while remaining available to the inference adapter.
            input_batch[key] = values
    return input_batch