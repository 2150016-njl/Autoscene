import math
import os

import numpy as np


def _wrap_to_pi(angle):
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def _track_state(current_state, timestep, agent_id, rear_axle_reference=False):
    tracks = current_state.get("tracks", {}) if isinstance(current_state, dict) else {}
    if agent_id is None:
        metadata = current_state.get("metadata", {}) if isinstance(current_state, dict) else {}
        agent_id = metadata.get("sdc_id")
    track = tracks.get(str(agent_id))
    if track is None:
        track = tracks.get(agent_id)
    if not isinstance(track, dict):
        return None

    state = track.get("state", {})
    position = np.asarray(state.get("position", []), dtype=np.float64)
    if len(position) == 0:
        return None
    idx = int(np.clip(timestep, 0, len(position) - 1))
    pos = np.asarray(position[idx], dtype=np.float64).reshape(-1)
    if pos.shape[0] < 2 or not np.isfinite(pos[:2]).all():
        return None

    heading_arr = np.asarray(state.get("heading", []), dtype=np.float64).reshape(-1)
    heading = float(heading_arr[idx]) if len(heading_arr) > idx else 0.0
    velocity = np.asarray(state.get("velocity", []), dtype=np.float64)
    speed = 0.0
    if len(velocity) > idx:
        vel = np.asarray(velocity[idx], dtype=np.float64).reshape(-1)
        if vel.shape[0] >= 2 and np.isfinite(vel[:2]).all():
            speed = float(np.linalg.norm(vel[:2]))

    xy = np.asarray(pos[:2], dtype=np.float64).copy()
    if rear_axle_reference:
        # Current PlutoDataset shifts the controlled vehicle from ScenarioNet/
        # MetaDrive center coordinates to rear-axle coordinates before model
        # inference. OSM candidates must use the same reference or the shared
        # downstream Pluto conversion would shift them a second time.
        metadata = current_state.get("metadata", {}) if isinstance(current_state, dict) else {}
        sdc_id = str(metadata.get("sdc_id", ""))
        resolved_id = str(agent_id)
        distance = 0.0
        if resolved_id == sdc_id:
            params = metadata.get("ego_vehicle_parameters", {}) or {}
            for key in ("rear_axle_to_center", "rear_axle_to_center_dist", "cog_position_from_rear_axle"):
                if key in params:
                    distance = float(params[key])
                    break
            else:
                distance = 1.67
        else:
            track_type = str(track.get("type", track.get("metadata", {}).get("type", ""))).upper()
            if track_type in {"VEHICLE", "CAR", "TRUCK", "BUS"}:
                lengths = np.asarray(state.get("length", []), dtype=np.float64).reshape(-1)
                if len(lengths) > 0:
                    length_idx = min(idx, len(lengths) - 1)
                    if np.isfinite(lengths[length_idx]):
                        distance = 0.35 * float(lengths[length_idx])
        if distance > 0.0:
            xy -= distance * np.asarray([math.cos(heading), math.sin(heading)], dtype=np.float64)

    return xy, heading, speed


def _current_road_network():
    try:
        from metadrive.engine.engine_utils import get_engine

        engine = get_engine()
    except Exception:
        return None
    current_map = getattr(engine, "current_map", None)
    if current_map is None and hasattr(engine, "map_manager"):
        current_map = getattr(engine.map_manager, "current_map", None)
    return getattr(current_map, "road_network", None)


def _current_engine():
    try:
        from metadrive.engine.engine_utils import get_engine

        return get_engine()
    except Exception:
        return None


def _resolve_lane(network, lane_id):
    if network is None or lane_id is None:
        return None
    try:
        return network.get_lane(lane_id)
    except Exception:
        return None


def _lane_id_from_engine_object(network, agent_id):
    engine = _current_engine()
    if engine is None:
        return None

    objects = {}
    try:
        objects.update(engine.get_objects())
    except Exception:
        pass
    objects.update(getattr(engine, "agents", {}) or {})

    candidate_obj_ids = []
    if agent_id is None:
        candidate_obj_ids.extend(["default_agent", "ego"])
    else:
        agent_id = str(agent_id)
        candidate_obj_ids.append(agent_id)
        traffic_manager = getattr(engine, "traffic_manager", None)
        mapping = getattr(traffic_manager, "obj_id_to_scenario_id", {}) if traffic_manager is not None else {}
        for obj_id, scenario_id in mapping.items():
            if str(scenario_id) == agent_id:
                candidate_obj_ids.append(str(obj_id))

    for obj_id in candidate_obj_ids:
        obj = objects.get(obj_id)
        if obj is None:
            continue
        lane_index = getattr(obj, "lane_index", None)
        if lane_index is None:
            lane = getattr(obj, "lane", None)
            lane_index = getattr(lane, "index", None)
        if lane_index is not None and _resolve_lane(network, lane_index) is not None:
            return lane_index
        navigation = getattr(obj, "navigation", None)
        lane = getattr(navigation, "_current_lane", None)
        lane_index = getattr(lane, "index", None)
        if lane_index is not None and _resolve_lane(network, lane_index) is not None:
            return lane_index
    return None


def _lane_id_from_reference_lines(network, pluto_feature, position, heading):
    ref = pluto_feature.get("reference_line", None) if isinstance(pluto_feature, dict) else None
    if not isinstance(ref, dict) or "route_id" not in ref:
        return None
    route_ids = np.asarray(ref.get("route_id", []))
    best = None
    for route_id in route_ids.reshape(-1):
        try:
            if int(route_id) < 0:
                continue
            lane_id = str(int(route_id))
        except Exception:
            lane_id = str(route_id)
        lane = _resolve_lane(network, lane_id)
        if lane is None:
            continue
        try:
            longitudinal, lateral = lane.local_coordinates(position)
            lane_length = float(getattr(lane, "length", max(float(longitudinal), 1.0)))
            lane_heading = _lane_heading(lane, longitudinal=np.clip(float(longitudinal), 0.0, lane_length))
            score = abs(float(lateral)) + 2.0 * abs(_wrap_to_pi(lane_heading - heading))
        except Exception:
            continue
        if best is None or score < best[0]:
            best = (score, lane_id)
    return None if best is None else best[1]


def _closest_lane_id(network, position, heading):
    if network is None:
        return None
    if hasattr(network, "get_closest_lane_index"):
        for kwargs in (
            {
                "position": position,
                "heading": heading,
                "ego_vehicle": None,
                "use_heading_filter": True,
            },
            {
                "position": position,
                "heading": heading,
                "ego_vehicle": None,
                "use_heading_filter": False,
            },
        ):
            try:
                lane_id, _ = network.get_closest_lane_index(**kwargs)
                if lane_id is not None:
                    return lane_id
            except TypeError:
                break
            except Exception:
                pass
        try:
            lane_id, _ = network.get_closest_lane_index(position)
            if lane_id is not None:
                return lane_id
        except Exception:
            pass
    return _closest_lane_id_from_graph(network, position, heading)


def _point_in_polygon(point, polygon):
    polygon = np.asarray(polygon, dtype=np.float64)
    if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] < 2:
        return False
    x, y = float(point[0]), float(point[1])
    inside = False
    xj, yj = polygon[-1, 0], polygon[-1, 1]
    for xi, yi in polygon[:, :2]:
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi
        )
        if intersects:
            inside = not inside
        xj, yj = xi, yi
    return inside


def _closest_lane_id_from_graph(network, position, heading):
    graph = getattr(network, "graph", {}) if network is not None else {}
    if not graph:
        return None

    best = None
    for lane_id, info in graph.items():
        lane = getattr(info, "lane", info)
        if lane is None:
            continue
        try:
            longitudinal, lateral = lane.local_coordinates(position)
            longitudinal = float(longitudinal)
            lateral = float(lateral)
        except Exception:
            continue
        lane_length = float(getattr(lane, "length", max(longitudinal, 1.0)))
        polygon = getattr(lane, "polygon", None)
        in_polygon = _point_in_polygon(position, polygon) if polygon is not None else False
        on_longitudinal = -2.0 <= longitudinal <= lane_length + 2.0
        lane_heading = _lane_heading(lane, longitudinal=np.clip(longitudinal, 0.0, lane_length))
        score = abs(lateral) + 2.0 * abs(_wrap_to_pi(lane_heading - heading))
        if not (in_polygon or on_longitudinal):
            score += 20.0
        if in_polygon:
            score -= 10.0
        if best is None or score < best[0]:
            best = (score, lane_id)
    return None if best is None else best[1]


def _graph_info(network, lane_id):
    graph = getattr(network, "graph", {}) if network is not None else {}
    key = lane_id
    if hasattr(network, "_resolve_graph_key"):
        try:
            key = network._resolve_graph_key(lane_id)
        except Exception:
            key = lane_id
    return graph.get(key), key


def _lane_heading(lane, position=None, longitudinal=None):
    try:
        if longitudinal is None and position is not None:
            longitudinal, _ = lane.local_coordinates(position)
        return float(lane.heading_theta_at(float(longitudinal or 0.0)))
    except Exception:
        return float(getattr(lane, "heading_theta", 0.0))


def _exit_turn_delta(lane, heading):
    lane_length = float(getattr(lane, "length", 1.0))
    probe_s = min(lane_length, max(8.0, 0.75 * lane_length))
    end_s = max(lane_length - 0.5, 0.0)
    probe_delta = _wrap_to_pi(_lane_heading(lane, longitudinal=probe_s) - heading)
    end_delta = _wrap_to_pi(_lane_heading(lane, longitudinal=end_s) - heading)
    return end_delta if abs(end_delta) >= abs(probe_delta) else probe_delta


def _exit_candidates(network, lane_id, heading):
    info, _ = _graph_info(network, lane_id)
    exits = list(getattr(info, "exit_lanes", []) or []) if info is not None else []
    scored = []
    for exit_id in exits:
        lane = _resolve_lane(network, exit_id)
        if lane is None:
            continue
        delta = _exit_turn_delta(lane, heading)
        scored.append((abs(delta), delta, exit_id))
    return scored


def _choose_exit(network, lane_id, heading, mode="straight", turn_threshold=0.35):
    scored = _exit_candidates(network, lane_id, heading)
    if not scored:
        return None

    if mode == "straight":
        candidates = [item for item in scored if item[0] <= turn_threshold]
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        return candidates[0][2]

    if mode == "left":
        candidates = [item for item in scored if item[1] > turn_threshold]
    elif mode == "right":
        candidates = [item for item in scored if item[1] < -turn_threshold]
    else:
        candidates = []
    if not candidates:
        return None
    candidates.sort(key=lambda x: -x[0])
    return candidates[0][2]


def _follow_straight_exits(network, first_lane_id, max_lanes=4, start_heading=0.0):
    path = [first_lane_id]
    lane_id = first_lane_id
    heading = start_heading
    for _ in range(max_lanes - 1):
        next_id = _choose_exit(network, lane_id, heading, mode="straight")
        if next_id is None or next_id in path:
            break
        path.append(next_id)
        lane = _resolve_lane(network, next_id)
        if lane is not None:
            heading = _lane_heading(lane, longitudinal=1.0)
        lane_id = next_id
    return path


def _exit_lane_ids(network, lane_id):
    info, _ = _graph_info(network, lane_id)
    exits = list(getattr(info, "exit_lanes", []) or []) if info is not None else []
    return [exit_id for exit_id in exits if _resolve_lane(network, exit_id) is not None]


def _reachable_by_lane_exits(network, start_lane_id, target_lane_id, max_lanes=6):
    if start_lane_id is None or target_lane_id is None:
        return False
    if str(start_lane_id) == str(target_lane_id):
        return True

    frontier = [(start_lane_id, 0)]
    visited = {str(start_lane_id)}
    while frontier:
        lane_id, depth = frontier.pop(0)
        if depth >= max(0, int(max_lanes) - 1):
            continue
        for next_id in _exit_lane_ids(network, lane_id):
            next_key = str(next_id)
            if next_key == str(target_lane_id):
                return True
            if next_key in visited:
                continue
            visited.add(next_key)
            frontier.append((next_id, depth + 1))
    return False


def _forward_lane_chain(network, start_lane_id, start_heading, max_lanes=6):
    if start_lane_id is None:
        return []
    path = [start_lane_id]
    lane_id = start_lane_id
    heading = float(start_heading)
    for _ in range(max(0, int(max_lanes) - 1)):
        next_id = _choose_exit(network, lane_id, heading, mode="straight")
        if next_id is None or str(next_id) in {str(item) for item in path}:
            break
        path.append(next_id)
        lane = _resolve_lane(network, next_id)
        if lane is not None:
            heading = _lane_heading(lane, longitudinal=1.0)
        lane_id = next_id
    return path


def _lane_anchor_point(lane, position):
    if lane is None:
        return None, None
    try:
        longitudinal, _ = lane.local_coordinates(position)
    except Exception:
        longitudinal = 0.0
    lane_length = float(getattr(lane, "length", max(float(longitudinal), 1.0)) or 1.0)
    query_s = float(np.clip(float(longitudinal), 0.0, lane_length))
    try:
        return np.asarray(lane.position(query_s, 0.0), dtype=np.float64)[:2], query_s
    except Exception:
        return None, query_s


def _forward_distance_from(position, heading, point):
    if point is None:
        return -float("inf")
    direction = np.asarray([math.cos(float(heading)), math.sin(float(heading))], dtype=np.float64)
    return float(np.dot(np.asarray(point, dtype=np.float64)[:2] - np.asarray(position, dtype=np.float64)[:2], direction))


def _select_forward_ego_chain_lane(
        network,
        ego_lane_id,
        ego_pos,
        ego_heading,
        agent_pos,
        agent_heading,
        cfg,
):
    max_lanes = int(cfg.get("osm_ego_lane_cutin_forward_chain_hops", cfg.get("osm_ego_lane_cutin_follow_lanes", 5)))
    min_ahead = float(cfg.get("osm_ego_lane_cutin_min_ahead_m", 1.0))
    chain = _forward_lane_chain(network, ego_lane_id, ego_heading, max_lanes=max_lanes)
    best = None
    lookahead_values = cfg.get("osm_ego_lane_cutin_forward_lookahead_m", [6.0, 10.0, 15.0, 20.0, 30.0])
    if isinstance(lookahead_values, str):
        lookahead_values = [item.strip() for item in lookahead_values.split(",") if item.strip()]
    try:
        lookahead_values = [float(item) for item in lookahead_values if float(item) > 0.0]
    except TypeError:
        lookahead_values = [float(lookahead_values)]
    if chain and lookahead_values:
        lookahead_xy, _ = _sample_lane_path(
            network,
            chain,
            ego_pos,
            np.asarray(lookahead_values, dtype=np.float32),
            start_l_override=0.0,
        )
        if lookahead_xy is not None:
            for lookahead_m, anchor in zip(lookahead_values, lookahead_xy):
                ahead = _forward_distance_from(agent_pos, agent_heading, anchor)
                if ahead < min_ahead:
                    continue
                lane_id = _closest_lane_id(network, anchor, ego_heading) or ego_lane_id
                try:
                    lane = _resolve_lane(network, lane_id)
                    anchor_s, _ = lane.local_coordinates(anchor) if lane is not None else (0.0, 0.0)
                except Exception:
                    anchor_s = 0.0
                score = ahead
                if best is None or score < best[0]:
                    best = (score, lane_id, ahead, 0, anchor_s, chain, f"ego_forward_lookahead_{lookahead_m:g}m")
    for depth, lane_id in enumerate(chain):
        lane = _resolve_lane(network, lane_id)
        if lane is None:
            continue
        anchors = []
        agent_anchor, agent_anchor_s = _lane_anchor_point(lane, agent_pos)
        if agent_anchor is not None:
            anchors.append((agent_anchor, agent_anchor_s, "agent_projection"))
        ego_anchor, ego_anchor_s = _lane_anchor_point(lane, ego_pos)
        if ego_anchor is not None:
            anchors.append((ego_anchor, ego_anchor_s, "ego_projection"))
        for anchor, anchor_s, source in anchors:
            ahead = _forward_distance_from(agent_pos, agent_heading, anchor)
            if ahead < min_ahead:
                continue
            score = ahead + depth * 0.25
            if best is None or score < best[0]:
                best = (score, lane_id, ahead, depth, anchor_s, chain, source)
    if best is None:
        return None, chain, None
    return best[1], best[5], {
        "ahead_m": float(best[2]),
        "depth": int(best[3]),
        "anchor_s": float(best[4]),
        "anchor_source": str(best[6]),
    }


def _same_topological_corridor(network, lane_a_id, lane_b_id, max_lanes=6):
    if lane_a_id is None or lane_b_id is None:
        return False
    if str(lane_a_id) == str(lane_b_id):
        return True
    return (
        _reachable_by_lane_exits(network, lane_a_id, lane_b_id, max_lanes=max_lanes)
        or _reachable_by_lane_exits(network, lane_b_id, lane_a_id, max_lanes=max_lanes)
    )


def _point_near_lane_corridor(lane, point, lateral_threshold=2.2, longitudinal_margin=25.0):
    if lane is None or point is None:
        return False
    try:
        longitudinal, lateral = lane.local_coordinates(point)
        lane_length = float(getattr(lane, "length", max(float(longitudinal), 1.0)))
    except Exception:
        return False
    return (
        abs(float(lateral)) <= float(lateral_threshold)
        and -float(longitudinal_margin) <= float(longitudinal) <= lane_length + float(longitudinal_margin)
    )


def _same_geometric_corridor(
        network,
        lane_a_id,
        lane_b_id,
        pos_a,
        pos_b,
        lateral_threshold=2.2,
        longitudinal_margin=25.0,
):
    if lane_a_id is None or lane_b_id is None:
        return False
    lane_a = _resolve_lane(network, lane_a_id)
    lane_b = _resolve_lane(network, lane_b_id)
    if lane_a is None or lane_b is None:
        return False
    return (
        _point_near_lane_corridor(
            lane_a,
            pos_b,
            lateral_threshold=lateral_threshold,
            longitudinal_margin=longitudinal_margin,
        )
        or _point_near_lane_corridor(
            lane_b,
            pos_a,
            lateral_threshold=lateral_threshold,
            longitudinal_margin=longitudinal_margin,
        )
    )


def _build_lane_modes(
        network,
        lane_id,
        heading,
):
    info, lane_key = _graph_info(network, lane_id)
    if info is None:
        return []
    modes = [
        ("straight", _follow_straight_exits(network, lane_key, start_heading=heading)),
    ]

    for label, turn_mode in (("left_turn", "left"), ("right_turn", "right")):
        exit_id = _choose_exit(network, lane_key, heading, mode=turn_mode)
        if exit_id is None:
            continue
        lane = _resolve_lane(network, exit_id)
        exit_heading = _lane_heading(lane, longitudinal=1.0) if lane is not None else heading
        tail = _follow_straight_exits(network, exit_id, start_heading=exit_heading)
        modes.append((label, [lane_key] + tail))

    modes.append(("brake", [lane_key]))

    unique = []
    seen = set()
    for label, path in modes:
        key = (str(label), tuple(str(x) for x in path)) if label == "brake" else tuple(str(x) for x in path)
        if path and key not in seen:
            unique.append((label, path))
            seen.add(key)
    return unique


def _sample_lane_path(network, path, start_position, distances, start_l_override=None):
    lanes = [_resolve_lane(network, lane_id) for lane_id in path]
    lanes = [lane for lane in lanes if lane is not None]
    if not lanes:
        return None, None
    try:
        start_s, start_l = lanes[0].local_coordinates(start_position)
    except Exception:
        start_s, start_l = 0.0, 0.0
    if start_l_override is not None:
        start_l = float(start_l_override)

    xy = []
    headings = []
    lane_idx = 0
    lane_s = float(start_s)
    last_xy = np.asarray(start_position, dtype=np.float64)[:2]
    last_heading = _lane_heading(lanes[0], longitudinal=lane_s)
    previous_distance = 0.0
    for distance in distances:
        delta_distance = float(distance - previous_distance)
        lane_s += delta_distance
        previous_distance = float(distance)
        while lane_idx < len(lanes) - 1 and lane_s > float(getattr(lanes[lane_idx], "length", 0.0)):
            lane_s -= float(getattr(lanes[lane_idx], "length", 0.0))
            lane_idx += 1
        lane = lanes[lane_idx]
        lane_length = float(getattr(lane, "length", max(lane_s, 1.0)))
        query_s = min(max(lane_s, 0.0), lane_length)
        try:
            point = np.asarray(lane.position(query_s, start_l), dtype=np.float64)[:2]
            hdg = _lane_heading(lane, longitudinal=query_s)
        except Exception:
            point = last_xy + np.array([math.cos(last_heading), math.sin(last_heading)]) * max(
                0.0, delta_distance
            )
            hdg = last_heading
        last_xy = point
        last_heading = hdg
        xy.append(point)
        headings.append(hdg)
    return np.asarray(xy, dtype=np.float32), np.asarray(headings, dtype=np.float32)


def _sample_cutin_to_lane(
        network,
        source_lane_id,
        target_lane_id,
        start_position,
        heading,
        distances,
        lane_change_distance,
        max_lanes=5,
        target_anchor_position=None,
):
    source_lane = _resolve_lane(network, source_lane_id)
    target_lane = _resolve_lane(network, target_lane_id)
    if source_lane is None or target_lane is None:
        return None, None
    target_anchor_position = start_position if target_anchor_position is None else target_anchor_position
    try:
        target_start_s, _ = target_lane.local_coordinates(target_anchor_position)
    except Exception:
        return None, None

    max_lanes = max(1, int(max_lanes))
    source_path = _follow_straight_exits(
        network,
        source_lane_id,
        max_lanes=max_lanes,
        start_heading=heading,
    )
    target_heading = _lane_heading(target_lane, longitudinal=max(float(target_start_s), 0.0))
    target_path = _follow_straight_exits(
        network,
        target_lane_id,
        max_lanes=max_lanes,
        start_heading=target_heading,
    )
    return _sample_cutin_between_paths(
        network,
        source_path,
        target_path,
        start_position,
        distances,
        lane_change_distance,
        target_start_position=target_anchor_position,
    )


def _sample_cutin_between_paths(
        network,
        source_path,
        target_path,
        start_position,
        distances,
        lane_change_distance,
        target_start_position=None,
):
    source_xy, _ = _sample_lane_path(network, source_path, start_position, distances)
    target_start_position = start_position if target_start_position is None else target_start_position
    target_xy, _ = _sample_lane_path(
        network,
        target_path,
        target_start_position,
        distances,
        start_l_override=0.0,
    )
    if source_xy is None or target_xy is None:
        return None, None

    lane_change_distance = max(float(lane_change_distance), 1e-3)
    progress = np.clip(np.asarray(distances, dtype=np.float32) / lane_change_distance, 0.0, 1.0)
    smooth = progress * progress * (3.0 - 2.0 * progress)
    xy = (1.0 - smooth[:, None]) * source_xy[:, :2] + smooth[:, None] * target_xy[:, :2]
    xy = xy.astype(np.float32)
    if len(xy) < 2:
        return None, None
    tangent = np.gradient(xy[:, :2], axis=0)
    headings = np.arctan2(tangent[:, 1], tangent[:, 0] + 1e-8).astype(np.float32)
    return xy, headings


def _cutin_distances(cfg):
    value = cfg.get("osm_adversarial_cutin_distance_m", [10.0, 14.0, 18.0])
    if isinstance(value, str):
        raw_values = [item.strip() for item in value.split(",") if item.strip()]
    else:
        try:
            raw_values = list(value)
        except TypeError:
            raw_values = [value]
    distances = [float(v) for v in raw_values if float(v) > 0.0]
    return distances or [14.0]


def _ego_lane_cutin_distances(cfg):
    value = cfg.get("osm_ego_lane_cutin_distance_m", [10.0, 14.0, 18.0])
    if isinstance(value, str):
        raw_values = [item.strip() for item in value.split(",") if item.strip()]
    else:
        try:
            raw_values = list(value)
        except TypeError:
            raw_values = [value]
    distances = [float(v) for v in raw_values if float(v) > 0.0]
    return distances or [14.0]


def _target_lane_cutin_modes(network, current_state, timestep, controlled_agent_id, target_agent_ids, position, heading, cfg):
    if not target_agent_ids:
        return []
    current_lane_id = _lane_id_from_engine_object(network, controlled_agent_id)
    current_lane_source = "engine"
    if current_lane_id is None:
        current_lane_id = _closest_lane_id(network, position, heading)
        current_lane_source = "closest"
    modes = []
    seen = set()
    max_heading_diff = float(cfg.get("osm_adversarial_cutin_heading_threshold", 0.8))
    for target_id in target_agent_ids:
        target_state = _track_state(current_state, timestep, str(target_id), rear_axle_reference=True)
        if target_state is None:
            continue
        target_pos, target_heading, _ = target_state
        target_lane_id = _lane_id_from_engine_object(network, str(target_id))
        if target_lane_id is None:
            target_lane_id = _closest_lane_id(network, target_pos, target_heading)
        heading_diff = abs(_wrap_to_pi(float(target_heading) - float(heading)))
        debug_enabled = os.environ.get("UNITRAJ_OSM_CUTIN_DEBUG", "0") not in ("0", "false", "False")
        same_lane = bool(
            current_lane_id is not None
            and target_lane_id is not None
            and str(target_lane_id) == str(current_lane_id)
        )
        same_corridor = _same_topological_corridor(
            network,
            current_lane_id,
            target_lane_id,
            max_lanes=int(cfg.get("osm_adversarial_same_corridor_hops", 6)),
        )
        same_geometric_corridor = _same_geometric_corridor(
            network,
            current_lane_id,
            target_lane_id,
            position,
            target_pos,
            lateral_threshold=float(cfg.get("osm_adversarial_same_corridor_lateral_m", 2.2)),
            longitudinal_margin=float(cfg.get("osm_adversarial_same_corridor_margin_m", 25.0)),
        )

        def _debug(decision):
            if not debug_enabled:
                return
            print(
                "[OSMCutinLaneDebug]",
                {
                    "t": int(timestep),
                    "agent": None if controlled_agent_id is None else str(controlled_agent_id),
                    "target": str(target_id),
                    "agent_lane": None if current_lane_id is None else str(current_lane_id),
                    "agent_lane_source": str(current_lane_source),
                    "target_lane": None if target_lane_id is None else str(target_lane_id),
                    "same_lane": bool(same_lane),
                    "same_corridor": bool(same_corridor),
                    "same_geometric_corridor": bool(same_geometric_corridor),
                    "heading_diff": round(float(heading_diff), 4),
                    "heading_threshold": round(float(max_heading_diff), 4),
                    "agent_xy": [round(float(x), 3) for x in np.asarray(position, dtype=np.float64)[:2]],
                    "target_xy": [round(float(x), 3) for x in np.asarray(target_pos, dtype=np.float64)[:2]],
                    "decision": str(decision),
                },
            )

        if target_lane_id is None:
            _debug("skip_no_target_lane")
            continue
        if str(target_lane_id) in seen:
            _debug("skip_seen_target_lane")
            continue
        if same_lane:
            _debug("skip_same_lane")
            continue
        if same_corridor:
            _debug("skip_same_corridor")
            continue
        if same_geometric_corridor:
            _debug("skip_same_geometric_corridor")
            continue
        if heading_diff > max_heading_diff:
            _debug("skip_heading")
            continue
        modes.append((f"target_cutin_{len(modes) + 1}", target_lane_id))
        seen.add(str(target_lane_id))
        _debug("add_target_cutin")
    return modes


def _ego_lane_cutin_modes(network, current_state, timestep, controlled_agent_id, position, heading, cfg):
    if not bool(cfg.get("osm_ego_lane_cutin_enabled", True)):
        return []
    agent_id = str(cfg.get("osm_ego_lane_cutin_agent_id", "adv_1"))
    if controlled_agent_id is None or str(controlled_agent_id) != agent_id:
        return []

    metadata = current_state.get("metadata", {}) if isinstance(current_state, dict) else {}
    ego_id = metadata.get("sdc_id", None)
    ego_state = _track_state(current_state, timestep, ego_id, rear_axle_reference=True)
    if ego_state is None:
        ego_state = _track_state(current_state, timestep, None, rear_axle_reference=True)
    if ego_state is None:
        return []
    ego_pos, ego_heading, _ = ego_state

    current_lane_id = _lane_id_from_engine_object(network, controlled_agent_id)
    current_lane_source = "engine"
    if current_lane_id is None:
        current_lane_id = _closest_lane_id(network, position, heading)
        current_lane_source = "closest"

    ego_lane_id = _lane_id_from_engine_object(network, None)
    ego_lane_source = "engine"
    if ego_lane_id is None:
        ego_lane_id = _closest_lane_id(network, ego_pos, ego_heading)
        ego_lane_source = "closest"

    forward_ego_lane_id = None
    ego_forward_chain = []
    forward_target_info = None
    if ego_lane_id is not None:
        selected_lane_id, ego_forward_chain, forward_target_info = _select_forward_ego_chain_lane(
            network,
            ego_lane_id,
            ego_pos,
            ego_heading,
            position,
            heading,
            cfg,
        )
        forward_ego_lane_id = selected_lane_id

    same_lane = bool(
        current_lane_id is not None
        and forward_ego_lane_id is not None
        and str(forward_ego_lane_id) == str(current_lane_id)
    )
    same_corridor = _same_topological_corridor(
        network,
        current_lane_id,
        forward_ego_lane_id,
        max_lanes=int(cfg.get("osm_ego_lane_cutin_same_corridor_hops", 6)),
    )
    same_geometric_corridor = _same_geometric_corridor(
        network,
        current_lane_id,
        forward_ego_lane_id,
        position,
        ego_pos,
        lateral_threshold=float(cfg.get("osm_ego_lane_cutin_same_corridor_lateral_m", 2.2)),
        longitudinal_margin=float(cfg.get("osm_ego_lane_cutin_same_corridor_margin_m", 25.0)),
    )
    heading_diff = abs(_wrap_to_pi(float(ego_heading) - float(heading)))
    debug_enabled = os.environ.get("UNITRAJ_OSM_EGO_CUTIN_DEBUG", "0") not in ("0", "false", "False")

    def _debug(decision):
        if not debug_enabled:
            return
        print(
            "[OSMEgoLaneCutinDebug]",
            {
                "t": int(timestep),
                "agent": None if controlled_agent_id is None else str(controlled_agent_id),
                "ego_id": None if ego_id is None else str(ego_id),
                "agent_lane": None if current_lane_id is None else str(current_lane_id),
                "agent_lane_source": str(current_lane_source),
                "ego_lane": None if ego_lane_id is None else str(ego_lane_id),
                "ego_lane_source": str(ego_lane_source),
                "ego_forward_chain": [str(item) for item in ego_forward_chain],
                "selected_forward_lane": None if forward_ego_lane_id is None else str(forward_ego_lane_id),
                "forward_target": forward_target_info,
                "same_lane": bool(same_lane),
                "same_corridor": bool(same_corridor),
                "same_geometric_corridor": bool(same_geometric_corridor),
                "heading_diff": round(float(heading_diff), 4),
                "agent_xy": [round(float(x), 3) for x in np.asarray(position, dtype=np.float64)[:2]],
                "ego_xy": [round(float(x), 3) for x in np.asarray(ego_pos, dtype=np.float64)[:2]],
                "decision": str(decision),
            },
        )

    if ego_lane_id is None:
        _debug("skip_no_ego_lane")
        return []
    if forward_ego_lane_id is None:
        _debug("skip_no_forward_ego_lane_ahead_of_agent")
        return []
    if same_lane:
        _debug("add_ego_lane_keep_same_lane")
        return [("ego_lane_keep", forward_ego_lane_id)]
    if same_corridor:
        _debug("add_ego_lane_keep_same_corridor")
        return [("ego_lane_keep", forward_ego_lane_id)]
    if same_geometric_corridor:
        _debug("add_ego_lane_keep_same_geometric_corridor")
        return [("ego_lane_keep", forward_ego_lane_id)]

    _debug("add_ego_lane_cutin")
    return [("ego_lane_cutin", forward_ego_lane_id)]


def _trajectory_from_xy_heading(xy, heading, channels=3):
    traj = np.zeros((len(xy), channels), dtype=np.float32)
    traj[:, :2] = xy[:, :2]
    if channels >= 3:
        traj[:, 2] = heading
    return traj


def _path_length(xy):
    xy = np.asarray(xy, dtype=np.float32)
    if xy.ndim != 2 or len(xy) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(xy[:, :2], axis=0), axis=-1).sum())


def _accelerating_distances(initial_speed, target_speed, accel, times):
    initial_speed = max(float(initial_speed), 0.0)
    target_speed = max(float(target_speed), 0.0)
    accel = max(float(accel), 1e-3)
    times = np.asarray(times, dtype=np.float32)
    if initial_speed >= target_speed:
        return initial_speed * times

    accel_time = (target_speed - initial_speed) / accel
    accel_t = np.minimum(times, accel_time)
    cruise_t = np.maximum(times - accel_time, 0.0)
    return (
        initial_speed * accel_t
        + 0.5 * accel * accel_t * accel_t
        + target_speed * cruise_t
    ).astype(np.float32)


def _braking_distances(initial_speed, decel, times):
    """Monotonic forward distance for constant braking until standstill."""
    initial_speed = max(float(initial_speed), 0.0)
    decel = max(float(decel), 1e-6)
    times = np.asarray(times, dtype=np.float32)
    stop_time = initial_speed / decel
    effective_t = np.minimum(times, stop_time)
    distances = (
        initial_speed * effective_t
        - 0.5 * decel * effective_t * effective_t
    )
    return np.maximum(distances, 0.0).astype(np.float32)


def _control_from_xy_heading(xy, heading, dt=0.1):
    vel = np.gradient(xy[:, :2], dt, axis=0).astype(np.float32)
    return np.concatenate(
        [
            xy[:, :2].astype(np.float32),
            np.cos(heading)[:, None].astype(np.float32),
            np.sin(heading)[:, None].astype(np.float32),
            vel[:, :2],
        ],
        axis=-1,
    )


def _to_local(planner_world, control_world, origin, angle):
    rot = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]], dtype=np.float64)
    planner = np.asarray(planner_world, dtype=np.float32).copy()
    control = np.asarray(control_world, dtype=np.float32).copy()
    planner[..., :2] = np.matmul(planner[..., :2] - origin, rot.T)
    if planner.shape[-1] >= 3:
        planner[..., 2] -= float(angle)
    control[..., :2] = np.matmul(control[..., :2] - origin, rot.T)
    control[..., 2:4] = np.matmul(control[..., 2:4], rot.T)
    control[..., 4:6] = np.matmul(control[..., 4:6], rot.T)
    return planner.astype(np.float32), control.astype(np.float32)


def build_osm_modal_candidates(
        current_state,
        timestep,
        controlled_agent_id,
        pluto_feature,
        input_data,
        batch_idx,
        cfg,
        horizon,
        channels,
        adversarial_target_agent_ids=None,
):
    if not bool(cfg.get("osm_modal_candidates_enabled", False)):
        return None
    # Hybrid OSM lane modes are kept for controlled BVs, but Ego defaults back to
    # the original learned Pluto candidate path.  Re-enable explicitly only for a
    # dedicated experiment.
    if controlled_agent_id is None and not bool(
        cfg.get("ego_osm_modal_candidates_enabled", False)
    ):
        return None

    state = _track_state(current_state, timestep, controlled_agent_id, rear_axle_reference=True)
    if state is None:
        return None
    position, heading, speed = state
    network = _current_road_network()
    lane_id = _lane_id_from_engine_object(network, controlled_agent_id)
    if lane_id is None:
        lane_id = _lane_id_from_reference_lines(network, pluto_feature, position, heading)
    if lane_id is None:
        lane_id = _closest_lane_id(network, position, heading)
    if lane_id is None:
        return None

    dt = float(cfg.get("osm_modal_candidates_dt", 0.1))
    actual_speed = max(float(speed), 0.0)
    # Do not inject an artificial cruise-speed floor into controlled BVs.  A BV
    # configured/observed at 0.5 m/s must generate lane-rule candidates from 0.5 m/s,
    # not from the legacy 2.0 m/s floor.  Keep the configurable floor only for the
    # optional Ego OSM mode, preserving its previous experiment semantics.
    if controlled_agent_id is None:
        min_speed = max(float(cfg.get("osm_modal_candidates_min_speed_mps", 2.0)), 0.0)
        speed = max(actual_speed, min_speed)
    else:
        speed = actual_speed
    times = (np.arange(horizon, dtype=np.float32) + 1.0) * dt
    modes = _build_lane_modes(
        network,
        lane_id,
        heading,
    )
    ego_lane_cutin_modes = _ego_lane_cutin_modes(
        network,
        current_state,
        timestep,
        controlled_agent_id,
        position,
        heading,
        cfg,
    )
    ego_cutin_anchor_position = position
    metadata = current_state.get("metadata", {}) if isinstance(current_state, dict) else {}
    ego_state_for_cutin = _track_state(current_state, timestep, metadata.get("sdc_id", None), rear_axle_reference=True)
    if ego_state_for_cutin is None:
        ego_state_for_cutin = _track_state(current_state, timestep, None, rear_axle_reference=True)
    if ego_state_for_cutin is not None:
        ego_cutin_anchor_position = ego_state_for_cutin[0]
    cutin_modes = _target_lane_cutin_modes(
        network,
        current_state,
        timestep,
        controlled_agent_id,
        adversarial_target_agent_ids,
        position,
        heading,
        cfg,
    )

    planner_world = []
    control_world = []
    labels = []
    min_motion_path = float(cfg.get("osm_modal_candidates_min_path_length_m", 2.0))
    mode_paths_by_label = {str(label): path for label, path in modes if label != "brake" and path}
    for label, path in modes:
        mode_speed = speed
        if label == "brake":
            decel = float(cfg.get("osm_modal_candidates_brake_decel_mps2", 6.0))
            # Clamp the braking integration at standstill.  The returned distance
            # is monotonic and remains fixed after v reaches zero.
            distances = _braking_distances(actual_speed, decel, times)
        elif controlled_agent_id is None and label == "straight":
            target_speed = float(cfg.get("ego_osm_straight_target_speed_km_h", 40.0)) / 3.6
            accel = float(cfg.get("ego_osm_straight_accel_mps2", 1.5))
            distances = _accelerating_distances(speed, target_speed, accel, times)
        else:
            distances = mode_speed * times
        xy, hdg = _sample_lane_path(network, path, position, distances)
        if xy is None:
            continue
        if label != "brake" and _path_length(xy) < min_motion_path:
            continue
        planner_world.append(_trajectory_from_xy_heading(xy, hdg, channels=max(channels, 3))[:, :channels])
        control_world.append(_control_from_xy_heading(xy, hdg, dt=dt))
        labels.append(label)

    for label, target_lane_id in ego_lane_cutin_modes:
        target_lane = _resolve_lane(network, target_lane_id)
        target_start_s = 0.0
        if target_lane is not None:
            try:
                target_start_s, _ = target_lane.local_coordinates(position)
            except Exception:
                target_start_s = 0.0
        target_heading = (
            _lane_heading(target_lane, longitudinal=max(float(target_start_s), 0.0))
            if target_lane is not None else heading
        )
        target_modes = _build_lane_modes(network, target_lane_id, target_heading)
        target_mode_paths_by_label = {
            str(mode_label): path
            for mode_label, path in target_modes
            if mode_label != "brake" and path
        }
        matched_mode_labels = [
            mode_label
            for mode_label in ("left_turn", "right_turn", "straight")
            if mode_label in mode_paths_by_label and mode_label in target_mode_paths_by_label
        ]
        added_mode_candidate = False
        for lane_change_distance in _ego_lane_cutin_distances(cfg):
            for mode_label in matched_mode_labels:
                xy, hdg = _sample_cutin_between_paths(
                    network,
                    mode_paths_by_label[mode_label],
                    target_mode_paths_by_label[mode_label],
                    position,
                    distances=speed * times,
                    lane_change_distance=lane_change_distance,
                    target_start_position=ego_cutin_anchor_position,
                )
                if xy is None:
                    continue
                if _path_length(xy) < min_motion_path:
                    continue
                planner_world.append(_trajectory_from_xy_heading(xy, hdg, channels=max(channels, 3))[:, :channels])
                control_world.append(_control_from_xy_heading(xy, hdg, dt=dt))
                labels.append(f"{label}_{mode_label}_{int(round(lane_change_distance))}m")
                added_mode_candidate = True
            if added_mode_candidate:
                continue
            xy, hdg = _sample_cutin_to_lane(
                network,
                lane_id,
                target_lane_id,
                position,
                heading,
                distances=speed * times,
                lane_change_distance=lane_change_distance,
                max_lanes=int(cfg.get("osm_ego_lane_cutin_follow_lanes", 5)),
                target_anchor_position=ego_cutin_anchor_position,
            )
            if xy is None:
                continue
            if _path_length(xy) < min_motion_path:
                continue
            planner_world.append(_trajectory_from_xy_heading(xy, hdg, channels=max(channels, 3))[:, :channels])
            control_world.append(_control_from_xy_heading(xy, hdg, dt=dt))
            labels.append(f"{label}_{int(round(lane_change_distance))}m")

    for label, target_lane_id in cutin_modes:
        for lane_change_distance in _cutin_distances(cfg):
            xy, hdg = _sample_cutin_to_lane(
                network,
                lane_id,
                target_lane_id,
                position,
                heading,
                distances=speed * times,
                lane_change_distance=lane_change_distance,
                max_lanes=int(cfg.get("osm_adversarial_cutin_follow_lanes", 5)),
            )
            if xy is None:
                continue
            if _path_length(xy) < min_motion_path:
                continue
            planner_world.append(_trajectory_from_xy_heading(xy, hdg, channels=max(channels, 3))[:, :channels])
            control_world.append(_control_from_xy_heading(xy, hdg, dt=dt))
            labels.append(f"{label}_{int(round(lane_change_distance))}m")

    if not planner_world:
        return None

    origin = input_data["origin"][batch_idx].detach().cpu().numpy()[:2]
    angle = float(input_data["angle"][batch_idx].detach().cpu().numpy())
    planner_local, control_local = _to_local(
        np.stack(planner_world, axis=0),
        np.stack(control_world, axis=0),
        origin,
        angle,
    )
    modal_logit = float(cfg.get("osm_modal_candidates_logit", -2.0))
    brake_logit = float(cfg.get("osm_modal_candidates_brake_logit", modal_logit))
    logits = np.asarray(
        [brake_logit if label == "brake" else modal_logit for label in labels],
        dtype=np.float32,
    )
    return {
        "planner": planner_local,
        "control": control_local,
        "logits": logits,
        "labels": labels,
    }