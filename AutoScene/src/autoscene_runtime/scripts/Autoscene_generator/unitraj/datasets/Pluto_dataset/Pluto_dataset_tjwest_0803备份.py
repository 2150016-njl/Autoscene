"""TJ-West Pluto feature builder.

This module intentionally keeps a distinct import path from the nuPlan builder.
The current algorithm-side implementation supports the ScenarioNet-normalized
schema used by both datasets, including short-history front padding.
"""

__all__ = ["PlutoDataset"]

import warnings

import traceback

import os
import pickle
import time
import numpy as np
import h5py
from scipy.spatial import cKDTree

from unitraj.datasets.Pluto_dataset.cost_map_manager import CostMapManager
from unitraj.datasets.Pluto_dataset.plot_lane_graph import visualize_lane_graph_integrity, plot_focused_lane_graph, \
    visualize_lane_graph_by_type
from unitraj.datasets.Pluto_dataset.utils import save_dict_to_hdf5
from unitraj.datasets.base_dataset import BaseDataset
from scenarionet.common_utils import read_scenario

from shapely.geometry import Polygon
from shapely.geometry import Point, LineString
from typing import List, Tuple
from unitraj.datasets.Pluto_dataset.pluto_utils import calculate_additional_ego_states, \
    PlutoFeature, interpolate_polyline, _is_lane_like

from unitraj.datasets.Pluto_dataset.map_topology import (
    build_lane_graph,
    estimate_route_lane_ids,
    expand_lateral_lane_ids,
    _as_int_id,
)

from unitraj.datasets.Pluto_dataset.refline_utils import _dedup_append, lane_length_xy, _crop_reference_line_from_ego, \
    _resample_reference_line, _line_with_heading, project_points_to_polyline


def _wrap_abs_angle(angle):
    return abs((float(angle) + np.pi) % (2.0 * np.pi) - np.pi)


def _clean_reference_segment(polyline):
    """Return finite XY points with consecutive duplicates removed."""
    arr = np.asarray(polyline, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
        return None
    xy = arr[:, :2]
    xy = xy[np.isfinite(xy).all(axis=1)]
    if len(xy) < 2:
        return None
    kept = [xy[0]]
    for point in xy[1:]:
        if float(np.linalg.norm(point - kept[-1])) > 1e-6:
            kept.append(point)
    if len(kept) < 2:
        return None
    return np.asarray(kept, dtype=np.float64)


def _local_polyline_heading(xy, index):
    """Tangent heading at a point, robust at the two endpoints."""
    index = int(np.clip(index, 0, len(xy) - 1))
    if index <= 0:
        vector = xy[1] - xy[0]
    elif index >= len(xy) - 1:
        vector = xy[-1] - xy[-2]
    else:
        vector = xy[index + 1] - xy[index - 1]
    return float(np.arctan2(vector[1], vector[0]))


def _orient_and_merge_reference_segments(
        segments,
        actor_position,
        actor_heading,
):
    """Orient stored polylines without changing the lane-ID path.

    A two-state dynamic program selects the forward/reversed storage direction
    for every lane/connector segment. Endpoint continuity is the primary cost;
    tangent continuity only resolves close endpoint choices. No heading limit is
    used, so legal turns and U-turns remain valid and lane IDs may change.
    """
    cleaned = []
    for segment in segments:
        xy = _clean_reference_segment(segment)
        if xy is None:
            return None, {"valid": False, "reason": "invalid_segment"}
        cleaned.append(xy)
    if not cleaned:
        return None, {"valid": False, "reason": "empty_path"}

    actor_xy = np.asarray(actor_position, dtype=np.float64).reshape(-1)[:2]
    variants = [(xy, xy[::-1].copy()) for xy in cleaned]

    dp = []
    first_states = []
    for candidate in variants[0]:
        nearest = int(np.argmin(np.linalg.norm(candidate - actor_xy[None, :], axis=1)))
        heading = _local_polyline_heading(candidate, nearest)
        heading_cost = _wrap_abs_angle(heading - actor_heading)
        position_cost = float(np.min(np.linalg.norm(candidate - actor_xy[None, :], axis=1)))
        first_states.append((4.0 * heading_cost + 0.01 * position_cost, None))
    dp.append(first_states)

    for index in range(1, len(variants)):
        states = []
        for candidate in variants[index]:
            candidate_start_heading = _local_polyline_heading(candidate, 0)
            best = None
            for previous_orientation, previous in enumerate(variants[index - 1]):
                previous_cost = dp[index - 1][previous_orientation][0]
                gap = float(np.linalg.norm(previous[-1] - candidate[0]))
                previous_end_heading = _local_polyline_heading(previous, len(previous) - 1)
                heading_delta = _wrap_abs_angle(candidate_start_heading - previous_end_heading)
                # Do not "repair" a real connected U-turn by creating a spatial
                # gap. Endpoint continuity dominates; heading is a tie-breaker.
                transition_cost = 100.0 * gap + heading_delta
                total = previous_cost + transition_cost
                if best is None or total < best[0]:
                    best = (total, previous_orientation)
            states.append(best)
        dp.append(states)

    orientation = min(range(2), key=lambda value: dp[-1][value][0])
    choices = [orientation]
    for index in range(len(variants) - 1, 0, -1):
        orientation = dp[index][orientation][1]
        choices.append(orientation)
    choices.reverse()

    oriented = [variants[index][choice] for index, choice in enumerate(choices)]
    merged = [oriented[0]]
    join_gaps = []
    join_heading_deltas = []
    for previous, current in zip(oriented[:-1], oriented[1:]):
        gap = float(np.linalg.norm(previous[-1] - current[0]))
        heading_delta = _wrap_abs_angle(
            _local_polyline_heading(current, 0)
            - _local_polyline_heading(previous, len(previous) - 1)
        )
        join_gaps.append(gap)
        join_heading_deltas.append(heading_delta)
        merged.append(current[1:] if gap <= 1e-6 else current)

    merged_xy = np.concatenate(merged, axis=0)
    if len(merged_xy) < 2:
        return None, {"valid": False, "reason": "too_short_after_merge"}

    delta = np.diff(merged_xy, axis=0)
    segment_heading = np.arctan2(delta[:, 1], delta[:, 0])
    point_heading = np.concatenate([segment_heading, segment_heading[-1:]])
    line = np.concatenate([merged_xy, point_heading[:, None]], axis=1)
    local_heading_deltas = (
        np.abs((np.diff(segment_heading) + np.pi) % (2.0 * np.pi) - np.pi)
        if len(segment_heading) > 1 else np.zeros((0,), dtype=np.float64)
    )
    return line, {
        "valid": True,
        "reversed": [bool(choice) for choice in choices],
        "join_gaps_m": [float(value) for value in join_gaps],
        "join_heading_deltas_rad": [float(value) for value in join_heading_deltas],
        "max_join_gap_m": None if not join_gaps else float(max(join_gaps)),
        "max_join_heading_delta_rad": (
            None if not join_heading_deltas else float(max(join_heading_deltas))
        ),
        "max_local_heading_delta_rad": (
            None if len(local_heading_deltas) == 0 else float(np.max(local_heading_deltas))
        ),
    }


def _get_ego_features(state, ego_category_idx: int = 0, present_idx: int = 20, history_samples: int = 20):
    pos = state['position']
    history_start = max(0, present_idx - history_samples)
    end = present_idx + 1

    pos = pos[history_start:end]
    T = len(pos)
    pad_len = (history_samples + 1) - T

    def _pad(arr, seq_idx=0):
        if pad_len <= 0:
            return arr
        pad_shape = ((pad_len,) + arr.shape[1:]) if isinstance(arr, np.ndarray) else (pad_len,)
        return np.concatenate([np.zeros(pad_shape, dtype=arr.dtype), arr], axis=0)

    position = _pad(pos[..., :2] if pos.shape[-1] >= 2 else pos)
    heading = _pad(state['heading'][history_start:end])
    vel = state['velocity'][history_start:end]
    velocity = _pad(vel[..., :2] if vel.shape[-1] >= 2 else vel)

    if 'acceleration' in state:
        accel = state['acceleration'][history_start:end]
        acceleration = _pad(accel[..., :2] if accel.shape[-1] >= 2 else accel)
    else:
        acceleration = np.zeros((history_samples + 1, 2), dtype=np.float64)

    width = _pad(np.array(state['width'][history_start:end]).reshape(-1, 1))
    length = _pad(np.array(state['length'][history_start:end]).reshape(-1, 1))
    # width = np.full((len(heading), 1), 1.8)  # 形状: (N, 1)
    # length = np.full((len(heading), 1), 4.8)  # 形状: (N, 1)
    shape = np.concatenate([width, length], axis=-1)

    valid_mask = _pad(state['valid'][history_start:end].astype(np.bool_))
    category = np.array(ego_category_idx, dtype=np.int8)

    return {
        "position": position.astype(np.float64),
        "heading": heading.astype(np.float64),
        "velocity": velocity.astype(np.float64),
        "acceleration": acceleration.astype(np.float64),
        "shape": shape.astype(np.float64),
        "category": category,
        "valid_mask": valid_mask.astype(np.bool_),
    }


def _box_corners_xy(center_xy: np.ndarray, heading: float, width: float, length: float) -> np.ndarray:
    """Return oriented 2D box corners (4,2) in world frame."""
    center_xy = np.asarray(center_xy, dtype=np.float64).reshape(-1)[:2]
    dx = float(length) / 2.0
    dy = float(width) / 2.0
    corners = np.array([[dx, dy], [-dx, dy], [-dx, -dy], [dx, -dy]], dtype=np.float64)
    c, s = np.cos(heading), np.sin(heading)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    return center_xy[None, :] + corners @ rot.T


def _sanitize_poly_xy(poly_xy: np.ndarray) -> np.ndarray:
    poly_xy = np.asarray(poly_xy, dtype=np.float64)
    if poly_xy.ndim != 2 or poly_xy.shape[0] < 3:
        return np.zeros((0, 2), dtype=np.float64)
    poly_xy = poly_xy[:, :2]
    valid = np.isfinite(poly_xy).all(axis=1)
    poly_xy = poly_xy[valid]
    if len(poly_xy) >= 2 and np.linalg.norm(poly_xy[0] - poly_xy[-1]) < 1e-6:
        poly_xy = poly_xy[:-1]
    if len(poly_xy) < 3:
        return np.zeros((0, 2), dtype=np.float64)

    dedup = [poly_xy[0]]
    for pt in poly_xy[1:]:
        if np.linalg.norm(pt - dedup[-1]) >= 1e-6:
            dedup.append(pt)
    poly_xy = np.asarray(dedup, dtype=np.float64)
    return poly_xy if len(poly_xy) >= 3 else np.zeros((0, 2), dtype=np.float64)


def _sample_closed_ring(poly_xy: np.ndarray, num_points: int) -> np.ndarray:
    poly_xy = _sanitize_poly_xy(poly_xy)
    if len(poly_xy) == 0:
        return np.zeros((0, 2), dtype=np.float64)

    ring = np.concatenate([poly_xy, poly_xy[:1]], axis=0)
    seg = np.linalg.norm(np.diff(ring, axis=0), axis=1)
    perimeter = float(seg.sum())
    if perimeter < 1e-6:
        return np.repeat(poly_xy[:1], num_points, axis=0)

    cum = np.concatenate([[0.0], np.cumsum(seg)])
    targets = np.linspace(0.0, perimeter, num_points, endpoint=False)
    sampled = np.zeros((num_points, 2), dtype=np.float64)
    for dim in range(2):
        sampled[:, dim] = np.interp(targets, cum, ring[:, dim])
    return sampled


def _compute_path_tangent(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if len(points) == 1:
        return np.array([[1.0, 0.0]], dtype=np.float64)

    tang = np.zeros_like(points)
    tang[0] = points[1] - points[0]
    tang[-1] = points[-1] - points[-2]
    if len(points) > 2:
        tang[1:-1] = points[2:] - points[:-2]
    norm = np.linalg.norm(tang, axis=1, keepdims=True)
    tang = tang / np.clip(norm, 1e-6, None)
    return tang


def _fill_polyline_gaps(polyline: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    polyline = np.asarray(polyline, dtype=np.float64).copy()
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if len(polyline) == 0 or valid_mask.all():
        return polyline
    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) == 0:
        return polyline
    if len(valid_idx) == 1:
        polyline[:] = polyline[valid_idx[0]]
        return polyline

    query = np.arange(len(polyline), dtype=np.float64)
    for dim in range(polyline.shape[1]):
        polyline[:, dim] = np.interp(query, valid_idx.astype(np.float64), polyline[valid_idx, dim])
    return polyline


def _smooth_polyline(polyline: np.ndarray) -> np.ndarray:
    polyline = np.asarray(polyline, dtype=np.float64).copy()
    if len(polyline) < 3:
        return polyline
    kernel = np.array([0.25, 0.5, 0.25], dtype=np.float64)
    for dim in range(polyline.shape[1]):
        tmp = np.convolve(polyline[:, dim], kernel, mode='same')
        tmp[0] = polyline[0, dim]
        tmp[-1] = polyline[-1, dim]
        polyline[:, dim] = tmp
    return polyline


def _smooth_series(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).copy()
    if len(values) < 3:
        return values
    kernel = np.array([0.25, 0.5, 0.25], dtype=np.float64)
    tmp = np.convolve(values, kernel, mode='same')
    tmp[0] = values[0]
    tmp[-1] = values[-1]
    return tmp


def _validate_reconstructed_lane_bounds(
        centerline: np.ndarray, left_bound: np.ndarray, right_bound: np.ndarray
) -> bool:
    centerline = np.asarray(centerline, dtype=np.float64)
    left_bound = np.asarray(left_bound, dtype=np.float64)
    right_bound = np.asarray(right_bound, dtype=np.float64)
    if len(centerline) < 2 or len(left_bound) != len(centerline) or len(right_bound) != len(centerline):
        return False

    tang = _compute_path_tangent(centerline)
    normals = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
    left_signed = np.sum((left_bound - centerline) * normals, axis=1)
    right_signed = np.sum((right_bound - centerline) * normals, axis=1)
    lane_width = left_signed - right_signed

    if np.any(left_signed < 0.2) or np.any(right_signed > -0.2):
        return False
    if np.any(lane_width < 1.5) or np.any(lane_width > 12.0):
        return False

    tang_left = _compute_path_tangent(left_bound)
    tang_right = _compute_path_tangent(right_bound)
    if np.mean(np.sum(tang_left * tang, axis=1)) < 0.5:
        return False
    if np.mean(np.sum(tang_right * tang, axis=1)) < 0.5:
        return False

    left_seg = np.linalg.norm(np.diff(left_bound, axis=0), axis=1)
    right_seg = np.linalg.norm(np.diff(right_bound, axis=0), axis=1)
    if len(left_seg) > 0 and np.mean(left_seg < 1e-3) > 0.1:
        return False
    if len(right_seg) > 0 and np.mean(right_seg < 1e-3) > 0.1:
        return False

    return True


def _extract_lane_bounds_from_polygon(
        centerline: np.ndarray, polygon_xy: np.ndarray, num_points: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate lane widths from polygon support and reconstruct ordered bounds along centerline normals."""
    centerline = np.asarray(centerline, dtype=np.float64)
    polygon_xy = _sanitize_poly_xy(polygon_xy)
    if len(centerline) < 2 or len(polygon_xy) < 3:
        return None, None

    dense_ring = _sample_closed_ring(polygon_xy, max(128, num_points * 12))
    if len(dense_ring) == 0:
        return None, None

    tang = _compute_path_tangent(centerline)
    normals = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
    seg = np.linalg.norm(np.diff(centerline, axis=0), axis=1)
    search_half_window = max(1.5, float(np.median(seg)) * 2.5 if len(seg) > 0 else 2.5)

    left_offset = np.zeros((len(centerline),), dtype=np.float64)
    right_offset = np.zeros((len(centerline),), dtype=np.float64)
    left_valid = np.zeros((len(centerline),), dtype=bool)
    right_valid = np.zeros((len(centerline),), dtype=bool)

    for i, center in enumerate(centerline):
        t = tang[i]
        n = normals[i]
        rel = dense_ring - center[None, :]
        longitudinal = rel @ t
        lateral = rel @ n

        near_mask = np.abs(longitudinal) <= search_half_window
        left_mask = near_mask & (lateral > 0.10)
        right_mask = near_mask & (lateral < -0.10)

        if left_mask.any():
            left_offset[i] = float(np.max(lateral[left_mask]))
            left_valid[i] = True
        if right_mask.any():
            right_offset[i] = float(-np.min(lateral[right_mask]))
            right_valid[i] = True

    min_valid = max(4, num_points // 3)
    if left_valid.sum() < min_valid or right_valid.sum() < min_valid:
        return None, None

    left_offset = _fill_polyline_gaps(left_offset[:, None], left_valid)[:, 0]
    right_offset = _fill_polyline_gaps(right_offset[:, None], right_valid)[:, 0]

    left_offset = _smooth_series(_smooth_series(left_offset))
    right_offset = _smooth_series(_smooth_series(right_offset))

    left_offset = np.clip(left_offset, 0.5, 6.0)
    right_offset = np.clip(right_offset, 0.5, 6.0)

    left_bound = centerline + left_offset[:, None] * normals
    right_bound = centerline - right_offset[:, None] * normals

    if not _validate_reconstructed_lane_bounds(centerline, left_bound, right_bound):
        return None, None

    return left_bound, right_bound


class PlutoDataset(BaseDataset):
    """
    Dataset class for Pluto feature format reading NuPlan db files (ScenarioDescription).
    """

    def __init__(self, config=None, is_validation=False):
        # Match pluto_feature_builder.py indexing semantics (no dependency on nuplan types)
        # agent categories: [EGO, VEHICLE, PEDESTRIAN, BICYCLE]
        self.interested_objects_types = ["EGO", "VEHICLE", "PEDESTRIAN", "BICYCLE"]
        # static obstacle categories (nuplan: [CZONE_SIGN, BARRIER, TRAFFIC_CONE, GENERIC_OBJECT])
        self.static_objects_types = ["CZONE_SIGN", "TRAFFIC_BARRIER", "TRAFFIC_CONE", "GENERIC_OBJECT"]
        # map polygon categories (nuplan: [LANE, LANE_CONNECTOR, CROSSWALK])
        self.polygon_types = ["LANE", "LANE_CONNECTOR", "CROSSWALK"]

        self.max_agents = 48
        self.max_static_obstacles = 10
        self.__collate_fn__ = PlutoFeature.collate

        self.radius = config.get('radius', 100)
        self.history_horizon = config.get('history_horizon', 2)
        self.future_horizon = config.get('future_horizon', 8)
        self.sample_interval = config.get('sample_interval', 0.1)
        self.history_samples = int(self.history_horizon / self.sample_interval)
        self.future_samples = int(self.future_horizon / self.sample_interval) - 1
        self.ego_params = None
        self.strict_route_refline = config.get("strict_route_refline", not is_validation)
        self.enable_route_lateral_refline = bool(config.get("enable_route_lateral_refline", False))
        self.ego_strict_route_refline = config.get("ego_strict_route_refline", None)
        self.ego_enable_route_lateral_refline = config.get("ego_enable_route_lateral_refline", None)
        self.ego_expert_reference_line_enabled = bool(config.get("ego_expert_reference_line_enabled", False))
        self.ego_expert_reference_line_first = bool(config.get("ego_expert_reference_line_first", True))
        self.ego_expert_reference_line_min_length = float(config.get("ego_expert_reference_line_min_length", 10.0))
        self.route_lateral_refline_hops = int(np.clip(
            int(config.get("route_lateral_refline_hops", 2)),
            0,
            2,
        ))
        self.route_lateral_same_direction_only = bool(config.get("route_lateral_same_direction_only", True))
        self.route_lateral_heading_threshold = float(config.get("route_lateral_heading_threshold", np.pi / 3))
        self.map_feature_cache_distance = float(config.get("map_feature_cache_distance", 8.0))
        self.map_cache_build_radius = float(config.get(
            "map_cache_build_radius",
            self.radius + self.map_feature_cache_distance,
        ))
        self.map_cache_build_radius = max(self.radius, self.map_cache_build_radius)
        self.scene_map_relevant_distance = float(config.get(
            "scene_map_relevant_distance",
            self.radius,
        ))
        self.scene_map_cache_margin = float(config.get("scene_map_cache_margin", 10.0))
        self.reference_line_cache_heading_threshold = float(
            config.get("reference_line_cache_heading_threshold", 0.7)
        )
        self.reference_line_cache_lateral_distance = float(
            config.get("reference_line_cache_lateral_distance", 12.0)
        )
        self.reference_line_cache_min_remaining = float(
            config.get("reference_line_cache_min_remaining", 20.0)
        )
        self.reference_line_build_radius = float(
            config.get("reference_line_build_radius", self.radius * 1.5)
        )
        self.reference_line_build_radius = max(
            self.radius,
            self.reference_line_build_radius,
        )
        # A dataset instance belongs to one planner/policy episode. Keeping these
        # caches class-global retains every scenario's full lane graph forever.
        self._lane_graph_cache = {}
        self._route_lane_cache = {}
        self._map_feature_cache = {}
        self._reference_line_cache = {}
        self._last_refline_audit_by_agent = {}
        self._last_refline_audit_latest = None
        self._refline_audit_lane_index_cache = {}
        super().__init__(config, is_validation)

    def _get_scene_map_cache_region(self, all_tracks, controlled_track_id):
        def valid_track_positions(track):
            state = track.get("state", {}) if isinstance(track, dict) else {}
            position = np.asarray(state.get("position", []), dtype=np.float64)
            if position.ndim != 2 or position.shape[0] == 0:
                return np.zeros((0, 2), dtype=np.float64)
            position = position[:, :2]
            valid = np.asarray(
                state.get("valid", np.ones((len(position),), dtype=np.bool_))
            ).reshape(-1)
            if len(valid) < len(position):
                valid = np.pad(
                    valid.astype(np.bool_),
                    (0, len(position) - len(valid)),
                    constant_values=False,
                )
            keep = valid[:len(position)].astype(np.bool_) & np.isfinite(position).all(axis=1)
            return position[keep]

        ego_track = all_tracks.get("ego")
        if ego_track is None:
            ego_track = all_tracks.get(controlled_track_id, {})
        ego_positions = valid_track_positions(ego_track)
        if len(ego_positions) == 0:
            return np.zeros((2,), dtype=np.float64), self.map_cache_build_radius

        included_positions = [ego_positions]
        for track_id, track in all_tracks.items():
            if str(track_id) == "ego":
                continue
            track_type = str(track.get("type", "")).upper()
            if track_type not in {"VEHICLE", "CAR", "TRUCK", "BUS", "EGO"}:
                continue
            positions = valid_track_positions(track)
            if len(positions) == 0:
                continue
            min_dist = np.min(
                np.linalg.norm(
                    positions[:, None, :] - ego_positions[None, :, :],
                    axis=-1,
                )
            )
            if (
                str(track_id) == str(controlled_track_id)
                or min_dist <= self.scene_map_relevant_distance
            ):
                included_positions.append(positions)

        coverage_positions = np.concatenate(included_positions, axis=0)
        bounds_min = np.min(coverage_positions, axis=0)
        bounds_max = np.max(coverage_positions, axis=0)
        center = (bounds_min + bounds_max) * 0.5
        trajectory_radius = float(
            np.max(np.linalg.norm(coverage_positions - center[None, :], axis=1))
        )
        build_radius = max(
            self.map_cache_build_radius,
            trajectory_radius + self.radius + self.scene_map_cache_margin,
        )
        return center, build_radius

    def process_data_chunk(self, worker_index):
        with open(os.path.join('tmp', '{}.pkl'.format(worker_index)), 'rb') as f:
            data_chunk = pickle.load(f)
        file_list = {}
        data_path, mapping, data_list, dataset_name = data_chunk
        hdf5_path = os.path.join(self.cache_path, f'{worker_index}.h5')

        with h5py.File(hdf5_path, 'w') as f:
            for cnt, file_name in enumerate(data_list):
                if worker_index == 0 and cnt % max(int(len(data_list) / 10), 1) == 0:
                    print(f'{cnt}/{len(data_list)} data processed', flush=True)
                scenario = read_scenario(data_path, mapping, file_name)

                try:
                    output = self.preprocess(scenario)

                    pluto_feature = self.process(output)

                    output = self.postprocess(pluto_feature)

                except Exception as e:
                    print('Warning: {} in {}'.format(e, file_name))
                    traceback.print_exc()
                    output = None

                if output is None: continue

                for i, record in enumerate(output):
                    grp_name = dataset_name + '-' + str(worker_index) + '-' + str(cnt) + '-' + str(i)
                    grp = f.create_group(grp_name)
                    save_dict_to_hdf5(grp, record)

                    file_info = {'h5_path': hdf5_path}
                    file_list[grp_name] = file_info
                del scenario
                del output

        return file_list

    def _resolve_controlled_agent_id(self, scenario, controlled_agent_id=None):
        tracks = scenario.get("tracks", {})
        if controlled_agent_id is None:
            return "ego"

        controlled_agent_id = str(controlled_agent_id)
        if controlled_agent_id in tracks:
            return controlled_agent_id

        for track_id in tracks.keys():
            if str(track_id) == controlled_agent_id:
                return track_id
        return "ego"

    def _resolve_controlled_category_idx(self, controlled_track_id, controlled_track) -> int:
        if str(controlled_track_id) == "ego":
            return self.interested_objects_types.index("EGO")

        track_type = str(controlled_track.get("type", "")).upper()
        if track_type in {"VEHICLE", "CAR", "TRUCK", "BUS", "EGO"}:
            return self.interested_objects_types.index("VEHICLE")
        if track_type in {"PEDESTRIAN", "PEDESTRIAN_ADULT", "PEDESTRIAN_CHILD"}:
            return self.interested_objects_types.index("PEDESTRIAN")
        if track_type in {"BICYCLE", "BICYCLIST", "CYCLIST"}:
            return self.interested_objects_types.index("BICYCLE")
        return self.interested_objects_types.index("VEHICLE")

    def _controlled_actor_type(self, controlled_track_id, controlled_track) -> str:
        if str(controlled_track_id) == "ego":
            return "EGO"
        return str(controlled_track.get("type", "")).upper()

    def _controlled_actor_is_vehicle(self) -> bool:
        actor_type = str(getattr(self, "controlled_actor_type", "EGO")).upper()
        return actor_type in {"EGO", "VEHICLE", "CAR", "TRUCK", "BUS"}

    def _estimate_dense_route_lane_ids(self, positions_xy, valid_mask, lane_graph, max_dist_m=4.0):
        """Recover an ordered lane sequence from the controlled agent trajectory."""
        if lane_graph is None or not getattr(lane_graph, "lanes", None):
            return []
        pos = np.asarray(positions_xy, dtype=np.float64)
        if pos.ndim != 2 or pos.shape[-1] < 2 or len(pos) == 0:
            return []
        valid = np.asarray(valid_mask, dtype=np.bool_).reshape(-1)
        if len(valid) < len(pos):
            valid = np.pad(valid, (0, len(pos) - len(valid)), constant_values=False)
        keep = valid[:len(pos)] & np.isfinite(pos[:, :2]).all(axis=1)
        pos = pos[keep, :2]
        if len(pos) == 0:
            return []

        lane_ids = list(lane_graph.lanes.keys())
        sampled_points = []
        sampled_lane_ids = []
        for lid in lane_ids:
            centerline = np.asarray(lane_graph.lanes[lid].centerline, dtype=np.float64)
            if centerline.ndim != 2 or len(centerline) < 2:
                continue
            if centerline.shape[0] > 50:
                indices = np.linspace(0, centerline.shape[0] - 1, 50).astype(np.int64)
                centerline = centerline[indices]
            sampled_points.append(centerline[:, :2])
            sampled_lane_ids.extend([int(lid)] * len(centerline))
        if not sampled_points:
            return []

        tree = cKDTree(np.concatenate(sampled_points, axis=0))
        sampled_lane_ids = np.asarray(sampled_lane_ids, dtype=np.int64)
        nearest_dist, nearest_idx = tree.query(pos, k=1)
        matched = sampled_lane_ids[np.asarray(nearest_idx, dtype=np.int64)]

        route = []
        last = None
        for lid, dist in zip(matched.tolist(), nearest_dist.tolist()):
            if float(dist) > float(max_dist_m):
                continue
            lid = int(lid)
            if lid == last:
                continue
            route.append(lid)
            last = lid

        # Remove short A-B-A jitter from nearest-lane ambiguity.
        cleaned = []
        for lid in route:
            if len(cleaned) >= 2 and cleaned[-2] == lid:
                cleaned.pop()
                continue
            cleaned.append(lid)
        return cleaned
    def _tjwest_ref_audit_enabled(self):
        return os.environ.get(
            "UNITRAJ_TJWEST_REF_AUDIT",
            "1",
        ) not in ("0", "false", "False")

    def _build_pkl_lane_audit(
            self,
            controlled_track_id,
            query_xy,
            route_lane_seq,
            lane_graph,
            map_cache_identity,
    ):
        """只构造诊断数据，不改变特征、候选参考线或轨迹选择。"""
        route_ids = []
        for value in (route_lane_seq if route_lane_seq is not None else []):
            try:
                lane_id = int(value)
            except (TypeError, ValueError):
                continue
            if lane_id in lane_graph.lanes and (
                not route_ids or route_ids[-1] != lane_id
            ):
                route_ids.append(lane_id)

        cache_key = int(map_cache_identity)
        cached = self._refline_audit_lane_index_cache.get(cache_key)
        if cached is None:
            sample_xy = []
            sample_ids = []
            for lane_id, lane in lane_graph.lanes.items():
                centerline = np.asarray(
                    getattr(lane, "centerline", []),
                    dtype=np.float64,
                )
                if centerline.ndim != 2 or len(centerline) == 0:
                    continue
                centerline = centerline[:, :2]
                finite = np.isfinite(centerline).all(axis=1)
                centerline = centerline[finite]
                if len(centerline) == 0:
                    continue
                sample_xy.append(centerline)
                sample_ids.extend([int(lane_id)] * len(centerline))
            if sample_xy:
                stacked_xy = np.concatenate(sample_xy, axis=0)
                cached = (
                    cKDTree(stacked_xy),
                    np.asarray(sample_ids, dtype=np.int64),
                )
            else:
                cached = (None, np.zeros((0,), dtype=np.int64))
            self._refline_audit_lane_index_cache[cache_key] = cached

        lane_tree, sampled_lane_ids = cached
        nearest_lane_id = None
        nearest_lane_distance = None
        if lane_tree is not None and len(sampled_lane_ids) > 0:
            distance, nearest_index = lane_tree.query(
                np.asarray(query_xy, dtype=np.float64)[:2],
                k=1,
            )
            nearest_lane_id = int(sampled_lane_ids[int(nearest_index)])
            nearest_lane_distance = float(distance)

        join_gap_threshold = float(
            self.config.get("tjwest_ref_audit_lane_join_gap_m", 3.0)
        )
        route_pairs = []
        discontinuous_pairs = []
        for first, second in zip(route_ids[:-1], route_ids[1:]):
            first_xy = np.asarray(
                lane_graph.lanes[first].centerline,
                dtype=np.float64,
            )[:, :2]
            second_xy = np.asarray(
                lane_graph.lanes[second].centerline,
                dtype=np.float64,
            )[:, :2]
            if len(first_xy) == 0 or len(second_xy) == 0:
                gap = float("inf")
            else:
                gap = float(np.linalg.norm(first_xy[-1] - second_xy[0]))
            pair = {
                "from": int(first),
                "to": int(second),
                "endpoint_gap_m": gap,
            }
            route_pairs.append(pair)
            if not np.isfinite(gap) or gap > join_gap_threshold:
                discontinuous_pairs.append(pair)

        audit = {
            "controlled_agent_id": str(controlled_track_id),
            "pkl_current_lane_id": nearest_lane_id,
            "pkl_current_lane_distance_m": nearest_lane_distance,
            "pkl_route_lane_ids": route_ids,
            "pkl_route_lane_count": int(len(route_ids)),
            "pkl_route_join_gap_threshold_m": join_gap_threshold,
            "pkl_route_pairs": route_pairs,
            "pkl_route_discontinuous_pairs": discontinuous_pairs,
            "pkl_route_is_continuous": len(discontinuous_pairs) == 0,
        }
        self._last_refline_audit_by_agent[str(controlled_track_id)] = audit
        self._last_refline_audit_latest = audit
        return audit


    def preprocess(self, scenario, current_step=20, controlled_agent_id=None):
        """
        In UniTraj, preprocess often generates intermediate dict.
        Here we skip intermediate formatting and construct pluto feature dict explicitly.
        """
        # We can either return the raw PlutoFeature or dict that HDF5 caches.
        # But HDF5 can't cache PlutoFeature directly if it contains custom objects.
        # Let's return primitive dict.
        all_tracks = scenario['tracks']
        controlled_track_id = self._resolve_controlled_agent_id(scenario, controlled_agent_id)
        controlled_track = all_tracks[controlled_track_id]
        self.controlled_actor_type = self._controlled_actor_type(controlled_track_id, controlled_track)
        self.ego_params = dict(scenario['metadata']['ego_vehicle_parameters'])
        if controlled_track_id != "ego" and self._controlled_actor_is_vehicle():
            controlled_length = float(np.asarray(controlled_track["state"].get("length", [4.8])).reshape(-1)[0])
            self.ego_params["rear_axle_to_center"] = 0.35 * controlled_length
        elif not self._controlled_actor_is_vehicle():
            self.ego_params = {}

        ego_track_state = self._build_local_ego_state_for_pluto(controlled_track['state'])
        ego_state_list = self.parse_tracks_to_states(controlled_track['state'])
        map_features_list = scenario['map_features']
        traffic_light_status = scenario['dynamic_map_states']
        return [
            ego_track_state,
            ego_state_list,
            map_features_list,
            traffic_light_status,
            all_tracks,
            controlled_track_id,
        ]

    def process(self, data, current_step=20, controlled_agent_id=None):
        profile_timing = os.environ.get("UNITRAJ_PROFILE_TIMING", "0") == "1"
        process_start = time.perf_counter()
        [
            ego_track_state,
            ego_state_list,
            map_features_list,
            traffic_light_status,
            all_tracks,
            controlled_track_id,
        ] = data

        # Use current_step as the present index instead of hardcoded self.history_samples
        present_idx = current_step
        present_idx = min(present_idx, len(ego_state_list) - 1)

        present_ego_state = ego_state_list[present_idx]
        query_xy = present_ego_state['position'][:2]
        # print('query_xy', query_xy)

        # Build lane topology once and reuse in downstream feature extraction.
        # Speed optimization: cache it for the same map instance (object id)
        map_id = id(map_features_list)
        if map_id not in self._lane_graph_cache:
            lane_graph = build_lane_graph(map_features_list, infer_from_geometry=True, geom_link_dist_m=2.0)
            self._lane_graph_cache[map_id] = lane_graph
        else:
            lane_graph = self._lane_graph_cache[map_id]

        controlled_cache_key = (map_id, str(controlled_track_id))
        if controlled_cache_key not in self._route_lane_cache:
            ego_xy = np.asarray([s['position'][:2] for s in ego_state_list], dtype=np.float64)
            route_lane_seq = estimate_route_lane_ids(
                ego_xy,
                lane_graph,
                max_dist_m=1.5,
                min_hold_frames=3,
                no_backtrack=True,
                include_lateral_neighbors=False,
            )
            if str(controlled_track_id) != "ego" and len(route_lane_seq) <= 1:
                state = all_tracks[controlled_track_id].get("state", {})
                dense_route_lane_seq = self._estimate_dense_route_lane_ids(
                    np.asarray(state.get("position", ego_xy), dtype=np.float64)[..., :2],
                    state.get("valid", np.ones((len(ego_xy),), dtype=np.bool_)),
                    lane_graph,
                    max_dist_m=float(self.config.get("cbv_dense_route_max_dist_m", 4.0)),
                )
                if len(dense_route_lane_seq) > len(route_lane_seq):
                    if os.environ.get("UNITRAJ_REF_LINE_DEBUG", "1") not in ("0", "false", "False"):
                        print(
                            "[RefLineDebug] dense route fallback",
                            {
                                "controlled_agent_id": str(controlled_track_id),
                                "old_route": [int(x) for x in route_lane_seq[:12]],
                                "dense_route_head": [int(x) for x in dense_route_lane_seq[:12]],
                                "dense_route_len": len(dense_route_lane_seq),
                            },
                        )
                    route_lane_seq = dense_route_lane_seq
            # print("\n================ ROUTE_SEQ ================")
            # print(route_lane_seq[:50])
            # print("===========================================\n")
            self._route_lane_cache[controlled_cache_key] = route_lane_seq

            # try:
            #     # DEBUG VISUALIZATION: Dump the lane graph mapping to json and image out
            #     import os
            #     os.environ['DUMP_LANE_GRAPH'] = '1'
            #     if os.environ.get('DUMP_LANE_GRAPH', '0') == '1':
            #         from unitraj.datasets.Pluto_dataset.plot_lane_graph import dump_lane_graph_to_json, plot_lane_graph
            #         # dump_lane_graph_to_json(lane_graph, out_file=f"lane_graph_{map_id}.json")
            #         # plot_focused_lane_graph(
            #         #     lane_graph=lane_graph,
            #         #     route_seq=route_lane_seq,  # 路由车道ID
            #         #     ego_pos=np.array([0, 0]),  # 自车位置
            #         #     out_file="focused_map.png",
            #         #     focus_radius=200.0,  # 100米范围
            #         #     figsize=(12, 12)
            #         # )
            #         visualize_lane_graph_by_type(
            #             lane_graph=lane_graph,
            #             route_seq=route_lane_seq,
            #             ego_pos=np.array([0, 0]),
            #             out_file="lanes_only.png"
            #         )
            #         # states = visualize_lane_graph_integrity( lane_graph=lane_graph, out_file="lane_graph_check.png" )
            #         # plot_lane_graph(lane_graph, route_seq=route_lane_seq, ego_pos=query_xy, out_file=f"lane_graph_viz_{map_id}.png", map_features_list=map_features_list)
            # except Exception as e:
            #     print(f"Warning: Failed to dump lane graph: {e}")

        else:
            route_lane_seq = self._route_lane_cache[controlled_cache_key]

        if self._tjwest_ref_audit_enabled():
            self._build_pkl_lane_audit(
                controlled_track_id=controlled_track_id,
                query_xy=query_xy,
                route_lane_seq=route_lane_seq,
                lane_graph=lane_graph,
                map_cache_identity=map_id,
            )
        route_seconds = time.perf_counter() - process_start

        route_lane_seq_for_map = route_lane_seq
        if self.enable_route_lateral_refline and self._controlled_actor_is_vehicle():
            route_lane_seq_for_map = expand_lateral_lane_ids(
                route_lane_seq,
                lane_graph,
                lateral_hops=self.route_lateral_refline_hops,
                same_direction_only=self.route_lateral_same_direction_only,
                heading_threshold=self.route_lateral_heading_threshold,
            )

        data = {}
        prev_idx = max(0, present_idx - 1)
        data["current_state"] = self._get_ego_current_state(
            ego_state_list[present_idx], ego_state_list[prev_idx]
        )

        controlled_features = _get_ego_features(
            ego_track_state,
            ego_category_idx=self._resolve_controlled_category_idx(
                controlled_track_id,
                all_tracks[controlled_track_id],
            ),
            present_idx=present_idx,
            history_samples=self.history_samples,
        )
        # print('controlled features', controlled_features['position'])
        agent_features, agent_tokens, agents_polygon = self._get_agent_features(
            query_xy=query_xy,
            present_idx=present_idx,
            all_tracks=all_tracks,
            controlled_agent_id=controlled_track_id,
        )
        agent_seconds = time.perf_counter() - process_start - route_seconds

        data["agent"] = {}
        for k in agent_features.keys():
            data["agent"][k] = np.concatenate(
                [controlled_features[k][np.newaxis, ...], agent_features[k]], axis=0
            )
        agent_tokens = [str(controlled_track_id)] + agent_tokens

        is_validation = getattr(self, 'is_validation', True)

        if is_validation:
            data["agent_tokens"] = agent_tokens
            # data["controlled_agent_id"] = str(controlled_track_id)

        data["static_objects"], static_objects = self._get_static_objects_features(
            query_xy=query_xy,
            present_idx=present_idx,
            all_tracks=all_tracks
        )
        static_seconds = time.perf_counter() - process_start - route_seconds - agent_seconds

        map_start = time.perf_counter()
        traffic_light_values = {}
        traffic_light_state_mapping = {
            "TRAFFIC_LIGHT_GREEN": 0,
            "TRAFFIC_LIGHT_YELLOW": 1,
            "TRAFFIC_LIGHT_RED": 2,
            "TRAFFIC_LIGHT_UNKNOWN": 3,
        }
        for lane_id, tl_info in sorted((traffic_light_status or {}).items(), key=lambda item: str(item[0])):
            if not isinstance(tl_info, dict):
                continue
            state = tl_info.get("state", {}) or {}
            object_state = state.get("object_state", [])
            try:
                light_value = object_state[present_idx] if present_idx < len(object_state) else None
            except TypeError:
                light_value = object_state
            traffic_light_values[str(lane_id)] = int(
                traffic_light_state_mapping.get(str(light_value).upper(), 3)
        )
        map_cache_key = map_id
        map_cache_entry = self._map_feature_cache.get(map_cache_key)
        if (
            map_cache_entry is not None
            and map_cache_entry.get("map_features_source") is not map_features_list
        ):
            map_cache_entry = None
        cache_displacement = (
            np.linalg.norm(np.asarray(query_xy) - map_cache_entry["center_xy"])
            if map_cache_entry is not None
            else float("inf")
        )
        map_cache_hit = (
            map_cache_entry is not None
            and cache_displacement + self.radius
            <= float(map_cache_entry["build_radius"])
        )
        if not map_cache_hit:
            scene_center_xy, scene_build_radius = self._get_scene_map_cache_region(
                all_tracks,
                controlled_track_id,
            )
            if (
                np.linalg.norm(np.asarray(query_xy) - scene_center_xy) + self.radius
                > scene_build_radius
            ):
                scene_center_xy = np.asarray(query_xy, dtype=np.float64)
                scene_build_radius = self.map_cache_build_radius

            cached_map, cached_polygon_tokens = self._get_map_features(
                map_features_list=map_features_list,
                query_xy=scene_center_xy,
                route_roadblock_ids=[],
                traffic_light_status=traffic_light_status,
                radius=scene_build_radius,
                present_idx=present_idx,
                lane_graph=lane_graph,
            )
            map_filter_centers = []
            for index, token in enumerate(cached_polygon_tokens):
                raw_feature = map_features_list.get(str(token), {})
                raw_points = raw_feature.get("polyline", None)
                if raw_points is None or len(raw_points) == 0:
                    raw_points = raw_feature.get("polygon", None)
                if raw_points is not None and len(raw_points) > 0:
                    map_filter_centers.append(
                        np.mean(np.asarray(raw_points)[:, :2], axis=0)
                    )
                else:
                    map_filter_centers.append(
                        cached_map["polygon_center"][index, :2]
                    )
            map_filter_centers = np.asarray(
                map_filter_centers,
                dtype=np.float64,
            ).reshape(-1, 2)
            map_cache_entry = {
                "map_features_source": map_features_list,
                "center_xy": np.asarray(scene_center_xy, dtype=np.float64).copy(),
                "build_radius": float(scene_build_radius),
                "filter_centers": map_filter_centers.copy(),
                "map_features": {
                    key: np.array(value, copy=True)
                    for key, value in cached_map.items()
                    if key != "_raw_map_features"
                },
                "map_polygon_tokens": list(cached_polygon_tokens),
            }
            self._map_feature_cache[map_cache_key] = map_cache_entry
            cache_displacement = float(
                np.linalg.norm(np.asarray(query_xy) - scene_center_xy)
            )

        map_filter_centers = map_cache_entry["filter_centers"]
        current_radius_mask = (
            np.linalg.norm(
                map_filter_centers - np.asarray(query_xy)[None, :2],
                axis=1,
            )
            <= self.radius
        )
        num_cached_polygons = len(current_radius_mask)
        data["map"] = {}
        for key, value in map_cache_entry["map_features"].items():
            if (
                isinstance(value, np.ndarray)
                and value.ndim > 0
                and value.shape[0] == num_cached_polygons
            ):
                data["map"][key] = np.array(value[current_radius_mask], copy=True)
            else:
                data["map"][key] = np.array(value, copy=True)
        map_polygon_tokens = [
            token
            for token, keep in zip(
                map_cache_entry["map_polygon_tokens"],
                current_radius_mask,
            )
            if keep
        ]
        num_input_polygons = len(map_polygon_tokens)

        route_id_set = set()
        for route_id in route_lane_seq_for_map:
            mapped_route_id = _as_int_id(route_id)
            if mapped_route_id is not None:
                route_id_set.add(mapped_route_id)
        polygon_road_block_id = data["map"]["polygon_road_block_id"]
        data["map"]["polygon_on_route"] = np.asarray(
            [int(lane_id) in route_id_set for lane_id in polygon_road_block_id],
            dtype=np.bool_,
        )
        data["map"]["polygon_tl_status"] = np.asarray(
            [
                traffic_light_values.get(str(int(lane_id)), 3)
                for lane_id in polygon_road_block_id
            ],
            dtype=np.int8,
        )
        data["map"]["_raw_map_features"] = map_features_list
        map_seconds = time.perf_counter() - map_start

        if not is_validation:
            data["causal"] = self.scenario_casual_reasoning_preprocess(
                ego_features=controlled_features,
                agent_features=agent_features,
                agents_tokens=agent_tokens,
                map_polygon_tokens=map_polygon_tokens,
                map_features=data["map"],
                present_idx=self.history_samples,
            )
            data["causal"]["interaction_label"] = self._get_interaction_label(
                controlled_features, agent_features
            )
            data["agent"]["valid_mask"][0, self.history_samples + 1:] = data["causal"][
                "fixed_ego_future_valid_mask"
            ]

            cost_map_manager = CostMapManager(
                origin=controlled_features["position"][self.history_samples],
                angle=controlled_features["heading"][self.history_samples],
                height=600,
                width=600,
                resolution=0.2
            )
            cost_maps_res = cost_map_manager.build_cost_maps(
                # CostMapManager expects static objects with shape info: [x,y,heading,width,length,cat]
                static_objects=static_objects,
                agents=agent_features,
                map_features_list=map_features_list,
                agents_polygon=agents_polygon,
                traffic_light_status=traffic_light_status,
                present_idx=present_idx,
                future_steps=self.future_samples,
                # Pluto default: consider VEHICLE/PEDESTRIAN/BICYCLE as obstacles
                dynamic_obstacle_types=(
                    self.interested_objects_types.index("VEHICLE"),
                    self.interested_objects_types.index("PEDESTRIAN"),
                    self.interested_objects_types.index("BICYCLE"),
                ),
                dynamic_dilation_radius_m=1.0,
            )
            data["cost_maps"] = cost_maps_res["cost_maps"]

        refline_start = time.perf_counter()
        data["reference_line"] = self._get_reference_line_feature(
            actor_features=controlled_features,
            map_features_list=map_features_list,
            agent_features=agent_features,
            route_roadblock_ids=route_lane_seq,
            lane_graph=lane_graph,
            training=bool(not self.is_validation),
            map_cache_identity=map_id,
            controlled_agent_id=controlled_track_id,
            controlled_track_state=ego_track_state,
            present_idx=present_idx,
        )
        refline_seconds = time.perf_counter() - refline_start

        normalize_start = time.perf_counter()
        result = PlutoFeature.normalize(data, first_time=True, radius=self.radius)

        normalize_seconds = time.perf_counter() - normalize_start
        total_seconds = time.perf_counter() - process_start
        accounted_seconds = (
            route_seconds
            + agent_seconds
            + static_seconds
            + map_seconds
            + refline_seconds
            + normalize_seconds
        )
        self._last_feature_timing = {
            "route_ms": route_seconds * 1000.0,
            # "agent_ms": agent_seconds * 1000.0,
            # "static_ms": static_seconds * 1000.0,
            "map_ms": map_seconds * 1000.0,
            "map_cache_hit": map_cache_hit,
            # "map_cache_displacement_m": (
            #     float(cache_displacement) if map_cache_hit else 0.0
            # ),
            # "map_cached_polygons": num_cached_polygons,
            # "map_input_polygons": num_input_polygons,
            "reference_line_ms": refline_seconds * 1000.0,
            "reference_line_cache_hit": bool(
                getattr(self, "_last_reference_line_cache_hit", False)
            ),
            # "normalize_ms": normalize_seconds * 1000.0,
            # "other_process_ms": max(0.0, total_seconds - accounted_seconds) * 1000.0,
        }
        return result

    def postprocess(self, pluto_feature):
        return [pluto_feature.data]
        # return pluto_feature

    # def __getitem__(self, idx):
    #     # We fetch the dict from parent class HDF5 cache,
    #     # and then wrap it in PlutoFeature at iteration time!
    #     record = super().__getitem__(idx)
    #     # Reconstruct structured dict
    #     data = {
    #         'agent': {
    #             'position': record['agent/position'],
    #             'heading': record['agent/heading'],
    #             'velocity': record['agent/velocity'],
    #             'shape': record['agent/shape'],
    #             'category': record['agent/category'],
    #             'valid_mask': record['agent/valid_mask'],
    #         },
    #         'map': {
    #             'point_position': record['map/point_position'],
    #             'point_vector': record['map/point_vector'],
    #             'polygon_center': record['map/polygon_center'],
    #             'polygon_type': record['map/polygon_type'],
    #             'valid_mask': record['map/valid_mask']
    #         },
    #         'current_state': record['current_state']
    #     }
    #     return PlutoFeature(data=data)

    def parse_tracks_to_states(self, controlled_track):
        """
        Parameters:
            scenario (dict): Contains 'tracks' key, which is a dictionary where keys are object IDs and values are
                dictionaries containing 'state' information.
                'state' contains 'position', 'heading', 'velocity', 'valid', 'length', 'width', 'height' fields.
                Each field is a 2D array of shape (T, ...) where T is the number of time frames.

        Returns:
            ego_state_list (list): List of ego states, one per frame, each element is a dictionary of the ego state.
            tracked_objects_list (list): List of tracked objects states, one per frame, each element is a list of
                dictionaries of the states of all tracked objects.
                Structure: [ [frame_0_ego_state], [frame_1_ego_state], ... ]
                                              and [ [frame_0_obj1_state, frame_0_obj2_state, ...], [frame_1_obj1_state, frame_1_obj2_state, ...], ... ]
        """
        ego_state = controlled_track
        T = len(ego_state['position'])
        ego_state_list = []

        # 逐帧提取状态
        for frame_idx in range(T):
            heading_val = ego_state['heading'][frame_idx].item()
            position_val = ego_state['position'][frame_idx].tolist()
            if self._controlled_actor_is_vehicle():
                position_val = self._shift_center_to_rear_xy(position_val, heading_val).tolist()
            ego_frame_state = {
                'position': position_val,
                'heading': heading_val,
                'velocity': ego_state['velocity'][frame_idx].tolist(),
                'valid': ego_state['valid'][frame_idx].item(),
                'length': ego_state['length'][frame_idx].item(),
                'width': ego_state['width'][frame_idx].item(),
                'height': ego_state['height'][frame_idx].item()
                # 'length': ego_state.get('length', [4.8] * T)[frame_idx].item() if 'length' in ego_state else 4.8,
                # 'width': ego_state.get('width', [1.8] * T)[frame_idx].item() if 'width' in ego_state else 1.8,
                # 'height': ego_state.get('height', [1.5] * T)[frame_idx].item() if 'height' in ego_state else 1.5
            }
            ego_state_list.append(ego_frame_state)
        return ego_state_list

    def _get_rear_axle_to_center(self) -> float:
        if not self._controlled_actor_is_vehicle():
            return 0.0
        params = self.ego_params or {}
        for key in ("rear_axle_to_center", "rear_axle_to_center_dist", "cog_position_from_rear_axle"):
            if key in params:
                return float(params[key])
        return 1.67

    def _shift_center_to_rear_xy(self, position, heading):
        pos = np.asarray(position, dtype=np.float64).copy()
        d = self._get_rear_axle_to_center()
        pos[:2] -= d * np.array([np.cos(heading), np.sin(heading)], dtype=np.float64)
        return pos

    def _build_local_ego_state_for_pluto(self, ego_state):
        ego_state = {k: np.array(v, copy=True) if isinstance(v, np.ndarray) else v for k, v in ego_state.items()}
        if not self._controlled_actor_is_vehicle():
            return ego_state
        pos = np.asarray(ego_state["position"], dtype=np.float64).copy()
        heading = np.asarray(ego_state["heading"], dtype=np.float64)

        d = self._get_rear_axle_to_center()
        offset = np.stack([np.cos(heading), np.sin(heading)], axis=-1) * d
        pos[..., :2] = pos[..., :2] - offset
        ego_state["position"] = pos
        return ego_state

    def _get_ego_current_state(self, ego_state, prev_state):
        state = np.zeros(7, dtype=np.float64)
        state[0:2] = ego_state['position'][:2]
        state[2] = ego_state['heading']
        heading = float(ego_state['heading'])
        heading_vec = np.array([np.cos(heading), np.sin(heading)], dtype=np.float64)
        cur_velocity = np.asarray(ego_state['velocity'][:2], dtype=np.float64)
        longitudinal_velocity = float(np.dot(cur_velocity, heading_vec))

        if 'acceleration' in ego_state:
            cur_acceleration = np.asarray(ego_state['acceleration'][:2], dtype=np.float64)
            longitudinal_acceleration = float(np.dot(cur_acceleration, heading_vec))
        else:
            prev_heading = float(prev_state['heading'])
            prev_heading_vec = np.array([np.cos(prev_heading), np.sin(prev_heading)], dtype=np.float64)
            prev_velocity = np.asarray(prev_state['velocity'][:2], dtype=np.float64)
            prev_longitudinal_velocity = float(np.dot(prev_velocity, prev_heading_vec))
            longitudinal_acceleration = (
                                                longitudinal_velocity - prev_longitudinal_velocity
                                        ) / max(float(self.sample_interval), 1e-6)

        state[3] = longitudinal_velocity
        state[4] = longitudinal_acceleration

        if self._controlled_actor_is_vehicle():
            steering_angle, yaw_rate = calculate_additional_ego_states(
                ego_state, prev_state, self.ego_params
            )
        else:
            angle_diff = float(ego_state['heading'] - prev_state['heading'])
            angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
            steering_angle = 0.0
            yaw_rate = angle_diff / max(float(self.sample_interval), 1e-6)
        state[5] = steering_angle
        state[6] = yaw_rate
        return state

    def _get_agent_features(
            self,
            query_xy,
            present_idx: int,
            all_tracks,
            controlled_agent_id='ego',
    ):
        # Find valid non-ego agents at present_idx
        present_agents = []
        for obj_id, track in all_tracks.items():
            if str(obj_id) == str(controlled_agent_id) or str(
                    track.get("type", "")).upper() not in self.interested_objects_types:
                continue
            state = track['state']
            if state['valid'][present_idx]:
                pos = state['position'][present_idx][:2]
                dist = np.linalg.norm(np.array(pos) - np.array(query_xy))
                present_agents.append((dist, obj_id, track))

        present_agents.sort(key=lambda x: x[0])
        present_agents = present_agents[:self.max_agents]

        N, T = min(len(present_agents), self.max_agents), self.history_samples + 1

        position = np.zeros((N, T, 2), dtype=np.float64)
        heading = np.zeros((N, T), dtype=np.float64)
        velocity = np.zeros((N, T, 2), dtype=np.float64)
        shape = np.zeros((N, T, 2), dtype=np.float64)
        category = np.zeros((N,), dtype=np.int8)
        valid_mask = np.zeros((N, T), dtype=np.bool_)
        polygon = [None] * N

        agent_tokens = []

        if N == 0:
            return (
                {
                    "position": position,
                    "heading": heading,
                    "velocity": velocity,
                    "shape": shape,
                    "category": category,
                    "valid_mask": valid_mask,
                },
                [],
                [],
            )

        for idx, (_dist, obj_id, track) in enumerate(present_agents):
            agent_tokens.append(obj_id)
            track_state = track['state']

            # Minimal, fast path: slice/pad to fixed horizon.
            history_start = max(0, present_idx - self.history_samples)
            end = present_idx + 1

            pos_seq = np.asarray(track_state['position'], dtype=np.float64)[history_start:end, :2]
            vel_seq = np.asarray(track_state['velocity'], dtype=np.float64)[history_start:end, :2]
            hdg_seq = np.asarray(track_state['heading'], dtype=np.float64)[history_start:end]
            vld_seq = np.asarray(track_state['valid'], dtype=bool)[history_start:end]

            # Pad missing history at front with zeros
            T_target = self.history_samples + 1
            pad_len = T_target - len(pos_seq)
            if pad_len > 0:
                pos_seq = np.pad(pos_seq, ((pad_len, 0), (0, 0)), mode='constant')
                vel_seq = np.pad(vel_seq, ((pad_len, 0), (0, 0)), mode='constant')
                hdg_seq = np.pad(hdg_seq, (pad_len, 0), mode='constant')
                vld_seq = np.pad(vld_seq, (pad_len, 0), mode='constant')

            position[idx] = pos_seq
            velocity[idx] = vel_seq
            heading[idx] = hdg_seq
            valid_mask[idx] = vld_seq

            # width/length: keep constant, expand to all timesteps
            w0 = float(np.asarray(track_state.get('width', 0.0)).reshape(-1)[0])
            l0 = float(np.asarray(track_state.get('length', 0.0)).reshape(-1)[0])
            shape[idx, :, 0] = w0
            shape[idx, :, 1] = l0

            track_type = str(track.get('type', '')).upper()
            if track_type in {"VEHICLE", "CAR", "TRUCK", "BUS"}:
                cat = self.interested_objects_types.index("VEHICLE")
            elif track_type in {"PEDESTRIAN", "PEDESTRIAN_ADULT", "PEDESTRIAN_CHILD"}:
                cat = self.interested_objects_types.index("PEDESTRIAN")
            elif track_type in {"BICYCLE", "BICYCLIST", "CYCLIST"}:
                cat = self.interested_objects_types.index("BICYCLE")
            else:
                # fallback: treat unknown dynamic agent as VEHICLE
                cat = self.interested_objects_types.index("VEHICLE")
            category[idx] = cat

            # Build present-time polygon for this agent (used by cost map parked-agent logic).
            center_now = pos_seq[self.history_samples]
            heading_now = float(hdg_seq[self.history_samples])
            poly_xy = _box_corners_xy(center_now, heading_now, float(w0), float(l0))
            polygon[idx] = Polygon(poly_xy)

        agent_features = {
            "position": position,
            "heading": heading,
            "velocity": velocity,
            "shape": shape,
            "category": category,
            "valid_mask": valid_mask,
        }

        return agent_features, agent_tokens, polygon

    def _get_static_objects_features(
            self,
            query_xy,
            present_idx: int,
            all_tracks,
    ):
        static_objects = []
        # dynamic_types = {'VEHICLE', 'PEDESTRIAN', 'BICYCLE'}

        for obj_id, track in all_tracks.items():
            track_type = str(track.get('type', '')).upper()
            if track_type in self.interested_objects_types:
                continue

            state = track['state']
            if not state['valid'][present_idx]:
                continue

            pos = state['position'][present_idx][:2]
            if np.linalg.norm(np.array(pos) - np.array(query_xy)) > self.radius:
                continue

            heading = state['heading'][present_idx]
            length = state['length'][present_idx] if isinstance(state['length'], (list, np.ndarray)) else state[
                'length']
            width = state['width'][present_idx] if isinstance(state['width'], (list, np.ndarray)) else state[
                'width']

            # Map static obstacle type to pluto_feature_builder indices
            if "BARRIER" in track_type or "TRAFFIC_BARRIER" in track_type:
                cat = self.static_objects_types.index("TRAFFIC_BARRIER")
            elif "CONE" in track_type or "TRAFFIC_CONE" in track_type:
                cat = self.static_objects_types.index("TRAFFIC_CONE")
            elif "SIGN" in track_type or "CZONE_SIGN" in track_type:
                cat = self.static_objects_types.index("CZONE_SIGN")
            else:
                cat = self.static_objects_types.index("GENERIC_OBJECT")
            static_objects.append([pos[0], pos[1], heading, float(width), float(length), cat])

        if len(static_objects) > 0:
            static_objects = np.array(static_objects, dtype=np.float64)
            valid_mask = np.ones(len(static_objects), dtype=np.bool_)
        else:
            static_objects = np.zeros((0, 6), dtype=np.float64)
            valid_mask = np.zeros(0, dtype=np.bool_)

        return {
            "position": static_objects[:, :2],
            "heading": static_objects[:, 2],
            "shape": static_objects[:, 3:5],
            "category": static_objects[:, -1].astype(np.int8),
            "valid_mask": valid_mask,
        }, static_objects

    def _get_map_features(
            self,
            map_features_list: dict,
            query_xy,
            route_roadblock_ids: list,
            traffic_light_status,
            radius: float,
            sample_points: int = 20,
            present_idx: int = None,
            lane_graph=None,
    ):
        present_idx_use = present_idx if present_idx is not None else self.history_samples
        route_ids = set(str(route_id) for route_id in route_roadblock_ids)

        # traffic_light_status (SD) is typically a dict keyed by lane_connector_id/lane_id.
        # We convert it to lane_id -> numeric status at present_idx.
        # IMPORTANT: Align with NuPlan TrafficLightStatusType used by Pluto pretrained weights:
        #   GREEN=0, YELLOW=1, RED=2, UNKNOWN=3
        state_mapping = {
            "TRAFFIC_LIGHT_GREEN": 0,
            "TRAFFIC_LIGHT_YELLOW": 1,
            "TRAFFIC_LIGHT_RED": 2,
            "TRAFFIC_LIGHT_UNKNOWN": 3,
        }

        def _tl_to_int(x) -> int:
            if x is None:
                return 3
            s = str(x).upper()
            return int(state_mapping.get(s, 3))

        tls = {}
        for lane_id, tl_info in (traffic_light_status or {}).items():
            state = tl_info.get('state', {}) if isinstance(tl_info.get('state', None), dict) else {}
            obj_state = state.get('object_state', None)
            cur_state = None
            if obj_state is not None and len(obj_state) > present_idx_use:
                cur_state = obj_state[present_idx_use]
            tls[str(lane_id)] = _tl_to_int(cur_state)

        lane_objects = []
        crosswalk_objects = []
        query_pos = query_xy

        for map_id, map_feat in map_features_list.items():
            # Check distance
            polyline = map_feat.get('polyline', None)
            polygon = map_feat.get('polygon', None)

            pts = polyline if polyline is not None and len(polyline) > 0 else polygon
            if pts is None or len(pts) == 0:
                continue

            center = np.mean(pts[:, :2], axis=0)
            if np.linalg.norm(center - query_pos) > radius:
                continue

            obj_type = str(map_feat.get('type', '')).upper()
            if 'CROSSWALK' in obj_type:
                crosswalk_objects.append((map_id, map_feat))
            elif _is_lane_like(obj_type):
                lane_objects.append((map_id, map_feat))

        # Build per-polygon records first to avoid shape/broadcast issues when
        # some polygons are invalid and must be skipped.
        P = sample_points
        rec_point_position = []
        rec_point_vector = []
        rec_point_side = []
        rec_point_orientation = []
        rec_polygon_center = []
        rec_polygon_position = []
        rec_polygon_orientation = []
        rec_polygon_type = []
        rec_polygon_on_route = []
        rec_polygon_tl_status = []
        rec_polygon_speed_limit = []
        rec_polygon_has_speed_limit = []
        rec_polygon_road_block_id = []

        kept_object_ids = []

        for map_id, lane in lane_objects:
            polyline = lane.get('polyline', np.zeros((2, 2)))
            if len(polyline) < 2:
                polyline = np.concatenate([polyline, np.zeros((2 - len(polyline), polyline.shape[-1]))])

            # Sample discrete path from polyline
            centerline = interpolate_polyline(polyline[:, :2], sample_points + 1)

            left_bound = None
            right_bound = None

            polygon = lane.get('polygon', None)
            poly_xy = None
            if polygon is not None:
                try:
                    poly_xy = _sanitize_poly_xy(np.asarray(polygon, dtype=np.float64))
                except Exception:
                    poly_xy = None

                if poly_xy is not None and len(poly_xy) >= 3:
                    left_bound, right_bound = _extract_lane_bounds_from_polygon(
                        centerline, poly_xy, sample_points + 1
                    )

                if left_bound is None or right_bound is None:
                    try:
                        lid = int(map_id) if str(map_id).isdigit() else None
                    except Exception:
                        lid = None

                    if lid is not None and lid in lane_graph.lanes:
                        node = lane_graph.lanes[lid]
                        if node.left_neighbors:
                            nb = node.left_neighbors[0]
                            if nb in lane_graph.lanes:
                                left_bound = interpolate_polyline(lane_graph.lanes[nb].centerline, sample_points + 1)
                        if node.right_neighbors:
                            nb = node.right_neighbors[0]
                            if nb in lane_graph.lanes:
                                right_bound = interpolate_polyline(lane_graph.lanes[nb].centerline, sample_points + 1)

                if left_bound is None or right_bound is None:
                    if poly_xy is not None:
                        d = np.min(np.linalg.norm(poly_xy[None, :, :] - centerline[:, None, :], axis=-1), axis=1)
                        half_w = float(np.clip(np.median(d), 0.5, 6.0))
                        # Build left/right offset using local tangent normals.
                        tang = np.diff(centerline, axis=0, prepend=centerline[0:1])
                        nrm = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
                        nrm_norm = np.linalg.norm(nrm, axis=1, keepdims=True)
                        nrm = nrm / np.clip(nrm_norm, 1e-6, None)
                        if left_bound is None:
                            left_bound = centerline + half_w * nrm
                        if right_bound is None:
                            right_bound = centerline - half_w * nrm

            # DEBUG: plot lane boundaries to check if they are correct
            dist_to_ego = np.min(np.linalg.norm(centerline - np.asarray(query_pos)[None, :], axis=1))
            if dist_to_ego < 20.0 and getattr(self, "_debug_lane_plot_cnt", 0) < 0:
                import matplotlib.pyplot as plt
                plt.figure()
                plt.plot(centerline[:, 0], centerline[:, 1], 'k--', label='centerline')
                plt.plot(left_bound[:, 0], left_bound[:, 1], 'b-', label='left_bound')
                plt.plot(right_bound[:, 0], right_bound[:, 1], 'r-', label='right_bound')
                if poly_xy is not None:
                    # append the first point to close the polygon
                    plot_poly = np.vstack([poly_xy, poly_xy[0]])
                    plt.plot(plot_poly[:, 0], plot_poly[:, 1], 'g:', label='original_poly')
                plt.legend()
                plt.title(f"Lane {map_id} Extraction")
                plt.axis('equal')
                plt.show()
                self._debug_lane_plot_cnt = getattr(self, "_debug_lane_plot_cnt", 0) + 1

            try:
                edges = np.stack([centerline, left_bound, right_bound], axis=0)
            except Exception:
                print('该车道边界有问题')
                continue

            vec = edges[:, 1:] - edges[:, :-1]
            pos = edges[:, :-1]
            ori = np.arctan2(vec[:, :, 1], vec[:, :, 0])

            rec_point_vector.append(vec)
            rec_point_position.append(pos)
            rec_point_orientation.append(ori)
            rec_point_side.append(np.arange(3, dtype=np.int8))

            rec_polygon_center.append(
                np.concatenate(
                    [
                        centerline[int(sample_points / 2)],
                        [ori[0, int(sample_points / 2)]],
                    ],
                    axis=-1,
                )
            )
            rec_polygon_position.append(centerline[0])
            rec_polygon_orientation.append(float(ori[0, 0]))
            # - MetaDriveType.LANE_SURFACE_STREET for ROADBLOCK interior edges (treat as LANE)
            # - MetaDriveType.LANE_SURFACE_UNSTRUCTURE for ROADBLOCK_CONNECTOR interior edges (treat as LANE_CONNECTOR)
            lane_type = str(lane.get('type', '')).upper()
            if "LANE_SURFACE_UNSTRUCTURE" in lane_type:
                rec_polygon_type.append(int(self.polygon_types.index("LANE_CONNECTOR")))
            else:
                rec_polygon_type.append(int(self.polygon_types.index("LANE")))
            rec_polygon_on_route.append(bool(str(map_id) in route_ids))
            rec_polygon_tl_status.append(int(tls.get(str(map_id), 3)))
            rec_polygon_has_speed_limit.append(False)
            rec_polygon_speed_limit.append(0.0)
            rec_polygon_road_block_id.append(int(map_id))

            # if (
            #         bool(str(map_id) in route_ids)
            #         and getattr(self, "_debug_lane_geom_log_cnt", 0) < 8
            # ):
            #     # #region debug-point E:lane-geometry-snapshot
            #     import json, urllib.request;
            #     _p = '.dbg/pluto-turning-drift.env';
            #     _u, _s = 'http://127.0.0.1:7777/event', 'pluto-turning-drift';
            #     exec(
            #         "try:\n with open(_p) as f: c=f.read(); _u=next((l.split('=',1)[1] for l in c.split('\\n') if l.startswith('DEBUG_SERVER_URL=')),_u); _s=next((l.split('=',1)[1] for l in c.split('\\n') if l.startswith('DEBUG_SESSION_ID=')),_s)\nexcept: pass")
            #     _dbg_n = int(min(10, centerline.shape[0]))
            #     _dbg_lane = {
            #         "map_id": str(map_id),
            #         "query_pos": np.asarray(query_pos, dtype=np.float64).tolist(),
            #         "dist_to_ego": float(dist_to_ego),
            #         "on_route": True,
            #         "centerline_first10": np.asarray(centerline[:_dbg_n], dtype=np.float64).tolist(),
            #         "left_bound_first10": np.asarray(left_bound[:_dbg_n], dtype=np.float64).tolist(),
            #         "right_bound_first10": np.asarray(right_bound[:_dbg_n], dtype=np.float64).tolist(),
            #         "center_orientation_first10": np.asarray(ori[0, :min(10, ori.shape[1])], dtype=np.float64).tolist(),
            #         "left_orientation_first10": np.asarray(ori[1, :min(10, ori.shape[1])], dtype=np.float64).tolist(),
            #         "right_orientation_first10": np.asarray(ori[2, :min(10, ori.shape[1])], dtype=np.float64).tolist(),
            #     }
            #     urllib.request.urlopen(urllib.request.Request(_u, data=json.dumps(
            #         {"sessionId": _s, "runId": "pre", "hypothesisId": "E",
            #          "location": "unitraj/datasets/Pluto_dataset/Pluto_dataset.py:_get_map_features",
            #          "msg": "[DEBUG] lane geometry snapshot", "data": _dbg_lane}).encode(),
            #                                                   headers={"Content-Type": "application/json"})).read()
            #     self._debug_lane_geom_log_cnt = getattr(self, "_debug_lane_geom_log_cnt", 0) + 1
            #     # #endregion

            kept_object_ids.append(map_id)

        for map_id, crosswalk in crosswalk_objects:
            polygon = crosswalk.get('polygon', np.zeros((4, 2)))
            if len(polygon) < 3:
                continue

            try:
                from shapely.geometry import Polygon
                import shapely
                poly = Polygon(polygon[:, :2])
                bbox = shapely.minimum_rotated_rectangle(poly)
                coords = np.array(bbox.exterior.coords)
                edge1 = coords[[3, 0]]  # right boundary
                edge2 = coords[[2, 1]]  # left boundary

                edges_geom = np.stack([(edge1 + edge2) * 0.5, edge2, edge1], axis=0)  # [3, 2, 2]
                v = edges_geom[:, 1] - edges_geom[:, 0]  # [3, 2]
                steps = np.linspace(0, 1, sample_points + 1, endpoint=True)[None, :]
                edges = edges_geom[:, 0][:, None, :] + v[:, None, :] * steps[:, :, None]  # [3, P+1, 2]

                # DEBUG: plot crosswalk edges
                dist_to_ego = np.min(np.linalg.norm(edges[0] - np.asarray(query_pos)[None, :], axis=1))
                if dist_to_ego < 20.0 and getattr(self, "_debug_cw_plot_cnt", 0) < 0:
                    import matplotlib.pyplot as plt
                    plt.figure()
                    plt.plot(edges[0, :, 0], edges[0, :, 1], 'k--', label='center')
                    plt.plot(edges[1, :, 0], edges[1, :, 1], 'b-', label='left_edge (edge2)')
                    plt.plot(edges[2, :, 0], edges[2, :, 1], 'r-', label='right_edge (edge1)')
                    plot_poly = np.vstack([polygon[:, :2], polygon[0, :2]])
                    plt.plot(plot_poly[:, 0], plot_poly[:, 1], 'g:', label='original_poly')
                    plt.legend()
                    plt.title(f"Crosswalk {map_id} Extraction")
                    plt.axis('equal')
                    plt.show()
                    self._debug_cw_plot_cnt = getattr(self, "_debug_cw_plot_cnt", 0) + 1

            except Exception as e:
                edges = np.tile(interpolate_polyline(polygon[:, :2], sample_points + 1)[None, :], (3, 1, 1))

            vec = edges[:, 1:] - edges[:, :-1]
            pos = edges[:, :-1]
            ori = np.arctan2(vec[:, :, 1], vec[:, :, 0])

            rec_point_vector.append(vec)
            rec_point_position.append(pos)
            rec_point_orientation.append(ori)
            rec_point_side.append(np.arange(3, dtype=np.int8))
            rec_polygon_center.append(
                np.concatenate(
                    [
                        edges[0, int(sample_points / 2)],
                        [ori[0, int(sample_points / 2)]],
                    ],
                    axis=-1,
                )
            )
            rec_polygon_position.append(edges[0, 0])
            rec_polygon_orientation.append(float(ori[0, 0]))
            rec_polygon_type.append(int(self.polygon_types.index("CROSSWALK")))
            rec_polygon_on_route.append(False)
            rec_polygon_tl_status.append(3)
            rec_polygon_has_speed_limit.append(False)
            rec_polygon_speed_limit.append(0.0)
            rec_polygon_road_block_id.append(int(map_id))

            kept_object_ids.append(map_id)

        # Stack records
        M = len(rec_point_position)
        point_position = np.stack(rec_point_position, axis=0) if M > 0 else np.zeros((0, 3, P, 2), dtype=np.float64)
        point_vector = np.stack(rec_point_vector, axis=0) if M > 0 else np.zeros((0, 3, P, 2), dtype=np.float64)
        point_orientation = np.stack(rec_point_orientation, axis=0) if M > 0 else np.zeros((0, 3), dtype=np.float64)
        point_side = np.stack(rec_point_side, axis=0) if M > 0 else np.zeros((0, 3), dtype=np.int8)
        polygon_center = np.stack(rec_polygon_center, axis=0) if M > 0 else np.zeros((0, 3), dtype=np.float64)
        polygon_position = np.stack(rec_polygon_position, axis=0) if M > 0 else np.zeros((0, 2), dtype=np.float64)
        polygon_orientation = np.asarray(rec_polygon_orientation, dtype=np.float64) if M > 0 else np.zeros((0,),
                                                                                                           dtype=np.float64)
        polygon_type = np.asarray(rec_polygon_type, dtype=np.int8) if M > 0 else np.zeros((0,), dtype=np.int8)
        polygon_on_route = np.asarray(rec_polygon_on_route, dtype=np.bool_) if M > 0 else np.zeros((0,), dtype=np.bool_)
        polygon_tl_status = np.asarray(rec_polygon_tl_status, dtype=np.int8) if M > 0 else np.zeros((0,), dtype=np.int8)
        polygon_speed_limit = np.asarray(rec_polygon_speed_limit, dtype=np.float64) if M > 0 else np.zeros((0,),
                                                                                                           dtype=np.float64)
        polygon_has_speed_limit = np.asarray(rec_polygon_has_speed_limit, dtype=np.bool_) if M > 0 else np.zeros((0,),
                                                                                                                 dtype=np.bool_)
        polygon_road_block_id = np.asarray(rec_polygon_road_block_id, dtype=np.int32) if M > 0 else np.zeros((0,),
                                                                                                             dtype=np.int32)

        object_ids = kept_object_ids

        map_features = {
            "point_position": point_position,
            "point_vector": point_vector,
            "point_orientation": point_orientation,
            "point_side": point_side,
            "polygon_center": polygon_center,
            "polygon_position": polygon_position,
            "polygon_orientation": polygon_orientation,
            "polygon_type": polygon_type,
            "polygon_on_route": polygon_on_route,
            "polygon_tl_status": polygon_tl_status,
            "polygon_has_speed_limit": polygon_has_speed_limit,
            "polygon_speed_limit": polygon_speed_limit,
            "polygon_road_block_id": polygon_road_block_id,
            # allow causal reasoning to access raw polygons by map_id
            "_raw_map_features": map_features_list,
        }

        return map_features, object_ids

    def scenario_casual_reasoning_preprocess(
            self,
            ego_features,
            agent_features,
            agents_tokens,
            map_polygon_tokens,
            map_features=None,
            present_idx=None,
    ):
        """Heuristic causal reasoning without nuPlan ScenarioManager.

        找到自车前方最近的动态车辆，识别对自车有影响的红灯区域，计算自车在考虑前方障碍物和红灯后的可行驶路径，判断自车未来轨迹是否会进入红灯区域

        Limitations vs nuplan:
        - no drivable-area query
        - no lane-graph-based leading-object inference
        - no precise stop-line occupancy; we use map polygon containment as proxy
        """
        if present_idx is None:
            present_idx = self.history_samples

        num_agents = len(agents_tokens)
        num_maps = len(map_polygon_tokens)
        T_future = self.future_samples

        leading_agent_mask = np.zeros(num_agents, dtype=bool)
        leading_distance = np.zeros(num_agents, dtype=np.float64)
        ego_care_red_light_mask = np.zeros(num_maps, dtype=bool)
        fixed_ego_future_valid_mask = np.ones(T_future, dtype=bool)

        # present ego pose
        ego_pos = ego_features["position"][present_idx]
        ego_heading = float(ego_features["heading"][present_idx])
        ego_dir = np.array([np.cos(ego_heading), np.sin(ego_heading)], dtype=np.float64)

        # --- leading dynamic agents (forward cone in ego heading) ---
        nearest_leading_agent_idx = None
        nearest_leading_agent_dist = None

        if agent_features is not None and agent_features["position"].shape[0] > 0:
            # agent_features are non-ego agents only, aligned with agents_tokens[1:]
            pos_now = agent_features["position"][:, present_idx]
            valid_now = agent_features["valid_mask"][:, present_idx]
            rel = pos_now - ego_pos[None, :]
            forward = rel @ ego_dir  # projection
            lateral = np.abs(rel[:, 0] * (-ego_dir[1]) + rel[:, 1] * ego_dir[0])

            # candidates in front within lateral band
            cand = valid_now & (forward > 0.0) & (lateral < 3.5)
            if cand.any():
                # sort by forward distance
                order = np.argsort(forward + (~cand) * 1e6)
                for j in order:
                    if not cand[j]:
                        continue
                    # mark this as leading
                    idx_token = j + 1  # shift because ego at 0 in agents_tokens
                    leading_agent_mask[idx_token] = True
                    leading_distance[idx_token] = float(forward[j])
                    if nearest_leading_agent_idx is None:
                        nearest_leading_agent_idx = idx_token
                        nearest_leading_agent_dist = float(forward[j])
                    # also mark other close-in-front agents as leading
                    if forward[j] < 30.0:
                        continue
                    break

        # --- red light polygons ---
        nearest_red_poly = None
        nearest_red_poly_dist = None

        def _is_red(x):
            # numeric encoding: RED==2,UNKNOWN==3
            if isinstance(x, (np.integer, int)):
                return int(x) == 2
            if isinstance(x, (bytes, str)):
                return "RED" in str(x).upper()
            return False

        if map_features is not None and "polygon_tl_status" in map_features:
            tl_status = map_features["polygon_tl_status"]
            poly_pos = map_features.get("polygon_position", None)
            for i in range(min(num_maps, len(tl_status))):
                if _is_red(tl_status[i]):
                    ego_care_red_light_mask[i] = True
                    if poly_pos is not None:
                        d = float(np.linalg.norm(poly_pos[i] - ego_pos))
                        if nearest_red_poly_dist is None or d < nearest_red_poly_dist:
                            nearest_red_poly_dist = d
                            nearest_red_poly = i

        # nuplan builder: "waiting for red light without lead" means nearest leading object is red light
        is_waiting_for_red_light_without_lead = bool(nearest_red_poly is not None and nearest_leading_agent_idx is None)

        # future valid mask: if ego future enters nearest red polygon, invalidate remaining future
        if nearest_red_poly is not None and map_features is not None:
            token = map_polygon_tokens[nearest_red_poly] if nearest_red_poly < len(map_polygon_tokens) else None
            poly = None
            raw_map = map_features.get("_raw_map_features", None)
            if isinstance(raw_map, dict) and token is not None:
                raw = raw_map.get(str(token), raw_map.get(token))
                if isinstance(raw, dict):
                    poly = raw.get("polygon", None)

            if poly is not None and len(poly) >= 3:
                shp = Polygon(poly[:, :2])
                future_pos = ego_features["position"][present_idx + 1: present_idx + 1 + T_future][:, :2]
                for i in range(min(T_future, len(future_pos))):
                    pt = future_pos[i]
                    if shp.contains(Polygon([pt, pt, pt]).centroid):
                        fixed_ego_future_valid_mask[i:] = False
                        break

        # free path points along ego heading, with end limited by nearest lead and red light
        ego_speed = float(np.linalg.norm(ego_features["velocity"][present_idx])) if "velocity" in ego_features else 0.0
        free_path_start = ego_speed ** 2 / (2 * 5.0) + 2.5
        free_path_end = max(7.0, ego_speed ** 2 / (2 * 1.5))
        if nearest_leading_agent_dist is not None:
            free_path_end = min(free_path_end, nearest_leading_agent_dist)
        if nearest_red_poly_dist is not None:
            free_path_end = min(free_path_end, nearest_red_poly_dist)

        if free_path_end <= free_path_start:
            free_path_points = np.zeros((0, 3), dtype=np.float64)
            free_path_points_angle = np.zeros((0,), dtype=np.float64)
        else:
            n = max(int((free_path_end - free_path_start) / 1.0), 2)
            ds = np.linspace(free_path_start + 3.0, max(free_path_start + 3.0, free_path_end - 3.0), n)
            points = (ego_pos[None, :] + ds[:, None] * ego_dir[None, :]).astype(np.float64)
            headings = np.full((points.shape[0], 1), ego_heading, dtype=np.float64)
            free_path_points = np.hstack([points, headings])

        return {
            "is_waiting_for_red_light_without_lead": is_waiting_for_red_light_without_lead,
            "leading_agent_mask": leading_agent_mask,
            "leading_distance": leading_distance,
            "ego_care_red_light_mask": ego_care_red_light_mask,
            "fixed_ego_future_valid_mask": fixed_ego_future_valid_mask,
            "free_path_points": free_path_points,
        }

    def _get_interaction_label(self, ego, agents):
        """Compute interaction label between ego and each agent.

        This is a dependency-free approximation of PlutoFeatureBuilder's interaction label.
        We:
        1) Find nearest ego-agent distance over future horizon (t > history_samples)
        2) If below threshold and boxes intersect, mark as interaction
        3) Label is time difference (ego_t - agent_t) in the closest pair, clipped, with 0 meaning no interaction.

        Output shape matches builder usage: (N_agents + 1,) including ego at index 0.
        """
        start = self.history_samples + 1
        ego_heading = ego["heading"][start:]
        ego_position = ego["position"][start:]

        agents_position = agents["position"][:, start:]
        agents_heading = agents["heading"][:, start:]
        agents_shape = agents["shape"][:, start:]
        agents_valid = agents["valid_mask"][:, start:]

        if agents_position.shape[0] == 0 or agents_position.shape[1] == 0 or ego_position.shape[0] == 0:
            return np.zeros(1, dtype=np.int64)

        N, T = agents_position.shape[:2]
        Te = ego_position.shape[0]

        # pairwise distances for each agent across time pairs (agent_t, ego_t)
        # Build (N, T, Te) distances in numpy for speed/memory constraints: do incremental min
        min_dist = np.full((N,), 1e9, dtype=np.float64)
        min_idx = np.full((N,), -1, dtype=np.int64)

        for i in range(N):
            if not agents_valid[i].any():
                continue
            # restrict to valid timesteps
            valid_ts = np.where(agents_valid[i])[0]
            if len(valid_ts) == 0:
                continue
            # compute cdist for valid agent steps to all ego steps
            a = agents_position[i, valid_ts]  # (Tv,2)
            e = ego_position  # (Te,2)
            d = np.linalg.norm(a[:, None, :] - e[None, :, :], axis=-1)  # (Tv,Te)
            flat = d.reshape(-1)
            j = int(flat.argmin())
            md = float(flat[j])
            if md < min_dist[i]:
                min_dist[i] = md
                # encode back to agent_t, ego_t in full-T coordinates
                agent_t = int(valid_ts[j // Te])
                ego_t = int(j % Te)
                min_idx[i] = agent_t * Te + ego_t

        interact_flag = min_dist < 4.0

        # collision check with oriented boxes
        for i in range(N):
            if not interact_flag[i] or min_idx[i] < 0:
                continue
            agent_t = int(min_idx[i] // Te)
            ego_t = int(min_idx[i] % Te)
            agent_shape = agents_shape[i, agent_t]
            agent_box = self._build_agent_bbox(
                agents_position[i, agent_t],
                agents_heading[i, agent_t],
                float(agent_shape[0]),
                float(agent_shape[1]),
            )
            ego_box = self._build_ego_bbox(ego_position[ego_t], float(ego_heading[ego_t]))
            if not agent_box.intersects(ego_box):
                interact_flag[i] = False

        # label
        interact_label = np.zeros((N,), dtype=np.int64)
        for i in range(N):
            if not interact_flag[i] or min_idx[i] < 0:
                continue
            agent_t = int(min_idx[i] // Te)
            ego_t = int(min_idx[i] % Te)
            interact_label[i] = ego_t - agent_t

        # prepend ego
        return np.concatenate([np.zeros(1, dtype=np.int64), interact_label])

    @staticmethod
    def _get_interact_type(index, T=80):
        row, col = index // T, index % T
        if row == col:
            return 0  # collision or self
        return col - row

    def _build_agent_bbox(self, xy, angle, width, length):
        dx = length / 2
        dy = width / 2
        corners = np.array([
            [dx, dy], [-dx, dy], [-dx, -dy], [dx, -dy]
        ])
        rot = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        center = xy[:2] if (isinstance(xy, np.ndarray) and xy.shape[-1] >= 2) else xy
        return Polygon(center + corners @ rot.T)

    def _build_ego_bbox(self, xy, angle):
        center = xy + 1.67 * np.array([np.cos(angle), np.sin(angle)])
        width = getattr(self, 'width', 2.0)
        length = getattr(self, 'length', 5.0)
        return self._build_agent_bbox(center, angle, width, length)

    def _get_ego_head_position(self, xy, angle):
        ego_len = getattr(self, 'length', 5.0)
        return xy + ego_len * np.array([np.cos(angle), np.sin(angle)]) / 2

    def _reference_lines_to_feature(
            self,
            reference_lines,
            reference_line_route_ids,
            actor_features,
            actor_pos,
            actor_heading,
            lines_are_cropped=False,
    ):
        n_points = int(self.radius)
        num_lines = len(reference_lines)

        position = np.zeros((num_lines, n_points, 2))
        vector = np.zeros((num_lines, n_points, 2))
        orientation = np.zeros((num_lines, n_points))
        valid_mask = np.zeros((num_lines, n_points), dtype=bool)
        future_projection = np.zeros((num_lines, 8, 2))

        route_id = np.full((num_lines,), -1, dtype=np.int64)
        if len(reference_line_route_ids) == num_lines:
            route_id[:] = np.asarray(reference_line_route_ids, dtype=np.int64)

        actor_future = actor_features["position"][self.history_samples + 1:]
        future_samples = np.zeros((0, 2), dtype=np.float64)
        if len(actor_future) > 0:
            future_samples = np.asarray(actor_future[9::10][:8, :2], dtype=np.float64)

        for i, line in enumerate(reference_lines):
            line = np.asarray(line, dtype=np.float64)
            cropped_line = (
                line
                if lines_are_cropped
                else _crop_reference_line_from_ego(
                    line, actor_pos[:2], actor_heading
                )
            )
            subsample = _resample_reference_line(
                cropped_line,
                n_points + 1,
                actor_heading,
                sample_spacing=1.0,
            )
            n_valid = len(subsample)

            position[i, : n_valid - 1] = subsample[:-1, :2]
            vector[i, : n_valid - 1] = np.diff(subsample[:, :2], axis=0)

            # Always derive heading from the final resampled XY geometry.
            # Stored headings become invalid whenever a segment is reversed or
            # several lane/connector segments are concatenated.
            orientation[i, : n_valid - 1] = np.arctan2(
                vector[i, : n_valid - 1, 1],
                vector[i, : n_valid - 1, 0],
            )
            valid_mask[i, : n_valid - 1] = True

            if 0 < n_valid - 1 < n_points:
                position[i, n_valid - 1:] = position[i, n_valid - 2]
                vector[i, n_valid - 1:] = vector[i, n_valid - 2]
                orientation[i, n_valid - 1:] = orientation[i, n_valid - 2]

            if len(actor_future) > 0:
                along, distance, _, _ = project_points_to_polyline(
                    future_samples,
                    cropped_line[:, :2],
                )
                future_projection[i, :len(future_samples), 0] = along
                future_projection[i, :len(future_samples), 1] = distance

        return {
            "position": position,
            "vector": vector,
            "orientation": orientation,
            "valid_mask": valid_mask,
            "future_projection": future_projection,
            "route_id": route_id,
        }

    def _get_reference_line_feature(
            self,
            actor_features=None,
            map_features_list=None,
            agent_features=None,
            route_roadblock_ids=None,
            lane_graph=None,
            training=False,
            map_cache_identity=None,
            controlled_agent_id=None,
            controlled_track_state=None,
            present_idx=None,
    ):
        actor_pos = actor_features["position"][-1]
        actor_heading = actor_features["heading"][-1]
        actor_speed = float(np.linalg.norm(actor_features["velocity"][-1]))
        debug_refline = os.environ.get("UNITRAJ_REF_LINE_DEBUG", "0") not in ("0", "false", "False")

        search_radius = self.radius
        build_radius = self.reference_line_build_radius

        # 路由ID转换
        route_seq = []
        for x in (route_roadblock_ids or []):
            mapped_x = _as_int_id(x)
            if mapped_x is not None:
                route_seq.append(mapped_x)

        # 补齐 route 中的拓扑缺口，BFS 最短路径连接
        # 如 [63367, 67341] → [63367, 63368, 69003, 67341]
        if lane_graph is not None and len(route_seq) > 1:
            def _route_bfs(start, goal, max_depth=10):
                if start == goal:
                    return [start]
                from collections import deque
                queue = deque([[start]])
                visited = {start}
                while queue:
                    path = queue.popleft()
                    cur = path[-1]
                    if len(path) > max_depth:
                        continue
                    node = lane_graph.lanes.get(cur)
                    if node is None:
                        continue
                    candidates = (list(getattr(node, 'successors', []) or []) +
                                  list(getattr(node, 'exit', []) or []) +
                                  list(node.left_neighbors or []) +
                                  list(node.right_neighbors or []))
                    for nxt in candidates:
                        if nxt in visited:
                            continue
                        new_path = path + [nxt]
                        if nxt == goal:
                            return new_path
                        visited.add(nxt)
                        queue.append(new_path)
                return None

            filled = [route_seq[0]]
            for i in range(1, len(route_seq)):
                prev = filled[-1]
                nxt = route_seq[i]
                gap_path = _route_bfs(prev, nxt)
                if gap_path and len(gap_path) > 1:
                    filled.extend(gap_path[1:])
                else:
                    filled.append(nxt)
            route_seq = filled

        def wrap_to_pi(angle):
            return (angle + np.pi) % (2 * np.pi) - np.pi

        original_route_seq = tuple(route_seq)
        # For controlled background agents/CBVs, do not anchor the start lane
        # to the possibly distant terminal route lane. By default they rebuild
        # from nearby lanes after closed-loop lane changes; route reachability
        # can be restored with cbv_reference_line_require_route_reachability.
        use_route_constraint = not (
            controlled_agent_id is not None
            and str(controlled_agent_id) != "ego"
        )
        route_seq_for_search = route_seq
        cbv_require_route_reachability = bool(
            getattr(self, "cbv_reference_line_require_route_reachability", False)
        )
        require_route_reachability = bool(route_seq) and (
            controlled_agent_id is None
            or str(controlled_agent_id) == "ego"
            or cbv_require_route_reachability
        )
        is_ego_actor = controlled_agent_id is None or str(controlled_agent_id) == "ego"
        strict_route_refline = bool(self.strict_route_refline)
        enable_route_lateral_refline = bool(self.enable_route_lateral_refline)
        if is_ego_actor:
            if self.ego_strict_route_refline is not None:
                strict_route_refline = bool(self.ego_strict_route_refline)
            if self.ego_enable_route_lateral_refline is not None:
                enable_route_lateral_refline = bool(self.ego_enable_route_lateral_refline)

        def _build_ego_expert_reference_line():
            if (
                not is_ego_actor
                or not bool(getattr(self, "ego_expert_reference_line_enabled", False))
                or controlled_track_state is None
            ):
                return None
            state = controlled_track_state
            pos = np.asarray(state.get("position", []), dtype=np.float64)
            if pos.ndim != 2 or pos.shape[0] < 2 or pos.shape[-1] < 2:
                return None
            start = int(self.history_samples if present_idx is None else present_idx)
            if start < 0 or start >= len(pos) - 1:
                return None
            end = min(len(pos), start + int(self.future_samples) + 1)
            xy = np.asarray(pos[start:end, :2], dtype=np.float64)
            valid = np.asarray(state.get("valid", np.ones((len(pos),), dtype=np.bool_)), dtype=np.bool_).reshape(-1)
            if len(valid) >= end:
                xy = xy[valid[start:end]]
            finite = np.isfinite(xy).all(axis=1)
            xy = xy[finite]
            if len(xy) < 2:
                return None

            filtered = [xy[0]]
            for point in xy[1:]:
                if float(np.linalg.norm(point - filtered[-1])) >= 0.2:
                    filtered.append(point)
            xy = np.asarray(filtered, dtype=np.float64)
            if len(xy) < 2:
                return None
            length = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))
            if length < float(getattr(self, "ego_expert_reference_line_min_length", 10.0)):
                return None

            return _line_with_heading(xy)

        def _prepend_or_append_expert_reference_line(lines, ids):
            expert_line = _build_ego_expert_reference_line()
            if expert_line is None:
                return lines, ids
            expert_id = -3000
            if bool(getattr(self, "ego_expert_reference_line_first", True)):
                merged_lines = []
                merged_ids = []
                _dedup_append(merged_lines, merged_ids, expert_line, expert_id, eps=0.5)
                for line, lid in zip(lines, ids):
                    _dedup_append(merged_lines, merged_ids, line, int(lid), eps=0.5)
                if debug_refline:
                    print(
                        "[RefLineDebug] added ego expert reference line",
                        {
                            "controlled_agent_id": str(controlled_agent_id),
                            "line_points": int(len(expert_line)),
                            "line_length_m": round(float(np.sum(np.linalg.norm(np.diff(expert_line[:, :2], axis=0), axis=1))), 3),
                            "first": True,
                            "reference_lines": len(merged_lines),
                        },
                    )
                return merged_lines, merged_ids
            _dedup_append(lines, ids, expert_line, expert_id, eps=0.5)
            if debug_refline:
                print(
                    "[RefLineDebug] added ego expert reference line",
                    {
                        "controlled_agent_id": str(controlled_agent_id),
                        "line_points": int(len(expert_line)),
                        "line_length_m": round(float(np.sum(np.linalg.norm(np.diff(expert_line[:, :2], axis=0), axis=1))), 3),
                        "first": False,
                        "reference_lines": len(lines),
                    },
                )
            return lines, ids
        route_goal_set = set(route_seq)
        route_index_by_id = {}
        for idx, lid in enumerate(route_seq_for_search):
            route_index_by_id.setdefault(int(lid), int(idx))
        cache_map_key = (
            str(map_cache_identity)
            if map_cache_identity is not None
            else id(lane_graph)
        )
        reference_line_cache_key = (cache_map_key, str(controlled_agent_id))
        cache_entry = self._reference_line_cache.get(reference_line_cache_key)
        cache_hit = (
            cache_entry is not None
            and tuple(cache_entry["route_seq"]) == tuple(route_seq_for_search)
            and bool(cache_entry["training"]) == bool(training)
            and not (
                is_ego_actor
                and bool(getattr(self, "ego_expert_reference_line_enabled", False))
            )
        )
        if cache_hit:
            cached_lines = cache_entry["reference_lines"]
            cached_ids = cache_entry["reference_line_route_ids"]
            usable_cached_lines = []
            usable_cached_ids = []
            for line, route_id in zip(cached_lines, cached_ids):
                line = np.asarray(line, dtype=np.float64)
                along, lateral_distance, seg_idx, _ = project_points_to_polyline(
                    actor_pos[None, :2],
                    line[:, :2],
                )
                segment = line[min(int(seg_idx[0]), len(line) - 2) + 1, :2] - line[
                    min(int(seg_idx[0]), len(line) - 2), :2
                ]
                segment_heading = float(np.arctan2(segment[1], segment[0]))
                total_length = float(
                    np.sum(np.linalg.norm(np.diff(line[:, :2], axis=0), axis=1))
                )
                remaining_length = total_length - float(along[0])
                if (
                    float(lateral_distance[0])
                    > self.reference_line_cache_lateral_distance
                    or abs(wrap_to_pi(segment_heading - actor_heading))
                    > self.reference_line_cache_heading_threshold
                    or remaining_length < self.reference_line_cache_min_remaining
                ):
                    continue
                cropped_line = _crop_reference_line_from_ego(
                    line,
                    actor_pos[:2],
                    actor_heading,
                )
                if (
                    cropped_line.ndim == 2
                    and cropped_line.shape[0] >= 2
                    and np.sum(
                        np.linalg.norm(
                            np.diff(cropped_line[:, :2], axis=0),
                            axis=1,
                        )
                    ) > 1.0
                ):
                    usable_cached_lines.append(cropped_line)
                    usable_cached_ids.append(route_id)
            if usable_cached_lines:
                self._last_reference_line_cache_hit = True
                if debug_refline and require_route_reachability:
                    print(
                        "[RefLineDebug] cache hit route-reachable reference lanes",
                        {
                            "controlled_agent_id": str(controlled_agent_id),
                            "actor_xy": np.round(actor_pos[:2], 3).tolist(),
                            "raw_route_head": list(route_seq[:8]),
                            "cached_route_ids": [int(x) for x in usable_cached_ids[:8]],
                            "cached_lines": len(usable_cached_lines),
                        },
                    )
                return self._reference_lines_to_feature(
                    usable_cached_lines,
                    usable_cached_ids,
                    actor_features,
                    actor_pos,
                    actor_heading,
                    lines_are_cropped=True,
                )

        self._last_reference_line_cache_hit = False

        # Search start lanes locally; only successor expansion uses the longer
        # build radius. This keeps cache misses bounded on long routes.
        # If route is available, restrict to route lanes (plus a small neighbor set) to reduce branching.
        lane_ids = []
        lane_polys = {}
        lane_xyz = {}
        lane_scan_stats = {
            "iter_lanes": 0,
            "missing_centerline": 0,
            "outside_search_radius": 0,
            "nearby_lanes": 0,
            "candidate_dist_reject": 0,
            "candidate_heading_reject": 0,
            "candidate_backward_reject": 0,
            "candidate_bad_tangent": 0,
            "candidate_route_unreachable": 0,
        }
        nearest_lane_samples = []
        candidate_reject_samples = []
        candidate_paths_to_route = {}
        from typing import Set
        neighbor_extra: Set[int] = set()
        enable_lateral_refline = (
            bool(enable_route_lateral_refline)
            and self._controlled_actor_is_vehicle()
        )

        if use_route_constraint and route_seq_for_search and enable_lateral_refline:
            expanded_route_seq = expand_lateral_lane_ids(
                route_seq_for_search,
                lane_graph,
                lateral_hops=getattr(self, "route_lateral_refline_hops", 2),
                same_direction_only=getattr(self, "route_lateral_same_direction_only", True),
                heading_threshold=getattr(self, "route_lateral_heading_threshold", np.pi / 3),
            )
            neighbor_extra.update(lid for lid in expanded_route_seq if lid not in route_seq_for_search)
        elif use_route_constraint and route_seq_for_search and not strict_route_refline:
            for lid in route_seq_for_search:
                node = lane_graph.lanes.get(lid)
                if node is None:
                    continue
                neighbor_extra.update(node.left_neighbors)
                neighbor_extra.update(node.right_neighbors)

        allowed_lanes = None
        if use_route_constraint and route_seq_for_search:
            # Keep a deterministic iteration order to avoid subtle randomness in candidate selection.
            if strict_route_refline and not enable_lateral_refline:
                allowed_lanes = set(route_seq_for_search)
            else:
                allowed_lanes = set(route_seq_for_search) | neighbor_extra # 不严格限制贴ref_line，路由车道 + 邻居车道
            allowed_lanes_ordered = []
            seen = set()
            for lid in route_seq_for_search:
                if lid in allowed_lanes and lid not in seen:
                    allowed_lanes_ordered.append(lid)
                    seen.add(lid)
            for lid in sorted(allowed_lanes):
                if lid not in seen:
                    allowed_lanes_ordered.append(lid)
                    seen.add(lid)
        else:
            allowed_lanes_ordered = None

        iter_lids = allowed_lanes_ordered if allowed_lanes_ordered is not None else list(lane_graph.lanes.keys())
        for lid in iter_lids:
            lane_scan_stats["iter_lanes"] += 1
            node = lane_graph.lanes.get(lid)
            if node is None:
                lane_scan_stats["missing_centerline"] += 1
                continue
            cl = node.centerline
            if cl is None or len(cl) < 2:
                lane_scan_stats["missing_centerline"] += 1
                continue
            # 距离过滤
            min_dist = float(np.min(np.linalg.norm(cl - actor_pos[None, :2], axis=1)))
            if len(nearest_lane_samples) < 8:
                nearest_lane_samples.append((float(min_dist), int(lid)))
            else:
                worst_idx = max(range(len(nearest_lane_samples)), key=lambda i: nearest_lane_samples[i][0])
                if min_dist < nearest_lane_samples[worst_idx][0]:
                    nearest_lane_samples[worst_idx] = (float(min_dist), int(lid))
            if min_dist > search_radius:
                lane_scan_stats["outside_search_radius"] += 1
                continue
            lane_scan_stats["nearby_lanes"] += 1
            lane_ids.append(lid)
            lane_polys[lid] = cl
            lane_xyz[lid] = node

        # 候选起始车道选择
        candidates: List[Tuple[int, float]] = []
        ego_dir = np.array([np.cos(actor_heading), np.sin(actor_heading)], dtype=np.float64)

        allow_neighbor_refs = bool(training) or enable_lateral_refline
        if use_route_constraint and route_seq_for_search and agent_features is not None and agent_features.get("position", None) is not None:
            try:
                pos_now = np.asarray(agent_features["position"][:, -1], dtype=np.float64)
                valid_now = np.asarray(agent_features["valid_mask"][:, -1], dtype=np.bool_)
                vel_now = (
                    np.asarray(agent_features["velocity"][:, -1], dtype=np.float64)
                    if "velocity" in agent_features and agent_features["velocity"].shape[0] > 0
                    else np.zeros((len(pos_now), 2), dtype=np.float64)
                )
                if len(pos_now) > 0:
                    rel = pos_now - actor_pos[None, :2]
                    forward = rel @ ego_dir
                    lateral = np.abs(rel[:, 0] * (-ego_dir[1]) + rel[:, 1] * ego_dir[0])
                    speed_now = np.linalg.norm(vel_now[..., :2], axis=-1)
                    blocking = valid_now & (forward > 0.0) & (forward < 12.0) & (lateral < 3.5)
                    if blocking.any():
                        nearest = int(np.argmin(np.where(blocking, forward, 1e6)))
                        speed_threshold = max(2.0, actor_speed * 0.35)
                        allow_neighbor_refs = enable_lateral_refline or bool(speed_now[nearest] < speed_threshold)
            except Exception:
                allow_neighbor_refs = bool(training) or enable_lateral_refline

        # 车道必须同时满足距离近、朝向匹配、在前方三个条件才能成为候选
        DIST_TH = 15.0
        HEADING_TH = 1.05
        FORWARD_TH = -2.0

        def _path_to_route(start_lid, max_depth=40):
            """Return a successor path that merges into and follows the raw route."""
            if not require_route_reachability:
                return None
            if start_lid not in lane_graph.lanes:
                return None
            if start_lid in route_index_by_id:
                return list(route_seq_for_search[route_index_by_id[start_lid]:])
            from collections import deque
            queue = deque([[start_lid]])
            visited = {start_lid}
            while queue:
                path = queue.popleft()
                if len(path) > max_depth:
                    continue
                cur = path[-1]
                succs = [s for s in lane_graph.successors(cur) if s in lane_graph.lanes]
                for nxt in succs:
                    if nxt in visited:
                        continue
                    next_path = path + [nxt]
                    if nxt in route_index_by_id:
                        return next_path + list(route_seq_for_search[route_index_by_id[nxt] + 1:])
                    visited.add(nxt)
                    queue.append(next_path)
            return None

        for lid in lane_ids:
            poly = lane_polys[lid]
            dists = np.linalg.norm(poly - actor_pos[None, :2], axis=1)
            k = int(np.argmin(dists))
            dist = float(dists[k])
            if dist > DIST_TH:
                lane_scan_stats["candidate_dist_reject"] += 1
                if len(candidate_reject_samples) < 8:
                    candidate_reject_samples.append((int(lid), "dist", round(dist, 3)))
                continue
            # local tangent
            if k == 0:
                p0, p1 = poly[0], poly[1]
            elif k >= len(poly) - 1:
                p0, p1 = poly[-2], poly[-1]
            else:
                p0, p1 = poly[k - 1], poly[k + 1]
            traj_vec = (p1 - p0).astype(np.float64)
            n = float(np.linalg.norm(traj_vec))
            if n < 1e-3:
                lane_scan_stats["candidate_bad_tangent"] += 1
                if len(candidate_reject_samples) < 8:
                    candidate_reject_samples.append((int(lid), "bad_tangent", round(n, 6)))
                continue
            traj_heading = float(np.arctan2(traj_vec[1], traj_vec[0]))
            heading_diff = abs(wrap_to_pi(traj_heading - actor_heading))
            if heading_diff > HEADING_TH:
                lane_scan_stats["candidate_heading_reject"] += 1
                if len(candidate_reject_samples) < 8:
                    candidate_reject_samples.append((int(lid), "heading", round(heading_diff, 3)))
                continue
            vec_near = (poly[k] - actor_pos[:2]).astype(np.float64)
            forward = float(np.dot(vec_near, ego_dir))
            if forward <= FORWARD_TH:
                lane_scan_stats["candidate_backward_reject"] += 1
                if len(candidate_reject_samples) < 8:
                    candidate_reject_samples.append((int(lid), "backward", round(forward, 3)))
                continue
            if require_route_reachability:
                route_path = _path_to_route(lid)
                if route_path is None:
                    lane_scan_stats["candidate_route_unreachable"] += 1
                    if len(candidate_reject_samples) < 8:
                        candidate_reject_samples.append((int(lid), "route_unreachable", None))
                    continue
                candidate_paths_to_route[int(lid)] = route_path
            candidates.append((lid, dist))



        def _nearest_lane_fallback_line(force_all_lanes=False):
            search_ids = lane_ids
            if (force_all_lanes or not search_ids) and lane_graph is not None:
                search_ids = list(lane_graph.lanes.keys())
            best = None
            best_dist = float("inf")
            best_idx = 0
            for lid in search_ids:
                node = lane_graph.lanes.get(lid) if lane_graph is not None else None
                poly = lane_polys.get(lid)
                if poly is None and node is not None:
                    poly = node.centerline
                if poly is None or len(poly) < 2:
                    continue
                poly = np.asarray(poly, dtype=np.float64)[:, :2]
                dists = np.linalg.norm(poly - actor_pos[None, :2], axis=1)
                k = int(np.argmin(dists))
                dist = float(dists[k])
                if dist < best_dist:
                    best = (lid, poly)
                    best_dist = dist
                    best_idx = k
            if best is None:
                return None, -1
            lid, poly = best

            start_idx = max(0, best_idx - 1)
            line = _line_with_heading(poly[start_idx:])
            return line if len(line) >= 2 else None, lid

        def _nearest_lanes_in_graph(limit=8):
            if lane_graph is None:
                return []
            nearest = []
            for lid, node in lane_graph.lanes.items():
                poly = getattr(node, "centerline", None)
                if poly is None or len(poly) < 2:
                    continue
                poly = np.asarray(poly, dtype=np.float64)[:, :2]
                dists = np.linalg.norm(poly - actor_pos[None, :2], axis=1)
                k = int(np.argmin(dists))
                dist = float(dists[k])
                if k == 0:
                    p0, p1 = poly[0], poly[1]
                elif k >= len(poly) - 1:
                    p0, p1 = poly[-2], poly[-1]
                else:
                    p0, p1 = poly[k - 1], poly[k + 1]
                tangent = (p1 - p0).astype(np.float64)
                tangent_norm = float(np.linalg.norm(tangent))
                lane_heading = (
                    float(np.arctan2(tangent[1], tangent[0]))
                    if tangent_norm > 1e-6 else float("nan")
                )
                heading_diff = (
                    abs(wrap_to_pi(lane_heading - actor_heading))
                    if np.isfinite(lane_heading) else float("nan")
                )
                forward = float(np.dot(poly[k] - actor_pos[:2], ego_dir))
                item = {
                    "lane_id": int(lid),
                    "dist": round(dist, 3),
                    "heading_diff": None if not np.isfinite(heading_diff) else round(float(heading_diff), 3),
                    "forward": round(forward, 3),
                    "in_allowed": bool(allowed_lanes is None or lid in allowed_lanes),
                    "route_reachable": None if not require_route_reachability else _path_to_route(lid) is not None,
                }
                nearest.append((dist, item))
            nearest.sort(key=lambda x: x[0])
            return [item for _, item in nearest[:int(limit)]]

        def _nearest_route_reachable_path(limit=200):
            if not require_route_reachability or lane_graph is None:
                return None
            scored = []
            for lid, node in lane_graph.lanes.items():
                poly = getattr(node, "centerline", None)
                if poly is None or len(poly) < 2:
                    continue
                poly = np.asarray(poly, dtype=np.float64)[:, :2]
                dist = float(np.min(np.linalg.norm(poly - actor_pos[None, :2], axis=1)))
                scored.append((dist, int(lid)))
            scored.sort(key=lambda x: x[0])
            for _, lid in scored[:int(limit)]:
                path = _path_to_route(lid)
                if path is not None:
                    return path
            return None

        def _line_from_path(path):
            if not path:
                return None, -1
            segments = []
            path_types = []
            for lid in path:
                raw = map_features_list.get(str(lid), None)
                poly = raw.get('polyline', None) if isinstance(raw, dict) else None
                path_types.append(
                    None if not isinstance(raw, dict)
                    else str(raw.get('type', ''))
                )
                if poly is None:
                    node = lane_graph.lanes.get(lid)
                    poly = None if node is None else node.centerline
                if poly is None:
                    return None, -1
                segments.append(poly)

            line, merge_audit = _orient_and_merge_reference_segments(
                segments,
                actor_position=actor_pos[:2],
                actor_heading=actor_heading,
            )
            if line is None:
                if self._tjwest_ref_audit_enabled():
                    print(
                        "[RefLineOrientationError]",
                        {
                            "controlled_agent_id": str(controlled_agent_id),
                            "path": [int(value) for value in path],
                            "path_types": path_types,
                            "audit": merge_audit,
                        },
                    )
                return None, -1

            gap_warn = float(self.config.get("tjwest_refline_join_gap_warn_m", 3.0))
            heading_warn = float(
                self.config.get("tjwest_refline_heading_jump_warn_rad", 1.0)
            )
            max_gap = merge_audit.get("max_join_gap_m")
            max_heading = merge_audit.get("max_local_heading_delta_rad")
            if self._tjwest_ref_audit_enabled() and (
                (max_gap is not None and max_gap > gap_warn)
                or (max_heading is not None and max_heading > heading_warn)
            ):
                print(
                    "[RefLineOrientationAudit]",
                    {
                        "controlled_agent_id": str(controlled_agent_id),
                        "path": [int(value) for value in path],
                        "path_types": path_types,
                        "audit": merge_audit,
                    },
                )
            return line, int(path[0])

        def _make_straight_fallback_line():
            print('—————————————— Warning,参考线为直线————————————')
            xs = actor_pos[0] + np.linspace(
                0, build_radius, int(build_radius)
            ) * np.cos(actor_heading)
            ys = actor_pos[1] + np.linspace(
                0, build_radius, int(build_radius)
            ) * np.sin(actor_heading)
            return np.stack([xs, ys, np.full_like(xs, actor_heading)], axis=-1)

        def _is_usable_reference_line(line):
            line = np.asarray(line, dtype=np.float64)
            if line.ndim != 2 or line.shape[0] < 2:
                return False
            cropped_line = _crop_reference_line_from_ego(line, actor_pos[:2], actor_heading)
            if cropped_line.ndim != 2 or cropped_line.shape[0] < 2:
                return False
            xy = cropped_line[:, :2]
            return bool(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)) > 1.0)

        def _ensure_usable_reference_lines(lines, ids):
            usable_lines = []
            usable_ids = []
            for line, lid in zip(lines, ids):
                if _is_usable_reference_line(line):
                    usable_lines.append(line)
                    usable_ids.append(lid)
            if usable_lines:
                return usable_lines, usable_ids

            # Route-constrained candidates can become stale after closed-loop drift.
            # When that happens, recover from the nearest lane in the full lane graph.
            fallback_line = None
            fallback_lid = -1
            if require_route_reachability:
                fallback_path = _nearest_route_reachable_path()
                fallback_line, fallback_lid = _line_from_path(fallback_path)
            else:
                fallback_line, fallback_lid = _nearest_lane_fallback_line(force_all_lanes=True)
            if fallback_line is None:
                fallback_line = _make_straight_fallback_line()
                fallback_lid = route_seq_for_search[0] if route_seq_for_search else -1
            return [fallback_line], [fallback_lid]

        # 从路由序列中找到当前位置对应的车道，只关心未来的路由
        if use_route_constraint and route_seq_for_search and candidates:
            start_idx = None
            for lid, _ in sorted(candidates, key=lambda x: x[1]):
                if lid in route_seq_for_search:
                    start_idx = route_seq_for_search.index(lid)
                    break

            if start_idx is None:
                best_dist = float('inf')
                best_idx = 0
                for i, r_lid in enumerate(route_seq_for_search):
                    # 找到目前 route_seq 里离主车最近的车道
                    node = lane_graph.lanes.get(r_lid)
                    if node and node.centerline is not None and len(node.centerline) > 0:
                        dist = float(np.min(np.linalg.norm(node.centerline - actor_pos[None, :2], axis=1)))
                        if dist < best_dist:
                            best_dist = dist
                            best_idx = i
                start_idx = best_idx

            if start_idx is not None:
                route_seq_for_search = route_seq_for_search[start_idx:]
        route_set = set(route_seq_for_search)

        if len(candidates) == 0:
            if debug_refline:
                nearest_lane_samples.sort(key=lambda x: x[0])
                print(
                    "[RefLineDebug] no candidate reference line, fallback",
                    {
                        "controlled_agent_id": str(controlled_agent_id),
                        "actor_type": str(getattr(self, "controlled_actor_type", "unknown")),
                        "actor_xy": np.round(actor_pos[:2], 3).tolist(),
                        "actor_heading": round(float(actor_heading), 3),
                        "actor_speed": round(float(actor_speed), 3),
                        "route_len": len(route_seq_for_search),
                        "route_head": list(route_seq_for_search[:8]),
                        "raw_route_len": len(route_seq),
                        "raw_route_head": list(route_seq[:8]),
                        "use_route_constraint": bool(use_route_constraint),
                        "require_route_reachability": bool(require_route_reachability),
                        "strict_route_refline": bool(strict_route_refline),
                        "enable_lateral_refline": bool(enable_lateral_refline),
                        "allowed_lanes": None if allowed_lanes is None else len(allowed_lanes),
                        "search_radius": round(float(search_radius), 3),
                        "dist_th": DIST_TH,
                        "heading_th": HEADING_TH,
                        "forward_th": FORWARD_TH,
                        "stats": lane_scan_stats,
                        "nearest_lanes": [(int(lid), round(float(dist), 3)) for dist, lid in nearest_lane_samples],
                        "nearest_lanes_all_graph": _nearest_lanes_in_graph(),
                        "reject_samples": candidate_reject_samples,
                    },
                )
            fallback_path = _nearest_route_reachable_path()
            fallback_line = None
            fallback_lid = -1
            if fallback_path is not None:
                fallback_line, fallback_lid = _line_from_path(fallback_path)
            if fallback_line is None and not route_seq:
                fallback_line, fallback_lid = _nearest_lane_fallback_line(force_all_lanes=True)
            if fallback_line is not None:
                if debug_refline:
                    print(
                        "[RefLineDebug] fallback selected route-reachable lane",
                        {
                            "controlled_agent_id": str(controlled_agent_id),
                            "fallback_lane_id": int(fallback_lid),
                            "fallback_path_head": [] if fallback_path is None else [int(x) for x in fallback_path[:8]],
                            "fallback_path_len": 0 if fallback_path is None else len(fallback_path),
                            "line_points": int(len(fallback_line)),
                        },
                    )
                reference_lines = [fallback_line]
                reference_line_route_ids = [fallback_lid]
            else:
                if debug_refline:
                    print(
                        "[RefLineDebug] fallback selected straight line",
                        {
                            "controlled_agent_id": str(controlled_agent_id),
                            "lane_graph_lanes": 0 if lane_graph is None else len(lane_graph.lanes),
                            "nearby_lanes": len(lane_ids),
                        },
                    )
                reference_lines = [_make_straight_fallback_line()]
                reference_line_route_ids = [route_seq_for_search[0] if route_seq_for_search else -1]
        else:
            # 优先选择在路由上的候选车道，作为起始车道
            cand_sorted = sorted(candidates, key=lambda x: x[1])
            start_lanes_on_route = [lid for lid, _ in cand_sorted if (not use_route_constraint) or (lid in route_set)]
            start_lanes = start_lanes_on_route[:2] if start_lanes_on_route else [lid for lid, _ in cand_sorted[:6]]

            if allow_neighbor_refs:
                # 添加邻近车道，允许变道
                extra_starts = []
                for lid in list(start_lanes):
                    if enable_lateral_refline:
                        lateral_ids = expand_lateral_lane_ids(
                            [lid],
                            lane_graph,
                            lateral_hops=getattr(self, "route_lateral_refline_hops", 2),
                            same_direction_only=getattr(self, "route_lateral_same_direction_only", True),
                            heading_threshold=getattr(self, "route_lateral_heading_threshold", np.pi / 3),
                        )[1:]
                    else:
                        node = lane_graph.lanes.get(lid)
                        lateral_ids = [] if node is None else (node.left_neighbors + node.right_neighbors)
                    for nb in lateral_ids:
                        if allowed_lanes is not None and nb not in allowed_lanes:
                            continue
                        extra_starts.append(nb)
                # unique preserve order
                seen = set(start_lanes)
                for nb in extra_starts:
                    if nb not in seen:
                        start_lanes.append(nb)
                        seen.add(nb)
                    if len(start_lanes) >= 12:
                        break

            # DFS along exit_lanes topology
            all_paths: List[List[int]] = []
            MAX_DFS_PATHS = 50
            MAX_DFS_DEPTH = 40

            # When no route, restrict expansion to lanes near the vehicle
            _lane_dist_threshold = search_radius if not route_seq_for_search else build_radius * 1.2

            # Build a quick route index for successor priority.
            route_index = {lid: i for i, lid in enumerate(route_seq_for_search)} if route_seq_for_search else {}

            def _ordered_successors(cur: int) -> List[int]:
                succs = [s for s in lane_graph.successors(cur) if s in lane_graph.lanes]
                if not succs:
                    return []
                if not route_seq_for_search or cur not in route_index:
                    return succs
                i = route_index[cur]

                # 允许跳过少量 topology 缺失，但不能大跳到出口
                MAX_SKIP = 3
                future_window = route_seq_for_search[i + 1: i + 1 + MAX_SKIP]
                ordered = [s for s in future_window if s in succs]
                return ordered

            def dfs(path: List[int], acc_len: float, offroute_budget: int):
                if len(all_paths) >= MAX_DFS_PATHS:
                    return
                if len(path) > MAX_DFS_DEPTH:
                    all_paths.append(path)
                    return
                cur = path[-1]
                if acc_len >= build_radius:
                    all_paths.append(path)
                    return
                succs = _ordered_successors(cur)
                if not succs:
                    all_paths.append(path)
                    return
                expanded = False
                for nxt in succs:
                    if nxt in path:
                        continue
                    # 保持在路由上; 允许邻边拓展
                    if route_seq_for_search and nxt not in route_set:
                        # allow neighbor lanes only
                        allow_offroute = (nxt in neighbor_extra) and (offroute_budget > 0)
                        if not allow_offroute:
                            continue
                    poly_xy = lane_graph.lanes[nxt].centerline
                    if poly_xy is None or len(poly_xy) < 2:
                        continue
                    # keep within radius by endpoint proximity
                    if (
                        float(
                            np.min(
                                np.linalg.norm(
                                    poly_xy - actor_pos[None, :2],
                                    axis=1,
                                )
                            )
                        )
                        > _lane_dist_threshold
                    ):
                        continue
                    expanded = True
                    next_budget = offroute_budget
                    if route_seq_for_search and (nxt not in route_set):
                        # 偏离 route 一步就扣预算，包括邻居车道
                        next_budget -= 1
                    dfs(path + [nxt], acc_len + lane_length_xy(poly_xy), next_budget)
                if not expanded:
                    all_paths.append(path)

            offroute_budget = 1 if allow_neighbor_refs else 0
            for lid in start_lanes:
                if len(all_paths) >= MAX_DFS_PATHS:
                    break
                # if route_seq exists, we can require start lane in trimmed route.
                if route_seq_for_search and lid not in route_set and lid not in neighbor_extra:
                    continue
                poly_xy = lane_graph.lanes[lid].centerline
                if poly_xy is None or len(poly_xy) < 2:
                    continue
                dfs([lid], lane_length_xy(poly_xy), offroute_budget=offroute_budget)

            # Merge lane centerlines into continuous polylines. Preserve original heading from polyline if present.
            reference_lines = []
            reference_line_route_ids = []



            if require_route_reachability:
                all_paths = [
                    candidate_paths_to_route[int(lid)]
                    for lid in start_lanes
                    if int(lid) in candidate_paths_to_route
                ][:MAX_DFS_PATHS]

            for path in all_paths:
                # Preserve the lane-ID path, including legal turn/connector ID
                # changes. Only fix each polyline's stored direction.
                line, route_id = _line_from_path(path)
                if line is None:
                    continue
                _dedup_append(
                    reference_lines,
                    reference_line_route_ids,
                    line,
                    route_id,
                )

                # Cap number of reference lines to keep tensors bounded
                if len(reference_lines) >= 12:
                    break

            if len(reference_lines) == 0:
                if debug_refline:
                    print(
                        "[RefLineDebug] dfs produced no reference lines, fallback to start lanes",
                        {
                            "controlled_agent_id": str(controlled_agent_id),
                            "route_set_size": len(route_set),
                            "start_lanes": [int(x) for x in start_lanes],
                            "all_paths": len(all_paths),
                            "require_route_reachability": bool(require_route_reachability),
                            "neighbor_extra": len(neighbor_extra),
                        },
                    )
                # Last resort: allow unfiltered expansion (ignore route_set)
                for lid in start_lanes:
                    poly = map_features_list.get(str(lid), {}).get('polyline', None)
                    if poly is None:
                        continue
                    reference_lines.append(np.asarray(poly))
                    reference_line_route_ids.append(int(lid))

        if debug_refline and require_route_reachability and len(candidates) > 0:
            path_summaries = []
            for lid in (start_lanes if 'start_lanes' in locals() else []):
                path = candidate_paths_to_route.get(int(lid))
                if path is None:
                    continue
                path_summaries.append({
                    "start_lane": int(lid),
                    "path_len": len(path),
                    "path_head": [int(x) for x in path[:8]],
                    "route_goal": int(path[-1]) if path else None,
                })
            print(
                "[RefLineDebug] selected route-reachable reference lanes",
                {
                    "controlled_agent_id": str(controlled_agent_id),
                    "actor_xy": np.round(actor_pos[:2], 3).tolist(),
                    "actor_heading": round(float(actor_heading), 3),
                    "raw_route_head": list(route_seq[:8]),
                    "candidate_count": len(candidates),
                    "start_lanes": [int(x) for x in (start_lanes if 'start_lanes' in locals() else [])],
                    "reference_lines": len(reference_lines),
                    "enable_lateral_refline": bool(enable_lateral_refline),
                    "neighbor_extra": len(neighbor_extra),
                    "paths": path_summaries[:8],
                },
            )

        reference_lines, reference_line_route_ids = _ensure_usable_reference_lines(
            reference_lines, reference_line_route_ids
        )

        # ---- 按与 route 方向一致性排序过滤 ----
        if route_seq_for_search and len(reference_lines) > 1:
            # Use the same orientation-only merge for the route-direction
            # baseline. Route IDs may legitimately change at turns/connectors.
            route_line, _ = _line_from_path(route_seq_for_search)
            if route_line is not None:
                route_full = route_line[:, :2]
                # 找 route 上离车辆最近的点，取前方 ~20m 处的朝向
                dists = np.linalg.norm(route_full - actor_pos[None, :2], axis=1)
                nearest = int(np.argmin(dists))
                lookahead = min(nearest + 40, len(route_full) - 1)  # ~20m ahead (0.5m resolution)
                if lookahead > nearest + 1:
                    vec = route_full[lookahead] - route_full[nearest]
                    route_heading = float(np.arctan2(vec[1], vec[0]))

                    scored = []
                    for i, line in enumerate(reference_lines):
                        line = np.asarray(line, dtype=np.float64)
                        # 取参考线上距离车辆前方 ~20m 处的朝向
                        line_dists = np.linalg.norm(line[:, :2] - actor_pos[None, :2], axis=1)
                        line_nearest = int(np.argmin(line_dists))
                        line_look = min(line_nearest + 40, len(line) - 1)
                        if line_look > line_nearest + 1:
                            vec_l = line[line_look, :2] - line[line_nearest, :2]
                        else:
                            vec_l = line[-1, :2] - line[0, :2]
                        line_heading = float(np.arctan2(vec_l[1], vec_l[0]))
                        angle_diff = abs(wrap_to_pi(line_heading - route_heading))
                        scored.append((angle_diff, i, line, reference_line_route_ids[i]))
                    if scored:
                        scored.sort(key=lambda x: x[0])
                        keep = [(line, rid) for diff, _, line, rid in scored if diff < np.deg2rad(60)]
                        if len(keep) >= 2:
                            reference_lines, reference_line_route_ids = zip(*keep)
                            reference_lines = list(reference_lines)
                            reference_line_route_ids = list(reference_line_route_ids)
                        elif len(keep) == 0:
                            best = scored[0]
                            reference_lines = [best[2]]
                            reference_line_route_ids = [best[3]]
        # ----------------------------------------------------------------
        reference_lines, reference_line_route_ids = _prepend_or_append_expert_reference_line(
            reference_lines,
            reference_line_route_ids,
        )

        self._reference_line_cache[reference_line_cache_key] = {
            "actor_xy": np.asarray(actor_pos[:2], dtype=np.float64).copy(),
            "actor_heading": float(actor_heading),
            "route_seq": tuple(route_seq_for_search),
            "training": bool(training),
            "reference_lines": [
                np.asarray(line, dtype=np.float64).copy()
                for line in reference_lines
            ],
            "reference_line_route_ids": list(reference_line_route_ids),
        }
        return self._reference_lines_to_feature(
            reference_lines,
            reference_line_route_ids,
            actor_features,
            actor_pos,
            actor_heading,
        )