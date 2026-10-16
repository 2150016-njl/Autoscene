"""Utilities to convert unified MetaDrive/ScenarioNet-style map_features into a lane graph.

Why this exists
---------------
Pluto's original NuPlan feature builder relies on the NuPlan map API which provides:
- Accurate lane/lane_connector geometry (centerline + boundaries)
- Explicit topology (incoming/outgoing edges)
- A route as an ordered sequence of roadblocks

In UniTraj we only have `scenario['map_features']`, which is a dict of lane-like features.
This module offers a best-effort conversion into a lightweight lane graph that can be reused
by feature extraction (reference line generation, route id estimation, map filtering, etc.).

Assumptions (as discussed)
-------------------------
Each lane-like map feature is a dict with (some of) the following keys:
- id: lane id (or the dict key itself)
- entry / exit: lists of predecessor/successor lane ids (or roadblock ids)
- polyline: centerline polyline points [(x, y), ...] or [(x, y, heading), ...]
- polygon: boundary polygon points [(x, y), ...]

The code is defensive: it tries multiple key aliases and gracefully degrades when topology
fields are missing by inferring adjacency from geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
from scipy.spatial import cKDTree


LANE_TYPE_HINTS = ("LANE", "LANE_CONNECTOR", "ROAD", "DRIVABLE")
NON_DRIVABLE_TYPE_HINTS = (
    "BOUNDARY",
    "ROAD_EDGE",
    "ROAD_LINE",
    "CURB",
    "CROSSWALK",
    "STOP_LINE",
    "STOP_SIGN",
    "BARRIER",
    "SPEED_BUMP",
)


def _is_drivable_lane_feature_type(feature_type: Any) -> bool:
    """Return True only for centerline-like drivable lane features.

    The previous substring-only allow-list treated ``ROAD_EDGE_BOUNDARY`` as a
    lane because it contains the word ``ROAD``.  Boundary/marking features must
    never enter the lane graph used for route/reference-line search.
    """
    t = str(feature_type or "").strip().upper()
    if not t:
        # Preserve compatibility with ScenarioNet variants that omit type but
        # still expose an actual lane centerline and explicit topology.
        return True
    if any(token in t for token in NON_DRIVABLE_TYPE_HINTS):
        return False
    if t.startswith("LANE") or "LANE_CONNECTOR" in t:
        return True
    # Keep exact generic road/drivable centerline tags, but do not use substring
    # matching so ROAD_EDGE_* cannot leak back in.
    return t in {"ROAD", "DRIVABLE", "DRIVABLE_LANE", "ROAD_CENTERLINE"}


def _endpoint_heading(centerline: np.ndarray, at_start: bool) -> Optional[float]:
    arr = np.asarray(centerline, dtype=np.float64)
    if arr.ndim != 2 or len(arr) < 2:
        return None
    if at_start:
        pairs = zip(arr[:-1], arr[1:])
    else:
        pairs = zip(arr[-2::-1], arr[:0:-1])
    for p0, p1 in pairs:
        delta = np.asarray(p1[:2] - p0[:2], dtype=np.float64)
        if float(np.linalg.norm(delta)) > 1e-6:
            return float(np.arctan2(delta[1], delta[0]))
    return None


def _as_int_id(x: Any) -> Optional[int]:
    if x is None:
        return None
    # preserve int
    if isinstance(x, (int, np.integer)):
        return int(x)
    # numeric strings
    s = str(x)
    if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
        try:
            return int(s)
        except Exception:
            return None
    # fallback to stable hash (deterministic within run, handles UUIDs like nuScenes map tokens)
    return int(abs(hash(s)) % (10**9))


def _to_xy(points: Any) -> Optional[np.ndarray]:
    """Convert common polyline-like structures to (N,2) float array."""
    if points is None:
        return None
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return None
    if arr.shape[1] >= 2:
        return arr[:, :2]
    return None


def _get_first_present(d: Dict[str, Any], keys: Iterable[str]) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


@dataclass
class LaneNode:
    lane_id: int
    centerline: np.ndarray  # (N,2)
    polygon: Optional[np.ndarray]  # (M,2)
    entry: List[int]
    exit: List[int]
    left_neighbors: List[int]
    right_neighbors: List[int]
    feature_type: str = ""

    @property
    def start(self) -> np.ndarray:
        return self.centerline[0]

    @property
    def end(self) -> np.ndarray:
        return self.centerline[-1]


@dataclass
class LaneGraph:
    lanes: Dict[int, LaneNode]
    entry_to: Dict[int, Set[int]]  # lane_id -> {successors}
    exit_from: Dict[int, Set[int]]  # lane_id -> {predecessors}

    def successors(self, lane_id: int) -> List[int]:
        return sorted(self.entry_to.get(lane_id, set()))

    def predecessors(self, lane_id: int) -> List[int]:
        return sorted(self.exit_from.get(lane_id, set()))


def _lane_mean_heading(node: LaneNode) -> Optional[float]:
    centerline = np.asarray(node.centerline, dtype=np.float64)
    if centerline.ndim != 2 or centerline.shape[0] < 2:
        return None
    delta = centerline[-1, :2] - centerline[0, :2]
    if float(np.linalg.norm(delta)) < 1e-6:
        seg = np.diff(centerline[:, :2], axis=0)
        seg_norm = np.linalg.norm(seg, axis=1)
        valid = seg_norm > 1e-6
        if not valid.any():
            return None
        delta = np.sum(seg[valid], axis=0)
    return float(np.arctan2(delta[1], delta[0]))


def _angle_diff_abs(a: float, b: float) -> float:
    return float(abs((a - b + np.pi) % (2 * np.pi) - np.pi))


def expand_lateral_lane_ids(
    route_seq: Iterable[int],
    lane_graph: LaneGraph,
    *,
    lateral_hops: int = 2,
    same_direction_only: bool = True,
    heading_threshold: float = np.pi / 3,
) -> List[int]:
    """Return route ids plus ordered same-direction lateral neighbors up to ``lateral_hops``.

    The route order is preserved, and each route lane is followed by its left/right neighbors
    before moving to the next route lane. Neighbor traversal only uses explicit topology from
    the map features.
    """
    route_ids = [lid for lid in route_seq if lid in lane_graph.lanes]
    if not route_ids:
        return []

    hops = int(np.clip(int(lateral_hops), 0, 2))
    if hops <= 0:
        return route_ids

    expanded: List[int] = []
    seen: Set[int] = set()

    def _add(lid: int) -> None:
        if lid in lane_graph.lanes and lid not in seen:
            seen.add(lid)
            expanded.append(lid)

    def _same_direction(base_id: int, nb_id: int) -> bool:
        if not same_direction_only:
            return True
        base_node = lane_graph.lanes.get(base_id)
        nb_node = lane_graph.lanes.get(nb_id)
        if base_node is None or nb_node is None:
            return False
        base_heading = _lane_mean_heading(base_node)
        nb_heading = _lane_mean_heading(nb_node)
        if base_heading is None or nb_heading is None:
            return False
        return _angle_diff_abs(base_heading, nb_heading) <= float(heading_threshold)

    def _ordered_lateral(lid: int) -> List[int]:
        node = lane_graph.lanes.get(lid)
        if node is None:
            return []
        return list(node.left_neighbors or []) + list(node.right_neighbors or [])

    for lid in route_ids:
        _add(lid)
        frontier = [lid]
        for _ in range(hops):
            next_frontier: List[int] = []
            for cur in frontier:
                for nb in _ordered_lateral(cur):
                    if nb not in lane_graph.lanes or nb in seen:
                        continue
                    if not _same_direction(lid, nb):
                        continue
                    _add(nb)
                    next_frontier.append(nb)
            frontier = next_frontier
            if not frontier:
                break

    return expanded


def build_lane_graph(
    map_features: Dict[Any, Dict[str, Any]],
    *,
    infer_from_geometry: bool = True,
    geom_link_dist_m: float = 2.0,
    geom_link_heading_max_rad: float = np.deg2rad(100.0),
) -> LaneGraph:
    """Build a lightweight lane graph.

    Parameters
    ----------
    map_features:
        scenario['map_features'] dict.
    infer_from_geometry:
        If explicit entry/exit lists are missing, infer adjacency by connecting lane A's end
        to lane B's start when distance < geom_link_dist_m.
    geom_link_dist_m:
        Threshold for geometry-based linking.
    """

    lanes: Dict[int, LaneNode] = {}

    # 1) Parse lane-like features
    for raw_id, feat in (map_features or {}).items():
        if not isinstance(feat, dict):
            continue

        # Only centerline-like drivable lanes belong in this graph.  In
        # particular, ROAD_EDGE_BOUNDARY must not be accepted merely because it
        # contains the substring "ROAD".
        t = str(feat.get("type", "")).upper()
        if not _is_drivable_lane_feature_type(t):
            continue

        # In our unified format, lane_id is the dict key (string form)
        lane_id = _as_int_id(raw_id)
        if lane_id is None:
            # fall back to stable hash (still deterministic within run)
            lane_id = int(abs(hash(str(raw_id))) % (10**9))

        # polyline points are (x, y, heading). We only keep (x, y) here.
        centerline = _to_xy(_get_first_present(feat, ["polyline", "centerline", "baseline_path", "line"]))
        if centerline is None:
            # Some datasets encode lane centerline as polygon boundary; we can't use it reliably.
            continue

        polygon = _to_xy(_get_first_present(feat, ["polygon", "boundary", "poly"]))

        entry = _get_first_present(
            feat,
            [
                "entry_lanes",
                "entry",
                "entries",
                "predecessors",
                "incoming",
                "in",
            ],
        )
        exit_ = _get_first_present(
            feat,
            [
                "exit_lanes",
                "exit",
                "exits",
                "successors",
                "outgoing",
                "out",
            ],
        )

        left_n = _get_first_present(feat, ["left_neighbor", "left_neighbors", "left"])
        right_n = _get_first_present(feat, ["right_neighbor", "right_neighbors", "right"])

        def _ids_list(v: Any) -> List[int]:
            if v is None:
                return []
            if isinstance(v, (int, np.integer, str)):
                v = [v]
            out: List[int] = []
            for x in list(v):
                xid = _as_int_id(x)
                if xid is not None:
                    out.append(xid)
            return out

        lanes[lane_id] = LaneNode(
            lane_id=lane_id,
            centerline=centerline,
            polygon=polygon,
            entry=_ids_list(entry),
            exit=_ids_list(exit_),
            left_neighbors=_ids_list(left_n),
            right_neighbors=_ids_list(right_n),
            feature_type=t,
        )

    # 2) Build adjacency from explicit topology
    succ: Dict[int, Set[int]] = {lid: set() for lid in lanes.keys()}
    pred: Dict[int, Set[int]] = {lid: set() for lid in lanes.keys()}

    for lid, node in lanes.items():
        for s in node.exit:
            if s in lanes:
                succ[lid].add(s)
                pred[s].add(lid)
        for p in node.entry:
            if p in lanes:
                pred[lid].add(p)
                succ[p].add(lid)

    # 3) Optional geometry-based linking if explicit data is weak
    if infer_from_geometry:
        lane_ids = list(lanes.keys())
        if lane_ids:
            ends = np.stack([lanes[lid].end for lid in lane_ids], axis=0)  # (L,2)
            starts = np.stack([lanes[lid].start for lid in lane_ids], axis=0)  # (L,2)

            # Compute pairwise distances end->start
            # For L up to ~1000 this is fine; if larger, consider KDTree.
            d2 = (
                (ends[:, None, 0] - starts[None, :, 0]) ** 2
                + (ends[:, None, 1] - starts[None, :, 1]) ** 2
            )
            thr2 = float(geom_link_dist_m) ** 2
            src_idx, dst_idx = np.where(d2 < thr2)
            for i, j in zip(src_idx.tolist(), dst_idx.tolist()):
                a = lane_ids[i]
                b = lane_ids[j]
                if a == b or b in succ[a]:
                    continue

                # Geometry-only linking is a fallback for missing explicit
                # topology.  Endpoint proximity alone can connect opposite-flow
                # lanes at the same road edge, producing an instantaneous ~pi
                # reference-line reversal.  Require tangent compatibility too.
                a_heading = _endpoint_heading(lanes[a].centerline, at_start=False)
                b_heading = _endpoint_heading(lanes[b].centerline, at_start=True)
                if a_heading is not None and b_heading is not None:
                    if _angle_diff_abs(a_heading, b_heading) > float(geom_link_heading_max_rad):
                        continue
                succ[a].add(b)
                pred[b].add(a)

    return LaneGraph(lanes=lanes, entry_to=succ, exit_from=pred)


def estimate_route_lane_ids(
    ego_positions_xy: np.ndarray,
    lane_graph: LaneGraph,
    *,
    max_dist_m: float = 1.5,
    min_hold_frames: int = 3,
    no_backtrack: bool = True,
    include_lateral_neighbors: bool = False,
    lateral_hops: int = 1,
) -> List[int]:
    """Estimate ordered route lane ids by matching ego positions to nearest lane polygon/centerline.

    This mirrors the idea used in `pluto_utils.get_route_roadblock_ids`, but operates on lane_graph.
    The output is a forward ordered sequence (with debouncing) and can be used as a route prior.

    Notes
    -----
    - If polygons are missing, we fall back to distance-to-centerline endpoints (rough).
    - For full fidelity you would want point-to-linestring distance; here we use a cheap proxy.
    """
    if ego_positions_xy is None:
        return []
    pos = np.asarray(ego_positions_xy, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[0] == 0:
        return []

    lane_ids = list(lane_graph.lanes.keys())
    if not lane_ids:
        return []

    route_index = getattr(lane_graph, "_route_point_index", None)
    if route_index is None:
        sampled_points = []
        sampled_lane_ids = []
        for lid in lane_ids:
            centerline = lane_graph.lanes[lid].centerline
            if centerline.shape[0] > 30:
                indices = np.linspace(
                    0, centerline.shape[0] - 1, 30
                ).astype(np.int64)
                centerline = centerline[indices]
            sampled_points.append(centerline)
            sampled_lane_ids.extend([lid] * len(centerline))
        flat_points = np.concatenate(sampled_points, axis=0)
        route_index = (
            cKDTree(flat_points),
            np.asarray(sampled_lane_ids, dtype=np.int64),
        )
        lane_graph._route_point_index = route_index

    route_tree, sampled_lane_ids = route_index
    nearest_dist, nearest_point_idx = route_tree.query(pos, k=1)
    matched_lane_ids = sampled_lane_ids[
        np.asarray(nearest_point_idx, dtype=np.int64)
    ]
    # Lanelet/OSM lane IDs may legitimately be negative (e.g. -101879 in
    # tjwest260725intersection).  Keep validity separate from the lane ID;
    # using -1 as a sentinel and then testing ``lid < 0`` drops every such
    # valid lane and leaves the ego route empty.
    matched_valid = np.asarray(nearest_dist) <= max_dist_m
    expert_tree = cKDTree(pos)

    route_seq: List[int] = []
    committed_set: Set[int] = set()

    last_committed: Optional[int] = None
    candidate: Optional[int] = None
    candidate_count = 0

    for matched_lid, is_valid_match in zip(matched_lane_ids, matched_valid):
        if not bool(is_valid_match):
            candidate = None
            candidate_count = 0
            continue

        lid = int(matched_lid)

        if lid == last_committed:
            candidate = None
            candidate_count = 0
            continue

        if candidate is None or lid != candidate:
            candidate = lid
            candidate_count = 1
        else:
            candidate_count += 1

        if candidate_count < min_hold_frames:
            continue

        def lane_close_to_expert(lid, expert_xy, max_d=2.0, ratio_th=0.6):
            cl = lane_graph.lanes[lid].centerline
            min_d, _ = expert_tree.query(cl, k=1)
            close_mask = min_d < max_d
            close_count = int(close_mask.sum())
            if close_mask.mean() >= ratio_th:
                return True
            # Intersection lanes are often long relative to the part covered
            # by the recorded ego motion.  Preserve a lane when a sufficient
            # local section agrees with the expert trajectory.
            min_required_points = min(int(np.ceil(ratio_th * len(cl))), 5)
            return close_count >= max(2, min_required_points)
        if not lane_close_to_expert(lid, pos, max_d=2.0, ratio_th=0.6):
            continue

        # commit
        if last_committed is None:
            route_seq.append(lid)
            committed_set.add(lid)
            last_committed = lid
        else:
            # no_backtrack: do not revisit an earlier lane id
            if no_backtrack and lid in committed_set:
                # If topology indicates it's a valid successor chain, keep; else skip.
                if lid in lane_graph.successors(last_committed):
                    route_seq.append(lid)
                    last_committed = lid
                # else ignore to avoid loops
            else:
                route_seq.append(lid)
                committed_set.add(lid)
                last_committed = lid

        candidate = None
        candidate_count = 0

    if not include_lateral_neighbors:
        return route_seq

    return expand_lateral_lane_ids(
        route_seq,
        lane_graph,
        lateral_hops=lateral_hops,
        same_direction_only=True,
    )