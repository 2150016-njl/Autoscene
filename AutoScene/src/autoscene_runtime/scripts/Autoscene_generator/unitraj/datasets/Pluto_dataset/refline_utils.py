from typing import List
import numpy as np


def project_points_to_polyline(points: np.ndarray, line_xy: np.ndarray):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    line_xy = np.asarray(line_xy, dtype=np.float64)
    if line_xy.ndim != 2 or line_xy.shape[0] < 2:
        return (
            np.zeros((len(points),), dtype=np.float64),
            np.full((len(points),), np.inf, dtype=np.float64),
            np.zeros((len(points),), dtype=np.int64),
            np.zeros((len(points),), dtype=np.float64),
        )

    seg_start = line_xy[:-1]
    seg_vec = np.diff(line_xy, axis=0)
    seg_len_sq = np.sum(seg_vec * seg_vec, axis=1)
    valid = seg_len_sq > 1e-12
    safe_len_sq = np.where(valid, seg_len_sq, 1.0)

    rel = points[:, None, :] - seg_start[None, :, :]
    alpha = np.sum(rel * seg_vec[None, :, :], axis=-1) / safe_len_sq[None, :]
    alpha = np.clip(alpha, 0.0, 1.0)
    projected = seg_start[None, :, :] + alpha[..., None] * seg_vec[None, :, :]
    dist_sq = np.sum((points[:, None, :] - projected) ** 2, axis=-1)
    dist_sq[:, ~valid] = np.inf

    seg_idx = np.argmin(dist_sq, axis=1)
    row_idx = np.arange(len(points))
    best_alpha = alpha[row_idx, seg_idx]
    best_dist = np.sqrt(dist_sq[row_idx, seg_idx])

    seg_len = np.sqrt(seg_len_sq)
    cum_len = np.concatenate([[0.0], np.cumsum(seg_len)])
    along = cum_len[seg_idx] + best_alpha * seg_len[seg_idx]
    return along, best_dist, seg_idx, best_alpha


def _dedup_append(lines: List[np.ndarray], ids: List[int], line: np.ndarray, route_id: int,
                  eps: float = 1.0) -> None:
    """Append line if it's not covered by existing ones. Remove existing ones covered by this line."""
    if line is None or len(line) < 2:
        return
    to_remove = []
    for i, ex in enumerate(lines):
        diff = line[:, None, :2] - ex[None, :, :2]
        dists = np.linalg.norm(diff, axis=-1)

        dist_line_to_ex = np.max(np.min(dists, axis=1))
        dist_ex_to_line = np.max(np.min(dists, axis=0))

        if dist_line_to_ex < eps:
            # 'line' is completely covered by 'ex' (within eps), so it provides no new path info
            return

        if dist_ex_to_line < eps:
            # 'ex' is completely covered by 'line', mark 'ex' for replacement
            to_remove.append(i)

    # Pop out in reverse to maintain indices
    for i in reversed(to_remove):
        lines.pop(i)
        ids.pop(i)

    lines.append(line)
    ids.append(route_id)


def lane_length_xy(poly_xy: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(poly_xy, axis=0), axis=1)))


def _crop_reference_line_from_ego(line: np.ndarray, ego_xy: np.ndarray, ego_heading: float) -> np.ndarray:
    line = np.asarray(line, dtype=np.float64)
    if line.ndim != 2 or line.shape[0] < 2:
        return line

    line_xy = line[:, :2]
    seg_vec = np.diff(line_xy, axis=0)
    seg_len = np.linalg.norm(seg_vec, axis=1)
    if not np.any(seg_len > 1e-6):
        return line

    cum_len = np.concatenate([[0.0], np.cumsum(seg_len)])
    projected, _, projected_seg_idx, projected_alpha = project_points_to_polyline(
        np.asarray(ego_xy[:2], dtype=np.float64)[None, :],
        line_xy,
    )
    proj_dist = float(projected[0])
    proj_dist = float(np.clip(proj_dist, 0.0, cum_len[-1]))
    seg_idx = int(projected_seg_idx[0])

    seg_start = float(cum_len[seg_idx])
    seg_total = float(seg_len[seg_idx])
    if seg_total > 1e-6:
        alpha = float(projected_alpha[0])
        proj_xy = line_xy[seg_idx] + alpha * seg_vec[seg_idx]
        proj_heading = float(np.arctan2(seg_vec[seg_idx, 1], seg_vec[seg_idx, 0]))
    else:
        alpha = 0.0
        proj_xy = line_xy[seg_idx].copy()
        proj_heading = float(line[seg_idx, 2] if line.shape[1] >= 3 else ego_heading)

    tail = line[seg_idx + 1:]
    if line.shape[1] >= 3:
        first = np.array([[proj_xy[0], proj_xy[1], proj_heading]], dtype=np.float64)
    else:
        first = np.array([[proj_xy[0], proj_xy[1]]], dtype=np.float64)

    cropped = np.concatenate([first, tail], axis=0)
    if cropped.shape[0] < 2:
        # Keep at least two points for downstream vector/orientation construction.
        cropped = line[max(seg_idx, 0): seg_idx + 2]
    return cropped


def _resample_reference_line(
        line: np.ndarray,
        target_points: int,
        actor_heading,
        sample_spacing: float = 1.0,
) -> np.ndarray:
    line = np.asarray(line, dtype=np.float64)
    if line.ndim != 2 or line.shape[0] == 0:
        return line
    if line.shape[0] == 1:
        base_xy = np.repeat(line[:, :2], target_points, axis=0)
        if line.shape[1] >= 3:
            base_hdg = np.repeat(line[:, 2], target_points, axis=0)
        else:
            base_hdg = np.zeros((target_points,), dtype=np.float64)
        return np.concatenate([base_xy, base_hdg[:, None]], axis=1)

    line_xy = line[:, :2]
    seg_vec = np.diff(line_xy, axis=0)
    seg_len = np.linalg.norm(seg_vec, axis=1)
    valid_seg = seg_len > 1e-6
    if not np.any(valid_seg):
        base_xy = np.repeat(line_xy[:1], target_points, axis=0)
        base_hdg = np.full((target_points,), float(actor_heading), dtype=np.float64)
        return np.concatenate([base_xy, base_hdg[:, None]], axis=1)

    cum_len = np.concatenate([[0.0], np.cumsum(seg_len)])
    sample_dist = np.arange(target_points, dtype=np.float64) * float(sample_spacing)
    if line.shape[1] >= 3:
        base_ori = np.asarray(line[:, 2], dtype=np.float64)
    else:
        base_ori = np.zeros((line.shape[0],), dtype=np.float64)
        base_ori[:-1] = np.arctan2(seg_vec[:, 1], seg_vec[:, 0])
        base_ori[-1] = base_ori[-2]

    last_valid_idx = int(np.where(valid_seg)[0][-1])
    last_heading = float(np.arctan2(seg_vec[last_valid_idx, 1], seg_vec[last_valid_idx, 0]))
    last_xy = line_xy[last_valid_idx + 1]
    total_len = float(cum_len[-1])

    inside = sample_dist <= total_len
    inside_dist = sample_dist[inside]
    seg_idx = np.searchsorted(cum_len[1:], inside_dist, side="right")
    seg_idx = np.clip(seg_idx, 0, len(seg_len) - 1)
    safe_seg_len = np.where(seg_len > 1e-6, seg_len, 1.0)
    alpha = (inside_dist - cum_len[seg_idx]) / safe_seg_len[seg_idx]
    alpha = np.clip(alpha, 0.0, 1.0)

    out_xy = np.empty((target_points, 2), dtype=np.float64)
    out_hdg = np.empty((target_points,), dtype=np.float64)
    out_xy[inside] = (
        line_xy[seg_idx] + alpha[:, None] * seg_vec[seg_idx]
    )
    segment_heading = np.arctan2(seg_vec[seg_idx, 1], seg_vec[seg_idx, 0])
    out_hdg[inside] = np.where(
        seg_len[seg_idx] > 1e-6,
        segment_heading,
        base_ori[seg_idx],
    )

    if np.any(~inside):
        extra = sample_dist[~inside] - total_len
        last_direction = np.array(
            [np.cos(last_heading), np.sin(last_heading)],
            dtype=np.float64,
        )
        out_xy[~inside] = last_xy + extra[:, None] * last_direction[None, :]
        out_hdg[~inside] = last_heading

    return np.concatenate([out_xy, out_hdg[:, None]], axis=1)


def _line_with_heading(poly_xy: np.ndarray) -> np.ndarray:
    poly_xy = np.asarray(poly_xy, dtype=np.float64)
    if len(poly_xy) < 2:
        return np.zeros((0, 3), dtype=np.float64)
    delta = np.diff(poly_xy, axis=0, append=poly_xy[-1:])
    if len(poly_xy) > 1:
        delta[-1] = poly_xy[-1] - poly_xy[-2]
    heading = np.arctan2(delta[:, 1], delta[:, 0])
    return np.concatenate([poly_xy[:, :2], heading[:, None]], axis=-1)
