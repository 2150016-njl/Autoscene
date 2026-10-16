from __future__ import annotations
from dataclasses import dataclass, fields
from typing import Dict, Optional

import numpy as np


def _to_numpy(x):
    if x is None:
        return None
    return x.detach().cpu().numpy() if hasattr(x, "detach") else (x.cpu().numpy() if hasattr(x, "cpu") else x)


def _wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


@dataclass
class FastTrajectoryEvaluatorConfig:
    dt: float = 0.1
    ego_length: float = 4.8
    ego_width: float = 2.0
    agent_length: float = 4.8
    agent_width: float = 2.0
    collision_penalty: float = 60.0
    wrong_way_penalty: float = 20.0
    weight_progress: float = 8.0
    weight_comfort: float = 1.0
    weight_ref_tracking: float = 5.0
    max_abs_accel: float = 6.0
    max_abs_yaw_diff: float = np.deg2rad(100.0)
    num_frames_eval: int = 30
    progress_displacement_scale: float = 30.0
    progress_path_length_scale: float = 40.0
    stall_distance_threshold: float = 8.0
    stall_penalty: float = 3.0
    ref_tracking_margin: float = 2.0
    ref_tracking_terminal_weight: float = 1.5


class FastTrajectoryEvaluator:
    """Lightweight post-forward candidate scorer.

    It keeps the important ranking signals used after model.forward:
    progress, comfort, reference tracking, wrong-way, and approximate dynamic
    collision. It intentionally avoids expensive map polygon point-in-poly checks.
    """

    def __init__(self, config: Optional[Dict] = None):
        cfg = dict(config or {})
        valid_keys = {f.name for f in fields(FastTrajectoryEvaluatorConfig)}
        filtered = {k: v for k, v in cfg.items() if k in valid_keys}
        self.config = FastTrajectoryEvaluatorConfig(**filtered)

    def evaluate(self, candidate_trajectories, input_data, predictions=None, candidate_ref_idx=None, debug_context=None):
        candidate_trajectories = _to_numpy(candidate_trajectories)
        input_data = input_data if isinstance(input_data, dict) else getattr(input_data, "data", input_data)
        predictions = _to_numpy(predictions)

        if candidate_trajectories is None or len(candidate_trajectories) == 0:
            return np.zeros((0,), dtype=np.float32)

        traj_xy = np.asarray(candidate_trajectories[..., :2], dtype=np.float32)
        traj_yaw = self._trajectory_yaw(candidate_trajectories, traj_xy)
        k = traj_xy.shape[0]

        if candidate_ref_idx is None:
            candidate_ref_idx = np.zeros((k,), dtype=np.int64)
        else:
            candidate_ref_idx = np.asarray(candidate_ref_idx, dtype=np.int64).reshape(-1)
            if len(candidate_ref_idx) != k:
                candidate_ref_idx = np.resize(candidate_ref_idx, k)

        collision = self._has_collision_three_circle(traj_xy, traj_yaw, predictions)
        wrong_way = self._wrong_way(traj_xy, traj_yaw, input_data, candidate_ref_idx)
        progress = self._progress(traj_xy)
        comfort = self._comfort(traj_xy)
        ref_tracking = self._ref_tracking(traj_xy, input_data, candidate_ref_idx)

        score = (
            -float(self.config.collision_penalty) * collision.astype(np.float32)
            -float(self.config.wrong_way_penalty) * wrong_way.astype(np.float32)
            + float(self.config.weight_progress) * progress.astype(np.float32)
            + float(self.config.weight_comfort) * comfort.astype(np.float32)
            + float(self.config.weight_ref_tracking) * ref_tracking.astype(np.float32)
        )
        return score.astype(np.float32)

    def _trajectory_yaw(self, candidate_trajectories, traj_xy):
        candidate_trajectories = np.asarray(candidate_trajectories)
        if candidate_trajectories.shape[-1] > 2:
            return np.asarray(candidate_trajectories[..., 2], dtype=np.float32)
        dxy = np.diff(traj_xy, axis=1, prepend=traj_xy[:, :1])
        return np.arctan2(dxy[..., 1], dxy[..., 0] + 1e-8).astype(np.float32)

    def _circle_centers(self, xy, yaw, length, width):
        radius = float(width) * 0.6
        offset = max(0.0, float(length) * 0.5 - radius)
        offsets = np.array([-offset, 0.0, offset], dtype=np.float32)
        direction = np.stack([np.cos(yaw), np.sin(yaw)], axis=-1).astype(np.float32)
        centers = xy[..., None, :] + direction[..., None, :] * offsets[None, None, :, None]
        return centers, radius

    def _has_collision_three_circle(self, traj_xy, traj_yaw, predictions):
        k, t = traj_xy.shape[:2]
        t_eval = t if int(self.config.num_frames_eval) <= 0 else min(t, int(self.config.num_frames_eval))
        if predictions is None or len(predictions) == 0:
            return np.zeros((k,), dtype=bool)

        pred = np.asarray(predictions, dtype=np.float32)
        if pred.ndim != 3 or pred.shape[1] == 0:
            return np.zeros((k,), dtype=bool)
        pred = pred[:, :t_eval]
        if pred.shape[1] < t_eval:
            pad_len = t_eval - pred.shape[1]
            pred = np.concatenate([pred, np.repeat(pred[:, -1:], pad_len, axis=1)], axis=1)

        ego_centers, ego_radius = self._circle_centers(
            traj_xy[:, :t_eval],
            traj_yaw[:, :t_eval],
            self.config.ego_length,
            self.config.ego_width,
        )
        other_xy = pred[..., :2]
        other_yaw = pred[..., 2] if pred.shape[-1] > 2 else np.zeros(other_xy.shape[:2], dtype=np.float32)
        other_centers, other_radius = self._circle_centers(
            other_xy,
            other_yaw,
            self.config.agent_length,
            self.config.agent_width,
        )

        threshold_sq = float((ego_radius + other_radius) ** 2)
        collided = np.zeros((k,), dtype=bool)
        other_centers_t = np.transpose(other_centers, (1, 0, 2, 3))
        for i in range(k):
            diff = (
                ego_centers[i, :, :, None, None, :]
                - other_centers_t[:, None, :, :, :]
            )
            dist_sq = np.sum(diff * diff, axis=-1)
            collided[i] = bool(np.any(dist_sq < threshold_sq))
        return collided

    def _progress(self, traj_xy):
        step_delta = np.diff(traj_xy, axis=1)
        path_length = np.linalg.norm(step_delta, axis=-1).sum(axis=-1)
        displacement = np.linalg.norm(traj_xy[:, -1] - traj_xy[:, 0], axis=-1)
        progress = (
            0.7 * np.clip(displacement / max(float(self.config.progress_displacement_scale), 1e-3), 0.0, 1.0)
            + 0.3 * np.clip(path_length / max(float(self.config.progress_path_length_scale), 1e-3), 0.0, 1.0)
        )
        stall_ratio = np.clip(displacement / max(float(self.config.stall_distance_threshold), 1e-3), 0.0, 1.0)
        stall_penalty = -float(self.config.stall_penalty) * (1.0 - stall_ratio)
        stall_penalty[displacement >= float(self.config.stall_distance_threshold)] = 0.0
        return progress + stall_penalty

    def _comfort(self, traj_xy):
        if traj_xy.shape[1] < 3:
            return np.ones((traj_xy.shape[0],), dtype=np.float32)
        speed = np.linalg.norm(np.diff(traj_xy, axis=1), axis=-1) / max(float(self.config.dt), 1e-3)
        accel = np.diff(speed, axis=1) / max(float(self.config.dt), 1e-3)
        max_abs_accel = np.max(np.abs(accel), axis=1, initial=0.0)
        return np.where(max_abs_accel > float(self.config.max_abs_accel), -1.0, 1.0).astype(np.float32)

    def _reference_line_arrays(self, input_data: Dict):
        ref = input_data.get("reference_line", None) if isinstance(input_data, dict) else None
        if not isinstance(ref, dict):
            return None, None, None
        ref_pos = _to_numpy(ref.get("position", None))
        ref_valid = _to_numpy(ref.get("valid_mask", None))
        ref_ori = _to_numpy(ref.get("orientation", None))
        if ref_pos is None or ref_valid is None:
            return None, None, None
        if ref_pos.ndim == 4:
            ref_pos = ref_pos[0]
            ref_valid = ref_valid[0]
            if ref_ori is not None and ref_ori.ndim == 3:
                ref_ori = ref_ori[0]
        return np.asarray(ref_pos), np.asarray(ref_valid).astype(bool), None if ref_ori is None else np.asarray(ref_ori)

    def _ref_tracking(self, traj_xy, input_data, candidate_ref_idx):
        ref_pos, ref_valid, _ = self._reference_line_arrays(input_data)
        if ref_pos is None or ref_valid is None or len(ref_pos) == 0:
            return np.zeros((traj_xy.shape[0],), dtype=np.float32)
        scores = np.zeros((traj_xy.shape[0],), dtype=np.float32)
        for ref_idx in np.unique(candidate_ref_idx):
            idx = int(np.clip(ref_idx, 0, len(ref_pos) - 1))
            pts = ref_pos[idx][ref_valid[idx]]
            if len(pts) < 2:
                continue
            mask = candidate_ref_idx == ref_idx
            cand = traj_xy[mask]
            dist = np.linalg.norm(cand[:, :, None, :] - pts[None, None, :, :], axis=-1)
            min_dist = dist.min(axis=-1)
            mean_excess = np.maximum(0.0, min_dist.mean(axis=-1) - float(self.config.ref_tracking_margin))
            terminal_excess = np.maximum(0.0, min_dist[:, -1] - float(self.config.ref_tracking_margin))
            scores[mask] = -(mean_excess + float(self.config.ref_tracking_terminal_weight) * terminal_excess)
        return scores

    def _wrong_way(self, traj_xy, traj_yaw, input_data, candidate_ref_idx):
        ref_pos, ref_valid, ref_ori = self._reference_line_arrays(input_data)
        if ref_pos is None or ref_valid is None or ref_ori is None or len(ref_pos) == 0:
            return np.zeros((traj_xy.shape[0],), dtype=bool)
        t = traj_xy.shape[1]
        t_eval = t if int(self.config.num_frames_eval) <= 0 else min(t, int(self.config.num_frames_eval))
        wrong = np.zeros((traj_xy.shape[0],), dtype=bool)
        for i in range(traj_xy.shape[0]):
            idx = int(np.clip(candidate_ref_idx[i], 0, len(ref_pos) - 1))
            pts = ref_pos[idx][ref_valid[idx]]
            oris = ref_ori[idx][ref_valid[idx]]
            if len(pts) < 2 or len(oris) < 2:
                continue
            dist = np.linalg.norm(pts[None, :, :] - traj_xy[i, :t_eval, None, :], axis=-1)
            nearest = np.argmin(dist, axis=-1)
            dyaw = _wrap_to_pi(traj_yaw[i, :t_eval] - oris[nearest])
            wrong[i] = bool(np.any(np.abs(dyaw) > float(self.config.max_abs_yaw_diff)))
        return wrong
