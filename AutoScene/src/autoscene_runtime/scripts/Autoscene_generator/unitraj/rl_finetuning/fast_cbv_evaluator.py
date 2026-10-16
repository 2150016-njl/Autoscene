from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Dict, Optional

import numpy as np


def _to_numpy(x):
    if x is None:
        return None
    return x.detach().cpu().numpy() if hasattr(x, "detach") else (x.cpu().numpy() if hasattr(x, "cpu") else x)


@dataclass
class FastCBVEvaluatorConfig:
    dt: float = 0.1
    feasibility_progress_weight: float = 2.0
    feasibility_ref_weight: float = 1.0
    feasibility_smoothness_weight: float = 0.15
    progress_displacement_scale: float = 30.0
    progress_path_length_scale: float = 40.0
    stall_distance_threshold: float = 8.0
    stall_penalty: float = 3.0
    ref_tracking_margin: float = 2.0
    ref_tracking_terminal_weight: float = 1.5
    max_abs_accel: float = 8.0


class FastCBVFeasibilityEvaluator:
    """Lightweight CBV feasibility scorer for adversarial trajectory selection.

    This evaluator intentionally avoids expensive map polygon and multi-agent collision
    checks. AV2 adversarial selection wants risky interactions, so collision should not
    be treated as an automatic hard failure here.
    """

    def __init__(self, config: Optional[Dict] = None):
        cfg = dict(config or {})
        valid_keys = {f.name for f in fields(FastCBVEvaluatorConfig)}
        filtered = {k: v for k, v in cfg.items() if k in valid_keys}
        self.config = FastCBVEvaluatorConfig(**filtered)

    def evaluate(self, candidate_trajectories, input_data, candidate_ref_idx=None) -> Dict[str, np.ndarray]:
        candidate_trajectories = _to_numpy(candidate_trajectories)
        input_data = input_data if isinstance(input_data, dict) else getattr(input_data, "data", input_data)
        if candidate_trajectories is None or len(candidate_trajectories) == 0:
            empty = np.zeros((0,), dtype=np.float32)
            return {
                "score": empty,
                "progress": empty,
                "ref_tracking": empty,
                "smoothness": empty,
            }

        traj_xy = np.asarray(candidate_trajectories[..., :2], dtype=np.float32)
        candidate_ref_idx = (
            np.zeros((len(traj_xy),), dtype=np.int64)
            if candidate_ref_idx is None
            else np.asarray(candidate_ref_idx, dtype=np.int64).reshape(-1)
        )
        if len(candidate_ref_idx) != len(traj_xy):
            candidate_ref_idx = np.resize(candidate_ref_idx, len(traj_xy))

        progress = self._progress(traj_xy)
        ref_tracking = self._ref_tracking(traj_xy, input_data, candidate_ref_idx)
        smoothness = self._smoothness(traj_xy)
        score = (
            self.config.feasibility_progress_weight * progress
            + self.config.feasibility_ref_weight * ref_tracking
            + self.config.feasibility_smoothness_weight * smoothness
        ).astype(np.float32)

        return {
            "score": score,
            "progress": progress.astype(np.float32),
            "ref_tracking": ref_tracking.astype(np.float32),
            "smoothness": smoothness.astype(np.float32),
        }

    def _progress(self, traj_xy: np.ndarray) -> np.ndarray:
        step_delta = np.diff(traj_xy, axis=1)
        path_length = np.linalg.norm(step_delta, axis=-1).sum(axis=-1)
        displacement = np.linalg.norm(traj_xy[:, -1] - traj_xy[:, 0], axis=-1)

        progress_from_disp = np.clip(
            displacement / max(float(self.config.progress_displacement_scale), 1e-3),
            0.0,
            1.0,
        )
        progress_from_path = np.clip(
            path_length / max(float(self.config.progress_path_length_scale), 1e-3),
            0.0,
            1.0,
        )
        progress = 0.7 * progress_from_disp + 0.3 * progress_from_path

        stall_ratio = np.clip(
            displacement / max(float(self.config.stall_distance_threshold), 1e-3),
            0.0,
            1.0,
        )
        stall_penalty = -float(self.config.stall_penalty) * (1.0 - stall_ratio)
        stall_penalty[displacement >= float(self.config.stall_distance_threshold)] = 0.0
        return progress + stall_penalty

    def _smoothness(self, traj_xy: np.ndarray) -> np.ndarray:
        if traj_xy.shape[1] < 3:
            return np.zeros((traj_xy.shape[0],), dtype=np.float32)
        speed = np.linalg.norm(np.diff(traj_xy, axis=1), axis=-1) / max(float(self.config.dt), 1e-3)
        accel = np.diff(speed, axis=1) / max(float(self.config.dt), 1e-3)
        max_abs_accel = np.max(np.abs(accel), axis=1, initial=0.0)
        return -np.clip(max_abs_accel / max(float(self.config.max_abs_accel), 1e-3), 0.0, 2.0)

    def _reference_line_arrays(self, input_data: Dict):
        ref = input_data.get("reference_line", None) if isinstance(input_data, dict) else None
        if not isinstance(ref, dict):
            return None, None
        ref_pos = _to_numpy(ref.get("position", None))
        ref_valid = _to_numpy(ref.get("valid_mask", None))
        if ref_pos is None or ref_valid is None:
            return None, None
        if ref_pos.ndim == 4:
            ref_pos = ref_pos[0]
            ref_valid = ref_valid[0]
        return np.asarray(ref_pos), np.asarray(ref_valid).astype(bool)

    def _ref_tracking(self, traj_xy: np.ndarray, input_data: Dict, candidate_ref_idx: np.ndarray) -> np.ndarray:
        ref_pos, ref_valid = self._reference_line_arrays(input_data)
        if ref_pos is None or ref_valid is None or len(ref_pos) == 0:
            return np.zeros((traj_xy.shape[0],), dtype=np.float32)

        scores = np.zeros((traj_xy.shape[0],), dtype=np.float32)
        for ref_idx in np.unique(candidate_ref_idx):
            idx = int(np.clip(ref_idx, 0, len(ref_pos) - 1))
            pts = ref_pos[idx][ref_valid[idx]]
            if len(pts) < 2:
                continue
            cand_mask = candidate_ref_idx == ref_idx
            cand_xy = traj_xy[cand_mask]
            if len(cand_xy) == 0:
                continue
            dist = np.linalg.norm(cand_xy[:, :, None, :] - pts[None, None, :, :], axis=-1)
            min_dist = dist.min(axis=-1)
            mean_excess = np.maximum(0.0, min_dist.mean(axis=-1) - float(self.config.ref_tracking_margin))
            terminal_excess = np.maximum(0.0, min_dist[:, -1] - float(self.config.ref_tracking_margin))
            scores[cand_mask] = -(
                mean_excess + float(self.config.ref_tracking_terminal_weight) * terminal_excess
            ).astype(np.float32)
        return scores
