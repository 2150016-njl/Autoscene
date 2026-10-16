from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Dict, Optional

import numpy as np

from unitraj.closeloop.pluto.trajectory_evaluator import TrajectoryEvaluator
from unitraj.rl_finetuning.fast_cbv_evaluator import FastCBVFeasibilityEvaluator


def _to_numpy(x):
    if x is None:
        return None
    return x.detach().cpu().numpy() if hasattr(x, "detach") else (x.cpu().numpy() if hasattr(x, "cpu") else x)


@dataclass
class CBVRewardConfig:
    ego_risk_weight: float = 3.0
    interaction_weight: float = 4.0
    approach_weight: float = 2.0
    cbv_feasibility_weight: float = 0.5
    risk_distance_threshold: float = 10.0
    interaction_distance_threshold: float = 8.0
    ttc_horizon_s: float = 5.0
    feasibility_progress_weight: float = 0.5
    feasibility_ref_weight: float = 1.0
    use_full_feasibility_evaluator: bool = False

class CBVAdversarialReward:
    """Adversarial reward for a critical background vehicle (CBV).

    The reward is designed for safe-critical scenario generation:
    - `ego_risk`: encourage lower TTC / smaller future clearance to the ego vehicle
    - `interaction`: encourage sustained interaction instead of trivial drive-away
    - `cbv_feasibility`: keep the CBV on-road, roughly on reference, and moving naturally
    """

    def __init__(self, config: Optional[Dict] = None):
        cfg = dict(config or {})
        if "cbv_adversarial_use_full_feasibility" in cfg and "use_full_feasibility_evaluator" not in cfg:
            cfg["use_full_feasibility_evaluator"] = bool(cfg["cbv_adversarial_use_full_feasibility"])
        valid_keys = {f.name for f in fields(CBVRewardConfig)}
        filtered = {k: v for k, v in cfg.items() if k in valid_keys}
        self.config = CBVRewardConfig(**filtered)
        self.feasibility_evaluator = (
            TrajectoryEvaluator(cfg)
            if self.config.use_full_feasibility_evaluator
            else FastCBVFeasibilityEvaluator(cfg)
        )

    def _compute_yaw(self, traj_xy: np.ndarray, fallback_yaw: Optional[np.ndarray] = None) -> np.ndarray:
        if fallback_yaw is not None:
            return fallback_yaw
        diff = np.diff(traj_xy, axis=1, prepend=traj_xy[:, :1])
        return np.arctan2(diff[..., 1], diff[..., 0])

    def _ego_risk(self, cbv_xy: np.ndarray, ego_xy: np.ndarray) -> np.ndarray:
        min_t = min(cbv_xy.shape[1], ego_xy.shape[0])
        if min_t == 0:
            return np.zeros((cbv_xy.shape[0],), dtype=np.float32)

        cbv_xy = cbv_xy[:, :min_t]
        ego_xy = ego_xy[:min_t]
        dist = np.linalg.norm(cbv_xy - ego_xy[None, :, :], axis=-1)
        min_dist = dist.min(axis=-1) # 每条轨迹距主车最短距离
        dist_score = np.clip(
            (self.config.risk_distance_threshold - min_dist) / max(self.config.risk_distance_threshold, 1e-3),
            0.0,
            1.0,
        )

        ttc_score = np.zeros_like(dist_score, dtype=np.float32)
        for i in range(cbv_xy.shape[0]):
            closing = dist[i, :-1] - dist[i, 1:]
            positive = closing > 1e-3
            if not np.any(positive):
                continue
            closing_speed = float(np.max(closing[positive]) / 0.1)
            ttc = float(min_dist[i] / max(closing_speed, 1e-3))
            ttc_score[i] = float(np.clip((self.config.ttc_horizon_s - ttc) / self.config.ttc_horizon_s, 0.0, 1.0))
        return dist_score + ttc_score

    def _interaction(self, cbv_xy: np.ndarray, ego_xy: np.ndarray) -> np.ndarray:
        min_t = min(cbv_xy.shape[1], ego_xy.shape[0])
        if min_t == 0:
            return np.zeros((cbv_xy.shape[0],), dtype=np.float32)
        dist = np.linalg.norm(cbv_xy[:, :min_t] - ego_xy[None, :min_t, :], axis=-1)
        sustained_overlap = (dist < self.config.interaction_distance_threshold).mean(axis=-1)
        terminal_proximity = np.clip(
            (self.config.interaction_distance_threshold - dist[:, -1]) / max(self.config.interaction_distance_threshold, 1e-3),
            0.0,
            1.0,
        )
        return sustained_overlap.astype(np.float32) + terminal_proximity.astype(np.float32)

    def _cbv_feasibility(
        self,
        candidate_trajectories: np.ndarray,
        input_data: Dict,
        predictions=None,
        candidate_ref_idx=None,
    ) -> Dict[str, np.ndarray]:
        traj_xy = candidate_trajectories[..., :2]
        traj_yaw = np.arctan2(candidate_trajectories[..., 3], candidate_trajectories[..., 2]) if candidate_trajectories.shape[-1] > 2 else self._compute_yaw(traj_xy)
        # pred_0 = None if predictions is None or len(predictions) == 0 else predictions[0]
        pred_others = predictions # 不包含自车

        if isinstance(self.feasibility_evaluator, FastCBVFeasibilityEvaluator):
            fast = self.feasibility_evaluator.evaluate(
                candidate_trajectories,
                input_data,
                candidate_ref_idx=candidate_ref_idx,
            )
            zeros_bool = np.zeros((len(candidate_trajectories),), dtype=np.bool_)
            zeros_float = np.zeros((len(candidate_trajectories),), dtype=np.float32)
            return {
                "score": fast["score"].astype(np.float32),
                "collision": zeros_bool,
                "out_of_drivable": zeros_float,
                "wrong_way": zeros_bool,
                "progress": fast["progress"].astype(np.float32),
                "ref_tracking": fast["ref_tracking"].astype(np.float32),
            }

        collision = self.feasibility_evaluator._has_collision(traj_xy, traj_yaw, input_data, pred_others)
        out_of_drivable = self.feasibility_evaluator._out_of_drivable(traj_xy, traj_yaw, input_data)
        wrong_way = self.feasibility_evaluator._wrong_way(traj_xy, traj_yaw, input_data, candidate_ref_idx)
        progress = self.feasibility_evaluator._progress(traj_xy)
        ref_tracking = self.feasibility_evaluator._ref_tracking(
            traj_xy,
            input_data,
            np.zeros((len(candidate_trajectories),), dtype=np.int64)
            if candidate_ref_idx is None else candidate_ref_idx,
        )

        feasibility = (
            -50.0 * collision.astype(np.float32)
            -40.0 * out_of_drivable.astype(np.float32)
            -30.0 * wrong_way.astype(np.float32)
            + self.config.feasibility_progress_weight * progress.astype(np.float32)
            + self.config.feasibility_ref_weight * ref_tracking.astype(np.float32)
        )
        return {
            "score": feasibility.astype(np.float32),
            "collision": collision.astype(np.bool_),
            "out_of_drivable": out_of_drivable.astype(np.float32),
            "wrong_way": wrong_way.astype(np.bool_),
            "progress": progress.astype(np.float32),
            "ref_tracking": ref_tracking.astype(np.float32),
        }

    def evaluate(
        self,
        candidate_trajectories,
        input_data,
        ego_future,
        predictions=None,
        candidate_ref_idx=None,
    ) -> Dict[str, np.ndarray]:
        candidate_trajectories = _to_numpy(candidate_trajectories)
        input_data = input_data if isinstance(input_data, dict) else getattr(input_data, "data", input_data)
        ego_future = _to_numpy(ego_future)
        predictions = _to_numpy(predictions)
        candidate_ref_idx = None if candidate_ref_idx is None else np.asarray(candidate_ref_idx, dtype=np.int64)

        if candidate_trajectories is None or len(candidate_trajectories) == 0:
            empty = np.zeros((0,), dtype=np.float32)
            return {
                "total": empty,
                "ego_risk": empty,
                "interaction": empty,
                "cbv_feasibility": empty,
            }

        cbv_xy = np.asarray(candidate_trajectories[..., :2], dtype=np.float32)
        ego_xy = np.asarray(ego_future[..., :2], dtype=np.float32)
        ego_risk = self._ego_risk(cbv_xy, ego_xy)
        interaction = self._interaction(cbv_xy, ego_xy)
        approach = self._approach(cbv_xy, ego_xy)
        feasibility = self._cbv_feasibility(
            np.asarray(candidate_trajectories),
            input_data,
            predictions=predictions,
            candidate_ref_idx=candidate_ref_idx,
        )
        total = (
            self.config.ego_risk_weight * ego_risk
            + self.config.interaction_weight * interaction
            + self.config.approach_weight * approach
            + self.config.cbv_feasibility_weight * feasibility["score"]
        ).astype(np.float32)
        return {
            "total": total,
            "ego_risk": ego_risk.astype(np.float32),
            "interaction": interaction.astype(np.float32),
            "approach": approach.astype(np.float32),
            "cbv_feasibility": feasibility["score"].astype(np.float32),
            "collision": feasibility["collision"],
            "out_of_drivable": feasibility["out_of_drivable"],
            "wrong_way": feasibility["wrong_way"],
            "progress": feasibility["progress"],
            "ref_tracking": feasibility["ref_tracking"],
        }

    def _approach(self, cbv_xy: np.ndarray, target_xy: np.ndarray) -> np.ndarray:
        min_t = min(cbv_xy.shape[1], target_xy.shape[0])
        if min_t < 3:
            return np.zeros((cbv_xy.shape[0],), dtype=np.float32)

        cbv = cbv_xy[:, :min_t]
        target = target_xy[:min_t]

        dist = np.linalg.norm(cbv - target[None, :, :], axis=-1)
        min_idx = dist.argmin(axis=-1)

        approach = np.zeros((cbv.shape[0],), dtype=np.float32)

        for i in range(cbv.shape[0]):
            k = int(np.clip(min_idx[i], 1, min_t - 2))

            v_cbv = (cbv[i, k + 1] - cbv[i, k - 1]) / (2.0 * 0.1)
            v_target = (target[k + 1] - target[k - 1]) / (2.0 * 0.1)

            rel_pos = target[k] - cbv[i, k]
            rel_norm = np.linalg.norm(rel_pos)

            if rel_norm < 1e-3:
                approach[i] = 1.0
                continue

            rel_dir = rel_pos / rel_norm
            rel_vel = v_cbv - v_target

            closing_speed = float(np.dot(rel_vel, rel_dir))
            approach[i] = float(np.clip(closing_speed / 5.0, 0.0, 1.0))

        return approach.astype(np.float32)