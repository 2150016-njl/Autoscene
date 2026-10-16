from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


class GRPOAdvantageComputer:
    """Group Relative Policy Optimization advantage helper for planner candidates."""

    def __init__(
        self,
        filter_invalid: bool = True,
        normalize_per_group: bool = True,
        min_valid_ratio: float = 0.05,
        clip_advantage: Optional[Tuple[float, float]] = (-5.0, 5.0),
    ):
        self.filter_invalid = filter_invalid
        self.normalize_per_group = normalize_per_group
        self.min_valid_ratio = min_valid_ratio
        self.clip_advantage = clip_advantage

    def compute_group_advantage(
        self,
        returns: np.ndarray,
        collision_mask: np.ndarray,
        off_road_mask: np.ndarray,
        valid_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        if returns.ndim != 2:
            raise ValueError(f"returns must be (R, M), got {returns.shape}")

        R, M = returns.shape
        final_valid_mask = np.ones((R, M), dtype=bool)
        if self.filter_invalid:
            final_valid_mask &= ~collision_mask.astype(bool)
            final_valid_mask &= ~off_road_mask.astype(bool)
        if valid_mask is not None:
            final_valid_mask &= valid_mask.astype(bool)

        advantages = np.zeros((R, M), dtype=np.float32)
        group_stats = []
        min_valid_threshold = max(1, int(np.ceil(M * self.min_valid_ratio)))

        for r in range(R):
            region_valid = final_valid_mask[r]
            num_valid = int(region_valid.sum())
            if num_valid < min_valid_threshold:
                final_valid_mask[r, :] = False
                group_stats.append({
                    "region_idx": r,
                    "num_valid": num_valid,
                    "status": "too_few_valid",
                })
                continue

            valid_returns = returns[r, region_valid]
            if self.normalize_per_group:
                mean_return = float(valid_returns.mean())
                std_return = max(0.5, float(valid_returns.std()))
                group_adv = (valid_returns - mean_return) / std_return
            else:
                group_adv = valid_returns

            advantages[r, region_valid] = group_adv.astype(np.float32)
            group_stats.append({
                "region_idx": r,
                "num_valid": num_valid,
                "mean_return": float(valid_returns.mean()),
                "std_return": float(valid_returns.std()),
                "min_advantage": float(group_adv.min()),
                "max_advantage": float(group_adv.max()),
                "status": "ok",
            })

        if self.clip_advantage is not None:
            advantages = np.clip(
                advantages,
                self.clip_advantage[0],
                self.clip_advantage[1],
            )

        return {
            "advantage": advantages,
            "valid_mask": final_valid_mask,
            "group_stats": group_stats,
        }

    def compute_advantage_from_flat(
        self,
        returns: np.ndarray,
        collision_mask: np.ndarray,
        off_road_mask: np.ndarray,
        num_regions: int,
        num_modes: int,
        valid_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        reshape = lambda x: np.asarray(x).reshape(num_regions, num_modes)
        return self.compute_group_advantage(
            reshape(returns),
            reshape(collision_mask),
            reshape(off_road_mask),
            None if valid_mask is None else reshape(valid_mask),
        )

    def get_statistics(self, advantage_result: Dict[str, np.ndarray]) -> Dict[str, float]:
        advantages = advantage_result["advantage"]
        valid_mask = advantage_result["valid_mask"]