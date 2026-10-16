from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import pickle

import numpy as np
import torch

from unitraj.datasets.Pluto_dataset.pluto_utils import PlutoFeature


@dataclass
class RolloutSample:
    pluto_feature: PlutoFeature
    old_group_logits: Dict[str, np.ndarray]
    group_advantage: Dict[str, np.ndarray]
    reward_matrix: Optional[np.ndarray]
    info: Dict


class RolloutBuffer:
    """On-policy rollout buffer for Pluto scorer fine-tuning.

    The buffer only stores the tensors needed for scorer/head updates:
    - current Pluto feature
    - old policy logits
    - group-relative advantage and valid mask
    - optional reward matrix / metadata
    """

    def __init__(
        self,
        buffer_size: int,
        scenario_repeat: int = 1,
        pin_memory: bool = True,
        enable_disk_cache: bool = False,
        cache_dir: str = "rl_buffer_cache",
    ):
        self.buffer_size = int(buffer_size)
        self.scenario_repeat = int(scenario_repeat)
        self.total_size = max(1, self.buffer_size * self.scenario_repeat)
        self.pin_memory = pin_memory
        self.enable_disk_cache = enable_disk_cache
        self.cache_dir = Path(cache_dir)
        if self.enable_disk_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.samples: List[Dict] = []
        self.buffer_pos = 0

    def _pin_recursive(self, data):
        if not torch.cuda.is_available():
            return data
        if isinstance(data, torch.Tensor):
            return data.pin_memory()
        if isinstance(data, dict):
            return {k: self._pin_recursive(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self._pin_recursive(v) for v in data]
        return data

    def _save_to_disk(self, idx: int, sample: Dict) -> None:
        cache_file = self.cache_dir / f"sample_{idx:06d}.pkl"
        with open(cache_file, "wb") as f:
            pickle.dump(sample, f)

    def _load_from_disk(self, idx: int) -> Dict:
        cache_file = self.cache_dir / f"sample_{idx:06d}.pkl"
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    def store(
        self,
        pluto_feature: PlutoFeature,
        old_logits: np.ndarray,
        group_advantage: Dict,
        reward_matrix: Optional[np.ndarray] = None,
        info: Optional[Dict] = None,
    ) -> None:
        sample = {
            "pluto_feature": pluto_feature,
            "old_group_logits": {
                "logits": np.asarray(old_logits, dtype=np.float32),
            },
            "group_advantage": {
                "advantage": np.asarray(group_advantage["advantage"], dtype=np.float32),
                "valid_mask": np.asarray(group_advantage["valid_mask"], dtype=np.bool_),
            },
            "reward_matrix": None if reward_matrix is None else (
                reward_matrix.detach().cpu().numpy()
                if isinstance(reward_matrix, torch.Tensor)
                else np.asarray(reward_matrix, dtype=np.float32)
            ),
            "info": info or {},
        }
        if self.pin_memory and not self.enable_disk_cache:
            sample = self._pin_recursive(sample)

        idx = self.buffer_pos % self.total_size
        if self.enable_disk_cache:
            self._save_to_disk(idx, sample)
            placeholder = {"_cache_idx": idx}
            if len(self.samples) >= self.total_size:
                self.samples[idx] = placeholder
            else:
                self.samples.append(placeholder)
        else:
            if len(self.samples) >= self.total_size:
                self.samples[idx] = sample
            else:
                self.samples.append(sample)
        self.buffer_pos += 1

    def sample(self, batch_size: int) -> List[Dict]:
        batch_size = min(int(batch_size), len(self.samples))
        if batch_size <= 0:
            return []
        indices = np.random.choice(len(self.samples), batch_size, replace=False)
        result = []
        for idx in indices:
            item = self.samples[idx]
            if self.enable_disk_cache and "_cache_idx" in item:
                result.append(self._load_from_disk(item["_cache_idx"]))
            else:
                result.append(item)
        return result

    def get_all_data(self) -> List[Dict]:
        data = []
        for item in self.samples:
            if self.enable_disk_cache and "_cache_idx" in item:
                data.append(self._load_from_disk(item["_cache_idx"]))
            else:
                data.append(item)
        return data

    def should_update(self) -> bool:
        return len(self.samples) >= self.total_size

    def reset(self) -> None:
        self.samples.clear()
        self.buffer_pos = 0
        if self.enable_disk_cache and self.cache_dir.exists():
            for file in self.cache_dir.glob("sample_*.pkl"):
                file.unlink()

    def get_statistics(self) -> Dict[str, float]:
        if len(self.samples) == 0:
            return {
                "size": 0,
                "capacity": self.total_size,
                "avg_reward": 0.0,
                "avg_valid_ratio": 0.0,
            }

        reward_means = []
        valid_ratios = []
        for sample in self.get_all_data():
            reward_matrix = sample.get("reward_matrix")
            if reward_matrix is not None and reward_matrix.size > 0:
                reward_means.append(float(np.mean(reward_matrix)))
            valid_mask = sample["group_advantage"]["valid_mask"]
            valid_ratios.append(float(valid_mask.mean()) if valid_mask.size > 0 else 0.0)

        return {
            "size": len(self.samples),
            "capacity": self.total_size,
            "avg_reward": float(np.mean(reward_means)) if reward_means else 0.0,
            "avg_valid_ratio": float(np.mean(valid_ratios)) if valid_ratios else 0.0,
        }

    def __len__(self) -> int:
        return len(self.samples)
