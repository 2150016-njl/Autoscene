import logging
import os
import yaml
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning.pytorch as pl
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from metadrive.envs.scenario_env import ScenarioEnv
from unitraj.closeloop.pluto.pluto_policy import PlutoPolicy
from unitraj.datasets.Pluto_dataset.pluto_utils import PlutoFeature
from unitraj.models import build_model
from unitraj.renderer.head_renderer import HeadTopDownRenderer
from unitraj.rl_finetuning.cbv_adversarial_reward import CBVAdversarialReward
from unitraj.rl_finetuning.grpo_advantage import GRPOAdvantageComputer
from unitraj.rl_finetuning.rift_trainer import RIFTTrainer
from unitraj.rl_finetuning.rollout_buffer import RolloutBuffer
from unitraj.utils.utils import set_seed


logger = logging.getLogger(__name__)


def _to_numpy(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _to_pluto_feature(feature: Any) -> PlutoFeature:
    if isinstance(feature, PlutoFeature):
        pf = feature
    elif isinstance(feature, dict):
        pf = PlutoFeature.deserialize(feature) if "data" in feature else PlutoFeature(data=feature)
    else:
        raise TypeError(f"Unsupported pluto_feature type: {type(feature)}")
    return pf.to_feature_tensor().to_device(torch.device("cpu"))


class RLBufferDataset(Dataset):
    def __init__(self, data_list: List[Dict]):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class CBVRLDataModule(pl.LightningDataModule):
    """Lightweight DataModule adapted to benchmark RolloutBuffer samples."""

    def __init__(self, cfg, buffer: RolloutBuffer, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.buffer = buffer
        self.device = device
        self.train_batch_size = int(cfg.get("rl_batch_size", 8))
        self.val_batch_size = int(cfg.get("rl_val_batch_size", self.train_batch_size))
        self.num_workers = int(cfg.get("rl_num_workers", 0))
        self.pin_memory = bool(cfg.get("rl_pin_memory", True))
        self.train_ratio = float(cfg.get("rl_train_ratio", 0.9))
        self.shuffle = bool(cfg.get("rl_shuffle", True))
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None):
        _ = stage
        all_data = self.buffer.get_all_data()
        if len(all_data) == 0:
            raise ValueError("RolloutBuffer is empty.")
        if len(all_data) == 1:
            train_data = all_data
            val_data = all_data
        else:
            train_size = max(1, min(int(len(all_data) * self.train_ratio), len(all_data) - 1))
            train_data = all_data[:train_size]
            val_data = all_data[train_size:]
        self.train_dataset = RLBufferDataset(train_data)
        self.val_dataset = RLBufferDataset(val_data)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=self.shuffle,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def collate_fn(self, batch: List[Dict]) -> Dict[str, Any]:
        features = PlutoFeature.collate([sample["pluto_feature"] for sample in batch]).to_device(self.device)
        R_max = int(features.data["reference_line"]["position"].shape[1])
        batched = {
            "cur_pluto_feature_torch": features,
            "group_advantage_torch": self._collate_group_advantage([s["group_advantage"] for s in batch], R_max),
            "group_advantage_mask_torch": self._collate_advantage_mask([s["group_advantage"] for s in batch], R_max),
            "old_group_logits_torch": self._collate_old_logits([s["old_group_logits"] for s in batch], R_max),
        }
        reward_list = [s.get("reward_matrix") for s in batch]
        if not all(r is None for r in reward_list):
            batched["reward_matrix"] = self._collate_reward_matrix(reward_list, R_max)
        return batched

    def _pad_2d(self, arr: np.ndarray, R_max: int, M: int, pad_value):
        out = np.full((R_max, M), pad_value, dtype=arr.dtype)
        R_i = min(int(arr.shape[0]), R_max)
        M_i = min(int(arr.shape[1]), M)
        out[:R_i, :M_i] = arr[:R_i, :M_i]
        return out

    def _collate_group_advantage(self, advantages: List[Dict], R_max: int) -> torch.Tensor:
        first = next(a for a in advantages if a is not None)
        M = int(first["advantage"].shape[1])
        stacked = []
        for item in advantages:
            arr = np.zeros((R_max, M), dtype=np.float32) if item is None else self._pad_2d(
                np.asarray(item["advantage"], dtype=np.float32),
                R_max,
                M,
                0.0,
            )
            stacked.append(arr)
        return torch.from_numpy(np.stack(stacked, axis=0)).float()

    def _collate_advantage_mask(self, advantages: List[Dict], R_max: int) -> torch.Tensor:
        first = next(a for a in advantages if a is not None)
        M = int(first["valid_mask"].shape[1])
        stacked = []
        for item in advantages:
            arr = np.zeros((R_max, M), dtype=np.bool_) if item is None else self._pad_2d(
                np.asarray(item["valid_mask"], dtype=np.bool_),
                R_max,
                M,
                False,
            )
            stacked.append(arr)
        return torch.from_numpy(np.stack(stacked, axis=0)).bool()

    def _collate_old_logits(self, logits_list: List[Dict], R_max: int) -> torch.Tensor:
        first = next(l for l in logits_list if l is not None)
        M = int(first["logits"].shape[1])
        stacked = []
        for item in logits_list:
            arr = np.zeros((R_max, M), dtype=np.float32) if item is None else self._pad_2d(
                np.asarray(item["logits"], dtype=np.float32),
                R_max,
                M,
                0.0,
            )
            stacked.append(arr)
        return torch.from_numpy(np.stack(stacked, axis=0)).float()

    def _collate_reward_matrix(self, rm_list: List[np.ndarray], R_max: int) -> torch.Tensor:
        first = next(r for r in rm_list if r is not None)
        M = int(first.shape[1])
        stacked = []
        for item in rm_list:
            arr = np.zeros((R_max, M), dtype=np.float32) if item is None else self._pad_2d(
                np.asarray(item, dtype=np.float32),
                R_max,
                M,
                0.0,
            )
            stacked.append(arr)
        return torch.from_numpy(np.stack(stacked, axis=0)).float()


class CBVRLFinetuner:
    """End-to-end CBV scorer fine-tuning entrypoint for benchmark."""

    def __init__(self, cfg):
        self.cfg = cfg
        use_gpu = torch.cuda.is_available() and int(cfg.get("devices", 1)) > 0
        self.device = torch.device("cuda" if use_gpu else "cpu")

        logger.info("Building base model...")
        self.model = build_model(cfg).to(self.device)
        self._load_initial_weights()
        self.model.eval()

        logger.info("Initializing CBV reward / advantage modules...")
        self.reward_fn = CBVAdversarialReward(cfg.get("reward", {}))
        grpo_cfg = dict(cfg.get("grpo", {}))
        self.advantage_computer = GRPOAdvantageComputer(
            filter_invalid=bool(grpo_cfg.get("filter_invalid", True)),
            normalize_per_group=bool(grpo_cfg.get("normalize_per_group", True)),
            min_valid_ratio=float(grpo_cfg.get("min_valid_ratio", 0.05)),
            clip_advantage=tuple(grpo_cfg.get("clip_advantage", (-5.0, 5.0))),
        )

        self.buffer = RolloutBuffer(
            buffer_size=int(cfg.get("rl_buffer_size", 256)),
            scenario_repeat=int(cfg.get("rl_scenario_repeat", 1)),
            enable_disk_cache=bool(cfg.get("rl_buffer_disk_cache", False)),
            cache_dir=str(cfg.get("cache_dir")),
        )
        self.offroad_invalid_threshold = float(cfg.get("offroad_invalid_threshold", 0.5))
        self.checkpoint_dir = Path(cfg.get("save_dir"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.num_updates = 0

    def _reshape_group_logits(
        self,
        old_logits: Any,
        candidate_trajectories: Any,
        num_refs: Optional[int] = None,
        num_modes: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        logits = np.asarray(_to_numpy(old_logits), dtype=np.float32)
        traj = np.asarray(_to_numpy(candidate_trajectories), dtype=np.float32)

        if logits.ndim == 2:
            R, M = logits.shape
        elif logits.ndim == 1 and num_refs is not None and num_modes is not None and logits.size == num_refs * num_modes:
            R, M = int(num_refs), int(num_modes)
            logits = logits.reshape(R, M)
        else:
            raise ValueError(
                f"old_logits must be (R, M) or flat with num_refs/num_modes. Got shape={logits.shape}"
            )

        if traj.ndim == 4:
            if traj.shape[0] != R or traj.shape[1] != M:
                raise ValueError(
                    f"candidate_trajectories shape {traj.shape} does not match logits {(R, M)}"
                )
            return logits, traj
        if traj.ndim == 3 and traj.shape[0] == R * M:
            return logits, traj.reshape(R, M, traj.shape[1], traj.shape[2])
        raise ValueError(
            "candidate_trajectories must be (R, M, T, D) or flat (R*M, T, D); "
            f"got shape={traj.shape}"
        )

    def _prepare_candidate_ref_idx(self, candidate_ref_idx: Any, R: int, M: int) -> np.ndarray:
        if candidate_ref_idx is None:
            return np.repeat(np.arange(R, dtype=np.int64)[:, None], M, axis=1)
        ref_idx = np.asarray(_to_numpy(candidate_ref_idx), dtype=np.int64)
        if ref_idx.ndim == 2:
            if ref_idx.shape != (R, M):
                raise ValueError(f"candidate_ref_idx shape {ref_idx.shape} does not match {(R, M)}")
            return ref_idx
        if ref_idx.ndim == 1 and ref_idx.size == R:
            return np.repeat(ref_idx[:, None], M, axis=1)
        if ref_idx.ndim == 1 and ref_idx.size == R * M:
            return ref_idx.reshape(R, M)
        raise ValueError(
            "candidate_ref_idx must be (R,), (R,M), or flat (R*M,); "
            f"got shape={ref_idx.shape}"
        )

    def _reference_region_valid_mask(self, pluto_feature: PlutoFeature, ref_idx_matrix: np.ndarray) -> np.ndarray:
        reference_line = pluto_feature.data.get("reference_line", None)
        if reference_line is None or "valid_mask" not in reference_line:
            return np.ones_like(ref_idx_matrix, dtype=np.bool_)

        ref_valid = _to_numpy(reference_line["valid_mask"])
        if ref_valid is None:
            return np.ones_like(ref_idx_matrix, dtype=np.bool_)
        ref_valid = np.asarray(ref_valid)
        if ref_valid.ndim >= 2:
            per_region = ref_valid.any(axis=-1)
        else:
            per_region = ref_valid.astype(bool)
        per_region = np.asarray(per_region, dtype=np.bool_).reshape(-1)
        region_indices = np.clip(ref_idx_matrix, 0, max(len(per_region) - 1, 0))
        return per_region[region_indices]

    def _normalize_reward_matrix(self, reward_matrix: Any, R: int, M: int) -> np.ndarray:
        reward = np.asarray(_to_numpy(reward_matrix), dtype=np.float32)
        if reward.shape == (R, M):
            return reward
        if reward.ndim == 1 and reward.size == R * M:
            return reward.reshape(R, M)
        raise ValueError(f"reward_matrix must be {(R, M)} or flat ({R*M},), got {reward.shape}")

    def _build_reward_metrics(
        self,
        candidate_trajectories: np.ndarray,
        pluto_feature: PlutoFeature,
        ego_future: Any,
        candidate_ref_idx: np.ndarray,
        predictions: Any = None,
        reward_matrix: Any = None,
        reward_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, np.ndarray]:
        flat_traj = candidate_trajectories.reshape(-1, candidate_trajectories.shape[-2], candidate_trajectories.shape[-1])
        flat_ref = candidate_ref_idx.reshape(-1)

        if reward_metrics is not None:
            metrics = {
                k: np.asarray(_to_numpy(v))
                for k, v in reward_metrics.items()
            }
        else:
            metrics = self.reward_fn.evaluate(
                flat_traj,
                pluto_feature.data,
                _to_numpy(ego_future),
                predictions=_to_numpy(predictions),
                candidate_ref_idx=flat_ref,
            )
        if reward_matrix is not None:
            metrics["total"] = self._normalize_reward_matrix(
                reward_matrix,
                candidate_trajectories.shape[0],
                candidate_trajectories.shape[1],
            ).reshape(-1)
        return metrics

    def rollout_step(
        self,
        pluto_feature: Any,
        old_logits: Any,
        candidate_trajectories: Any,
        ego_future: Any,
        candidate_ref_idx: Any = None,
        predictions: Any = None,
        reward_matrix: Any = None,
        reward_metrics: Optional[Dict[str, Any]] = None,
        info: Optional[Dict[str, Any]] = None,
        num_refs: Optional[int] = None,
        num_modes: Optional[int] = None,
    ) -> bool:
        feature = _to_pluto_feature(pluto_feature)
        logits_2d, grouped_traj = self._reshape_group_logits(
            old_logits,
            candidate_trajectories,
            num_refs=num_refs,
            num_modes=num_modes,
        )
        R, M = logits_2d.shape
        ref_idx_matrix = self._prepare_candidate_ref_idx(candidate_ref_idx, R, M)

        metrics = self._build_reward_metrics(
            grouped_traj,
            feature,
            ego_future,
            ref_idx_matrix,
            predictions=predictions,
            reward_matrix=reward_matrix,
            reward_metrics=reward_metrics,
        )
        reward_matrix_2d = np.asarray(metrics["total"], dtype=np.float32).reshape(R, M)
        collision_mask = np.asarray(metrics["collision"], dtype=np.bool_).reshape(R, M)
        off_road_mask = np.asarray(metrics["out_of_drivable"], dtype=np.float32).reshape(R, M) > self.offroad_invalid_threshold
        wrong_way_mask = np.asarray(metrics["wrong_way"], dtype=np.bool_).reshape(R, M)
        region_valid_mask = self._reference_region_valid_mask(feature, ref_idx_matrix)
        advantage_result = self.advantage_computer.compute_group_advantage(
            returns=reward_matrix_2d,
            collision_mask=collision_mask,
            off_road_mask=off_road_mask,
            valid_mask=region_valid_mask & (~wrong_way_mask),
        )
        if not advantage_result["valid_mask"].any():
            logger.debug("Skip rollout sample due to empty valid_mask.")
            return False

        sample_info = dict(info or {})
        sample_info.update(
            {
                "candidate_ref_idx": ref_idx_matrix.astype(np.int64),
                "collision_mask": collision_mask.astype(np.bool_),
                "off_road_mask": off_road_mask.astype(np.bool_),
                "wrong_way_mask": wrong_way_mask.astype(np.bool_),
                "reward_terms": {
                    "ego_risk": np.asarray(metrics.get("ego_risk"), dtype=np.float32).reshape(R, M),
                    "interaction": np.asarray(metrics.get("interaction"), dtype=np.float32).reshape(R, M),
                    "cbv_feasibility": np.asarray(metrics.get("cbv_feasibility"), dtype=np.float32).reshape(R, M),
                },
            }
        )
        self.buffer.store(
            pluto_feature=feature,
            old_logits=logits_2d,
            group_advantage=advantage_result,
            reward_matrix=reward_matrix_2d,
            info=sample_info,
        )
        return True

    def _iter_rollout_samples(self, payload: Any) -> List[Dict[str, Any]]:
        if payload is None:
            return []
        if isinstance(payload, list):
            return [p for p in payload if isinstance(p, dict)]
        if isinstance(payload, dict):
            if "cbv_rollout_samples" in payload and isinstance(payload["cbv_rollout_samples"], list):
                return [p for p in payload["cbv_rollout_samples"] if isinstance(p, dict)]
            if "default_agent" in payload and isinstance(payload["default_agent"], dict):
                nested = payload["default_agent"].get("cbv_rollout_samples", [])
                return [p for p in nested if isinstance(p, dict)]
        return []

    def collect_rollout_samples(self, info: Any) -> int:
        samples = self._iter_rollout_samples(info)
        num_collected = 0
        for sample in samples:
            try:
                if self.rollout_step(**sample):
                    num_collected += 1
            except Exception as exc:
                logger.warning("Skip malformed cbv rollout sample: %s", exc)
        return num_collected

    def run_finetuning_loop(self) -> bool:
        if len(self.buffer) == 0:
            logger.warning("Buffer is empty, skipping RL update.")
            return False

        stats = self.buffer.get_statistics()
        logger.info(
            "Starting CBV RL update: size=%s avg_reward=%.4f valid_ratio=%.4f",
            stats["size"],
            stats["avg_reward"],
            stats["avg_valid_ratio"],
        )

        datamodule = CBVRLDataModule(self.cfg, self.buffer, self.device)
        trainer_module = RIFTTrainer(
            model=self.model,
            lr=float(self.cfg.get("rl_lr", 1e-4)),
            weight_decay=float(self.cfg.get("rl_weight_decay", 1e-4)),
            clip_ratio=float(self.cfg.get("rl_clip_ratio", 0.2)),
            entropy_coef=float(self.cfg.get("rl_entropy_coef", 0.05)),
            trainable_layers=self.cfg.get("trainable_layers", ["planning_decoder.pi_head"]),
            kl_warmup_steps=int(self.cfg.get("rl_kl_warmup_steps", 20)),
            kl_warn=float(self.cfg.get("rl_kl_warn", 0.15)),
            kl_stop=float(self.cfg.get("rl_kl_stop", 0.30)),
        )
        trainer = pl.Trainer(
            max_epochs=int(self.cfg.get("rl_epochs", 5)),
            accelerator="gpu" if self.device.type == "cuda" else "cpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            num_sanity_val_steps=0,
            log_every_n_steps=1,
            enable_model_summary=False,
            gradient_clip_val=float(self.cfg.get("rl_grad_clip", 0.5)),
            gradient_clip_algorithm="norm",
        )
        torch.set_grad_enabled(True)
        trainer.fit(model=trainer_module, datamodule=datamodule)
        torch.set_grad_enabled(False)

        self.model.load_state_dict(trainer_module.model.state_dict(), strict=False)
        self.model.eval()
        self.num_updates += 1
        self._save_checkpoint()
        self.buffer.reset()
        return True

    def _save_checkpoint(self) -> None:
        ckpt = {
            "state_dict": self.model.state_dict(),
            "num_updates": self.num_updates,
            "trainable_layers": list(self.cfg.get("trainable_layers", ["planning_decoder.pi_head"])),
        }
        latest_path = self.checkpoint_dir / "latest_cbv_rl.ckpt"
        history_path = self.checkpoint_dir / f"cbv_rl_update_{self.num_updates}.ckpt"
        torch.save(ckpt, latest_path)
        torch.save(ckpt, history_path)
        logger.info("Saved CBV RL checkpoint to %s", latest_path)

    def _resolve_model_ckpt_path(self) -> Optional[Path]:
        resume_candidates = [
            self.cfg.get("rl_resume_ckpt", None),
            self.cfg.get("resume_rl_ckpt", None),
        ]
        for path in resume_candidates:
            if path and str(path).lower() != "null":
                candidate = Path(path)
                if candidate.exists():
                    return candidate

        latest_rl_ckpt = Path(self.cfg.get("save_dir")) / "latest_cbv_rl.ckpt"
        if latest_rl_ckpt.exists():
            return latest_rl_ckpt

        base_candidates = [
            self.cfg.get("ckpt_path", None),
            self.cfg.get("init_from", None),
        ]
        for path in base_candidates:
            if path and str(path).lower() != "null":
                candidate = Path(path)
                if candidate.exists():
                    return candidate
        return None

    def _load_initial_weights(self) -> None:
        ckpt_path = self._resolve_model_ckpt_path()
        if ckpt_path is None:
            logger.warning("No model checkpoint found for CBV RL fine-tuning, using random-initialized weights.")
            return

        logger.info("Loading model weights from %s", ckpt_path)
        ckpt = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
        if not isinstance(state_dict, dict):
            raise ValueError(f"Unsupported checkpoint format in {ckpt_path}")

        normalized_state = {}
        for key, value in state_dict.items():
            if key.startswith("model."):
                normalized_state[key.replace("model.", "", 1)] = value
            else:
                normalized_state[key] = value

        missing_keys, unexpected_keys = self.model.load_state_dict(normalized_state, strict=False)
        if missing_keys:
            logger.info("Checkpoint missing keys: %s", missing_keys[:20])
        if unexpected_keys:
            logger.info("Checkpoint unexpected keys: %s", unexpected_keys[:20])


def main():
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "rl_finetuning.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        base_raw = yaml.safe_load(f)
    cfg = OmegaConf.create(base_raw)
    method = cfg.get("method", "Pluto")
    method_file = os.path.join(os.path.dirname(config_path), "method", f"{method}.yaml")
    with open(method_file, 'r', encoding='utf-8') as f:
        method_cfg = yaml.safe_load(f)
    cfg = OmegaConf.merge(cfg, OmegaConf.create(method_cfg))
    OmegaConf.set_struct(cfg, False)  # Open the struct
    set_seed(cfg.seed)
    os.environ["WANDB_MODE"] = "offline"
    os.environ["UNITRAJ_RL_FINETUNING_MODE"] = "1" if bool(cfg.get("rl_finetuning_mode", False)) else "0"
    os.environ["UNITRAJ_ADV_CONTROL"] = "1" if bool(cfg.get("adv_control", False)) else "0"
    os.environ["UNITRAJ_ADV_NUM"] = str(int(cfg.get("adv_num", 2)))

    def create_env():
        env = ScenarioEnv(
            {
                "use_render": False,
                "agent_policy": PlutoPolicy,
                "manual_control": False,
                "log_level": logging.CRITICAL,
                "num_scenarios": 6,
                "sequential_seed": False,
                "start_scenario_index": 0,
                "horizon": 100,
                "vehicle_config": dict(
                    show_navi_mark=True,
                    show_line_to_dest=True,
                    show_dest_mark=True,
                    no_wheel_friction=False,
                ),
                "data_directory": cfg.get("eval_data_path", ""),
            }
        )
        return env


    finetuner = CBVRLFinetuner(cfg)

    logger.info("Initializing MetaDrive ScenarioEnv for CBV RL fine-tuning...")
    eval_env = create_env()

    n_iterations = int(cfg.get("rl_iterations", 1))
    update_at_iteration_end = bool(cfg.get("rl_update_at_iteration_end", True))
    render_rollout = bool(cfg.get("rl_render_rollout", False))
    render_every_n_steps = max(1, int(cfg.get("rl_render_every_n_steps", 1)))
    render_film_size = tuple(cfg.get("rl_render_film_size", [1200, 800]))

    for iteration in range(n_iterations):
        logger.info("===== CBV RL Iteration %s / %s =====", iteration + 1, n_iterations)
        for episode_idx in range(int(cfg.get("eval_eps", 10))):
            try:
                eval_env.reset()
                scenario = deepcopy(eval_env.engine.data_manager.current_scenario)
                sdc_id = scenario["metadata"]["sdc_id"]
                if scenario["metadata"]["object_summary"][sdc_id]["moving_distance"] < 5:
                    print('Too short ,skip------------')
                    continue
                if render_rollout:
                    eval_env.head_renderer = HeadTopDownRenderer(eval_env)
                    eval_env.engine.head_renderer = eval_env.head_renderer
                eval_env.engine.agents["default_agent"].expert_traj = scenario["tracks"][sdc_id]["state"][
                    "position"]

                for step_idx in range(int(cfg.get("max_step", 100))):
                    _, _, terminated, truncated, info = eval_env.step([0, 0])
                    num_new = finetuner.collect_rollout_samples(info)

                    if render_rollout and step_idx % render_every_n_steps == 0:
                        eval_env.head_renderer.render(
                            screen_record=False,
                            show_plan_traj=True,
                            scaling=6,
                            film_size=render_film_size,
                            mode="topdown",
                            text={"step": step_idx, "id": scenario["id"]},
                        )

                    if num_new > 0:
                        logger.debug("Collected %s CBV rollout samples at step %s", num_new, step_idx)
                    if finetuner.buffer.should_update():
                        finetuner.run_finetuning_loop()
                    if terminated or truncated:
                        break
            except Exception as exc:
                logger.error("Error running scenario %s in iteration %s: %s", episode_idx, iteration, exc)
                continue

        if update_at_iteration_end and len(finetuner.buffer) > 0:
            finetuner.run_finetuning_loop()

    eval_env.close()
    logger.info("CBV RL finetuning finished with %s updates.", finetuner.num_updates)

if __name__ == "__main__":
    main()
