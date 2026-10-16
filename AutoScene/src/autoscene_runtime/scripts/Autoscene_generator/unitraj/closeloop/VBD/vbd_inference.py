import json
import math
import os
import time

import numpy as np
import torch

from unitraj.datasets.VBD_dataset.VBD_test_dataset import VBDTestDataset
from unitraj.models.vbd.sim_agent.sim_actor import VBDTest


class VBDInference:
    """AutoScene adapter for unguided VBD joint diffusion.

    Model/checkpoint contract:
      - 11 observed states = 10 history + current;
      - native model horizon = 80 future states at 0.1 s;
      - AutoScene export horizon = 50 future states (5.0 s) from the request time.

    VBD predicts all selected agents jointly. Compatible ego/CBV requests share
    a sample. Temporal reuse is opt-in; cached world-coordinate trajectories are
    advanced by simulation steps, never wall time or spatial nearest matching.
    """

    trajectory_reference = "center"
    trajectory_format = "xy_cos_sin_vx_vy"
    trajectory_model_name = "vbd"

    def __init__(self, cfg):
        self.cfg = cfg
        # ROS /algorithm overrides are process-local, not edits to checkpoint or
        # generator YAML. Standalone without this payload still uses cfg only.
        self._cadence_overrides = json.loads(os.environ.get("UNITRAJ_VBD_CADENCE_OVERRIDES", "{}"))
        if not isinstance(self._cadence_overrides, dict):
            raise ValueError("UNITRAJ_VBD_CADENCE_OVERRIDES must be a JSON object")
        self.model = None
        self.dataset = None
        self._joint_cache = None
        self._last_timing = None
        self._last_request_timestep = None
        self.inference_count = 0
        self.cache_hit_count = 0
        self.last_diagnostics = {}

        interval = self._positive_steps("vbd_inference_interval_steps", 1)
        self.inference_intervals = {
            role: self._positive_steps("vbd_%s_inference_interval_steps" % role, interval)
            for role in ("ego", "adv")
        }
        self._configured_intervals = dict(self.inference_intervals)
        self.adaptive_cadence = bool(self._get("vbd_adaptive_cadence", False))
        self._inference_cost_ema = None
        self.cadence_budget_exceeded = False

        self.agents_len = int(self._get("agents_len", 16))
        self.history_steps = int(self._get("vbd_history_steps", self._get("history_steps", 11)))
        self.model_future_steps = int(self._get("future_len", 80))
        self.output_future_steps = int(self._get("output_future_steps", 50))
        self.max_polylines = int(self._get("max_polylines", 256))
        self.num_points_polyline = int(self._get("num_points_polyline", 30))
        self.max_traffic_lights = int(self._get("max_traffic_lights", 16))
        self.map_radius_m = float(self._get("map_radius_m", 150.0))
        self.use_tqdm = bool(self._get("vbd_use_tqdm", False))
        self.inference_mode = str(self._get("inference_mode", "diffusion")).strip().lower()

        if self.inference_mode != "diffusion":
            raise ValueError(
                "AutoScene VBD integration currently supports unguided diffusion only; "
                f"got inference_mode={self.inference_mode}"
            )
        if self.history_steps != 11:
            raise ValueError(
                f"VBD history contract requires 11 states (10 history + current), got {self.history_steps}"
            )
        if self.model_future_steps != 80:
            raise ValueError(
                f"VBD checkpoint contract requires future_len=80, got {self.model_future_steps}"
            )
        if self.output_future_steps != 50:
            raise ValueError(
                f"AutoScene platform contract requires output_future_steps=50, got {self.output_future_steps}"
            )
        if self.output_future_steps > self.model_future_steps:
            raise ValueError("output_future_steps cannot exceed VBD future_len")
        if self.agents_len <= 0:
            raise ValueError("VBD agents_len must be > 0")

        self.generator_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../..")
        )
        self.ckpt_path = self._resolve_path(self._get("ckpt_path"))
        self.anchor_path = self._resolve_path(self._get("anchor_path"))
        self.device = self._resolve_device()

    def _get(self, key, default=None):
        if hasattr(self.cfg, "get"):
            return self.cfg.get(key, default)
        try:
            return self.cfg[key]
        except Exception:
            return default

    def _positive_steps(self, key, default):
        value = self._cadence_overrides.get(key, self._get(key, default))
        if value is None:
            value = default
        # Do not silently truncate a fractional interval or accept YAML booleans.
        try:
            parsed = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("%s must be a positive integer, got %r" % (key, value)) from exc
        if isinstance(value, bool) or str(value).strip() != str(parsed) or parsed < 1:
            raise ValueError("%s must be a positive integer, got %r" % (key, value))
        return parsed

    def reset_cache(self):
        """Invalidate predictions on reset/rebinding, including a same-ID replay.

        Counters are lifetime totals. Call this explicitly for in-place scene
        replacement; ordinary policy scene changes and time rewinds also reset.
        """
        self._joint_cache = None
        self._last_request_timestep = None
        self._last_timing = None
        self.last_diagnostics = {}

    def _adapt_cadence(self, sample_wall_sec):
        """Budget at most 35% of real-time playback for VBD, horizon permitting.

        This never changes model denoising steps or trajectory point times.
        Sampling remains synchronous; an over-budget GPU is reported, not hidden
        by extrapolating beyond the native prediction horizon.
        """
        if not self.adaptive_cadence:
            return
        cost = max(0.0, float(sample_wall_sec))
        self._inference_cost_ema = (cost if self._inference_cost_ema is None else
                                    0.75 * self._inference_cost_ema + 0.25 * cost)
        # Use current cost too, so a sudden slowdown is not hidden by smoothing.
        budget_cost = max(cost, self._inference_cost_ema)
        desired = int(math.ceil(budget_cost / (0.1 * 0.35 * 5))) * 5
        maximum = self.model_future_steps - self.output_future_steps  # 30 real steps
        for role, configured in self._configured_intervals.items():
            self.inference_intervals[role] = min(maximum, max(10, configured, desired))
        self.cadence_budget_exceeded = budget_cost > maximum * 0.1 * 0.35

    def _resolve_path(self, value):
        if value is None or str(value).strip() == "":
            return None
        path = os.path.expanduser(str(value).strip())
        if not os.path.isabs(path):
            path = os.path.join(self.generator_root, path)
        return os.path.abspath(path)

    def _resolve_device(self):
        configured = self._get("device", None)
        if configured is not None and str(configured).strip():
            device = str(configured).strip()
        else:
            devices = self._get("devices", [0])
            try:
                gpu_index = int(devices[0])
            except Exception as exc:
                raise ValueError(f"invalid VBD devices config: {devices!r}") from exc
            device = f"cuda:{gpu_index}"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"VBD is configured for {device}, but torch.cuda.is_available() is False"
            )
        return torch.device(device)

    def initialize_model(self):
        if self.model is not None:
            return
        if not self.ckpt_path or not os.path.isfile(self.ckpt_path):
            raise FileNotFoundError(f"VBD checkpoint not found: {self.ckpt_path}")
        if not self.anchor_path or not os.path.isfile(self.anchor_path):
            raise FileNotFoundError(f"VBD anchor file not found: {self.anchor_path}")

        self.dataset = VBDTestDataset(
            data_dir=None,
            anchor_path=self.anchor_path,
            max_object=self.agents_len,
            max_polylines=self.max_polylines,
            history_length=self.history_steps,
            future_steps=self.model_future_steps,
            num_points_polyline=self.num_points_polyline,
            max_traffic_lights=self.max_traffic_lights,
            map_radius_m=self.map_radius_m,
        )

        # Newer PyTorch defaults torch.load(weights_only=True). Lightning checkpoints
        # need the full checkpoint object. Scope the compatibility shim to this load
        # only; never leave a process-global monkey patch installed.
        import lightning.pytorch.core.saving as saving
        original_torch_load = saving.torch.load

        def _torch_load_full_checkpoint(*args, **kwargs):
            kwargs["weights_only"] = False
            return original_torch_load(*args, **kwargs)

        saving.torch.load = _torch_load_full_checkpoint
        try:
            self.model = VBDTest.load_from_checkpoint(
                self.ckpt_path,
                cfg=self.cfg,
                map_location=self.device,
            )
        finally:
            saving.torch.load = original_torch_load

        self.model.reset_agent_length(self.agents_len)
        self.model.to(self.device)
        self.model.eval()
        # Explicitly keep this integration unguided.
        if hasattr(self.model, "reward_func"):
            self.model.reward_func = None

        model_future_len = int(getattr(self.model, "_future_len", self.model_future_steps))
        if model_future_len != self.model_future_steps:
            raise RuntimeError(
                f"loaded VBD checkpoint future_len={model_future_len} != config={self.model_future_steps}"
            )
        model_agent_len = int(getattr(self.model, "_agents_len", self.agents_len))
        if model_agent_len != self.agents_len:
            raise RuntimeError(
                f"loaded VBD checkpoint agents_len={model_agent_len} != config={self.agents_len}"
            )

    @staticmethod
    def _scenario_id(current_state):
        metadata = current_state.get("metadata", {}) if isinstance(current_state, dict) else {}
        return str(current_state.get("id", metadata.get("scenario_id", metadata.get("id", "unknown"))))

    @staticmethod
    def _dedupe_ids(ids):
        out = []
        seen = set()
        for agent_id in ids or []:
            sid = str(agent_id)
            if sid not in seen:
                out.append(sid)
                seen.add(sid)
        return out

    @staticmethod
    def _native_to_common_center(pred_native):
        arr = np.asarray(pred_native, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[-1] < 5:
            raise RuntimeError(
                f"VBD denoised_trajs must have shape [agent,time,>=5], got {arr.shape}"
            )
        if not np.isfinite(arr[..., :5]).all():
            raise RuntimeError("VBD denoised_trajs contains non-finite x/y/yaw/vx/vy")
        yaw = arr[..., 2]
        return np.stack(
            [
                arr[..., 0],
                arr[..., 1],
                np.cos(yaw),
                np.sin(yaw),
                arr[..., 3],
                arr[..., 4],
            ],
            axis=-1,
        ).astype(np.float32, copy=False)

    def _motion_diverged(self, cache, state, timestep):
        """A time-valid prediction may still be invalid after a vehicle reacts.

        Compare actual CENTER pose/velocity with the sampled joint world. Missing
        geometry is allowed for legacy fixtures, but real online tracks provide it.
        """
        age = int(timestep) - cache["sample_timestep"]
        if age <= 0 or age > cache["trajectory"].shape[1]:
            return False
        tracks = state.get("tracks", {})
        max_position = float(self._get("vbd_position_error_m", 1.0))
        max_speed = float(self._get("vbd_velocity_error_mps", 2.0))
        max_heading = float(self._get("vbd_heading_error_rad", .45))
        owned = getattr(self, "interaction_controlled_ids", set(cache["key"][2]))
        anchors = []
        for sid in owned:
            position = tracks.get(sid, {}).get("state", {}).get("position")
            if position is not None and timestep < len(position):
                anchors.append(np.asarray(position[timestep])[:2])
        for sid, slot in cache["token_to_slot"].items():
            values = tracks.get(sid, {}).get("state", {})
            position = values.get("position")
            valid = values.get("valid")
            if position is None or timestep >= len(position):
                continue
            if valid is not None and not bool(np.asarray(valid[timestep]).item()):
                continue
            actual = np.asarray(position[timestep])[:2]
            predicted = cache["trajectory"][slot, age - 1]
            # Far-away actors whose sampled trajectories are NOT executed must
            # not force diffusion every tick. Observe all owned actors plus the
            # neighborhood around either their actual or predicted position.
            if sid not in owned and anchors and all(min(np.linalg.norm(actual-p), np.linalg.norm(predicted[:2]-p)) > 25. for p in anchors):
                continue
            if not np.isfinite(actual).all() or np.linalg.norm(actual - predicted[:2]) > max_position:
                self.last_motion_divergence = {"track":sid,"kind":"position","error":float(np.linalg.norm(actual-predicted[:2]))}
                return True
            velocity, heading = values.get("velocity"), values.get("heading")
            if velocity is not None and timestep < len(velocity):
                if np.linalg.norm(np.asarray(velocity[timestep])[:2] - predicted[4:6]) > max_speed:
                    self.last_motion_divergence = {"track":sid,"kind":"velocity"}
                    return True
            if heading is not None and timestep < len(heading):
                delta = (float(np.asarray(heading[timestep]).item()) - np.arctan2(predicted[3], predicted[2]) + np.pi) % (2*np.pi) - np.pi
                if abs(delta) > max_heading:
                    self.last_motion_divergence = {"track":sid,"kind":"heading","error":float(abs(delta))}
                    return True
        return False

    def _prepare_joint_prediction(self, current_state, timestep, required_agent_ids=None, role="ego"):
        if self.model is None or self.dataset is None:
            raise RuntimeError("VBDInference.initialize_model() must be called before inference")
        if not isinstance(current_state, dict):
            raise TypeError("VBD current_state must be a ScenarioNet dict")

        timestep = int(timestep)
        metadata = current_state.get("metadata", {})
        sdc_id = str(metadata.get("sdc_id", ""))
        if not sdc_id:
            raise KeyError("ScenarioNet metadata.sdc_id is required for VBD")

        required = self._dedupe_ids(required_agent_ids)
        if sdc_id not in required:
            required.insert(0, sdc_id)
        t0 = time.perf_counter()
        cache_key = (self._scenario_id(current_state), sdc_id, tuple(sorted(required)))
        cache = self._joint_cache
        interval = self.inference_intervals[role]
        reason = "empty_cache"
        if cache is not None:
            age = timestep - cache["sample_timestep"]
            # Initial CBV selection happens after the ego prediction. If it only
            # adds already sampled agents at this exact tick, retain the joint
            # world and lock the expanded required set for subsequent ticks.
            same_tick_expansion = (
                age == 0
                and set(cache["key"][2]).issubset(cache_key[2])
                and set(cache_key[2]).issubset(cache["token_to_slot"])
            )
            if self._last_request_timestep is not None and timestep < self._last_request_timestep:
                reason = "time_rewind"
            elif (cache["key"][:2] != cache_key[:2]
                  or cache["scene_metadata"] is not metadata
                  or cache["scene_map"] is not current_state.get("map_features")):
                # update_state shallow-copies the scene/tracks each planning tick,
                # but retains metadata/map identity. Keep references, not just ids.
                reason = "scene_changed"
            elif ((cache["key"][2] != cache_key[2] and not same_tick_expansion)
                  or cache["track_ids"] != tuple(sorted(str(x) for x in current_state.get("tracks", {})))):
                reason = "candidates_changed"
            elif not self._required_agents_valid(current_state, timestep, required):
                reason = "required_agent_invalid"
            elif age < 0 or age + self.output_future_steps > cache["trajectory"].shape[1]:
                reason = "horizon_exhausted"
            elif self._motion_diverged(cache, current_state, timestep):
                reason = "motion_diverged"
            elif age >= interval:
                reason = "interval_due"
            else:
                cache["key"] = cache_key
                self._last_request_timestep = timestep
                self.cache_hit_count += 1
                return self._export_window(cache, timestep, role, True, "cache_hit", t0)

        # Fail closed: a failed refresh must not leave an older usable prediction.
        self._joint_cache = None
        self._last_timing = None
        self._last_request_timestep = timestep
        self.last_diagnostics = {}
        sample = self.dataset.process_scenario(
            current_state,
            current_index=timestep,
            use_log=True,
            required_agent_ids=required,
        )
        tokens = [str(token) for token in sample.get("agent_tokens", [])]
        if len(tokens) != len(set(tokens)):
            raise RuntimeError(f"VBD agent token mapping contains duplicates: {tokens}")
        missing = [sid for sid in required if sid not in tokens]
        if missing:
            raise RuntimeError(
                f"VBD joint sample omitted required agents {missing}; selected tokens={tokens}"
            )
        if sample.get("history_end_index") != timestep:
            raise RuntimeError("VBD history_end_index does not match requested timestep")
        if sample.get("history_start_index") != timestep - self.history_steps + 1:
            raise RuntimeError("VBD history window is not exactly 10 history + current")

        batch = self.dataset.__collate_fn__([sample])
        t1 = time.perf_counter()
        with torch.no_grad():
            pred = self.model.sample_denoiser(
                batch,
                use_tqdm=self.use_tqdm,
            )
        if not isinstance(pred, dict) or "denoised_trajs" not in pred:
            raise RuntimeError("VBD sample_denoiser did not return denoised_trajs")
        native = pred["denoised_trajs"].detach().cpu().numpy()
        if native.ndim != 4 or native.shape[0] != 1:
            raise RuntimeError(f"VBD denoised_trajs expected [1,A,T,C], got {native.shape}")
        native = native[0]
        if native.shape[0] < len(tokens):
            raise RuntimeError(
                f"VBD output agents={native.shape[0]} < selected tokens={len(tokens)}"
            )
        if native.shape[1] < self.model_future_steps:
            raise RuntimeError(
                f"VBD output horizon={native.shape[1]} < checkpoint horizon={self.model_future_steps}"
            )
        # Preserve the entire native horizon. Row k predicts sample_timestep+k+1.
        # Cropping to 50 here would make any later 50-point export invalid.
        native = native[: len(tokens), : self.model_future_steps, :5]
        common = self._native_to_common_center(native)
        t2 = time.perf_counter()

        token_to_slot = {token: idx for idx, token in enumerate(tokens)}
        self._last_timing = {
            "preprocess_sec": float(t1 - t0),
            "inference_sec": float(t2 - t1),
            "total_sec": float(t2 - t0),
        }
        self._adapt_cadence(t2 - t0)
        self._joint_cache = {
            "key": cache_key,
            "sample_timestep": timestep,
            "scene_metadata": metadata,
            "scene_map": current_state.get("map_features"),
            "track_ids": tuple(sorted(str(x) for x in current_state.get("tracks", {}))),
            "tokens": tokens,
            "token_to_slot": token_to_slot,
            "trajectory": common,
            "timing": dict(self._last_timing),
            "history_start_index": int(sample["history_start_index"]),
            "history_end_index": int(sample["history_end_index"]),
        }
        self.inference_count += 1
        return self._export_window(self._joint_cache, timestep, role, False, reason, t0)

    @staticmethod
    def _required_agents_valid(current_state, timestep, required):
        tracks = current_state.get("tracks", {})
        for agent_id in required:
            if agent_id not in tracks:
                return False
            valid = tracks[agent_id].get("state", {}).get("valid")
            if valid is not None:
                valid = np.asarray(valid).reshape(-1)
                if timestep < 0 or timestep >= len(valid) or not bool(valid[timestep]):
                    return False
        return True

    def _export_window(self, cache, timestep, role, reused, reason, started):
        age = timestep - cache["sample_timestep"]
        remaining = cache["trajectory"].shape[1] - age
        if age < 0 or remaining < self.output_future_steps:
            raise RuntimeError("VBD cache has no complete future export window")
        timing = dict(cache["timing"])
        if reused:
            timing = {"preprocess_sec": 0.0, "inference_sec": 0.0,
                      "total_sec": float(time.perf_counter() - started)}
        diagnostics = {
            "role": role,
            "cache_hit": bool(reused),
            "refresh_reason": reason,
            "inference_interval_steps": self.inference_intervals[role],
            "sample_timestep": cache["sample_timestep"],
            "request_timestep": timestep,
            "cache_age_steps": age,
            "remaining_future_steps": remaining,
            "valid_until_index": cache["sample_timestep"] + cache["trajectory"].shape[1],
            "export_start_index": timestep + 1,
            "export_end_index": timestep + self.output_future_steps,
            "inference_count": self.inference_count,
            "cache_hit_count": self.cache_hit_count,
            "adaptive_cadence": self.adaptive_cadence,
            "cadence_budget_exceeded": self.cadence_budget_exceeded,
        }
        self.last_diagnostics = dict(diagnostics, timing=dict(timing))
        self._last_timing = timing
        window = dict(cache)
        window["trajectory"] = cache["trajectory"][:, age:age + self.output_future_steps]
        window["sample_timing"] = dict(cache["timing"])
        window["timing"] = timing
        window["cadence"] = diagnostics
        return window

    def _result_for_agent(self, cache, primary_agent_id):
        primary_agent_id = str(primary_agent_id)
        token_to_slot = cache["token_to_slot"]
        if primary_agent_id not in token_to_slot:
            raise RuntimeError(
                f"VBD primary agent {primary_agent_id} not in joint sample {cache['tokens']}"
            )
        primary_slot = token_to_slot[primary_agent_id]
        primary = cache["trajectory"][primary_slot]

        background_tokens = [token for token in cache["tokens"] if token != primary_agent_id]
        if background_tokens:
            background = np.stack(
                [cache["trajectory"][token_to_slot[token]] for token in background_tokens],
                axis=0,
            )
        else:
            background = np.zeros(
                (0, self.output_future_steps, 6),
                dtype=np.float32,
            )

        # Consumers may mutate their control arrays; never expose cache storage.
        pred_traj = primary[None, ...].copy()
        info = {
            "model_name": "VBD",
            "trajectory_reference": self.trajectory_reference,
            "trajectory_format": self.trajectory_format,
            "controlled_agent_id": primary_agent_id,
            # Existing AutoScene CBV selector accepts the convention
            # [primary token] + background tokens while prediction excludes primary.
            "agent_tokens": [primary_agent_id] + background_tokens,
            "prediction_agent_tokens": background_tokens,
            "prediction": background,
            "joint_agent_tokens": list(cache["tokens"]),
            "joint_prediction": cache["trajectory"].copy(),
            "selected_future_traj": primary.copy(),
            "score_selection": "max",
            "history_start_index": cache["history_start_index"],
            "history_end_index": cache["history_end_index"],
            "model_future_steps": self.model_future_steps,
            "export_future_steps": self.output_future_steps,
            "timing": dict(cache["timing"]),
            "sample_timing": dict(cache["sample_timing"]),
            "cadence": dict(cache["cadence"]),
        }
        return pred_traj, None, None, info

    def run_inference(
        self,
        current_state,
        timestep,
        controlled_agent_id=None,
        score_selection="max",
        adversarial_target_agent_ids=None,
        required_agent_ids=None,
    ):
        score_selection = str(score_selection or "max").strip().lower()
        if score_selection not in ("max", "unguided"):
            raise ValueError(
                f"VBD AutoScene integration is unguided only; score_selection={score_selection} is unsupported"
            )
        if adversarial_target_agent_ids:
            raise ValueError("VBD adversarial_target_agent_ids is unsupported in unguided integration")

        metadata = current_state.get("metadata", {})
        primary = str(controlled_agent_id) if controlled_agent_id is not None else str(metadata.get("sdc_id", ""))
        if not primary:
            raise KeyError("VBD cannot resolve primary agent id")
        required = self._dedupe_ids(required_agent_ids)
        if primary not in required:
            required.append(primary)
        role = "ego" if primary == str(metadata.get("sdc_id", "")) else "adv"
        cache = self._prepare_joint_prediction(current_state, timestep, required, role=role)
        return self._result_for_agent(cache, primary)

    def run_inference_batch(
        self,
        current_state,
        timestep,
        controlled_agent_ids,
        score_selections=None,
        adversarial_target_agent_ids_by_id=None,
        **_unused,
    ):
        controlled_agent_ids = self._dedupe_ids(controlled_agent_ids)
        if not controlled_agent_ids:
            return {}
        score_selections = score_selections or {}
        for agent_id in controlled_agent_ids:
            mode = str(score_selections.get(agent_id, "max")).strip().lower()
            if mode not in ("max", "unguided"):
                raise ValueError(
                    f"VBD controlled agent {agent_id} requests unsupported planner mode={mode}"
                )
        adversarial_target_agent_ids_by_id = adversarial_target_agent_ids_by_id or {}
        if any(adversarial_target_agent_ids_by_id.get(agent_id) for agent_id in controlled_agent_ids):
            raise ValueError("VBD adversarial guidance is not enabled")

        cache = self._prepare_joint_prediction(
            current_state,
            timestep,
            required_agent_ids=controlled_agent_ids,
            role="adv",
        )
        return {
            agent_id: self._result_for_agent(cache, agent_id)
            for agent_id in controlled_agent_ids
        }
