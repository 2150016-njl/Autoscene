import inspect

import os
import numpy as np
import yaml
from metadrive.policy.env_input_policy import EnvInputPolicy
from metadrive.policy.base_policy import BasePolicy
from metadrive.policy.replay_policy import ReplayTrafficParticipantPolicy
from metadrive.component.vehicle.PID_controller import PIDController
from metadrive.scenario.parse_object_state import parse_object_state
from omegaconf import OmegaConf
from unitraj.closeloop.VBD.vbd_inference import VBDInference
from unitraj.closeloop.pluto.pluto_inference import PlutoInference
from unitraj.utils.utils import set_seed

# --- 控制器参数 ---
LATERAL_PID_GAINS = (0.85, 0.05, 0.02)  # Kp, Ki, Kd
LONGITUDINAL_PID_GAINS = (50, 0, 0)  # Kp, Ki, Kd
FIXED_DT = 0.1  # (s) 仿真步长
MATCH_ROLLBACK_DIST_M = 2.5
MATCH_ROLLBACK_CROSS_TRACK_M = 1.5
FORWARD_DOT_MIN = 0.5


def _vehicle_speed_mps(vehicle):
    """Return actual planar world speed in m/s from the velocity vector.

    Do not use ``vehicle.speed`` for control/history here: the velocity vector is
    the same world-frame quantity exported to AutoScene and avoids any simulator
    scalar-speed unit/definition ambiguity.
    """
    try:
        velocity = np.asarray(getattr(vehicle, "velocity"), dtype=np.float64).reshape(-1)
        if len(velocity) >= 2 and np.isfinite(velocity[:2]).all():
            return float(np.linalg.norm(velocity[:2]))
    except Exception:
        pass
    fallback = float(getattr(vehicle, "speed", 0.0))
    return fallback if np.isfinite(fallback) else 0.0


class DynamicReplayTrafficParticipantPolicy(ReplayTrafficParticipantPolicy):
    """A replay-like policy that reads the provided track *dynamically* every step.

    MetaDrive's default `ReplayTrafficParticipantPolicy` caches `traj_info` at init time.
    When we want to overwrite a CBV's future states with a newly predicted trajectory,
    we need a policy that always reads from the (mutated) track.

    This class still subclasses `ReplayTrafficParticipantPolicy` so ScenarioTrafficManager
    will keep calling it in `after_step()` (it checks `isinstance(policy, ReplayTrafficParticipantPolicy)`).
    """

    def __init__(self, control_object, track, random_seed=None):
        # NOTE: intentionally NOT calling ReplayTrafficParticipantPolicy.__init__
        # to avoid caching traj_info.
        BasePolicy.__init__(self, control_object=control_object, random_seed=random_seed)
        self.start_index = 0
        self._velocity_local_frame = False
        self.track = track
        self.physics_tracking_active = False
        self.execution_max_speed_mps = None
        self.execution_speed_kp = 1.5
        self.execution_overspeed_margin_mps = 0.05

    def configure_execution_limits(
            self,
            max_speed_mps=None,
            speed_kp=1.5,
            overspeed_margin_mps=0.05,
    ):
        """Configure the physics-execution envelope for simulated CBVs.

        Planner feasibility is necessary but not sufficient: Bullet dynamics can
        deviate from the discrete plan.  This layer prevents a simulated CBV from
        continuing to accelerate beyond the same speed ceiling used by the planner.
        It is never installed on ExternalFeedbackMirrorPolicy hardware objects.
        """
        if max_speed_mps is None:
            self.execution_max_speed_mps = None
        else:
            value = float(max_speed_mps)
            self.execution_max_speed_mps = value if np.isfinite(value) and value > 0.0 else None
        self.execution_speed_kp = max(float(speed_kp), 0.0)
        self.execution_overspeed_margin_mps = max(float(overspeed_margin_mps), 0.0)

    def activate_tracking(self):
        # Warmup replay must remain exact. Switch only once a real planner takes
        # execution ownership; an external feedback mirror never enters here.
        self.physics_tracking_active = True
        self.control_object.set_static(False)

    def compute_action(self, index=None):
        index = max(int(self.episode_step if index is None else index), 0)
        obj = self.control_object
        # Collision is an evaluation/physics event, not a permanent control override.
        # MetaDrive contact dynamics may change pose/velocity at impact, but after the
        # impact this policy must keep following the latest Pluto/Hybrid plan so the
        # executed post-collision state can flow back into the next planning history.
        state = parse_object_state(self.track, index)
        if state is None or not state.get("valid", False):
            return [0.0, -1.0]
        speed = _vehicle_speed_mps(obj)
        # Respect planned zero velocity: never impose a minimum cruising speed.
        target_speed_raw = float(np.linalg.norm(np.asarray(state["velocity"], dtype=np.float64)[:2]))
        target_speed = target_speed_raw
        max_speed = self.execution_max_speed_mps
        if max_speed is not None:
            target_speed = min(target_speed, max_speed)
        positions = np.asarray(self.track["state"]["position"])
        target_index = min(index + max(2, int(np.clip(speed * .6, 2., 5.) / max(target_speed * .1, .2))), len(positions)-1)
        target = positions[target_index, :2]
        delta = target - np.asarray(obj.position)[:2]
        heading_error = (float(np.arctan2(delta[1], delta[0])) - float(obj.heading_theta) + np.pi) % (2*np.pi) - np.pi
        # Pure pursuit expressed in normalized steering (BaseVehicle.before_step).
        wheelbase = float(getattr(obj, "LENGTH", 4.5)) * .6
        steering = np.arctan2(2 * wheelbase * np.sin(heading_error), max(np.linalg.norm(delta), 1.))
        steering /= np.deg2rad(float(getattr(obj, "max_steering", 40.)))
        throttle = np.clip(self.execution_speed_kp * (target_speed - speed), -1., 1.)
        overspeed_brake = bool(
            max_speed is not None
            and speed > max_speed + self.execution_overspeed_margin_mps
        )
        if overspeed_brake:
            # The actual plant has already escaped the planner envelope.  Apply a
            # bounded physical brake until it re-enters; never keep accelerating.
            throttle = -1.0
        elif max_speed is not None and speed >= max_speed and throttle > 0.0:
            throttle = 0.0
        if target_speed < .05:
            throttle = -1.0 if speed > .05 else 0.0
        steering_cmd = float(np.clip(steering, -.8, .8))
        throttle_cmd = float(np.clip(throttle, -1.0, 1.0))

        # Execution audit only; it does not change control.  This proves whether
        # a planned turn reaches MetaDrive's traffic vehicle controller.
        if os.environ.get("AUTOSCENE_CBV_EXEC_DEBUG", "0") in ("1", "true", "True"):
            if int(self.episode_step) % 5 == 0:
                scenario_id = None
                try:
                    scenario_id = self.engine.traffic_manager.obj_id_to_scenario_id.get(obj.id)
                except Exception:
                    scenario_id = None
                heading_values = np.asarray(self.track["state"].get("heading", []), dtype=np.float64).reshape(-1)
                planned_heading = (
                    float(heading_values[min(index, len(heading_values) - 1)])
                    if len(heading_values) > 0 else None
                )
                print(
                    "[CBVExecution]",
                    {
                        "step": int(self.episode_step),
                        "scenario_id": str(scenario_id) if scenario_id is not None else str(getattr(obj, "id", "?")),
                        "actual_speed_mps": round(float(speed), 4),
                        "target_speed_raw_mps": round(float(target_speed_raw), 4),
                        "target_speed_cmd_mps": round(float(target_speed), 4),
                        "execution_max_speed_mps": None if max_speed is None else round(float(max_speed), 4),
                        "overspeed_brake": bool(overspeed_brake),
                        "actual_heading": round(float(obj.heading_theta), 6),
                        "planned_heading": None if planned_heading is None else round(planned_heading, 6),
                        "heading_error_to_lookahead": round(float(heading_error), 6),
                        "lookahead_index": int(target_index),
                        "steering": round(steering_cmd, 6),
                        "throttle": round(throttle_cmd, 6),
                    },
                )
        return [steering_cmd, throttle_cmd]

    @property
    def is_current_step_valid(self):
        # Once Pluto/Hybrid owns execution, dataset validity must no longer delete
        # the vehicle when the recorded track ends.  The planner/physics object is
        # now the lifecycle owner.
        if self.physics_tracking_active:
            return self.control_object is not None
        index = max(int(self.episode_step), 0)
        state = parse_object_state(self.track, index)
        return state is not None and bool(state.get("valid", False))

    def act(self, *args, **kwargs):
        if self.physics_tracking_active:
            # ScenarioTrafficManager applies compute_action BEFORE physics.
            # Do not teleport the vehicle back onto the plan after physics.
            return None
        index = max(int(self.episode_step), 0)
        state = parse_object_state(self.track, index)
        if state is None or not bool(state.get("valid", False)):
            return None

        if "throttle_brake" in state and hasattr(self.control_object, "set_throttle_brake"):
            self.control_object.set_throttle_brake(float(np.asarray(state["throttle_brake"]).item()))
        if "steering" in state and hasattr(self.control_object, "set_steering"):
            self.control_object.set_steering(float(np.asarray(state["steering"]).item()))

        self.control_object.set_position(state["position"])
        self.control_object.set_velocity(state["velocity"], in_local_frame=self._velocity_local_frame)
        self.control_object.set_heading_theta(state["heading"])
        if "angular_velocity" in state:
            self.control_object.set_angular_velocity(state["angular_velocity"])

        if self.engine.global_config.get("set_static", False):
            self.control_object.set_static(True)

        return None


class ExternalFeedbackMirrorPolicy(BasePolicy):
    """Mirror an actual platform object; runtime feedback is its sole pose owner."""

    def act(self, *args, **kwargs):
        return None


class InferencePolicy(EnvInputPolicy):
    """
    一个专业的混合策略，通过继承 ReplayEgoCarPolicy 来实现行为切换
    - 前 WARMUP_STEPS 步: 自动使用父类 ReplayEgoCarPolicy 的 act 方法
    - WARMUP_STEPS 步后: 使用横向和纵向PID控制器，精准跟随模型生成的轨迹。
    """

    def __init__(self, obj, seed):
        super(InferencePolicy, self).__init__(obj, seed)

        self.Sim = None
        self.base_config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "config.yaml")

        # ROS/UI create_env() binds the already-resolved effective UniTraj config
        # to the concrete policy subclass.  Consume that exact snapshot here so
        # ego_method/adv_method cannot silently fall back to the on-disk
        # configs/config.yaml.  Standalone policy construction keeps the original
        # disk-config behavior when no snapshot was supplied.
        policy_type = type(self)
        effective_base_raw = getattr(
            policy_type,
            "_autoscene_effective_base_cfg",
            None,
        )
        effective_required = bool(
            getattr(policy_type, "_autoscene_require_effective_base_cfg", False)
        )
        if effective_base_raw is None:
            if effective_required:
                raise RuntimeError(
                    "AutoScene configured policy is missing its effective base config"
                )
            with open(self.base_config_path, "r", encoding="utf-8") as f:
                effective_base_raw = yaml.safe_load(f) or {}
        elif not isinstance(effective_base_raw, dict):
            raise TypeError(
                "_autoscene_effective_base_cfg must be a dict, got %s"
                % type(effective_base_raw).__name__
            )
        self.base_cfg = OmegaConf.create(effective_base_raw)
        self._apply_runtime_overrides(self.base_cfg)

        self.WARMUP_STEPS = 20
        self.replan_frequency = 5  # 中频重规划 (比如 5 帧/0.5s 重新推理一次主周车意图)
        self.current_cbv_ids = []  # 单场景锁定的多辆关键周车
        self.max_cbv_num = self.base_cfg.get('adv_num', 2)
        self.cbv_selection_locked = False
        self.allow_pedestrian_cbv = self.base_cfg.get('allow_pedestrian_cbv', False)
        self.allow_cyclist_cbv = self.base_cfg.get('allow_cyclist_cbv', False)

        self.ego_method = self.base_cfg.get('ego_method', self.base_cfg.defaults[0]['method'])
        self.ego_cfg = self.get_final_config(self.ego_method)
        print(f"[Policy] Ego Planner: {self.ego_method}")

        self.use_adv = self.base_cfg.adv_control
        if self.use_adv:
            self.adv_method = self.base_cfg.get('adv_method', self.base_cfg.defaults[0]['method'])
            self.adv_cfg = self.get_final_config(self.adv_method)
            self.AdvSim = self._init_inference_engine(self.adv_cfg)
        else:
            self.AdvSim = None
        print(f"[Policy] Adv Planner: {self.adv_method if self.use_adv else 'None'}")

        # 初始化横向和纵向PID控制器。MetaDrive policy action is normalized
        # to [-1, 1], so use an m/s-scale longitudinal gain and clamp the final
        # command before it reaches Bullet.
        self.lateral_pid = PIDController(*LATERAL_PID_GAINS)
        ego_longitudinal_kp = float(self.base_cfg.get("ego_longitudinal_speed_kp", 1.5))
        self.longitudinal_pid = PIDController(ego_longitudinal_kp, 0.0, 0.0)
        self.current_state = self.engine.data_manager.current_scenario

        # MetaDrive horizon can exceed the recorded dataset length.  Reserve enough
        # array capacity for the full runtime plus one planner horizon, but keep the
        # added tail invalid until it is actually produced by closed-loop execution.
        runtime_horizon = int(self.engine.global_config.get(
            "horizon", self.base_cfg.get("horizon", 500)
        ))
        future_margin = int(max(
            50,
            self.ego_cfg.get("future_steps", 80),
            self.adv_cfg.get("future_steps", 80) if self.use_adv else 0,
        ))
        target_track_length = runtime_horizon + future_margin + 1
        extended_track_count = self._extend_scenario_track_capacity(
            self.current_state, target_track_length
        )
        if extended_track_count:
            print(
                "[Policy] Extended scenario track capacity to %d frames (%d tracks)"
                % (target_track_length, extended_track_count)
            )

        # 深拷贝保存最初的、纯净的（不受污染的）回放场景数据作为基准线底座
        import copy
        self._base_scenario_cache = copy.deepcopy(self.current_state)

        self.sdc_id = str(self.current_state["metadata"]["sdc_id"])

        self.last_match_idx = 0

        self._pred_traj = None
        # buffer 需要能记下自接管以来的所有历史，而不仅是最近21帧
        # 为了保险起见，设置足够大的容量，以储存整局游戏的轨迹
        self.hist_buffer = {}
        self.agent_hist_buffer = {}

        # 存储周车推理结果
        self.current_adv_trajs = None
        self.current_aligned_adv_trajs = {}
        self.current_adv_patch_info = {}
        self.adv_agent_ids = []
        self.current_adv_ref_lines = {}
        self.current_adv_candidates = {}
        self.platform_output_point_num = 50
        # Keep the previous full CBV plan with its absolute start step.  At the
        # next 0.5 s replan it can be time-shifted and supplied to the batch
        # collision checker.  This is interaction context only; it is never
        # replayed as actual history.
        self._previous_adv_plan_by_id = {}
        self.info = None

        # Pending rollout payloads to be exposed via MetaDrive env.step() info.
        # We must NOT keep them in action_info across frames, otherwise RL collector
        # will see duplicates.
        self._pending_cbv_rollout_samples = []

        # For keeping inference-state consistent with the actually controlled CBV
        self._controlled_cbv_scenario_ids = set()
        self.external_history_owned_ids = set()
        self.external_execution_owned_ids = set()
        self.external_ego_control = False
        # Runtime fills this mapping with the configured mode for every locked CBV.
        # Keep it explicit here so inference never falls back to batch-order defaults.
        self.planner_mode_by_scenario_id = {}

        self.control_debug = []

    @staticmethod
    def _extend_scenario_track_capacity(scenario, target_length):
        """Extend track arrays without inventing valid future ground truth.

        New slots are capacity only: ``valid=False``.  Shape/static attributes are
        repeated, dynamic motion fields are zero-filled, and actual closed-loop
        frames become valid later through update_state()/planner track patching.
        """
        if not isinstance(scenario, dict):
            return 0
        target_length = max(int(target_length), 1)
        tracks = scenario.get("tracks", {}) or {}
        extended = 0
        dynamic_zero_keys = {
            "velocity", "acceleration", "angular_velocity",
            "steering", "throttle_brake",
        }
        for track in tracks.values():
            if not isinstance(track, dict):
                continue
            state = track.get("state")
            if not isinstance(state, dict):
                continue
            position = np.asarray(state.get("position", []))
            if position.ndim < 1 or len(position) <= 0:
                continue
            old_len = int(len(position))
            if old_len >= target_length:
                continue
            pad_n = target_length - old_len
            for key, value in list(state.items()):
                if isinstance(value, np.ndarray):
                    arr = value
                else:
                    try:
                        arr = np.asarray(value)
                    except Exception:
                        continue
                if arr.ndim < 1 or len(arr) != old_len:
                    continue
                if key == "valid":
                    fill = np.zeros((pad_n,) + arr.shape[1:], dtype=bool)
                    base = arr.astype(bool, copy=False)
                elif key in dynamic_zero_keys:
                    fill = np.zeros((pad_n,) + arr.shape[1:], dtype=arr.dtype)
                    base = arr
                elif old_len > 0:
                    fill = np.repeat(arr[-1:], pad_n, axis=0)
                    base = arr
                else:
                    fill = np.zeros((pad_n,) + arr.shape[1:], dtype=arr.dtype)
                    base = arr
                state[key] = np.concatenate([base, fill], axis=0)
            # valid must exist and must be false in the capacity-only tail.
            valid = np.asarray(state.get("valid", np.zeros((target_length,), dtype=bool)), dtype=bool).reshape(-1)
            if len(valid) < target_length:
                valid = np.pad(valid, (0, target_length - len(valid)), constant_values=False)
            valid[old_len:target_length] = False
            state["valid"] = valid
            extended += 1
        scenario["length"] = max(int(scenario.get("length", 0) or 0), target_length)
        return extended

    def _planner_mode_for_scenario_id(self, scenario_id):
        """Return explicit configured selection; missing/invalid mapping fails safe to min."""
        sid = str(scenario_id)
        raw = self.planner_mode_by_scenario_id.get(sid)
        if raw is None:
            suffix = sid[-5:]
            matches = [
                value for key, value in self.planner_mode_by_scenario_id.items()
                if str(key).endswith(suffix)
            ]
            if len(matches) == 1:
                raw = matches[0]
        mode = str(raw or "").strip().lower()
        if mode not in {"max", "min", "adversarial"}:
            warned = getattr(self, "_missing_planner_mode_warned", set())
            if sid not in warned:
                print(
                    "[Policy] Warning: planner selection missing/invalid for %s; "
                    "falling back to min/safe-brake" % sid
                )
                warned = set(warned)
                warned.add(sid)
                self._missing_planner_mode_warned = warned
            return "min"
        return mode

    def get_final_config(self, method_name):
        """Merge base config with the selected planner method config.

        pluto_hybrid deliberately reuses the existing Pluto method/checkpoint
        config and overlays only hybrid-candidate settings. pluto_hybrid_blend
        adds one further scoring-only overlay while preserving the original Pluto
        and pluto_hybrid branches. Existing Pluto/VBD loading remains unchanged.
        """
        # Merge method-specific settings on top of the same effective base config
        # used by this policy instance.  Reopening config.yaml here would discard
        # UI/runtime ego_method/adv_method overrides and recreate split-brain config.
        raw = OmegaConf.to_container(
            self.base_cfg,
            resolve=False,
        )
        if not isinstance(raw, dict):
            raise TypeError("effective base config must resolve to a mapping")

        method_name_str = str(method_name).strip()
        method_dir = os.path.join(os.path.dirname(self.base_config_path), "method")
        if method_name_str.lower() == "pluto_hybrid":
            method_files = [
                os.path.join(method_dir, "Pluto.yaml"),
                os.path.join(method_dir, "pluto_hybrid.yaml"),
            ]
        elif method_name_str.lower() == "pluto_hybrid_blend":
            method_files = [
                os.path.join(method_dir, "Pluto.yaml"),
                os.path.join(method_dir, "pluto_hybrid.yaml"),
                os.path.join(method_dir, "pluto_hybrid_blend.yaml"),
            ]
        else:
            method_files = [os.path.join(method_dir, f"{method_name_str}.yaml")]

        cfg = OmegaConf.create(raw)
        for method_file in method_files:
            with open(method_file, 'r', encoding='utf-8') as f:
                method_cfg = yaml.safe_load(f) or {}
            cfg = OmegaConf.merge(cfg, OmegaConf.create(method_cfg))
        OmegaConf.set_struct(cfg, False)
        self._apply_runtime_overrides(cfg)
        set_seed(cfg.seed)
        return cfg

    def _apply_runtime_overrides(self, cfg):
        if "UNITRAJ_RL_FINETUNING_MODE" in os.environ:
            cfg.rl_finetuning_mode = os.environ["UNITRAJ_RL_FINETUNING_MODE"] in ("1", "true", "True")
        if "UNITRAJ_ADV_CONTROL" in os.environ:
            cfg.adv_control = os.environ["UNITRAJ_ADV_CONTROL"] in ("1", "true", "True")
        if "UNITRAJ_ADV_NUM" in os.environ:
            cfg.adv_num = int(os.environ["UNITRAJ_ADV_NUM"])
        return cfg

    def _init_inference_engine(self, specific_cfg):
        method_name = specific_cfg.get('model_name')
        planner_variant = str(specific_cfg.get('planner_variant', '')).strip().lower()
        if method_name == 'GT':
            return None

        if planner_variant == 'pluto_hybrid_blend':
            from unitraj.closeloop.pluto_hybrid_blend.pluto_hybrid_blend_inference import PlutoHybridBlendInference
            sim = PlutoHybridBlendInference(specific_cfg)
        elif planner_variant == 'pluto_hybrid':
            from unitraj.closeloop.pluto_hybrid.pluto_hybrid_inference import PlutoHybridInference
            sim = PlutoHybridInference(specific_cfg)
        elif method_name == 'VBD':
            sim = VBDInference(specific_cfg)
        elif method_name == 'Pluto':
            sim = PlutoInference(specific_cfg)
        elif method_name in ['wayformer', 'autobot', 'MTR']:
            from unitraj.closeloop.unitraj.unitraj_inference import UnitrajInference
            sim = UnitrajInference(specific_cfg)
        else:
            raise ValueError(f"Unknown method: {method_name}")

        sim.initialize_model()
        return sim

    def _call_inference_engine(
            self,
            sim,
            scenario_state,
            time_index,
            controlled_agent_id=None,
            score_selection=None,
            adversarial_target_agent_ids=None,
            required_agent_ids=None,
    ):
        """Call an inference engine without silently dropping configured semantics."""
        if sim is None:
            return None

        run_inference = sim.run_inference
        params = inspect.signature(run_inference).parameters
        kwargs = {}
        if controlled_agent_id is not None:
            if "controlled_agent_id" not in params:
                raise RuntimeError(
                    "inference engine does not accept controlled_agent_id"
                )
            kwargs["controlled_agent_id"] = controlled_agent_id
        if score_selection is not None:
            if "score_selection" not in params:
                raise RuntimeError(
                    "inference engine does not accept score_selection"
                )
            kwargs["score_selection"] = str(score_selection)
        if adversarial_target_agent_ids is not None:
            if "adversarial_target_agent_ids" not in params:
                raise RuntimeError(
                    "inference engine does not accept adversarial_target_agent_ids"
                )
            kwargs["adversarial_target_agent_ids"] = [
                str(target_id) for target_id in adversarial_target_agent_ids
            ]
        if required_agent_ids is not None and "required_agent_ids" in params:
            kwargs["required_agent_ids"] = [str(agent_id) for agent_id in required_agent_ids]
        return run_inference(scenario_state, time_index, **kwargs)

    def _normalize_inference_output(self, infer_out):
        """Normalize planner outputs to `(pred_traj, ref_lines, candidates, info)`."""
        if infer_out is None:
            return None, None, None, {}
        if isinstance(infer_out, tuple):
            pred_traj = infer_out[0]
            ref_lines = infer_out[1] if len(infer_out) > 1 else None
            candidates = infer_out[2] if len(infer_out) > 2 else None
            info = infer_out[3] if len(infer_out) > 3 else {}
            return pred_traj, ref_lines, candidates, info or {}
        return infer_out, None, None, {}

    def _resolve_track_id(self, track_id, available_ids):
        """Resolve ids robustly: exact match first, unique suffix match as fallback."""
        if track_id is None:
            return None

        track_id = str(track_id)
        available_ids = [str(aid) for aid in available_ids]
        if track_id in available_ids:
            return track_id

        suffix = track_id[-5:]
        matches = [aid for aid in available_ids if aid.endswith(suffix)]
        if len(matches) == 1:
            return matches[0]
        return None

    def _get_scenario_to_object_id_map(self):
        return {
            str(scenario_id): obj_id
            for obj_id, scenario_id in self.engine.traffic_manager.obj_id_to_scenario_id.items()
        }

    def install_external_feedback_policies(self):
        """Replace hardware-platform replay policies before the first warmup step."""
        scenario_to_obj_id = self._get_scenario_to_object_id_map()
        for scenario_id in sorted(self.external_execution_owned_ids):
            obj_id = scenario_to_obj_id.get(str(scenario_id))
            if obj_id is None:
                raise RuntimeError(
                    "cannot find MetaDrive object for external platform track %s"
                    % scenario_id
                )
            obj = self.engine.get_objects().get(obj_id)
            if obj is None:
                raise RuntimeError(
                    "cannot find MetaDrive object instance for external platform track %s"
                    % scenario_id
                )
            existing = self.engine.get_policy(obj_id)
            if not isinstance(existing, ExternalFeedbackMirrorPolicy):
                self.engine.add_policy(obj_id, ExternalFeedbackMirrorPolicy, obj)


    def install_initial_history_replay_policies(
            self,
            scenario_ids,
            replay_until_step,
    ):
        """Install dynamic replay before frame 0 for strict initial-track execution."""
        scenario = self.engine.data_manager.current_scenario
        tracks = scenario.get("tracks", {}) if isinstance(scenario, dict) else {}
        scenario_to_obj_id = self._get_scenario_to_object_id_map()
        replay_until_step = int(replay_until_step)
        if replay_until_step != int(self.WARMUP_STEPS):
            raise RuntimeError(
                "initial replay endpoint %d must equal WARMUP_STEPS=%d"
                % (replay_until_step, int(self.WARMUP_STEPS))
            )

        for scenario_id in scenario_ids:
            scenario_id = str(scenario_id)
            if scenario_id in self.external_execution_owned_ids:
                continue

            track = tracks.get(scenario_id)
            if track is None:
                raise RuntimeError(
                    "cannot find initial-history track %s" % scenario_id
                )

            for step_index in range(replay_until_step + 1):
                state = parse_object_state(track, step_index)
                if state is None or not bool(state.get("valid", False)):
                    raise RuntimeError(
                        "initial-history track %s is invalid at step %d"
                        % (scenario_id, step_index)
                    )

            obj_id = scenario_to_obj_id.get(scenario_id)
            if obj_id is None:
                raise RuntimeError(
                    "cannot find MetaDrive object for initial-history track %s"
                    % scenario_id
                )
            obj = self.engine.get_objects().get(obj_id)
            if obj is None:
                raise RuntimeError(
                    "cannot find MetaDrive object instance for initial-history track %s"
                    % scenario_id
                )

            self.engine.add_policy(
                obj_id,
                DynamicReplayTrafficParticipantPolicy,
                obj,
                track,
            )
            installed = self.engine.get_policy(obj_id)
            if not isinstance(installed, DynamicReplayTrafficParticipantPolicy):
                raise RuntimeError(
                    "failed to install dynamic initial replay for track %s"
                    % scenario_id
                )


    def act(self, agent_id: str):
        # MetaDrive will call policy.get_action_info() right after act().
        # So anything we want to surface to env.step()'s info must be placed into
        # self.action_info *within this function call*.
        self.action_info.clear()

        current_step = self.engine.episode_step
        time_index = current_step - 1
        sdc_track = self.current_state["tracks"][self.sdc_id]

        # --- 阶段一: 数据采集 (完全模仿 ReplayEgoCarPolicy) ---
        if time_index < self.WARMUP_STEPS:
            if not self.external_ego_control:
                self._handle_warmup(time_index, sdc_track)
            # 把前期的真实回放数据也录入 buffer
            ego_vehicle = self.control_object
            self.get_hist_buffer(time_index, [ego_vehicle.position, ego_vehicle.heading_theta, _vehicle_speed_mps(ego_vehicle)])

            return None

        # The first planning input must end at dataset/initial frame WARMUP_STEPS.
        # Without this transition placement, frame WARMUP_STEPS-1 is duplicated
        # and the user-confirmed initial endpoint is absent from Pluto history.
        if time_index == self.WARMUP_STEPS and not self.external_ego_control:
            self._handle_warmup(time_index, sdc_track)

        # 记录每帧闭环控制真实状态到历史里
        ego_vehicle = self.control_object
        self.get_hist_buffer(time_index, [ego_vehicle.position, ego_vehicle.heading_theta, _vehicle_speed_mps(ego_vehicle)])
        if self.use_adv:
            self.get_agent_hist_buffer(time_index)

        # --- 阶段二: 闭环规划 ---
        if time_index % self.replan_frequency == 0:
            self._plan_trajectory(time_index)

        # --- 阶段三: 主车闭环控制 ---
        # external ego is pose-owned by runtime actual feedback, not Pluto PID.
        ego_control_value = None if self.external_ego_control else self._compute_control()

        # Export info for rollout collection.
        if self._pending_cbv_rollout_samples:
            self.action_info["cbv_rollout_samples"] = self._pending_cbv_rollout_samples
            self._pending_cbv_rollout_samples = []

        if isinstance(self.info, dict):
            if "top_k_ids" in self.info:
                self.action_info["top_k_ids"] = self.info.get("top_k_ids")

        self.action_info["action"] = ego_control_value

        # --- 阶段四: 周车控制 (Adv Control) ---
        if time_index % self.replan_frequency == 0 and self.use_adv:
            self._apply_adv_control()
        return ego_control_value

    def _handle_warmup(self, time_index, sdc_track):
        """处理预热阶段：回放数据"""
        if time_index == 0:
            self.engine.agents['default_agent'].plan_traj = self.current_state['tracks'][self.sdc_id]['state'][
                "position"][:self.WARMUP_STEPS]
        state = parse_object_state(sdc_track, time_index)

        if state["valid"]:
            self.control_object.set_position(state["position"])
            self.control_object.set_velocity(state["velocity"])
            self.control_object.set_heading_theta(state["heading"])

    def _plan_trajectory(self, time_index):
        """按设定频率进行本车轨迹推理规划"""
        self.current_state = self.update_state(time_index)

        # ================= [1. 主车推理 ] =================
        infer_out = self._call_inference_engine(
            self.Sim,
            self.current_state,
            time_index,
            required_agent_ids=(self.current_cbv_ids if self.use_adv else None),
        )
        self._pred_traj, ref_lines, candidates, self.info = self._normalize_inference_output(infer_out)
        if self._pred_traj is None:
            return
        ego_vehicle = self.control_object
        current_heading = ego_vehicle.heading_theta
        self.engine.agents['default_agent'].plan_traj = self._convert_ego_model_traj_xy_to_center(
            self._pred_traj[0],
            default_heading=current_heading,
        )

        if hasattr(self.engine, "head_renderer") and self.engine.head_renderer is not None:
            self.engine.head_renderer.set_reference_lines(ref_lines)

        if candidates is not None:
            self.engine.agents['default_agent'].candidate_trajectories = self._convert_ego_candidate_trajs_xy_to_center(
                candidates,
                default_heading=current_heading,
            )

        if self.info is None:
            self.info = {}

        current_pos_rear = self._ego_model_reference_position(ego_vehicle)
        # 在新轨迹上找最接近当前位置的点
        pred_traj = self._pred_traj[0, :, :]
        distances = np.linalg.norm(pred_traj[:, :2] - current_pos_rear, axis=1)
        heading_vec = np.array([np.cos(current_heading), np.sin(current_heading)])
        vec_to_points = pred_traj[:, :2] - current_pos_rear
        dot = np.dot(vec_to_points, heading_vec)
        forward_mask = dot > 0
        if forward_mask.any():
            self.last_match_idx = int(np.argmin(distances * forward_mask + (~forward_mask) * 1e6))
        else:
            self.last_match_idx = int(np.argmin(distances))
        self.last_match_idx = max(0, min(self.last_match_idx, len(pred_traj) - 1))
        # print(self.last_match_idx)

        # ================= [2. 周车推理 ] =================
        self.adv_agent_ids = []
        self.current_adv_trajs = None
        self.current_aligned_adv_trajs = {}
        self.current_adv_patch_info = {}

        if self.AdvSim is not None and self.use_adv:
            # --- 从主车预测中选择一次关键周车并锁定 ---
            self._select_cbv(time_index)
            self._run_adv_inference(time_index)

        if self.info is not None and 'top_k_ids' in self.info:
            self.engine.pred_agent_ids = self.info['top_k_ids']
        else:
            self.engine.pred_agent_ids = [self.sdc_id]

        # if not hasattr(self, "traj_debug"):
        #     self.traj_debug = []
        # self.traj_debug.append({
        #     "frame": self.engine.episode_step,
        #     "traj": self._pred_traj[0, :, :2].copy()
        # })

    def _select_cbv(self, time_index):
        """利用主车推理给出的周车未来预测，单场景锁定一辆关键周车 CBV。"""
        self.info['top_k_ids'] = []
        if 'prediction' not in self.info or 'agent_tokens' not in self.info or self._pred_traj is None:
            return

        tokens = self.info['agent_tokens']

        if self.cbv_selection_locked:
            if (self.current_cbv_ids is not None and
                all(cbv_id in self.current_state['tracks']  for cbv_id in self.current_cbv_ids)):
                self.info['top_k_ids'] = self.current_cbv_ids
            return

        predictions = self.info['prediction']
        if predictions is None or len(predictions) == 0 or len(tokens) <= 1:
            return

        ego_future = self._pred_traj[0, :, :2]
        ego_now = np.asarray(self.current_state['tracks'][self.sdc_id]['state']['position'][time_index][:2], dtype=np.float64)
        ego_heading = float(self.current_state['tracks'][self.sdc_id]['state']['heading'][time_index])
        ego_heading_vec = np.array([np.cos(ego_heading), np.sin(ego_heading)], dtype=np.float64)

        best_score = -np.inf
        best_entry = None
        debug_entries = []
        token_offset = 1 if len(tokens) == len(predictions) + 1 else 0

        for pred_idx in range(len(predictions)):
            token_idx = pred_idx + token_offset
            if token_idx >= len(tokens):
                break

            cbv_id = tokens[token_idx]
            if cbv_id not in self.current_state['tracks']:
                continue

            track_type = self.current_state['tracks'][cbv_id].get('type', '')
            is_pedestrian = ('PEDESTRIAN' in str(track_type).upper()) or (track_type == 1)
            is_cyclist = ('CYCLIST' in str(track_type).upper()) or (track_type == 2)
            if is_pedestrian and not self.allow_pedestrian_cbv: continue
            if is_cyclist and not self.allow_cyclist_cbv: continue
            agent_future = np.asarray(predictions[pred_idx, :, :2], dtype=np.float64)
            min_t = min(len(ego_future), len(agent_future))
            if min_t < 2:
                continue

            ego_xy = np.asarray(ego_future[:min_t], dtype=np.float64)
            agent_xy = agent_future[:min_t]
            dist = np.linalg.norm(ego_xy - agent_xy, axis=-1)

            min_dist = float(dist.min())
            risk_distance = float(np.clip((12.0 - min_dist) / 12.0, 0.0, 1.0))
            overlap_ratio = float((dist < 4.0).mean())

            cbv_now = np.asarray(self.current_state['tracks'][cbv_id]['state']['position'][time_index][:2], dtype=np.float64)
            rel_now = cbv_now - ego_now
            front_score = float(np.clip(np.dot(rel_now, ego_heading_vec) / 20.0, 0.0, 1.0))

            dist_delta = dist[:-1] - dist[1:]
            closing_mask = dist_delta > 1e-3
            if closing_mask.any():
                closing_speed = float(np.max(dist_delta[closing_mask]) / max(FIXED_DT, 1e-3))
                ttc = min_dist / max(closing_speed, 1e-3)
                ttc_score = float(np.clip((5.0 - ttc) / 5.0, 0.0, 1.0))
            else:
                ttc = float("inf")
                ttc_score = 0.0

            score = 0.45 * risk_distance + 0.30 * ttc_score + 0.15 * overlap_ratio + 0.10 * front_score
            debug_entries.append({
                "agent_id": str(cbv_id),
                "score": float(score),
                "min_dist_m": min_dist,
                "ttc_s": None if not np.isfinite(ttc) else float(ttc),
                "overlap_ratio": overlap_ratio,
                "front_score": front_score,
            })
            if score > best_score:
                best_score = score
                best_entry = cbv_id

        self.info['cbv_candidate_scores'] = debug_entries
        if best_entry is None:
            return

        sorted_entries = sorted(debug_entries, key=lambda x: x['score'], reverse=True)
        # TODO： 是否增加规则过滤，距离速度等
        self.current_cbv_ids = [
            entry['agent_id']
            for entry in sorted_entries[:self.max_cbv_num]
        ]

        self.cbv_selection_locked = True
        self.info['top_k_ids'] = self.current_cbv_ids


    def _planner_interaction_kwargs(self):
        """Provide time-aligned world-frame futures for CBV collision checking.

        TJ-West previously called ``run_inference_batch`` without an Ego future.
        The batch planner explicitly excludes Ego from replay futures, so normal
        CBV collision checking could not see the planned Ego path.  We provide
        the current Ego plan plus time-shifted previous CBV plans here.
        """
        kwargs = {}
        horizon = int(self.platform_output_point_num)

        if self._pred_traj is not None:
            ego_arr = np.asarray(self._pred_traj)
            if ego_arr.ndim == 3 and len(ego_arr) > 0:
                ego_arr = ego_arr[0]
            if ego_arr.ndim == 2 and len(ego_arr) >= horizon:
                kwargs["ego_future_traj"] = ego_arr[:horizon].copy()

        previous = {}
        current_step = int(self.engine.episode_step)
        for agent_id, record in dict(self._previous_adv_plan_by_id).items():
            if not isinstance(record, dict):
                continue
            traj = np.asarray(record.get("traj"))
            start_step = int(record.get("start_step", current_step))
            age = max(0, current_step - start_step)
            if traj.ndim != 2 or age + horizon > len(traj):
                continue
            previous[str(agent_id)] = traj[age:age + horizon].copy()
        if previous:
            kwargs["previous_agent_futures_by_id"] = previous
        return kwargs

    def _run_adv_inference(self, time_index):
        """处理周车推理，按 controlled_agent_id 为关键周车构造 Pluto 特征。"""
        dynamic_adv_ids = self.info.get('top_k_ids', [])
        if dynamic_adv_ids:
            target_ids = dynamic_adv_ids
        else:
            target_ids = []
        if not target_ids:
            print("[Policy] Warning: target_ids is empty, AdvSim will not run.")
            return

        all_cbv_trajs = []
        all_cbv_ids = []
        real_adv_ids = []
        self.current_adv_ref_lines = {}
        self.current_adv_candidates = {}
        if hasattr(self.engine, "head_renderer") and self.engine.head_renderer is not None:
            if hasattr(self.engine.head_renderer, "clear_overlays"):
                # Ego reference lines are set in _plan_trajectory; only clear CBV reference lines here.
                self.engine.head_renderer.cbv_reference_lines = []

        for adv_id_str in target_ids:
            real_adv_id = self._resolve_track_id(adv_id_str, self.current_state['tracks'].keys())
            if not real_adv_id or real_adv_id not in self.current_state['tracks']:
                continue
            real_adv_ids.append(real_adv_id)

        if not real_adv_ids:
            return

        # Keep the platform policy aligned with the algorithm-side simulation
        # policy. The currently delivered algorithm stack does not provide a
        # CBVAdversarialReward.evaluate(..., candidate_labels=...) interface,
        # so explicitly requesting "adversarial" here enters an incompatible
        # algorithm branch and raises TypeError. Do not pass planner-specific
        # arguments until the matching reward implementation is delivered.
        score_selections = {
            real_adv_id: self._planner_mode_for_scenario_id(real_adv_id)
            for real_adv_id in real_adv_ids
        }
        interaction_kwargs = (
            self._planner_interaction_kwargs()
            if hasattr(self, "_planner_interaction_kwargs") else {}
        )
        if hasattr(self.AdvSim, "run_inference_batch"):
            batch_runner = self.AdvSim.run_inference_batch
            batch_params = inspect.signature(batch_runner).parameters
            # Interaction futures are Pluto/Hybrid-specific.  Filter against the
            # concrete batch signature so VBD and other backends keep their
            # existing call contract.
            batch_kwargs = {
                key: value
                for key, value in interaction_kwargs.items()
                if key in batch_params
            }
            if "score_selections" in batch_params:
                batch_kwargs["score_selections"] = score_selections
            elif str(self.adv_method).strip().lower() in {"pluto", "pluto_hybrid", "pluto_hybrid_blend"}:
                raise RuntimeError(
                    "%s batch inference must accept score_selections"
                    % self.adv_method
                )
            adv_outputs = batch_runner(
                self.current_state,
                time_index,
                real_adv_ids,
                **batch_kwargs,
            )
        else:
            adv_outputs = {}
            run_params = inspect.signature(self.AdvSim.run_inference).parameters
            accepts_selection = "score_selection" in run_params
            for real_adv_id in real_adv_ids:
                adv_outputs[real_adv_id] = self._call_inference_engine(
                    self.AdvSim,
                    self.current_state,
                    time_index,
                    controlled_agent_id=real_adv_id,
                    score_selection=(
                        score_selections[real_adv_id]
                        if accepts_selection else None
                    ),
                )

        for real_adv_id in real_adv_ids:
            adv_infer_out = adv_outputs.get(real_adv_id)
            adv_traj_all, adv_ref_lines, adv_candidates, adv_info = self._normalize_inference_output(adv_infer_out)
            if adv_traj_all is None:
                continue

            if adv_traj_all.ndim == 3:  # (16, T, 5)
                all_cbv_trajs.append(adv_traj_all[0])
            else:
                all_cbv_trajs.append(adv_traj_all[0, 0])
            all_cbv_ids.append(real_adv_id)
            self.current_adv_ref_lines[str(real_adv_id)] = adv_ref_lines
            self.current_adv_candidates[str(real_adv_id)] = adv_candidates

            if adv_ref_lines is not None and hasattr(self.engine, "head_renderer") and self.engine.head_renderer is not None:
                cbv_idx = len(all_cbv_ids) - 1
                cbv_colors = [(0, 200, 255), (255, 100, 100), (80, 220, 120), (255, 180, 0)]
                color = cbv_colors[cbv_idx % len(cbv_colors)]
                if hasattr(self.engine.head_renderer, "add_cbv_reference_lines"):
                    self.engine.head_renderer.add_cbv_reference_lines(adv_ref_lines, color=color)

            rl_finetuning_enabled = bool(self.base_cfg.get("rl_finetuning_mode", False))
            if rl_finetuning_enabled and adv_info is not None and isinstance(adv_info, dict):
                cbv_samples = adv_info.get('cbv_rollout_samples', None)
                if cbv_samples:
                    merged_samples = self.info.setdefault('cbv_rollout_samples', [])
                    for sample in cbv_samples:
                        if not isinstance(sample, dict):
                            continue
                        sample = dict(sample)
                        sample_info = dict(sample.get("info", {}))
                        sample_info.setdefault("frame_idx", int(time_index))
                        sample_info.setdefault("cbv_id", str(real_adv_id))
                        sample_info.setdefault("controlled_agent_id", str(real_adv_id))
                        sample["info"] = sample_info
                        merged_samples.append(sample)

                        # Also enqueue for immediate export via env.step() info.
                        self._pending_cbv_rollout_samples.append(sample)

        if len(all_cbv_trajs) > 0:
            self.current_adv_trajs = np.stack(all_cbv_trajs)
            self.adv_agent_ids = all_cbv_ids

    def _apply_adv_control(self):
        """将周车的预测轨迹应用到引擎中"""
        if self.current_adv_trajs is None or self.info is None:
            return

        # IMPORTANT:
        # - `time_index = engine.episode_step - 1` in our act() logic
        # - Replay traffic is applied in ScenarioTrafficManager.after_step(), using policy.episode_step
        #   which equals engine.episode_step for *this* env.step call.
        # Therefore we should start patching from `engine.episode_step`.
        start_step = max(int(self.engine.episode_step), 0)

        # 1. 建立输出 ID 到 Index 的映射，优先使用完整 scenario id
        adv_id_map = {str(aid): i for i, aid in enumerate(self.adv_agent_ids)}

        # 2. 建立 ScenarioID 到 ObjID 的映射
        scenario_to_obj_id = self._get_scenario_to_object_id_map()

        # 算关键周车id
        dynamic_adv_ids = self.info.get('top_k_ids', [])
        if dynamic_adv_ids:
            target_ids = dynamic_adv_ids
        else:
            target_ids = []

        if not hasattr(self, 'adv_expert_trajs_cache'):
            self.adv_expert_trajs_cache = {}

        target_scene_ids = set()
        for aid in target_ids:
            resolved_id = self._resolve_track_id(aid, scenario_to_obj_id.keys())
            if resolved_id is not None:
                target_scene_ids.add(str(resolved_id))
        # 抹除已经过去（不再是 CBV）的对抗车轨迹（排除主车）
        for oid, scene_id in self.engine.traffic_manager.obj_id_to_scenario_id.items():
            if str(scene_id) != self.sdc_id and str(scene_id) not in target_scene_ids:
                obj = self.engine.get_objects().get(oid)
                if obj is not None:
                    if hasattr(obj, 'plan_traj'):
                        delattr(obj, 'plan_traj')
                    if hasattr(obj, 'expert_traj'):
                        delattr(obj, 'expert_traj')

        for adv_id_str in target_ids:
            resolved_adv_id = self._resolve_track_id(adv_id_str, self.adv_agent_ids)
            if resolved_adv_id is None:
                continue
            idx = adv_id_map.get(str(resolved_adv_id), None)
            if idx is None:
                continue

            adv_traj = self.current_adv_trajs[idx]
            adv_obj_id = scenario_to_obj_id.get(str(resolved_adv_id), None)

            if adv_obj_id is None:
                continue

            adv_obj = self.engine.get_objects().get(adv_obj_id)
            if adv_obj is None:
                continue

            if str(resolved_adv_id) in self.external_execution_owned_ids:
                source_traj = np.asarray(adv_traj)
                if source_traj.ndim != 2 or len(source_traj) < int(self.platform_output_point_num):
                    raise RuntimeError(
                        "external platform Pluto trajectory %s requires at least %d points"
                        % (resolved_adv_id, int(self.platform_output_point_num))
                    )
                self.current_aligned_adv_trajs[str(resolved_adv_id)] = source_traj
                self._previous_adv_plan_by_id[str(resolved_adv_id)] = {
                    "start_step": int(start_step),
                    "traj": np.array(source_traj, copy=True),
                }
                planner_source = str(getattr(self.AdvSim, "trajectory_model_name", getattr(self, "adv_method", "Pluto"))).strip().lower()
                self.current_adv_patch_info[str(resolved_adv_id)] = {
                    "patch_offset": 0,
                    "match_info": {},
                    "source": "%s_raw_external_platform" % planner_source,
                }
                # Planning output is exported/rendered only. Never patch track or replay it.
                source_xy_center = self._convert_adv_model_traj_xy_to_center(
                    adv_obj,
                    source_traj,
                    default_heading=float(getattr(adv_obj, "heading_theta", 0.0)),
                )
                adv_obj.plan_traj = source_xy_center
                existing_policy = self.engine.get_policy(adv_obj_id)
                if not isinstance(existing_policy, ExternalFeedbackMirrorPolicy):
                    self.engine.add_policy(
                        adv_obj_id,
                        ExternalFeedbackMirrorPolicy,
                        adv_obj,
                    )
                self._controlled_cbv_scenario_ids.add(str(resolved_adv_id))
                continue

            # Pluto output is a temporal sequence: row 0 is the state at t+0.1 s.
            # Spatially cropping this array changes its time meaning and was the cause
            # of the first planned point no longer corresponding to initial frame 20.
            source_traj = np.asarray(adv_traj)
            required = max(int(self.platform_output_point_num), 2)
            if source_traj.ndim != 2 or len(source_traj) < required:
                raise RuntimeError(
                    "CBV Pluto trajectory %s requires at least %d temporal points"
                    % (resolved_adv_id, required)
                )
            self.current_aligned_adv_trajs[str(resolved_adv_id)] = source_traj
            self._previous_adv_plan_by_id[str(resolved_adv_id)] = {
                "start_step": int(start_step),
                "traj": np.array(source_traj, copy=True),
            }
            planner_source = str(getattr(self.AdvSim, "trajectory_model_name", getattr(self, "adv_method", "Pluto"))).strip().lower()
            self.current_adv_patch_info[str(resolved_adv_id)] = {
                "patch_offset": 0,
                "match_info": {},
                "source": "%s_temporal_plan" % planner_source,
            }

            # NOTE: 主车后轴，cbv应为中心
            source_heading = self._extract_traj_heading(
                source_traj,
                default_heading=float(getattr(adv_obj, "heading_theta", 0.0)),
            )
            source_velocity = self._extract_traj_velocity_xy(source_traj)
            has_heading = source_heading is not None
            source_xy_center = self._convert_adv_model_traj_xy_to_center(
                adv_obj,
                source_traj,
                default_heading=float(getattr(adv_obj, "heading_theta", 0.0)),
            )

            # 可视化更新（CENTER）
            candidate_trajs = self.current_adv_candidates.get(str(resolved_adv_id), None)
            if candidate_trajs is not None:
                adv_obj.candidate_trajectories = self._convert_candidate_trajs_xy_to_center(
                    adv_obj,
                    candidate_trajs,
                    default_heading=float(getattr(adv_obj, "heading_theta", 0.0)),
                )
            elif hasattr(adv_obj, 'candidate_trajectories'):
                delattr(adv_obj, 'candidate_trajectories')
            adv_obj.plan_traj = source_xy_center
            top_down_renderer = getattr(getattr(self.engine, "head_renderer", None), "head_top_down_renderer", None)
            if top_down_renderer is not None:
                top_down_renderer.cbv_id.append(adv_obj_id)

            real_adv_id = self.engine.traffic_manager.obj_id_to_scenario_id[adv_obj_id]

            # Patch BOTH:
            # 1) engine.data_manager.current_scenario (so traffic replay uses it)
            # 2) self.current_state (for visualization caches / debugging)
            engine_scene = self.engine.data_manager.current_scenario
            if engine_scene is None or 'tracks' not in engine_scene or real_adv_id not in engine_scene['tracks']:
                continue
            engine_track = engine_scene['tracks'][real_adv_id]
            engine_state = engine_track['state']

            # Make sure this CBV uses a dynamic replay policy, otherwise the cached traj_info will ignore our patches.
            existing_policy = self.engine.get_policy(adv_obj_id)
            if not isinstance(existing_policy, DynamicReplayTrafficParticipantPolicy):
                self.engine.add_policy(adv_obj_id, DynamicReplayTrafficParticipantPolicy, adv_obj, engine_track)
            tracking_policy = self.engine.get_policy(adv_obj_id)
            if hasattr(tracking_policy, "configure_execution_limits"):
                tracking_policy.configure_execution_limits(
                    max_speed_mps=float(self.base_cfg.get("candidate_max_speed_mps", 12.0)),
                    speed_kp=float(self.base_cfg.get("cbv_execution_speed_kp", 1.5)),
                    overspeed_margin_mps=float(
                        self.base_cfg.get("cbv_execution_overspeed_margin_mps", 0.05)
                    ),
                )
            if hasattr(tracking_policy, "activate_tracking"):
                tracking_policy.activate_tracking()

            # Remember we have a controlled CBV so update_state() can include its updated track.
            self._controlled_cbv_scenario_ids.add(real_adv_id)

            adv_pos = engine_state.get('position', None)
            adv_heading = engine_state.get('heading', None)
            adv_velocity = engine_state.get('velocity', None)
            adv_valid = engine_state.get('valid', None)

            if adv_pos is None:
                continue

            # 使用实例变量对每辆车分别保存真值轨迹，避免多个不同车辆发生轨迹串台乱画
            if real_adv_id not in self.adv_expert_trajs_cache:
                adv_valid_mask = np.array(engine_state.get('valid', []), dtype=bool).reshape(-1)
                if adv_valid_mask.any():
                    self.adv_expert_trajs_cache[real_adv_id] = np.copy(engine_state["position"][adv_valid_mask])
            adv_obj.expert_traj = self.adv_expert_trajs_cache.get(real_adv_id, None)

            # 处理轨迹长度
            pred_length = len(source_traj)
            adv_length = len(adv_pos)
            has_vel = source_velocity is not None

            remain_len = min(pred_length, max(adv_length - start_step, 0))
            if remain_len <= 0:
                continue

            # 1) position (CENTER)
            adv_pos[start_step:start_step + remain_len, :2] = source_xy_center[:remain_len, :2]

            # 2) heading
            if has_heading and adv_heading is not None:
                target_slice = adv_heading[start_step:start_step + remain_len]
                source_data = source_heading[:remain_len]
                if getattr(target_slice, 'ndim', 1) == 2 and getattr(source_data, 'ndim', 1) == 1:
                    source_data = source_data.reshape(-1, 1)
                adv_heading[start_step:start_step + remain_len] = source_data

            # 3) velocity
            if adv_velocity is not None:
                if not has_vel:
                    raise RuntimeError(
                        "Pluto CBV trajectory is missing vx/vy for %s" % resolved_adv_id
                    )
                adv_velocity[start_step:start_step + remain_len, :2] = source_velocity[:remain_len, :2]

            # 4) valid mask: keep the controlled CBV alive during the horizon
            if adv_valid is not None:
                adv_valid[start_step:start_step + remain_len] = True

            # Keep self.current_state in sync (best-effort; it is used for inference input snapshots).
            if self.current_state is not None and 'tracks' in self.current_state and real_adv_id in self.current_state[
                'tracks']:
                try:
                    cs = self.current_state['tracks'][real_adv_id]['state']
                    if 'position' in cs:
                        cs['position'][start_step:start_step + remain_len, :2] = source_xy_center[:remain_len, :2]
                    if has_heading and 'heading' in cs:
                        cs['heading'][start_step:start_step + remain_len] = source_heading[:remain_len]
                    if has_vel and 'velocity' in cs:
                        cs['velocity'][start_step:start_step + remain_len, :2] = source_velocity[:remain_len, :2]
                    if 'valid' in cs:
                        cs['valid'][start_step:start_step + remain_len] = True
                except Exception:
                    pass

    def _compute_control(self):
        """计算本车 PID 控制动作"""
        pred_traj = self._pred_traj[0, :-1, :]

        ego_vehicle = self.control_object
        current_heading = ego_vehicle.heading_theta
        current_speed = _vehicle_speed_mps(ego_vehicle)
        current_pos = ego_vehicle.position
        current_pos_rear = self._ego_model_reference_position(ego_vehicle)

        # ===== RLPlanningPolicy-style tracking =====
        # 1) match 点用于横向误差，应贴近当前后轴；lookahead 只用于目标点
        heading_vec = np.array([np.cos(current_heading), np.sin(current_heading)], dtype=np.float32)

        # 2) 找离预瞄点最近且在车头前方的 match 点
        # distances = np.linalg.norm(pred_traj[:, :2] - preview_pos, axis=1)
        # match_idx = self.get_match_idx(pred_traj, distances, current_pos_rear, current_heading)
        # p_match = pred_traj[match_idx, :2]

        distances = np.linalg.norm(pred_traj[:, :2] - current_pos_rear, axis=1)
        match_idx, _ = self._compute_match_idx(
            pred_traj=pred_traj,
            distances=distances,
            current_pos=current_pos_rear,
            current_heading=current_heading,
            last_match_idx=self.last_match_idx,
            allow_stateful_smoothing=False,
        )
        match_idx = min(match_idx + 2, len(pred_traj[:, :2]) - 1)
        self.last_match_idx = match_idx
        p_match = pred_traj[match_idx, :2]

        # 3) 选取目标点 (lookahead)
        lookahead_distance = np.clip(current_speed * 1.2, 3.0, 6.0)
        target_idx = self._find_lookahead_idx(pred_traj[:, :2], match_idx, lookahead_distance)
        target_idx = self._ensure_forward_target(
            pred_traj[:, :2], match_idx, target_idx, current_pos_rear, current_heading
        )
        p_target = pred_traj[target_idx, :2]

        # # ========== 目标点合法性最终校验与外推 ==========
        # fallback_mode = "none"
        # if np.dot(p_target - current_pos_rear, heading_vec) <= 0:
        #     fallback_idx = self._find_forward_anchor_idx(
        #         pred_traj[:, :2],
        #         current_pos_rear,
        #         current_heading,
        #         preferred_start=max(match_idx - 5, 0),
        #     )
        #     if fallback_idx is not None:
        #         target_idx = int(fallback_idx)
        #         p_target = pred_traj[target_idx, :2]
        #         if np.dot(p_match - current_pos_rear, heading_vec) <= 0:
        #             p_match = pred_traj[target_idx, :2]
        #         fallback_mode = "trajectory_forward_anchor"
        #     else:
        #         # Last resort: stay close to the matched path point and only nudge slightly forward.
        #         p_target = p_match + heading_vec * min(lookahead_distance, 1.5)
        #         fallback_mode = "matched_point_nudge"
        self.engine.agents['default_agent'].key_points = [p_target, p_match, current_pos_rear]

        # 4) 算轨迹朝向 (切向)
        K_TANGENT = 3
        idx2 = min(match_idx + K_TANGENT, len(pred_traj) - 1)
        p_dir = pred_traj[idx2, :2]
        vec_traj_dir = p_dir - p_match
        if np.linalg.norm(vec_traj_dir) < 1e-3:
            vec_traj_dir = heading_vec
        else:
            if np.dot(vec_traj_dir, heading_vec) < 0:
                vec_traj_dir = heading_vec

        # 5) 横向与航向误差
        lateral_error_m = self._signed_cross_track_error(current_pos_rear, p_target, vec_traj_dir)

        if pred_traj.shape[-1] >= 4:
            # 索引 2, 3 对应模型输出的 [cos(yaw), sin(yaw)]
            traj_heading = np.arctan2(pred_traj[idx2, 3], pred_traj[idx2, 2])
        else:
            traj_heading = np.arctan2(vec_traj_dir[1], vec_traj_dir[0])

        heading_error = traj_heading - current_heading
        heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi
        # combined_error = lateral_error + 25 * heading_error
        k_stanley = 1.4
        combined_error = heading_error - np.arctan2(k_stanley * lateral_error_m, max(current_speed, 2.0))

        # 6) 纵向误差: 优先从模型预测中直接读取速度以避免震荡
        if pred_traj.shape[-1] >= 6:
            # 索引 4, 5 对应模型输出的 [vx, vy]
            # pred_vx = pred_traj[min(match_idx + 1, len(pred_traj) - 1), 4]
            # pred_vy = pred_traj[min(match_idx + 1, len(pred_traj) - 1), 5]
            speed_idx = min(max(match_idx + 1, target_idx), len(pred_traj) - 1)
            pred_vx = pred_traj[speed_idx, 4]
            pred_vy = pred_traj[speed_idx, 5]
            target_speed = float(np.hypot(pred_vx, pred_vy))
        else:
            # 回退到根据距离的空间差分法计算目标速度
            step_diff = 5
            if match_idx + step_diff < len(pred_traj):
                target_speed_idx = match_idx + step_diff
                p_target_speed = pred_traj[target_speed_idx, :2]
                target_speed = float(np.linalg.norm(p_target_speed - p_match) / (step_diff * FIXED_DT))
            else:
                available_steps = len(pred_traj) - 1 - match_idx
                if available_steps > 0:
                    p_target_speed = pred_traj[-1, :2]
                    target_speed = float(np.linalg.norm(p_target_speed - p_match) / (available_steps * FIXED_DT))
                elif len(pred_traj) > 1:
                    target_speed = float(np.linalg.norm(pred_traj[-1, :2] - pred_traj[-2, :2]) / FIXED_DT)
                else:
                    target_speed = current_speed

        def estimate_traj_turning(pred_traj, start_idx, horizon=12):
            xy = pred_traj[:, :2]
            end = min(start_idx + horizon, len(xy) - 1)
            if end <= start_idx + 2:
                return 0.0
            vec = np.diff(xy[start_idx:end + 1], axis=0)
            heading = np.arctan2(vec[:, 1], vec[:, 0])
            dheading = np.diff(np.unwrap(heading))
            return float(np.sum(np.abs(dheading)))
        turn_amount = estimate_traj_turning(pred_traj, match_idx, horizon=12)
        if turn_amount > 0.45:
            target_speed = min(target_speed, 3.0)
        elif turn_amount > 0.25:
            target_speed = min(target_speed, 4.0)
        steering = self.lateral_pid.get_result(-combined_error)

        speed_error = current_speed - target_speed
        throttle_brake_raw = self.longitudinal_pid.get_result(speed_error)
        steering_cmd = float(np.clip(steering, -1.0, 1.0))
        throttle_brake = float(np.clip(throttle_brake_raw, -1.0, 1.0))

        if os.environ.get("AUTOSCENE_EGO_EXEC_DEBUG", "0") in ("1", "true", "True"):
            if int(self.episode_step) % 5 == 0:
                selected_label = self.info.get("selected_osm_label") if isinstance(self.info, dict) else None
                print(
                    "[EgoExecution]",
                    {
                        "step": int(self.episode_step),
                        "actual_speed_mps": round(float(current_speed), 4),
                        "target_speed_mps": round(float(target_speed), 4),
                        "speed_error_mps": round(float(speed_error), 4),
                        "throttle_raw": round(float(throttle_brake_raw), 4),
                        "throttle_cmd": round(float(throttle_brake), 4),
                        "steering_raw": round(float(steering), 4),
                        "steering_cmd": round(float(steering_cmd), 4),
                        "match_idx": int(match_idx),
                        "target_idx": int(target_idx),
                        "turn_amount": round(float(turn_amount), 4),
                        "selected_osm_label": selected_label,
                    },
                )
        return [steering_cmd, throttle_brake]

    def _find_lookahead_idx(self, traj_xy: np.ndarray, start_idx: int, lookahead_m: float) -> int:
        """Find an index forward along the trajectory by accumulating arc-length from start_idx."""
        n = len(traj_xy)
        if n == 0:
            return 0
        if n == 1:
            return 0
        start_idx = int(np.clip(start_idx, 0, n - 1))

        acc = 0.0
        i = start_idx
        while i < n - 1 and acc < lookahead_m:
            step = float(np.linalg.norm(traj_xy[i + 1] - traj_xy[i]))
            # protect against duplicate points
            if step < 1e-4:
                step = 0.0
            acc += step
            i += 1
        return i

    def _ensure_forward_target(
            self,
            traj_xy: np.ndarray,
            match_idx: int,
            target_idx: int,
            current_pos: np.ndarray,
            current_heading: float,
            max_extra_steps: int = 20,
    ) -> int:
        """Ensure the chosen target point lies in front of the ego (dot>0); otherwise step forward."""
        n = len(traj_xy)
        if n == 0:
            return 0
        heading_vec = np.array([np.cos(current_heading), np.sin(current_heading)], dtype=np.float32)

        idx = int(np.clip(target_idx, match_idx, n - 1))
        for _ in range(max_extra_steps):
            v = traj_xy[idx] - current_pos
            if float(np.dot(v, heading_vec)) > FORWARD_DOT_MIN:
                return idx
            if idx >= n - 1:
                break
            idx += 1
        return idx

    def _find_forward_anchor_idx(
            self,
            traj_xy: np.ndarray,
            current_pos: np.ndarray,
            current_heading: float,
            preferred_start: int = 0,
    ):
        """Pick the closest point that is genuinely in front of the ego, preferring later points."""
        n = len(traj_xy)
        if n == 0:
            return None
        heading_vec = np.array([np.cos(current_heading), np.sin(current_heading)], dtype=np.float32)
        start = int(np.clip(preferred_start, 0, n - 1))
        best_idx = None
        best_dist = None
        for idx in range(start, n):
            v = traj_xy[idx] - current_pos
            if float(np.dot(v, heading_vec)) <= FORWARD_DOT_MIN:
                continue
            dist = float(np.linalg.norm(v))
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_idx = idx
        return best_idx

    def _signed_cross_track_error(self, p: np.ndarray, p_on_path: np.ndarray, path_tangent: np.ndarray) -> float:
        """Signed lateral distance from point p to the line passing p_on_path with direction path_tangent."""
        t = np.asarray(path_tangent, dtype=np.float32)
        tn = float(np.linalg.norm(t))
        if tn < 1e-6:
            return 0.0
        t = t / tn
        # left normal of tangent
        n = np.array([-t[1], t[0]], dtype=np.float32)
        return float(np.dot(np.asarray(p, dtype=np.float32) - np.asarray(p_on_path, dtype=np.float32), n))

    def update_state(self, time_index):
        """Return a patched scenario state for inference without mutating the engine's original scenario.

        由于 `base_state` 每次从 self._base_scenario_cache 拉取最原始的未被污染的纯净录像数据，
        为了确保从仿真第 0 步到当前的 `time_index` 的所有闭环发生后的状态都不丢失被覆盖，
        我们需要把截止目前所有沉积在 hist_buffer 中的实际仿真轨迹全部覆写上去。
        """
        # Base is ALWAYS the pure original scenario (Deepcopied once at __init__)
        base_state = self._base_scenario_cache
        length = int(base_state.get('length', 0))

        # During warmup we should not patch; also if nothing to patch, return base_state.
        if time_index <= 0 or length <= 0 or len(self.hist_buffer) == 0:
            return base_state

        # Create a lightweight shallow copy of dicts to preserve pure base_state references
        patched_state = dict(base_state)
        patched_tracks = dict(base_state.get('tracks', {}))
        patched_state['tracks'] = patched_tracks

        base_track = base_state['tracks'][self.sdc_id]
        patched_track = dict(base_track)
        patched_tracks[self.sdc_id] = patched_track

        base_track_state = base_track['state']
        patched_track_state = dict(base_track_state)
        patched_track['state'] = patched_track_state

        # NumPy Copy arrays so in-place assignment won't affect the pure fallback cache
        patched_traj = np.array(base_track_state['position'], copy=True)
        patched_head = np.array(base_track_state['heading'], copy=True)
        patched_vel = np.array(base_track_state['velocity'], copy=True)
        patched_acc = np.array(
            base_track_state.get('acceleration', np.zeros_like(patched_vel)),
            copy=True,
        )
        patched_valid = np.array(
            base_track_state.get('valid', np.ones((len(patched_traj),), dtype=bool)),
            dtype=bool,
            copy=True,
        ).reshape(-1)
        if len(patched_valid) < len(patched_traj):
            patched_valid = np.pad(
                patched_valid,
                (0, len(patched_traj) - len(patched_valid)),
                constant_values=False,
            )
        # print('GT',patched_traj)

        # 为了保证不漏掉任何历史信息，强制对齐：
        # hist_buffer 是每仿真步 append 新状态的列表，因此:
        # hist_buffer[0] 对应 timestep = 0
        # hist_buffer[h] 对应 timestep = h

        # 将整个已发生的所有闭环历史逐帧覆写进去
        for step_idx, state in self.hist_buffer.items():
            if 0 <= step_idx < length:
                patched_traj[step_idx, :2] = state['position']
                patched_head[step_idx] = state['heading']
                patched_vel[step_idx, :2] = state['velocity']
                if 'acceleration' in state:
                    patched_acc[step_idx, :2] = state['acceleration'][:2]
                # ``history_source=initial`` invalidates untouched dataset future.
                # A frame becomes semantically valid again only after it has really
                # occurred in the closed-loop history. CBVs already did this; Ego
                # must obey the same rule.
                patched_valid[step_idx] = True

        # print(f'[StateUpdate] Successfully patched {min(k, length)} history frames up to {time_index}')

        patched_track_state['position'] = patched_traj
        patched_track_state['heading'] = patched_head
        patched_track_state['velocity'] = patched_vel
        patched_track_state['acceleration'] = patched_acc
        patched_track_state['valid'] = patched_valid

        # CBV 历史更新
        for sid, hist in self.agent_hist_buffer.items():
            if sid not in base_state.get("tracks", {}):
                continue
            base_cbv_track = base_state["tracks"][sid]
            base_cbv_state = base_cbv_track["state"]

            patched_track = dict(base_cbv_track)
            patched_tracks[sid] = patched_track

            track_state = dict(base_cbv_state)
            patched_track["state"] = track_state

            pos = np.array(base_cbv_state["position"], copy=True)
            heading = np.array(base_cbv_state["heading"], copy=True)
            velocity = np.array(base_cbv_state["velocity"], copy=True)
            acceleration = np.array(
                base_cbv_state.get("acceleration", np.zeros_like(velocity)),
                copy=True,
            )
            valid = np.array(base_cbv_state["valid"], copy=True)

            for step_idx, st in hist.items():
                if 0 <= step_idx < len(pos):
                    pos[step_idx, :2] = st["position"][:2]
                    heading[step_idx] = st["heading"]
                    velocity[step_idx, :2] = st["velocity"][:2]
                    if "acceleration" in st:
                        acceleration[step_idx, :2] = st["acceleration"][:2]
                    valid[step_idx] = True

            track_state["position"] = pos
            track_state["heading"] = heading
            track_state["velocity"] = velocity
            track_state["acceleration"] = acceleration
            track_state["valid"] = valid

        return patched_state

    def _ego_center_to_rear_position(self, position, heading):
        d = float(self.current_state["metadata"]["ego_vehicle_parameters"].get("rear_axle_to_center", 1.67))
        pos = np.asarray(position, dtype=np.float64).copy()
        pos[:2] -= d * np.array([np.cos(heading), np.sin(heading)], dtype=np.float64)
        return pos

    @staticmethod
    def _trajectory_reference_for_engine(sim):
        reference = str(getattr(sim, "trajectory_reference", "rear_axle")).strip().lower()
        if reference not in ("rear_axle", "center"):
            raise RuntimeError("unsupported planner trajectory_reference=%s" % reference)
        return reference

    def _ego_model_reference_position(self, ego_vehicle):
        if self._trajectory_reference_for_engine(self.Sim) == "center":
            return np.asarray(ego_vehicle.position, dtype=np.float64).copy()
        return self._ego_center_to_rear_position(
            ego_vehicle.position,
            ego_vehicle.heading_theta,
        )

    def _convert_ego_model_traj_xy_to_center(self, traj, default_heading=0.0):
        if self._trajectory_reference_for_engine(self.Sim) == "center":
            return np.asarray(traj[:, :2], dtype=np.float64)
        return self._convert_ego_rear_traj_xy_to_center(
            traj,
            default_heading=default_heading,
        )

    def _convert_adv_model_traj_xy_to_center(self, obj, traj, default_heading=0.0):
        if self._trajectory_reference_for_engine(self.AdvSim) == "center":
            return np.asarray(traj[:, :2], dtype=np.float64)
        return self._convert_rear_axle_traj_xy_to_center(
            obj,
            traj,
            default_heading=default_heading,
        )

    def _ego_rear_to_center_distance(self) -> float:
        return float(self.current_state["metadata"]["ego_vehicle_parameters"].get("rear_axle_to_center", 1.67))

    def _convert_ego_rear_traj_xy_to_center(self, traj: np.ndarray, default_heading: float = 0.0) -> np.ndarray:
        """Convert ego prediction xy from rear axle to center for rendering only."""
        if traj is None or len(traj) == 0:
            return np.zeros((0, 2), dtype=np.float64)

        xy = np.asarray(traj[:, :2], dtype=np.float64)
        d = self._ego_rear_to_center_distance()
        if d <= 1e-6:
            return xy

        heading = self._extract_traj_heading(traj, default_heading=default_heading)
        if heading is None:
            heading = np.full((len(traj),), float(default_heading), dtype=np.float64)
        return xy + np.stack([np.cos(heading), np.sin(heading)], axis=-1) * d

    def _convert_ego_candidate_trajs_xy_to_center(
            self,
            candidate_trajs: np.ndarray,
            default_heading: float = 0.0,
    ) -> np.ndarray:
        """Convert ego candidate trajectories from rear-axle xy to center xy for rendering."""
        candidate_trajs = np.asarray(candidate_trajs)
        if candidate_trajs.ndim != 3 or candidate_trajs.shape[0] == 0:
            return candidate_trajs

        converted = np.array(candidate_trajs, copy=True)
        for idx in range(converted.shape[0]):
            center_xy = self._convert_ego_rear_traj_xy_to_center(
                converted[idx],
                default_heading=default_heading,
            )
            converted[idx, :, :2] = center_xy[:, :2]
        return converted

    def _history_acceleration(self, history, time_index, velocity, seeded_entry=None):
        """Use seeded/actual acceleration first, otherwise deterministic finite difference."""
        if isinstance(seeded_entry, dict) and "acceleration" in seeded_entry:
            return np.asarray(seeded_entry["acceleration"], dtype=np.float64)[:2].copy()
        previous = history.get(int(time_index) - 1)
        if isinstance(previous, dict) and "velocity" in previous:
            return (
                np.asarray(velocity, dtype=np.float64)[:2]
                - np.asarray(previous["velocity"], dtype=np.float64)[:2]
            ) / max(FIXED_DT, 1e-6)
        return np.zeros(2, dtype=np.float64)

    def get_hist_buffer(self, time_index, state):
        current_pos, current_heading, current_speed = state
        velocity_2d = np.array([
            current_speed * np.cos(current_heading),
            current_speed * np.sin(current_heading)
        ], dtype=np.float64)
        seeded_entry = self.hist_buffer.get(int(time_index))
        acceleration_2d = self._history_acceleration(
            self.hist_buffer,
            time_index,
            velocity_2d,
            seeded_entry=seeded_entry,
        )
        self.hist_buffer[int(time_index)] = {
            "position": np.array(current_pos, dtype=np.float64).copy(),
            "heading": float(current_heading),
            "velocity": velocity_2d,
            "acceleration": acceleration_2d,
            "valid": True,
        }

    def get_agent_hist_buffer(self, time_index):
        """Record actual planner-controlled motion, or time-aligned warmup replay.

        ScenarioTrafficManager applies replay after ego policy.act(), so reading the
        object here is one frame stale at the warmup/planning boundary. The track is
        the warmup source; once physics tracking starts, actual motion is used.
        """
        scene = self.engine.data_manager.current_scenario
        tracks = scene.get("tracks", {}) if isinstance(scene, dict) else {}

        for sid in list(self._controlled_cbv_scenario_ids):
            sid = str(sid)
            if sid in self.external_history_owned_ids:
                continue
            track = tracks.get(sid)
            if track is None:
                continue

            object_id = self._get_scenario_to_object_id_map().get(sid)
            obj = self.engine.get_objects().get(object_id) if object_id is not None else None
            policy = self.engine.get_policy(object_id) if obj is not None else None
            if getattr(policy, "physics_tracking_active", False) and obj is not None:
                # Planner/physics owns the vehicle after handoff.  Do not require a
                # still-valid dataset frame before recording the actually executed
                # pose; that would drop history as soon as the recording ends.
                yaw = float(obj.heading_theta)
                state = {
                    "position": np.asarray(obj.position).copy(),
                    "heading": yaw,
                    "velocity": np.asarray(obj.velocity).copy(),
                    "valid": True,
                }
            else:
                state = parse_object_state(track, int(time_index))
                if state is None or not bool(state.get("valid", False)):
                    continue

            velocity_2d = np.asarray(state.get("velocity", [0.0, 0.0]), dtype=np.float64)[:2]
            history = self.agent_hist_buffer.setdefault(sid, {})
            seeded_entry = history.get(int(time_index))
            if "acceleration" in state:
                acceleration_2d = np.asarray(state["acceleration"], dtype=np.float64)[:2]
            else:
                acceleration_2d = self._history_acceleration(
                    history,
                    time_index,
                    velocity_2d,
                    seeded_entry=seeded_entry,
                )
            history[int(time_index)] = {
                "position": np.asarray(state["position"], dtype=np.float64)[:2].copy(),
                "heading": float(state["heading"]),
                "velocity": velocity_2d.copy(),
                "acceleration": acceleration_2d.copy(),
                "valid": True,
            }

    def get_match_idx(self, pred_traj, distances, current_pos, current_heading):
        match_idx, _ = self._compute_match_idx(
            pred_traj,
            distances,
            current_pos,
            current_heading,
            last_match_idx=self.last_match_idx,
            allow_stateful_smoothing=True,
        )
        # print(f'[MatchIdx] match_idx={match_idx}, last_match_idx={self.last_match_idx}')
        self.last_match_idx = match_idx
        return match_idx

    def _compute_match_idx(
            self,
            pred_traj,
            distances,
            current_pos,
            current_heading,
            last_match_idx=0,
            allow_stateful_smoothing=True,
    ):
        # 当前车头方向向量
        heading_vec = np.array([np.cos(current_heading), np.sin(current_heading)])
        vec_to_points = pred_traj[:, :2] - current_pos

        # 只保留车辆前方点（夹角 < 90°）
        dot = np.dot(vec_to_points, heading_vec)
        forward_mask = dot > 0

        # 再在前方点中找最近的点
        if forward_mask.any():
            match_idx = np.argmin(distances * forward_mask + (~forward_mask) * 1e6)
        else:
            match_idx = np.argmin(distances)  # 极端情况备用

        candidate_idx = int(match_idx)
        candidate_point = pred_traj[candidate_idx, :2]
        candidate_vec = candidate_point - current_pos
        nearest_dist = float(np.linalg.norm(candidate_vec))

        tangent_idx = min(candidate_idx + 3, len(pred_traj) - 1)
        tangent = pred_traj[tangent_idx, :2] - candidate_point
        if np.linalg.norm(tangent) < 1e-3:
            tangent = heading_vec
        cross_track = abs(self._signed_cross_track_error(current_pos, candidate_point, tangent))

        allow_rollback = (
                nearest_dist > MATCH_ROLLBACK_DIST_M
                or cross_track > MATCH_ROLLBACK_CROSS_TRACK_M
        )

        # 平滑更新
        if allow_stateful_smoothing and candidate_idx < last_match_idx and not allow_rollback:
            match_idx = last_match_idx
        else:
            match_idx = candidate_idx
        return match_idx, {
            "nearest_dist": nearest_dist,
            "cross_track": cross_track,
            "allow_rollback": allow_rollback,
        }

    def _select_cbv_patch_offset(self, adv_obj, adv_traj):
        """Find the most continuous future point in a fresh CBV plan.

        CBV trajectories are written directly into replay state, so the chosen
        patch start must align with the object's current physical state.
        """
        if adv_traj is None or len(adv_traj) == 0:
            return 0

        current_pos = np.asarray(adv_obj.position, dtype=np.float64)[:2]
        current_heading = float(adv_obj.heading_theta)

        # Convert predicted (rear axle) to CENTER before matching
        match_traj = self._convert_adv_model_traj_xy_to_center(
            adv_obj,
            adv_traj,
            default_heading=current_heading,
        )

        distances = np.linalg.norm(match_traj - current_pos[None, :], axis=1)
        match_idx, match_info = self._compute_match_idx(
            match_traj,
            distances,
            current_pos,
            current_heading,
            last_match_idx=0,
            allow_stateful_smoothing=False,
        )

        # Never jump backward to a point behind the current vehicle. If the
        # closest continuous point is still the trajectory origin, compare the
        # first few points and pick the closest forward-progressive point.
        patch_offset = int(np.clip(match_idx, 0, len(adv_traj) - 1))
        if patch_offset == 0 and len(adv_traj) > 1:
            prefix = min(3, len(adv_traj))
            prefix_dists = distances[:prefix]
            patch_offset = int(np.argmin(prefix_dists))

        return patch_offset, match_info

    def _rear_axle_to_center_distance(self, obj) -> float:
        """Heuristic rear-axle->center distance used by PlutoDataset for non-ego vehicles.

        PlutoDataset.preprocess uses: rear_axle_to_center = 0.35 * length.
        Here we apply the same for MetaDrive vehicles when converting predicted CBV
        trajectories (rear-axle) back into scenario track coordinates (center).
        """
        length_val = getattr(obj, "top_down_length", None)
        if length_val is None:
            length_val = getattr(obj, "LENGTH", None)

        if length_val is None:
            return 0.0

        try:
            length_val = float(length_val)
        except Exception:
            return 0.0

        # For pedestrians/cyclists, length is typically small; we only apply this shift for vehicle-like agents.
        if length_val < 1.0:
            return 0.0
        return 0.35 * length_val


    def _extract_traj_heading(self, traj: np.ndarray, default_heading: float = 0.0):
        """Extract heading angle from trajectory channels.

        Pluto control trajectories are typically `[x, y, cos(yaw), sin(yaw), vx, vy]`.
        Older/fallback paths may still use `[x, y, yaw]`.
        """
        if traj is None or len(traj) == 0:
            return None

        if traj.shape[1] >= 4:
            return np.arctan2(
                np.asarray(traj[:, 3], dtype=np.float64),
                np.asarray(traj[:, 2], dtype=np.float64),
            )
        if traj.shape[1] >= 3:
            return np.asarray(traj[:, 2], dtype=np.float64).reshape(-1)
        return np.full((len(traj),), float(default_heading), dtype=np.float64)

    def _extract_traj_velocity_xy(self, traj: np.ndarray):
        """Extract velocity channels if the trajectory explicitly provides them."""
        if traj is None or len(traj) == 0:
            return None
        if traj.shape[1] >= 6:
            return np.asarray(traj[:, 4:6], dtype=np.float64)
        return None

    def _convert_rear_axle_traj_xy_to_center(self, obj, traj: np.ndarray, default_heading: float = 0.0) -> np.ndarray:
        """Convert a predicted trajectory's xy from rear-axle to center coordinates.

        Args:
            obj: MetaDrive object instance (used to estimate vehicle length).
            traj: np.ndarray with shape (T, C). We use traj[:, :2] as xy and traj[:, 2] as heading if present.
            default_heading: used when trajectory doesn't contain heading.

        Returns:
            (T, 2) float64 array in center coordinates.
        """
        if traj is None or len(traj) == 0:
            return np.zeros((0, 2), dtype=np.float64)

        xy = np.asarray(traj[:, :2], dtype=np.float64)
        d = float(self._rear_axle_to_center_distance(obj))
        if d <= 1e-6:
            return xy

        heading = self._extract_traj_heading(traj, default_heading=default_heading)
        if heading is None:
            heading = np.full((len(traj),), float(default_heading), dtype=np.float64)

        offset = np.stack([np.cos(heading), np.sin(heading)], axis=-1) * d
        return xy + offset

    def _convert_candidate_trajs_xy_to_center(
            self,
            obj,
            candidate_trajs: np.ndarray,
            default_heading: float = 0.0,
    ) -> np.ndarray:
        """Convert candidate trajectories from rear-axle xy to center xy for rendering."""
        candidate_trajs = np.asarray(candidate_trajs)
        if candidate_trajs.ndim != 3 or candidate_trajs.shape[0] == 0:
            return candidate_trajs

        converted = np.array(candidate_trajs, copy=True)
        for idx in range(converted.shape[0]):
            center_xy = self._convert_rear_axle_traj_xy_to_center(
                obj,
                converted[idx],
                default_heading=default_heading,
            )
            converted[idx, :, :2] = center_xy[:, :2]
        return converted

    def save_history(self):
        self.history_logger.save()
