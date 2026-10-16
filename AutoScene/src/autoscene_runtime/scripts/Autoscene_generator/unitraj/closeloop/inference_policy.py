"""AutoScene 当前主闭环策略。

本文件运行在 MetaDrive policy 内部，不直接订阅或发布 ROS topic：
1. 使用 create_env 绑定的 effective config（standalone 时读取 config.yaml），初始化 ego/adv planner；
2. 在每个 env.step() 中维护 ego/CBV 历史；
3. 调用 Pluto/VBD 推理生成 ego 轨迹和关键周车轨迹；
4. 用 PID 把预测轨迹转成 MetaDrive 控制量，或把 CBV 轨迹写回 replay track。

ROS hybrid 中，scenario_runtime_node 会在 env.step() 前把 actual state 覆盖到
MetaDrive mirror；真实平台历史由 runtime 写入，ego 和 virtual BV 历史由 policy 写入。
"""
import inspect

import os
import time
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

# Controller gains retained from the validated generator baseline.
LATERAL_PID_GAINS = (3.0, 0.05, 0.0)
LONGITUDINAL_PID_GAINS = (50, 0, 0)
CBV_LATERAL_PID_GAINS = (0.55, 0.0, 0.35)
CBV_LONGITUDINAL_PID_GAINS = (1.5, 0.0, 0.05)
CBV_MIN_TARGET_SPEED = 2.5
CBV_LOW_SPEED_STEER_LIMIT = 0.45
FIXED_DT = 0.1  # (s) 仿真步长
MATCH_ROLLBACK_DIST_M = 2.5
MATCH_ROLLBACK_CROSS_TRACK_M = 1.5
FORWARD_DOT_MIN = 0.5


class DynamicReplayTrafficParticipantPolicy(ReplayTrafficParticipantPolicy):
    """可动态读取 track 的周车 replay/PID 策略。

    MetaDrive 原始 ReplayTrafficParticipantPolicy 会在初始化时缓存轨迹。
    AutoScene 会在闭环中不断覆盖 CBV 的未来轨迹，因此这里每一帧都从 track
    重新取数据，并用 PID 追踪该动态轨迹。
    """

    def __init__(self, control_object, track, random_seed=None):
        # NOTE: intentionally NOT calling ReplayTrafficParticipantPolicy.__init__
        # to avoid caching traj_info.
        BasePolicy.__init__(self, control_object=control_object, random_seed=random_seed)
        self.start_index = 0
        self._velocity_local_frame = False
        self.track = track
        self.lateral_pid = PIDController(*CBV_LATERAL_PID_GAINS)
        self.longitudinal_pid = PIDController(*CBV_LONGITUDINAL_PID_GAINS)
        self.last_match_idx = 0

    @property
    def is_current_step_valid(self):
        index = max(int(self.episode_step), 0)
        state = parse_object_state(self.track, index)
        return state is not None and bool(state.get("valid", False))

    def act(self, *args, **kwargs):
        index = max(int(self.episode_step), 0)
        state = parse_object_state(self.track, index)
        if state is None or not bool(state.get("valid", False)):
            return None

        action = self.compute_action(index)
        if action is None:
            return None

        steering, throttle_brake = action
        if hasattr(self.control_object, "set_steering"):
            self.control_object.set_steering(steering)
        if hasattr(self.control_object, "set_throttle_brake"):
            self.control_object.set_throttle_brake(throttle_brake)

        self.control_object.set_static(False)

        return [steering, throttle_brake]

    def compute_action(self, index=None):
        if index is None:
            index = max(int(self.episode_step), 0)
        return self._pid_track(int(index))

    def _pid_track(self, index):
        pred_traj = self._future_track(index)
        if pred_traj is None or len(pred_traj) < 2:
            return None

        current_pos = np.asarray(self.control_object.position, dtype=np.float64)[:2]
        current_heading = float(self.control_object.heading_theta)
        current_speed = float(getattr(self.control_object, "speed", 0.0))
        heading_vec = np.array([np.cos(current_heading), np.sin(current_heading)], dtype=np.float64)

        preview_distance = float(np.clip(current_speed * 0.6, 2.0, 5.0))
        preview_pos = current_pos + preview_distance * heading_vec
        match_idx = self._continuous_match_idx(pred_traj[:, :2], preview_pos, current_pos, current_heading)
        self.last_match_idx = match_idx

        lookahead_distance = float(np.clip(current_speed * 0.8, 2.0, 6.0))
        target_idx = self._find_lookahead_idx(pred_traj[:, :2], match_idx, lookahead_distance)
        target_idx = self._ensure_forward_target(pred_traj[:, :2], match_idx, target_idx, current_pos, current_heading)

        p_match = pred_traj[match_idx, :2]
        tangent_idx = min(match_idx + 3, len(pred_traj) - 1)
        traj_tangent = pred_traj[tangent_idx, :2] - p_match
        if np.linalg.norm(traj_tangent) < 1e-3 or float(np.dot(traj_tangent, heading_vec)) < 0.0:
            traj_tangent = heading_vec

        lateral_error = self._signed_cross_track_error(current_pos, p_match, traj_tangent)
        if pred_traj.shape[-1] >= 3 and np.isfinite(pred_traj[tangent_idx, 2]):
            traj_heading = float(pred_traj[tangent_idx, 2])
        else:
            traj_heading = float(np.arctan2(traj_tangent[1], traj_tangent[0]))
        heading_error = self._wrap_to_pi(traj_heading - current_heading)
        combined_error = heading_error - np.arctan2(1.5 * lateral_error, max(current_speed, 1.0))

        target_speed = self._target_speed(pred_traj, match_idx, target_idx, current_speed)
        if np.linalg.norm(pred_traj[target_idx, :2] - current_pos) > 1.5:
            target_speed = max(float(target_speed), CBV_MIN_TARGET_SPEED)
        steering = float(np.clip(self.lateral_pid.get_result(-combined_error), -1.0, 1.0))
        steer_limit = CBV_LOW_SPEED_STEER_LIMIT + (1.0 - CBV_LOW_SPEED_STEER_LIMIT) * np.clip(current_speed / 5.0, 0.0, 1.0)
        steering = float(np.clip(steering, -steer_limit, steer_limit))
        throttle_brake = float(np.clip(self.longitudinal_pid.get_result(current_speed - target_speed), -1.0, 1.0))
        return steering, throttle_brake

    def _future_track(self, index, horizon=80):
        state = self.track.get("state", {}) if isinstance(self.track, dict) else {}
        pos = state.get("position", None)
        if pos is None:
            return None
        pos = np.asarray(pos)
        if index >= len(pos):
            return None
        end = min(index + int(horizon), len(pos))
        valid = np.asarray(state.get("valid", np.ones((len(pos),), dtype=bool))).reshape(-1)
        keep = valid[index:end] if len(valid) >= end else np.ones((end - index,), dtype=bool)
        keep = np.asarray(keep).astype(bool)
        xy = np.asarray(pos[index:end, :2], dtype=np.float64)
        heading = np.asarray(state.get("heading", np.full((len(pos),), np.nan)), dtype=np.float64).reshape(-1)
        heading = heading[index:end] if len(heading) >= end else np.full((end - index,), np.nan)
        traj = np.concatenate([xy, heading[:, None]], axis=-1)
        traj = traj[keep]
        return traj if len(traj) >= 2 else None

    def _continuous_match_idx(self, traj_xy, preview_pos, current_pos, current_heading):
        n = len(traj_xy)
        if n == 0:
            return 0
        old_idx = int(np.clip(self.last_match_idx, 0, n - 1))
        search_start = max(0, old_idx - 5)
        search_end = min(n, old_idx + 15)
        local = traj_xy[search_start:search_end]
        dist = np.linalg.norm(local - preview_pos, axis=1)
        heading_vec = np.array([np.cos(current_heading), np.sin(current_heading)], dtype=np.float64)
        forward = np.dot(local - current_pos, heading_vec) > 0.0
        local_idx = int(np.argmin(np.where(forward, dist, 1e6))) if forward.any() else int(np.argmin(dist))
        return search_start + local_idx

    def _find_lookahead_idx(self, traj_xy, start_idx, lookahead_m):
        idx = int(np.clip(start_idx, 0, len(traj_xy) - 1))
        acc = 0.0
        while idx < len(traj_xy) - 1 and acc < lookahead_m:
            acc += float(np.linalg.norm(traj_xy[idx + 1] - traj_xy[idx]))
            idx += 1
        return idx

    def _ensure_forward_target(self, traj_xy, match_idx, target_idx, current_pos, current_heading):
        heading_vec = np.array([np.cos(current_heading), np.sin(current_heading)], dtype=np.float64)
        idx = int(np.clip(target_idx, match_idx, len(traj_xy) - 1))
        while idx < len(traj_xy) - 1:
            if float(np.dot(traj_xy[idx] - current_pos, heading_vec)) > FORWARD_DOT_MIN:
                return idx
            idx += 1
        return idx

    def _target_speed(self, pred_traj, match_idx, target_idx, current_speed):
        if target_idx > match_idx:
            segment = pred_traj[match_idx:target_idx + 1, :2]
            if len(segment) > 1:
                return float(np.linalg.norm(np.diff(segment, axis=0), axis=1).sum() / max((target_idx - match_idx) * FIXED_DT, 1e-6))
        if len(pred_traj) > 1:
            idx = min(match_idx + 5, len(pred_traj) - 1)
            steps = max(idx - match_idx, 1)
            return float(np.linalg.norm(pred_traj[idx, :2] - pred_traj[match_idx, :2]) / (steps * FIXED_DT))
        return float(current_speed)

    @staticmethod
    def _signed_cross_track_error(point, path_point, path_tangent):
        tangent = np.asarray(path_tangent, dtype=np.float64)
        norm = float(np.linalg.norm(tangent))
        if norm < 1e-6:
            return 0.0
        tangent = tangent / norm
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
        return float(np.dot(np.asarray(point, dtype=np.float64) - np.asarray(path_point, dtype=np.float64), normal))

    @staticmethod
    def _wrap_to_pi(angle):
        return float((angle + np.pi) % (2 * np.pi) - np.pi)


class InferencePolicy(EnvInputPolicy):
    """Ego/CBV 闭环推理策略。

    运行阶段：
    - warmup：dataset 模式前 20 帧回放真实历史；
    - plan：按 replan_frequency 调用 Pluto/VBD 生成 ego 和 CBV 轨迹；
    - control：ego 用 PID 跟踪规划轨迹，CBV 用动态 replay/PID 跟踪对抗轨迹；
    - export：把 top_k_ids、候选轨迹、控制调试信息暴露给 renderer/runtime。
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
        self.replan_frequency = 5
        self.current_cbv_ids = []  # 单场景锁定的多辆关键周车
        self.max_cbv_num = self.base_cfg.get('adv_num', 2)
        self.cbv_select_min_longitudinal_m = float(self.base_cfg.get('cbv_select_min_longitudinal_m', -8.0))
        self.cbv_select_max_longitudinal_m = float(self.base_cfg.get('cbv_select_max_longitudinal_m', 45.0))
        self.cbv_select_max_lateral_m = float(self.base_cfg.get('cbv_select_max_lateral_m', 8.0))
        self.cbv_select_max_heading_diff_rad = float(self.base_cfg.get('cbv_select_max_heading_diff_rad', 1.35))
        self.cbv_select_max_current_dist_m = float(self.base_cfg.get('cbv_select_max_current_dist_m', 55.0))
        self.cbv_select_max_future_min_dist_m = float(self.base_cfg.get('cbv_select_max_future_min_dist_m', 30.0))
        self.cbv_select_allow_opposite = bool(self.base_cfg.get('cbv_select_allow_opposite', True))
        self.cbv_select_allow_crossing = bool(self.base_cfg.get('cbv_select_allow_crossing', True))
        self.cbv_select_opposite_min_heading_diff_rad = float(
            self.base_cfg.get('cbv_select_opposite_min_heading_diff_rad', 2.2)
        )
        self.cbv_select_opposite_max_lateral_m = float(
            self.base_cfg.get('cbv_select_opposite_max_lateral_m', 4.5)
        )
        self.cbv_select_opposite_min_longitudinal_m = float(
            self.base_cfg.get('cbv_select_opposite_min_longitudinal_m', -2.0)
        )
        self.cbv_select_opposite_max_longitudinal_m = float(
            self.base_cfg.get('cbv_select_opposite_max_longitudinal_m', 35.0)
        )
        self.cbv_select_opposite_max_future_min_dist_m = float(
            self.base_cfg.get('cbv_select_opposite_max_future_min_dist_m', 10.0)
        )
        self.cbv_select_crossing_min_heading_diff_rad = float(
            self.base_cfg.get('cbv_select_crossing_min_heading_diff_rad', 0.9)
        )
        self.cbv_select_crossing_max_heading_diff_rad = float(
            self.base_cfg.get('cbv_select_crossing_max_heading_diff_rad', 2.2)
        )
        self.cbv_select_crossing_max_lateral_m = float(
            self.base_cfg.get('cbv_select_crossing_max_lateral_m', 20.0)
        )
        self.cbv_select_crossing_min_longitudinal_m = float(
            self.base_cfg.get('cbv_select_crossing_min_longitudinal_m', -12.0)
        )
        self.cbv_select_crossing_max_longitudinal_m = float(
            self.base_cfg.get('cbv_select_crossing_max_longitudinal_m', 45.0)
        )
        self.cbv_select_crossing_max_future_min_dist_m = float(
            self.base_cfg.get('cbv_select_crossing_max_future_min_dist_m', 12.0)
        )
        self.cbv_selection_locked = False
        self.allow_pedestrian_cbv = self.base_cfg.get('allow_pedestrian_cbv', False)
        self.allow_cyclist_cbv = self.base_cfg.get('allow_cyclist_cbv', False)

        self.ego_method = self.base_cfg.get(
            'ego_method',
            self.base_cfg.defaults[0]['method'],
        )
        self.ego_cfg = self.get_final_config(self.ego_method)
        print(f"[Policy] Ego Planner: {self.ego_method}")

        # _plan_trajectory() 实际读取 self.Sim，因此必须初始化同名成员。
        self.Sim = self._init_inference_engine(self.ego_cfg)

        self.use_adv = self.base_cfg.adv_control
        if self.use_adv:
            self.adv_method = self.base_cfg.get('adv_method', self.base_cfg.defaults[0]['method'])
            self.adv_cfg = self.get_final_config(self.adv_method)
            self.AdvSim = self._init_inference_engine(self.adv_cfg)
        else:
            self.AdvSim = None
        print(f"[Policy] Adv Planner: {self.adv_method if self.use_adv else 'None'}")

        # 初始化横向和纵向PID控制器
        self.lateral_pid = PIDController(*LATERAL_PID_GAINS)
        self.longitudinal_pid = PIDController(*LONGITUDINAL_PID_GAINS)
        self.current_state = self.engine.data_manager.current_scenario

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
        self.current_adv_candidates = {}
        self.adv_agent_ids = []
        self.cbv_score_selection_by_id = {}
        self.planner_mode_by_scenario_id = {}
        self.platform_output_point_num = 50
        self.info = None
        # Pending rollout payloads to be exposed via MetaDrive env.step() info.
        # We must NOT keep them in action_info across frames, otherwise RL collector
        # will see duplicates.
        self._pending_cbv_rollout_samples = []

        # For keeping inference-state consistent with the actually controlled CBV
        self._controlled_cbv_scenario_ids = set()
        # IDs whose history is written by scenario_runtime from actual feedback.
        self.external_history_owned_ids = set()
        self.external_ego_control = False


    def get_final_config(self, method_name):
        """Merge base config with the selected planner method config.

        pluto_hybrid deliberately reuses the existing Pluto method/checkpoint
        config and overlays only hybrid-candidate settings. Existing Pluto/VBD
        method loading remains byte-for-byte equivalent in behavior.
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
        """接收环境变量传入的运行时覆盖项，供 ROS runtime 控制对抗开关。"""
        if "UNITRAJ_RL_FINETUNING_MODE" in os.environ:
            cfg.rl_finetuning_mode = os.environ["UNITRAJ_RL_FINETUNING_MODE"] in ("1", "true", "True")
        if "UNITRAJ_ADV_CONTROL" in os.environ:
            cfg.adv_control = os.environ["UNITRAJ_ADV_CONTROL"] in ("1", "true", "True")
        if "UNITRAJ_ADV_NUM" in os.environ:
            cfg.adv_num = int(os.environ["UNITRAJ_ADV_NUM"])
        return cfg

    def _init_inference_engine(self, specific_cfg):
        """初始化精简后保留下来的模型推理引擎。"""
        method_name = specific_cfg.get('model_name')
        planner_variant = str(specific_cfg.get('planner_variant', '')).strip().lower()
        if method_name == 'GT':
            return None

        if planner_variant == 'pluto_hybrid':
            from unitraj.closeloop.pluto_hybrid.pluto_hybrid_inference import PlutoHybridInference
            sim = PlutoHybridInference(specific_cfg)
        elif method_name == 'VBD':
            sim = VBDInference(specific_cfg)
        elif method_name == 'Pluto':
            sim = PlutoInference(specific_cfg)
        else:
            raise ValueError(
                f"Unsupported method after Autoscene_generator cleanup: {method_name}. "
                "Only VBD and Pluto are kept."
            )

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
        """兼容不同推理引擎的 run_inference 参数签名。"""
        if sim is None:
            return None

        run_inference = sim.run_inference
        params = inspect.signature(run_inference).parameters
        kwargs = {}
        if controlled_agent_id is not None:
            if "controlled_agent_id" not in params:
                return None
            kwargs["controlled_agent_id"] = controlled_agent_id
        if score_selection is not None and "score_selection" in params:
            kwargs["score_selection"] = score_selection
        if adversarial_target_agent_ids is not None and "adversarial_target_agent_ids" in params:
            kwargs["adversarial_target_agent_ids"] = adversarial_target_agent_ids
        if required_agent_ids is not None and "required_agent_ids" in params:
            kwargs["required_agent_ids"] = [str(agent_id) for agent_id in required_agent_ids]
        return run_inference(scenario_state, time_index, **kwargs)

    def _normalize_inference_output(self, infer_out):
        """统一 planner 输出格式为 `(pred_traj, ref_lines, candidates, info)`。"""
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

    def _clear_render_overlays(self):
        """Remove planning/debug overlays that can otherwise survive env.reset()."""
        if hasattr(self.engine, "head_renderer") and self.engine.head_renderer is not None:
            if hasattr(self.engine.head_renderer, "clear_overlays"):
                self.engine.head_renderer.clear_overlays()
            else:
                if hasattr(self.engine.head_renderer, "set_reference_lines"):
                    self.engine.head_renderer.set_reference_lines(None)
                if hasattr(self.engine.head_renderer, "set_av2_reference_lines"):
                    self.engine.head_renderer.set_av2_reference_lines(None)

        top_down_renderer = getattr(getattr(self.engine, "head_renderer", None), "head_top_down_renderer", None)
        if top_down_renderer is not None:
            top_down_renderer.cbv_id = []
            top_down_renderer.max_score_cbv_id = []

        objects = {}
        if hasattr(self.engine, "get_objects"):
            try:
                objects.update(self.engine.get_objects())
            except Exception:
                pass
        if hasattr(self.engine, "agents"):
            objects.update(getattr(self.engine, "agents", {}))

        default_agent = getattr(self.engine, "agents", {}).get("default_agent", None) if hasattr(self.engine, "agents") else None
        for obj in objects.values():
            if obj is None:
                continue
            attrs = [
                "plan_traj",
                "pre_plan_traj",
                "safe_plan_traj",
                "candidate_trajectories",
                "key_points",
            ]
            if obj is not default_agent:
                attrs.append("expert_traj")
            for attr in attrs:
                if hasattr(obj, attr):
                    try:
                        delattr(obj, attr)
                    except Exception:
                        pass

    def act(self, agent_id: str):
        # MetaDrive 会在 act() 后立即读取 action_info；
        # 需要暴露给 env.step() info 的内容必须在本函数内写入。
        self.action_info.clear()

        current_step = self.engine.episode_step
        time_index = current_step - 1

        if current_step <= 1:
            self._clear_render_overlays()
        sdc_track = self.current_state["tracks"][self.sdc_id]

        # dataset 模式回放原始历史；initial 模式由 runtime 在 reset 后把
        # ego/选定 BV 的前 21 帧 track 改成启动历史，因此这里仍按同一时间轴读取。
        if time_index < self.WARMUP_STEPS:
            if not self.external_ego_control:
                self._handle_warmup(time_index, sdc_track)
            ego_vehicle = self.control_object
            self.get_hist_buffer(
                time_index,
                [
                    ego_vehicle.position,
                    ego_vehicle.heading_theta,
                    ego_vehicle.speed,
                ],
            )
            return None

        # 记录每帧闭环后的真实执行状态。
        # 在 ROS hybrid 中，runtime 会在 env.step() 前先把 actual feedback 覆盖到 mirror，
        # 因此这里采到的是融合后的闭环状态。
        ego_vehicle = self.control_object
        self.get_hist_buffer(time_index, [ego_vehicle.position, ego_vehicle.heading_theta, ego_vehicle.speed])
        if self.use_adv:
            self.get_agent_hist_buffer(time_index)

        # --- 阶段二: 闭环规划 ---
        if time_index % self.replan_frequency == 0:
            self._plan_trajectory(time_index)

        # --- 阶段三: 主车闭环控制 (PID跟踪) ---
        ego_control_start = time.perf_counter()
        ego_control_value = self._compute_control()
        ego_control_seconds = time.perf_counter() - ego_control_start

        # 导出给 env.step() info，供 runtime/renderer/rollout 观察。
        if self._pending_cbv_rollout_samples:
            self.action_info["cbv_rollout_samples"] = self._pending_cbv_rollout_samples
            self._pending_cbv_rollout_samples = []

        if isinstance(self.info, dict):
            if "top_k_ids" in self.info:
                self.action_info["top_k_ids"] = self.info.get("top_k_ids")
            if self.info.get("terminate_scene", False):
                self.action_info["terminate_scene"] = True
                self.action_info["terminate_reason"] = self.info.get("terminate_reason", "unknown")
                self._clear_render_overlays()
            if "latent" in self.info:
                self.action_info["ego_latent"] = self.info.get("latent")
            if "selected_future_traj" in self.info:
                self.action_info["ego_future_traj"] = self.info.get("selected_future_traj")

        self.action_info["action"] = ego_control_value

        # --- 阶段四: 周车控制 (Adv Control) ---
        cbv_control_seconds = 0.0
        if self.use_adv:
            cbv_control_start = time.perf_counter()
            self._apply_adv_control()
            cbv_control_seconds = time.perf_counter() - cbv_control_start
        if os.environ.get("UNITRAJ_PROFILE_TIMING", "0") == "1":
            print(
                "[Profile]",
                {
                    "stage": "control_execution",
                    "ego_ms": round(ego_control_seconds * 1000.0, 3),
                    "cbv_ms": round(cbv_control_seconds * 1000.0, 3),
                    "total_ms": round(
                        (ego_control_seconds + cbv_control_seconds) * 1000.0, 3
                    ),
                },
            )
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
        """按设定频率进行 ego 轨迹和 CBV 轨迹推理。"""
        self.current_state = self.update_state(time_index)

        # ================= [1. 主车推理 ] =================
        infer_out = self._call_inference_engine(
            self.Sim,
            self.current_state,
            time_index,
            required_agent_ids=(self.current_cbv_ids if self.use_adv else None),
        )
        self._pred_traj, ref_lines, candidates, self.info = self._normalize_inference_output(infer_out)
        # ego 推理必须返回至少一条包含两个轨迹点的二维位置轨迹。
        # 输出为空或形状错误时立即报错，避免错误延迟到 PID 控制阶段。
        if self._pred_traj is None:
            raise RuntimeError(
                "ego inference returned no trajectory "
                "at time_index=%d" % time_index
            )

        if (
                self._pred_traj.ndim != 3
                or self._pred_traj.shape[0] < 1
                or self._pred_traj.shape[1] < 2
                or self._pred_traj.shape[2] < 2
        ):
            raise RuntimeError(
                "invalid ego trajectory shape "
                "at time_index=%d: %s"
                % (
                    time_index,
                    tuple(self._pred_traj.shape),
                )
            )
        self.engine.agents['default_agent'].plan_traj = self._pred_traj[0, :, :2]

        if hasattr(self.engine, "head_renderer") and self.engine.head_renderer is not None:
            self.engine.head_renderer.set_reference_lines(ref_lines)

        if candidates is not None:
            self.engine.agents['default_agent'].candidate_trajectories = candidates

        if self.info is None:
            self.info = {}

        ego_vehicle = self.control_object
        current_pos_rear = self._ego_model_reference_position(ego_vehicle)
        current_heading = ego_vehicle.heading_theta
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


    def _select_cbv(self, time_index):
        """按 ego-agent 风险排序选择 CBV。

        评分综合未来最小距离、TTC、轨迹重叠比例和当前位置前向性；
        选中后在单场景内锁定，避免每帧换车造成轨迹和平台绑定跳变。
        """
        self.info['top_k_ids'] = []
        if 'prediction' not in self.info or 'agent_tokens' not in self.info or self._pred_traj is None:
            return

        tokens = self.info['agent_tokens']

        if self.cbv_selection_locked:
            if (
                self.current_cbv_ids is not None
                and all(cbv_id in self.current_state['tracks'] for cbv_id in self.current_cbv_ids)
            ):
                self.info['top_k_ids'] = self.current_cbv_ids
            return

        predictions = self.info['prediction']
        if predictions is None or len(predictions) == 0 or len(tokens) <= 1:
            return

        ego_future = self._pred_traj[0, :, :2]
        ego_now = np.asarray(
            self.current_state['tracks'][self.sdc_id]['state']['position'][time_index][:2],
            dtype=np.float64,
        )
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

            cbv_id = self._resolve_track_id(tokens[token_idx], self.current_state['tracks'].keys())
            if cbv_id is None or cbv_id not in self.current_state['tracks']:
                continue

            track_type = self.current_state['tracks'][cbv_id].get('type', '')
            is_pedestrian = ('PEDESTRIAN' in str(track_type).upper()) or (track_type == 1)
            is_cyclist = ('CYCLIST' in str(track_type).upper()) or (track_type == 2)
            if is_pedestrian and not self.allow_pedestrian_cbv:
                continue
            if is_cyclist and not self.allow_cyclist_cbv:
                continue

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

            cbv_now = np.asarray(
                self.current_state['tracks'][cbv_id]['state']['position'][time_index][:2],
                dtype=np.float64,
            )
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
                "agent_id": cbv_id,
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
        self.info['three_vehicle_pair_debug'] = []
        if best_entry is None:
            return

        sorted_entries = sorted(debug_entries, key=lambda x: x['score'], reverse=True)
        self.current_cbv_ids = [
            entry['agent_id']
            for entry in sorted_entries[:self.max_cbv_num]
        ]
        self.cbv_selection_locked = True
        self.info['top_k_ids'] = self.current_cbv_ids


    def _run_adv_inference(self, time_index):
        """处理周车推理，按 controlled_agent_id 为关键周车构造 Pluto 特征。"""
        dynamic_adv_ids = self.info.get('top_k_ids', [])
        if dynamic_adv_ids:
            target_ids = dynamic_adv_ids
        else:
            target_ids = []
        if not target_ids:
            print("[Policy] Warning: target_ids is empty, AdvSim will not run.")
            if isinstance(self.info, dict):
                self.info["terminate_scene"] = True
                self.info["terminate_reason"] = "empty_cbv_target_ids"
            return

        all_cbv_trajs = []
        all_cbv_ids = []
        all_cbv_candidates = {}
        self.cbv_score_selection_by_id = {}
        if hasattr(self.engine, "head_renderer") and self.engine.head_renderer is not None:
            if hasattr(self.engine.head_renderer, "set_av2_reference_lines"):
                self.engine.head_renderer.set_av2_reference_lines(None)

        real_adv_ids = []
        score_selections = {}
        adversarial_targets_by_id = {}
        for adv_rank, adv_id_str in enumerate(target_ids):
            real_adv_id = self._resolve_track_id(
                adv_id_str,
                self.current_state['tracks'].keys(),
            )
            if not real_adv_id or real_adv_id not in self.current_state['tracks']:
                raise RuntimeError(
                    "cannot resolve CBV track id %s at time_index=%d"
                    % (adv_id_str, time_index)
                )

            planner_mode = str(
                self.planner_mode_by_scenario_id.get(
                    str(real_adv_id),
                    "max",
                )
            ).strip().lower()
            if planner_mode not in ("adversarial", "max"):
                raise ValueError(
                    "invalid planner_mode=%s for CBV %s"
                    % (planner_mode, real_adv_id)
                )
            is_adversarial = planner_mode == "adversarial"
            score_selection = planner_mode
            real_adv_ids.append(str(real_adv_id))
            score_selections[str(real_adv_id)] = score_selection
            if is_adversarial:
                adversarial_targets_by_id[str(real_adv_id)] = [
                    str(target_id)
                    for target_rank, target_id in enumerate(target_ids)
                    if target_rank != adv_rank
                ]
                adversarial_targets_by_id[str(real_adv_id)].append(str(self.sdc_id))

        if hasattr(self.AdvSim, "run_inference_batch"):
            # Pluto 新路径支持一次性推理多个 CBV，可减少重复特征构造。
            adv_outputs = self.AdvSim.run_inference_batch(
                self.current_state,
                time_index,
                real_adv_ids,
                score_selections=score_selections,
                adversarial_target_agent_ids_by_id=adversarial_targets_by_id,
                **(self._planner_interaction_kwargs() if hasattr(self, "_planner_interaction_kwargs") else {}),
            )
        else:
            # 兼容只支持单车 run_inference 的旧推理引擎。
            adv_outputs = {
                real_adv_id: self._call_inference_engine(
                    self.AdvSim,
                    self.current_state,
                    time_index,
                    controlled_agent_id=real_adv_id,
                    score_selection=score_selections[real_adv_id],
                    adversarial_target_agent_ids=adversarial_targets_by_id.get(real_adv_id),
                )
                for real_adv_id in real_adv_ids
            }

        for real_adv_id in real_adv_ids:
            score_selection = score_selections[real_adv_id]
            adv_infer_out = adv_outputs.get(real_adv_id)
            adv_traj_all, _, adv_candidates, adv_info = (
                self._normalize_inference_output(adv_infer_out)
            )
            if adv_traj_all is None:
                raise RuntimeError(
                    "CBV inference returned no trajectory for %s "
                    "at time_index=%d"
                    % (real_adv_id, time_index)
                )
            if not isinstance(adv_info, dict):
                raise RuntimeError(
                    "CBV inference returned invalid info for %s "
                    "at time_index=%d: %s"
                    % (real_adv_id, time_index, type(adv_info).__name__)
                )

            score_selection = str(
                adv_info.get("score_selection", score_selection)
            ).lower()
            self.cbv_score_selection_by_id[str(real_adv_id)] = score_selection
            if score_selection == "adversarial":
                av2_ref_lines = adv_info.get("adversarial_reference_lines", None)
                if av2_ref_lines is not None:
                    if hasattr(self.engine, "head_renderer") and self.engine.head_renderer is not None:
                        if hasattr(self.engine.head_renderer, "set_av2_reference_lines"):
                            self.engine.head_renderer.set_av2_reference_lines(av2_ref_lines)

            adv_traj_all = np.asarray(adv_traj_all)
            if adv_traj_all.ndim == 3:
                # (candidate, time, feature)
                selected_traj = adv_traj_all[0]
            elif adv_traj_all.ndim == 4:
                # (batch, candidate, time, feature)
                selected_traj = adv_traj_all[0, 0]
            else:
                raise RuntimeError(
                    "invalid CBV trajectory rank for %s "
                    "at time_index=%d: shape=%s"
                    % (
                        real_adv_id,
                        time_index,
                        tuple(adv_traj_all.shape),
                    )
                )

            if (
                    selected_traj.ndim != 2
                    or selected_traj.shape[0] < 2
                    or selected_traj.shape[1] < 2
            ):
                raise RuntimeError(
                    "invalid CBV trajectory shape for %s "
                    "at time_index=%d: shape=%s"
                    % (
                        real_adv_id,
                        time_index,
                        tuple(selected_traj.shape),
                    )
                )

            all_cbv_trajs.append(selected_traj)
            all_cbv_ids.append(real_adv_id)
            if score_selection == "adversarial" and adv_candidates is not None:
                all_cbv_candidates[str(real_adv_id)] = np.asarray(adv_candidates)

            rl_finetuning_enabled = bool(
                self.adv_cfg.get("rl_finetuning_mode", self.base_cfg.get("rl_finetuning_mode", False))
            )
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

        if len(all_cbv_trajs) != len(real_adv_ids):
            raise RuntimeError(
                "CBV inference result count mismatch at time_index=%d: "
                "expected=%d actual=%d"
                % (
                    time_index,
                    len(real_adv_ids),
                    len(all_cbv_trajs),
                )
            )

        if all_cbv_trajs:
            self.current_adv_trajs = np.stack(all_cbv_trajs)
            self.current_adv_candidates = all_cbv_candidates
            self.adv_agent_ids = all_cbv_ids

    def _apply_adv_control(self):
        """将周车预测轨迹应用到 MetaDrive 引擎中。

        对虚拟 BV：轨迹直接驱动 MetaDrive 对象。
        对实物 hybrid 的受控 BV：这里仍写入 mirror/规划轨迹，但最终执行状态会被
        platform_comm_node 的真实反馈覆盖。
        """
        if self.current_adv_trajs is None or self.info is None:
            return

        self.current_aligned_adv_trajs = {}
        self.current_adv_patch_info = {}

        # 注意：
        # - act() 中 time_index = engine.episode_step - 1；
        # - MetaDrive 的 replay traffic 在 ScenarioTrafficManager.after_step() 中应用；
        # - after_step() 使用本次 env.step() 的 policy.episode_step。
        # 因此轨迹 patch 应从 engine.episode_step 开始，避免错一帧。
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

        top_down_renderer = getattr(getattr(self.engine, "head_renderer", None), "head_top_down_renderer", None)
        if top_down_renderer is not None:
            top_down_renderer.cbv_id = []
            top_down_renderer.max_score_cbv_id = []

        # 抹除已经过去（不再是 CBV）的对抗车轨迹（排除主车）
        for oid, scene_id in self.engine.traffic_manager.obj_id_to_scenario_id.items():
            if str(scene_id) != self.sdc_id and str(scene_id) not in target_scene_ids:
                obj = self.engine.get_objects().get(oid)
                if obj is not None:
                    if hasattr(obj, 'plan_traj'):
                        delattr(obj, 'plan_traj')
                    if hasattr(obj, 'expert_traj'):
                        delattr(obj, 'expert_traj')
                    if hasattr(obj, 'candidate_trajectories'):
                        delattr(obj, 'candidate_trajectories')

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
            adv_obj.set_static(False)

            trajectory_reference = self._trajectory_reference_for_engine(self.AdvSim)
            if trajectory_reference == "center":
                source_traj = np.asarray(adv_traj)
                patch_offset = 0
                patch_match_info = {}
            else:
                source_traj, patch_offset, patch_match_info = (
                    self.align_cbv_control_trajectory(
                        adv_obj,
                        adv_traj,
                    )
                )
            if len(source_traj) == 0:
                continue

            self.current_aligned_adv_trajs[str(resolved_adv_id)] = (
                np.asarray(source_traj).copy()
            )
            self.current_adv_patch_info[str(resolved_adv_id)] = {
                "patch_offset": int(patch_offset),
                "match_info": dict(patch_match_info or {}),
            }

            # Pluto 控制轨迹按后轴参考点输出；MetaDrive track 和平台状态使用车辆中心。
            # 因此这里只做一次“后轴 -> 中心”转换，后续内部执行和 ROS 导出共用同一裁剪逻辑。
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
            if self.cbv_score_selection_by_id.get(str(resolved_adv_id)) == "adversarial":
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
            if top_down_renderer is not None:
                top_down_renderer.cbv_id.append(adv_obj_id)  # 默认 CBV 设置为红色
                if self.cbv_score_selection_by_id.get(str(resolved_adv_id)) == "adversarial":
                    top_down_renderer.max_score_cbv_id.append(adv_obj_id)  # AV2 对抗车设置为黑色

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
                if has_vel:
                    adv_velocity[start_step:start_step + remain_len, :2] = source_velocity[:remain_len, :2]
                else:
                    # 兜底：由位置差分估计速度，首帧速度保持当前值。
                    for t in range(start_step + 1, start_step + remain_len):
                        adv_velocity[t, :2] = (adv_pos[t, :2] - adv_pos[t - 1, :2]) / max(FIXED_DT, 1e-6)

            # 4) valid mask：保持 CBV 在预测窗口内有效，避免 replay 认为对象消失。
            if adv_valid is not None:
                adv_valid[start_step:start_step + remain_len] = True

            cbv_policy = self.engine.get_policy(adv_obj_id)
            if hasattr(cbv_policy, "compute_action"):
                cbv_action = cbv_policy.compute_action(start_step)
                if cbv_action is not None:
                    adv_obj.before_step(cbv_action)

            # 同步本次推理使用的 current_state。
            # 这里与 engine.data_manager.current_scenario 写入相同的 CBV 预测状态；
            # 数据结构或数组形状不正确时直接抛出异常，不静默跳过。
            if (
                    self.current_state is not None
                    and "tracks" in self.current_state
                    and real_adv_id in self.current_state["tracks"]
            ):
                cs = self.current_state["tracks"][real_adv_id]["state"]

                if "position" in cs:
                    cs["position"][
                    start_step:start_step + remain_len,
                    :2,
                    ] = source_xy_center[:remain_len, :2]

                if has_heading and "heading" in cs:
                    heading_target = cs["heading"][
                                     start_step:start_step + remain_len
                                     ]
                    heading_source = source_heading[:remain_len]

                    # ScenarioNet heading 可能是 (T,) 或 (T, 1)，
                    # 只做对应的维度转换，不修改 heading 数值。
                    if heading_target.ndim == 2:
                        heading_source = heading_source.reshape(-1, 1)

                    cs["heading"][
                    start_step:start_step + remain_len
                    ] = heading_source

                if has_vel and "velocity" in cs:
                    cs["velocity"][
                    start_step:start_step + remain_len,
                    :2,
                    ] = source_velocity[:remain_len, :2]

                if "valid" in cs:
                    cs["valid"][
                    start_step:start_step + remain_len
                    ] = True

                if self.info is not None:
                    cbv_patch_info = self.info.setdefault(
                        "cbv_patch_debug",
                        {},
                    )
                    debug_entry = {
                        "trajectory_reference": str(trajectory_reference),
                        "patch_offset": int(patch_offset),
                        "start_step": int(start_step),
                        "current_obj_xy": np.asarray(
                            adv_obj.position,
                            dtype=np.float64,
                        )[:2].tolist(),
                        "patched_first_xy": np.asarray(
                            source_xy_center[0, :2],
                            dtype=np.float64,
                        ).tolist(),
                    }

                    # nearest_dist/cross_track/allow_rollback are produced only by
                    # align_cbv_control_trajectory(), i.e. the rear-axle/Pluto path.
                    # Center-referenced planners such as VBD intentionally do not run
                    # that alignment, so do not fabricate Pluto-only debug metrics.
                    if trajectory_reference != "center":
                        required_debug_keys = (
                            "nearest_dist",
                            "cross_track",
                            "allow_rollback",
                        )
                        missing_debug_keys = [
                            key
                            for key in required_debug_keys
                            if key not in patch_match_info
                        ]
                        if missing_debug_keys:
                            raise RuntimeError(
                                "CBV alignment debug info missing keys %s for "
                                "trajectory_reference=%s"
                                % (missing_debug_keys, trajectory_reference)
                            )
                        debug_entry.update({
                            "nearest_dist": float(patch_match_info["nearest_dist"]),
                            "cross_track": float(patch_match_info["cross_track"]),
                            "allow_rollback": bool(patch_match_info["allow_rollback"]),
                        })

                    cbv_patch_info[str(real_adv_id)] = debug_entry

    def _compute_control(self):
        """计算 ego PID 控制动作。

        模型输出的是未来轨迹；这里用预瞄点匹配、Stanley-like 横向误差和速度误差
        转成 MetaDrive throttle/steering。
        """
        pred_traj = self._pred_traj[0, :-1, :]

        ego_vehicle = self.control_object
        current_heading = ego_vehicle.heading_theta
        current_speed = ego_vehicle.speed
        current_pos_rear = self._ego_model_reference_position(ego_vehicle)

        # ===== RLPlanningPolicy-style tracking =====
        # 1) 用“预瞄点”匹配
        preview_distance = np.clip(current_speed * 0.6, 3.0, 6.0)
        preview_pos = self._preview_point(current_pos_rear, current_heading, preview_distance)
        heading_vec = np.array([np.cos(current_heading), np.sin(current_heading)], dtype=np.float32)

        # 2) 找离预瞄点最近且在车头前方的 match 点
        match_idx = self.get_continuous_match_idx(
            pred_traj=pred_traj,
            preview_pos=preview_pos,
            current_pos=current_pos_rear,
            current_heading=current_heading,
            last_match_idx=self.last_match_idx,
        )
        self.last_match_idx = match_idx
        p_match = pred_traj[min(match_idx, len(pred_traj)-1), :2]

        # 3) 选取目标点 (lookahead)
        lookahead_distance = np.clip(current_speed * 0.8, 3.0, 6.0)
        target_idx = self._find_lookahead_idx(pred_traj[:, :2], match_idx, lookahead_distance)
        target_idx = self._ensure_forward_target(
            pred_traj[:, :2], match_idx, target_idx, current_pos_rear, current_heading
        )
        p_target = pred_traj[target_idx, :2]

        # ========== 目标点合法性最终校验与外推 ==========
        fallback_mode = "none"
        if np.dot(p_target - current_pos_rear, heading_vec) <= 0:
            fallback_idx = self._find_forward_anchor_idx(
                pred_traj[:, :2],
                current_pos_rear,
                current_heading,
                preferred_start=max(match_idx - 5, 0),
            )
            if fallback_idx is not None:
                target_idx = int(fallback_idx)
                p_target = pred_traj[target_idx, :2]
                if np.dot(p_match - current_pos_rear, heading_vec) <= 0:
                    p_match = pred_traj[target_idx, :2]
                fallback_mode = "trajectory_forward_anchor"
            else:
                # Last resort: stay close to the matched path point and only nudge slightly forward.
                p_target = p_match + heading_vec * min(lookahead_distance, 1.5)
                fallback_mode = "matched_point_nudge"
        self.engine.agents['default_agent'].key_points = [p_target]

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
        lateral_error_m = self._signed_cross_track_error(current_pos_rear, p_match, vec_traj_dir)

        if pred_traj.shape[-1] >= 4:
            # 索引 2, 3 对应模型输出的 [cos(yaw), sin(yaw)]
            traj_heading = np.arctan2(pred_traj[idx2, 3], pred_traj[idx2, 2])
        else:
            traj_heading = np.arctan2(vec_traj_dir[1], vec_traj_dir[0])

        heading_error = traj_heading - current_heading
        heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi
        k_stanley = 1.5
        combined_error = heading_error - np.arctan2(k_stanley * lateral_error_m, max(current_speed, 1.0))

        # 6) 纵向误差: 优先从模型预测中直接读取速度以避免震荡
        target_speed = current_speed
        arc_speed = None
        if target_idx > match_idx:
            segment = pred_traj[match_idx:target_idx+1, :2]
            if len(segment) > 1:
                arc_len = float(np.linalg.norm(np.diff(segment, axis=0), axis=1).sum())
                arc_speed = arc_len / max((target_idx - match_idx)*FIXED_DT, 1e-6)
        if pred_traj.shape[-1] >= 6:
            # 索引 4, 5 对应模型输出的 [vx, vy]
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

        if arc_speed is not None:
            target_speed = max(float(target_speed), float(arc_speed))
        steering = self.lateral_pid.get_result(-combined_error)
        speed_error = current_speed - target_speed
        throttle_brake = self.longitudinal_pid.get_result(speed_error)
        steering = float(np.clip(steering, -1.0, 1.0))
        throttle_brake = float(np.clip(throttle_brake, -1.0, 1.0))
        return [steering, throttle_brake]

    def _preview_point(self, current_pos, current_heading, preview_distance: float) -> np.ndarray:
        """沿车头方向生成 world-frame 预瞄点。"""
        return np.asarray(current_pos, dtype=np.float32) + preview_distance * np.array(
            [np.cos(current_heading), np.sin(current_heading)], dtype=np.float32
        )

    def _find_lookahead_idx(self, traj_xy: np.ndarray, start_idx: int, lookahead_m: float) -> int:
        """从 start_idx 沿轨迹累积弧长，找到 lookahead 目标点。"""
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
            # 防止重复点导致目标索引抖动。
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
        """确保目标点位于 ego 前方；否则沿轨迹继续向前找。"""
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
        """选择真正位于车头前方的最近锚点，优先使用较靠后的点。"""
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
        """计算点到路径切线的有符号横向误差。"""
        t = np.asarray(path_tangent, dtype=np.float32)
        tn = float(np.linalg.norm(t))
        if tn < 1e-6:
            return 0.0
        t = t / tn
        # 路径切线的左法向。
        n = np.array([-t[1], t[0]], dtype=np.float32)
        return float(np.dot(np.asarray(p, dtype=np.float32) - np.asarray(p_on_path, dtype=np.float32), n))

    def update_state(self, time_index):
        """Return a patched scenario state for inference without mutating the engine's original scenario.

        由于 `base_state` 每次从 self._base_scenario_cache 拉取最原始的未被污染的纯净录像数据，
        为了确保从仿真第 0 步到当前的 `time_index` 的所有闭环发生后的状态都不丢失被覆盖，
        我们需要把截止目前所有沉积在 hist_buffer 中的实际仿真轨迹全部覆写上去。
        """
        # base_state 始终保持初始化时深拷贝的纯净原始场景。
        base_state = self._base_scenario_cache
        length = int(base_state.get('length', 0))

        # warmup 或没有历史时不 patch，直接返回原始场景。
        if time_index <= 0 or length <= 0 or len(self.hist_buffer) == 0:
            return base_state

        # 浅拷贝 dict、深拷贝数组：既不污染 base_state，也避免整场景重复深拷贝。
        patched_state = dict(base_state)
        patched_tracks = dict(base_state.get('tracks', {}))
        patched_state['tracks'] = patched_tracks

        base_track = base_state['tracks'][self.sdc_id]
        patched_track = dict(base_track)
        patched_tracks[self.sdc_id] = patched_track

        base_track_state = base_track['state']
        patched_track_state = dict(base_track_state)
        patched_track['state'] = patched_track_state

        # NumPy 数组必须 copy，否则会把原始 fallback 场景改脏。
        patched_traj = np.array(
            base_track_state["position"],
            copy=True,
        )
        patched_head = np.array(
            base_track_state["heading"],
            copy=True,
        )
        patched_vel = np.array(
            base_track_state["velocity"],
            copy=True,
        )
        patched_valid = np.array(
            base_track_state["valid"],
            copy=True,
        )

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
                patched_valid[step_idx] = True


        patched_track_state['position'] = patched_traj
        patched_track_state['heading'] = patched_head
        patched_track_state['velocity'] = patched_vel
        patched_track_state["valid"] = patched_valid

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
            valid = np.array(base_cbv_state["valid"], copy=True)

            for step_idx, st in hist.items():
                if 0 <= step_idx < len(pos):
                    pos[step_idx, :2] = st["position"][:2]
                    heading[step_idx] = st["heading"]
                    velocity[step_idx, :2] = st["velocity"][:2]
                    valid[step_idx] = True

            track_state["position"] = pos
            track_state["heading"] = heading
            track_state["velocity"] = velocity
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

    def _convert_adv_model_traj_xy_to_center(self, obj, traj, default_heading=0.0):
        if self._trajectory_reference_for_engine(self.AdvSim) == "center":
            return np.asarray(traj[:, :2], dtype=np.float64)
        return self._convert_rear_axle_traj_xy_to_center(
            obj,
            traj,
            default_heading=default_heading,
        )

    def _ego_rear_to_center_distance(self) -> float:
        return float(
            self.current_state["metadata"]["ego_vehicle_parameters"].get(
                "rear_axle_to_center",
                1.67,
            )
        )

    def _convert_ego_rear_traj_xy_to_center(
            self,
            traj: np.ndarray,
            default_heading: float = 0.0,
    ) -> np.ndarray:
        """Convert Pluto ego trajectory xy from rear axle to vehicle center."""
        if traj is None or len(traj) == 0:
            return np.zeros((0, 2), dtype=np.float64)

        xy = np.asarray(traj[:, :2], dtype=np.float64)
        d = self._ego_rear_to_center_distance()
        if d <= 1e-6:
            return xy

        heading = self._extract_traj_heading(
            traj,
            default_heading=default_heading,
        )
        if heading is None:
            heading = np.full(
                (len(traj),),
                float(default_heading),
                dtype=np.float64,
            )
        return xy + np.stack(
            [np.cos(heading), np.sin(heading)],
            axis=-1,
        ) * d

    def get_hist_buffer(self, time_index, state):
        """记录 ego 当前实际执行状态，供下一次 update_state() patch 场景历史。"""
        current_pos, current_heading, current_speed = state
        velocity_2d = np.array([
            current_speed * np.cos(current_heading),
            current_speed * np.sin(current_heading)
        ])
        self.hist_buffer[int(time_index)] = {
            "position": np.array(current_pos, dtype=np.float64).copy(),
            "heading": float(current_heading),
            "velocity": velocity_2d,
            "valid": True,
        }

    def get_agent_hist_buffer(self, time_index):
        """记录受控 CBV 当前实际状态，供 Pluto 下一次推理使用。"""
        scenario_to_obj_id = self._get_scenario_to_object_id_map()

        for sid in list(self._controlled_cbv_scenario_ids):
            if str(sid) in self.external_history_owned_ids:
                continue
            obj_id = scenario_to_obj_id.get(str(sid), None)
            if obj_id is None:
                continue

            obj = self.engine.get_objects().get(obj_id)
            if obj is None:
                continue

            heading = float(obj.heading_theta)
            speed = float(obj.speed)
            velocity_2d = np.array([
                speed * np.cos(heading),
                speed * np.sin(heading)
            ], dtype=np.float64)

            self.agent_hist_buffer.setdefault(str(sid), {})[int(time_index)] = {
                "position": np.array(obj.position, dtype=np.float64).copy(),
                "heading": heading,
                "velocity": velocity_2d,
                "valid": True,
            }

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

    def get_continuous_match_idx(
            self,
            pred_traj,
            preview_pos,
            current_pos,
            current_heading,
            last_match_idx,
            search_backward=5,
            search_forward=10,
            allow_stateful_smoothing=True
    ):
        traj_xy = pred_traj[:, :2]
        old_idx = int(last_match_idx)
        search_start = max(0, old_idx - search_backward)
        search_end = min(len(traj_xy), old_idx + search_forward)
        local_traj = traj_xy[search_start:search_end]
        local_distances = np.linalg.norm(
            local_traj - preview_pos,
            axis=1
        )
        heading_vec = np.array([
            np.cos(current_heading),
            np.sin(current_heading)
        ], dtype=np.float32)
        vec_to_points = local_traj - current_pos
        forward_mask = (
                np.dot(vec_to_points, heading_vec) > 0.0
        )
        if forward_mask.any():
            masked_dist = np.where(
                forward_mask,
                local_distances,
                1e6
            )
            local_idx = int(np.argmin(masked_dist))
        else:
            local_idx = int(np.argmin(local_distances))
        match_idx = search_start + local_idx

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
        return match_idx

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

    def align_cbv_control_trajectory(self, adv_obj, adv_traj):
        """按当前 CBV 实际中心状态裁剪最新 Pluto 控制轨迹。

        返回：
        - aligned_traj：从连续匹配点开始的 Pluto 原始控制轨迹，坐标语义仍是后轴；
        - patch_offset：在原始轨迹中的起始索引；
        - match_info：最近距离、横向误差等连续性调试信息。

        policy 在控制阶段只执行一次对齐并缓存结果；runtime 直接读取该缓存，
        保证 MetaDrive mirror 与 ROS 平台轨迹使用同一段规划。
        """
        if adv_traj is None:
            return np.zeros((0, 0), dtype=np.float64), 0, {}

        raw = np.asarray(adv_traj)
        if raw.ndim != 2 or len(raw) == 0:
            return raw, 0, {}

        patch_offset, match_info = self._select_cbv_patch_offset(
            adv_obj,
            raw,
        )
        required = max(int(self.platform_output_point_num), 2)
        max_offset = max(len(raw) - required, 0)
        patch_offset = int(np.clip(patch_offset, 0, max_offset))
        return raw[patch_offset:], patch_offset, match_info

    def _rear_axle_to_center_distance(self, obj) -> float:
        """返回当前工程用于非 ego 车辆的后轴到中心近似距离。

        当前 Pluto/Policy 链路把控制轨迹当作后轴参考点；MetaDrive 对象位置和平台
        状态使用车辆中心，因此按 PlutoDataset 约定使用 0.35 * length 转回中心。
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