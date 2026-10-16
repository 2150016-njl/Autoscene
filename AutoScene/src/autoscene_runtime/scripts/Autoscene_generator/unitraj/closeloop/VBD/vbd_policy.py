import os
import time


dataset_type = os.environ.get("AUTOSCENE_DATASET_TYPE", "tjwest").strip().lower()
if dataset_type == "nuplan":
    from unitraj.closeloop.inference_policy import InferencePolicy
elif dataset_type == "tjwest":
    from unitraj.closeloop.inference_policy_for_tjwest import InferencePolicy
else:
    raise ValueError(
        "AUTOSCENE_DATASET_TYPE must be nuplan or tjwest, got %s" % dataset_type
    )

from unitraj.closeloop.VBD.vbd_inference import VBDInference


class VBDPolicy(InferencePolicy):
    """Dataset-aware AutoScene policy using one shared VBD joint sampler.

    The parent policy still owns MetaDrive warmup, history patching, PID tracking,
    CBV execution ownership and actual-platform mirror behavior. This subclass
    supplies the VBD engine, reset hooks and scheduling diagnostics. Pluto calls
    remain on the parent's planning ticks; only VBD sampling can be less frequent.
    """

    def __init__(self, obj, seed):
        self._shared_vbd_engine = None
        super(VBDPolicy, self).__init__(obj, seed)

        # Mixed VBD branches use the same 0.5 s planning cadence as the platform
        # contract, without changing the pure PlutoPolicy class.
        self.replan_frequency = int(self.base_cfg.get("replan_frequency", 5))
        if self.replan_frequency <= 0:
            raise ValueError("replan_frequency must be > 0")

        # TJ-West parent intentionally leaves ego Sim unset; generic nuPlan parent
        # initializes it through _init_inference_engine during super().__init__().
        if getattr(self, "Sim", None) is None:
            self.Sim = self._init_inference_engine(self.ego_cfg)

        # When both ego and CBVs use VBD they must slice one joint diffusion sample,
        # not independently resample the traffic scene.
        if (
            getattr(self, "use_adv", False)
            and str(getattr(self, "ego_method", "")).strip().lower() == "vbd"
            and str(getattr(self, "adv_method", "")).strip().lower() == "vbd"
        ):
            self.AdvSim = self.Sim
            self._shared_vbd_engine = self.Sim

            # A joint sample cannot simultaneously describe two independently
            # resampled worlds. With VBD in both roles, the faster role wins.
            interval = min(self.Sim.inference_intervals.values())
            self.Sim.inference_intervals = dict(ego=interval, adv=interval)

        self._cadence_scene = self.engine.data_manager.current_scenario
        self._cadence_last_step = None
        self.last_planning_diagnostics = {}
        self._role_wall_seconds = {}

    def reset_inference_cache(self):
        """Explicit invalidation hook for an in-place scene/candidate replacement."""
        seen = set()
        for sim in (self.Sim, self.AdvSim):
            if sim is not None and id(sim) not in seen:
                seen.add(id(sim))
                if hasattr(sim, "reset_cache"):
                    sim.reset_cache()
        if isinstance(self.Sim, VBDInference):
            self._pred_traj = None
            self.info = None
        if isinstance(self.AdvSim, VBDInference):
            self.current_adv_trajs = None
            self.current_aligned_adv_trajs = {}
            self.current_adv_patch_info = {}
        self.last_planning_diagnostics = {}

    def act(self, agent_id):
        scene = self.engine.data_manager.current_scenario
        step = int(self.engine.episode_step)
        if (scene is not self._cadence_scene
                or (self._cadence_last_step is not None and step <= self._cadence_last_step)):
            self.reset_inference_cache()
        self._cadence_scene = scene
        self._cadence_last_step = step
        result = super(VBDPolicy, self).act(agent_id)
        if self.last_planning_diagnostics:
            self.action_info["planner_scheduling"] = dict(
                self.last_planning_diagnostics,
                planning_tick=(step - 1) % self.replan_frequency == 0,
            )
        return result

    def _call_inference_engine(self, sim, *args, **kwargs):
        started = time.perf_counter()
        result = super(VBDPolicy, self)._call_inference_engine(sim, *args, **kwargs)
        if sim is self.Sim:
            self._role_wall_seconds["ego"] = time.perf_counter() - started
        return result

    def _run_adv_inference(self, time_index):
        started = time.perf_counter()
        result = super(VBDPolicy, self)._run_adv_inference(time_index)
        self._role_wall_seconds["adv"] = time.perf_counter() - started
        return result

    def _plan_trajectory(self, time_index):
        self._role_wall_seconds = {}
        started = time.perf_counter()
        super(VBDPolicy, self)._plan_trajectory(time_index)
        diagnostics = {"time_index": int(time_index),
                       "total_wall_sec": time.perf_counter() - started}
        for role, sim in (("ego", self.Sim), ("adv", self.AdvSim)):
            if sim is None or role not in self._role_wall_seconds:
                continue
            detail = {"wall_sec": self._role_wall_seconds[role],
                      "model": getattr(sim, "trajectory_model_name", "unknown")}
            if isinstance(sim, VBDInference):
                # The ego info is captured before the shared sampler serves CBVs.
                info = self.info if role == "ego" and isinstance(self.info, dict) else {}
                cadence = info.get("cadence", sim.last_diagnostics)
                detail["inference_requested"] = cadence.get("request_timestep") == time_index
                if detail["inference_requested"]:
                    detail.update(cadence)
            diagnostics[role] = detail
        self.last_planning_diagnostics = diagnostics
        if os.environ.get("UNITRAJ_PROFILE_TIMING", "0") == "1":
            print("[Profile]", dict(diagnostics, stage="mixed_planning"))

    def _apply_adv_control(self):
        # nuPlan's parent patches every control step, unlike TJ-West. Re-applying
        # VBD row 0 at a later absolute track index restarts the old prediction.
        # Leave the already time-indexed replay track in place between exports.
        if (isinstance(self.AdvSim, VBDInference)
                and (int(self.engine.episode_step) - 1) % self.replan_frequency != 0):
            return
        return super(VBDPolicy, self)._apply_adv_control()

    def _init_inference_engine(self, specific_cfg):
        method_name = str(specific_cfg.get("model_name", "")).strip()
        if method_name == "VBD":
            if self._shared_vbd_engine is None:
                if specific_cfg.get("vbd_async_prefetch", False):
                    from unitraj.closeloop.VBD.async_vbd_inference import AsyncVBDInference
                    engine = AsyncVBDInference(specific_cfg)
                else:
                    engine = VBDInference(specific_cfg)
                engine.initialize_model()
                self._shared_vbd_engine = engine
            return self._shared_vbd_engine
        return super(VBDPolicy, self)._init_inference_engine(specific_cfg)

    def destroy(self):
        engine = getattr(self, "_shared_vbd_engine", None)
        if engine is not None and hasattr(engine, "close"):
            engine.close()
        return super(VBDPolicy, self).destroy()
