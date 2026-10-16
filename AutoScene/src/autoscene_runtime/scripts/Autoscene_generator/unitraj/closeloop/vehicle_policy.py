"""Per-track planner routing; physical execution ownership stays with runtime.

Each model sees the same world snapshot. VBD cars share a joint sampler, while
Pluto/Hybrid cars are batched by model. Never mix center and rear-axle outputs
when patching replay tracks or exporting physical-platform trajectories.
"""
import time
import inspect

import numpy as np

from unitraj.closeloop.VBD.vbd_policy import VBDPolicy
from unitraj.closeloop.planner_routing import apply_planner_overrides, planner_name


class PerVehiclePolicy(VBDPolicy):
    def __init__(self, obj, seed):
        self._planner_engines = {}
        self.vehicle_planners = {}
        self._adv_groups = []
        self._group_diagnostics = {}
        self._interaction_futures = {}
        super().__init__(obj, seed)
        self.configure_vehicle_planners(dict(self.base_cfg.get("vehicle_planners", {})))

    def _center_future(self, trajectory, engine, scenario_id=None):
        result = np.asarray(trajectory)[:50].copy()
        if getattr(engine, "trajectory_reference", "center") == "rear_axle":
            if scenario_id is None:
                distance = float(self.current_state.get("metadata", {}).get("ego_vehicle_parameters", {}).get("rear_axle_to_center", 1.67))
            else:
                objects = self.engine.get_objects() if hasattr(self.engine, "get_objects") else {}
                mapping = self._get_scenario_to_object_id_map() if hasattr(self.engine, "traffic_manager") else {}
                obj = objects.get(mapping.get(str(scenario_id)))
                distance = self._rear_axle_to_center_distance(obj) if obj is not None else 1.67
            result[:, :2] += distance * result[:, 2:4]
        return result

    def _planner_interaction_kwargs(self):
        """Provide executed-plan intentions, not original PKL futures, to Pluto."""
        if str(getattr(self.AdvSim, "trajectory_model_name", "")).lower() == "vbd":
            return {}
        params = inspect.signature(self.AdvSim.run_inference_batch).parameters
        if "previous_agent_futures_by_id" not in params:
            return {}
        ego = self._center_future(self._pred_traj[0], self.Sim)
        return {"ego_future_traj": ego,
                "previous_agent_futures_by_id": dict(getattr(self, "_interaction_futures", {}))}

    def _apply_runtime_overrides(self, cfg):
        super()._apply_runtime_overrides(cfg)
        return apply_planner_overrides(cfg)

    def _init_inference_engine(self, cfg):
        variant = str(cfg.get("planner_variant", "")).strip().lower()
        name = planner_name(variant if variant == "pluto_hybrid" else cfg.get("model_name"))
        if name not in self._planner_engines:
            self._planner_engines[name] = super()._init_inference_engine(cfg)
        return self._planner_engines[name]

    def configure_vehicle_planners(self, mapping):
        choices = {str(key): planner_name(value) for key, value in mapping.items()}
        for method in set(choices.values()):
            if method not in self._planner_engines:
                self._init_inference_engine(self.get_final_config(method))
        if choices != self.vehicle_planners:
            self.reset_inference_cache()
        self.vehicle_planners = choices

    def planner_for_track(self, scenario_id):
        return self.vehicle_planners.get(str(scenario_id), planner_name(self.adv_method))

    def waiting_for_predictions(self, timestep):
        """All actors share one clock; never advance only the fast model."""
        if timestep < self.WARMUP_STEPS or timestep % self.replan_frequency:
            return []
        return [name for name, engine in self._planner_engines.items()
                if hasattr(engine, "prediction_pending_at")
                and engine.prediction_pending_at(timestep)]

    def reset_inference_cache(self):
        self._interaction_futures = {}
        super().reset_inference_cache()
        for engine in self._planner_engines.values():
            if hasattr(engine, "reset_cache"):
                engine.reset_cache()
        self._adv_groups = []
        self.current_adv_trajs = None
        self.current_aligned_adv_trajs = {}
        self.current_adv_patch_info = {}
        self._group_diagnostics = {}

    def _call_inference_engine(self, sim, *args, **kwargs):
        # A VBD ego and the VBD subset must request the same required set, even
        # when other configured cars are planned by Pluto. Other actors remain
        # in the joint context but their VBD predictions are never executed.
        if sim is self.Sim and str(getattr(sim, "trajectory_model_name", "")).lower() == "vbd":
            kwargs["required_agent_ids"] = [sid for sid in self.current_cbv_ids
                                            if self.planner_for_track(sid) == "VBD"]
        return super()._call_inference_engine(sim, *args, **kwargs)

    def _run_adv_inference(self, time_index):
        self._adv_groups = []
        self._group_diagnostics = {}
        self._interaction_futures = {}
        targets = [str(sid) for sid in self.current_cbv_ids]
        if len(targets) != len(set(targets)):
            raise RuntimeError("duplicate configured planner track IDs")
        grouped = {}
        for sid in targets:
            grouped.setdefault(self.planner_for_track(sid), []).append(sid)
        original_engine, original_method = self.AdvSim, self.adv_method
        all_ids, all_trajs, refs, candidates, selections = [], [], {}, {}, {}
        try:
            # VBD's joint diffusion is unguided. Plan its subset first, then let
            # Pluto/Hybrid score candidates against those actual selected plans.
            for method in sorted(grouped, key=lambda name: (name != "VBD", name)):
                ids = grouped[method]
                self.AdvSim = self._planner_engines[method]
                self.adv_method = method
                self.info["top_k_ids"] = list(ids)
                self.adv_agent_ids, self.current_adv_trajs = [], None
                self.current_adv_ref_lines, self.current_adv_candidates = {}, {}
                started = time.perf_counter()
                super()._run_adv_inference(time_index)
                produced = [str(sid) for sid in self.adv_agent_ids]
                if set(produced) != set(ids) or len(produced) != len(ids) or self.current_adv_trajs is None:
                    raise RuntimeError("%s did not produce exactly the configured tracks %s (got %s)"
                                       % (method, ids, produced))
                trajs = np.asarray(self.current_adv_trajs)
                if trajs.ndim != 3 or trajs.shape[1] < 50 or trajs.shape[2] != 6 or not np.isfinite(trajs).all():
                    raise RuntimeError("%s per-vehicle output must be finite [A,>=50,6]" % method)
                trajs = trajs[:, :50].copy()
                for sid, trajectory in zip(produced, trajs):
                    self._interaction_futures[sid] = self._center_future(trajectory, self.AdvSim, sid)
                self._adv_groups.append((method, produced, trajs))
                all_ids.extend(produced)
                all_trajs.extend(trajs)
                refs.update(self.current_adv_ref_lines)
                candidates.update(self.current_adv_candidates)
                selections.update(getattr(self, "cbv_score_selection_by_id", {}))
                self._group_diagnostics[method] = {
                    "tracks": list(produced), "wall_sec": time.perf_counter() - started,
                    "cadence": dict(getattr(self.AdvSim, "last_diagnostics", {})),
                }
        except Exception:
            self._interaction_futures = {}
            self._adv_groups = []
            self.current_adv_trajs = None
            self.adv_agent_ids = []
            self.current_aligned_adv_trajs = {}
            self.current_adv_patch_info = {}
            raise
        finally:
            self.AdvSim, self.adv_method = original_engine, original_method
            self.info["top_k_ids"] = targets
        self.adv_agent_ids = all_ids
        self.current_adv_trajs = np.stack(all_trajs) if all_trajs else None
        self.current_adv_ref_lines, self.current_adv_candidates = refs, candidates
        self.cbv_score_selection_by_id = selections

    def _apply_adv_control(self):
        # Row 0 always means planning-time + 0.1 s, for ALL model groups.
        if (int(self.engine.episode_step) - 1) % self.replan_frequency != 0:
            return
        original = self.AdvSim, self.adv_method, self.adv_agent_ids, self.current_adv_trajs
        aligned, patch_info = {}, {}
        try:
            for method, ids, trajectories in self._adv_groups:
                self.AdvSim = self._planner_engines[method]
                self.adv_method, self.adv_agent_ids, self.current_adv_trajs = method, ids, trajectories
                # Keep top_k_ids as the union: parent cleanup must not erase the
                # other model group's overlays or execution ownership.
                super()._apply_adv_control()
                for sid in ids:
                    if sid in self.current_adv_patch_info:
                        self.current_adv_patch_info[sid].setdefault("source", "%s_temporal_plan" % method.lower())
                # nuPlan's parent clears these dictionaries on every call.
                aligned.update(self.current_aligned_adv_trajs)
                patch_info.update(self.current_adv_patch_info)
        finally:
            self.AdvSim, self.adv_method, self.adv_agent_ids, self.current_adv_trajs = original
        self.current_aligned_adv_trajs, self.current_adv_patch_info = aligned, patch_info

    def _plan_trajectory(self, time_index):
        vbd = self._planner_engines.get("VBD")
        if vbd is not None:
            owned = {str(sid) for sid in self.current_cbv_ids if self.planner_for_track(sid) == "VBD"}
            if str(self.ego_method).lower() == "vbd":
                owned.add(str(self.sdc_id))
            vbd.interaction_controlled_ids = owned
        super()._plan_trajectory(time_index)
        # A Pluto ego was initially planned before the VBD subset. Re-score it
        # once with the actual BV plans; do not pretend VBD's imagined ego will
        # be executed. This is a bounded extra Pluto call, not a diffusion loop.
        futures = getattr(self, "_interaction_futures", {})
        params = inspect.signature(self.Sim.run_inference).parameters
        if futures and "extra_collision_trajs" in params and any(method == "VBD" for method, _, _ in self._adv_groups):
            result = self.Sim.run_inference(self.current_state, time_index,
                                           extra_collision_trajs=list(futures.values()),
                                           actual_target_futures_by_id=futures)
            trajectory, refs, candidates, info = self._normalize_inference_output(result)
            if trajectory is None or not np.isfinite(trajectory).all():
                raise RuntimeError("Interaction-aware ego replan failed")
            self._pred_traj = trajectory
            self.engine.agents["default_agent"].plan_traj = self._center_future(trajectory[0], self.Sim)[:, :2]
            self.last_match_idx = 0
            if isinstance(info, dict):
                info["top_k_ids"] = list(self.current_cbv_ids)
                self.info = info
            renderer = getattr(self.engine, "head_renderer", None)
            if renderer is not None:
                renderer.set_reference_lines(refs)
            if candidates is not None:
                self.engine.agents["default_agent"].candidate_trajectories = (
                    self._convert_ego_candidate_trajs_xy_to_center(candidates, default_heading=float(self.control_object.heading_theta))
                    if hasattr(self, "_convert_ego_candidate_trajs_xy_to_center") else candidates)
        self.last_planning_diagnostics["vehicle_groups"] = dict(self._group_diagnostics)
        self.last_planning_diagnostics["shared_plan_ids"] = sorted(futures)
