"""One bounded background VBD worker, immutable requests, time-valid exports.

First prediction/reset and motion-invalidated predictions may wait up to 10s.
Steady-state reuse uses a complete native future window, or raises on a missed
deadline. No fabricated tail, stale-scene result, or unbounded request queue is
allowed. Model access is serialized on the worker.
"""
import copy
from concurrent.futures import ThreadPoolExecutor
import math
import time

from unitraj.closeloop.VBD.vbd_inference import VBDInference


class AsyncVBDInference(VBDInference):
    def __init__(self, cfg):
        super().__init__(cfg)
        self._executor = None
        self._pending = None
        self._generation = 0
        self._closed = False

    def close(self):
        self._closed = True
        self.reset_cache()
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None

    def reset_cache(self):
        super().reset_cache()
        self._generation += 1
        if self._pending is not None:
            self._pending["future"].cancel()
        self._pending = None

    def _submit(self, current_state, timestep, required, role):
        if self._closed:
            raise RuntimeError("VBD inference engine is closed")
        if self._pending is not None:
            return
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="autoscene-vbd")
        # MetaDrive may patch tracks on the next control tick. Never pass live
        # arrays to the background model, including map and metadata arrays.
        snapshot = copy.deepcopy(current_state)
        worker = object.__new__(VBDInference)
        worker.__dict__ = dict(self.__dict__)
        worker.inference_intervals = dict(self.inference_intervals)
        worker._configured_intervals = dict(self._configured_intervals)
        worker._joint_cache = None
        worker._last_request_timestep = None
        worker.inference_count = 0
        self._pending = {
            "generation": self._generation,
            "metadata": current_state.get("metadata"),
            "map": current_state.get("map_features"),
            "future": self._executor.submit(self._sample, worker, snapshot, timestep, list(required), role),
        }

    @staticmethod
    def _sample(worker, snapshot, timestep, required, role):
        worker._prepare_joint_prediction(snapshot, timestep, required, role)
        return (worker._joint_cache, worker.inference_intervals, worker._inference_cost_ema,
                worker.cadence_budget_exceeded)

    def _collect(self, wait=False):
        pending = self._pending
        if pending is None or (not wait and not pending["future"].done()):
            return False
        self._pending = None
        try:
            cache, intervals, cost, overloaded = pending["future"].result(timeout=10.0 if wait else 0.0)
        except Exception:
            self.reset_cache()
            raise
        if pending["generation"] != self._generation:
            return False
        cache["scene_metadata"] = pending["metadata"]
        cache["scene_map"] = pending["map"]
        self._joint_cache = cache
        self.inference_intervals = dict(intervals)
        self._inference_cost_ema = cost
        self.cadence_budget_exceeded = overloaded
        self.inference_count += 1
        return True

    def _compatible(self, cache, state, timestep, required):
        key = (self._scenario_id(state), str(state.get("metadata", {}).get("sdc_id", "")), tuple(sorted(required)))
        same_tick_expansion = (timestep == cache["sample_timestep"]
                               and set(cache["key"][2]).issubset(key[2])
                               and set(key[2]).issubset(cache["token_to_slot"]))
        if (cache["key"][:2] != key[:2] or cache["scene_metadata"] is not state.get("metadata")
                or cache["scene_map"] is not state.get("map_features")
                or (cache["key"][2] != key[2] and not same_tick_expansion)
                or cache["track_ids"] != tuple(sorted(str(x) for x in state.get("tracks", {})))):
            return False
        cache["key"] = key
        return True

    def prediction_pending_at(self, timestep):
        """Read-only preflight BEFORE physics advances (simulation backpressure).

        Do not collect here: scene identity and worker errors must still be
        checked by the normal inference path. Hardware callers keep fail-fast.
        """
        cache, pending = self._joint_cache, self._pending
        if cache is None or pending is None or pending["future"].done():
            return False
        age = int(timestep) - cache["sample_timestep"]
        return age > cache["trajectory"].shape[1] - self.output_future_steps

    def _prepare_joint_prediction(self, current_state, timestep, required_agent_ids=None, role="ego"):
        if self._closed or self.model is None or self.dataset is None:
            raise RuntimeError("VBD must be initialized and open before inference")
        timestep = int(timestep)
        started = time.perf_counter()
        required = self._dedupe_ids(required_agent_ids)
        sdc = str(current_state.get("metadata", {}).get("sdc_id", ""))
        if not sdc:
            raise ValueError("ScenarioNet metadata.sdc_id is required")
        if sdc not in required:
            required.insert(0, sdc)
        if not self._required_agents_valid(current_state, timestep, required):
            self.reset_cache()
            raise RuntimeError("VBD required agent is not valid at the requested time")
        cache = self._joint_cache
        if cache is not None and (
                (self._last_request_timestep is not None and timestep < self._last_request_timestep)
                or not self._compatible(cache, current_state, timestep, required)):
            self.reset_cache()
            cache = None
        cold = cache is None
        if cold:
            self._submit(current_state, timestep, required, role)
            self._collect(wait=True)  # Only first plan/reset/rebinding, not steady-state ticks.
        else:
            self._collect(wait=False)
        cache = self._joint_cache
        if cache is None or not self._compatible(cache, current_state, timestep, required):
            self.reset_cache()
            raise RuntimeError("VBD background result does not match the current scene/vehicles")
        refreshed_for_motion = self._motion_diverged(cache, current_state, timestep)
        if refreshed_for_motion:
            # A risk-triggered refresh takes precedence over throughput. No old
            # result may be committed after a new observation invalidates it.
            self.reset_cache()
            self._submit(current_state, timestep, required, role)
            self._collect(wait=True)
            cache = self._joint_cache
        age = timestep - cache["sample_timestep"]
        maximum = cache["trajectory"].shape[1] - self.output_future_steps
        if age < 0 or age > maximum:
            self.reset_cache()
            raise RuntimeError("VBD missed its prediction deadline; no complete 50-point future remains")
        # Leave measured inference time + one scheduling margin before expiry.
        sample_cost = float(cache["timing"]["total_sec"])
        # Reserve a full planning tick plus 50% measured latency for jitter.
        # The native 80-point horizon still limits reuse to 30 actual steps.
        lead = int(math.ceil((sample_cost * 1.5 + 0.5) / (0.1 * 5))) * 5
        trigger = max(5, min(self.inference_intervals[role], maximum - lead))
        if age >= trigger and self._pending is None:
            self._submit(current_state, timestep, required, role)
        self._last_request_timestep = timestep
        if not cold and not refreshed_for_motion:
            self.cache_hit_count += 1
        result = self._export_window(cache, timestep, role, not cold and not refreshed_for_motion,
                                     "motion_diverged" if refreshed_for_motion else ("cold_start" if cold else "async_valid_window"), started)
        extra = {"async_prefetch": True, "inference_pending": self._pending is not None,
                 "prefetch_trigger_steps": trigger,
                 "motion_refresh": dict(getattr(self, "last_motion_divergence", {})) if refreshed_for_motion else {},
                 "control_call_sec": time.perf_counter() - started}
        result["cadence"].update(extra)
        self.last_diagnostics.update(extra)
        return result
