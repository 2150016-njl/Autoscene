# AutoScene 1.3: per-vehicle planning and asynchronous VBD

## Current behavior

The GUI no longer exposes VBD cadence controls. The generator defaults are:

```yaml
vbd_inference_interval_steps: 10
vbd_adaptive_cadence: true
vbd_async_prefetch: true
```

`ego.planner_method` and each `vehicles.agents[].planner_method` select `Pluto`,
`VBD`, or `pluto_hybrid`. Runtime validates the complete configuration and sends
`UNITRAJ_VEHICLE_PLANNERS` before environment/policy creation. The factory and
policy both apply these overrides. `PerVehiclePolicy` groups configured track
IDs by model, reuses one instance per model, then applies each group's own
coordinate reference. Platform trajectory export also selects by track ID,
not by the old global BV model. Unconfigured background traffic remains replay.

Physical execution ownership is unchanged. Explicit model selection with
`initial_once` is rejected because that mode uses a fixed trajectory instead;
model-planned platforms require `pluto_replan`. Changing a GUI algorithm
invalidates preparation and requires prepare/start; no live model replacement.

`AsyncVBDInference` uses one worker and independent scenario snapshots. First
prediction/reset can block; steady-state control calls do not wait for diffusion.
They slice a complete 50-point window from the 80-point native prediction using
its original sampling time. Adaptive cadence considers measured inference cost;
prefetch begins early enough to leave a measured-time margin before expiry.
One pending request is permitted. Failures, invalid IDs and missed horizon
deadlines raise errors and invalidate the cache. No extrapolated tail or silent
reuse of a result from a previous scene is allowed. Reset cancels queued work;
the single worker serializes any already-running job and the new cold start.

Diagnostics include `async_prefetch`, `inference_pending`, `sample_timestep`,
`cache_age_steps`, `prefetch_trigger_steps`, `control_call_sec`, and model groups.
Model weights and the 50 diffusion steps are unchanged. Predictive reuse changes
the replanning rate and result age; it is not equivalent to fresh inference at
each tick. `UNITRAJ_PROFILE_TIMING=1` remains available for end-to-end profiling.

## Verification on 2026-09-15

RTX 4060, PyTorch 2.1.2 CUDA 12.1, actual bundled `epoch=15.ckpt`; isolated
VBD inference on one TJ-East and one nuPlan file, not a closed-loop ROS/vehicle
experiment. Package-level training registries were bypassed, not model code.
Temporary Lightning 2.2.5 loaded the newer 2.3.3 checkpoint with a version warning.

| Scene dataset | Cold first request | Maximum subsequent control request |
| --- | ---: | ---: |
| TjEast_custom_0803 | 0.824 s | 0.00184 s |
| pittsburgh_convert10000_0 | 0.665 s | 0.00468 s |

Requests used simulation frames 20–70 at five-frame/0.5 s intervals, with two
configured VBD targets and three completed joint predictions per case. Raw VBD
sampling still takes about 0.65–0.70 s. The improvement is removal of that wait
from steady-state control requests, not faster diffusion. This short isolated
test does not establish full mixed-model FPS, hardware safety, ROS/network
latency, rollout quality, or behavior under all GPU workloads.

Regression suites exercise actual parent policy methods for TJ-West and nuPlan,
per-model batching, physical mirror ownership, ROS export routing, saved YAML,
and deterministic pending/failure/reset/deadline scenarios. No real platform
commands were sent. Frontend and backend must both be deployed and restarted.

## Historical 1.2 audit (configuration below is superseded)

## Configuration

Configure `unitraj/configs/config.yaml` before creating the policy:

```yaml
replan_frequency: 5                 # unchanged runtime/export contract
vbd_inference_interval_steps: 1     # default: sample on every planning tick
vbd_ego_inference_interval_steps: null
vbd_adv_inference_interval_steps: null
```

The interval keys are positive integer **simulation steps**, each 0.1 s, not
wall-clock timers or Hz. Null role overrides inherit the common interval. Set
the common interval to `10` to request one VBD sample per simulated second;
Pluto still runs every five steps, and each planning tick still exports 50 points.
The ego override affects VBD ego only; the adv override affects the joint VBD CBV
batch, not individual CBVs. These keys never throttle Pluto/pluto_hybrid.

Suggested UI options are `1` (every planning tick), `5`, `10`, `15`, `20`, `25`,
`30`, with an optional inherit/null choice for each role. `1` and `5` have the
same sampling frequency in the existing five-step loop. Other positive integers
are accepted: requests occur on planning ticks, so e.g. `7` takes effect at ten
steps. Both-VBD configurations retain one shared model/joint sample and use the
faster role interval for both; they do not generate inconsistent independent
world samples. Changes take effect on policy creation, not live mutation.

### ROS / GUI knobs

`scenario_runtime_node.py` now forwards these same optional keys from
`/algorithm` before `create_env()` / scene reset creates the policies. GUI
controls can write that dictionary directly; no generator-source edits are
needed. Explicit recommended opt-in example:

```yaml
algorithm:
  vbd_inference_interval_steps: 10  # one sample/1.0 s; legacy default remains 1
  # Optional role overrides; null means inherit the effective common value.
  vbd_ego_inference_interval_steps: null
  vbd_adv_inference_interval_steps: null
```

For one GUI control, retain `1` as the legacy/default value and offer
`5/10/15/20/25/30`; selecting `10` enables reduced cadence explicitly. Leave
`replan_frequency` alone. Omitted `/algorithm` keys use the generator YAML;
explicit role nulls override a YAML role value to inherit the common interval.
ROS keys override YAML via the process-local JSON environment payload
`UNITRAJ_VBD_CADENCE_OVERRIDES`. Runtime replaces the whole payload on backend
initialization, including `{}` if no keys were provided, preventing stale ROS
overrides from a previous backend. Values are validated before environment/model
creation and again in the adapter. Config changes require backend restart;
this is not a live ROS-param watcher. No YAML file is rewritten at runtime.

## Code evidence in the original project

Paths below are relative to `AutoScene/src/autoscene_runtime/scripts/` and line
numbers refer to the read-only baseline, not a runtime profile:

- `Autoscene_generator/unitraj/closeloop/inference_policy.py:547` and
  `inference_policy_for_tjwest.py:415` gate `_plan_trajectory` on the same
  `replan_frequency`. Their planning functions call ego first, then CBVs. There
  is no separate VBD sampling cadence.
- `Autoscene_generator/unitraj/closeloop/VBD/vbd_inference.py:218` keys reuse by
  exact timestep and required IDs, so the next planning tick rebuilds features
  and calls diffusion again. At line 267 it discards the last 30 native points.
- `Autoscene_generator/unitraj/models/vbd/sim_agent/sim_actor.py:541` encodes the
  scene for each sample. Lines 561–592 loop through the denoising schedule,
  including CPU copies of intermediate arrays. `configs/method/VBD.yaml` sets
  `diffusion_steps: 50`; default sampler skip is 1. This is repeated work visible
  in code, **not a measured attribution of GPU time or a measured speedup**.
- `scenario_runtime_node.py:1418` rejects a replan period other than five
  frames/0.5 s. Ego and VBD CBV conversion require at least 50 points. Therefore
  increasing the shared `replan_frequency` or returning a shortened cached
  prefix would violate existing contracts.
- `Autoscene_generator/unitraj/closeloop_sim.py:50` selects `VBDPolicy` if either
  active role is VBD, including Pluto/VBD and VBD/Pluto (and pluto_hybrid mixes).
  `create_env` uses that selector for standalone and ROS. It needs no code change.

## Cache and execution contract

The adapter retains all 80 genuine native future points in world/vehicle-center
coordinates. A prediction made at simulation index `s` has row zero at `s+1`.
At request index `t`, export `cache[:, t-s:t-s+50]` for primary, background and
joint outputs alike. No repeated tail, extrapolation, spatial alignment or
wall-clock advancement is used. Controllers receive copies, not cache storage.

Reuse requires both `t-s < interval` and `t-s+50 <= 80`. Age 30 is the last
complete export; the next request must resample even if the configured interval
is larger. This bounds reuse by real available horizon, not fabricated points.

Reset/restart, time rollback (relative to the last request, not just the last
sample), a new same-ID scene instance, map replacement, changed required IDs,
changed track roster, or invalid required agents prevent reuse. Refresh failure
clears the old cache and propagates the error. Ordinary policy state updates
shallow-copy scene dictionaries, preserving metadata/map identity, so they do
not accidentally invalidate every tick. For an in-place same-object scene/map
edit with no time reset, explicitly call `policy.reset_inference_cache()` (or
`engine.reset_cache()` when using the adapter directly). Normal MetaDrive reset
recreates the policy and its history; cache invalidation is not a replacement
for the environment's full scene/history reset.

The initial ego prediction can select CBVs from its already sampled joint
agents. Adding those IDs at the **same timestep** promotes the required set
without resampling. A later candidate change, or a candidate not in the joint
sample, forces refresh. This avoids duplicate diffusion and inconsistent worlds
on the first both-VBD planning tick.

nuPlan previously repatched VBD row zero at each successive control frame.
The mixed wrapper now patches VBD CBVs only on planning ticks, like TJ-West;
the absolute-time replay track remains in place between ticks. Existing replay
policies still execute each control step. Pluto CBV behavior is untouched.

The nuPlan Pluto spatial alignment at `inference_policy.py:1846` only matches
points within the selected CBV's own prediction and clips its crop offset to
retain 50 points. It does not search nearby vehicles or reassign track IDs.
VBD bypasses that alignment (`inference_policy.py:1060`). Neither is evidence
of a nearby-car placement/snap bug in the UI; the temporal repatching issue
above is distinct.

## Diagnostics and limits

`info["cadence"]` reports role, requested interval, sample/request indices, cache
age/hit, remaining horizon, inclusive validity/export indices, refresh reason,
and lifetime inference/cache-hit counts. `info["timing"]` is current-request
adapter timing (zero preprocessing/inference on a hit); `sample_timing` retains
the originating sample timing. Timing uses `perf_counter`; inference timing
includes the adapter's CPU transfer, not a CUDA-event kernel measurement.

`policy.last_planning_diagnostics` and `action_info["planner_scheduling"]` expose
per-role wall time and VBD cadence; `UNITRAJ_PROFILE_TIMING=1` also prints it.
Ego wall time covers the engine call, CBV wall time covers batch dispatch/output
handling, and total covers the parent's planning pass. These are deliberately
wall timings, not GPU utilization statistics.

Cache advancement occurs at existing planning/export ticks. This patch does not
redesign ROS's intervening 10 Hz republishes of the most recent plan, alter
feedback ownership, add async GPU work, or reduce diffusion denoising steps.
Slower inference means older scene conditioning and may degrade responsiveness;
dynamic scene changes not listed above do not trigger a deviation-based refresh.
GPU latency, closed-loop quality and hardware safety still need separate
controlled validation before enabling temporal reuse in operation.

## CPU verification and integration scope

From the staged/project root:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test_vbd_cadence.py' -v
```

Tests use NumPy plus standard-library dependency fakes; no torch, ROS,
MetaDrive, checkpoint loads, or simulator startup. They execute the actual
adapter, both real parent planning loops and the mixed wrapper, including
absolute-time replay patching and the launch selector. Call-count assertions
are deterministic scheduling tests, not performance benchmarks.
The final backend-specific suite contains 31 tests and passes on CPU. The runtime
smoke test executes its actual initialization and override bridge with a fake
environment factory, then checks real adapter offsets and sample-call counts.
Additional bridge checks cover positive-integer validation, role precedence,
null inheritance and removal of stale settings.

Modified existing files: `unitraj/closeloop/VBD/vbd_inference.py`,
`unitraj/closeloop/VBD/vbd_policy.py`, `unitraj/configs/config.yaml`, and
`AutoScene/src/autoscene_runtime/scripts/scenario_runtime_node.py`.
New files: this note and root `tests/test_vbd_cadence.py`.
`unitraj/closeloop_sim.py` was copied unchanged into staging to test its actual
selector. Both parent policy files remain unchanged. Do not interpret files
absent from the sparse staged tree as deletions during integration.
