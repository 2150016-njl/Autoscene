"""Autoscene_generator standalone loop and import-safe environment factory.

ROS runtime imports :func:`create_env` directly. Importing this module never
parses command-line arguments and never mutates the working directory.
"""
from copy import deepcopy
import argparse
import logging
import os
import time
import traceback

import yaml
from tqdm import tqdm
from metadrive.envs.scenario_env import ScenarioEnv

from unitraj.closeloop.VBD.vbd_policy import VBDPolicy
from unitraj.closeloop.pluto.pluto_policy import PlutoPolicy
from unitraj.closeloop.pluto_hybrid.pluto_hybrid_policy import PlutoHybridPolicy
from unitraj.renderer.head_renderer import HeadTopDownRenderer
from unitraj.utils.evaluate_utils import EvaluateMetrics

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
logging.getLogger("lightning_fabric.utilities.seed").setLevel(logging.WARNING)


def load_config(config_path, planner_overrides=None):
    """加载生成器配置，并保持原工程的相对数据路径语义。

    ``planner_overrides`` 只允许覆盖全局 ``ego_method`` / ``adv_method``。
    不接受逐车 planner 配置，也不对非法值做回退。

    原始 closeloop_sim.py 从 Autoscene_generator 根目录运行，因此
    eval_data_path 的相对路径也以该目录为基准，而不是以 configs 目录为基准。
    """
    config_path = os.path.abspath(os.path.expanduser(config_path))
    with open(config_path, "r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream) or {}

    runtime_data_path = os.environ.get("AUTOSCENE_DATASET_PATH", "").strip()
    if runtime_data_path:
        cfg["eval_data_path"] = runtime_data_path

    data_path = cfg.get("eval_data_path")
    if data_path and not os.path.isabs(data_path):
        generator_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cfg["eval_data_path"] = os.path.abspath(
            os.path.join(generator_root, data_path)
        )

    if planner_overrides is not None:
        if not isinstance(planner_overrides, dict):
            raise TypeError("planner_overrides must be a dict or None")
        allowed_keys = {"ego_method", "adv_method"}
        unknown = sorted(set(planner_overrides) - allowed_keys)
        if unknown:
            raise ValueError("unsupported planner override keys: %s" % unknown)
        allowed_methods = {"Pluto", "VBD", "pluto_hybrid", "pluto_hybrid_blend"}
        for key in ("ego_method", "adv_method"):
            if key not in planner_overrides:
                continue
            value = str(planner_overrides[key]).strip()
            if value not in allowed_methods:
                raise ValueError(
                    "%s must be Pluto, VBD, pluto_hybrid or pluto_hybrid_blend, got %s"
                    % (key, value)
                )
            cfg[key] = value
    return cfg


def select_policy(cfg):
    """Return a policy wrapper that supports the effective ego/BV planner mix.

    ``defaults[0].method`` is retained only as a backward-compatible default.
    ``ego_method`` and ``adv_method`` select the two planner roles independently.
    VBDPolicy is the mixed-model wrapper; pure Pluto/Pluto keeps the original
    PlutoPolicy path untouched.
    """
    defaults = cfg.get("defaults") or []
    if not defaults or not isinstance(defaults[0], dict):
        raise ValueError("config.defaults[0].method is required")

    default_method = str(defaults[0].get("method", "")).strip()
    ego_method = str(cfg.get("ego_method", default_method)).strip()
    adv_method = str(cfg.get("adv_method", default_method)).strip()
    allowed = {"Pluto", "VBD", "pluto_hybrid", "pluto_hybrid_blend"}
    if ego_method not in allowed:
        raise ValueError(
            "ego_method must be Pluto, VBD, pluto_hybrid or pluto_hybrid_blend, got %s" % ego_method
        )
    if bool(cfg.get("adv_control", True)) and adv_method not in allowed:
        raise ValueError(
            "adv_method must be Pluto, VBD, pluto_hybrid or pluto_hybrid_blend, got %s" % adv_method
        )

    adv_enabled = bool(cfg.get("adv_control", True))
    effective_adv = adv_method if adv_enabled else None

    # Preserve the validated pure Pluto path exactly.
    if ego_method == "Pluto" and (not adv_enabled or effective_adv == "Pluto"):
        return PlutoPolicy

    # Any VBD role still needs the VBD wrapper so VBD/VBD can share one joint
    # diffusion sample. Non-VBD roles are delegated back to the parent factory.
    if ego_method == "VBD" or effective_adv == "VBD":
        return VBDPolicy

    # Remaining non-pure-Pluto combinations contain a Pluto hybrid-family planner.
    return PlutoHybridPolicy


def configured_policy_class(policy_cls, cfg):
    """Bind one immutable-by-convention effective config snapshot to a policy class.

    MetaDrive's ``ScenarioEnv`` accepts an ``agent_policy`` class rather than a
    policy instance or arbitrary constructor arguments.  The ROS/UI planner
    overrides therefore have to cross that class boundary explicitly; otherwise
    ``InferencePolicy`` would reopen ``configs/config.yaml`` and lose the effective
    ``ego_method`` / ``adv_method`` selected by the UI.

    A fresh subclass is created per environment, so multiple environments in the
    same process do not share or overwrite planner configuration through mutable
    class attributes on the original policy wrappers.
    """
    if not isinstance(cfg, dict):
        raise TypeError("effective policy config must be a dict")
    effective_cfg = deepcopy(cfg)
    configured_name = "AutoSceneConfigured%s" % policy_cls.__name__
    return type(
        configured_name,
        (policy_cls,),
        {
            "_autoscene_effective_base_cfg": effective_cfg,
            "_autoscene_require_effective_base_cfg": True,
            "__module__": __name__,
        },
    )


def _autoscene_should_suppress_route_done(done, info, config):
    """Return True for route-geometry termination that AutoScene must ignore.

    AutoScene allows users to place the edited initial pose outside MetaDrive's
    recorded road polygon and then let Pluto recover/plan from that state.  Therefore
    both ``arrive_dest`` and ``out_of_road`` are diagnostic flags only in the ROS
    continuous runtime and must not terminate the experiment.  Real crash termination
    (when enabled) and max-step/horizon termination are still preserved.
    """
    if not bool(done) or not isinstance(info, dict):
        return False
    if bool(info.get("max_step", False)):
        return False

    crash_done = (
        (bool(info.get("crash_human", False)) and bool(config.get("crash_human_done", False)))
        or (bool(info.get("crash_vehicle", False)) and bool(config.get("crash_vehicle_done", False)))
        or (
            (bool(info.get("crash_object", False)) or bool(info.get("crash_building", False)))
            and bool(config.get("crash_object_done", False))
        )
    )
    if crash_done:
        return False

    return bool(
        info.get("arrive_dest", False)
        or info.get("out_of_road", False)
    )


class AutoSceneRuntimeScenarioEnv(ScenarioEnv):
    """ScenarioEnv variant used only by the ROS runtime.

    Route geometry flags such as ``arrive_dest`` and ``out_of_road`` remain visible
    in ``info`` for diagnostics, but they do not stop AutoScene continuous runtime.
    This is required because a user-defined initial pose is allowed to start outside
    the recorded road polygon.  Crash/max-step termination keeps MetaDrive semantics.
    """

    def done_function(self, vehicle_id):
        done, info = super().done_function(vehicle_id)
        if _autoscene_should_suppress_route_done(done, info, self.config):
            info = dict(info)
            info["autoscene_suppressed_route_done"] = True
            return False, info
        return done, info


def create_env(num_scenarios, config_path=None, planner_overrides=None,
               continuous_runtime=False):
    """Create the MetaDrive ScenarioEnv used by standalone and ROS runtime.

    ``continuous_runtime=True`` is reserved for AutoScene ROS runtime. Standalone
    evaluation keeps the original ScenarioEnv termination behavior unchanged.
    """
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__),
            "configs",
            "config.yaml",
        )

    cfg = load_config(config_path, planner_overrides=planner_overrides)
    database_path = cfg.get("eval_data_path")
    if not database_path:
        raise ValueError("eval_data_path is required in %s" % config_path)

    base_policy_cls = select_policy(cfg)
    policy_cls = configured_policy_class(base_policy_cls, cfg)

    env_cls = AutoSceneRuntimeScenarioEnv if continuous_runtime else ScenarioEnv

    return env_cls(
        {
            "use_render": False,
            "agent_policy": policy_cls,
            "manual_control": False,
            "log_level": logging.CRITICAL,
            "num_scenarios": int(num_scenarios),
            "sequential_seed": False,
            "start_scenario_index": 0,
            "horizon": int(cfg.get("horizon", 500)),
            "store_data": False,
            "store_map": False,
            "vehicle_config": {
                "show_navi_mark": True,
                "show_line_to_dest": True,
                "show_dest_mark": True,
                "no_wheel_friction": False,
            },
            "data_directory": database_path,
            "top_down_show_real_size": True,
        }
    )


def clear_render_overlays(env):
    """Remove per-scenario visualization state before or after reset."""
    engine = getattr(env, "engine", None)
    if engine is None:
        return

    renderer = getattr(engine, "head_renderer", None)
    if renderer is not None:
        if hasattr(renderer, "reset"):
            renderer.reset()
        elif hasattr(renderer, "clear_overlays"):
            renderer.clear_overlays()

    objects = {}
    if hasattr(engine, "get_objects"):
        try:
            objects.update(engine.get_objects())
        except Exception:
            pass
    if hasattr(engine, "agents"):
        objects.update(getattr(engine, "agents", {}))

    default_agent = getattr(engine, "agents", {}).get("default_agent")
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Load a database into MetaDrive and run closed-loop replay"
    )
    parser.add_argument(
        "--file_path",
        default=os.path.join(os.path.dirname(__file__), "configs", "config.yaml"),
    )
    parser.add_argument("--render", default="none", choices=["none", "2D"])
    parser.add_argument("--max_step", default=100, type=int)
    parser.add_argument("--eval_eps", default=24, type=int)
    return parser.parse_args(argv)


def run_standalone(args):
    cfg = load_config(args.file_path)
    eval_env = create_env(args.eval_eps, args.file_path)
    scene_score = []
    success_list = []
    total_steps = 0
    total_reward = 0.0
    mean_reward_all = 0.0
    success_rate = 0.0
    evaluate = EvaluateMetrics()
    started_at = time.time()

    pbar = tqdm(range(args.eval_eps), desc="Eval eps", unit="ep")
    try:
        for scenario_index in pbar:
            try:
                clear_render_overlays(eval_env)
                eval_env.reset(seed=scenario_index)
                clear_render_overlays(eval_env)

                scenario = deepcopy(eval_env.engine.data_manager.current_scenario)
                sdc_id = scenario["metadata"]["sdc_id"]
                summary = scenario["metadata"]["object_summary"][sdc_id]
                if summary["moving_distance"] < 5:
                    continue

                if args.render == "2D":
                    eval_env.head_renderer = HeadTopDownRenderer(eval_env)
                    eval_env.engine.head_renderer = eval_env.head_renderer
                    clear_render_overlays(eval_env)

                eval_env.engine.agents["default_agent"].expert_traj = (
                    scenario["tracks"][sdc_id]["state"]["position"]
                )
                print("id", scenario["id"])

                episode_steps = 0
                episode_reward = 0.0
                for step_index in range(args.max_step):
                    obs, reward, terminated, truncated, info = eval_env.step([0, 0])
                    if info.get("terminate_scene", False):
                        print(
                            "Terminate scenario %d: %s"
                            % (scenario_index, info.get("terminate_reason", "unknown"))
                        )
                        break

                    evaluate.step(info, obs, step_index, eval_env)
                    if args.render == "2D":
                        eval_env.head_renderer.render(
                            screen_record=False,
                            show_plan_traj=True,
                            scaling=6,
                            film_size=(6000, 4000),
                            mode="topdown",
                            text={"step": step_index, "id": scenario["id"]},
                        )

                    episode_steps += 1
                    total_steps += 1
                    episode_reward += reward
                    total_reward += reward

                    if terminated or truncated:
                        score, final_score, _ = evaluate.reset(total_steps, eval_env)
                        scene_score.append(score)
                        print(final_score)
                        print("scene_score:", score)
                        print("average_score:", sum(scene_score) / len(scene_score))
                        crashed = any(
                            info.get(key, False)
                            for key in (
                                "crash_object",
                                "crash_vehicle",
                                "crash_human",
                                "crash_building",
                                "crash_sidewalk",
                            )
                        )
                        success_list.append(0 if crashed else 1)
                        success_rate = sum(success_list) / len(success_list)
                        pbar.set_postfix({"success_rate": "%.3f" % success_rate})
                        break

                if episode_steps:
                    mean_reward_all = total_reward / max(total_steps, 1)
                clear_render_overlays(eval_env)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                print("scenario %d failed: %s" % (scenario_index, exc))
                traceback.print_exc()
                clear_render_overlays(eval_env)
    finally:
        try:
            eval_env.close()
        except Exception:
            pass

    evaluate.print_metrics(
        {"mean_reward_all": mean_reward_all, "success_rate": success_rate},
        file_name=cfg.get("exp_name", "autoscene"),
    )
    print("elapsed:", time.time() - started_at)


def main(argv=None):
    run_standalone(parse_args(argv))


if __name__ == "__main__":
    main()
