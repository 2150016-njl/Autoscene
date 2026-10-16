#!/usr/bin/env bash
set +e
AUTOSCENE_ROOT="${AUTOSCENE_ROOT:-$HOME/AutoScene}"
OUT="${1:-autoscene_runtime_diag_$(date +%Y%m%d_%H%M%S).log}"
exec > >(tee "$OUT") 2>&1

echo "===== AutoScene runtime diagnostics ====="; date
echo "AUTOSCENE_ROOT=$AUTOSCENE_ROOT"
[[ -f /opt/ros/noetic/setup.bash ]] && source /opt/ros/noetic/setup.bash || echo "[FAIL] no /opt/ros/noetic/setup.bash"
[[ -f "$AUTOSCENE_ROOT/devel/setup.bash" ]] && source "$AUTOSCENE_ROOT/devel/setup.bash" || echo "[FAIL] no $AUTOSCENE_ROOT/devel/setup.bash"

echo; echo "===== ROS environment ====="
env | grep -E '^ROS_(MASTER_URI|IP|HOSTNAME|NAMESPACE|DISTRO|VERSION)=' | sort

echo; echo "===== rosnode list ====="; rosnode list 2>&1

echo; echo "===== relevant processes ====="
ps -ef | grep -E 'scenario_runtime_node|experiment_manager_node|platform_comm_node|evaluator_node|autoscene_experiment_nodes|roslaunch' | grep -v grep

echo; echo "===== relevant topics ====="
rostopic list 2>&1 | grep -E 'autoscene|world/state|ego/state|agents/state|scenario/status|simulation/status' | sort

for t in /autoscene/experiment/status /autoscene/scenario/status /autoscene/simulation/status /autoscene/world/state /autoscene/ego/state /autoscene/agents/state; do
  echo; echo "===== rostopic info $t ====="; rostopic info "$t" 2>&1
done

echo; echo "===== one ExperimentStatus ====="; timeout 3s rostopic echo -n 1 /autoscene/experiment/status 2>&1
echo; echo "===== one ScenarioStatus ====="; timeout 3s rostopic echo -n 1 /autoscene/scenario/status 2>&1
echo; echo "===== one WorldState ====="; timeout 3s rostopic echo -n 1 /autoscene/world/state 2>&1

echo; echo "===== frequency checks ====="
for t in /autoscene/world/state /autoscene/ego/state /autoscene/agents/state; do
  echo "--- $t ---"; timeout 5s rostopic hz "$t" 2>&1; echo
done

echo "===== key params ====="
for p in /runtime /scenario /vehicles /ego /platform; do echo "--- $p ---"; rosparam get "$p" 2>&1; echo; done

echo "===== relevant services ====="; rosservice list 2>&1 | grep -E '^/autoscene/(experiment|scenario|simulation|platform)' | sort

echo; echo "===== latest ROS errors ====="
LATEST_LOG_DIR="$(ls -td "$HOME"/.ros/log/* 2>/dev/null | head -n 1)"
echo "latest log dir: $LATEST_LOG_DIR"
if [[ -n "$LATEST_LOG_DIR" && -d "$LATEST_LOG_DIR" ]]; then
  grep -RniE 'scenario_runtime|Traceback|ERROR|FATAL|failed|exception|ValueError|RuntimeError' "$LATEST_LOG_DIR" 2>/dev/null | tail -n 250
fi

echo; echo "===== done ====="; echo "saved: $OUT"
