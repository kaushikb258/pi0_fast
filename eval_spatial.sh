#!/usr/bin/env bash
# Evaluate the SAME ten tasks used for Spatial-only training. No training here.
# Usage: bash eval_spatial.sh /absolute/path/to/checkpoint [--dry-run]
# Default: 10 episodes/task = 100 trials. Full suite: EPISODES_PER_TASK=50 (500 trials).
set -eo pipefail
PROJECT_DIR=/home/kb/pi0_fast
if (( $# < 1 || $# > 2 )); then
  printf 'Usage: bash %s /path/to/checkpoint [--dry-run]\n' "$0" >&2; exit 2
fi
# Resolve against the caller directory before activation changes directories.
CHECKPOINT="$(realpath -m -- "$1")"
DRY_RUN=0
if (( $# == 2 )); then
  [[ "$2" == --dry-run ]] || exit 2
  DRY_RUN=1
fi
CONFIG=pi0_fast_libero_spatial_lora
EPISODES_PER_TASK="${EPISODES_PER_TASK:-10}"
[[ "$EPISODES_PER_TASK" =~ ^[1-9][0-9]*$ ]] && (( EPISODES_PER_TASK<=50 )) || exit 2
METHOD_ARGS=()
OUTPUT_PREFIX=spatial
if [[ -n "${ACTION_EMA_LAMBDA:-}" ]]; then
  # Validate before creating files or starting the server (also in dry-run mode).
  ACTION_EMA_LAMBDA="$("/home/kb/miniconda3/envs/pi0/bin/python" - "$PROJECT_DIR" "$ACTION_EMA_LAMBDA" <<'PYCODE'
import sys
sys.path.insert(0, sys.argv[1])
from action_ema import validate_lambda
print(validate_lambda(sys.argv[2]))
PYCODE
)"
  METHOD_ARGS=( --action-ema-lambda "$ACTION_EMA_LAMBDA" )
  OUTPUT_PREFIX="spatial_action_ema_lambda_${ACTION_EMA_LAMBDA}"
  printf 'Method: action-ema; lambda=%s; motion only; gripper unchanged\n' "$ACTION_EMA_LAMBDA"
else
  printf 'Method: baseline (no action smoothing)\n'
fi
SERVER_ARGS=()
if [[ -n "${RESIDUAL_CONTROLLER:-}" ]]; then
  RESIDUAL_CONTROLLER="$(realpath -e -- "$RESIDUAL_CONTROLLER")"
  RESIDUAL_BETA="${RESIDUAL_BETA:-1.0}"
  SERVER_ARGS=( --residual-controller "$RESIDUAL_CONTROLLER" --residual-beta "$RESIDUAL_BETA" )
  OUTPUT_PREFIX="${OUTPUT_PREFIX}_residual_beta_${RESIDUAL_BETA}"
  printf 'Residual controller: %s | beta=%s (correction before EMA)\n' "$RESIDUAL_CONTROLLER" "$RESIDUAL_BETA"
fi
OUTPUT_DIR="$PROJECT_DIR/evaluation/${OUTPUT_PREFIX}_$(date +%Y%m%d_%H%M%S)"
PORT=8011
printf 'Config: %s\nCheckpoint: %s\nTasks: ALL 10 LIBERO-Spatial tasks\nEpisodes per task: %s\nOutput: %s\n' "$CONFIG" "$CHECKPOINT" "$EPISODES_PER_TASK" "$OUTPUT_DIR"
if (( DRY_RUN )); then
  printf 'Dry run: checkpoint is not loaded; no evaluation is launched.\n'; exit 0
fi
[[ -d "$CHECKPOINT/params" && -f "$CHECKPOINT/assets/local/libero_spatial/norm_stats.json" ]] || { printf 'Expected a completed Spatial-only checkpoint with its normalization assets.\n' >&2; exit 1; }
source "$PROJECT_DIR/activate_pi0.sh"
python - "$PORT" <<'PY'
import socket,sys
with socket.socket() as s:
    s.bind(('127.0.0.1',int(sys.argv[1])))
PY
mkdir -p "$OUTPUT_DIR"
cp "$PROJECT_DIR/action_ema.py" "$OUTPUT_DIR/action_ema_source.py"
HF_HUB_OFFLINE=1 python "$PROJECT_DIR/serve_audited_policy.py" --config "$CONFIG" --checkpoint "$CHECKPOINT" --port "$PORT" "${SERVER_ARGS[@]}" > "$OUTPUT_DIR/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true' EXIT
if ! python - "$PORT" "$SERVER_PID" <<'PY'
import os,sys,time,urllib.request
port,pid=map(int,sys.argv[1:])
for _ in range(180):
    try:
        os.kill(pid,0)
    except ProcessLookupError:
        raise SystemExit("Policy server exited during startup; server log follows.")
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz',timeout=1) as response:
            if response.status==200:
                break
    except OSError:
        time.sleep(1)
else:
    raise SystemExit('Server startup timed out; inspect server.log')
PY
then
  printf 'Policy server failed. Log: %s/server.log\n' "$OUTPUT_DIR" >&2
  tail -n 60 "$OUTPUT_DIR/server.log" >&2
  exit 1
fi
source "$PROJECT_DIR/activate_libero.sh"
python "$PROJECT_DIR/evaluate_spatial.py" --port "$PORT" --episodes-per-task "$EPISODES_PER_TASK" --output-dir "$OUTPUT_DIR" "${METHOD_ARGS[@]}" 2>&1 | tee "$OUTPUT_DIR/evaluation.log"
