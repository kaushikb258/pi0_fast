#!/usr/bin/env bash
# Run from /home/kb/pi0_fast. Evaluation only; never starts training.
# Default: newly trained controller + EMA lambda 0.2, 100 episodes.
# Use "control" for the identical inference path with correction disabled.
set -euo pipefail
PROJECT_DIR=/home/kb/pi0_fast
HEAD="${LIBERO_HEAD:-$PROJECT_DIR/checkpoints/libero_residual/libero_residual_25k_decode_checked_20261006_145418/best.pt}"
MODE=residual
EXTRA_ARGS=()
usage() {
  printf 'Usage: bash %s [residual|control] [--dry-run]\n' "${0##*/}" >&2
}
if [[ "${1:-}" == residual || "${1:-}" == control ]]; then
  MODE="$1"
  shift
fi
if [[ "${1:-}" == --dry-run ]]; then
  EXTRA_ARGS=(--dry-run)
  shift
fi
if (( $# != 0 )); then usage; exit 2; fi
[[ -f "$HEAD" ]] || { printf 'Missing controller: %s\n' "$HEAD" >&2; exit 1; }

# Refuse the previous epoch-0, zero-output controller if the path is edited.
PYTHONDONTWRITEBYTECODE=1 /home/kb/miniconda3/envs/pi0/bin/python - "$HEAD" <<'PY'
import sys, torch
bundle=torch.load(sys.argv[1],map_location='cpu',weights_only=True)
if bundle.get('config')!='pi0_fast_libero_spatial_lora' or bundle.get('layers')!=[4,8,12,18]:
    raise SystemExit('Expected the LIBERO controller with layers 4,8,12,18')
if bundle['epoch']<=0:
    raise SystemExit('Refusing an epoch-0 controller')
output=[bundle['state_dict'][key] for key in ['mlp.4.weight','mlp.4.bias']]
if not all(torch.isfinite(value).all() for value in output):
    raise SystemExit('Controller output projection has nonfinite weights')
if not any(torch.count_nonzero(value).item() for value in output):
    raise SystemExit('Refusing a zero-output controller')
print('Verified trained controller | epoch:',bundle['epoch'],flush=True)
PY
export EPISODES_PER_TASK="${EPISODES_PER_TASK:-10}"
if [[ "$MODE" == control ]]; then
  export RESIDUAL_BETA=0
  printf 'EMA-only control: correction disabled (beta=0); lambda=0.2\n'
else
  export RESIDUAL_BETA=1
  printf 'Trained residual controller + EMA: beta=1; lambda=0.2\n'
fi
printf 'Tasks: 10 | Episodes/task: %s\nController: %s\n' "$EPISODES_PER_TASK" "$HEAD"
# Existing launcher selects the 25k VLA, lambda 0.2, and separate timestamped outputs.
# Both modes use the same decoder validation and feature-extraction path.
exec bash "$PROJECT_DIR/eval_libero_residual.sh" "$HEAD" "${EXTRA_ARGS[@]}"
