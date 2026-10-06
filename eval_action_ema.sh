#!/usr/bin/env bash
# Action-EMA evaluation: same checkpoint, tasks, seeds and replanning as baseline.
# From pi0_fast: bash eval_action_ema.sh
# Optional: bash eval_action_ema.sh /path/to/checkpoint [--dry-run]
# Preview default checkpoint: bash eval_action_ema.sh --dry-run
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# EDIT THIS: 0 disables smoothing; compare 0.2 and 0.3. Must be 0 <= lambda < 1.
LAMBDA=0.2
EPISODES_PER_TASK="${EPISODES_PER_TASK:-10}"
CHECKPOINT="$PROJECT_DIR/checkpoints/pi0_fast_libero_spatial_lora/spatial_fast_plus_25k_b8_20260919_163905/24999"

if [[ "${1:-}" != "" && "${1:-}" != --dry-run ]]; then
  CHECKPOINT="$1"
  shift
fi
if (( $# > 1 )) || { (( $# == 1 )) && [[ "$1" != --dry-run ]]; }; then
  printf 'Usage: bash %s [checkpoint] [--dry-run]\n' "$0" >&2
  exit 2
fi
export ACTION_EMA_LAMBDA="$LAMBDA" EPISODES_PER_TASK
exec bash "$PROJECT_DIR/eval_spatial.sh" "$CHECKPOINT" "$@"
