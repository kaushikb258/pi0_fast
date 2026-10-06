#!/usr/bin/env bash
set -eo pipefail
PROJECT_DIR=/home/kb/pi0_fast
CHECKPOINT="${LIBERO_CHECKPOINT:-$PROJECT_DIR/checkpoints/pi0_fast_libero_spatial_lora/spatial_fast_plus_25k_b8_20260919_163905/24999}"
CHECKPOINT="$(realpath -m -- "$CHECKPOINT")"
STRIDE=5
CACHE_DIR="$PROJECT_DIR/cache/libero_residual_25k_stride${STRIDE}_fp32"
[[ $# == 0 || ( $# == 1 && "$1" == --dry-run ) ]] || { echo 'Usage: bash cache_libero_residual.sh [--dry-run]'; exit 2; }
source "$PROJECT_DIR/activate_pi0.sh"
COMMAND=(python -u "$PROJECT_DIR/cache_libero_residual.py" --checkpoint "$CHECKPOINT" --output "$CACHE_DIR" --stride "$STRIDE" --resume)
printf 'Frozen LIBERO 25k VLA; Gemma layers 4,8,12,18; no weight updates.\nCommand: '
printf '%q ' "${COMMAND[@]}"; printf '\n'
[[ "${1:-}" == --dry-run ]] && exit 0
mkdir -p "$PROJECT_DIR/setup_logs/libero_residual"
export HF_HUB_OFFLINE=1
{ date -Is; time "${COMMAND[@]}"; } 2>&1 | tee "$PROJECT_DIR/setup_logs/libero_residual/cache_$(date +%Y%m%d_%H%M%S).log"
