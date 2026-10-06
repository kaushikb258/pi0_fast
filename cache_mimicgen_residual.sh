#!/usr/bin/env bash
set -eo pipefail
PROJECT_DIR=/home/kb/pi0_fast
# Frozen fine-tuned VLA, NOT the original pi0_fast_base.
CHECKPOINT="${MIMICGEN_CHECKPOINT:-$PROJECT_DIR/checkpoints/pi0_fast_mimicgen_square_lora/mimicgen_square_fast_plus_continue_60k_b8_20260920_190251/50000}"
CHECKPOINT="$(realpath -m -- "$CHECKPOINT")"
STRIDE=5
CACHE_DIR="$PROJECT_DIR/cache/mimicgen_residual_50k_stride${STRIDE}_fp32"
source "$PROJECT_DIR/activate_pi0.sh"
COMMAND=(python "$PROJECT_DIR/cache_mimicgen_residual.py" --checkpoint "$CHECKPOINT" --output "$CACHE_DIR" --stride "$STRIDE" --resume)
printf 'Command: '; printf '%q ' "${COMMAND[@]}"; printf '\n'
if [[ "${1:-}" == --dry-run ]]; then exit 0; fi
[[ $# == 0 ]] || { echo 'Only --dry-run is accepted'; exit 2; }
mkdir -p "$PROJECT_DIR/setup_logs/residual"
LOG="$PROJECT_DIR/setup_logs/residual/cache_$(date +%Y%m%d_%H%M%S).log"
export HF_HUB_OFFLINE=1
{ date -Is; time "${COMMAND[@]}"; } 2>&1 | tee "$LOG"
