#!/usr/bin/env bash
set -eo pipefail
PROJECT_DIR=/home/kb/pi0_fast
# Residual correction first; existing constant action-EMA afterward.
LAMBDA=0.3
BETA="${RESIDUAL_BETA:-1.0}" # Set RESIDUAL_BETA=0 for a base-policy parity check.
[[ $# -ge 1 && $# -le 2 ]] || { echo 'Usage: bash eval_mimicgen_residual.sh path/to/best.pt [--dry-run]'; exit 2; }
export RESIDUAL_CONTROLLER="$(realpath -e -- "$1")"
export RESIDUAL_BETA="$BETA"
export ACTION_EMA_LAMBDA="$LAMBDA"
unset ACTION_ADAPTIVE_SCALES ACTION_HYBRID_TRANSPORT_LAMBDA
CHECKPOINT="${MIMICGEN_CHECKPOINT:-$PROJECT_DIR/checkpoints/pi0_fast_mimicgen_square_lora/mimicgen_square_fast_plus_continue_60k_b8_20260920_190251/50000}"
CHECKPOINT="$(realpath -m -- "$CHECKPOINT")"
shift
exec bash "$PROJECT_DIR/eval_mimicgen_square.sh" "$CHECKPOINT" "$@"
