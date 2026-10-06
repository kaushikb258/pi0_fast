#!/usr/bin/env bash
set -eo pipefail
PROJECT_DIR=/home/kb/pi0_fast
CACHE_DIR="$PROJECT_DIR/cache/libero_residual_25k_stride5_fp32_decode_checked"
# Same MSE/architecture/optimization as MimicGen; corrupt decoded chunks excluded by audit.
EPOCHS=100
PATIENCE=10
BATCH_SIZE=256
LR=0.001
RESIDUAL_PENALTY=0.001
MAX_CORRECTION=1.0
RUN="libero_residual_25k_decode_checked_$(date +%Y%m%d_%H%M%S)"
OUTPUT="$PROJECT_DIR/checkpoints/libero_residual/$RUN"
[[ $# == 0 || ( $# == 1 && "$1" == --dry-run ) ]] || { echo 'Usage: bash train_libero_residual.sh [--dry-run]'; exit 2; }
source "$PROJECT_DIR/activate_pi0.sh"
COMMAND=(python -u "$PROJECT_DIR/train_residual_controller.py" --cache "$CACHE_DIR" --output "$OUTPUT" --epochs "$EPOCHS" --patience "$PATIENCE" --batch-size "$BATCH_SIZE" --lr "$LR" --residual-penalty "$RESIDUAL_PENALTY" --max-correction "$MAX_CORRECTION" --require-learned-head --select-trained-only)
printf 'Train only the residual head; all VLA weights frozen.\nCommand: '
printf '%q ' "${COMMAND[@]}"; printf '\nBest head weights: %s/best.pt\n' "$OUTPUT"
[[ "${1:-}" == --dry-run ]] && exit 0
mkdir -p "$PROJECT_DIR/setup_logs/libero_residual"
{ date -Is; time "${COMMAND[@]}"; } 2>&1 | tee "$PROJECT_DIR/setup_logs/libero_residual/$RUN.log"
