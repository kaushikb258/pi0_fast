#!/usr/bin/env bash
set -eo pipefail
PROJECT_DIR=/home/kb/pi0_fast
CACHE_DIR="$PROJECT_DIR/cache/mimicgen_residual_50k_stride5_fp32"
EPOCHS=100 # Fresh head: recipe used for the selected 83% result.
PATIENCE=10
BATCH_SIZE=256
LR=0.001
RESIDUAL_PENALTY=0.001
MAX_CORRECTION=1.0 # Bound in training-data standard deviation units per motion dimension.
RUN="residual_50k_$(date +%Y%m%d_%H%M%S)"
OUTPUT="$PROJECT_DIR/checkpoints/mimicgen_residual/$RUN"
source "$PROJECT_DIR/activate_pi0.sh"
COMMAND=(python "$PROJECT_DIR/train_residual_controller.py" --cache "$CACHE_DIR" --output "$OUTPUT" --epochs "$EPOCHS" --patience "$PATIENCE" --batch-size "$BATCH_SIZE" --lr "$LR" --residual-penalty "$RESIDUAL_PENALTY" --max-correction "$MAX_CORRECTION")
printf 'Fresh residual controller; frozen 50k VLA; LR=%s
' "$LR"
printf 'Command: '; printf '%q ' "${COMMAND[@]}"; printf '\nBest head weights: %s/best.pt\n' "$OUTPUT"
if [[ "${1:-}" == --dry-run ]]; then exit 0; fi
[[ $# == 0 ]] || { echo 'Only --dry-run is accepted'; exit 2; }
mkdir -p "$PROJECT_DIR/setup_logs/residual"
{ date -Is; time "${COMMAND[@]}"; } 2>&1 | tee "$PROJECT_DIR/setup_logs/residual/$RUN.log"
