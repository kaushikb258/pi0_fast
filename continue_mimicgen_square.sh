#!/usr/bin/env bash
# Resume MimicGen Square D0 from 20k to 60k TOTAL optimizer updates (40k additional).
# Run: bash /home/kb/pi0_fast/continue_mimicgen_square.sh
# Preview: bash /home/kb/pi0_fast/continue_mimicgen_square.sh --dry-run
# Original Gemma weights frozen; Gemma LoRA + full SigLIP vision backbone trainable.
# Preserves optimizer state and original checkpoint; restarts LR at 2e-5.
set -eo pipefail
case "${1:-}" in
  '') DRY_RUN=0 ;;
  --dry-run) DRY_RUN=1 ;;
  *) printf 'Usage: bash %s [--dry-run]\n' "$0" >&2; exit 2 ;;
esac
(( $# <= 1 )) || exit 2
SCRIPT_PATH="$(realpath -- "${BASH_SOURCE[0]}")"
PROJECT_DIR=/home/kb/pi0_fast
if [[ -n "${MIMICGEN_20K_CHECKPOINT:-}" ]]; then
  MIMICGEN_20K_CHECKPOINT="$(realpath -m -- "$MIMICGEN_20K_CHECKPOINT")"
fi
source "$PROJECT_DIR/activate_pi0.sh"
set -u
export HF_HUB_OFFLINE=1 WANDB_MODE=disabled PYTHONUNBUFFERED=1
CONFIG=pi0_fast_mimicgen_square_lora
EXP_NAME="${EXP_NAME:-mimicgen_square_fast_plus_continue_60k_b8_$(date +%Y%m%d_%H%M%S)}"
TOTAL_STEPS=60000
COMPLETED_STEPS=20000
SOURCE_CHECKPOINT="${MIMICGEN_20K_CHECKPOINT:-$PROJECT_DIR/checkpoints/$CONFIG/mimicgen_square_fast_plus_20k_b8_20260920_094341/19999}"
BATCH_SIZE=8
# Use 8 here for microbatch 1 if memory is tight; effective batch stays 8.
GRADIENT_ACCUMULATION_STEPS=4
# EDIT THIS to choose where model weights and optimizer checkpoints are saved.
CHECKPOINT_BASE_DIR="/home/kb/pi0_fast/checkpoints"
CHECKPOINT_DIR="$CHECKPOINT_BASE_DIR/$CONFIG/$EXP_NAME"
FINAL_CHECKPOINT="$CHECKPOINT_DIR/$((TOTAL_STEPS-1))"
LOG_DIR="$PROJECT_DIR/setup_logs/training/$EXP_NAME"
DATASET_DIR="$PROJECT_DIR/datasets/lerobot/local/mimicgen_square_d0"
NORM_STATS="$PROJECT_DIR/openpi/assets/$CONFIG/local/mimicgen_square_d0/norm_stats.json"
[[ "$EXP_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || exit 2
[[ "$CONDA_PREFIX" == /home/kb/miniconda3/envs/pi0 ]] || exit 1
for item in "$DATASET_DIR/source_manifest.json" "$NORM_STATS" "$SOURCE_CHECKPOINT/params/_METADATA" "$SOURCE_CHECKPOINT/train_state/_METADATA" "$SOURCE_CHECKPOINT/_CHECKPOINT_METADATA"; do
  [[ -e "$item" ]] || { printf 'Missing input: %s\n' "$item" >&2; exit 1; }
done
[[ ! -e "$CHECKPOINT_DIR" && ! -e "$LOG_DIR" ]] || { printf 'Choose a new EXP_NAME; this experiment already exists.\n' >&2; exit 1; }
# Refuse to silently change normalization when resuming.
cmp -- "$NORM_STATS" "$SOURCE_CHECKPOINT/assets/local/mimicgen_square_d0/norm_stats.json" || {
  printf 'Checkpoint normalization differs from current training assets. Stop and inspect.\n' >&2; exit 1;
}
COMMAND=( "$CONDA_PREFIX/bin/python" scripts/train.py "$CONFIG"
  --exp-name "$EXP_NAME" --checkpoint-base-dir "$CHECKPOINT_BASE_DIR"
  --num-train-steps "$TOTAL_STEPS" --batch-size "$BATCH_SIZE" --num-workers 0
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
  --lr-schedule.warmup-steps 0 --lr-schedule.peak-lr 2e-5
  --lr-schedule.restart-step "$COMPLETED_STEPS"
  --lr-schedule.decay-steps "$TOTAL_STEPS" --lr-schedule.decay-lr 2e-6
  --log-interval 10 --save-interval 10000 --keep-period 10000
  --resume --no-overwrite --no-wandb-enabled )
printf 'Initialization: saved 20k model AND optimizer state\nDataset: %s\nSteps: %s; batch: %s\nFinal weights: %s/params/\nLogs: %s\n' "$DATASET_DIR" "$TOTAL_STEPS" "$BATCH_SIZE" "$FINAL_CHECKPOINT" "$LOG_DIR"
printf 'Gradient accumulation: %s microbatches per update; microbatch size: %s\n' "$GRADIENT_ACCUMULATION_STEPS" "$((BATCH_SIZE/GRADIENT_ACCUMULATION_STEPS))"
printf 'Source checkpoint: %s\nAdditional updates: %s; cumulative dataset passes: ~3.13\n' "$SOURCE_CHECKPOINT" "$((TOTAL_STEPS-COMPLETED_STEPS))"
printf 'LR restart: 2e-5 at 20k, cosine decay to 2e-6 at 60k; no warmup.\n'
printf 'New saved checkpoint labels: 30000, 40000, 50000, 59999 (final). Original 19999 retained separately.\n'
printf 'The shuffled data loader restarts; this is not a bit-for-bit uninterrupted run.\n' 
printf 'Command: '; printf '%q ' "${COMMAND[@]}"; printf '\n'
printf 'Evaluation afterward: bash %q %q\n' "$PROJECT_DIR/eval_mimicgen_square.sh" "$FINAL_CHECKPOINT"
if (( DRY_RUN )); then
  printf 'Dry run complete: no checkpoint copied, model loaded or training started.\n'; exit 0
fi
mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR"
# Independent copy: no hard links. Copy-on-write if the filesystem supports it.
# Copy the complete checkpoint, including Adam moments, counters and normalization.
printf 'Copying source checkpoint into the new experiment...\n'
cp -a --reflink=auto "$SOURCE_CHECKPOINT" "$CHECKPOINT_DIR/19999"
printf '%s\n' "$SOURCE_CHECKPOINT" > "$LOG_DIR/source-checkpoint.txt"
printf '%q ' "${COMMAND[@]}" > "$LOG_DIR/command.sh"
printf '\n' >> "$LOG_DIR/command.sh"
git rev-parse HEAD > "$LOG_DIR/git-commit.txt"
git diff > "$LOG_DIR/source.patch"
"$CONDA_PREFIX/bin/python" -m pip freeze > "$LOG_DIR/pip-freeze.txt"
cp "$SCRIPT_PATH" "$LOG_DIR/launcher.sh"
cp "$PROJECT_DIR/mimicgen_square_common.py" "$LOG_DIR/"
cp "$PROJECT_DIR/openpi/src/openpi/training/gradient_accumulation.py" "$LOG_DIR/"
cp "$DATASET_DIR/source_manifest.json" "$NORM_STATS" "$LOG_DIR/"
# train.py prints dynamic total/trainable/LoRA counts and actual backbone freeze status.
# Restore the existing optimizer schedule counter; do not repeat warmup.
# 60k * 8 / 153,477 starting frames ~= 3.128 cumulative dataset passes.
# Cosine uses the restored global optimizer counter minus the 20k restart offset.
# Wall time includes model/data loading, JIT compilation, updates and final checkpoint flush.
TIMING_LOG="$LOG_DIR/timing.log"
printf 'Started: %s\n' "$(date --iso-8601=seconds)" | tee "$TIMING_LOG"
set +e
/usr/bin/time -a -o "$TIMING_LOG" \
  -f 'Elapsed wall time: %E\nElapsed seconds: %e\nTraining process exit status: %x' \
  "${COMMAND[@]}" 2>&1 | tee "$LOG_DIR/train.log"
RUN_STATUS=$?
set -e
printf 'Finished: %s\nPipeline exit status: %s\n' "$(date --iso-8601=seconds)" "$RUN_STATUS" >> "$TIMING_LOG"
cat "$TIMING_LOG" | tee -a "$LOG_DIR/train.log"
(( RUN_STATUS == 0 )) || exit "$RUN_STATUS"
printf '\nTraining complete. Final checkpoint: %s\n' "$FINAL_CHECKPOINT"
printf 'Evaluate MimicGen Square D0: bash %q %q\n' "$PROJECT_DIR/eval_mimicgen_square.sh" "$FINAL_CHECKPOINT"
