#!/usr/bin/env bash
set -eo pipefail
PROJECT_DIR=/home/kb/pi0_fast
# Match our best LIBERO EMA comparison; change to 0 for no smoothing.
LAMBDA=0.2
BETA="${RESIDUAL_BETA:-1.0}"
[[ $# -ge 1 && $# -le 2 ]] || { echo 'Usage: bash eval_libero_residual.sh path/to/best.pt [--dry-run]'; exit 2; }
if [[ $# == 2 && "$2" != --dry-run ]]; then exit 2; fi
export RESIDUAL_CONTROLLER="$(realpath -e -- "$1")"
export RESIDUAL_BETA="$BETA"
export ACTION_EMA_LAMBDA="$LAMBDA"
CHECKPOINT="${LIBERO_CHECKPOINT:-$PROJECT_DIR/checkpoints/pi0_fast_libero_spatial_lora/spatial_fast_plus_25k_b8_20260919_163905/24999}"
CHECKPOINT="$(realpath -m -- "$CHECKPOINT")"
# Reject an accidental MimicGen or wrong-VLA head before starting any server.
/home/kb/miniconda3/envs/pi0/bin/python - "$RESIDUAL_CONTROLLER" "$CHECKPOINT" "$BETA" <<'PY'
import math,sys,torch
from pathlib import Path
b=torch.load(sys.argv[1],map_location='cpu',weights_only=True)
if b.get('config')!='pi0_fast_libero_spatial_lora' or b.get('layers')!=[4,8,12,18]:
    raise SystemExit('Expected a LIBERO controller with layers 4,8,12,18')
if b['base_checkpoint']!=str(Path(sys.argv[2]).resolve()):raise SystemExit('Frozen VLA checkpoint mismatch')
v=float(sys.argv[3])
if not math.isfinite(v) or not 0<=v<=1:raise SystemExit('Beta must be in [0,1]')
PY
shift
exec bash "$PROJECT_DIR/eval_spatial.sh" "$CHECKPOINT" "$@"
