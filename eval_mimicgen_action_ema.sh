#!/usr/bin/env bash
# Optional action-EMA evaluation of a trained MimicGen Square checkpoint.
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LAMBDA=0.3  # EMA weight on previous filtered motion; 0 is identity.
export ACTION_EMA_LAMBDA="$LAMBDA"
exec bash "$PROJECT_DIR/eval_mimicgen_square.sh" "$@"
