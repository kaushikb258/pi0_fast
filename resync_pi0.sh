#!/usr/bin/env bash
set -euo pipefail
source /home/kb/pi0_fast/activate_pi0.sh
# Keep Conda bootstrap packages while enforcing the checked-in dependency versions.
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --inexact
GIT_LFS_SKIP_SMUDGE=1 uv pip install --python "$CONDA_PREFIX/bin/python" --no-deps -e .
