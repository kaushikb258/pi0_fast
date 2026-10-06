# Source this file: source /home/kb/pi0_fast/activate_pi0.sh
source /home/kb/miniconda3/etc/profile.d/conda.sh
conda activate pi0
export UV_PROJECT_ENVIRONMENT=/home/kb/miniconda3/envs/pi0
export UV_PYTHON_DOWNLOADS=never
export UV_CACHE_DIR=/home/kb/pi0_fast/cache/uv
export OPENPI_DATA_HOME=/home/kb/pi0_fast/cache/openpi
export HF_LEROBOT_HOME=/home/kb/pi0_fast/datasets/lerobot
export HF_HOME=/home/kb/pi0_fast/cache/huggingface
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda
export GIT_LFS_SKIP_SMUDGE=1
# Prefer environment-provided CUDA wheels over host toolkits in this shell.
unset LD_LIBRARY_PATH CUDA_HOME CUDA_PATH
cd /home/kb/pi0_fast/openpi
