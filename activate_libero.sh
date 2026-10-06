# Source this file in the simulator terminal.
source /home/kb/miniconda3/etc/profile.d/conda.sh
conda activate libero
export LIBERO_CONFIG_PATH=/home/kb/pi0_fast/libero_config
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
export NUMBA_CACHE_DIR=/home/kb/pi0_fast/cache/numba-libero
export HF_HOME=/home/kb/pi0_fast/cache/huggingface
unset UV_PROJECT_ENVIRONMENT JAX_PLATFORMS LD_LIBRARY_PATH CUDA_HOME CUDA_PATH
cd /home/kb/pi0_fast/openpi
