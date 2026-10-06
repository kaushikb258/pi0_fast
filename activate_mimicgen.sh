# Source for direct simulator commands; model training remains in pi0.
source /home/kb/miniconda3/etc/profile.d/conda.sh
conda activate mimicgen
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export NUMBA_CACHE_DIR=/home/kb/pi0_fast/cache/numba-mimicgen
export HF_HOME=/home/kb/pi0_fast/cache/huggingface
unset UV_PROJECT_ENVIRONMENT JAX_PLATFORMS LD_LIBRARY_PATH CUDA_HOME CUDA_PATH
cd /home/kb/pi0_fast
