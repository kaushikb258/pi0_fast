# OpenPI / pi0-FAST setup record

## Scope

Local setup and one synthetic GPU inference only. No LIBERO training, data conversion, normalization-statistics computation, or tokenizer fitting.

## Locations and provenance

- Project: `/home/kb/pi0_fast`
- Conda environment: `pi0`, `/home/kb/miniconda3/envs/pi0`
- OpenPI source: `/home/kb/pi0_fast/openpi`, editable installation
- Upstream: https://github.com/Physical-Intelligence/openpi
- Initial upstream commit: `215abfb217dbac7d5f1273282331b9b1866c0479`
- Local setup branch: `codex/rtx5090-setup`
- Official FAST+ source and fitted tokenizer: `/home/kb/pi0_fast/fast-tokenizer`
- Tokenizer upstream: https://huggingface.co/physical-intelligence/fast
- Tokenizer revision: `ec4d7aa71691cac0b8bed6942be45684db2110f4`
- Checkpoint: `gs://openpi-assets/checkpoints/pi0_fast_base`
- Local checkpoint: `cache/openpi/openpi-assets/checkpoints/pi0_fast_base`
- Download verified against GCS MD5 metadata for all 35 objects, 10,850,405,453 bytes. Manifest and download log are in `setup_logs`.

## Machine

Ubuntu 24.04.4 LTS; NVIDIA GeForce RTX 5090; 32607 MiB VRAM; compute capability 12.0; driver 595.84. Existing system CUDA compiler is 13.2.86. No driver or system CUDA installation was changed. The command sandbox hides GPU device nodes, so GPU checks must run in a normal host terminal or with authorized host access.

## Installation and compatibility choices

1. Read upstream README, pyproject.toml, .python-version, uv.lock, model/loading/tokenizer code, and LIBERO configs before installing OpenPI dependencies. Upstream documents Ubuntu 22.04 as its tested OS; this machine is being verified directly on 24.04.
2. Cloned OpenPI including its ALOHA and LIBERO submodules via HTTPS. No SSH credentials were needed.
3. At user request, created `pi0` using `conda create -n pi0 --clone myenv1 --copy -y`. Existing environments were not installation targets. The earlier temporary project Conda environment was subsequently removed.
4. The clone initially had Python 3.12.13, NumPy 2.5.2, and PyTorch 2.12.1+cu130. Python 3.12 satisfies OpenPI's declared floor but the locked MuJoCo 2.3.7 has no cp312 wheel; installation attempted a source build and failed. Changed **pi0 only** to Python 3.11.16, matching upstream's .python-version. Removed the clone's copied Python/CUDA 13 packages before installing the OpenPI stack. `myenv1` remains unchanged.
5. Upstream pins JAX 0.5.3, Flax 0.10.2, and torch 2.7.1. Preserved those release versions. Selected official torch **2.7.1+cu128** and torchvision **0.22.1+cu128** wheels for Blackwell, rather than the default CUDA 12.6 PyPI build. Added the explicit official PyTorch CUDA 12.8 index to local pyproject.toml and regenerated uv.lock. Source: https://pytorch.org/blog/pytorch-2-7/
6. CUDA runtime, cuBLAS, cuDNN and related libraries are Python wheels inside pi0. The existing lock's NVIDIA CUDA NVCC package remains 12.9.41 for JAX compilation; this does not replace system nvcc. No global CUDA symlink or driver change.
7. Used `GIT_LFS_SKIP_SMUDGE=1` as upstream requires for LeRobot. `uv sync --frozen --inexact` targets the Conda prefix, preserving Conda's bootstrap packages. Because uv's wheel transfers were unusually slow, exported exact pinned requirements with uv, installed those using pip, and then completed uv sync. Editable installation uses `uv pip install --no-deps -e .` after the sync, avoiding a second dependency resolution.
8. The activation helper routes caches into this project, disables JAX VRAM preallocation, requires the CUDA backend (no silent CPU fallback), and unsets CUDA_HOME/CUDA_PATH/LD_LIBRARY_PATH in the current shell to prefer environment libraries. It does not edit shell profiles or global settings.
9. uv's initial lock invocation automatically downloaded a managed Python 3.11 into its user cache; the actual pi0 environment uses Conda Python. Subsequent commands explicitly target pi0 and disable managed Python downloads.

## Source experimentation and FAST+

See `SOURCE_MAP.md`. Both the OpenPI model/wrapper and FAST+ DCT/BPE processor source are local. FAST+ is the pretrained universal tokenizer released at `physical-intelligence/fast`; it is already OpenPI's default tokenizer artifact. The smoke test explicitly uses the local copy. `tokenizer_experiment.py` offers `fast_plus` and an explicit future `custom` saved-directory option; it never trains.

Existing configs in `openpi/src/openpi/training/config.py`:

- `pi0_fast_libero`: full fine-tuning configuration.
- `pi0_fast_libero_low_mem_finetune`: LoRA configuration (`gemma_2b_lora`, EMA disabled).

Both initialize from `pi0_fast_base/params`, use the preconverted `physical-intelligence/libero` dataset, action_dim=7, action_horizon=10, max_token_len=180. They were inspected, not run. Base checkpoint assets contain several robot-specific normalization statistics but no LIBERO statistics; the smoke test uses synthetic normalized inputs and does not invent training-derived statistics.

## Commands for later use

```bash
source /home/kb/pi0_fast/activate_pi0.sh
python /home/kb/pi0_fast/verify_gpu.py jax
python /home/kb/pi0_fast/verify_gpu.py torch
python /home/kb/pi0_fast/tokenizer_experiment.py --tokenizer fast_plus
python /home/kb/pi0_fast/smoke_pi0_fast.py
```

`resync_pi0.sh` reapplies the local compatible lock and editable package. Source changes to installed OpenPI modules take effect in a fresh Python process without reinstalling. No model algorithm or tokenizer behavior was modified during setup.

## Verification results

**PASS: RTX 5090 -> OpenPI -> pi0_fast_base -> successful GPU inference.**

- Python 3.11.16; JAX/jaxlib 0.5.3; Flax 0.10.2; torch 2.7.1+cu128; torchvision 0.22.1+cu128.
- `uv sync --frozen --inexact` and editable installation completed successfully.
- `pip check`: no broken requirements.
- JAX and PyTorch each passed a BF16 256x256 matrix multiplication on CUDA. PyTorch reports sm_120 support and NVIDIA GeForce RTX 5090.
- Loaded 2,923,335,408 actual pretrained parameters onto the GPU (asserted every parameter leaf's device).
- One batch-one autoregressive inference returned a [1,256] token buffer on CUDA and decoded valid [32,32] finite, nonzero actions. The decoder's fallback path was explicitly rejected.
- Checkpoint restore: about 4.4 s. First compilation plus inference: 8.03 s. This is a smoke test, not a warmed latency benchmark or LIBERO task-quality evaluation.
- FAST+ standalone encode/decode test: [1,10,7] synthetic normalized actions, 42 tokens, RMSE 0.02964. No fitting performed.
- An initial probe used LIBERO's 10x7 shape with the unadapted base weights. Neural GPU generation succeeded, but the emitted 1024 DCT coefficients did not match that shape, and the upstream decoder fell back to zeros. That probe is preserved as `smoke-libero-shape-probe.*` and is **not** the final passing result. The final smoke test uses Pi0FASTConfig's native base defaults (32x32) and verifies real decoding.
- Local editable source and environment-local NVIDIA library paths were verified in `environment-provenance.log`.
- Existing myenv1 retains Python 3.12 and torch 2.12.1+cu130 with its original CUDA 13 packages. System nvcc remains 13.2.86.

Files in `setup_logs`: `commands.log`, `inspection-commands.txt`, `checkpoint-manifest.json`, `checkpoint-download.log`, `gpu-jax.log`, `gpu-torch.log`, `tokenizer-and-dependencies.log`, `smoke-pi0-fast.log`, `smoke_result.json`, `smoke_outputs.npz`, package manifests, and `blackwell-compatibility.patch`.

Setup stopped after successful GPU inference. No training process was launched.

## Subsequent LIBERO milestone

LIBERO simulator and the OpenPI-ready demonstration dataset were installed and verified in a later user-authorized step. See [LIBERO_SETUP.md](LIBERO_SETUP.md). No training has been started.
