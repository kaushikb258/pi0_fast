# LIBERO installation and dataset

LIBERO is an independent benchmark by Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone—not a Physical Intelligence release. Official repository: https://github.com/Lifelong-Robot-Learning/LIBERO

Physical Intelligence supplies the OpenPI integration and the converted dataset used here: https://huggingface.co/datasets/physical-intelligence/libero

## Two environments, distinct purposes

- **pi0**: working OpenPI / pi0-FAST environment for offline learning from recorded demonstrations and GPU inference. No packages were changed during LIBERO installation; the before/after pip manifests match exactly.
- **libero**: Python 3.8.20 simulator environment, `/home/kb/miniconda3/envs/libero`. Used for task execution, success-rate evaluation, and visualization. Offline pi0-FAST imitation learning does not need the simulator running.

Simulator source is local and editable at `/home/kb/pi0_fast/openpi/third_party/libero`, the official OpenPI submodule revision `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`. OpenPI's lightweight client is also installed editable in this environment so future evaluations can communicate with a model server running in pi0.

## Installation choices

Inspected `examples/libero/README.md`, its requirements files, LIBERO's requirements/setup/configuration code, and the upstream documentation before installing. Followed the separated simulator/model layout documented by OpenPI.

Used OpenPI's pinned Python 3.8 simulator requirements together with LIBERO's requirements. Replaced only the PyTorch family CUDA 11.3 wheels with corresponding CPU wheels: torch 1.11.0+cpu, torchvision 0.12.0+cpu, torchaudio 0.11.0+cpu. These utilities do not run the OpenPI model. Rendering uses NVIDIA EGL on the RTX 5090, independently of PyTorch CUDA support. Installed CMake 3.31.6 **inside libero** because egl_probe must compile its helper. No system CUDA, drivers, apt packages, or packages in existing Conda environments were modified. The project dataset path was added as an environment-local variable in pi0.

MuJoCo 3.2.3, robosuite 1.4.1, NumPy 1.22.4. Full resolved package versions are saved in `setup_logs/libero/libero-pip-freeze.txt`; Conda specifications and installation logs are beside it.

Configuration is local at `/home/kb/pi0_fast/libero_config/config.yaml`; no global `~/.libero` configuration was overwritten. `datasets/libero_raw` is an empty configured location for optional raw LIBERO files, not the downloaded converted training dataset.

## Simulator verification: PASS

- pip dependency check: no broken requirements.
- Supported suites: Spatial (10), Object (10), Goal (10), LIBERO-90 (90), LIBERO-10 (10): 130 distinct tasks.
- Reset the first Spatial task, loaded a supplied initial state, and executed 10 dummy actions.
- Captured a nonempty 256x256 RGB image and finite robot state.
- OpenGL vendor: NVIDIA Corporation; renderer: NVIDIA GeForce RTX 5090/PCIe/SSE2.
- OpenPI's LIBERO evaluation CLI imports and its `--help` command work.
- No trained policy rollout, benchmark scoring, training, or tokenizer fitting was performed.

An initial smoke script enumerated a legacy `libero_100` alias whose task map is absent in the pinned source. The final smoke script uses the five suites in LIBERO's supported `libero_suites` list (LIBERO-100 is represented by the 90/10 split). No upstream code change was needed.

Logs/results/image: `setup_logs/libero/simulator-smoke.log`, `simulator-result.json`, `libero-smoke.png`.

## Demonstration dataset

Download requested by the user in addition to the simulator.

- Repository: `physical-intelligence/libero`
- Published format: LeRobot v2.0; supported by the existing LeRobot reader in pi0.
- Pinned revision: `9dfa69510ea9e1613fc54112bc706444b686a231` (`v2.0` branch).
- Expected size: 34,938,926,941 bytes across 1,699 files.
- Metadata: 1,693 episodes, 273,465 frames, 40 tasks, 10 FPS. Combines Spatial, Object, Goal, and LIBERO-10.
- Destination: `/home/kb/pi0_fast/datasets/lerobot/physical-intelligence/libero`.
- Images are embedded in episode Parquet files; no separate videos are required for this version.
- State dimension: 8; action dimension: 7. Language task descriptions live in metadata.
- Attribution chain: original LIBERO demonstrations -> OpenVLA modified RLDS data -> Physical Intelligence's LeRobot conversion. Dataset license: CC BY 4.0.

Both pi0’s Conda environment configuration and `activate_pi0.sh` now set `HF_LEROBOT_HOME=/home/kb/pi0_fast/datasets/lerobot`, so OpenPI's existing dataset ID resolves to the local download. No dataset normalization statistics were computed for your training configuration yet.

**Dataset verification: PASS.** All 1,699 expected files have matching sizes. Every available LFS SHA-256 hash matches. All 1,693 episode Parquet files were checked; their row counts sum to 273,465 frames. A real episode loaded entirely offline through the existing LeRobot reader in pi0, with both images shaped [3,256,256], state [8], actions [10,7], and a nonempty natural-language instruction. All tested arrays were finite.

LeRobot emits an informational compatibility warning because this published v2.0 dataset uses global rather than per-episode statistics. The installed reader explicitly supports it and the offline check passed. The dataset was not silently migrated or modified; custom OpenPI normalization-statistics preparation remains a later step.

Verification files: `setup_logs/libero/dataset-manifest.json`, `dataset-download.log`, `dataset-verification.log`, and `dataset-result.json`. One interrupted HTTP connection was recovered using the resumable downloader; final hashes passed. No training or tokenizer fitting was started.

## Use

For model work and future offline training:

```bash
source /home/kb/pi0_fast/activate_pi0.sh
```

For the simulator, in its own terminal (its paths and EGL settings are saved in Conda):

```bash
conda activate libero
python /home/kb/pi0_fast/smoke_libero.py
```

For download recovery (completed files are reused):

```bash
source /home/kb/pi0_fast/activate_pi0.sh
HF_HUB_DISABLE_XET=1 python /home/kb/pi0_fast/download_libero_dataset.py
```

To repeat the full download integrity and one-episode reader check after completion:

```bash
python /home/kb/pi0_fast/verify_libero_dataset.py
```

The two environments can run concurrently in separate terminals. The simulator consumes actions from a policy during evaluation; offline training consumes the recorded demonstrations. Nothing in these setup commands launches training.

Plain `conda activate pi0` also loads HF_LEROBOT_HOME. If pi0 was already active before setup, deactivate/reactivate it or source the helper to load the new dataset path. LIBERO variables are likewise environment-local and do not modify shell profiles.
