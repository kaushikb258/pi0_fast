"""Shared Square D0 state and image conventions for conversion and evaluation."""
import numpy as np
PROMPT='pick up the square nut and place it onto the square peg'
CONFIG='pi0_fast_mimicgen_square_lora'
REPO_ID='local/mimicgen_square_d0'
DATASET='/home/kb/pi0_fast/datasets/mimicgen/core/square_d0.hdf5'

def state_from_obs(obs):
    # robosuite stores quaternions as xyzw; matches its quat2axisangle convention.
    q=np.asarray(obs['robot0_eef_quat'],dtype=np.float64).copy()
    q[...,3]=np.clip(q[...,3],-1,1)
    den=np.sqrt(1-q[...,3]**2)
    factor=np.divide(2*np.arccos(q[...,3]),den,out=np.zeros_like(den),where=den>1e-8)
    angle=q[...,:3]*factor[...,None]
    return np.concatenate([obs['robot0_eef_pos'],angle,obs['robot0_gripper_qpos']],axis=-1).astype(np.float32)

def make_env():
    import h5py,json,mimicgen
    from robomimic.utils import obs_utils
    from robomimic.envs.env_robosuite import EnvRobosuite
    obs_utils.initialize_obs_utils_with_obs_specs({'obs':{'rgb':['agentview_image','robot0_eye_in_hand_image'], 'low_dim':['robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos']}})
    with h5py.File(DATASET,'r') as f: meta=json.loads(f['data'].attrs['env_args'])
    # Wrapper flips only the vertical axis, matching released HDF5 RGB images.
    return EnvRobosuite(env_name=meta['env_name'],render=False,render_offscreen=True,use_image_obs=True,
                        postprocess_visual_obs=False,**meta['env_kwargs'])
