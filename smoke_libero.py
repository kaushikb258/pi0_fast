"""Exercise one LIBERO environment, GPU offscreen rendering, and ten dummy steps. No policy/training."""
import json, pathlib
import numpy as np
from PIL import Image
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from OpenGL import GL
import torch, mujoco, robosuite

out=pathlib.Path('/home/kb/pi0_fast/setup_logs/libero')
suites=benchmark.get_benchmark_dict()
counts={name:suites[name]().n_tasks for name in benchmark.libero_suites}
print('Task suites:',counts,flush=True)
assert sum(counts.values())==130
suite=suites['libero_spatial']()
task=suite.get_task(0)
env=OffScreenRenderEnv(bddl_file_name=str(pathlib.Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),camera_heights=256,camera_widths=256)
try:
 env.seed(0)
 obs=env.reset()
 obs=env.set_init_state(suite.get_task_init_states(0)[0])
 for _ in range(10):
  obs,reward,done,info=env.step([0.0]*6+[-1.0])
 frame=obs['agentview_image']
 assert frame.shape==(256,256,3) and frame.std()>1
 assert np.isfinite(obs['robot0_eef_pos']).all()
 renderer=GL.glGetString(GL.GL_RENDERER).decode()
 vendor=GL.glGetString(GL.GL_VENDOR).decode()
 assert 'NVIDIA' in vendor and '5090' in renderer,(vendor,renderer)
 Image.fromarray(frame[::-1]).save(out/'libero-smoke.png')
 result={'status':'LIBERO_SIMULATOR_PASS','task_suites':counts,'task':task.language,'steps':10,'image_shape':list(frame.shape),'render_vendor':vendor,'renderer':renderer,'torch':torch.__version__,'mujoco':mujoco.__version__,'robosuite':robosuite.__version__,'training_started':False}
 (out/'simulator-result.json').write_text(json.dumps(result,indent=2))
 print(json.dumps(result,indent=2),flush=True)
finally:
 env.close()
