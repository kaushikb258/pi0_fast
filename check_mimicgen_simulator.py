"""Verify rendering, source-image conventions and a recorded demo, without a learned policy."""
import json
from pathlib import Path
import numpy as np
import h5py,imageio
from mimicgen_square_common import make_env,DATASET,state_from_obs
out=Path('/home/kb/pi0_fast/setup_logs/mimicgen');out.mkdir(exist_ok=True)
env=make_env()
try:
 with h5py.File(DATASET,'r') as f:
  d=f['data/demo_0'];obs=env.reset_to({'model':d.attrs['model_file'],'states':d['states'][0]})
  comparison={}
  for key in ['agentview_image','robot0_eye_in_hand_image']:
   source=d['obs'][key][0];actual=obs[key]
   assert actual.shape==source.shape==(84,84,3)
   comparison[key]={n:float(np.abs(source.astype(float)-v).mean()) for n,v in [('same',actual),('vertical_flip',actual[::-1]),('rotate180',actual[::-1,::-1])]}
   assert comparison[key]['same'] < comparison[key]['vertical_flip']
   imageio.imwrite(out/(key+'_source_and_render.png'),np.concatenate([source,actual],axis=1))
  expected=state_from_obs({k:d['obs'][k][0] for k in ['robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos']})
  np.testing.assert_allclose(state_from_obs(obs),expected,rtol=0,atol=1e-4)
  frames=[]
  for a in d['actions'][:]:
   frames.append(np.concatenate([obs['agentview_image'],obs['robot0_eye_in_hand_image']],axis=1))
   obs,_,_,_=env.step(a)
  success=bool(env.is_success()['task'])
  imageio.mimwrite(out/'square_demo_replay.mp4',frames,fps=20,macro_block_size=1)
  result={'status':'PASS' if success else 'FAIL','image_orientation_errors':comparison,'recorded_demo_replay_success':success,'state_alignment_pass':True,'image_shape':[84,84,3],'control_frequency':20,'source_demo':'demo_0','model_updates':0}
  (out/'simulator_check.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));assert success
finally:
 env.env.close()
