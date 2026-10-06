"""Verify conversion against source at episode starts, midpoints and ends."""
import json
from pathlib import Path
import h5py,numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from mimicgen_square_common import DATASET,REPO_ID,state_from_obs
root=Path('/home/kb/pi0_fast/datasets/lerobot')/REPO_ID
meta=json.loads((root/'source_manifest.json').read_text())
ds=LeRobotDataset(REPO_ID,root=root)
assert len(ds)==153477 and ds.num_episodes==1000
checks=0
with h5py.File(DATASET,'r') as f:
 offsets=np.cumsum([0]+[x['frames'] for x in meta['episode_mapping']])
 for ep in [0,1,10,100,250,500,750,998,999]:
  d=f['data'][meta['episode_mapping'][ep]['source_demo']]
  for frame in [0,len(d['actions'])//2,len(d['actions'])-1]:
   sample=ds[int(offsets[ep]+frame)]
   obs={k:d['obs'][k][frame] for k in ['robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos']}
   np.testing.assert_allclose(sample['state'],state_from_obs(obs),atol=1e-6)
   np.testing.assert_allclose(sample['actions'],d['actions'][frame],atol=1e-6)
   for key,source in [('image','agentview_image'),('wrist_image','robot0_eye_in_hand_image')]:
    img=np.rint(sample[key].numpy().transpose(1,2,0)*255).astype(np.uint8)
    np.testing.assert_array_equal(img,d['obs'][source][frame])
   checks+=1
result={'status':'PASS','episodes':1000,'frames':153477,'checked_frames':checks,'checks':'numeric states/actions + exact RGB equality, starts/midpoints/ends of nine episodes'}
Path('/home/kb/pi0_fast/setup_logs/mimicgen/data_check.json').write_text(json.dumps(result,indent=2));print(result)
