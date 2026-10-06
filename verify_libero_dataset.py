"""Verify download integrity, all episode row counts, and a LeRobot action chunk; no training."""
import concurrent.futures, hashlib, json, pathlib
import numpy as np
import pyarrow.parquet as pq
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
root=pathlib.Path('/home/kb/pi0_fast/datasets/lerobot/physical-intelligence/libero')
out=pathlib.Path('/home/kb/pi0_fast/setup_logs/libero')
manifest=json.loads((out/'dataset-manifest.json').read_text())
def verify(f):
 p=root/f['path']
 assert p.stat().st_size==f['size'],str(p)
 if f['sha256']:
  with p.open('rb') as stream:
   assert hashlib.file_digest(stream,'sha256').hexdigest()==f['sha256'],str(p)
 return pq.read_metadata(p).num_rows if p.suffix=='.parquet' else 0
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 frames=sum(pool.map(verify,manifest['files']))
info=json.loads((root/'meta/info.json').read_text())
assert frames==info['total_frames']
episodes=list(root.glob('data/*/*.parquet'))
assert len(episodes)==info['total_episodes']
print('All available LFS SHA-256 hashes, file sizes, and episode row counts verified.',flush=True)
ds=LeRobotDataset('physical-intelligence/libero',root=root,episodes=[0],revision=manifest['revision'],delta_timestamps={'actions':[i/info['fps'] for i in range(10)]},video_backend='pyav')
sample=ds[0]
assert isinstance(sample.get('task'), str) and sample['task'].strip()
assert tuple(sample['image'].shape)==(3,256,256)
assert tuple(sample['wrist_image'].shape)==(3,256,256)
assert tuple(sample['state'].shape)==(8,)
assert tuple(sample['actions'].shape)==(10,7)
for k in ['image','wrist_image','state','actions']:
 assert np.isfinite(sample[k].numpy()).all(),k
result={'status':'LIBERO_DATASET_PASS','repo_id':manifest['repo_id'],'revision':manifest['revision'],'path':str(root),'bytes':manifest['total_bytes'],'episodes':len(episodes),'frames':frames,'tasks':info['total_tasks'],'fps':info['fps'],'reader_sample_keys':list(sample),'sample_shapes':{k:list(sample[k].shape) for k in ['image','wrist_image','state','actions']},'prompt':sample.get('task'),'lfs_sha256_verified':True, 'all_file_sizes_verified':True,'training_started':False}
(out/'dataset-result.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2),flush=True)
