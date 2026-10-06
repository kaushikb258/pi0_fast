"""Read original Spatial demonstration actions in final simulator-command units."""
import hashlib
import io
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from openpi_client import image_tools

DATASET = Path('/home/kb/pi0_fast/datasets/lerobot/local/libero_spatial')

def episode_path(dataset, episode):
    return Path(dataset) / 'data/chunk-000' / f'episode_{episode:06d}.parquet'

def load_metadata(dataset=DATASET):
    dataset = Path(dataset)
    episodes = [json.loads(x) for x in (dataset/'meta/episodes.jsonl').read_text().splitlines()]
    tasks = {x['task']: x['task_index'] for x in
             (json.loads(line) for line in (dataset/'meta/tasks.jsonl').read_text().splitlines())}
    if len(episodes) != 432 or len(tasks) != 10:
        raise ValueError('Expected our complete 432-demo, ten-task Spatial dataset')
    for e in episodes:
        if len(e['tasks']) != 1: raise ValueError('Expected one task per episode')
        e['task_index'] = tasks[e['tasks'][0]]
    return episodes

def validation_split(episodes, seed=42):
    rng = np.random.default_rng(seed)
    validation = set()
    for task in sorted({e['task_index'] for e in episodes}):
        ids = sorted(e['episode_index'] for e in episodes if e['task_index'] == task)
        if len(ids) < 2: raise ValueError('Need train and validation demonstrations per task')
        n = min(len(ids)-1, max(1, round(.1*len(ids))))
        validation.update(rng.permutation(ids)[:n].tolist())
    return validation

def read_episode(dataset, meta, images=True):
    columns = ['state', 'actions', 'task_index'] + (['image', 'wrist_image'] if images else [])
    table = pq.read_table(episode_path(dataset, meta['episode_index']), columns=columns)
    if len(table) != meta['length']: raise ValueError('Episode length mismatch')
    d = {k: table[k].to_pylist() for k in columns}
    for key, size in [('state',8), ('actions',7)]:
        d[key] = np.asarray(d[key], dtype=np.float32)
        if d[key].shape != (len(table),size) or not np.isfinite(d[key]).all():
            raise ValueError(f'Invalid {key}')
    if set(d['task_index']) != {meta['task_index']}: raise ValueError('Task mapping mismatch')
    d['prompt'] = meta['tasks'][0]
    return d

def observation(d, t):
    def image(key):
        entry = d[key][t]
        if entry['bytes'] is None: raise ValueError('Expected embedded RGB image bytes')
        rgb = np.array(Image.open(io.BytesIO(entry['bytes'])).convert('RGB'))
        # Same resize path as the evaluation client; stored images are already oriented.
        return image_tools.convert_to_uint8(image_tools.resize_with_pad(rgb,224,224))
    return {'observation/image':image('image'), 'observation/wrist_image':image('wrist_image'),
            'observation/state':d['state'][t].copy(), 'prompt':d['prompt']}

def target_chunk(actions, t, horizon=10):
    indices = np.arange(t,t+horizon)
    return actions[np.minimum(indices,len(actions)-1)].copy(), indices < len(actions)

def dataset_digest(dataset, episodes):
    h=hashlib.sha256()
    paths=[Path(dataset)/'meta'/n for n in ['info.json','tasks.jsonl','episodes.jsonl']]
    paths += [episode_path(dataset,e['episode_index']) for e in episodes]
    for path in paths:
        h.update(str(path.relative_to(dataset)).encode())
        with path.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
