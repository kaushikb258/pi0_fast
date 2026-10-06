"""Compute OpenPI normalization on every Square D0 frame; no training."""
import dataclasses,json,sys
from pathlib import Path
sys.path.insert(0,'/home/kb/pi0_fast/openpi')
from scripts import compute_norm_stats
from openpi.training import config
if __name__ == '__main__':
    name='pi0_fast_mimicgen_square_lora'
    info=json.loads(Path('/home/kb/pi0_fast/datasets/lerobot/local/mimicgen_square_d0/meta/info.json').read_text())
    assert info['total_episodes']==1000 and info['total_frames']==153477
    batch=next(n for n in range(32,0,-1) if info['total_frames']%n==0)
    config._CONFIGS_DICT[name]=dataclasses.replace(config.get_config(name),batch_size=batch,num_workers=4)
    print('Normalization batch (no dropped frames):',batch,flush=True)
    compute_norm_stats.main(name)
