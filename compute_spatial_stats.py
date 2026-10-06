"""Use upstream normalization code on every Spatial frame; no model training."""
import dataclasses
import json
from pathlib import Path
import sys
sys.path.insert(0, '/home/kb/pi0_fast/openpi')
from scripts import compute_norm_stats
from openpi.training import config

if __name__ == '__main__':
    name='pi0_fast_libero_spatial_lora'
    root=Path('/home/kb/pi0_fast/datasets/lerobot/local/libero_spatial')
    info=json.loads((root/'meta/info.json').read_text())
    assert info['total_tasks']==10 and info['total_episodes']==432 and info['total_frames']==52970
    # 10 divides 52,970 exactly, so no frames are dropped by the upstream loader.
    config._CONFIGS_DICT[name]=dataclasses.replace(config.get_config(name),batch_size=10,num_workers=4)
    compute_norm_stats.main(name)
