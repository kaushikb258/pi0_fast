"""Check normalized Square action targets and FAST+ reconstruction, no training."""
import json,numpy as np
from pathlib import Path
from openpi.training import config,data_loader
from openpi.models.tokenizer import FASTTokenizer
cfg=config.get_config('pi0_fast_mimicgen_square_lora');dc=cfg.data.create(cfg.assets_dirs,cfg.model)
assert cfg.data.extra_delta_transform is False
for key,s in dc.norm_stats.items():
 assert np.isfinite(s.mean).all() and np.isfinite(s.std).all() and np.isfinite(s.q01).all() and np.isfinite(s.q99).all()
d=data_loader.transform_dataset(data_loader.create_torch_dataset(dc,cfg.model.action_horizon,cfg.model),dc)
tok=FASTTokenizer(cfg.model.max_token_len,fast_tokenizer_path='/home/kb/pi0_fast/fast-tokenizer')
rows=[]
for index in [0,135,136,10000,50000,100000,153476]:
 x=d[index];actions=np.asarray(x['actions']);assert actions.shape==(10,7)
 assert x['tokenized_prompt_mask'].sum()<cfg.model.max_token_len
 decoded=tok.extract_actions(x['tokenized_prompt'],10,7)
 assert decoded.shape==(10,7) and np.isfinite(decoded).all()
 error=float(np.sqrt(np.mean((decoded-actions)**2)))
 assert error<.051,error
 rows.append({'index':index,'active_tokens':int(x['tokenized_prompt_mask'].sum()),'roundtrip_rms':error})
result={'status':'PASS','action_shape':[10,7],'checks':rows,'extra_delta_transform':False}
Path('/home/kb/pi0_fast/setup_logs/mimicgen/token_check.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
