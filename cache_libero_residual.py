"""Cache frozen Spatial VLA features and predictions. No controller training."""
import argparse
import json
import time
from pathlib import Path
from fast_decode_validation import DECODE_VALIDATION_VERSION
import numpy as np
from tqdm import tqdm
from libero_residual_data import DATASET, load_metadata, validation_split, read_episode, observation, target_chunk, dataset_digest

def atomic_json(path, data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)

def main():
    from libero_residual_policy import CONFIG, DEFAULT_CHECKPOINT, LAYERS, feature_code_hash
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,default=DEFAULT_CHECKPOINT)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dataset',type=Path,default=DATASET)
    p.add_argument('--stride',type=int,default=5);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--max-samples',type=int,default=0,help='Smoke check only; never train from this cache')
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    if a.stride<1 or a.max_samples<0:p.error('Invalid sample selection')
    episodes=load_metadata(a.dataset);val=validation_split(episodes,a.seed)
    print('Reading dataset fingerprint and training-only motion statistics...',flush=True)
    training=[read_episode(a.dataset,e,False)['actions'][:,:6] for e in episodes if e['episode_index'] not in val]
    scales=np.maximum(np.concatenate(training).std(axis=0),1e-3).tolist();del training
    items=[(e['episode_index'],t) for e in episodes for t in range(0,e['length'],a.stride)]
    if a.max_samples:items=items[:a.max_samples]
    signature={'config':CONFIG,'checkpoint':str(a.checkpoint.resolve()),'dataset':str(a.dataset.resolve()),
        'dataset_sha256':dataset_digest(a.dataset,episodes),'stride':a.stride,'seed':a.seed,
        'max_samples':a.max_samples,'layers':LAYERS,'feature_code_hash':feature_code_hash(),
        'decode_validation':DECODE_VALIDATION_VERSION,'feature_dtype':'float32','horizon':10,'motion_scales':scales,'samples':len(items),
        'validation_episode_indices':sorted(val),'episodes':episodes,
        'input_contract':'Observation prefix only; no target actions; base and target in final simulator-command units before EMA.',
        'eval_script':'eval_libero_residual.sh'}
    a.output.mkdir(parents=True,exist_ok=True);manifest=a.output/'manifest.json'
    if manifest.exists():
        old=json.loads(manifest.read_text())
        if old['signature']!=signature:raise ValueError('Cache provenance changed; use a new output directory')
        if old['complete']:print('Cache already complete:',a.output);return
        if not a.resume:raise ValueError('Incomplete cache: pass --resume')
        completed=old['completed'];mode='r+'
    else:
        if any(a.output.iterdir()):raise ValueError('Output must be empty')
        completed=0;mode='w+'
    n=len(items)
    specs={'features':((n,4,2048),np.float32),'base_actions':((n,10,7),np.float32),
        'target_actions':((n,10,7),np.float32),'time_mask':((n,10),np.bool_),
        'valid_decode':((n,),np.bool_),'is_validation':((n,),np.bool_),
        'episode_index':((n,),np.int32),'frame_index':((n,),np.int32),'task_index':((n,),np.int32)}
    arrays={k:np.lib.format.open_memmap(a.output/(k+'.npy'),mode=mode,dtype=d,shape=shape) for k,(shape,d) in specs.items()}
    def progress(done,complete=False):
        for array in arrays.values():array.flush()
        atomic_json(manifest,{'signature':signature,'completed':done,'complete':complete,
                             'valid_samples':int(arrays['valid_decode'][:done].sum())})
    progress(completed)
    print(f'Frozen Spatial 25k VLA; {n} samples; {len(episodes)-len(val)}/{len(val)} train/validation demos',flush=True)
    import jax
    from libero_residual_policy import FrozenLiberoFeaturePolicy
    if jax.default_backend()!='gpu':raise RuntimeError('GPU required for VLA caching')
    policy=FrozenLiberoFeaturePolicy(a.checkpoint);start=time.monotonic()
    by_id={e['episode_index']:e for e in episodes};loaded=None;done=completed
    try:
        for index in tqdm(range(completed,n),initial=completed,total=n,desc='Cache frozen LIBERO VLA',unit='sample'):
            episode,t=items[index];meta=by_id[episode]
            if loaded!=episode:d=read_episode(a.dataset,meta);loaded=episode
            output,features=policy.infer_with_features(observation(d,t))
            base=np.asarray(output['actions'],dtype=np.float32)
            if base.shape!=(10,7) or not np.isfinite(base).all():raise ValueError('Invalid continuous action output')
            target,mask=target_chunk(d['actions'],t)
            values={'features':features,'base_actions':base,'target_actions':target,'time_mask':mask,
                'valid_decode':bool(output['action_decode']['valid']),'is_validation':episode in val,
                'episode_index':episode,'frame_index':t,'task_index':meta['task_index']}
            for k,v in values.items():arrays[k][index]=v
            done=index+1
            if done%64==0:progress(done)
    except BaseException:
        progress(done);raise
    progress(done,True)
    print('CACHE_COMPLETE',a.output,'elapsed_seconds',time.monotonic()-start,flush=True)
if __name__=='__main__':main()
