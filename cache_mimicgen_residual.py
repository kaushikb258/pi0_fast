"""Cache observation-only features and actual greedy actions; no controller training."""
import argparse,hashlib,json,time
from pathlib import Path
from fast_decode_validation import DECODE_VALIDATION_VERSION
import h5py,numpy as np
from tqdm import tqdm
from mimicgen_square_common import DATASET,PROMPT,state_from_obs
from residual_policy import FrozenFeaturePolicy,DEFAULT_CHECKPOINT,LAYERS,feature_code_hash


def atomic_json(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)

def observation(d,t):
    state=state_from_obs({k:d['obs'][k][t] for k in ['robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos']})
    return {'observation/image':d['obs/agentview_image'][t],
            'observation/wrist_image':d['obs/robot0_eye_in_hand_image'][t],
            'observation/state':state,'prompt':PROMPT}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,default=DEFAULT_CHECKPOINT);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stride',type=int,default=5);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--max-samples',type=int,default=0,help='For smoke checks only; 0 means all selected samples')
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    if a.stride<1 or a.max_samples<0:p.error('Invalid sample selection')
    with h5py.File(DATASET,'r') as f:
        names=sorted(f['data'],key=lambda x:int(x.split('_')[-1]))
        rng=np.random.default_rng(a.seed);order=rng.permutation(len(names));val=set(order[:round(.1*len(names))].tolist())
        items=[(i,t) for i,name in enumerate(names) for t in range(0,len(f['data'][name]['actions']),a.stride)]
        # Standard deviations use training demonstrations only, in raw simulator command units.
        train_actions=np.concatenate([f['data'][name]['actions'][:,:6] for i,name in enumerate(names) if i not in val])
        scales=np.maximum(train_actions.std(axis=0),1e-3).tolist()
    if a.max_samples:items=items[:a.max_samples]
    signature={'checkpoint':str(a.checkpoint.resolve()),'dataset':DATASET,'dataset_bytes':Path(DATASET).stat().st_size,
        'stride':a.stride,'seed':a.seed,'max_samples':a.max_samples,'layers':LAYERS,'feature_code_hash':feature_code_hash(),
        'decode_validation':DECODE_VALIDATION_VERSION,'feature_dtype':'float32','horizon':10,'motion_scales':scales,'episodes':names,'validation_episode_indices':sorted(val),
        'samples':len(items),'input_contract':'Observation prefix only; no action targets; greedy model predictions; raw simulator action units.'}
    a.output.mkdir(parents=True,exist_ok=True);manifest_path=a.output/'manifest.json'
    if manifest_path.exists():
        old=json.loads(manifest_path.read_text())
        if old['signature']!=signature:raise ValueError('Cache settings/source changed; choose a new directory')
        if old['complete']:print('Cache already complete:',a.output);return
        if not a.resume:raise ValueError('Incomplete cache: pass --resume')
        completed=old['completed'];mode='r+'
    else:
        if any(a.output.iterdir()):raise ValueError('Nonempty cache directory without manifest')
        completed=0;mode='w+'
    shapes={'features':((len(items),4,2048),np.float32),'base_actions':((len(items),10,7),np.float32),
        'target_actions':((len(items),10,7),np.float32),'time_mask':((len(items),10),np.bool_),
        'valid_decode':((len(items),),np.bool_),'is_validation':((len(items),),np.bool_),
        'episode_index':((len(items),),np.int32),'frame_index':((len(items),),np.int32)}
    arrays={k:np.lib.format.open_memmap(a.output/(k+'.npy'),mode=mode,dtype=d,shape=shape) for k,(shape,d) in shapes.items()}
    def save_progress(done,complete=False):
        for x in arrays.values():x.flush()
        atomic_json(manifest_path,{'signature':signature,'completed':done,'complete':complete,
            'valid_samples':int(arrays['valid_decode'][:done].sum())})
    save_progress(completed)
    import jax
    assert jax.default_backend()=='gpu','Frozen VLA extraction must use GPU'
    policy=FrozenFeaturePolicy(a.checkpoint);start=time.monotonic()
    with h5py.File(DATASET,'r') as f:
        for n in tqdm(range(completed,len(items)),initial=completed,total=len(items),desc='Cache frozen VLA',unit='sample'):
            i,t=items[n];d=f['data'][names[i]];T=len(d['actions'])
            try:
                output,features=policy.infer_with_features(observation(d,t))
            except Exception as exc:
                save_progress(n)
                raise RuntimeError(f'Cache sample {n}: {names[i]}, frame {t}; progress saved') from exc
            base=np.asarray(output['actions']);assert base.shape==(10,7) and np.isfinite(base).all()
            # h5py fancy indexing rejects repeated tail indices; read the small contiguous chunk first.
            chunk=d['actions'][t:min(t+10,T)];target=chunk[np.minimum(np.arange(10),len(chunk)-1)]
            values={'features':features,'base_actions':base,'target_actions':target,'time_mask':np.arange(t,t+10)<T,
                'valid_decode':output['action_decode']['valid'],'is_validation':i in val,'episode_index':i,'frame_index':t}
            for k,v in values.items():arrays[k][n]=v
            if (n+1)%64==0:save_progress(n+1)
    save_progress(len(items),True)
    print('CACHE_COMPLETE',a.output,'elapsed_seconds',time.monotonic()-start,flush=True)
if __name__=='__main__':main()
