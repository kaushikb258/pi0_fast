"""Train only the residual MLP from a completed frozen-VLA cache."""
import argparse, hashlib, json, random, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from residual_controller import ResidualMLP, controller_loss

class CachedSamples(Dataset):
    def __init__(self, directory, validation):
        self.arrays = {k: np.load(directory / (k+'.npy'), mmap_mode='r') for k in
                       ['features','base_actions','target_actions','time_mask','valid_decode','is_validation','episode_index']}
        self.indices = np.flatnonzero(self.arrays['valid_decode'] & (self.arrays['is_validation']==validation))
        if not len(self.indices): raise ValueError('No valid samples in train or validation split')
    def __len__(self): return len(self.indices)
    def __getitem__(self, index):
        i=self.indices[index]
        return tuple(torch.from_numpy(np.array(self.arrays[k][i],copy=True)) for k in
                     ['features','base_actions','target_actions','time_mask'])

def atomic_save(bundle, path):
    temporary=path.with_suffix('.tmp');torch.save(bundle,temporary);temporary.replace(path)

def initialize_controller(model, path, architecture, signature, manifest_hash):
    """Load head weights only; reject incompatible cached features or targets."""
    bundle=torch.load(path,map_location='cpu',weights_only=True)
    expected={'architecture':architecture,'base_checkpoint':signature['checkpoint'],
              'feature_code_hash':signature['feature_code_hash'],
              'cache_manifest_hash':manifest_hash,'motion_scales':signature['motion_scales']}
    for key,value in expected.items():
        if bundle[key]!=value:raise ValueError(f'Initial controller {key} does not match this training cache/config')
    if bundle.get('layers',[4,8,12,18])!=signature['layers']:
        raise ValueError('Initial controller layer selection does not match cache')
    model.load_state_dict(bundle['state_dict'],strict=True)
    return int(bundle['epoch'])

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--select-trained-only',action='store_true',help='Keep initial head as zero_reference.pt; select best.pt from nonzero trained epochs')
    p.add_argument('--require-learned-head',action='store_true',help='Exit 3 if the selected best head has zero correction')
    p.add_argument('--init-controller',type=Path,help='Continue from saved head weights with a fresh AdamW optimizer')
    p.add_argument('--epochs',type=int,default=100,help='Number of additional epochs in this run');p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--patience',type=int,default=10);p.add_argument('--lr',type=float,default=1e-3)
    p.add_argument('--residual-penalty',type=float,default=1e-3)
    p.add_argument('--max-correction',type=float,default=1.)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    a=p.parse_args()
    if min(a.epochs,a.batch_size,a.patience)<1 or a.lr<=0 or a.residual_penalty<0: p.error('Invalid training settings')
    manifest_bytes=(a.cache/'manifest.json').read_bytes();manifest=json.loads(manifest_bytes);s=manifest['signature']
    if not manifest['complete'] or manifest['completed']!=s['samples']:raise ValueError('Finish the feature cache first')
    if s['max_samples']:raise ValueError('Refusing a smoke-test cache; generate the complete cache')
    layers=s['layers']
    if layers != [4,8,12,18]:raise ValueError('Unsupported cached layer selection')
    train=CachedSamples(a.cache,False);val=CachedSamples(a.cache,True)
    if train.arrays['features'].shape!=(s['samples'],len(layers),2048):raise ValueError('Cache feature shape mismatch')
    train_episodes=set(train.arrays['episode_index'][train.indices].tolist())
    val_episodes=set(val.arrays['episode_index'][val.indices].tolist())
    assert train_episodes.isdisjoint(val_episodes)
    a.output.mkdir(parents=True,exist_ok=True)
    if any((a.output/name).exists() for name in ['best.pt','last.pt','metrics.jsonl']):raise ValueError('Choose a new training output directory')
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(4)
    architecture={'feature_dim':2048,'num_layers':len(layers),'horizon':10,'hidden_dims':[256,128],
                  'max_normalized_correction':a.max_correction}
    device=torch.device(a.device);model=ResidualMLP(**architecture).to(device)
    manifest_hash=hashlib.sha256(manifest_bytes).hexdigest()
    source_epoch=initialize_controller(model,a.init_controller,architecture,s,manifest_hash) if a.init_controller else 0
    scales=torch.tensor(s['motion_scales'],dtype=torch.float32,device=device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
    loaders={name:DataLoader(ds,batch_size=a.batch_size,shuffle=(name=='train'),num_workers=0)
             for name,ds in [('train',train),('validation',val)]}
    print('Frozen VLA:',s['checkpoint'],flush=True)
    print('Gemma feature layers (one-based):',layers)
    print('Initial controller:',a.init_controller or 'zero-initialized head','| source epoch:',source_epoch)
    print('Fresh AdamW optimizer | constant LR:',a.lr,'| additional epoch limit:',a.epochs,'| patience:',a.patience)
    print('SigLIP, Gemma, and existing LoRA: FROZEN; VLA is not loaded into this training process.')
    print('Head total parameters:',sum(x.numel() for x in model.parameters()),
          '| trainable:',sum(x.numel() for x in model.parameters() if x.requires_grad))
    print('Train/validation valid samples:',len(train),len(val),'| demos:',len(train_episodes),len(val_episodes))
    print('Excluded invalid chunks:',s['samples']-len(train)-len(val),'| device:',device,flush=True)
    def evaluate(loader,training=False):
        model.train(training);mse_sum=base_sum=loss_sum=denominator=0.
        context=torch.enable_grad() if training else torch.no_grad()
        with context:
            for features,base,target,mask in loader:
                features,base,target,mask=(x.to(device) for x in (features,base,target,mask))
                prediction=model(features)
                loss,mse=controller_loss(prediction,base,target,scales,mask,a.residual_penalty)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                if training:
                    optimizer.zero_grad(set_to_none=True);loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                count=float(mask.sum().item()*6)
                mse_sum+=mse.item()*count;loss_sum+=loss.item()*count;denominator+=count
                base_sum+=float(((((target[...,:6]-base[...,:6])/scales)**2)*mask[...,None]).sum().item())
        return {'motion_mse':mse_sum/denominator,'base_motion_mse':base_sum/denominator,'loss':loss_sum/denominator}
    initial=evaluate(loaders['validation']);best=initial['motion_mse'];best_epoch=source_epoch;stale=0;started=time.monotonic()
    def save(path,epoch,metrics):
        atomic_save({'architecture':architecture,'layers':layers,'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
            'config':s.get('config'),'motion_scales':s['motion_scales'],'base_checkpoint':s['checkpoint'],'feature_code_hash':s['feature_code_hash'],
            'cache_manifest_hash':manifest_hash,'epoch':epoch,
            'source_epoch':source_epoch,'additional_epochs':epoch-source_epoch,'optimizer_initialization':'fresh AdamW','validation':metrics,
            'training_args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}},path)
    # Keep the starting head if continuation never improves validation MSE.
    save(a.output/('zero_reference.pt' if a.select_trained_only else 'best.pt'),source_epoch,initial)
    if a.select_trained_only:best=float('inf');best_epoch=None
    print('Initial validation (starting head):',initial,flush=True)
    for additional_epoch in tqdm(range(1,a.epochs+1),desc='Residual MLP',unit='epoch'):
        epoch=source_epoch+additional_epoch
        training=evaluate(loaders['train'],True);validation=evaluate(loaders['validation'])
        row={'epoch':epoch,'additional_epoch':additional_epoch,'train':training,'validation':validation,'layer_weights':model.layer_weights().detach().cpu().tolist(),
             'elapsed_seconds':time.monotonic()-started}
        with (a.output/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        tqdm.write(json.dumps(row));save(a.output/'last.pt',epoch,validation)
        nonzero_head=any(bool(torch.count_nonzero(x)) for x in model.mlp[-1].parameters())
        if validation['motion_mse']<best and (nonzero_head or not a.select_trained_only):
            best=validation['motion_mse'];best_epoch=epoch;stale=0;save(a.output/'best.pt',epoch,validation)
        else:stale+=1
        if stale>=a.patience:break
    if not (a.output/'best.pt').exists():
        raise RuntimeError('No nonzero trained head was produced; inspect last.pt and metrics')
    selected=torch.load(a.output/'best.pt',map_location=device,weights_only=True)
    model.load_state_dict(selected['state_dict']);model.eval()
    correction_sum=0.;correction_count=0
    with torch.inference_mode():
        for features,_,_,mask in loaders['validation']:
            correction=model(features.to(device));valid=mask.to(device)[...,None]
            correction_sum+=float((correction.square()*valid).sum().item())
            correction_count+=int(mask.sum().item())*6
    correction_rms=(correction_sum/correction_count)**.5
    nonzero=sum(int(torch.count_nonzero(x).item()) for x in model.mlp[-1].parameters())
    status={'best_epoch':best_epoch,'best_validation_motion_mse':best,
        'initial_validation_motion_mse':initial['motion_mse'],
        'output_projection_nonzero_parameters':nonzero,
        'validation_normalized_correction_rms':correction_rms,
        'beats_initial_validation_mse':best<initial['motion_mse'],
        'zero_correction_head':correction_rms==0.,'simulator_success_not_yet_evaluated':True}
    (a.output/'training_summary.json').write_text(json.dumps(status,indent=2)+'\n')
    print('SELECTED_HEAD_AUDIT',json.dumps(status),flush=True)
    print('CONTROLLER_TRAINING_COMPLETE | best epoch:',best_epoch,'| best validation motion MSE:',best)
    print('Head weights:',a.output/'best.pt','| elapsed seconds:',time.monotonic()-started)
    if correction_rms==0.:
        print('WARNING: selected best.pt produces ZERO residual correction; not a learned-controller gain.',flush=True)
        if a.require_learned_head:raise SystemExit(3)
    print('Evaluate separately: bash',Path('/home/kb/pi0_fast')/s.get('eval_script','eval_mimicgen_residual.sh'),a.output/'best.pt')
if __name__=='__main__':main()
