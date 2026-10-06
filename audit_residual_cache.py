"""Audit cached actions for the confirmed U+FFFD FAST corruption; preserve source cache.

Original token IDs were not cached. Recover quantized DCT coefficients through the
inverse output transform and flag only a rounded codepoint of U+FFFD. This does not
claim to recover every possible original token error, or impose magnitude clipping.
"""
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np
from scipy.fft import dct

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,help='Optional new independent cache; original is never edited')
    p.add_argument('--report',type=Path,required=True)
    a=p.parse_args();raw=(a.cache/'manifest.json').read_bytes();manifest=json.loads(raw);s=manifest['signature']
    if not manifest['complete'] or manifest['completed']!=s['samples']:raise ValueError('Cache incomplete')
    arrays={k:np.load(a.cache/(k+'.npy'),mmap_mode='r') for k in ['base_actions','target_actions','valid_decode','is_validation','time_mask','episode_index','frame_index']}
    libero=s.get('config')=='pi0_fast_libero_spatial_lora'
    asset_id='local/libero_spatial' if libero else 'local/mimicgen_square_d0'
    stats=json.loads((Path(s['checkpoint'])/'assets'/asset_id/'norm_stats.json').read_text())['norm_stats']['actions']
    q01=np.array(stats['q01']);q99=np.array(stats['q99'])
    processor=json.loads((Path(__file__).resolve().parent/'fast-tokenizer/processor_config.json').read_text())
    corrections=[];valid=np.array(arrays['valid_decode'],copy=True);m=arrays['time_mask']
    if libero:
        from libero_residual_data import load_metadata,read_episode
        metadata={e['episode_index']:e for e in load_metadata(Path(s['dataset']))}
    for ep in np.unique(arrays['episode_index']):
        ix=np.flatnonzero(arrays['episode_index']==ep)
        x=np.array(arrays['base_actions'][ix],dtype=np.float64)
        if libero:
            data=read_episode(Path(s['dataset']),metadata[int(ep)],False)
            x[:,:,:6]-=data['state'][arrays['frame_index'][ix],None,:6]
        normalized=2*(x-q01)/(q99-q01+1e-6)-1
        coefficients=np.rint(dct(normalized,axis=1,norm='ortho')*processor['scale']).astype(np.int64)
        codepoints=coefficients-processor['min_token']
        for offset in np.flatnonzero(np.any(codepoints==0xfffd,axis=(1,2))):
            i=int(ix[offset]);valid[i]=False
            corrections.append({'sample':i,'episode':int(ep),'frame':int(arrays['frame_index'][i]),
                'validation':bool(arrays['is_validation'][i]),'was_valid':bool(arrays['valid_decode'][i]),
                'coefficient_positions':np.argwhere(codepoints[offset]==0xfffd).tolist(),
                'max_abs_base':float(np.abs(arrays['base_actions'][i]).max())})
    metrics={}
    residual=(arrays['target_actions'][...,:6].astype(float)-arrays['base_actions'][...,:6])/np.array(s['motion_scales'])
    for split in [False,True]:
        metrics['validation' if split else 'train']={}
        for label,mask in [('before',arrays['valid_decode']),('after',valid)]:
            ix=mask&(arrays['is_validation']==split);weight=m[ix,...,None]
            metrics['validation' if split else 'train'][label]={'samples':int(ix.sum()),
                'base_motion_mse':float((residual[ix]**2*weight).sum()/(6*weight.sum()))}
    report={'source_cache':str(a.cache.resolve()),'source_manifest_sha256':hashlib.sha256(raw).hexdigest(),
        'rule':'inverse-output-transform/DCT reconstructs U+FFFD; not residual/error magnitude filtering',
        'limitations':'Original token IDs absent; no claim of complete strict token validation of historical cache.',
        'affected':corrections,'metrics':metrics,'changes':'valid_decode mask only; features, targets, predictions, scales and splits unchanged'}
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(report,indent=2)+'\n')
    if a.output:
        if a.output.exists():raise ValueError('Output exists; choose a new cache directory')
        a.output.mkdir(parents=True)
        for f in a.cache.iterdir():
            if f.is_file() and f.name not in ['manifest.json','valid_decode.npy']:shutil.copy2(f,a.output/f.name)
        np.save(a.output/'valid_decode.npy',valid)
        manifest['signature']['cache_decode_audit']={'source_manifest_sha256':report['source_manifest_sha256'],
            'rule':report['rule'],'excluded_samples':[x['sample'] for x in corrections]}
        manifest['valid_samples']=int(valid.sum())
        (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        (a.output/'decode_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
