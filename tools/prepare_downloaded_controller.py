"""Validate downloaded weights and path-only source relocation, then bind a copy of a controller.

Original best.pt is never rewritten. Does not change any tensor or disable source checks.
Run in the configured pi0 environment after downloading this complete model repository.
"""
import argparse,hashlib,json
from pathlib import Path
import torch

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def bind(release,project,dataset):
    release=Path(release).resolve();project=Path(project).resolve()
    manifest=json.loads((release/'release_manifest.json').read_text())
    provenance=manifest['controller_source_provenance']
    # Source must be byte-identical to the published implementation except configured root paths.
    settings=project/'.local-paths.json'
    local=json.loads(settings.read_text()) if settings.exists() else {'project_root':str(project),'conda_base':'/home/kb/miniconda3'}
    relevant=provenance['libero_files'] if dataset=='libero' else provenance['mimicgen_files']
    for name in relevant:
        source=project/name
        if not source.is_file():raise ValueError(f'Missing required source: {name}')
        text=source.read_text().replace(local['project_root'],provenance['original_project']).replace(local['conda_base'],provenance['original_conda'])
        if hashlib.sha256(text.encode()).hexdigest()!=provenance['source_sha256'][name]:
            raise ValueError(f'Source changed beyond path configuration: {name}; use the matching release code')
    # Verify all selected inference weights, norm assets and controller before rebinding.
    for entry in manifest['files']:
        if entry['path'].startswith(dataset+'/'):
            path=release/entry['path']
            if not path.is_file() or path.stat().st_size!=entry['bytes'] or sha(path)!=entry['sha256']:
                raise ValueError(f'Artifact checksum mismatch: {entry["path"]}')
    original=release/dataset/'controller/best.pt'
    destination=release/dataset/'controller/best.local.pt'
    if destination.exists():raise ValueError(f'Output exists; not overwriting: {destination}')
    bundle=torch.load(original,map_location='cpu',weights_only=True)
    before={k:v.clone() for k,v in bundle['state_dict'].items()}
    old_path=bundle['base_checkpoint'];old_hash=bundle['feature_code_hash']
    # Match the exact formulas in residual_policy.py and libero_residual_policy.py.
    digest=hashlib.sha256()
    for name in provenance['feature_hash_files']:digest.update((project/name).read_bytes())
    new_hash=digest.hexdigest()
    if dataset=='libero':
        digest=hashlib.sha256(new_hash.encode())
        for name in ['libero_residual_policy.py','libero_residual_data.py']:digest.update((project/name).read_bytes())
        digest.update(b'pi0_fast_libero_spatial_lora');new_hash=digest.hexdigest()
    expected=manifest['artifacts'][dataset]['controller']
    if old_path!=expected['original_base_checkpoint'] or old_hash!=expected['original_feature_code_hash']:
        raise ValueError('Controller provenance does not match release manifest')
    bundle['base_checkpoint']=str(release/dataset/'vla');bundle['feature_code_hash']=new_hash
    bundle['relocation']={'original_base_checkpoint':old_path,'original_feature_code_hash':old_hash,
                          'release_manifest_sha256':sha(release/'release_manifest.json'),'tensor_changes':False}
    temporary=destination.with_suffix('.tmp')
    torch.save(bundle,temporary)
    saved=torch.load(temporary,map_location='cpu',weights_only=True)
    if not all(torch.equal(v,saved['state_dict'][k]) for k,v in before.items()):
        temporary.unlink();raise AssertionError('Tensor changed during metadata migration')
    temporary.replace(destination)
    print('Bound controller (weights unchanged):',destination)
    print('VLA checkpoint:',bundle['base_checkpoint'])
    return destination

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--project',type=Path,required=True)
    p.add_argument('--dataset',choices=['libero','mimicgen'],required=True)
    a=p.parse_args();bind(a.release,a.project,a.dataset)
