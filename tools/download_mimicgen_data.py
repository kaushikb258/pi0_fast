"""Download the pinned official Square D0 HDF5 and verify its SHA-256."""
import hashlib,json
from pathlib import Path
from huggingface_hub import hf_hub_download

def main():
    root=Path(__file__).resolve().parents[1]/'datasets/mimicgen'
    manifest={'repo':'amandlek/mimicgen_datasets',
        'revision':'33016f8a62c02334f929f2913af8fdd2a8a129e1',
        'file':'core/square_d0.hdf5','size':1621351476,
        'sha256':'41fc24bce0f88343099c0b1b5bf6eee08cbc35851e71276d4509a01c9b75481c'}
    filename=hf_hub_download(repo_id=manifest['repo'],repo_type='dataset',
        filename=manifest['file'],revision=manifest['revision'],local_dir=root)
    digest=hashlib.sha256()
    with open(filename,'rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    if Path(filename).stat().st_size!=manifest['size'] or digest.hexdigest()!=manifest['sha256']:
        raise RuntimeError('Square D0 file size or SHA-256 mismatch')
    (root/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Verified Square D0:',filename)

if __name__=='__main__':main()
