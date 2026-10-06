"""Download a pinned official OpenPI-ready LIBERO snapshot; never starts training."""
import json, pathlib, time, logging
logging.getLogger("huggingface_hub.file_download").setLevel(logging.ERROR)
from huggingface_hub import HfApi, snapshot_download
repo='physical-intelligence/libero'
root=pathlib.Path('/home/kb/pi0_fast/datasets/lerobot')/repo
saved=pathlib.Path('/home/kb/pi0_fast/setup_logs/libero/dataset-manifest.json')
saved.parent.mkdir(parents=True,exist_ok=True)
revision=json.loads(saved.read_text())['revision'] if saved.exists() else '9dfa69510ea9e1613fc54112bc706444b686a231'
info=HfApi().dataset_info(repo, revision=revision, files_metadata=True)
files=[]
for f in info.siblings:
    files.append({'path': f.rfilename, 'size': f.size, 'sha256': f.lfs.sha256 if f.lfs else None})
manifest={'repo_id':repo,'revision':info.sha,'version_ref':'v2.0','files':files,'total_bytes':sum(f['size'] or 0 for f in files)}
pathlib.Path('/home/kb/pi0_fast/setup_logs/libero/dataset-manifest.json').write_text(json.dumps(manifest,indent=2))
print('Pinned revision:',info.sha,'files:',len(files),'bytes:',manifest['total_bytes'],'destination:',root,flush=True)
for attempt in range(1,6):
    try:
        snapshot_download(repo,repo_type='dataset',revision=info.sha,local_dir=root,max_workers=8)
        break
    except Exception as exc:
        if attempt==5: raise
        print('Resuming interrupted download:',type(exc).__name__, 'attempt',attempt,flush=True)
        time.sleep(5)
for f in files:
    p=root/f['path']
    assert p.is_file() and p.stat().st_size==f['size'],str(p)
print('DATASET DOWNLOAD COMPLETE; all expected file sizes match.',flush=True)
