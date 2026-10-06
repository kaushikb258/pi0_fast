"""Configure a fresh source checkout before downloading or training.

No environments, weights, or system settings are modified. Preview by default.
"""
import argparse,json,re
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--conda-base',type=Path,required=True)
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    conda=args.conda_base.expanduser().resolve()
    for path in (root,conda):
        if not re.fullmatch(r'/[A-Za-z0-9_./-]+',str(path)):
            parser.error('Use paths without spaces, quotes, or shell metacharacters')
    if not (conda/'etc/profile.d/conda.sh').is_file():
        parser.error('No conda.sh in the specified Conda installation')
    state=root/'.local-paths.json'
    desired={'project_root':str(root),'conda_base':str(conda)}
    if state.exists():
        if json.loads(state.read_text()) != desired:
            parser.error('Already configured elsewhere; use a fresh checkout before caching/training')
        print('Already configured:',root);return
    for name in ('checkpoints','cache'):
        if (root/name).exists() and any((root/name).iterdir()):
            parser.error('Configure a fresh checkout before creating checkpoints or caches; existing heads have path/hash guards')
    paths=list(root.glob('*.py'))+list(root.glob('*.sh'))+list((root/'openpi').glob('src/**/*.py'))
    changes=[]
    for path in paths:
        old=path.read_text()
        new=old.replace('/home/kb/pi0_fast',str(root)).replace('/home/kb/miniconda3',str(conda))
        if old!=new:changes.append((path,new))
    print('Project:',root,'\nConda:',conda,'\nSource files to configure:',len(changes))
    if not args.apply:
        print('Preview only. Add --apply before installing, downloading or caching.');return
    for path,new in changes:path.write_text(new)
    base=root/'openpi/third_party/libero/libero/libero'
    config={'benchmark_root':str(base),'bddl_files':str(base/'bddl_files'),
            'init_states':str(base/'init_files'),'assets':str(base/'assets'),
            'datasets':str(root/'datasets/libero_raw')}
    (root/'libero_config').mkdir(exist_ok=True)
    (root/'libero_config/config.yaml').write_text(json.dumps(config,indent=2)+'\n')
    state.write_text(json.dumps(desired,indent=2)+'\n')
    print('Configured. Source hash changes are expected: create new controller caches/heads here.')

if __name__=='__main__':main()
