"""Fetch exact upstream simulator source revisions, without installing packages."""
import argparse,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--dry-run',action='store_true')
a=p.parse_args()
root=Path(__file__).resolve().parents[1]
for dep in json.loads((root/'provenance/dependencies.json').read_text()):
 target=root/dep['path']
 print(dep['path'],dep['url'],dep['commit'],flush=True)
 if a.dry_run:continue
 if target.exists():
  try:head=subprocess.check_output(['git','-C',str(target),'rev-parse','HEAD'],text=True).strip()
  except subprocess.CalledProcessError:raise SystemExit(f'Existing non-Git destination: {target}')
  if head!=dep['commit']:raise SystemExit(f'Existing checkout differs: {target}; not modifying it')
  continue
 target.parent.mkdir(parents=True,exist_ok=True)
 subprocess.run(['git','clone','--no-checkout',dep['url'],str(target)],check=True)
 subprocess.run(['git','-C',str(target),'checkout','--detach',dep['commit']],check=True)
