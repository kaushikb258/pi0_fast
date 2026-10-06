"""Download the public official checkpoint and verify each object's GCS MD5."""
import base64, concurrent.futures, hashlib, json, pathlib, urllib.parse, urllib.request
root = pathlib.Path('/home/kb/pi0_fast/cache/openpi/openpi-assets')
pathlib.Path('/home/kb/pi0_fast/setup_logs').mkdir(parents=True, exist_ok=True)
prefix = 'checkpoints/pi0_fast_base/'
url = 'https://storage.googleapis.com/storage/v1/b/openpi-assets/o?prefix=' + urllib.parse.quote(prefix)
items = []
while url:
    with urllib.request.urlopen(url) as r: page = json.load(r)
    items.extend(page.get('items', []))
    token = page.get('nextPageToken')
    url = ('https://storage.googleapis.com/storage/v1/b/openpi-assets/o?prefix=' + urllib.parse.quote(prefix) + '&pageToken=' + urllib.parse.quote(token)) if token else None
pathlib.Path('/home/kb/pi0_fast/setup_logs/checkpoint-manifest.json').write_text(json.dumps(items, indent=2))
def fetch(item):
    path = root / item['name']
    path.parent.mkdir(parents=True, exist_ok=True)
    def valid(p):
        if not p.exists() or p.stat().st_size != int(item['size']): return False
        h = hashlib.md5()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(8*1024*1024), b''): h.update(b)
        return base64.b64encode(h.digest()).decode() == item['md5Hash']
    if valid(path): return str(path)
    temp = path.with_name(path.name + '.partial')
    req = 'https://storage.googleapis.com/openpi-assets/' + urllib.parse.quote(item['name'], safe='/')
    with urllib.request.urlopen(req, timeout=120) as src, temp.open('wb') as dst:
        while b := src.read(8*1024*1024): dst.write(b)
    assert valid(temp), f'Checksum mismatch: {path}'
    temp.replace(path)
    print('Verified:', item['name'], item['size'], flush=True)
    return str(path)
print('Downloading', len(items), 'objects;', sum(int(i['size']) for i in items), 'bytes', flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(fetch, items))
print('CHECKPOINT DOWNLOAD VERIFIED', flush=True)
