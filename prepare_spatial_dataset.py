"""Extract all Spatial demonstrations into a standalone, reindexed LeRobot v2 dataset."""
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import io
import json
import math
from pathlib import Path
import time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

ROOT=Path('/home/kb/pi0_fast')
SOURCE=ROOT/'datasets/lerobot/physical-intelligence/libero'
DEST=ROOT/'datasets/lerobot/local/libero_spatial'
LOG=ROOT/'setup_logs/spatial'

def main():
    assert not DEST.exists(), f'Destination already exists: {DEST}'
    names=ast.literal_eval(ast.parse((ROOT/'openpi/third_party/libero/libero/libero/benchmark/libero_suite_task_map.py').read_text()).body[0].value)['libero_spatial']
    prompts=[name.replace('_',' ') for name in names]
    task_ids={p:i for i,p in enumerate(prompts)}
    tasks={r['task_index']:r['task'] for r in map(json.loads,(SOURCE/'meta/tasks.jsonl').read_text().splitlines())}
    selected=[]
    for e in map(json.loads,(SOURCE/'meta/episodes.jsonl').read_text().splitlines()):
        if any(t in task_ids for t in e['tasks']):
            assert len(e['tasks'])==1 and e['tasks'][0] in task_ids
            selected.append(e)
    assert len(selected)==432
    offsets=np.cumsum([0]+[e['length'] for e in selected]).tolist()
    (DEST/'meta').mkdir(parents=True)
    (DEST/'data/chunk-000').mkdir(parents=True)
    def convert(pair):
        i,e=pair
        old=e['episode_index']
        source=SOURCE/f'data/chunk-{old//1000:03d}/episode_{old:06d}.parquet'
        table=pq.read_table(source)
        assert table.num_rows==e['length']
        old_task=table['task_index'].to_numpy()
        assert len(set(old_task))==1 and tasks[int(old_task[0])]==e['tasks'][0]
        new_task=task_ids[e['tasks'][0]]
        for name,values in {
            'episode_index':np.full(table.num_rows,i,dtype=np.int64),
            'index':np.arange(offsets[i],offsets[i+1],dtype=np.int64),
            'task_index':np.full(table.num_rows,new_task,dtype=np.int64),
        }.items():
            table=table.set_column(table.schema.get_field_index(name),table.schema.field(name),pa.array(values))
        numeric={}
        for name in ['state','actions','timestamp','frame_index','episode_index','index','task_index']:
            arr=np.asarray(table[name].to_pylist(),dtype=np.float64).reshape(table.num_rows,-1)
            assert np.isfinite(arr).all()
            numeric[name]={'count':len(arr),'sum':arr.sum(axis=0),'sumsq':(arr*arr).sum(axis=0),'min':arr.min(axis=0),'max':arr.max(axis=0)}
        histograms={}
        for name in ['image','wrist_image']:
            hist=np.zeros((3,256),dtype=np.int64)
            for item in table[name].to_pylist():
                with Image.open(io.BytesIO(item['bytes'])) as image:
                    assert image.size==(256,256)
                    hist+=np.asarray(image.convert('RGB').histogram()).reshape(3,256)
            histograms[name]=hist
        destination=DEST/f'data/chunk-000/episode_{i:06d}.parquet'
        pq.write_table(table,destination,compression='snappy')
        # Compare all immutable payload columns after writing; image bytes are preserved exactly.
        copied=pq.read_table(destination)
        for name in ['image','wrist_image','state','actions','timestamp','frame_index']:
            assert table[name].equals(copied[name]),(i,name)
        assert int(copied['index'][0].as_py())==offsets[i]
        assert int(copied['index'][-1].as_py())==offsets[i+1]-1
        return numeric,histograms
    numeric_all={}
    hist_all={k:np.zeros((3,256),dtype=np.int64) for k in ['image','wrist_image']}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,(numeric,histograms) in enumerate(pool.map(convert,enumerate(selected))):
            for key,value in numeric.items():
                if key not in numeric_all:
                    numeric_all[key]=value
                else:
                    total=numeric_all[key]
                    for op in ['count','sum','sumsq']:
                        total[op]+=value[op]
                    total['min']=np.minimum(total['min'],value['min'])
                    total['max']=np.maximum(total['max'],value['max'])
            for key,value in histograms.items():
                hist_all[key]+=value
            if (i+1)%25==0:
                print(f'Extracted and verified {i+1}/{len(selected)} episodes',flush=True)
    stats={}
    for key,value in numeric_all.items():
        mean=value['sum']/value['count']
        std=np.sqrt(np.maximum(value['sumsq']/value['count']-mean**2,0))
        stats[key]={'mean':mean.tolist(),'std':std.tolist(),'min':value['min'].tolist(),'max':value['max'].tolist()}
    levels=np.arange(256)/255
    for key,hist in hist_all.items():
        count=hist.sum(axis=1)
        mean=(hist*levels).sum(axis=1)/count
        std=np.sqrt(np.maximum((hist*levels**2).sum(axis=1)/count-mean**2,0))
        mins=np.array([np.flatnonzero(h)[0]/255 for h in hist])
        maxs=np.array([np.flatnonzero(h)[-1]/255 for h in hist])
        stats[key]={k:v.reshape(3,1,1).tolist() for k,v in {'mean':mean,'std':std,'min':mins,'max':maxs}.items()}
    info=json.loads((SOURCE/'meta/info.json').read_text())
    info.update(total_episodes=len(selected),total_frames=offsets[-1],total_tasks=10,total_chunks=1,splits={'train':f'0:{len(selected)}'})
    (DEST/'meta/info.json').write_text(json.dumps(info,indent=2))
    (DEST/'meta/stats.json').write_text(json.dumps(stats,indent=2))
    (DEST/'meta/tasks.jsonl').write_text(''.join(json.dumps({'task_index':i,'task':p})+'\n' for i,p in enumerate(prompts)))
    (DEST/'meta/episodes.jsonl').write_text(''.join(json.dumps({**e,'episode_index':i})+'\n' for i,e in enumerate(selected)))
    counts=Counter(e['tasks'][0] for e in selected)
    manifest={'repo_id':'local/libero_spatial','source_repo_id':'physical-intelligence/libero','source_revision':'9dfa69510ea9e1613fc54112bc706444b686a231','episodes':len(selected),'frames':offsets[-1],'tasks':[{'spatial_task_id':i,'prompt':p,'episodes':counts[p]} for i,p in enumerate(prompts)],'episode_mapping':[{'source':e['episode_index'],'subset':i,'length':e['length']} for i,e in enumerate(selected)],'all_payloads_verified':True,'metadata_statistics':'recomputed from all subset frames and all image pixels','source_modified':False}
    (DEST/'subset_manifest.json').write_text(json.dumps(manifest,indent=2))
    (LOG/'dataset-result.json').write_text(json.dumps({k:v for k,v in manifest.items() if k!='episode_mapping'},indent=2))
    print(f'SPATIAL_DATASET_PASS episodes={len(selected)} frames={offsets[-1]} tasks=10',flush=True)

if __name__=='__main__':
    main()
