"""Evaluate all ten Spatial tasks, verifying they match the training subset."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil
import textwrap
from action_ema import validate_lambda
from libero.libero import benchmark
from openpi_client.websocket_client_policy import WebsocketClientPolicy

ROOT=Path('/home/kb/pi0_fast')


def print_summary_box(summary, output_dir):
    """Readable terminal summary; result.json remains the machine-readable record."""
    width = max(40, min(100, shutil.get_terminal_size((100, 24)).columns) - 4)
    calls = summary['policy_calls']
    invalid = summary['invalid_action_chunks']
    lines = [
        'LIBERO-SPATIAL EVALUATION COMPLETE',
        'Method: {} | lambda: {}'.format(summary.get('method', 'baseline'),
                                         summary.get('action_ema_lambda', 'off')),
        '',
        'Tasks: {} | Episodes per task: {} | Total episodes: {}'.format(
            len(summary['tasks']), summary['episodes_per_task'], summary['episodes_completed']),
        'SUCCESS RATE: {:.1%}  ({}/{} episodes)'.format(
            summary['success_rate'], summary['successes'], summary['episodes_completed']),
        'Valid action chunks: {:.2%}  ({}/{})'.format(
            summary['valid_action_fraction'], calls - invalid, calls),
        'Invalid action chunks: {}'.format(invalid),
        '',
        'Per-task successes:',
    ]
    if summary.get('residual_controller'):
        lines.extend(['Controller: '+summary['residual_controller']['controller'],
                      'Residual beta: '+str(summary['residual_controller']['beta'])])
    for task in summary['tasks']:
        lines.append('  Task {:02d}: {}/{} ({:.1%})'.format(
            task['task_id'], task['successes'], task['episodes'], task['success_rate']))
    lines.extend(['', 'Checkpoint:', str(summary['checkpoint']), '',
                  'Results and videos:', str(output_dir),
                  'Metrics: ' + str(Path(output_dir) / 'result.json')])
    border = '+' + '-' * (width + 2) + '+'
    rendered = [border]
    for line in lines:
        for part in textwrap.wrap(line, width=width, break_long_words=True,
                                  break_on_hyphens=False) or ['']:
            rendered.append('| ' + part.ljust(width) + ' |')
    rendered.append(border)
    print('\n' + '\n'.join(rendered) + '\n', flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8011)
    parser.add_argument('--episodes-per-task',type=int,default=10)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--action-ema-lambda', type=validate_lambda, default=None)
    args=parser.parse_args()
    manifest=json.loads((ROOT/'datasets/lerobot/local/libero_spatial/subset_manifest.json').read_text())
    suite=benchmark.get_benchmark_dict()['libero_spatial']()
    assert suite.n_tasks==10
    assert [suite.get_task(i).language for i in range(10)]==[t['prompt'] for t in manifest['tasks']]
    assert 1<=args.episodes_per_task<=50
    client=WebsocketClientPolicy('127.0.0.1',args.port)
    metadata=client.get_server_metadata()
    client._ws.close()
    assert metadata['config']=='pi0_fast_libero_spatial_lora',metadata
    assert metadata['repo_id']=='local/libero_spatial',metadata
    args.output_dir.mkdir(parents=True,exist_ok=True)
    summary={'suite':'libero_spatial','checkpoint':metadata['checkpoint'],
             'training_repo_id':metadata['repo_id'],'all_training_tasks_matched':True,
             'episodes_per_task':args.episodes_per_task,'tasks':[],
             'method':'action-ema' if args.action_ema_lambda is not None else 'baseline',
             'action_ema_lambda':args.action_ema_lambda,
             'smoothing_scope':'final motion commands 0:6; gripper unchanged; first action unchanged; reset per episode',
             'protocol':'Seed 7; first N standard initial states per task; 220 control steps; replan every 5; upstream decode fallback counted.'}
    summary['decode_validation']=metadata.get('decode_validation','legacy-shape-only')
    if 'residual_controller' in metadata:
        summary['residual_controller']=metadata['residual_controller']
        summary['method']='residual-mlp'+('+action-ema' if args.action_ema_lambda is not None else '')
    method_args=[] if args.action_ema_lambda is None else ['--action-ema-lambda',str(args.action_ema_lambda)]
    for task_id in range(10):
        task_dir=args.output_dir/f'task_{task_id:02d}'
        print(f'Evaluating Spatial task {task_id}/9: {suite.get_task(task_id).language}',flush=True)
        with (args.output_dir/f'task_{task_id:02d}.log').open('w') as log:
            completed=subprocess.run([sys.executable,str(ROOT/'evaluate_libero_smoke.py'),
                '--port',str(args.port),'--suite','libero_spatial','--task-id',str(task_id),
                '--episodes',str(args.episodes_per_task),'--output-dir',str(task_dir),*method_args],stdout=log,stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError(f'Task {task_id} evaluation execution failed; inspect {log.name}')
        result=json.loads((task_dir/'result.json').read_text())
        row={'task_id':task_id,'prompt':result['prompt'],'episodes':len(result['episodes']),
             'successes':result['successes'],'success_rate':result['success_rate'],
             'policy_calls':result['policy_calls'],'invalid_action_chunks':result['invalid_action_chunks']}
        summary['tasks'].append(row)
        summary['episodes_completed']=sum(x['episodes'] for x in summary['tasks'])
        summary['successes']=sum(x['successes'] for x in summary['tasks'])
        summary['success_rate']=summary['successes']/summary['episodes_completed']
        summary['policy_calls']=sum(x['policy_calls'] for x in summary['tasks'])
        summary['invalid_action_chunks']=sum(x['invalid_action_chunks'] for x in summary['tasks'])
        summary['valid_action_fraction']=1-summary['invalid_action_chunks']/max(summary['policy_calls'],1)
        (args.output_dir/'result.json').write_text(json.dumps(summary,indent=2))
        print(json.dumps(row),flush=True)
    print('ALL_10_SPATIAL_TASKS_COMPLETE',json.dumps({k:v for k,v in summary.items() if k!='tasks'}),flush=True)
    print_summary_box(summary, args.output_dir)
