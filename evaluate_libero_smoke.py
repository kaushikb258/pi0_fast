"""Small simulator evaluation using upstream observation/controller conventions."""
import argparse
import collections
import json
from pathlib import Path
import sys
import time
import imageio
import numpy as np
from action_ema import ActionEMA, action_change_metrics, validate_lambda
sys.path.insert(0, '/home/kb/pi0_fast/openpi')
from examples.libero.main import _get_libero_env, _quat2axisangle, LIBERO_DUMMY_ACTION
from libero.libero import benchmark
from openpi_client import image_tools
from openpi_client.websocket_client_policy import WebsocketClientPolicy

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8011)
    parser.add_argument('--suite',default='libero_spatial')
    parser.add_argument('--task-id',type=int,default=0)
    parser.add_argument('--episodes',type=int,default=3)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--action-ema-lambda', type=validate_lambda, default=None)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    np.random.seed(7)
    suite=benchmark.get_benchmark_dict()[args.suite]()
    task=suite.get_task(args.task_id)
    initial_states=suite.get_task_init_states(args.task_id)
    assert 0 < args.episodes <= len(initial_states)
    max_steps={'libero_spatial':220,'libero_object':280,'libero_goal':300,'libero_10':520}[args.suite]
    client=WebsocketClientPolicy('127.0.0.1',args.port)
    env,prompt=_get_libero_env(task,256,7)
    summary={'suite':args.suite,'task_id':args.task_id,'prompt':prompt,'seed':7,
             'checkpoint':client.get_server_metadata()['checkpoint'],'replan_steps':5,
             'max_control_steps':max_steps,'episodes':[],
             'method':'action-ema' if args.action_ema_lambda is not None else 'baseline',
             'action_ema_lambda':args.action_ema_lambda,
             'smoothing_scope':'final motion commands 0:6; gripper unchanged; first action unchanged; reset per episode',
             'note':'Small same-task diagnostic, not full benchmark; official decoding fallback retained and counted.'}
    metadata=client.get_server_metadata()
    summary['decode_validation']=metadata.get('decode_validation','legacy-shape-only')
    if 'residual_controller' in metadata:
        summary['residual_controller']=metadata['residual_controller']
        summary['method']='residual-mlp'+('+action-ema' if args.action_ema_lambda is not None else '')
    try:
        for episode in range(args.episodes):
            env.reset()
            obs=env.set_init_state(initial_states[episode])
            for _ in range(10):
                obs,_,_,_=env.step(LIBERO_DUMMY_ACTION)
            plan=collections.deque()
            frames=[]
            actions=[]
            raw_actions=[]
            smoother=ActionEMA(args.action_ema_lambda) if args.action_ema_lambda is not None else None
            chunks=[]
            success=False
            error=None
            start=time.monotonic()
            try:
                for step in range(max_steps):
                    image=np.ascontiguousarray(obs['agentview_image'][::-1,::-1])
                    wrist=np.ascontiguousarray(obs['robot0_eye_in_hand_image'][::-1,::-1])
                    frames.append(np.concatenate([image,wrist],axis=1))
                    if not plan:
                        result=client.infer({
                            'observation/image':image_tools.convert_to_uint8(image_tools.resize_with_pad(image,224,224)),
                            'observation/wrist_image':image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist,224,224)),
                            'observation/state':np.concatenate([obs['robot0_eef_pos'],_quat2axisangle(obs['robot0_eef_quat'].copy()),obs['robot0_gripper_qpos']]),
                            'prompt':prompt})
                        chunk=np.asarray(result['actions'])
                        assert chunk.shape==(10,7) and np.isfinite(chunk).all()
                        chunks.append({'control_step':step,**result['action_decode']})
                        plan.extend(chunk[:5])
                    raw_action=plan.popleft()
                    action=smoother.apply(raw_action) if smoother is not None else raw_action.copy()
                    raw_actions.append(raw_action.copy())
                    actions.append(action)
                    obs,_,done,_=env.step(action.tolist())
                    if done:
                        success=True
                        break
            except Exception as exc:
                error=repr(exc)
            video=args.output_dir/f'episode_{episode}_{"success" if success else "failure"}.mp4'
            if frames:
                # One frame per environment action, at the environment's 20 Hz control rate.
                imageio.mimwrite(video,frames,fps=20)
            np.save(args.output_dir/f'episode_{episode}_actions.npy',np.asarray(actions))
            np.save(args.output_dir/f'episode_{episode}_raw_actions.npy',np.asarray(raw_actions))
            row={'episode':episode,'initial_state_index':episode,'success':success,'control_steps':len(actions),
                 'policy_calls':len(chunks),'invalid_action_chunks':sum(not r['valid'] for r in chunks),
                 'seconds':time.monotonic()-start,'video':str(video),'error':error,'chunks':chunks,
                 'action_change_metrics':{'raw':action_change_metrics(raw_actions),
                                          'executed':action_change_metrics(actions)}}
            summary['episodes'].append(row)
            summary['successes']=sum(r['success'] for r in summary['episodes'])
            summary['success_rate']=summary['successes']/len(summary['episodes'])
            summary['policy_calls']=sum(r['policy_calls'] for r in summary['episodes'])
            summary['invalid_action_chunks']=sum(r['invalid_action_chunks'] for r in summary['episodes'])
            (args.output_dir/'result.json').write_text(json.dumps(summary,indent=2))
            print(json.dumps({k:v for k,v in row.items() if k!='chunks'}),flush=True)
            if error:
                raise RuntimeError(error)
        print('EVALUATION_COMPLETE',summary['successes'],'/',len(summary['episodes']),flush=True)
    finally:
        env.close()
