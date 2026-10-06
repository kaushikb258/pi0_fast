"""Seeded Square D0 policy evaluation; no training. Simulator lives in mimicgen env."""
import argparse,collections,json,time,textwrap
from pathlib import Path
import imageio,numpy as np
from openpi_client.websocket_client_policy import WebsocketClientPolicy
from mimicgen_square_common import CONFIG,REPO_ID,PROMPT,make_env,state_from_obs
from action_ema import ActionEMA,validate_lambda,action_change_metrics

def print_box(summary,out):
    lines=['MIMICGEN SQUARE D0 EVALUATION COMPLETE',
      'Method: {} | lambda: {}'.format(summary['method'],summary['action_ema_lambda']),
      'SUCCESS RATE: {:.1%} ({}/{})'.format(summary['success_rate'],summary['successes'],len(summary['episodes'])),
      'Valid action chunks: {:.2%} | Invalid: {}/{}'.format(summary['valid_action_fraction'],summary['invalid_action_chunks'],summary['policy_calls']),
      'Checkpoint: '+summary['checkpoint'],'Results and videos: '+str(out)]
    if summary.get('residual_controller'):
      lines.insert(2,'Residual controller: {} | beta: {}'.format(summary['residual_controller']['controller'],summary['residual_controller']['beta']))
    width=90;border='+'+'-'*(width+2)+'+';print('\n'+border)
    for line in lines:
      for part in textwrap.wrap(line,width=width,break_on_hyphens=False):print('| '+part.ljust(width)+' |')
    print(border,flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8012)
    parser.add_argument('--episodes',type=int,default=100)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--action-ema-lambda',type=validate_lambda,default=None)
    args=parser.parse_args()
    if not 1<=args.episodes<=1000:parser.error('episodes must be 1..1000')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    client=WebsocketClientPolicy('127.0.0.1',args.port);metadata=client.get_server_metadata()
    assert metadata['config']==CONFIG and metadata['repo_id']==REPO_ID,metadata
    env=make_env()
    summary={'suite':'mimicgen_square_d0','checkpoint':metadata['checkpoint'],'training_repo_id':REPO_ID,
      'method':('baseline' if args.action_ema_lambda is None else 'action-ema'),'action_ema_lambda':args.action_ema_lambda,
      'prompt':PROMPT,'protocol':'Fresh Square_D0 resets; episode seed=100000+index; 400 control steps; 20Hz; horizon10, execute5; greedy policy; no dummy settling steps; upstream decode fallback retained.',
      'episodes':[]}
    if metadata.get('residual_controller'):
      summary['residual_controller']=metadata['residual_controller']
      summary['method']='residual-mlp+'+summary['method']
    try:
      for episode in range(args.episodes):
        seed=100000+episode;np.random.seed(seed);obs=env.reset()
        initial=env.get_state()
        np.save(args.output_dir/f'episode_{episode}_initial_state.npy',initial['states'])
        (args.output_dir/f'episode_{episode}_model.xml').write_text(initial['model'])
        plan=collections.deque();frames=[];raw_actions=[];actions=[];states=[];chunks=[];success=False;error=None
        ema=ActionEMA(0 if args.action_ema_lambda is None else args.action_ema_lambda)
        lambda_history=[]
        started=time.monotonic()
        try:
          for step in range(400):
            frames.append(np.concatenate([obs['agentview_image'],obs['robot0_eye_in_hand_image']],axis=1))
            state=state_from_obs(obs);states.append(state)
            if not plan:
              result=client.infer({'observation/image':np.ascontiguousarray(obs['agentview_image']),
                'observation/wrist_image':np.ascontiguousarray(obs['robot0_eye_in_hand_image']),
                'observation/state':state,'prompt':PROMPT})
              chunk=np.asarray(result['actions']);assert chunk.shape==(10,7) and np.isfinite(chunk).all()
              chunks.append({'control_step':step,**result['action_decode']});plan.extend(chunk[:5])
            raw=plan.popleft()
            action=ema.apply(raw)
            lambda_history.append(ema.last_lambdas.copy())
            raw_actions.append(raw.copy());actions.append(action.copy())
            obs,_,_,_=env.step(action)
            success=bool(env.is_success()['task'])
            if success:break
        except Exception as exc:error=repr(exc)
        video=args.output_dir/f'episode_{episode}_{"success" if success else "failure"}.mp4'
        if frames:imageio.mimwrite(video,frames,fps=20,macro_block_size=1)
        for name,values in [('actions',actions),('raw_actions',raw_actions),('states',states),('lambdas',lambda_history)]:
          np.save(args.output_dir/f'episode_{episode}_{name}.npy',np.asarray(values))
        row={'episode':episode,'seed':seed,'success':success,'control_steps':len(actions),'seconds':time.monotonic()-started,
          'video':str(video),'error':error,'policy_calls':len(chunks),'invalid_action_chunks':sum(not c['valid'] for c in chunks),
          'chunks':chunks,'action_change_metrics':{'raw':action_change_metrics(raw_actions),'executed':action_change_metrics(actions)}}
        summary['episodes'].append(row);summary['successes']=sum(e['success'] for e in summary['episodes'])
        summary['success_rate']=summary['successes']/len(summary['episodes'])
        summary['policy_calls']=sum(e['policy_calls'] for e in summary['episodes'])
        summary['invalid_action_chunks']=sum(e['invalid_action_chunks'] for e in summary['episodes'])
        summary['valid_action_fraction']=1-summary['invalid_action_chunks']/max(summary['policy_calls'],1)
        (args.output_dir/'result.json').write_text(json.dumps(summary,indent=2))
        print(json.dumps({k:v for k,v in row.items() if k not in ['chunks']}),flush=True)
        if error:raise RuntimeError(error)
      print_box(summary,args.output_dir)
    finally:
      env.env.close();client._ws.close()

if __name__=='__main__':main()
