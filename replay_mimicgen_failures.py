"""Replay saved commands headlessly and classify failure stages; no policy inference."""
import argparse,json,time,collections
from pathlib import Path
import numpy as np,h5py
from mimicgen_square_common import DATASET,state_from_obs

def make_headless():
 import mimicgen
 from robomimic.utils import obs_utils
 from robomimic.envs.env_robosuite import EnvRobosuite
 obs_utils.initialize_obs_utils_with_obs_specs({'obs':{'low_dim':['robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos']}})
 with h5py.File(DATASET,'r') as f:m=json.loads(f['data'].attrs['env_args'])
 return EnvRobosuite(env_name=m['env_name'],render=False,render_offscreen=False,use_image_obs=False,**m['env_kwargs'])

def sustained(mask,n=3):
 hits=np.convolve(np.asarray(mask,dtype=int),np.ones(n,dtype=int),'valid') if len(mask)>=n else []
 ix=np.flatnonzero(np.asarray(hits)==n)
 return int(ix[0]) if len(ix) else None

def classify(telemetry,success,lift=.04,near=.06):
 z=telemetry[:,2]-telemetry[0,2];grasp=telemetry[:,4]>0
 lifted=sustained((z>=lift)&grasp)
 if success:return 'success',lifted,None
 if lifted is None:
  return ('uncertain' if sustained(z>=lift) is not None else 'picking'),None,None
 reached=sustained((np.arange(len(z))>=lifted)&(telemetry[:,3]<=near)&(z>=.02))
 return ('placement_insertion' if reached is not None else 'transport'),lifted,reached

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--episode',type=int);a=p.parse_args()
 r=json.loads((a.run/'result.json').read_text());assert len(r['episodes'])==100,'Use completed evaluations only'
 selected=[e for e in r['episodes'] if not e['success']]+[e for e in r['episodes'] if e['success']][:3]
 if a.episode is not None:selected=[e for e in r['episodes'] if e['episode']==a.episode]
 a.output.mkdir(parents=True,exist_ok=True);env=make_headless();results=[]
 summary={'source':str(a.run),'checkpoint':r['checkpoint'],'definitions':{'lift':'Nut center >=4cm above initial height AND bilateral grasp contact for >=3 control samples (0.15s).','placement':'After lift, nut center within 6cm XY of correct peg and >=2cm above initial height for >=3 samples.','picking':'No confirmed sustained lift; lift without grasp is uncertain.','transport':'Confirmed lift but no sustained approach to peg; does not necessarily mean a drop.','placement_insertion':'Reached peg region but failed task; includes alignment, insertion or release failures.','validation':'Initial full simulator state, every saved 8D robot state (pre-action), and final success predicate; object trajectories were not saved originally.'},'episodes':results}
 try:
  for e in selected:
   i=e['episode'];start=time.monotonic();np.random.seed(e['seed']);obs=env.reset()
   initial=np.load(a.run/f'episode_{i}_initial_state.npy');initial_err=float(np.max(np.abs(env.get_state()['states']-initial)))
   actions=np.load(a.run/f'episode_{i}_actions.npy');states=np.load(a.run/f'episode_{i}_states.npy')
   assert len(actions)==len(states)==e['control_steps']
   sim=env.env;nut=sim.nuts[sim.nut_id];body=sim.obj_body_id[nut.name];peg=sim.peg1_body_id if sim.nut_id==0 else sim.peg2_body_id
   def measure():
    pos=sim.sim.data.body_xpos[body].copy();pegpos=sim.sim.data.body_xpos[peg]
    grasp=sim._check_grasp(gripper=sim.robots[0].gripper,object_geoms=nut.contact_geoms)
    return np.r_[pos,np.linalg.norm(pos[:2]-pegpos[:2]),float(grasp)]
   telemetry=[measure()];max_error=0.
   for t,action in enumerate(actions):
    max_error=max(max_error,float(np.max(np.abs(state_from_obs(obs)-states[t]))))
    obs,_,_,_=env.step(action);telemetry.append(measure())
   success=bool(env.is_success()['task']);v=np.array(telemetry)
   valid=initial_err<=1e-8 and max_error<=1e-4 and success==e['success']
   category,lifted,near=classify(v,success) if valid else ('uncertain_replay',None,None)
   sensitivity={f'lift_{lift}_near_{near}':classify(v,success,lift,near)[0] for lift in [.03,.04,.05] for near in [.04,.06,.08]} if valid else {}
   row={'episode':i,'recorded_success':e['success'],'replayed_success':success,'replay_valid':valid,'initial_state_max_error':initial_err,'robot_state_max_error':max_error,'category':category,'first_lift_sample':lifted,'first_near_peg_sample':near,'max_lift_m':float(np.max(v[:,2]-v[0,2])),'min_peg_xy_distance_m':float(v[:,3].min()),'threshold_sensitivity':sensitivity,'seconds':time.monotonic()-start}
   np.save(a.output/f'episode_{i}_telemetry.npy',v);results.append(row)
   summary['failure_counts']=dict(collections.Counter(x['category'] for x in results if not x['recorded_success']))
   (a.output/'result.json').write_text(json.dumps(summary,indent=2));print(json.dumps(row),flush=True)
   if not valid:raise RuntimeError('Replay mismatch; stop before interpreting remaining episodes')
 finally:env.env.close()
 print('COMPLETE',summary['failure_counts'],flush=True)
if __name__=='__main__':main()
