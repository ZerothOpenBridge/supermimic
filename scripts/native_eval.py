"""Native Mini3 evaluation on pilot12 or dev316, never final-test model selection."""
import argparse,json
from pathlib import Path
from isaaclab.app import add_launcher_args,launch_simulation
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--library',choices=['pilot','dev316'],required=True);p.add_argument('--output',required=True);add_launcher_args(p);args=p.parse_args()
if Path(args.output).exists():raise FileExistsError(args.output)
import numpy as np,torch
from supermimic import networks
from supermimic.env import TrackingSim,make_sim
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import MDP
from supermimic.motion import MotionLib
from supermimic.networks import Policy
from supermimic.robot import default_joint_pos_list
from supermimic.rotations import yaw_mat
R=Path('/mnt/cpfs/workspace/yxq/supermimic');torch.manual_seed(0);dev='cuda:0';networks.AMP_DTYPE=None
kin=G1Kinematics(dev);motion=MotionLib.from_dir(str(R/('shared/data/mini3_v1/packed/'+args.library+'.pt')),kin,dev);C=len(motion.names)
mdp=MDP(kin,motion,torch.tensor(default_joint_pos_list(),device=dev));ck=torch.load(args.checkpoint,map_location=dev,weights_only=False)
pi=Policy(75,570,hist=10).to(dev);pi.load_state_dict(ck['pi'],strict=True);pi.eval();cfg=make_sim(dev,args)
with launch_simulation(cfg,args),torch.no_grad():
 env=TrackingSim(cfg,C,motion,mdp,eval_mode=True);ids=torch.arange(C,device=dev);env.reset(ids,motion.clip_start.clone());env.refresh()
 ret=torch.zeros(C,device=dev);steps=torch.zeros(C,device=dev);local=torch.zeros(C,device=dev);hist=None
 while not bool(env.done_eval.all()):
  S=env.state();o=mdp.obs(S,env.a_prev,env.xi,env.f);hist=o[:,None,:].repeat(1,10,1) if hist is None else torch.cat((hist[:,1:],o[:,None,:]),1)
  a=pi(hist,motion.goal(env.f,10));alive=~env.done_eval.clone()
  # Evaluate current physical state against the current physical reference frame.
  actual=kin.forward(*S)[0];reference=motion.body_p[env.f];rp=motion.root_p[env.f];rR=motion.root_R[env.f]
  al=(yaw_mat(S[1]).transpose(-1,-2)[:,None]@(actual-S[0][:,None,:])[...,None]).squeeze(-1)
  rl=(yaw_mat(rR).transpose(-1,-2)[:,None]@(reference-rp[:,None,:])[...,None]).squeeze(-1)
  local+=torch.where(alive,(al-rl).abs().mean((1,2)),0.)
  out=env.step(a);assert torch.isfinite(out['r'][alive]).all();ret+=torch.where(alive,out['r'],0.);steps+=alive.float()
  env.f=torch.minimum(env.f,motion.clip_end-1)
  # Inactive worlds no longer provide evidence. Reset them to safe reference states.
  inactive=env.done_eval.nonzero().flatten()
  if len(inactive):env.reset(inactive,motion.clip_end[inactive]-1)
 failed=env.failed_eval.clone();progress=torch.clamp(steps/(motion.clip_len-1),max=1.)
 rows=[{'name':motion.names[i],'frames':int(motion.clip_len[i]),'steps':int(steps[i]),'failed':bool(failed[i]),'progress':float(progress[i]),'norm_return':float(ret[i]/(steps[i].clamp_min(1)*.1)),
        'full_clip_norm_return':float(ret[i]/((motion.clip_len[i]-1)*.1)),'local_kpt_mae':float(local[i]/steps[i].clamp_min(1))} for i in range(C)]
 summary={'clips':C,'failed':int(failed.sum()),'completion_rate':float((~failed).float().mean()),'mean_progress':float(progress.mean()),'mean_local_kpt_mae':float((local/steps.clamp_min(1)).mean()),
          'mean_norm_return':float((ret/(steps.clamp_min(1)*.1)).mean()),'mean_full_clip_norm_return':float((ret/((motion.clip_len-1)*.1)).mean())}
 result={'summary':summary,'clips':rows,'checkpoint':args.checkpoint,'library':args.library,'policy_precision':'FP32 evaluation; native training may use bf16 MLP matmuls','iteration':ck.get('iteration',0),'sim_transitions':ck.get('sim_steps')}
 assert all(np.isfinite(v) for v in summary.values()),summary
 assert all(all(np.isfinite(x[k]) for k in ['progress','norm_return','local_kpt_mae']) for x in rows)
 assert len({x['name'] for x in rows})==C;Path(args.output).write_text(json.dumps(result,indent=2));print(json.dumps(summary),flush=True)
