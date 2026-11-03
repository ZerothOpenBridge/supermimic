"""Strict CPU checkpoint -> merged FP32 ONNX package and executed producer/source parity."""
import argparse,hashlib,json,sys
import time as walltime
from pathlib import Path
import numpy as np,torch,yaml,onnx,onnxruntime as ort
p=argparse.ArgumentParser();p.add_argument('--backend',choices=['cuboi','zita'],required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);a=p.parse_args()
R=Path('/mnt/cpfs/workspace/yxq/supermimic');D=R/'shared/data/mini3_v1';sys.path.insert(0,str(D/'evaluator-frozen-v1/evaluator/src'));sys.path.insert(0,str(D/'contract-tooling/tools'))
from teleop_evaluator.policy.supermimic_contract import EXT,JOINTS,DEFAULT,descriptors,semantic_extension
from teleop_evaluator.policy.supermimic_features import Features,_quat_matrix_wxyz
from teleop_evaluator.policy.ufo import _quat_mul_wxyz
from teleop_evaluator.robot.base import RobotState
from onnx_contract import finalize_package,NoAliasDumper
out=Path(a.output);assert not out.exists();(out/'tests').mkdir(parents=True)
ckpath=Path(a.checkpoint);assert ckpath.is_file();ck=torch.load(ckpath,map_location='cpu',weights_only=False)
assert ck.get('iteration',0)>0,'Publishing requires a trained checkpoint'
if a.backend=='cuboi':
 from supermimic.policy.nets import Policy,PolicyCfg
 pc=ck['config']['policy'];policy=Policy(PolicyCfg(**{k:tuple(v) if isinstance(v,list) else v for k,v in pc.items()}));policy.load_state_dict(ck['policy'],strict=True)
 from supermimic.world_model.heads import analytic_observation,flatten_history_term_major
 from supermimic.envs.mdp.motion_library import MotionLibrary,goal_features,RefState
else:
 from supermimic import networks
 networks.AMP_DTYPE=None
 from supermimic.networks import Policy
 policy=Policy(75,570,hist=10);policy.load_state_dict(ck['pi'],strict=True)
 from supermimic.motion import MotionLib
 from supermimic.mdp import MDP
import supermimic
source_root=Path(next(iter(supermimic.__path__)))
source_files={str(x.relative_to(source_root)):hashlib.sha256(x.read_bytes()).hexdigest() for x in sorted(source_root.rglob('*.py')) if '__pycache__' not in x.parts}
source_identity=hashlib.sha256(json.dumps(source_files,sort_keys=True,separators=(',',':')).encode()).hexdigest()
policy=policy.float().eval();print('STRICT_LOAD',a.backend,len(policy.state_dict()),flush=True)
class Merged(torch.nn.Module):
 def __init__(self,source):super().__init__();self.source=source
 def forward(self,obs_hist,goal):return self.source(obs_hist.reshape(1,10,75),goal) if a.backend=='zita' else self.source(obs_hist,goal)
model=Merged(policy).eval();obs_width=690 if a.backend=='cuboi' else 750;goal_width=520 if a.backend=='cuboi' else 570
with torch.no_grad():torch.onnx.export(model,(torch.zeros(1,obs_width),torch.zeros(1,goal_width)),str(out/'policy.onnx'),input_names=['obs_hist','goal'],output_names=['raw_action'],opset_version=17,dynamo=False)
onnx.checker.check_model(str(out/'policy.onnx'))
report={'backend':a.backend,'checkpoint':str(ckpath),'iteration':ck.get('iteration',0),'strict_state_dict_keys':len(policy.state_dict()),'tolerance':{'absolute':1e-5,'relative':1e-5},'cases':[]}
feeds=[];source_outputs=[];producer_max=0.
# Three reset sequences:20 ordinary ticks,22 turning ticks,22 ticks approaching a clip boundary.
for label,ticks,N,turn in [('reset20',20,80,.2),('turning22',22,80,1.7),('boundary22',22,24,.5)]:
 time=np.arange(N)*.02;angle=turn*time;c,s=np.cos(angle/2),np.sin(angle/2)
 q=np.zeros((N,28));q[:,:3]=np.stack((time*.3,np.sin(time)*.04,.5+np.sin(time*2)*.01),axis=1);q[:,3]=c;q[:,6]=s
 roll=.15*np.sin(time*2);pitch=.12*np.cos(time*1.3);qr=np.stack((np.cos(roll/2),np.sin(roll/2),np.zeros(N),np.zeros(N)),axis=1);qp=np.stack((np.cos(pitch/2),np.zeros(N),np.sin(pitch/2),np.zeros(N)),axis=1);q[:,3:7]=_quat_mul_wxyz(q[:,3:7],_quat_mul_wxyz(qp,qr))
 q[:,7:]=np.asarray(DEFAULT)+.03*np.sin(time[:,None]*2+np.arange(21)[None]*.2)
 q=q.astype(np.float32)
 f=Features(a.backend,DEFAULT);f.reset(q)
 ref=torch.tensor(np.concatenate([f.reference,f.v,f.w,f.qd],axis=1),dtype=torch.float32)
 if a.backend=='zita':
  lib=MotionLib.__new__(MotionLib);lib.device='cpu';lib.clip_id=torch.zeros(N,dtype=torch.long);lib.clip_end=torch.tensor([N-1])
  lib.root_p=ref[:,:3];lib.root_R=torch.tensor(f.R,dtype=torch.float32);lib.root_v=ref[:,28:31];lib.root_w=ref[:,31:34];lib.q=ref[:,7:28];lib.qd=ref[:,34:]
  lib.root_v_local=(lib.root_R.transpose(-1,-2)@lib.root_v[...,None]).squeeze(-1);lib.root_w_local=(lib.root_R.transpose(-1,-2)@lib.root_w[...,None]).squeeze(-1)
  mdp=MDP.__new__(MDP);mdp.motion=lib;mdp.default_q=torch.tensor(DEFAULT)
 native_hist=[];fixture={};expected=[];raw_fixture={"reference_qpos":q.astype(np.float32),"robot_qpos":[],"robot_qvel":[],"previous_raw_action":[],"expected_observation":[],"expected_goal":[],"tick":[]}
 for tick in range(ticks):
  ii=min(tick,N-1);rq=f.reference[ii].copy();rq[7:]+=.005*np.cos(tick*.2+np.arange(21));rr=_quat_matrix_wxyz(rq[3:7]);qvel=np.concatenate((f.v[ii],rr.T@f.w[ii],f.qd[ii]))
  state=RobotState(rq.astype(np.float32),qvel.astype(np.float32),rq[:3].astype(np.float32),rq[3:7].astype(np.float32),rq[7:].astype(np.float32),qvel[6:].astype(np.float32))
  feed=f.assemble(state,tick);prev=torch.tensor(f.previous[None],dtype=torch.float32);xi=torch.zeros(1,25)
  if a.backend=='cuboi':
   st=torch.tensor(np.concatenate((state.qpos,state.qvel))[None],dtype=torch.float32);o=analytic_observation(st,prev,xi,torch.tensor(DEFAULT))
   native_hist=[o]*10 if not native_hist else native_hist[1:]+[o];native_obs=flatten_history_term_major(torch.stack(native_hist))
   idx=np.minimum(tick+np.arange(1,11),N-1);r=ref[idx];native_goal=goal_features(RefState(r[:,:3],r[:,3:7],r[:,7:28],r[:,28:31],r[:,31:34],r[:,34:]),torch.tensor(DEFAULT)).reshape(1,520)
  else:
   S=(torch.tensor(rq[:3][None],dtype=torch.float32),torch.tensor(rr[None],dtype=torch.float32),torch.tensor(f.v[ii][None],dtype=torch.float32),torch.tensor(f.w[ii][None],dtype=torch.float32),torch.tensor(state.joint_pos[None]),torch.tensor(state.joint_vel[None]))
   o=mdp.obs(S,prev,xi,torch.tensor([ii]));native_hist=[o]*10 if not native_hist else native_hist[1:]+[o];native_obs=torch.stack(native_hist,1).reshape(1,750);native_goal=lib.goal(torch.tensor([ii]),10)
  for name,val in [('obs_hist',native_obs),('goal',native_goal)]:
   err=float(np.max(np.abs(feed[name]-val.numpy())));producer_max=max(producer_max,err)
   np.testing.assert_allclose(feed[name],val.numpy(),atol=1e-5,rtol=1e-5)
  raw_fixture["robot_qpos"].append(state.qpos.copy());raw_fixture["robot_qvel"].append(state.qvel.copy());raw_fixture["previous_raw_action"].append(f.previous.copy());raw_fixture["expected_observation"].append(native_obs.numpy()[0].copy());raw_fixture["expected_goal"].append(native_goal.numpy()[0].copy());raw_fixture["tick"].append(tick)
  with torch.no_grad():source=model(native_obs,native_goal).numpy()
  feeds.append(feed);source_outputs.append(source.copy());f.accept_action(source[0]);expected.append((np.asarray(DEFAULT,dtype=np.float32)+.25*source[0]))
  for tensor in descriptors(a.backend):
   flat=feed[tensor['name']].reshape(-1)
   for item in tensor['features']:
    if 'packing' in item:
     pk=item['packing'];idx=np.concatenate([np.arange(item['offset']+j*pk['stride'],item['offset']+j*pk['stride']+pk['block_size']) for j in range(10)])
    else:idx=np.arange(item['offset'],item['offset']+item['size'])
    fixture.setdefault('feature/policy/'+tensor['name']+'/'+item['id'],[]).append(flat[idx].copy())
 np.savez(out/'tests'/(label+'_inputs.npz'),**{k:np.asarray(v,dtype=np.float32) for k,v in fixture.items()})
 np.savez(out/'tests'/(label+'_expected.npz'),**{'output/joint_position_target':np.asarray(expected,dtype=np.float32)})
 producer_path=out.parent/(out.name+'-'+label+'-producer-conformance.npz')
 np.savez_compressed(producer_path,**{k:np.asarray(v,dtype=np.float32 if k!='tick' else np.int64) for k,v in raw_fixture.items()})
 report['cases'].append({'id':label,'ticks':ticks,'producer_conformance':producer_path.name,'producer_conformance_sha256':hashlib.sha256(producer_path.read_bytes()).hexdigest()})
zero='sha256:'+'0'*64
contract={'contract':'robot-policy-onnx','contract_version':'1.0','package':{'name':'supermimic-'+a.backend+'-mini3-seed0','version':'1.0.0','interface_id':zero,'package_id':zero},'profile':{'id':'humanoid-motion-control','version':1},
 'compatibility':{'targets':[{'id':'mini3_v031','robot':{'vendor':'Zeroth','model':'Mini3','revision':'0.3.1'},'control_interface':{'id':'com.zeroth0.mini3.joint_position','version':1},'joint_set':'policy_joints'}]},
 'runtime':{'control_period_s':.02,'reset':{'recurrent_state':'declared_initial_value','history':'repeat_first_sample'},'tick':{'sample_phase':'before_inference'}},'joint_sets':{'policy_joints':JOINTS},
 'artifacts':[{'id':'policy','path':'policy.onnx','format':'onnx','sha256':zero,'size_bytes':(out/'policy.onnx').stat().st_size}],
 'graphs':{'policy':{'artifact':'policy','inputs':descriptors(a.backend),'outputs':[{'name':'raw_action','dtype':'float32','shape':[1,21]}]}},
 'execution':{'stages':[{'id':'policy','graph':'policy'}],'connections':[],'state':[]},
 'outputs':[{'id':'joint_position_target','source':{'graph':'policy','tensor':'raw_action'},'target':{'interface':'com.zeroth0.mini3.joint_position','field':'position','unit':'rad','joint_set':'policy_joints'},'transforms':[{'op':'affine','scale':.25,'offset':DEFAULT}]}],
 'tests':[{'id':x['id'],'ticks':x['ticks'],'inputs':{'path':'tests/'+x['id']+'_inputs.npz','sha256':zero,'size_bytes':(out/'tests'/(x['id']+'_inputs.npz')).stat().st_size},'expected':{'path':'tests/'+x['id']+'_expected.npz','sha256':zero,'size_bytes':(out/'tests'/(x['id']+'_expected.npz')).stat().st_size},'tolerance':{'absolute':1e-5,'relative':1e-5}} for x in report['cases']],
 'extensions':{EXT:semantic_extension(a.backend)},'provenance':{'checkpoint_name':ckpath.name,'source_variant_tree_sha256':source_identity,'exporter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'control_sha256':hashlib.sha256((D/'control/control.json').read_bytes()).hexdigest(),'iteration':ck.get('iteration',0),'source_commit':'ba6105349069bf040f1fdad60873d13138d505a2' if a.backend=='cuboi' else 'ee0608f917d08a591356b9bdc7d94bd949157341'}}
(out/'onnx_contract.yaml').write_text(yaml.dump(contract,Dumper=NoAliasDumper,sort_keys=False));validation=finalize_package(out)
options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1;options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL;options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
session=ort.InferenceSession(str(out/'policy.onnx'),sess_options=options,providers=['CPUExecutionProvider']);max_abs=0.;target_abs=0.
for feed,expected in zip(feeds,source_outputs):
 actual=session.run(['raw_action'],feed)[0];np.testing.assert_allclose(actual,expected,atol=1e-5,rtol=1e-5)
 max_abs=max(max_abs,float(np.max(np.abs(actual-expected))))
 actual_target=np.asarray(DEFAULT,dtype=np.float32)+.25*actual;expected_target=np.asarray(DEFAULT,dtype=np.float32)+.25*expected
 np.testing.assert_allclose(actual_target,expected_target,atol=1e-5,rtol=1e-5);target_abs=max(target_abs,float(np.max(np.abs(actual_target-expected_target))))
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(16<<20),b''):h.update(b)
 return h.hexdigest()
for _ in range(30):session.run(['raw_action'],feeds[0])
latencies=[]
for j in range(100):
 start=walltime.perf_counter();session.run(['raw_action'],feeds[j%len(feeds)]);latencies.append((walltime.perf_counter()-start)*1000)
report['network_only_latency_ms']={'mean':float(np.mean(latencies)),'p50':float(np.percentile(latencies,50)),'p95':float(np.percentile(latencies,95)),'provider':'CPUExecutionProvider','threads':1,'warmup':30,'samples':100,'end_to_end_teleop':None}
report.update(source_variant_tree_sha256=source_identity,exporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),passed=True,inputs=len(feeds),producer_max_abs=producer_max,raw_action_max_abs=max_abs,joint_target_max_abs=target_abs,final_onnx_sha256=sha(out/'policy.onnx'),checkpoint_sha256=sha(ckpath),torch=torch.__version__,onnx=onnx.__version__,ort=ort.__version__,package_validation=validation)
evidence=out.parent/(out.name+'-source-parity.json');evidence.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
