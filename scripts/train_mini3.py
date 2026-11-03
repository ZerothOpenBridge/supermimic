"""Fresh seed0 full-data Mini3 training with dev316 selection, bounded externally by GPU lease."""
import argparse,json,random,signal,time,hashlib,os,resource
from dataclasses import asdict
from pathlib import Path
import numpy as np,torch
from supermimic.config import TrainConfig
from supermimic.training.loop import Trainer
from supermimic.eval.benchmark import run_benchmark,save_results,summarize
R=Path('/mnt/cpfs/workspace/yxq/supermimic');D=R/'shared/data/mini3_v1'
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--resume');p.add_argument('--run-root',type=Path,required=True);p.add_argument('--print-config',action='store_true');a=p.parse_args()
c=TrainConfig();c.seed=0;c.log_root=a.run_root;c.run_name=a.run;c.data.library=D/'packed/train.pt';c.env.compact_library=True
c.env.num_envs=4096;c.wm_trainer.batch_size=2048;c.policy_trainer.batch_size=4096;c.policy_trainer.horizon=32;c.schedule.total_iterations=320000
c.schedule.checkpoint_interval=10**9;c.schedule.checkpoint_seconds=7200.
def apply_config(dst,src):
 from dataclasses import is_dataclass
 for key,value in src.items():
  current=getattr(dst,key)
  if is_dataclass(current):apply_config(current,value)
  else:setattr(dst,key,Path(value) if isinstance(current,Path) and value is not None else tuple(value) if isinstance(current,tuple) else value)
recipe=json.loads((Path(__file__).resolve().parents[1]/'configs/formal.json').read_text())
recipe.pop('run_name',None);recipe.pop('log_root',None);apply_config(c,recipe)
c.log_root=a.run_root;c.run_name=a.run;c.data.library=D/'packed/train.pt'
if a.resume:c.resume=Path(a.resume)
assert a.resume or not (c.log_root/a.run).exists()
if a.print_config:print(json.dumps(asdict(c),indent=2,default=str));raise SystemExit(0)
t=Trainer(c);signal.signal(signal.SIGTERM,lambda s,f:setattr(t,'stop_requested',True));signal.signal(signal.SIGINT,lambda s,f:setattr(t,'stop_requested',True))
selection=t.log_dir/'selection.json';best=json.loads(selection.read_text()) if selection.exists() else None
if a.resume:
 held=[t._pending_resume];resume_iteration=t.iteration
 loaded_optimizer_step=max(int(v['step']) for v in t.policy_trainer.opt.state.values())
 def verify_actual_resume(trainer,obs):
  state=held.pop();rng=state['rng'];numpy_state=np.random.get_state()
  checks={'replay_equal':torch.equal(trainer.buffer.s,state['objects']['buffer']['s'].to(trainer.buffer.s.device)),
          'sampler_equal':torch.equal(trainer.command.bin_return,state['objects']['command']['bin_return'].to(trainer.command.bin_return.device)),
          'history_equal':all(torch.equal(v,state['obs'][k].to(v.device)) for k,v in obs.items()),
          'python_rng_equal':random.getstate()==rng['python'],
          'numpy_rng_equal':numpy_state[0]==rng['numpy'][0] and np.array_equal(numpy_state[1],rng['numpy'][1]) and numpy_state[2:]==rng['numpy'][2:],
          'torch_rng_equal':torch.equal(torch.get_rng_state(),rng['torch'].cpu()),
          'cuda_rng_equal':all(torch.equal(x,y.cpu()) for x,y in zip(torch.cuda.get_rng_state_all(),rng['cuda']))}
  assert all(checks.values()),checks
  checks.update(passed=True,iteration=trainer.iteration,updates=trainer.num_updates,optimizer_step=loaded_optimizer_step,checkpoint=str(a.resume))
  (t.log_dir/'resume-start-verification.json').write_text(json.dumps(checks,indent=2));print('ACTUAL_RESUME_VERIFIED',json.dumps(checks),flush=True)
 t.on_resume=verify_actual_resume

def evaluate_candidate(trainer):
 global best
 if trainer.iteration==0:return
 name=f'candidate_{trainer.iteration:07d}.pt';path=t.log_dir/name
 if path.exists() and (t.log_dir/(path.stem+'-dev316.json')).exists():return
 if float(os.environ.get('SM_JOB_DEADLINE_UNIX','inf'))-time.time()<300:
  (t.log_dir/'final-candidate-deferred.json').write_text(json.dumps({'iteration':t.iteration,'reason':'budget guard reserves time for final full checkpoint; selected earlier validated candidate retained'}));return
 if not path.exists():torch.save({'policy':t.policy.state_dict(),'wm':t.wm.state_dict(),'critic':t.critic.state_dict(),'iteration':t.iteration,'num_updates':t.num_updates,'config':asdict(c)},path)
 rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
 t.policy.eval()
 try:rows=run_benchmark(t.policy,[str(D/'packed/dev316.pt')],num_envs=316,verbose=False)
 finally:
  t.policy.train();random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2]);torch.cuda.set_rng_state_all(rng[3])
 assert len(rows)==316 and len({x.name for x in rows})==316
 save_results(rows,t.log_dir/(path.stem+'-dev316.json'));m=summarize(rows)
 key=[m['completion_rate'],m['mean_progress'],-m['mean_local_kpt_mae'],-t.iteration]
 if not np.isfinite(key).all():raise FloatingPointError('Nonfinite development score')
 record={'checkpoint':str(path),'iteration':t.iteration,'updates':t.num_updates,'criterion_key':key,'development':m,'development_only':True}
 if best is None or tuple(key)>tuple(best['criterion_key']):best=record;selection.write_text(json.dumps(best,indent=2))
 print('DEV316_CANDIDATE',json.dumps(record),flush=True)
t.on_checkpoint=evaluate_candidate;t.run()
report={'status':'training_stopped_or_native_steps_complete','iteration':t.iteration,'updates':t.num_updates,'sim_transitions':t.iteration*c.env.num_envs,
        'wm_optimizer_updates':t.num_updates*c.wm_trainer.batches_per_update,'policy_optimizer_updates':t.num_updates,
        'imagined_steps':t.num_updates*c.policy_trainer.batch_size*c.policy_trainer.horizon,'selected':best,'seed':0,'fresh_training':not bool(a.resume),
        'peak_main_process_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'peak_torch_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_torch_reserved_bytes':torch.cuda.max_memory_reserved()}
if a.resume:
 report['resumed_from']=str(a.resume);report['iteration_at_resume']=resume_iteration
 report['optimizer_step_at_resume']=loaded_optimizer_step;report['optimizer_step_at_end']=max(int(v['step']) for v in t.policy_trainer.opt.state.values())
(t.log_dir/'training-result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True);t.env.close()
