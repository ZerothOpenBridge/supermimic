"""Finite zita Mini3 seed0 sequence on the explicitly authorized spare node, same 48 GPU-hour ledger."""
import fcntl,hashlib,json,os,subprocess,time,traceback
from pathlib import Path
R=Path('/mnt/cpfs/workspace/yxq/supermimic');D=R/'shared/data/mini3_v1';E=R/'experiments/mini3_v1'
CLI=R/'deploy/docker-zhangyu.sh';CONTAINER='yxq-supermimic-zita-zhangyu-gpu0';IMAGE='sha256:4f29ffba9da63bc2ec812b02053db1b4c29cea89ceddc5c66848e89895422f91'
state=E/'zita-parallel-state.json';lock=open(E/'zita-parallel.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(16<<20),b''):h.update(b)
 return h.hexdigest()
def write(p,d):
 t=p.with_name(p.name+'.'+str(os.getpid())+'.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
def update(phase,**kw):
 obj={'phase':phase,'updated_unix':time.time(),'host':'zhangyu','physical_gpu':0,'cap_gpu_hours':48,**kw};write(state,obj);print(json.dumps(obj),flush=True)
def stop_check():
 if any((E/x).exists() for x in ['STOP_PIPELINE','STOP_ZITA_PARALLEL']):raise RuntimeError('Explicit stop marker: no subsequent jobs launched')
def budget():
 b=json.loads((E/'zita-budget.json').read_text());assert b['cap_seconds']==172800;return b['cap_seconds']-b['spent_seconds']
def cpu(script,args,log):
 envpath='/opt/isaaclab-venv/lib/python3.12/site-packages:'+str(D/'runtime/export-runtime')+':'+str(D/'runtime/contracts/lib/python3.12/site-packages')+':'+str(D/'sources/zita')
 cmd=[str(CLI),'exec','-e','CUDA_VISIBLE_DEVICES=','-e','OMP_NUM_THREADS=8','-e','OPENBLAS_NUM_THREADS=1','-e','PYTHONPATH='+envpath,CONTAINER,'python',str(script),*map(str,args)]
 with open(log,'x') as f:subprocess.run(cmd,check=True,stdout=f,stderr=subprocess.STDOUT)
def gpu(label,seconds,script,args):
 stop_check();update(label,remaining_gpu_hours=budget()/3600)
 subprocess.run(['python3',str(D/'gpu_job.py'),'--backend','zita','--label',label,'--seconds',str(seconds),'--script',str(script),'--gpu-index','0','--host-alias','zhangyu','--lease-name','zhangyu-gpu0','--docker-cli',str(CLI),'--container',CONTAINER,'--cache-label','zita-zhangyu','--',*map(str,args)],check=True)
def publish(checkpoint):
 update('export-and-conformance',checkpoint=str(checkpoint));stop_check();building=D/'packages/building-final-zita-mini3-v1';final=D/'packages/final-zita-mini3';assert not building.exists() and not final.exists()
 cpu(D/'export_policy.py',['--backend','zita','--checkpoint',checkpoint,'--output',building],E/'zita-final-export.log')
 report=building.parent/(building.name+'-source-parity.json');d=json.loads(report.read_text());assert d['passed'] and d['inputs']>=64 and d['checkpoint_sha256']==sha(checkpoint)
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(D/'evaluator-frozen-v1/evaluator/src')+':'+str(D/'runtime/export-runtime'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 subprocess.run([str(R/'cuboi/.venv/bin/python'),str(D/'check_producer_vectors.py'),'--report',str(report)],env=env,check=True)
 clean=report.with_name(report.stem+'-clean-producers.json');assert json.loads(clean.read_text())['passed'];building.rename(final);assert sha(final/'policy.onnx')==d['final_onnx_sha256']
 ready={'backend':'zita','package':str(final),'package_id':d['package_validation']['package_id'],'interface_id':d['package_validation']['interface_id'],'onnx_sha256':d['final_onnx_sha256'],'source_parity':str(report),'source_parity_sha256':sha(report),'producer_conformance':str(clean),'producer_conformance_sha256':sha(clean),'checkpoint':str(checkpoint),'training_host':'zhangyu','physical_gpu':0}
 (E/'packages-ready').mkdir(exist_ok=True);write(E/'packages-ready/zita.json',ready);update('package-accepted',**ready)
try:
 stop_check();update('waiting-image-archive')
 deadline=time.monotonic()+1800
 while True:
  stop_check();r=json.loads((E/'zita-image-transfer-20261030.json').read_text())
  if r['status']=='saved':break
  assert time.monotonic()<deadline,'Image archive not ready within 30 minutes';time.sleep(10)
 archive=Path(r['archive']);assert r['image_id']==IMAGE and sha(archive)==r['archive_sha256']
 update('checking-isolated-image',archive_bytes=r['bytes'])
 if subprocess.run([str(CLI),'image','inspect',IMAGE],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode!=0:
  update('loading-isolated-image',archive_bytes=r['bytes']);subprocess.run([str(CLI),'load','--input',str(archive)],check=True,timeout=1800)
 assert subprocess.check_output([str(CLI),'image','inspect',IMAGE,'--format','{{.Id}}'],text=True).strip()==IMAGE
 used=int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip());assert used<=1024,used
 container_exists=subprocess.run([str(CLI),'container','inspect',CONTAINER],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0
 cache=D/'runtime/cache/zita-zhangyu';cache.mkdir(parents=True,exist_ok=True)
 mounts=[(R/'experiments',R/'experiments',False),(R/'shared/data',R/'shared/data',False),(cache,Path('/root/.cache'),False),(R/'zita',Path('/workspace/supermimic'),False),(R/'deploy',R/'deploy',True)]
 cmd=[str(CLI),'run','--detach','--name',CONTAINER,'--runtime','nvidia','--gpus','device=0','--shm-size','8g','--workdir','/workspace/supermimic']
 for src,dst,ro in mounts:cmd+=['--volume',str(src)+':'+str(dst)+(':ro' if ro else '')]
 cmd += ['--entrypoint','sleep',IMAGE,'infinity']
 if not container_exists:subprocess.run(cmd,check=True)
 inspect=json.loads(subprocess.check_output([str(CLI),'inspect',CONTAINER],text=True))[0];assert inspect['Image']==IMAGE and inspect['State']['Running']
 assert {x['Destination']:x['Source'] for x in inspect['Mounts']}=={str(dst):str(src) for src,dst,ro in mounts}
 assert inspect['HostConfig']['DeviceRequests'][0]['DeviceIDs']==['0']
 write(E/'zita-zhangyu-environment.json',{'image_id':IMAGE,'archive_sha256':r['archive_sha256'],'container':CONTAINER,'device_requests':inspect['HostConfig']['DeviceRequests'],'docker_root':'/tmp/yxq-supermimic-docker-zhangyu-overlay/storage/docker','storage_backing_file':str(R/'shared/docker-zhangyu-overlay-v1.ext4'),'storage_driver':'overlay2 on an exclusive CPFS-backed ext4 loop mount','host':'zhangyu','created_unix':time.time()})
 common=['--num_envs','4096','--pi_batch','4096','--wm_batch','2048','--horizon','32','--seed','0']
 gpu('zhangyu-sim-smoke-r2',900,D/'zita_sim_check.py',['--output',E/'zita-zhangyu-sim-check.json'])
 assert json.loads((E/'zita-zhangyu-sim-check.json').read_text())['passed']
 profile=E/'zita/zhangyu-profile4096-r1'
 gpu('zhangyu-profile4096-r1',1800,D/'sources/zita/supermimic/train.py',common+['--phase','profile','--iterations','4','--wm_only_iters','0','--log_every','1','--motion_dir',D/'packed/train.pt','--log_dir',profile])
 b=json.loads((E/'zita-budget.json').read_text());peak=b['jobs'][-1]['peak_gpu_memory_mib'];total=int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.total','--format=csv,noheader,nounits'],text=True).strip());assert peak<=.8*total,(peak,total)
 write(E/'zita-zhangyu-resource-gate.json',{'passed':True,'peak_mib':peak,'total_mib':total,'common_envs':4096,'policy_batch':4096,'world_model_batch':2048,'horizon':32})
 zrun=E/'zita/pilot12-4096-r1';assert not zrun.exists(),'Never silently overwrite an existing pilot'
 gpu('pilot12-4096-r1',7200,D/'sources/zita/supermimic/train.py',common+['--phase','pilot','--iterations','5000','--motion_dir',D/'packed/pilot.pt','--log_dir',zrun])
 for which,checkpoint in [('initial',zrun/'initial_policy.pt'),('final',zrun/'last.pt')]:gpu('pilot-eval-'+which+'-r1',900,D/'evaluate_zita_mini3.py',['--checkpoint',checkpoint,'--library','pilot','--output',zrun/(which+'-evaluation.json')])
 cpu(D/'validate_zita_pilot.py',[],E/'zita-pilot-learning-gate.log');assert json.loads((zrun/'learning-gate.json').read_text())['passed']
 zrun=E/'zita/formal-seed0-r1';assert not zrun.exists();resume=None;best=None;chunk=0;last_iteration=0
 while last_iteration<320000 and budget()>=600:
  chunk+=1;remaining=budget();cadence=min(7200,max(60,remaining-600))
  args=common+['--phase','formal','--iterations','320000','--ckpt_every','1000000000','--checkpoint_seconds',str(cadence),'--stop_after_timed_checkpoint','--motion_dir',D/'packed/train.pt','--log_dir',zrun]
  if resume:args+=['--resume',resume]
  gpu(f'formal-seed0-r1-chunk{chunk:02d}',min(7700,remaining-300),D/'sources/zita/supermimic/train.py',args)
  counters=json.loads((zrun/'training-result.json').read_text());last_iteration=counters['iteration'];checkpoint=zrun/f'ckpt_{last_iteration:06d}.pt';resume=zrun/'last.pt'
  if budget()<180:break
  result=zrun/f'dev316_{last_iteration:06d}.json';gpu(f'dev316-{last_iteration:06d}',min(900,budget()),D/'evaluate_zita_mini3.py',['--checkpoint',checkpoint,'--library','dev316','--output',result])
  dev=json.loads(result.read_text());assert len(dev['clips'])==316 and len({x['name'] for x in dev['clips']})==316
  m=dev['summary'];key=[m['completion_rate'],m['mean_progress'],-m['mean_local_kpt_mae'],-last_iteration]
  if best is None or tuple(key)>tuple(best['criterion_key']):
   best={'checkpoint':str(checkpoint),'iteration':last_iteration,'criterion_key':key,'development':m,'development_only':True};write(zrun/'selection.json',best)
  update('candidate-validated',iteration=last_iteration,remaining_gpu_hours=budget()/3600,selected=best)
 assert best is not None,'No accepted development candidate within GPU budget';publish(Path(best['checkpoint']))
except Exception as exc:update('stage-stopped',error=str(exc),traceback=traceback.format_exc());raise
