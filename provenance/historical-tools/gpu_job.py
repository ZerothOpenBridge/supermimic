"""Compose an isolated single-GPU job without shell interpolation; persistent receipts on CPFS."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
R=Path('/mnt/cpfs/workspace/yxq/supermimic');D=R/'shared/data/mini3_v1';E=R/'experiments/mini3_v1'
p=argparse.ArgumentParser();p.add_argument('--backend',choices=['cuboi','zita'],required=True);p.add_argument('--label',required=True);p.add_argument('--seconds',type=float,required=True);p.add_argument('--script',required=True)
p.add_argument('--gpu-index',type=int,default=2);p.add_argument('--host-alias',default='zero-2');p.add_argument('--lease-name',default='gpu2');p.add_argument('--docker-cli',default=str(R/'deploy/docker-isolated.sh'));p.add_argument('--container',default='yxq-supermimic-zita-gpu2');p.add_argument('--cache-label',default='zita')
p.add_argument('args',nargs=argparse.REMAINDER);a=p.parse_args()
args=a.args[1:] if a.args[:1]==['--'] else a.args
log=E/(a.backend+'-'+a.label+'.log');assert not log.exists(),log
source=D/'sources'/a.backend;files={str(x.relative_to(source)):hashlib.sha256(x.read_bytes()).hexdigest() for x in source.rglob('*.py') if '__pycache__' not in x.parts}
(E/(a.backend+'-'+a.label+'-source.json')).write_text(json.dumps({'source_files':files,'control_sha256':hashlib.sha256((D/'control/control.json').read_bytes()).hexdigest(),'script':a.script,'script_sha256':hashlib.sha256(Path(a.script).read_bytes()).hexdigest(),'arguments':args},indent=2))
cmd=['python3',str(D/'budget_run.py'),'--backend',a.backend,'--label',a.label,'--max-seconds',str(a.seconds),'--gpu-index',str(a.gpu_index),'--host-alias',a.host_alias,'--lease-name',a.lease_name,'--docker-cli',a.docker_cli]
env=os.environ.copy();env.update(OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='1')
if a.backend=='cuboi':
 env.update(CUDA_VISIBLE_DEVICES=str(a.gpu_index),PYTHONPATH=str(source/'src'));cmd+=['--',str(R/'cuboi/.venv/bin/python'),a.script,*args]
else:
 pidfile=E/(a.label+'.container-pid.json');cmd+=['--docker-container',a.container,'--container-pidfile',str(pidfile),'--',a.docker_cli,'exec','-e','SM_JOB_DEADLINE_UNIX','-e','OMP_NUM_THREADS=8','-e','OPENBLAS_NUM_THREADS=1','-e','PYTHONPATH='+str(source)]
 for key,suffix in [('CUDA_CACHE_PATH','cuda'),('TORCHINDUCTOR_CACHE_DIR','inductor'),('TRITON_CACHE_DIR','triton'),('XDG_CACHE_HOME','xdg'),('TMPDIR','tmp')]:
  folder=D/'runtime/cache'/a.cache_label/suffix;folder.mkdir(parents=True,exist_ok=True);cmd+=['-e',key+'='+str(folder)]
 cmd+=[a.container,'python',str(D/'zita_entry.py'),str(pidfile),a.script,*args]
with log.open('x') as f:code=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
print(json.dumps({'log':str(log),'returncode':code}),flush=True);raise SystemExit(code)
