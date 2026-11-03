"""CPFS-only launch adapter. --dry-run never imports torch or starts a job."""
import argparse,json,os,re,shlex,subprocess,sys,time,hashlib
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
R=Path('/mnt/cpfs/workspace/yxq/supermimic')
D=R/'shared/data/mini3_v1'
METHOD=json.loads((REPO/'configs/implementation.json').read_text())['backend']

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['preflight','check-fk','train','resume','native-eval','export','verify-package','evaluate'])
    p.add_argument('--run-id',default='')
    p.add_argument('--gpu',type=int,help='physical host GPU; required for GPU actions')
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--package',type=Path)
    p.add_argument('--clip',action='append',default=[],help='evaluate only: exact clip ID; repeat for a small interface check')
    p.add_argument('--workers',type=int,default=32,help='evaluate only: CPU workers, 1..32')
    p.add_argument('--run-root',type=Path,default=R/'shared/data/collaboration/runs')
    p.add_argument('--max-seconds',type=float,default=7800,help='one invocation, including checkpoint grace; run ledger cap is 48 GPUh')
    p.add_argument('--python',default=os.environ.get('SM_PYTHON',str(R/'cuboi/.venv/bin/python')))
    p.add_argument('--docker-cli',default=os.environ.get('SM_DOCKER_CLI',str(R/'deploy/docker-zhangyu.sh')))
    p.add_argument('--container',default=os.environ.get('SM_CONTAINER','yxq-supermimic-zita-zhangyu-gpu0'))
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    gpu_action=a.action in ['train','resume','native-eval']
    if a.action not in ['preflight','check-fk'] and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,80}',a.run_id):p.error('Supply a new --run-id (letters, numbers, _, . and -)')
    if gpu_action and (a.gpu is None or a.gpu<0):p.error('--gpu is required for GPU actions')
    if not 1<=a.workers<=32:p.error('--workers must be 1..32')
    if a.clip and a.action!='evaluate':p.error('--clip is only valid for evaluate')
    if a.max_seconds<300:p.error('--max-seconds must reserve at least 300 seconds')
    run=a.run_root/METHOD/a.run_id if a.run_id else a.run_root/METHOD
    if not str(run).startswith('/mnt/cpfs/'):p.error('run-root must be on CPFS')
    if (R/'experiments').resolve() in run.resolve().parents:p.error('Historical experiments are read-only; choose collaboration/runs')
    training=run/'training'
    ck=a.checkpoint
    if not ck and a.action in ['resume','native-eval']:
        ck=training/('ckpt_latest.pt' if METHOD=='cuboi' else 'last.pt')
        if a.action=='native-eval' and METHOD=='zita' and (training/'training-result.json').exists():
            ck=training/('ckpt_'+str(json.loads((training/'training-result.json').read_text())['iteration']).zfill(6)+'.pt')
    selection=training/'selection.json'
    if not ck and a.action=='export':
        if selection.exists():ck=Path(json.loads(selection.read_text())['checkpoint'])
        else:p.error('export requires --checkpoint or this run\'s selection.json')
    if a.action=='preflight':target=[str(REPO/'scripts/preflight.py')]
    elif a.action=='check-fk':target=[str(REPO/'scripts/check_fk.py'),'--backend',METHOD]
    elif a.action in ['train','resume']:
        if METHOD=='cuboi':
            target=[str(REPO/'scripts/train_mini3.py'),'--run','training','--run-root',str(run)]
        else:
            c=json.loads((REPO/'configs/formal.json').read_text())
            keys=['num_envs','iterations','buffer_steps','utd','horizon','goal_frames','hist','wm_window','wm_batches','wm_batch','pi_batch','lr','gamma','lam','action_noise','critic_epochs','critic_minibatches','warmup_steps','wm_only_iters','lr_decay_iters','log_every','amp','seed']
            target=[str(REPO/'supermimic/train.py')]+[v for k in keys for v in ['--'+k,str(c[k])]]
            target+=['--phase','formal','--motion_dir',str(D/'packed/train.pt'),'--log_dir',str(training),'--checkpoint_seconds','7200','--ckpt_every','1000000000','--stop_after_timed_checkpoint']
        if a.action=='resume':target+=['--resume',str(ck)]
    elif a.action=='native-eval':
        eval_output=run/('dev316-'+str(int(time.time()))+'.json')
        target=[str(REPO/'scripts/native_eval.py'),'--checkpoint',str(ck),'--output',str(eval_output)]
        if METHOD=='zita':target+=['--library','dev316']
    elif a.action=='export':target=[str(REPO/'scripts/export_policy.py'),'--backend',METHOD,'--checkpoint',str(ck),'--output',str(run/'policy-package')]
    elif a.action=='verify-package':
        if a.package is None:p.error('verify-package requires --package')
        target=[str(REPO/'scripts/verify_package.py'),'--package',str(a.package)]
    else:
        if a.package is None:p.error('evaluate requires --package (an exported package directory)')
        target=[str(REPO/'scripts/evaluate_package.py'),'--package',str(a.package),'--output',str(run/'evaluation'),'--workers',str(a.workers)]+[v for c in a.clip for v in ['--clip',c]]
    source=REPO/'src' if METHOD=='cuboi' else REPO
    # Native export retains training dependencies; its script appends only missing export tools.
    # The independent consumer intentionally gives the frozen MuJoCo/ORT runtime priority.
    extra=[D/'evaluator-frozen-v1/evaluator/src'] if a.action=='export' else [D/'evaluator-frozen-v1/evaluator/src',D/'runtime/export-runtime',D/'runtime/contracts/lib/python3.12/site-packages'] if a.action in ['verify-package','evaluate'] else []
    pythonpath=':'.join(map(str,[source,*extra]))
    cache=run/'cache' if a.run_id else a.run_root/METHOD/'preflight-cache'
    env=os.environ.copy();env.update(PYTHONPATH=pythonpath,OMP_NUM_THREADS='8' if gpu_action else '1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='8' if gpu_action else '1',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=str(a.gpu) if gpu_action and METHOD=='cuboi' else '',SM_RUN_BUDGET_ROOT=str(run/'resource-ledger'))
    for key,suffix in [('CUDA_CACHE_PATH','cuda'),('TRITON_CACHE_DIR','triton'),('TORCHINDUCTOR_CACHE_DIR','inductor'),('XDG_CACHE_HOME','xdg'),('TMPDIR','tmp'),('WARP_CACHE_PATH','warp'),('MPLCONFIGDIR','matplotlib')]:env[key]=str(cache/suffix)
    if METHOD=='zita' and a.action in ['preflight','check-fk','train','resume','native-eval','export']:
        assignments={k:env[k] for k in ['PYTHONPATH','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','PYTHONDONTWRITEBYTECODE','CUDA_CACHE_PATH','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','XDG_CACHE_HOME','TMPDIR','WARP_CACHE_PATH','MPLCONFIGDIR']}
        assignments['CUDA_VISIBLE_DEVICES']='0' if gpu_action else ''
        cmd=[a.docker_cli,'exec','--workdir',str(REPO)]+[v for k,val in assignments.items() for v in ['-e',k+'='+val]]
        if gpu_action:
            pidfile=run/'container-pid.json'
            cmd+=['-e','SM_JOB_DEADLINE_UNIX',a.container,'python',str(REPO/'scripts/container_entry.py'),str(pidfile),*target]
        else:cmd+=[a.container,'python',*target]
    else:cmd=[a.python,*target]
    if gpu_action:
        guard=[sys.executable,str(REPO/'scripts/budget_guard.py'),'--backend',METHOD,'--label',a.action+'-'+str(int(time.time())),'--max-seconds',str(a.max_seconds),'--gpu-index',str(a.gpu),'--lease-name','gpu'+str(a.gpu)]
        if METHOD=='zita':guard+=['--docker-container',a.container,'--container-pidfile',str(run/'container-pid.json'),'--docker-cli',a.docker_cli]
        cmd=guard+['--',*cmd]
    plan={'backend':METHOD,'action':a.action,'run_directory':str(run),'command':cmd,'native_source':str(source),'dry_run':a.dry_run,'historical_data_root':str(D),'recorded_config':'configs/recorded-formal.json','fresh_config':'configs/formal.json'}
    module=source/'supermimic'
    source_files={str(f.relative_to(module)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(module.rglob('*.py')) if not f.name.startswith('._') and '__pycache__' not in f.parts}
    plan['source_files']=source_files
    plan['fresh_config_sha256']=hashlib.sha256((REPO/'configs/formal.json').read_bytes()).hexdigest()
    rev=subprocess.run(['git','-C',str(REPO),'rev-parse','HEAD'],text=True,capture_output=True)
    plan['git_commit']=rev.stdout.strip() if rev.returncode==0 else None
    if rev.returncode==0:plan['git_dirty']=bool(subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'],text=True).strip())
    print(json.dumps(plan,ensure_ascii=False,indent=2),flush=True)
    if a.dry_run:return
    if not D.is_dir():raise RuntimeError('Shared CPFS assets are missing; read docs/CPFS_ASSETS.md')
    if ck and not ck.is_file():raise FileNotFoundError(ck)
    if a.action=='train' and run.exists():raise FileExistsError('Fresh training refuses an existing run: '+str(run))
    if a.action=='resume' and not run.exists():raise FileNotFoundError('Resume requires an existing collaboration run; fork manually into a new run if needed')
    if METHOD=='zita' and a.action in ['preflight','check-fk','train','resume','native-eval','export']:
        inspect=json.loads(subprocess.check_output([a.docker_cli,'inspect',a.container],text=True))[0]
        if not inspect['State']['Running']:raise RuntimeError('Selected container is not running; no container is started automatically')
        if not any(str(REPO)==m['Destination'] or str(REPO).startswith(m['Destination'].rstrip('/')+'/') for m in inspect['Mounts']):raise RuntimeError('Checkout is outside the selected container mounts. Clone under '+str(R/'shared/data/collaboration/checkouts/zita'))
        if gpu_action:
            devices=[x for q in inspect['HostConfig'].get('DeviceRequests',[]) for x in (q.get('DeviceIDs') or [])]
            uuid=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
            if devices not in [[str(a.gpu)],[uuid]]:raise RuntimeError('Container must expose exactly the requested physical GPU; create a separate container instead of modifying an existing one')
    for key in ['CUDA_CACHE_PATH','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','XDG_CACHE_HOME','TMPDIR','WARP_CACHE_PATH','MPLCONFIGDIR']:Path(env[key]).mkdir(parents=True,exist_ok=True)
    if a.run_id:run.mkdir(parents=True,exist_ok=True)
    if gpu_action:
        log=run/(a.action+'-'+str(time.time_ns())+'.stdout.log')
        (run/('launch-'+str(time.time_ns())+'.json')).write_text(json.dumps(plan,indent=2))
        print('stdout:',log,flush=True)
        with log.open('x') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        if a.action=='native-eval':
            result=json.loads(eval_output.read_text());m=result['summary'];it=result['iteration']
            import math
            key=[m['completion_rate'],m['mean_progress'],-m['mean_local_kpt_mae'],-it]
            if not all(math.isfinite(float(x)) for x in key):raise ValueError('Nonfinite development score')
            best=json.loads(selection.read_text()) if selection.exists() else None
            if best is None or tuple(key)>tuple(best['criterion_key']):
                selection.write_text(json.dumps({'checkpoint':result['checkpoint'],'iteration':it,'criterion_key':key,'development':m,'development_only':True},indent=2))
    else:subprocess.run(cmd,env=env,check=True)

if __name__=='__main__':main()
