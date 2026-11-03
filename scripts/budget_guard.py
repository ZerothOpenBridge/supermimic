"""Single-GPU lease and persistent per-repository wall-time budget. Never stops other jobs."""
import argparse,datetime,fcntl,json,os,signal,subprocess,time,hashlib,threading,atexit,re,socket
from pathlib import Path

ROOT=Path(os.environ['SM_RUN_BUDGET_ROOT'])
def atomic(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));os.replace(tmp,path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=['cuboi','zita'],required=True);p.add_argument('--label',required=True)
    p.add_argument('--max-seconds',type=float,default=172800)
    p.add_argument('--docker-container');p.add_argument('--container-pidfile')
    p.add_argument('--gpu-index',type=int,default=2);p.add_argument('--host-alias',default=socket.gethostname());p.add_argument('--lease-name',default='gpu2')
    p.add_argument('--docker-cli',default='/mnt/cpfs/workspace/yxq/supermimic/deploy/docker-isolated.sh')
    p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    cmd=a.command[1:] if a.command[:1]==['--'] else a.command
    assert a.gpu_index>=0 and re.fullmatch('[a-z0-9-]+',a.lease_name)
    ROOT.mkdir(exist_ok=True);lock=open(ROOT/(a.lease_name+'.lock'),'a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    used=int(subprocess.check_output(['nvidia-smi','-i',str(a.gpu_index),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True,timeout=5).strip())
    if used>1024:raise RuntimeError(f'GPU{a.gpu_index} occupied ({used}MiB); no job launched')
    uuid=subprocess.check_output(['nvidia-smi','-i',str(a.gpu_index),'--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=5).strip()
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=5)
    if any(row.split(',')[0].strip()==uuid for row in apps.splitlines() if row.strip()):raise RuntimeError('GPU has another compute process; no job launched')
    # Atomic mkdir is the cross-host exclusion mechanism on this nolock NFS mount.
    lease=Path('/mnt/cpfs/workspace/yxq/supermimic/shared/data/collaboration/gpu-leases')/uuid;lease.parent.mkdir(parents=True,exist_ok=True);lease.mkdir()
    (lease/'owner.json').write_text(json.dumps({'host_alias':a.host_alias,'launcher_pid':os.getpid(),'gpu_index':a.gpu_index,'backend':a.backend}))
    child=None
    def release_lease():
        if child is None or child.poll() is not None:
            try:(lease/'owner.json').unlink();lease.rmdir()
            except OSError:pass
    atexit.register(release_lease)
    path=ROOT/(a.backend+'-budget.json');d=json.loads(path.read_text()) if path.exists() else {'backend':a.backend,'cap_seconds':172800,'spent_seconds':0.,'jobs':[]}
    remaining=d['cap_seconds']-d['spent_seconds'];limit=min(a.max_seconds,remaining)
    if limit<=0:raise RuntimeError('Repository GPU budget exhausted')
    start=time.monotonic();base=d['spent_seconds'];job={'label':a.label,'started':datetime.datetime.now(datetime.timezone.utc).isoformat(),'limit_seconds':limit,'status':'running','command':cmd,'host_alias':a.host_alias,'gpu_index':a.gpu_index};d['jobs'].append(job)
    if a.docker_container:
        if not a.container_pidfile:raise ValueError('Container job requires a dedicated PID receipt')
        Path(a.container_pidfile).unlink(missing_ok=True)
    cache=ROOT.parent/'cache'
    for key,suffix in [('CUDA_CACHE_PATH','cuda'),('TRITON_CACHE_DIR','triton'),('TORCHINDUCTOR_CACHE_DIR','inductor'),('XDG_CACHE_HOME','xdg')]:
        folder=cache/suffix;folder.mkdir(parents=True,exist_ok=True);os.environ.setdefault(key,str(folder))
    os.environ.setdefault('TMPDIR',str(ROOT.parent/'cache/tmp'))
    os.environ['SM_JOB_DEADLINE_UNIX']=str(time.time()+limit)
    child=subprocess.Popen(cmd,start_new_session=True);job['pid']=child.pid
    job['peak_gpu_memory_mib']=0
    stopped=[False]
    def container_signal(sig):
        if not a.docker_container:return
        receipt=Path(a.container_pidfile)
        if not receipt.exists():return
        pid=json.loads(receipt.read_text())['pid']
        docker=a.docker_cli
        # PID namespace isolation and a job-specific receipt limit signals to our job.
        subprocess.run([str(docker),'exec',a.docker_container,'kill','-'+str(int(sig)),str(pid)],check=False,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=10)
    def stop(sig,frame):
        stopped[0]=True
        container_signal(signal.SIGTERM)
        # Keep docker exec attached until the trainer finishes its checkpoint and exits.
        # Killing only that client would release accounting while the container job still saves.
        if not a.docker_container:
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    # Enforce the lease even if a hard-mounted filesystem write or NVML call blocks.
    finished=threading.Event()
    def hard_deadline():
        if finished.wait(max(0,limit-120-(time.monotonic()-start))):return
        stop(None,None)
        if finished.wait(max(0,limit-(time.monotonic()-start))):return
        container_signal(signal.SIGKILL)
        try:os.killpg(child.pid,signal.SIGKILL)
        except ProcessLookupError:pass
    threading.Thread(target=hard_deadline,daemon=True).start()
    last_write_warning=0
    while child.poll() is None:
        elapsed=time.monotonic()-start;d['spent_seconds']=base+elapsed;job['elapsed_seconds']=elapsed
        try:
            memory=int(subprocess.check_output(['nvidia-smi','-i',str(a.gpu_index),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True,timeout=5).strip());job['peak_gpu_memory_mib']=max(job['peak_gpu_memory_mib'],memory);job['current_gpu_memory_mib']=memory
        except Exception:pass
        try:atomic(path,d)
        except OSError as exc:
            if elapsed-last_write_warning>30:
                try:print(f'BUDGET_RECEIPT_WRITE_ERROR {exc}; independent deadline remains active',flush=True)
                except OSError:pass
                last_write_warning=elapsed
        if elapsed>=max(0,limit-120) and not stopped[0]:stop(None,None)
        if elapsed>=limit:
            container_signal(signal.SIGKILL)
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
        time.sleep(1)
    finished.set()
    elapsed=time.monotonic()-start;d['spent_seconds']=base+elapsed;job.update(status='stopped_by_limit' if stopped[0] else ('passed' if child.returncode==0 else 'failed'),returncode=child.returncode,elapsed_seconds=elapsed);atomic(path,d)
    print(json.dumps({'job':job,'total_GPU_hours':d['spent_seconds']/3600}),flush=True)
    raise SystemExit(child.returncode if child.returncode>=0 else 128-child.returncode)

if __name__=='__main__':main()
