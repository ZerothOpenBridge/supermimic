"""Build a compact shared Mini3 state library from immutable source NPZ motions (CPU only)."""
import argparse, concurrent.futures, hashlib, json, os, time
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation

def load_motion(path):
    blob=path.read_bytes()
    import io
    with np.load(io.BytesIO(blob),allow_pickle=False) as z:q=z['qpos'].astype(np.float64)
    if q.ndim!=2 or q.shape[1]!=28 or len(q)<2 or not np.isfinite(q).all():raise ValueError(f'invalid qpos: {path}')
    norms=np.linalg.norm(q[:,3:7],axis=1)
    if np.max(np.abs(norms-1))>1e-3:raise ValueError(f'invalid quaternion: {path}')
    q[:,3:7]/=norms[:,None]
    r=Rotation.from_quat(q[:,[4,5,6,3]])
    w=np.empty((len(q),3));w[1:-1]=(r[2:]*r[:-2].inv()).as_rotvec()/0.04
    w[0]=(r[1]*r[0].inv()).as_rotvec()/0.02;w[-1]=(r[-1]*r[-2].inv()).as_rotvec()/0.02
    state=np.concatenate([q,np.gradient(q[:,:3],0.02,axis=0),w,np.gradient(q[:,7:],0.02,axis=0)],axis=1)
    assert state.shape[1]==55 and np.isfinite(state).all()
    half=state.astype(np.float16)
    if not np.isfinite(half).all():raise ValueError(f'float16 overflow: {path}')
    return half,{'name':path.stem,'source':str(path),'sha256':hashlib.sha256(blob).hexdigest(),'frames':len(q),
                 'fp16_max_error':float(np.max(np.abs(half.astype(np.float64)-state)))}

def build(paths,out):
    assert not out.exists(),out
    lengths=[];starts=[];names=[];proof=[];chunks=[];total=0;started=time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for i,(arr,record) in enumerate(pool.map(load_motion,paths)):
            names.append(record['name']);lengths.append(len(arr));starts.append(total);total+=len(arr)
            chunks.append(arr);proof.append(record)
            if (i+1)%5000==0:print(json.dumps({'clips':i+1,'frames':total,'seconds':round(time.time()-started)}),flush=True)
    ref=np.concatenate(chunks);del chunks
    pack={'ref':torch.from_numpy(ref),'names':names,'starts':starts,'lengths':lengths,'fps':50,
          'layout':'p3,quat_wxyz4,joint21,v_world3,omega_world3,joint_velocity21','robot':'mini3_v031'}
    tmp=out.with_suffix('.partial');torch.save(pack,tmp);os.replace(tmp,out)
    out.with_suffix('.manifest.json').write_text(json.dumps({'clips':len(names),'frames':total,'fps':50,'sources':proof,
        'stems_sha256':hashlib.sha256('\n'.join(sorted(names)).encode()).hexdigest(),'bytes':out.stat().st_size,
        'no_action_based_filtering':True,'state_layout':pack['layout']},ensure_ascii=False))
    print(json.dumps({'saved':str(out),'clips':len(names),'frames':total,'bytes':out.stat().st_size}),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--benchmark',required=True);a=p.parse_args()
    src=Path(a.source);out=Path(a.output);out.mkdir(exist_ok=True)
    train=sorted((src/'sonic_train/motions').rglob('*.npz'));test=sorted((src/'sonic_test/motions').rglob('*.npz'))
    assert len(train)==119925 and len(test)==6316
    train_names={x.stem for x in train};test_names={x.stem for x in test};assert not train_names&test_names
    bench=Path(a.benchmark);full={Path(x.strip()).stem for x in (bench/'splits/full.txt').read_text().splitlines() if x.strip()}
    # Dataset split files can contain IDs resolved by clips metadata; inspect rather than guess.
    if not full<=test_names:raise ValueError(f'benchmark IDs not source stems: {list(full-test_names)[:3]}')
    assert len(full)==6000
    dev=[x for x in test if x.stem not in full];assert len(dev)==316
    (out/'split-proof.json').write_text(json.dumps({'train':119925,'development':316,'test':6000,
        'train_names':sorted(train_names),'development_names':[x.stem for x in dev],'final_test_names':sorted(full)},ensure_ascii=False))
    build(dev,out/'dev316.pt')
    simple=[x for x in train if x.stem.startswith(('walk_ff_loop_','walk_ff_start_','jog_ff_loop_','squat_','crouch_','dance_basic_'))]
    # Freeze a deterministic balanced smoke curriculum before observing policy outcomes.
    selected=[]
    for prefix in ['walk_ff_loop_','walk_ff_start_','jog_ff_loop_','squat_','crouch_','dance_basic_']:
        group=[x for x in simple if x.stem.startswith(prefix)]
        selected.extend(group[:2])
    assert len(selected)>=4
    build(selected,out/'pilot.pt')
    build(train,out/'train.pt')

if __name__=='__main__':main()
