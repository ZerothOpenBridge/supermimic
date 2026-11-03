"""Reference FK and differentiability check against fresh MuJoCo kinematics (CPU)."""
import argparse,json
from pathlib import Path
import mujoco,numpy as np,torch
from scipy.spatial.transform import Rotation
p=argparse.ArgumentParser();p.add_argument('--backend',required=True);a=p.parse_args()
from supermimic.mini3.spec import CONTROL,TRACKED_BODIES
path=CONTROL['mjcf'];m=mujoco.MjModel.from_xml_path(path);d=mujoco.MjData(m)
rng=np.random.default_rng(0);n=128;q=np.broadcast_to(m.qpos0,(n,m.nq)).copy();q[:,:3]=rng.normal(0,.2,(n,3));q[:,2]+=.5
rot=Rotation.random(n,random_state=rng);q[:,3:7]=rot.as_quat()[:,[3,0,1,2]];q[:,7:]=rng.uniform(m.jnt_range[1:,0],m.jnt_range[1:,1],(n,21));v=rng.normal(0,1,(n,27))
ids=[m.body(b).id for b in TRACKED_BODIES];P=[];R=[];V=[];W=[]
for qi,vi in zip(q,v):
 d.qpos[:]=qi;d.qvel[:]=vi;mujoco.mj_forward(m,d);P.append(d.xpos[ids].copy());R.append(d.xmat[ids].reshape(-1,3,3).copy());bv=[];bw=[]
 for b in ids:
  vel=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_XBODY,b,vel,0);bw.append(vel[:3]);bv.append(vel[3:])
 V.append(bv);W.append(bw)
qt=torch.tensor(q,dtype=torch.float32,requires_grad=True);vt=torch.tensor(v,dtype=torch.float32)
if a.backend=='cuboi':
 from supermimic.kinematics.g1_fk import G1Kinematics
 from supermimic.kinematics.rotations import quat_apply
 def quat_to_matrix(q):
  return torch.stack([quat_apply(q,torch.eye(3)[i].expand(*q.shape[:-1],3)) for i in range(3)],-1)
 kin=G1Kinematics(m);o=kin.forward(qt,vt);ii=kin.body_indices(TRACKED_BODIES);pred=[o['pos'][:,ii],quat_to_matrix(o['quat'][:,ii]),o['lin_vel'][:,ii],o['ang_vel'][:,ii]]
else:
 from supermimic.kinematics import G1Kinematics
 kin=G1Kinematics('cpu');rt=torch.tensor(rot.as_matrix(),dtype=torch.float32);omega=(rt@vt[:,3:6,None]).squeeze(-1)
 pred=kin.forward(qt[:,:3],rt,vt[:,:3],omega,qt[:,7:],vt[:,6:])
err={key:float(np.max(np.abs(x.detach().numpy()-np.asarray(y)))) for key,x,y in zip(['position','rotation_matrix','linear_velocity','angular_velocity'],pred,[P,R,V,W])}
pred[0].square().sum().backward();assert torch.isfinite(qt.grad).all() and qt.grad[:,7:].abs().max()>0
assert max(err.values())<2e-5,err
out={'backend':a.backend,'samples':n,'max_absolute_error':err,'gradient_finite':True,'passed':True,'mujoco':mujoco.__version__}
print(json.dumps(out))
