"""Portable full single-rank checkpoint contents for Mini3 zita training."""
import random,numpy as np,torch
from .motor import MotorTorqueEngine

def tensors(obj):
    return {k:v.detach().cpu().clone() if torch.is_tensor(v) else v for k,v in vars(obj).items() if torch.is_tensor(v) or type(v) in (int,float,bool)}
def restore_fields(obj,state):
    for k,v in state.items():
        old=getattr(obj,k)
        if torch.is_tensor(v):
            assert old.shape==v.shape,(k,old.shape,v.shape);old.copy_(v.to(old.device))
        else:setattr(obj,k,v)
def capture(env,buffer,obs_hist,step_acc,counts):
    return {'version':1,'env':tensors(env),'physics_state':tuple(x.cpu() for x in env.state()),
            'buffer':{k:v.cpu().clone() for k,v in buffer.data.items()},'ptr':buffer.ptr,'size':buffer.size,
            'obs_hist':obs_hist.cpu().clone(),'step_acc':step_acc,'counts':counts.copy(),
            'rng':{'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}}
def restore(env,buffer,snapshot):
    assert snapshot['version']==1
    restore_fields(env,snapshot['env'])
    p,R,v,w,q,qd=(x.to(env.device) for x in snapshot['physics_state'])
    from ..rotations import mat_to_quat
    quat=mat_to_quat(R);pose=torch.cat((p+env.scene.env_origins,quat[:,1:4],quat[:,:1]),-1)
    env.robot.write_root_link_pose_to_sim_index(root_pose=pose)
    env.robot.write_root_link_velocity_to_sim_index(root_velocity=torch.cat((v,w),-1))
    env.robot.write_joint_position_to_sim_index(position=q);env.robot.write_joint_velocity_to_sim_index(velocity=qd)
    env.robot.set_coms_index(coms=(env.nominal_torso_com+env.xi[:,22:25]).unsqueeze(1),body_ids=[env.torso_id]);env._apply_friction()
    env.scene.write_data_to_sim();env.scene.update(0.)
    for k,v in snapshot['buffer'].items():buffer.data[k].copy_(v.to(env.device))
    buffer.ptr,buffer.size=snapshot['ptr'],snapshot['size']
    rng=snapshot['rng'];random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in rng['cuda']])
    return snapshot['obs_hist'].to(env.device),snapshot['step_acc'],snapshot['counts'].copy()
