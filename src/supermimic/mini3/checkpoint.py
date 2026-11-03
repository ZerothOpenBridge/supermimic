"""Complete single-rank Mini3 resume state; no replay refill or fresh optimizer on resume."""
import random
import numpy as np
import torch


def fields(obj):
    return {k:v.detach().cpu().clone() if torch.is_tensor(v) else v
            for k,v in vars(obj).items() if torch.is_tensor(v) or type(v) in (int,float,bool)}


def put(obj,state):
    for k,v in state.items():
        old=getattr(obj,k,None)
        if torch.is_tensor(v):
            if old is None:setattr(obj,k,v.to(getattr(obj,'_device','cuda:0')))
            else:
                assert old.shape==v.shape,(k,old.shape,v.shape)
                old.copy_(v.to(old.device))
        else:setattr(obj,k,v)


def objects(t):
    e=t.env
    out={'env':e,'command':t.command,'recorder':t.recorder,'buffer':t.buffer,'action_manager':e.action_manager,
         'action_term':e.action_manager.get_term('joint_pos')}
    for group,terms in e.observation_manager._group_obs_term_history_buffer.items():
        for term,buf in terms.items():out['history/'+group+'/'+term]=buf
    return out


def capture(t):
    e=t.env
    arrays={}
    for key in ['qpos','qvel','act','ctrl','qacc_warmstart','time','qfrc_applied','xfrc_applied']:
        if hasattr(e.sim.data,key):arrays[key]=getattr(e.sim.data,key)[:].detach().cpu().clone()
    model={key:getattr(e.sim.model,key)[:].detach().cpu().clone() for key in ['geom_friction','body_ipos']}
    return {'version':1,'objects':{k:fields(o) for k,o in objects(t).items()},'sim':arrays,'model':model,
            'encoder_bias':e.scene['robot'].data.encoder_bias.detach().cpu().clone(),
            'obs':{k:v.detach().cpu().clone() for k,v in t._last_obs.items()},
            'rng':{'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
                   'cuda':torch.cuda.get_rng_state_all()},'utd':t._utd_acc,'start_iteration':t.start_iteration,
            'ep_reward_sum':t.ep_reward_sum.cpu(),'ep_len':t.ep_len.cpu(),
            'elapsed_training_seconds':getattr(t,'elapsed_training_seconds',0.)}


def restore(t,state):
    assert state['version']==1
    e=t.env
    for key,val in state['model'].items():getattr(e.sim.model,key)[:]=val.to(t.device)
    e.scene['robot'].data.encoder_bias.copy_(state['encoder_bias'].to(t.device))
    for key,val in state['sim'].items():getattr(e.sim.data,key)[:]=val.to(t.device)
    e.sim.forward();e.scene.update(e.physics_dt)
    for name,obj in objects(t).items():put(obj,state['objects'][name])
    t.command._refresh_ref()
    # forward may overwrite warmstart; restore it last.
    if 'qacc_warmstart' in state['sim']:e.sim.data.qacc_warmstart[:]=state['sim']['qacc_warmstart'].to(t.device)
    t._utd_acc=state['utd'];t.start_iteration=state['start_iteration']
    t.ep_reward_sum.copy_(state['ep_reward_sum']);t.ep_len.copy_(state['ep_len'])
    t.elapsed_training_seconds=state['elapsed_training_seconds']
    rng=state['rng'];random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch'].cpu())
    torch.cuda.set_rng_state_all([x.cpu() for x in rng['cuda']])
    return {k:v.to(t.device) for k,v in state['obs'].items()}
