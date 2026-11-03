"""Shared 55-component reference state, with FK computed only for requested frames."""
from pathlib import Path
import torch
from ..rotations import quat_to_mat

class FeatureView:
    def __init__(self,lib,name):self.lib,self.name=lib,name
    def __getitem__(self,idx):
        r=self.lib._ref[idx].float();p,q,j,v,w,qd=r[...,:3],r[...,3:7],r[...,7:28],r[...,28:31],r[...,31:34],r[...,34:55]
        if self.name=='root_p':return p
        if self.name=='q':return j
        if self.name=='qd':return qd
        if self.name=='root_v':return v
        if self.name=='root_w':return w
        R=quat_to_mat(q/torch.linalg.vector_norm(q,dim=-1,keepdim=True))
        if self.name=='root_R':return R
        if self.name=='root_gz':return R[...,2,2]
        if self.name=='root_v_local':return (R.transpose(-1,-2)@v.unsqueeze(-1)).squeeze(-1)
        if self.name=='root_w_local':return (R.transpose(-1,-2)@w.unsqueeze(-1)).squeeze(-1)
        values=self.lib.kin.forward(p,R,v,w,j,qd)
        return values[{'body_p':0,'body_R':1,'body_v':2,'body_w':3}[self.name]]

def initialize(lib,files,kin,device):
    assert len(files)==1 and str(files[0]).endswith('.pt')
    pack=torch.load(files[0],map_location='cpu',weights_only=False)
    assert pack['robot']=='mini3_v031' and pack['ref'].shape[1]==55
    lib.device=device;lib.kin=kin;lib.names=pack['names'];lib._ref=pack['ref'].to(device)
    lib.clip_len=torch.tensor(pack['lengths'],device=device,dtype=torch.long)
    lib.clip_start=torch.tensor(pack['starts'],device=device,dtype=torch.long)
    lib.clip_end=lib.clip_start+lib.clip_len-1;lib.num_frames=len(lib._ref)
    lib.clip_id=torch.repeat_interleave(torch.arange(len(lib.names),device=device),lib.clip_len)
    for name in ['root_p','root_R','root_v','root_w','q','qd','root_v_local','root_w_local','root_gz','body_p','body_R','body_v','body_w']:
        setattr(lib,name,FeatureView(lib,name))
    counts=(lib.clip_len+49)//50;offset=torch.cumsum(counts,0)-counts
    lib.bin_clip=torch.repeat_interleave(torch.arange(len(lib.names),device=device),counts)
    lib.num_bins=int(counts.sum());b=torch.arange(lib.num_bins,device=device)
    lib.bin_start=lib.clip_start[lib.bin_clip]+50*(b-offset[lib.bin_clip])
    f=torch.arange(lib.num_frames,device=device)
    lib.frame_bin=offset[lib.clip_id]+(f-lib.clip_start[lib.clip_id])//50
    print(f'Mini3 compact library: {len(lib.names)} motions, {lib.num_frames} frames',flush=True)
