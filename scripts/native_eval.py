"""Native full-length dev316 evaluation; new output file required."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);a=p.parse_args()
out=Path(a.output)
if out.exists():raise FileExistsError(out)
import torch
from supermimic.policy.nets import Policy,PolicyCfg
from supermimic.eval.benchmark import run_benchmark,save_results,summarize
ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
checkpoint=Path(a.checkpoint)
if checkpoint.name=='ckpt_latest.pt':
 checkpoint=checkpoint.parent/('candidate_'+str(ck['iteration']).zfill(7)+'.pt')
 if not checkpoint.exists():torch.save({k:ck[k] for k in ['policy','wm','critic','config','iteration','num_updates']},checkpoint)
c=ck['config']['policy'];pi=Policy(PolicyCfg(**{k:tuple(v) if isinstance(v,list) else v for k,v in c.items()})).cuda().eval();pi.load_state_dict(ck['policy'],strict=True)
with torch.no_grad():rows=run_benchmark(pi,['/mnt/cpfs/workspace/yxq/supermimic/shared/data/mini3_v1/packed/dev316.pt'],num_envs=316,verbose=False)
assert len(rows)==316 and len({r.name for r in rows})==316
save_results(rows,out)
result=json.loads(out.read_text())
if not isinstance(result,dict):result={'clips':result}
result.update(summary=summarize(rows),iteration=ck['iteration'],checkpoint=str(checkpoint));out.write_text(json.dumps(result,indent=2))
print(json.dumps({'iteration':ck['iteration'],'summary':summarize(rows),'output':str(out)},indent=2))
