"""Frozen common Mini3 evaluator; full6000 is never used for model selection."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--package',required=True);p.add_argument('--output',required=True);a=p.parse_args()
out=Path(a.output)
if out.exists():raise FileExistsError(out)
import mujoco,onnxruntime,yaml
from teleop_evaluator.api import EvaluationRequest,evaluate
assert mujoco.__version__=='3.12.0' and onnxruntime.__version__=='1.29.0','Use docs/ENVIRONMENT.md common-evaluation runtime'
D=Path('/mnt/cpfs/workspace/yxq/supermimic/shared/data/mini3_v1')
config=Path(a.output).parent/'ort-one-thread.yaml'
config.write_text('ort_intra_op_threads: 1\nort_inter_op_threads: 1\nort_execution_mode: sequential\nort_graph_optimization: basic\n')
contract=yaml.safe_load((Path(a.package)/'onnx_contract.yaml').read_text())
result=evaluate(EvaluationRequest(adapter='supermimic',model={'contract':str(Path(a.package)/'onnx_contract.yaml')},dataset_dir=str(D/'benchmark'),split='full',robot='mini3_v031',profile='mini3_teleop_v1',output_dir=str(out),workers=32,device=('cpu',),seeds=(0,),render='none',checksums='full',resume=False,user_config=str(config),extra={'evaluator_identity':json.loads((D/'evaluator-frozen-v1/identity.json').read_text())['overlay_tree_sha256'],'diagnostics':{'fresh_fk':True,'trace_dir':str(out/'scoring-traces'),'trace_clip_ids':[],'package_id':contract['package']['package_id'],'interface_id':contract['package']['interface_id']}}))
assert len(result.clip_results)==6000 and len({(r['clip_id'],r['seed']) for r in result.clip_results})==6000
assert all(r['status'] in ('completed','failed') for r in result.clip_results)
print(json.dumps({'tasks':6000,'summary':result.summary,'scope':'Frozen scoring protocol; historical fresh-FK/scoring-trace extension remains at CPFS evaluate_batch.py'},indent=2))
