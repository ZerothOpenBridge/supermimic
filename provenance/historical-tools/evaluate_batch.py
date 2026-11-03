"""Same-environment model runs; partial/technical-error batches never become the final table."""
import argparse,hashlib,json,os,time
from pathlib import Path
import mujoco,onnxruntime,yaml
from teleop_evaluator.api import EvaluationRequest,evaluate
R=Path('/mnt/cpfs/workspace/yxq/supermimic');D=R/'shared/data/mini3_v1';E=R/'experiments/mini3_v1'
MODELS={'trackbase':('mimiclite',D/'packages/baselines/mimiclite-online-retargeting-residual-4000'),
 'smpl_adapter_v1':('mimiclite',D/'packages/baselines/mimiclite-smpl-adapter-large-4000'),
 'bfm_final':('ufo',D/'packages/baselines/ufo-fb-bones-seed-191994880'),
 'sonic100k':('sonic',D/'packages/baselines/package_sonic_100000_raw'),
 'supermimic_cuboi':('supermimic',D/'packages/final-cuboi-mini3'),
 'supermimic_zita':('supermimic',D/'packages/final-zita-mini3')}
def main():
 p=argparse.ArgumentParser();p.add_argument('--models',default='trackbase,smpl_adapter_v1,bfm_final,sonic100k');p.add_argument('--smoke',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args()
 (D/'ort-one-thread.yaml').write_text('ort_intra_op_threads: 1\nort_inter_op_threads: 1\nort_execution_mode: sequential\nort_graph_optimization: basic\n')
 assert mujoco.__version__=='3.12.0' and onnxruntime.__version__=='1.29.0'
 selection=json.loads((E/'demo-selection.json').read_text());demo=[x['name'] for x in selection['selection'] if x['dataset_role']=='Full-6000 final test']
 for name in a.models.split(','):
  adapter,package=MODELS[name];contract=yaml.safe_load((package/'onnx_contract.yaml').read_text());out=E/('evaluator-smoke-v1' if a.smoke else 'cohort-full-v1')/name
  req=EvaluationRequest(adapter=adapter,model={'contract':str(package/'onnx_contract.yaml')},dataset_dir=str(D/'benchmark'),split='full',robot='mini3_v031',profile='mini3_teleop_v1',output_dir=str(out),
     workers=2 if a.smoke else 32,device=('cpu',),seeds=(0,),clip=tuple(demo[:2]) if a.smoke else (),render='none',checksums='full',resume=a.resume,
     user_config=str(D/'ort-one-thread.yaml'),
     extra={'evaluator_identity':json.loads((D/'evaluator-frozen-v1/identity.json').read_text())['overlay_tree_sha256'],'cohort':'supermimic-mini3-v1-20261029','diagnostics':{'fresh_fk':True,'trace_dir':str(out/'scoring-traces'),'trace_clip_ids':demo,
                  'package_id':contract['package']['package_id'],'interface_id':contract['package']['interface_id']}})
  started=time.time();result=evaluate(req);rows=result.clip_results
  assert len(rows)==(2 if a.smoke else 6000) and len({(x['clip_id'],x['seed']) for x in rows})==len(rows)
  assert all(x['status'] in ('completed','failed') for x in rows),'Technical errors require diagnosis; batch is not official'
  report={'complete':True,'model':name,'tasks':len(rows),'seconds':time.time()-started,'package_id':contract['package']['package_id'],
          'interface_id':contract['package']['interface_id'],'mujoco':mujoco.__version__,'ort':onnxruntime.__version__,'summary':result.summary,
          'is_final_cohort':not a.smoke,'all_six_models_complete':False}
  (out/'acceptance.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
