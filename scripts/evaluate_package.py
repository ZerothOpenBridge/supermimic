"""Frozen common Mini3 evaluator; an explicit clip subset is an interface check."""
import argparse
import json
import sys
from pathlib import Path

import mujoco
import onnxruntime
import yaml
from teleop_evaluator.api import EvaluationRequest, evaluate

D = Path('/mnt/cpfs/workspace/yxq/supermimic/shared/data/mini3_v1')
from runtime_contract import verify_runtime_sources
RUNTIME_IDENTITY = verify_runtime_sources()
sys.path.insert(0, str(D / 'contract-tooling/tools'))
from onnx_contract import validate_package


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--clip', action='append', default=[])
    parser.add_argument('--workers', type=int, default=32)
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        parser.error('--workers must be 1..32')
    if len(set(args.clip)) != len(args.clip):
        parser.error('Duplicate clip IDs are not allowed')
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.output.parent.is_dir():
        raise FileNotFoundError(args.output.parent)
    assert mujoco.__version__ == '3.12.0' and onnxruntime.__version__ == '1.29.0'
    # Never rely on an earlier manual verify step: reject invalid canonical IDs,
    # file hashes, graph signatures and golden outputs before physics starts.
    package_validation = validate_package(args.package, run_tests=True)
    config = args.output.parent / 'ort-one-thread.yaml'
    config.write_text('ort_intra_op_threads: 1\nort_inter_op_threads: 1\n'
                      'ort_execution_mode: sequential\nort_graph_optimization: basic\n')
    contract = yaml.safe_load((args.package / 'onnx_contract.yaml').read_text())
    identity = json.loads((D / 'evaluator-frozen-v1/identity.json').read_text())
    result = evaluate(EvaluationRequest(
        adapter='supermimic', model={'contract': str(args.package / 'onnx_contract.yaml')},
        dataset_dir=str(D / 'benchmark'), split='full', robot='mini3_v031',
        profile='mini3_teleop_v1', output_dir=str(args.output), workers=args.workers,
        device=('cpu',), seeds=(0,), clip=tuple(args.clip), render='none',
        checksums='full', resume=False, user_config=str(config),
        extra={'evaluator_identity': identity['overlay_tree_sha256'],
               'scope': 'interface-check' if args.clip else 'full6000',
               'diagnostics': {'fresh_fk': True,
                   'trace_dir': str(args.output / 'scoring-traces'),
                   'trace_clip_ids': args.clip,
                   'package_id': contract['package']['package_id'],
                   'interface_id': contract['package']['interface_id']}}))
    rows = result.clip_results
    assert len(rows) == (len(args.clip) if args.clip else 6000)
    assert len({(r['clip_id'], r['seed']) for r in rows}) == len(rows)
    assert all(r['status'] in ('completed', 'failed') for r in rows), 'Technical errors'
    if args.clip:
        assert {r['clip_id'] for r in rows} == set(args.clip), 'Clip filter changed'
    receipt = {'passed': True, 'package_validation': package_validation, 'runtime_identity': RUNTIME_IDENTITY, 'tasks': len(rows), 'clips': args.clip,
        'full6000': not bool(args.clip), 'summary': result.summary,
        'package_id': contract['package']['package_id'],
        'interface_id': contract['package']['interface_id'],
        'evaluator_identity': identity['overlay_tree_sha256'],
        'mujoco': mujoco.__version__, 'ort': onnxruntime.__version__,
        'workers': args.workers, 'ort_threads': 1, 'traces': str(args.output / 'scoring-traces')}
    (args.output / 'acceptance.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
