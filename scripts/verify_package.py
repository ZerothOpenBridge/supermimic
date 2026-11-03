"""Read-only contract/golden/consumer validation. No native training imports."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

D = Path('/mnt/cpfs/workspace/yxq/supermimic/shared/data/mini3_v1')
sys.path.insert(0, str(D / 'contract-tooling/tools'))
from onnx_contract import validate_package
from teleop_evaluator.artifacts import ArtifactBundle
from teleop_evaluator.policy.supermimic import SuperMimicAdapter
from teleop_evaluator.policy.supermimic_features import Features
from teleop_evaluator.policy.supermimic_contract import DEFAULT
from teleop_evaluator.robot.base import RobotState


def verify_producers(package, validation, backend):
    path = package.parent / (package.name + '-source-parity.json')
    if not path.is_file():
        return {'status': 'not_supplied', 'note': 'Golden tests passed; raw producer sidecars were not checked'}
    report = json.loads(path.read_text())
    assert report['passed'] and report['backend'] == backend
    assert report['package_validation']['package_id'] == validation['package_id']
    assert hashlib.sha256((package / 'policy.onnx').read_bytes()).hexdigest() == report['final_onnx_sha256']
    ticks, maximum = 0, 0.
    for case in report['cases']:
        relative = Path(case['producer_conformance'])
        assert relative.name == str(relative), 'Producer fixtures must be sibling files'
        fixture = path.parent / relative
        assert hashlib.sha256(fixture.read_bytes()).hexdigest() == case['producer_conformance_sha256']
        with np.load(fixture) as z:
            f = Features(backend, DEFAULT)
            f.reset(z['reference_qpos'])
            for j, tick in enumerate(z['tick']):
                q, v = z['robot_qpos'][j], z['robot_qvel'][j]
                state = RobotState(q, v, q[:3], q[3:7], q[7:], v[6:])
                feed = f.assemble(state, int(tick))
                for key, expected in [('obs_hist', z['expected_observation'][j]), ('goal', z['expected_goal'][j])]:
                    np.testing.assert_allclose(feed[key][0], expected, atol=1e-5, rtol=1e-5)
                    maximum = max(maximum, float(np.max(np.abs(feed[key][0] - expected))))
                f.accept_action(z['previous_raw_action'][j+1] if j+1 < len(z['tick']) else np.zeros(21))
                ticks += 1
    assert ticks >= 64
    return {'status': 'passed', 'ticks': ticks, 'maximum_absolute_error': maximum, 'report': str(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    args = parser.parse_args()
    assert mujoco.__version__ == '3.12.0' and ort.__version__ == '1.29.0'
    validation = validate_package(args.package, run_tests=True)
    bundle = ArtifactBundle.from_mapping({'contract': str(args.package / 'onnx_contract.yaml')})
    schema = SuperMimicAdapter.inspect_artifacts(bundle)
    adapter = SuperMimicAdapter(bundle)
    adapter.create_session(['CPUExecutionProvider'], ort_options={
        'ort_intra_op_threads': 1, 'ort_inter_op_threads': 1,
        'ort_execution_mode': 'sequential', 'ort_graph_optimization': 'basic',
    })
    adapter.close()
    producers = verify_producers(args.package, validation, adapter.backend)
    print(json.dumps({
        'passed': True, 'package': str(args.package),
        'validation': validation, 'adapter': schema.adapter_id,
        'backend': adapter.backend, 'mujoco': mujoco.__version__,
        'ort': ort.__version__, 'training_framework_imported': 'torch' in sys.modules,
        'native_supermimic_imported': 'supermimic' in sys.modules,
        'producer_conformance': producers,
    }, indent=2))


if __name__ == '__main__':
    main()
