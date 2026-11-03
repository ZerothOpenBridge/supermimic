"""Fail closed if the frozen evaluator or contract tooling has changed."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/mnt/cpfs/workspace/yxq/supermimic/shared/data/mini3_v1')


def check_files(root, files):
    root = root.resolve()
    for relative, expected in files.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise RuntimeError('Frozen dependency missing or outside root: ' + relative)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError('Frozen dependency changed: ' + relative)


def verify_runtime_sources():
    evaluator = json.loads((ROOT / 'provenance/evaluator-identity.json').read_text())
    deployed = json.loads((DATA / 'evaluator-frozen-v1/identity.json').read_text())
    if evaluator['overlay_tree_sha256'] != deployed['overlay_tree_sha256']:
        raise RuntimeError('Frozen evaluator identity differs from repository pin')
    check_files(DATA / 'evaluator-frozen-v1/evaluator', evaluator['files'])
    tooling = json.loads((ROOT / 'provenance/contract-tooling-sha256.json').read_text())
    check_files(DATA / 'contract-tooling', tooling['files'])
    return {'evaluator': evaluator['overlay_tree_sha256'],
            'evaluator_files': len(evaluator['files']),
            'contract_tooling_files': len(tooling['files']), 'sha256_verified': True}
