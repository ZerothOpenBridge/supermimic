"""Read-only CPFS + import + source identity validation. CUDA stays hidden."""
from pathlib import Path
import hashlib,json,importlib.metadata,os,sys
root=Path(__file__).resolve().parents[1]
meta=json.loads((root/'configs/implementation.json').read_text())
assets=json.loads((root/'configs/cpfs-assets.json').read_text())
for row in assets['required']:
 p=Path(row['path'])
 if not p.exists():raise FileNotFoundError(p)
 if row.get('bytes') is not None and p.is_file() and p.stat().st_size!=row['bytes']:raise ValueError('Size mismatch: '+str(p))
for relative,digest in json.loads((root/'provenance/native-source-sha256.json').read_text()).items():
 p=root/relative
 if hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise ValueError('Native source changed: '+relative)
import supermimic
import supermimic.mini3.spec as spec
expected=root/('src/supermimic' if meta['backend']=='cuboi' else 'supermimic')
actual=Path(next(iter(supermimic.__path__))).resolve()
if actual!=expected.resolve():raise RuntimeError('Imported another checkout: '+str(actual))
assert len(spec.JOINT_NAMES)==21 and len(spec.DEFAULT_POS)==21
control=Path('/mnt/cpfs/workspace/yxq/supermimic/shared/data/mini3_v1/control/control.json')
assert hashlib.sha256(control.read_bytes()).hexdigest()==assets['control_sha256']
versions={k:importlib.metadata.version(k) for k in ['torch','mujoco','warp-lang','numpy','scipy']}
print(json.dumps({'passed':True,'backend':meta['backend'],'python':sys.version,'imported_source':str(actual),'joint_count':21,'control_sha256':assets['control_sha256'],'versions':versions,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'large_data_loaded':False,'gpu_simulation_started':False},indent=2))
