"""Write a container-namespace PID receipt so the host budget guard can stop only this task."""
import json,os,runpy,sys
from pathlib import Path
receipt=Path(sys.argv[1]);target=sys.argv[2];sys.argv=sys.argv[2:]
receipt.write_text(json.dumps({'pid':os.getpid(),'target':target}))
runpy.run_path(target,run_name='__main__')
