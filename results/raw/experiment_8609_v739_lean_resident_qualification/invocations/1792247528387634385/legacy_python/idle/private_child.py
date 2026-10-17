import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,'/home/ianblenke/github.com/ianblenke/carnot/python/carnot/inference')
from carnot.reporting.resident_worker_8572 import serve
try:
    result=serve(Path('/home/ianblenke/github.com/ianblenke/carnot/results/raw/experiment_8609_v739_lean_resident_qualification/invocations/1792247528387634385/legacy_python/idle/configuration.json'),'sha256:6a8a8a072a6960482aec973d7f13b431ac19c2a3d910a96a3fd9915793221bc1','python',0.2)
finally:
    rows=[]
    hashes={}
    for name,module in list(sys.modules.items()):
        path=Path(getattr(module,'__file__','') or '')
        if path.is_file():
            path=path.absolute()
            if str(path) not in hashes:
                hashes[str(path)]='sha256:'+hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append(dict(name=name,path=str(path),sha256=hashes[str(path)]))
    Path('/home/ianblenke/github.com/ianblenke/carnot/results/raw/experiment_8609_v739_lean_resident_qualification/invocations/1792247528387634385/legacy_python/idle/modules.json').write_text(json.dumps(rows,sort_keys=True))
raise SystemExit(result)
