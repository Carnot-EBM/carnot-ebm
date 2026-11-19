from pathlib import Path
import sys
from carnot.reporting import human_methods_9084 as d
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v756_branch_qualification import measured
p=Path(sys.argv[1])
v=measured(list(p.glob('.coverage.*')),d.OWNED,p/'measured')
atomic_json(p/'coverage.json',v)
print(v,flush=True)
raise SystemExit(0 if v['complete'] else 1)
