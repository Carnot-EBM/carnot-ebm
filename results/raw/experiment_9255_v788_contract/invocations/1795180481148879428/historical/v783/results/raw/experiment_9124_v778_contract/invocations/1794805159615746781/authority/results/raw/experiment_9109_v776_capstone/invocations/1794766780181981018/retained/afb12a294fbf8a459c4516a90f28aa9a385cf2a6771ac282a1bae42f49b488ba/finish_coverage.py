"""Require real, current, owned-only statements with no exclusions."""
import json
from pathlib import Path
import sys
from carnot.reporting import human_authority_9097 as d
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v756_branch_qualification import measured, verify_measured
raw = Path(sys.argv[1])
report = measured(list((raw / 'coverage').glob('.coverage.*')), d.OWNED, raw / 'measured')
atomic_json(raw / 'owned_coverage.json', report)
print(json.dumps(report), flush=True)
raise SystemExit(0 if verify_measured(report, d.OWNED) else 1)
