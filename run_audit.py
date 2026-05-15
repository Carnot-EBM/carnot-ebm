import sys
import glob
from pathlib import Path
from scripts.adversarial_verify import verify_artifact

files = glob.glob("results/experiment_17*.json") + glob.glob("results/experiment_21*.json")

target_files = []
for f in files:
    try:
        parts = f.split('_')
        exp_id_str = parts[1]
        exp_id = int(exp_id_str)
        if (1709 <= exp_id <= 1716) or (2101 <= exp_id <= 2114):
            target_files.append(Path(f))
    except Exception:
        pass

for f in sorted(target_files):
    report = verify_artifact(f)
    flags = report.get("flags", [])
    if flags:
        print(f"File: {f}")
        for flag in flags:
            print(f"  - {flag['severity']} {flag['kind']}: {flag['detail']}")
