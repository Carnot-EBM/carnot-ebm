#!/usr/bin/env python3
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.reporting.archive_v328_activate_v329_3572 import write_artifact

def main() -> None:
    start = time.perf_counter()

    artifact_path = write_artifact()
    artifact = json.loads(artifact_path.read_text())
    artifact["duration_s"] = round(time.perf_counter() - start, 6)
    artifact_path.write_text(json.dumps(artifact, indent=2))

    print(f"Written: {artifact_path}")
    print(f"honest_verdict: {artifact['honest_verdict']}")

if __name__ == "__main__":
    main()
