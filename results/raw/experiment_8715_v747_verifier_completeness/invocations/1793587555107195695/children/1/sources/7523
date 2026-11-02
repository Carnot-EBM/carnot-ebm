"""Runner for Capstone v325 (Depth-Over-Breadth XI).

Produces results/experiment_3538_capstone_v325.json.

Usage:
    cd /home/ianblenke/github.com/ianblenke/carnot
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_3538_capstone_v325.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

# Project root is two levels up from this script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_PATH = RESULTS_DIR / "experiment_3538_capstone_v325.json"


def main() -> None:
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "python"))

    from carnot.reporting.capstone_v325_3538 import run_capstone

    t0 = time.monotonic()
    result = run_capstone(RESULTS_DIR)
    duration = max(time.monotonic() - t0, 0.0001)   # sub-second floor for aggregation

    result["duration_s"] = round(duration, 6)

    # Re-compute checksum now that duration_s is set (duration excluded from hash)
    import hashlib
    stable = {k: v for k, v in result.items() if k not in ("reproducibility_checksum", "duration_s")}
    result["reproducibility_checksum"] = hashlib.sha256(
        json.dumps(stable, sort_keys=True).encode("utf-8")
    ).hexdigest()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Written: {OUTPUT_PATH}")
    print(f"honest_verdict: {result['honest_verdict']}")
    print(f"capstone_v325_ready: {result['capstone_v325_ready']}")
    print(f"duration_s: {result['duration_s']}")
    print(f"unmet_gates: {result['unmet_gates']}")
    print(f"p0_1_status: {result['p0_1_status'][:80]}...")


if __name__ == "__main__":
    main()
