#!/usr/bin/env python3
"""Runner for Exp 3867 PolarFire SoC Ising dispatch v4.

Spec refs: REQ-HW-3867, SCENARIO-HW-3867.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))


def main() -> None:
    from carnot.experiment_3867_polarfire_soc_smoke_v4 import (
        OUTPUT_REL_PATH,
        run_experiment,
    )

    artifact = run_experiment(repo_root=REPO_ROOT)
    print(f"artifact: {OUTPUT_REL_PATH}")
    print(f"honest_verdict: {artifact['honest_verdict']}")
    print(f"polarfire_workload_validated: {artifact['polarfire_workload_validated']}")
    print(f"result_hash_match: {artifact['result_hash_match']}")
    print(f"run_duration_s: {artifact['run_duration_s']}")
    print(f"soc_temp_max_c: {artifact['soc_temp_max_c']}")


if __name__ == "__main__":  # pragma: no cover
    main()
