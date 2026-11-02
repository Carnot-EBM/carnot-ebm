#!/usr/bin/env python3
"""Runner for Exp 3901 PolarFire plus KV260 continuity audit.

Spec refs: REQ-HW-3901, SCENARIO-HW-3901.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))


def main() -> None:
    from carnot.experiment_3901_polarfire_kv260_continuity import run_experiment

    out_path = run_experiment(repo_root=REPO_ROOT)
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    print(f"artifact: {out_path.relative_to(REPO_ROOT)}")
    print(f"honest_verdict: {artifact['honest_verdict']}")
    print(f"polarfire_reachable: {artifact['polarfire_reachable']}")
    print(f"kv260_reachable: {artifact['kv260_reachable']}")
    print(f"polarfire_state: {artifact['polarfire_state']}")
    print(f"kv260_state: {artifact['kv260_state']}")
    print(f"fabric_acceleration_claimed: {artifact['fabric_acceleration_claimed']}")


if __name__ == "__main__":  # pragma: no cover
    main()
