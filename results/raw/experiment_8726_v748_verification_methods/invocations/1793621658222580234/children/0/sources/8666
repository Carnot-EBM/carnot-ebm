#!/usr/bin/env python3
"""Runner for Exp 3931 clean hardware continuity rerun.

Spec refs: REQ-HW-3931, SCENARIO-HW-3931.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))


def main() -> None:
    from carnot.experiment_3922_hardware_continuity_consolidated import _gatemate
    from carnot.experiment_3931_hardware_continuity_clean_rerun import run_experiment

    _gatemate.resolve_toolchain_path()
    out_path = run_experiment(repo_root=REPO_ROOT)
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    print(f"artifact: {out_path.relative_to(REPO_ROOT)}")
    print(f"honest_verdict: {artifact['honest_verdict']}")
    print(f"gatemate_reachable: {artifact['gatemate_reachable']}")
    print(f"polarfire_reachable: {artifact['polarfire_reachable']}")
    print(f"kv260_reachable: {artifact['kv260_reachable']}")
    print(f"duration_s: {artifact['duration_s']}")
    print(f"run_duration_s: {artifact['run_duration_s']}")
    print(f"fabric_acceleration_claimed: {artifact['fabric_acceleration_claimed']}")


if __name__ == "__main__":  # pragma: no cover
    main()
