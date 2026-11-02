#!/usr/bin/env python3
"""Runner for Exp 3983 ARC hardware continuity.

Spec refs: REQ-HW-3983, SCENARIO-HW-3983.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))


def main() -> None:
    from carnot.experiment_3983_hardware_continuity import run_experiment

    out_path = run_experiment(repo_root=REPO_ROOT)
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    print(f"artifact: {out_path.relative_to(REPO_ROOT)}")
    print(f"honest_verdict: {artifact['honest_verdict']}")
    print(f"kv260_reachable: {artifact['kv260_reachable']}")
    print(f"gatemate_reachable: {artifact['gatemate_reachable']}")
    print(f"polarfire_reachable: {artifact['polarfire_reachable']}")
    print(f"duration_s: {artifact['duration_s']}")
    print(f"per_board_duration_s: {artifact['per_board_duration_s']}")


if __name__ == "__main__":  # pragma: no cover
    main()
