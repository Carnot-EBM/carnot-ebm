#!/usr/bin/env python3
"""Runner for Exp 4096 hardware continuity.

Spec refs: REQ-HW-4096, SCENARIO-HW-4096.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))


def main() -> None:
    from carnot.experiment_4096_hardware_continuity import run_experiment

    out_path = run_experiment(repo_root=REPO_ROOT)
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    print(f"artifact: {out_path.relative_to(REPO_ROOT)}")
    print(f"honest_verdict: {artifact['honest_verdict']}")
    print(f"per_board_reachability: {artifact['per_board_reachability']}")
    print(f"gatemate_step_taken: {artifact['gatemate_step_taken']}")
    print(f"polarfire_step_taken: {artifact['polarfire_step_taken']}")
    print(f"kv260_terminal_confirmed: {artifact['kv260_terminal_confirmed']}")
    print(f"per_board_duration_s: {artifact['per_board_duration_s']}")
    print(f"duration_s: {artifact['duration_s']}")


if __name__ == "__main__":  # pragma: no cover
    main()
