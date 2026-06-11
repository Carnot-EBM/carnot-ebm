#!/usr/bin/env python3
"""Runner for Exp 4052 hardware continuity.

Spec refs: REQ-HW-4052, SCENARIO-HW-4052.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))


def main() -> None:
    from carnot.experiment_4052_hardware_continuity import run_experiment

    out_path = run_experiment(repo_root=REPO_ROOT)
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    print(f"artifact: {out_path.relative_to(REPO_ROOT)}")
    print(f"honest_verdict: {artifact['honest_verdict']}")
    print(f"per_board_reachability: {artifact['per_board_reachability']}")
    print(f"kv260_overlay_loaded: {artifact['kv260_overlay_loaded']}")
    print(f"kv260_latency_step_taken: {artifact['kv260_latency_step_taken']}")
    print(f"per_board_next_step: {artifact['per_board_next_step']}")
    print(f"per_board_duration_s: {artifact['per_board_duration_s']}")
    print(f"duration_s: {artifact['duration_s']}")


if __name__ == "__main__":  # pragma: no cover
    main()
