#!/usr/bin/env python3
"""Runner for Exp 4422 KV260 SSH-only hardware continuity.

Spec refs: REQ-HW-4422, SCENARIO-HW-4422.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))


def main() -> None:
    from carnot.experiment_4422_hardware_continuity_kv260 import run_experiment

    out_path = run_experiment(repo_root=REPO_ROOT)
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    print(f"artifact: {out_path.relative_to(REPO_ROOT)}")
    print(f"honest_verdict: {artifact['honest_verdict']}")
    print(f"kv260_reachable: {artifact['kv260_reachable']}")
    print(f"loaded_overlay: {artifact['loaded_overlay']}")
    print(f"uio_present: {artifact['uio_present']}")


if __name__ == "__main__":  # pragma: no cover
    main()
