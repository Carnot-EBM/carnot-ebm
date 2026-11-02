#!/usr/bin/env python3
"""Run Exp 1133: PRM-BiasBench-style adversarial verifier evaluation.

Spec: REQ-VERIFY-1133, SCENARIO-VERIFY-1133
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.eval.prm_biasbench_adversarial import (  # noqa: E402
    OUTPUT_PATH,
    run_experiment,
    write_artifact,
)


def main() -> int:
    """Write the Exp 1133 result artifact and print the headline rates."""

    artifact = run_experiment()
    write_artifact(artifact, OUTPUT_PATH)
    print(
        "[exp1133] "
        f"honest_verdict={artifact['honest_verdict']} "
        f"k5_attack_tp_rate={artifact['k5_attack_tp_rate']:.3f} "
        f"semenergy_alone_attack_tp_rate={artifact['semenergy_alone_attack_tp_rate']:.3f} "
        f"z3_attack_immune={artifact['z3_attack_immune']} "
        f"output={OUTPUT_PATH.relative_to(REPO_ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
