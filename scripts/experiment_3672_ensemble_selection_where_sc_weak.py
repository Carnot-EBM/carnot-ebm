#!/usr/bin/env python3
"""Run Exp 3672 ensemble selection where self-consistency is weak.

Spec: REQ-VERIFY-3672, SCENARIO-VERIFY-3672.
"""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.verify.ensemble_selection_sc_weak_3672 import (  # noqa: E402
    OUTPUT_REL_PATH,
    run_experiment,
)


def main() -> int:
    artifact = run_experiment(
        repo_root=REPO_ROOT,
        output_path=REPO_ROOT / OUTPUT_REL_PATH,
    )
    print(artifact["honest_verdict"])
    print(
        "sc_accuracy="
        f"{artifact['sc_accuracy']} "
        "oracle_bestofn_accuracy="
        f"{artifact['oracle_bestofn_accuracy']} "
        "ensemble_selection_accuracy="
        f"{artifact['ensemble_selection_accuracy']} "
        "confidence_selection_accuracy="
        f"{artifact['confidence_selection_accuracy']} "
        "flip_count="
        f"{artifact['flip_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
