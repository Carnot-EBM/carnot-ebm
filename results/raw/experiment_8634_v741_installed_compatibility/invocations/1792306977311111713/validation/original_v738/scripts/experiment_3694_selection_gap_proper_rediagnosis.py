#!/usr/bin/env python3
"""Run Exp 3694 proper selection-gap rediagnosis.

Spec: REQ-VERIFY-3694, SCENARIO-VERIFY-3694.
"""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.verify.selection_gap_proper_rediagnosis_3694 import (  # noqa: E402
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
        "per_candidate_auroc="
        f"{artifact['per_candidate_auroc']} "
        "within_question_rank_corr="
        f"{artifact['within_question_rank_corr']} "
        "sc_selection_accuracy="
        f"{artifact['sc_selection_accuracy']} "
        "oracle_bestofn_accuracy="
        f"{artifact['oracle_bestofn_accuracy']} "
        "non_degeneracy_assert="
        f"{artifact['non_degeneracy_assert']} "
        "positive_control_valid="
        f"{artifact['positive_control_valid']} "
        "selection_gap_closed="
        f"{artifact['selection_gap_closed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

