#!/usr/bin/env python3
"""Run Exp 4394 ARC E3 deeper fidelity gate.

Spec refs: REQ-VERIFY-4394, SCENARIO-VERIFY-4394.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.agentic.arc_e3_fidelity_gate import ExperimentConfig, run_experiment


def main() -> None:
    config = ExperimentConfig.from_repo_root(REPO_ROOT)
    artifact = run_experiment(config)
    for card in artifact["per_target_scorecard"]:
        rounds = zip(card["verifier_accuracy_per_round"], card["lookahead_fidelity_per_round"])
        for index, (verifier_accuracy, lookahead_fidelity) in enumerate(rounds, start=1):
            print(
                f"{card['game']} round {index}: "
                f"verifier_accuracy={verifier_accuracy:.6f} "
                f"lookahead_fidelity={lookahead_fidelity:.6f} "
                f"gate_passed={card['fidelity_gate_passed']}"
            )
        print(
            f"{card['game']} checkpoint={card['checkpoint_status']} "
            f"offline_reproduced={card['offline_reproduced']} "
            f"new_reproduced_level={card['new_reproduced_level']}"
        )
    print(f"honest_verdict={artifact['honest_verdict']}")
    print(f"reproducible_total_levels={artifact['reproducible_total_levels']}")
    print(f"new_levels_reproduced={artifact['new_levels_reproduced']}")
    print(f"artifact={artifact['artifact_path']}")


if __name__ == "__main__":
    main()
