import sys

def append_to_file(filepath, lines):
    with open(filepath, 'a') as f:
        for line in lines:
            f.write(line + '\n')

changelog_lines = [
    "- 2026-06-01: FR-11 continuous self-learning v6 — online-calibrate the REAL grounding verifier threshold via conservative-default; confirm no collapse (mandatory continuous self-learning) (✅ Complete) — honest_verdict=complete: fr11_conservative_default_calibrates_real_grounding_verifier_holds_quality_maintained; results/experiment_3604_fr11_continuous_self_learning_v6.json"
]

status_lines = [
    "| 2026-06-01 | Exp 3604: FR-11 Continuous Self Learning v6 | ✅ Complete | honest_verdict=complete: fr11_conservative_default_calibrates_real_grounding_verifier_holds_quality_maintained; introduces REQ-LEARN-3604 and SCENARIO-LEARN-3604 | results/experiment_3604_fr11_continuous_self_learning_v6.json |"
]

traceability_lines = [
    "| REQ-LEARN-3604 | Exp 3604: FR-11 Continuous Self Learning v6 | Implemented | results/experiment_3604_fr11_continuous_self_learning_v6.json |",
    "| REQ-LEARN-3604-1 | Exp 3604: FR-11 Continuous Self Learning v6 | Implemented | results/experiment_3604_fr11_continuous_self_learning_v6.json |",
    "| REQ-LEARN-3604-2 | Exp 3604: FR-11 Continuous Self Learning v6 | Implemented | results/experiment_3604_fr11_continuous_self_learning_v6.json |",
    "| REQ-LEARN-3604-3 | Exp 3604: FR-11 Continuous Self Learning v6 | Implemented | results/experiment_3604_fr11_continuous_self_learning_v6.json |",
    "| SCENARIO-LEARN-3604 | Exp 3604: FR-11 Continuous Self Learning v6 | Implemented | results/experiment_3604_fr11_continuous_self_learning_v6.json |"
]

append_to_file('ops/changelog.md', changelog_lines)
append_to_file('ops/status.md', status_lines)
append_to_file('_bmad/traceability.md', traceability_lines)

print("Appended to logs.")
