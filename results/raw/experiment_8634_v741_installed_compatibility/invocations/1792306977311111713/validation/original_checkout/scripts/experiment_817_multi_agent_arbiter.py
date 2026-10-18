#!/usr/bin/env python3
"""Exp 817: Multi-Agent Arbiter — score competing agent outputs via EBM energy.

**Researcher summary:**
    Implements the Tier B product "Multi-Agent Arbiter": score competing agent
    responses, pick the lowest-energy response as the final answer.  This
    experiment wires MultiAgentArbiter (backed by VerifyRepairPipeline) into
    a synthetic 3-agent math debate and measures how often the EBM correctly
    identifies the arithmetically correct answer as the winner.

**Evaluation protocol:**
    6 scenarios, each with 3 agent responses (one correct, two wrong).
    arbiter_accuracy = fraction of scenarios where winner_index points to the
    correct agent.

    honest_verdict:
      - "arbiter_correct"  if arbiter_accuracy >= 0.80
      - "arbiter_partial"  if arbiter_accuracy in [0.60, 0.80)
      - "arbiter_incorrect" if arbiter_accuracy < 0.60

Spec: REQ-AGENT-003, REQ-AGENT-004, SCENARIO-AGENT-004
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Force CPU JAX — this experiment uses constraint extraction only, no GPU needed.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from carnot.pipeline.multi_agent_arbiter import MultiAgentArbiter  # noqa: E402
from carnot.pipeline.verify_repair import VerifyRepairPipeline  # noqa: E402
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

EXP_ID = 817
TITLE = "Multi-Agent Arbiter — EBM energy ranking of competing agent outputs"
DELIVERABLE = "results/experiment_817_multi_agent_arbiter.json"
TIMEOUT_MINUTES = 30

# ---------------------------------------------------------------------------
# Synthetic scenarios
# ---------------------------------------------------------------------------

# Each scenario: (question, responses_list, correct_agent_index)
# The correct agent's response contains the arithmetically accurate answer.
# The wrong agents introduce errors of varying magnitudes so the EBM can
# order them by how much they violate the constraint.
SCENARIOS = [
    (
        "What is 47 + 28?",
        [
            "47 + 28 = 75",  # correct
            "47 + 28 = 76",  # off by 1
            "47 + 28 = 70",  # off by 5
        ],
        0,  # expected winner: Agent A
    ),
    (
        "What is 13 + 19?",
        [
            "13 + 19 = 33",  # wrong by 1
            "13 + 19 = 32",  # correct
            "13 + 19 = 25",  # wrong by 7
        ],
        1,  # expected winner: Agent B
    ),
    (
        "What is 100 - 37?",
        [
            "100 - 37 = 50",  # wrong by 13
            "100 - 37 = 64",  # wrong by 1
            "100 - 37 = 63",  # correct
        ],
        2,  # expected winner: Agent C
    ),
    (
        "What is 8 * 7?",
        [
            "8 * 7 = 56",  # correct
            "8 * 7 = 57",  # wrong by 1
            "8 * 7 = 48",  # wrong by 8
        ],
        0,  # expected winner: Agent A
    ),
    (
        "What is 144 / 12?",
        [
            "144 / 12 = 11",  # wrong by 1
            "144 / 12 = 15",  # wrong by 3
            "144 / 12 = 12",  # correct
        ],
        2,  # expected winner: Agent C
    ),
    (
        "What is 55 + 45?",
        [
            "55 + 45 = 95",  # wrong by 5
            "55 + 45 = 100",  # correct
            "55 + 45 = 102",  # wrong by 2
        ],
        1,  # expected winner: Agent B
    ),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    apply_env_autofix()

    tmpl = ExperimentTemplate(
        exp_id=EXP_ID,
        title=TITLE,
        deliverable=DELIVERABLE,
        requires_gpu=False,
    )
    tmpl.setup()

    _watchdog = ExperimentTimeoutWatchdog(EXP_ID, timeout_minutes=TIMEOUT_MINUTES)

    pipeline = VerifyRepairPipeline(model=None)
    arbiter = MultiAgentArbiter(pipeline)

    scenario_results = []
    n_correct = 0

    for scenario_idx, (question, responses, expected_winner) in enumerate(SCENARIOS):
        print(f"\n[Scenario {scenario_idx + 1}/{len(SCENARIOS)}] {question}")

        result = arbiter.rank_agents(question, responses)
        arbiter_winner = result.winner_index
        is_correct = arbiter_winner == expected_winner

        if is_correct:
            n_correct += 1

        scenario_record = {
            "scenario_index": scenario_idx,
            "question": question,
            "expected_winner_index": expected_winner,
            "arbiter_winner_index": arbiter_winner,
            "arbiter_correct": is_correct,
            "winner_energy": result.winner_energy,
            "all_scores": [s.to_dict() for s in result.all_scores],
        }
        scenario_results.append(scenario_record)

        print(
            f"  Expected: Agent {expected_winner}  "
            f"  Got: Agent {arbiter_winner}  "
            f"  Energy: {result.winner_energy:.4f}  "
            f"  {'PASS' if is_correct else 'FAIL'}"
        )

    arbiter_accuracy = n_correct / len(SCENARIOS)

    if arbiter_accuracy >= 0.80:
        honest_verdict = "arbiter_correct"
    elif arbiter_accuracy >= 0.60:
        honest_verdict = "arbiter_partial"
    else:
        honest_verdict = "arbiter_incorrect"

    print(f"\narbiter_accuracy={arbiter_accuracy:.2f}  honest_verdict={honest_verdict}")

    artifact = tmpl.build_result(
        {
            "scenarios": scenario_results,
            "n_scenarios": len(SCENARIOS),
            "n_correct": n_correct,
            "arbiter_accuracy": arbiter_accuracy,
            "honest_verdict": honest_verdict,
            "inference_mode": "cpu_ebm",
            "mcp_tool": "score_agent_outputs",
        },
        status="success",
    )

    output_path = Path(DELIVERABLE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2))
    print(f"\nDeliverable written to {DELIVERABLE}")

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
