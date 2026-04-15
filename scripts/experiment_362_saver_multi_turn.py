#!/usr/bin/env python3
"""Experiment 362 — SAVeR Multi-Turn Verification Wrapper.

**Researcher summary:**
    Implements and benchmarks the SAVeR (Self-Auditing Verification and Repair)
    multi-turn verification wrapper from arXiv 2604.08401.

    SAVeR inserts an auditor between each agent reasoning step: before a step's
    conclusion is allowed to inform the next step, it is checked against
    accumulated constraints from prior steps.  If violations are found, repair
    is attempted.  If repair fails after ``max_repair_attempts``, the step is
    blocked and does not poison downstream conclusions.

    This experiment evaluates SAVeR on 5 multi-step math reasoning chains and
    measures:

        Primary:   faithfulness — fraction of steps that commit (0.0–1.0)
        Secondary: repairs per chain, blocked steps, accuracy with/without SAVeR

    When ``CARNOT_FORCE_LIVE=1`` and a GPU-backed ThreeTierPipeline is available:
        Uses a live VerifyRepairPipeline for real constraint checking.

    When ``CARNOT_FORCE_LIVE`` is unset (CI / default):
        Uses a ``pipeline=None`` CI-safe stub — all steps commit, all chains
        show 100% faithfulness.  This exercises all SAVeR code paths without
        requiring a model.

**Architecture:**
    SAVeRVerifier wraps VerifyRepairPipeline (or None for CI):
    - propose_step(): verify action_cot → repair if needed → commit or block
    - run_chain(): iterate steps, propagate ConstraintState
    - compute_faithfulness(): fraction of committed steps

**Output:** results/experiment_362_saver_multi_turn.json

Spec: REQ-AGENT-001, REQ-AGENT-002,
      SCENARIO-AGENT-001, SCENARIO-AGENT-002, SCENARIO-AGENT-003
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo root on sys.path so scripts/ and python/ imports resolve.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from carnot.pipeline.saver_verifier import (
    AgentStep,
    ConstraintState,
    SAVeRVerifier,
    build_saver_artifact,
)
from scripts.experiment_template import ExperimentTemplate

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DELIVERABLE = "results/experiment_362_saver_multi_turn.json"
MAX_REPAIR_ATTEMPTS = 3
FORCE_LIVE = os.environ.get("CARNOT_FORCE_LIVE", "0") == "1"

# ---------------------------------------------------------------------------
# 5 multi-step reasoning chains (math word problems)
# Each chain has 3-5 reasoning steps.  Ground-truth final answer included for
# accuracy comparison (with and without SAVeR).
#
# Format: {"chain_id": int, "description": str, "steps": [(question, cot)],
#          "ground_truth_answer": str}
# ---------------------------------------------------------------------------

_CHAINS: list[dict] = [
    {
        "chain_id": 0,
        "description": "Shopping change calculation (3 steps)",
        "steps": [
            (
                "Step 1: What is the total cost of items costing $3.50, $5.25, and $2.75?",
                "Adding costs: 3.50 + 5.25 = 8.75, then 8.75 + 2.75 = 11.50. "
                "Total cost is $11.50.",
            ),
            (
                "Step 2: How much change from $20 after paying $11.50?",
                "Change = 20.00 - 11.50 = 8.50. The change is $8.50.",
            ),
            (
                "Step 3: If the customer gives an extra $0.50 tip, what is the final payment?",
                "Final payment = 11.50 + 0.50 = 12.00. The customer pays $12.00.",
            ),
        ],
        "ground_truth_answer": "$8.50 change",
    },
    {
        "chain_id": 1,
        "description": "Train distance calculation (3 steps)",
        "steps": [
            (
                "Step 1: A train travels 60 mph for 2 hours. How far does it go?",
                "Distance = speed * time = 60 * 2 = 120 miles.",
            ),
            (
                "Step 2: The train then travels 80 mph for 1.5 hours. How far in this leg?",
                "Distance = 80 * 1.5 = 120 miles.",
            ),
            (
                "Step 3: What is the total distance for both legs?",
                "Total = 120 + 120 = 240 miles.",
            ),
        ],
        "ground_truth_answer": "240 miles",
    },
    {
        "chain_id": 2,
        "description": "Rectangle perimeter and area (4 steps)",
        "steps": [
            (
                "Step 1: A rectangle is 8 meters wide and 5 meters tall. What is the perimeter?",
                "Perimeter = 2 * (width + height) = 2 * (8 + 5) = 2 * 13 = 26 meters.",
            ),
            (
                "Step 2: What is the area of the same rectangle?",
                "Area = width * height = 8 * 5 = 40 square meters.",
            ),
            (
                "Step 3: If the width doubles, what is the new area?",
                "New width = 8 * 2 = 16 meters. New area = 16 * 5 = 80 square meters.",
            ),
            (
                "Step 4: How much larger is the new area than the original?",
                "Difference = 80 - 40 = 40 square meters. The new area is 40 sq m larger.",
            ),
        ],
        "ground_truth_answer": "perimeter=26m, area=40sq_m",
    },
    {
        "chain_id": 3,
        "description": "Discounted price with tax (3 steps)",
        "steps": [
            (
                "Step 1: A $120 item has a 25% discount applied. What is the discounted price?",
                "Discount amount = 120 * 0.25 = 30. Discounted price = 120 - 30 = $90.",
            ),
            (
                "Step 2: A 10% sales tax is applied to the discounted price. What is the tax?",
                "Tax = 90 * 0.10 = $9.",
            ),
            (
                "Step 3: What is the final price including tax?",
                "Final price = 90 + 9 = $99.",
            ),
        ],
        "ground_truth_answer": "$99.00",
    },
    {
        "chain_id": 4,
        "description": "Worker scheduling (4 steps)",
        "steps": [
            (
                "Step 1: If 3 workers take 4 days to complete a job, how many worker-days is the job?",
                "Worker-days = 3 workers * 4 days = 12 worker-days.",
            ),
            (
                "Step 2: How many workers are needed to finish the same job in 2 days?",
                "Workers = 12 worker-days / 2 days = 6 workers.",
            ),
            (
                "Step 3: If workers cost $150/day each, what is the total cost for the 2-day scenario?",
                "Cost = 6 workers * 2 days * $150/day = $1,800.",
            ),
            (
                "Step 4: Compare to the original cost (3 workers for 4 days).",
                "Original cost = 3 * 4 * 150 = $1,800. Both scenarios cost the same.",
            ),
        ],
        "ground_truth_answer": "6 workers needed",
    },
]

# ---------------------------------------------------------------------------
# Ground-truth answers for accuracy comparison
# A simple keyword match heuristic — sufficient for CI validation.
# ---------------------------------------------------------------------------

_CORRECT_KEYWORDS: dict[int, list[str]] = {
    0: ["8.50"],
    1: ["240"],
    2: ["26", "40"],
    3: ["99"],
    4: ["6"],
}


def _check_accuracy_without_saver(chain: dict) -> bool:
    """Check chain accuracy without SAVeR (use the last step's CoT verbatim).

    In CI stub mode, the final step's CoT is the proposed action, which is
    pre-written to be correct.  In live mode, this would use raw LLM output.

    Spec: REQ-AGENT-001
    """
    if not chain["steps"]:
        return False
    last_cot = chain["steps"][-1][1].lower()
    keywords = _CORRECT_KEYWORDS.get(chain["chain_id"], [])
    return all(kw in last_cot for kw in keywords)


def _check_accuracy_with_saver(agent_steps: list[AgentStep], chain: dict) -> bool:
    """Check chain accuracy with SAVeR (use committed final step's action).

    When a step is committed, we trust its content.  When all steps pass,
    the final committed step's proposed_action is checked against keywords.

    Spec: REQ-AGENT-001
    """
    committed_steps = [s for s in agent_steps if s.committed]
    if not committed_steps:
        return False
    final_action = committed_steps[-1].proposed_action.lower()
    keywords = _CORRECT_KEYWORDS.get(chain["chain_id"], [])
    return all(kw in final_action for kw in keywords)


# ---------------------------------------------------------------------------
# Main experiment function
# ---------------------------------------------------------------------------


def run_experiment(tmpl: ExperimentTemplate) -> dict:
    """Run the SAVeR multi-turn verification experiment.

    Spec: REQ-AGENT-001, REQ-AGENT-002, SCENARIO-AGENT-001
    """
    # ------------------------------------------------------------------
    # Setup pipeline
    # ------------------------------------------------------------------
    if FORCE_LIVE:
        # Live mode: attempt to load VerifyRepairPipeline.
        # If GPU is not available or imports fail, fall back to CI stub.
        try:
            from carnot.pipeline.verify_repair import VerifyRepairPipeline

            pipeline = VerifyRepairPipeline(
                model=None,  # verify-only mode (no LLM repair in live mode)
                max_repairs=MAX_REPAIR_ATTEMPTS,
            )
        except Exception:
            pipeline = None
    else:
        pipeline = None  # CI-safe stub: all steps approved

    inference_mode = "live_gpu" if (FORCE_LIVE and pipeline is not None) else "ci_stub"

    verifier = SAVeRVerifier(pipeline=pipeline, max_repair_attempts=MAX_REPAIR_ATTEMPTS)

    # ------------------------------------------------------------------
    # Run all chains
    # ------------------------------------------------------------------
    chain_results = []
    total_repairs = 0
    total_blocked = 0
    n_correct_with_saver = 0
    n_correct_without_saver = 0

    for chain in _CHAINS:
        initial_state = ConstraintState(model_id=f"chain-{chain['chain_id']}")
        agent_steps = verifier.run_chain(chain["steps"], initial_state)
        faithfulness = verifier.compute_faithfulness(agent_steps)

        n_repairs = sum(s.repair_attempts for s in agent_steps)
        n_blocked = sum(1 for s in agent_steps if not s.committed)
        total_repairs += n_repairs
        total_blocked += n_blocked

        acc_with = _check_accuracy_with_saver(agent_steps, chain)
        acc_without = _check_accuracy_without_saver(chain)
        if acc_with:
            n_correct_with_saver += 1
        if acc_without:
            n_correct_without_saver += 1

        artifact = build_saver_artifact(agent_steps, faithfulness)
        chain_results.append(
            {
                "chain_id": chain["chain_id"],
                "description": chain["description"],
                "n_steps": len(agent_steps),
                "faithfulness": faithfulness,
                "n_repairs": n_repairs,
                "n_blocked": n_blocked,
                "accuracy_with_saver": acc_with,
                "accuracy_without_saver": acc_without,
                "saver_artifact": artifact,
            }
        )

    n_chains = len(_CHAINS)
    mean_faithfulness = (
        sum(r["faithfulness"] for r in chain_results) / n_chains if n_chains > 0 else 0.0
    )
    mean_repairs_per_chain = total_repairs / n_chains if n_chains > 0 else 0.0
    accuracy_with_saver = n_correct_with_saver / n_chains if n_chains > 0 else 0.0
    accuracy_without_saver = n_correct_without_saver / n_chains if n_chains > 0 else 0.0
    improvement_delta = accuracy_with_saver - accuracy_without_saver

    return {
        "n_chains": n_chains,
        "mean_faithfulness": mean_faithfulness,
        "mean_repairs_per_chain": mean_repairs_per_chain,
        "n_blocked_steps": total_blocked,
        "accuracy_with_saver": accuracy_with_saver,
        "accuracy_without_saver": accuracy_without_saver,
        "improvement_delta": improvement_delta,
        "inference_mode": inference_mode,
        "chains": chain_results,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Experiment 362 and write the result artifact.

    Spec: REQ-AGENT-001, REQ-AGENT-002
    """
    tmpl = ExperimentTemplate(
        exp_id=362,
        title="SAVeR Multi-Turn Verification Wrapper",
        deliverable=DELIVERABLE,
        requires_gpu=False,
    )
    tmpl.setup()

    results = run_experiment(tmpl)

    artifact = tmpl.build_result(
        {
            "schema": "carnot.saver_verifier.v1",
            **results,
        },
        status="success",
    )

    out_path = _REPO_ROOT / DELIVERABLE
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))

    print(f"Experiment 362 complete.")
    print(f"  n_chains:              {results['n_chains']}")
    print(f"  mean_faithfulness:     {results['mean_faithfulness']:.3f}")
    print(f"  mean_repairs/chain:    {results['mean_repairs_per_chain']:.2f}")
    print(f"  n_blocked_steps:       {results['n_blocked_steps']}")
    print(f"  accuracy_with_saver:   {results['accuracy_with_saver']:.3f}")
    print(f"  accuracy_without_saver:{results['accuracy_without_saver']:.3f}")
    print(f"  improvement_delta:     {results['improvement_delta']:+.3f}")
    print(f"  inference_mode:        {results['inference_mode']}")
    print(f"  Output: {DELIVERABLE}")


if __name__ == "__main__":
    main()
