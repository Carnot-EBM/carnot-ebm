#!/usr/bin/env python3
"""Exp 835: Arbiter Calibration Fix v2 — Z-Score Normalization.

**Researcher summary:**
    Exp 822 showed accuracy_standard=0.17 (arbiter_still_wrong).  Analysis revealed that
    even though the arbiter correctly uses compute_energy_with_external_field(), the raw
    energy magnitudes vary widely per-query.  Without per-query normalization the consensus
    detection threshold of 0.01 is meaningless — some queries have energy ranges of 0.001,
    others 0.5, and both are treated identically.

    This experiment validates the z-score normalization fix added to MultiAgentArbiter.arbitrate():
    energies are now normalized to (mean=0, std=1) before consensus detection and agent
    selection.  The normalization does NOT change the ORDER of raw energies, but it makes
    the consensus threshold consistent across queries (0.01 standard deviations).

    Because z-score normalization preserves ordering, accuracy improvement depends entirely
    on whether the external field correctly ranks agents for each query.  The experiment
    reports honestly: if the underlying energy still does not discriminate, accuracy stays
    at baseline.

**Scenarios (identical structure to Exp 822):**
    12 synthetic scenarios, CPU-only, fixed seed=42.
    6 standard:    3 agents disagree; arbiter calls score_agents() for real energies.
    6 adversarial: 2 agents share the SAME wrong response string; consensus penalty fires.
    Correct agent is always index 0 in both scenario types.

**honest_verdict:**
    - "arbiter_calibrated"  if accuracy_standard >= 0.67
    - "arbiter_partial"     if 0.50 <= accuracy_standard < 0.67
    - "arbiter_improvement" if 0.17 < accuracy_standard < 0.50  (beat Exp 822 baseline)
    - "arbiter_still_wrong" if accuracy_standard <= 0.17

Spec: REQ-VERIFY-143, REQ-VERIFY-144, SCENARIO-VERIFY-172
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

apply_env_autofix()

import numpy as np  # noqa: E402

from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from carnot.inference.multi_agent_arbiter import MultiAgentArbiter  # noqa: E402
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

EXP_ID = 835
TITLE = "Arbiter Calibration Fix v2 — Z-Score Normalization"
DELIVERABLE = "results/experiment_835_arbiter_calibration_fix_v2.json"
TIMEOUT_MINUTES = 30

SEED = 42
N_SPINS = 16
EMB_DIM = 384
N_STANDARD = 6
N_ADVERSARIAL = 6


def _make_constraint_embeddings(rng: np.random.Generator, n: int = 5) -> list[list[float]]:
    """Generate synthetic constraint embeddings for testing.

    Random unit vectors in embedding space (384-dim) — the same approach as Exp 822.
    These produce a non-zero external field h, giving the arbiter a real energy signal to
    work with rather than zero energies from empty constraint lists.

    Args:
        rng: Seeded RNG for reproducibility across scenarios.
        n: Number of constraint embeddings to generate.

    Returns:
        List of n float lists, each of length EMB_DIM.
    """
    embs = rng.standard_normal((n, EMB_DIM))
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / np.maximum(norms, 1e-8)
    return embs.tolist()


def _run_standard_scenarios(
    arbiter: MultiAgentArbiter, rng: np.random.Generator
) -> list[dict]:
    """Run 6 standard scenarios using real arbitrate() with z-score normalization.

    Standard scenario: 3 agents with distinct responses (correct_std_N, wrong_std_a_N,
    wrong_std_b_N).  The correct agent is index 0.  Real energies are computed by
    score_agents() via compute_energy_with_external_field() — no synthetic assignment.
    The z-score normalized energies are then used for agent selection.

    If the external field correctly ranks the correct agent (index 0) as lowest energy,
    the arbiter picks correctly.  This is the discriminating test for REQ-VERIFY-143/144.

    Args:
        arbiter: MultiAgentArbiter instance with z-score normalization.
        rng: Seeded RNG for generating constraint embeddings.

    Returns:
        List of 6 scenario result dicts with energies_raw, energies_normalized, etc.

    Spec: REQ-VERIFY-143, REQ-VERIFY-144, SCENARIO-VERIFY-172
    """
    results = []
    for i in range(N_STANDARD):
        responses = [f"correct_std_{i}", f"wrong_std_a_{i}", f"wrong_std_b_{i}"]
        constraint_embeddings = _make_constraint_embeddings(rng)

        result = arbiter.arbitrate(responses, constraint_embeddings)
        is_correct = result["arbiter_index"] == 0

        results.append(
            {
                "scenario_id": f"standard_{i + 1}",
                "type": "standard",
                "arbiter_index": result["arbiter_index"],
                "is_correct": is_correct,
                "used_consensus_penalty": result["used_consensus_penalty"],
                "energies_raw": result["energies_raw"],
                "energies_normalized": result["energies_normalized"],
                "energies_adjusted": result["energies_adjusted"],
            }
        )

    return results


def _run_adversarial_scenarios(
    arbiter: MultiAgentArbiter, rng: np.random.Generator
) -> list[dict]:
    """Run 6 adversarial scenarios where 2 of 3 agents share the wrong answer.

    Adversarial scenario: agents = [correct_adv_N, wrong_adv_N, wrong_adv_N].
    Two agents share the identical wrong response string, forming a majority cluster.
    detect_consensus() should fire on the response cluster (>= 2 identical) and the
    consensus penalty should be applied, boosting the two wrong agents' energies.
    The arbiter then picks the correct agent (index 0) as the minimum-penalty-adjusted energy.

    Args:
        arbiter: MultiAgentArbiter instance with z-score normalization.
        rng: Seeded RNG for generating constraint embeddings.

    Returns:
        List of 6 scenario result dicts.

    Spec: REQ-VERIFY-144, SCENARIO-VERIFY-172
    """
    results = []
    for i in range(N_ADVERSARIAL):
        # Two agents share wrong_adv_N — this triggers the response-cluster consensus check.
        responses = [f"correct_adv_{i}", f"wrong_adv_{i}", f"wrong_adv_{i}"]
        constraint_embeddings = _make_constraint_embeddings(rng)

        result = arbiter.arbitrate(responses, constraint_embeddings)
        is_correct = result["arbiter_index"] == 0

        results.append(
            {
                "scenario_id": f"adversarial_{i + 1}",
                "type": "adversarial",
                "arbiter_index": result["arbiter_index"],
                "is_correct": is_correct,
                "used_consensus_penalty": result["used_consensus_penalty"],
                "energies_raw": result["energies_raw"],
                "energies_normalized": result["energies_normalized"],
                "energies_adjusted": result["energies_adjusted"],
            }
        )

    return results


def map_honest_verdict(accuracy_standard: float) -> str:
    """Map standard accuracy to an honest_verdict string.

    The verdict hierarchy is relative to the Exp 822 baseline (accuracy_standard=0.17):
        - arbiter_calibrated:  >= 0.67 (4/6 — satisfies SCENARIO-VERIFY-172)
        - arbiter_partial:     >= 0.50 (3/6 — meaningful improvement, not yet passing)
        - arbiter_improvement: > 0.17  (beats Exp 822 but below 50%)
        - arbiter_still_wrong: <= 0.17 (no improvement over baseline)

    Args:
        accuracy_standard: Fraction of 6 standard scenarios correctly arbitrated.

    Returns:
        One of the four verdict strings above.

    Spec: REQ-VERIFY-143, REQ-VERIFY-144
    """
    if accuracy_standard >= 0.67:
        return "arbiter_calibrated"
    if accuracy_standard >= 0.50:
        return "arbiter_partial"
    if accuracy_standard > 0.17:
        return "arbiter_improvement"
    return "arbiter_still_wrong"


def main() -> None:
    """Run Exp 835 and write the deliverable artifact."""
    tmpl = ExperimentTemplate(
        EXP_ID,
        TITLE,
        DELIVERABLE,
        requires_gpu=False,
    )
    tmpl.setup()

    _watchdog = ExperimentTimeoutWatchdog(EXP_ID, timeout_minutes=TIMEOUT_MINUTES)
    output_path = Path(_REPO / DELIVERABLE)

    arbiter = MultiAgentArbiter(
        n_spins=N_SPINS,
        embedding_dim=EMB_DIM,
        consensus_threshold=0.01,
        consensus_penalty=0.1,
    )

    rng = np.random.default_rng(SEED)

    standard_results = _run_standard_scenarios(arbiter, rng)
    adversarial_results = _run_adversarial_scenarios(arbiter, rng)
    all_results = standard_results + adversarial_results

    accuracy_standard = sum(r["is_correct"] for r in standard_results) / len(standard_results)
    accuracy_adversarial = sum(r["is_correct"] for r in adversarial_results) / len(adversarial_results)
    accuracy_overall = sum(r["is_correct"] for r in all_results) / len(all_results)
    consensus_penalty_triggered_n = sum(r["used_consensus_penalty"] for r in all_results)

    honest_verdict = map_honest_verdict(accuracy_standard)

    artifact = tmpl.build_result(
        {
            "accuracy_standard": accuracy_standard,
            "accuracy_adversarial": accuracy_adversarial,
            "accuracy_overall": accuracy_overall,
            "consensus_penalty_triggered_n": consensus_penalty_triggered_n,
            "honest_verdict": honest_verdict,
            "scenario_results": all_results,
            "n_standard": len(standard_results),
            "n_adversarial": len(adversarial_results),
            "n_total": len(all_results),
        },
        status="success",
        honest_verdict=honest_verdict,
    )

    with open(output_path, "w") as fh:
        json.dump(artifact, fh, indent=2)

    _watchdog.stop()
    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
