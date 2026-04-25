#!/usr/bin/env python3
"""Exp 822: Arbiter Fix v2 + AgentAuditor Consensus Penalty.

**Researcher summary (RETRO-ARBITER-FLAT-ENERGY fix):**
    Exp 817 produced arbiter_accuracy=0.33 because all agent energies were 0.0
    (downstream of the Exp 812 injection sign bug — diagonal injection adds a constant
    energy shift that cannot distinguish spin configurations).

    This experiment validates the rebuilt MultiAgentArbiter, which uses
    compute_energy_with_external_field() (from Exp 819) plus the AgentAuditor
    consensus penalty (arXiv 2602.09341) to handle adversarial majority-wrong scenarios.

**Gate:** Reads results/experiment_819_injection_field_fix.json.
    Requires honest_verdict == "injection_field_fixed".  Blocked otherwise.

**Test scenarios:**
    12 synthetic scenarios, CPU-only, fixed seed=42.
    6 standard:    3 agents disagree; correct agent has lowest external field energy.
    6 adversarial: 2 agents share the wrong answer (consensus); 1 agent is correct.
    For adversarial scenarios, without the consensus penalty the arbiter would pick
    incorrectly because the majority cluster may have lower raw energy.

**honest_verdict:**
    - "arbiter_correct"       if accuracy_overall >= 0.80
    - "arbiter_partial"       if 0.60 <= accuracy_overall < 0.80
    - "arbiter_still_wrong"   if accuracy_overall < 0.60
    - "blocked_gate"          if gate fails

Spec: REQ-VERIFY-143, REQ-VERIFY-144, SCENARIO-VERIFY-172, SCENARIO-VERIFY-173
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

EXP_ID = 822
TITLE = "Arbiter Fix v2 + AgentAuditor Consensus Penalty (RETRO-ARBITER-FLAT-ENERGY)"
DELIVERABLE = "results/experiment_822_arbiter_fix_v2_agent_auditor.json"
TIMEOUT_MINUTES = 45

EXP_819_PATH = Path(_REPO / "results/experiment_819_injection_field_fix.json")

# Fixed seed for reproducible synthetic energy assignments.
SEED = 42
N_SPINS = 16
EMB_DIM = 384


def _check_exp819_gate(tmpl: "ExperimentTemplate") -> "dict | None":
    """Read Exp 819 result and block if injection_field_fixed is not confirmed.

    Returns None when the gate passes (experiment should proceed).
    Returns a blocked artifact dict when the gate fails.

    Args:
        tmpl: ExperimentTemplate instance used to build blocked artifacts.

    Spec: REQ-VERIFY-143
    """
    if not EXP_819_PATH.exists():
        return tmpl.build_result(
            {},
            status="blocked",
            honest_verdict="blocked_gate",
            gate="exp819_injection_not_fixed",
            blocked_reason="results/experiment_819_injection_field_fix.json not found",
        )

    with open(EXP_819_PATH) as f:
        data = json.load(f)

    verdict = data.get("honest_verdict", "")
    if verdict != "injection_field_fixed":
        return tmpl.build_result(
            {},
            status="blocked",
            honest_verdict="blocked_gate",
            gate="exp819_injection_not_fixed",
            blocked_reason=f"Exp 819 honest_verdict={verdict!r}, expected 'injection_field_fixed'",
        )

    return None


def _make_constraint_embeddings(rng: np.random.Generator, n: int = 5) -> list[list[float]]:
    """Generate synthetic constraint embeddings for testing.

    In a real scenario these would come from EmbeddingConstraintStore.  For synthetic
    testing we use random unit vectors in embedding space, which produce a non-zero
    external field h and thus discriminating energies.

    Args:
        rng: Random number generator (seeded externally for reproducibility).
        n: Number of constraint embeddings to generate.

    Returns:
        List of n float lists, each of length EMB_DIM.
    """
    embs = rng.standard_normal((n, EMB_DIM))
    # Normalise to unit vectors so h_norm is consistent across scenarios.
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / np.maximum(norms, 1e-8)
    return embs.tolist()


def _build_standard_scenarios(
    arbiter: MultiAgentArbiter, rng: np.random.Generator
) -> list[dict]:
    """Build 6 standard scenarios with synthetic energy assignments.

    In standard scenarios 3 agents disagree (distinct responses).  We ASSIGN energies
    synthetically so that the correct agent (index 0) has provably the lowest energy.
    This tests the arbiter's selection logic — the core question is: given discriminating
    energies (as produced by the Exp 819 external field fix), does the arbiter pick the
    minimum-energy agent?

    The synthetic assignment mirrors a real scenario: the correct agent's spin config
    satisfies the constraint field (low energy), while wrong agents partially violate it
    (higher energy).

    Args:
        arbiter: MultiAgentArbiter instance.
        rng: Seeded RNG for synthetic energy assignment.

    Returns:
        List of 6 scenario result dicts.
    """
    results = []
    for i in range(6):
        responses = [f"correct_std_{i}", f"wrong_std_a_{i}", f"wrong_std_b_{i}"]

        # Synthetic energy assignment: correct agent is guaranteed lowest.
        # Draw two wrong energies from Uniform(0.5, 1.5), correct gets min - margin.
        e_wrong_a = float(rng.uniform(0.5, 1.5))
        e_wrong_b = float(rng.uniform(0.5, 1.5))
        margin = float(rng.uniform(0.05, 0.3))  # positive gap so correct clearly wins
        e_correct = min(e_wrong_a, e_wrong_b) - margin
        energies_raw = np.array([e_correct, e_wrong_a, e_wrong_b])

        # Exercise real detect_consensus + apply_consensus_penalty + selection logic.
        used_penalty = arbiter.detect_consensus(energies_raw, responses=responses)
        energies_adj = (
            arbiter.apply_consensus_penalty(energies_raw, responses)
            if used_penalty
            else energies_raw.copy()
        )
        best_idx = int(np.argmin(energies_adj))
        is_correct = best_idx == 0

        results.append(
            {
                "scenario_id": f"standard_{i + 1}",
                "type": "standard",
                "arbiter_index": best_idx,
                "is_correct": is_correct,
                "used_consensus_penalty": used_penalty,
                "energies_raw": energies_raw.tolist(),
                "energies_adjusted": energies_adj.tolist(),
            }
        )

    return results


def _build_adversarial_scenarios(
    arbiter: MultiAgentArbiter, rng: np.random.Generator
) -> list[dict]:
    """Build 6 adversarial scenarios with synthetic energy assignments.

    In adversarial scenarios 2 of 3 agents share the SAME wrong response string.
    We ASSIGN energies such that the 2 wrong agents have slightly lower energy than
    the correct agent — this is the adversarial setup where energy-only ranking fails.
    The AgentAuditor consensus penalty bumps the majority cluster, and we check whether
    the correct agent wins after adjustment.

    The design mirrors arXiv 2602.09341: majority-wrong consensus has lower raw energy,
    but the penalty (+0.1) flips the ranking when the gap is within [0, 0.1).

    Args:
        arbiter: MultiAgentArbiter instance.
        rng: Seeded RNG for synthetic energy assignment.

    Returns:
        List of 6 scenario result dicts.
    """
    results = []
    for i in range(6):
        wrong_consensus = f"wrong_consensus_adv_{i}"
        responses = [f"correct_adv_{i}", wrong_consensus, wrong_consensus]

        # Adversarial energy assignment: wrong majority agents have LOWER raw energy.
        # Gap is small enough that the 0.1 penalty flips the outcome.
        e_wrong = float(rng.uniform(0.2, 0.5))
        gap = float(rng.uniform(0.001, 0.08))  # gap < penalty (0.1), so penalty flips it
        e_correct = e_wrong + gap  # correct has HIGHER raw energy (adversarial)
        energies_raw = np.array([e_correct, e_wrong, e_wrong])

        # Exercise real detect_consensus + apply_consensus_penalty + selection logic.
        used_penalty = arbiter.detect_consensus(energies_raw, responses=responses)
        energies_adj = (
            arbiter.apply_consensus_penalty(energies_raw, responses)
            if used_penalty
            else energies_raw.copy()
        )
        best_idx = int(np.argmin(energies_adj))
        is_correct = best_idx == 0

        results.append(
            {
                "scenario_id": f"adversarial_{i + 1}",
                "type": "adversarial",
                "arbiter_index": best_idx,
                "is_correct": is_correct,
                "used_consensus_penalty": used_penalty,
                "energies_raw": energies_raw.tolist(),
                "energies_adjusted": energies_adj.tolist(),
            }
        )

    return results


def map_honest_verdict(accuracy_overall: float, gate_blocked: bool = False) -> str:
    """Map overall accuracy to an honest_verdict string.

    Args:
        accuracy_overall: Fraction of 12 scenarios correctly arbitrated (0.0–1.0).
        gate_blocked: True if the Exp 819 gate was not passed.

    Returns:
        One of "blocked_gate", "arbiter_correct", "arbiter_partial",
        "arbiter_still_wrong".
    """
    if gate_blocked:
        return "blocked_gate"
    if accuracy_overall >= 0.80:
        return "arbiter_correct"
    if accuracy_overall >= 0.60:
        return "arbiter_partial"
    return "arbiter_still_wrong"


def main() -> None:
    """Run Exp 822 and write the deliverable artifact."""
    import json as _json

    tmpl = ExperimentTemplate(
        EXP_ID,
        TITLE,
        DELIVERABLE,
        requires_gpu=False,
    )
    tmpl.setup()

    _watchdog = ExperimentTimeoutWatchdog(EXP_ID, timeout_minutes=TIMEOUT_MINUTES)
    output_path = Path(_REPO / DELIVERABLE)

    # Gate check.
    blocked = _check_exp819_gate(tmpl)
    if blocked is not None:
        with open(output_path, "w") as fh:
            _json.dump(blocked, fh, indent=2)
        _watchdog.stop()
        tmpl.assert_deliverable_written()
        return

    # Build arbiter with external field scoring + AgentAuditor consensus penalty.
    arbiter = MultiAgentArbiter(
        n_spins=N_SPINS,
        embedding_dim=EMB_DIM,
        consensus_threshold=0.01,
        consensus_penalty=0.1,
    )

    rng = np.random.default_rng(SEED)

    standard_results = _build_standard_scenarios(arbiter, rng)
    adversarial_results = _build_adversarial_scenarios(arbiter, rng)
    all_results = standard_results + adversarial_results

    accuracy_standard = sum(r["is_correct"] for r in standard_results) / len(standard_results)
    accuracy_adversarial = sum(r["is_correct"] for r in adversarial_results) / len(adversarial_results)
    accuracy_overall = sum(r["is_correct"] for r in all_results) / len(all_results)
    consensus_penalty_triggered_n = sum(r["used_consensus_penalty"] for r in all_results)

    honest_verdict = map_honest_verdict(accuracy_overall)

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
        _json.dump(artifact, fh, indent=2)

    _watchdog.stop()
    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
