"""Tests for exp3437 energy-vote vs self-consistency real-harness premise v3.

Spec: REQ-KONA-3437, SCENARIO-KONA-3437, SCENARIO-KONA-3437-DEGENERATE.

These tests pin the v3 additions on top of the (already-tested) v2 surface:
  * the per-sample answer extraction works on EACH sampled generation, not just
    the greedy one — the exp3426 bug that returned 0.0 for every k-sample
    condition;
  * the NON-DEGENERATE-SC gate (step 0e) that makes the 0.0-vs-0.0 tie
    impossible to ship;
  * the v3 verdict mapping including the degenerate-harness blocked terminal.
"""

from __future__ import annotations

import pytest

from carnot.phase3.energy_premise_v3 import (
    DEGENERATE_VERDICT,
    ScDegeneracyGate,
    derive_premise_v3_verdict,
    evaluate_sc_non_degenerate,
    extract_candidate_answers,
)

pytestmark = pytest.mark.kona


def test_module_imports():
    # SCENARIO-KONA-3437: the v3 surface is importable as one coherent module.
    import carnot.phase3.energy_premise_v3 as mod

    # Re-exported v2 primitives must be reachable through the v3 module so the
    # experiment script imports a single surface.
    assert hasattr(mod, "energy_weighted_vote")
    assert hasattr(mod, "self_certainty_select")
    assert hasattr(mod, "load_gsm8k_subset")


def test_extract_candidate_answers_runs_per_sample():
    # SCENARIO-KONA-3437: extraction must work on EACH non-empty sampled
    # generation — the exact thing exp3426 never exercised (every sample empty).
    texts = [
        "The cost is 3 apples plus 2 more.\n#### 5",
        "Step 1: 4 * 6 = 24. So the answer is 24.",
        "",  # an empty/failed generation extracts to None, scored as a non-vote.
    ]
    answers = extract_candidate_answers(texts)
    assert answers == [5, 24, None]


def test_non_degenerate_gate_passes_when_sc_beats_greedy():
    # SCENARIO-KONA-3437: a healthy harness has SC >= greedy and SC above floor.
    ar = [True, True, False, False, True]  # greedy 0.6
    sc = [True, True, True, False, True]  # self-consistency 0.8
    gate = evaluate_sc_non_degenerate(ar, sc, min_abs=0.30)
    assert isinstance(gate, ScDegeneracyGate)
    assert gate.passed is True
    assert gate.self_consistency_accuracy == pytest.approx(0.8)
    assert gate.ar_greedy_accuracy == pytest.approx(0.6)
    assert gate.reason == ""


def test_non_degenerate_gate_fails_on_zero_self_consistency():
    # SCENARIO-KONA-3437-DEGENERATE: the exact exp3426 failure — SC=0.0 next to a
    # nonzero greedy is the broken-harness signature.
    ar = [True, True, True, False]  # greedy 0.75
    sc = [False, False, False, False]  # self-consistency 0.0 (all-empty samples)
    gate = evaluate_sc_non_degenerate(ar, sc, min_abs=0.30)
    assert gate.passed is False
    assert gate.self_consistency_accuracy == pytest.approx(0.0)
    assert "broken" in gate.reason


def test_non_degenerate_gate_fails_when_sc_below_greedy():
    # SCENARIO-KONA-3437-DEGENERATE: SC above the floor but below greedy is still
    # degenerate — majority vote should never lose to one greedy sample on GSM8K.
    ar = [True, True, True, True, True, True, True, True, False, False]  # 0.8
    sc = [True, True, True, True, False, False, False, False, False, False]  # 0.4
    gate = evaluate_sc_non_degenerate(ar, sc, min_abs=0.30)
    assert gate.passed is False
    assert gate.self_consistency_accuracy == pytest.approx(0.4)
    assert gate.ar_greedy_accuracy == pytest.approx(0.8)
    assert "< greedy" in gate.reason


def test_non_degenerate_gate_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="equal-length"):
        evaluate_sc_non_degenerate([True, False], [True])


def test_non_degenerate_gate_rejects_empty_batch():
    with pytest.raises(ValueError, match="non-empty"):
        evaluate_sc_non_degenerate([], [])


def test_v3_verdict_blocks_on_degenerate_gate():
    # SCENARIO-KONA-3437-DEGENERATE: a failed gate forces the blocked terminal
    # regardless of the (meaningless) downstream accuracies.
    failed_gate = ScDegeneracyGate(
        passed=False,
        self_consistency_accuracy=0.0,
        ar_greedy_accuracy=0.75,
        min_abs_threshold=0.30,
        reason="broken",
    )
    verdict = derive_premise_v3_verdict(
        failed_gate,
        self_consistency_accuracy=0.0,
        energy_weighted_vote_accuracy=0.0,
        p_value=1.0,
        ci=(0.0, 0.0),
        direction=0.0,
    )
    assert verdict.verdict == DEGENERATE_VERDICT
    assert verdict.g1_energy_non_inferior is False
    assert verdict.g2_energy_adds_value is False


def test_v3_verdict_delegates_to_v2_when_gate_passes_energy_wins():
    # SCENARIO-KONA-3437: a passing gate delegates to the v2 gates; a significant
    # energy win maps to the validated terminal (G2).
    passed_gate = ScDegeneracyGate(
        passed=True,
        self_consistency_accuracy=0.80,
        ar_greedy_accuracy=0.75,
        min_abs_threshold=0.30,
        reason="",
    )
    verdict = derive_premise_v3_verdict(
        passed_gate,
        self_consistency_accuracy=0.80,
        energy_weighted_vote_accuracy=0.86,
        p_value=0.01,
        ci=(0.02, 0.10),
        direction=1.0,
    )
    assert verdict.verdict == "complete: energy_beats_self_consistency_premise_validated"
    assert verdict.g2_energy_adds_value is True


def test_v3_verdict_delegates_to_v2_when_gate_passes_energy_below():
    # SCENARIO-KONA-3437: a passing gate with energy significantly below SC maps
    # to the premise-unsupported terminal (G1 fails).
    passed_gate = ScDegeneracyGate(
        passed=True,
        self_consistency_accuracy=0.85,
        ar_greedy_accuracy=0.75,
        min_abs_threshold=0.30,
        reason="",
    )
    verdict = derive_premise_v3_verdict(
        passed_gate,
        self_consistency_accuracy=0.85,
        energy_weighted_vote_accuracy=0.70,
        p_value=0.001,
        ci=(-0.20, -0.05),
        direction=-1.0,
    )
    assert verdict.verdict == (
        "complete: energy_below_self_consistency_premise_unsupported_retire_superiority_framing"
    )
    assert verdict.g1_energy_non_inferior is False
