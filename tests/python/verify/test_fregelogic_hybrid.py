"""Tests for Exp 2395 FregeLogic neural-prefilter/Z3 tiebreaker.

Spec: REQ-TIER0-010, SCENARIO-TIER0-010
"""

from __future__ import annotations

from carnot.verify.fregelogic_hybrid import FregeLogicHybrid


def test_fregelogic_uses_neural_consensus_without_z3() -> None:
    """REQ-TIER0-010-2: agreeing pre-filters skip the symbolic tiebreaker."""
    entry = {
        "semantic_energy_score": 0.9,
        "laab_score": 0.8,
        "prompt": "Verify claim: 2 + 5 = 7. Return 1 if true, 0 if false.",
        "response_text": "1",
    }

    result = FregeLogicHybrid(semantic_threshold=0.5, laab_threshold=0.5).verify(entry)

    assert result["fregelogic_verdict"] == "high_risk"
    assert result["tiebreaker_invoked"] is False
    assert result["z3_verdict"] is None


def test_fregelogic_invokes_z3_on_neural_disagreement() -> None:
    """SCENARIO-TIER0-010: disagreement routes to Z3 answer validation."""
    entry = {
        "semantic_energy_score": 0.1,
        "laab_score": 0.9,
        "prompt": (
            "Telemetry obedience check. Do not solve the task. Return exactly "
            "this integer and no other text: 4"
        ),
        "response_text": "2",
    }

    result = FregeLogicHybrid(semantic_threshold=0.5, laab_threshold=0.5).verify(entry)

    assert result["fregelogic_verdict"] == "high_risk"
    assert result["tiebreaker_invoked"] is True
    assert result["z3_verdict"] == "high_risk"
    assert "(assert (= response_answer expected_answer))" in result["z3_smtlib"]


def test_fregelogic_unknown_z3_tiebreaker_stays_nontrivial() -> None:
    """REQ-TIER0-010-3: unsupported responses produce an honest unknown verdict."""
    entry = {
        "semantic_energy_score": 0.9,
        "laab_score": 0.1,
        "prompt": "Verify claim: 4 * 3 = 11. Return 1 if true, 0 if false.",
        "response_text": "<think>unfinished reasoning trace",
    }

    result = FregeLogicHybrid(semantic_threshold=0.5, laab_threshold=0.5).verify(entry)

    assert result["tiebreaker_invoked"] is True
    assert result["z3_verdict"] == "unknown"
    assert result["fregelogic_verdict"] == "uncertain"
