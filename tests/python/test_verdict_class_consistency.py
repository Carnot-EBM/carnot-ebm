"""Tests for the declared verdict-class structural cross-check.

REQ: REQ-CONDUCTOR-VERDICT-1 (openspec/capabilities/research-harnesses/spec.md).
SCENARIOs: SCENARIO-CONDUCTOR-VERDICT-1 (oracle-circular result may not be
`positive`), SCENARIO-CONDUCTOR-VERDICT-2 (declared circular_positive is
clean), SCENARIO-CONDUCTOR-VERDICT-3 (values outside the closed enum flag).

Origin: verdict semantics lived in four drifting substring token lists,
patched at least six times. The enum is declared by the artifact and
cross-checked against structural fields the linter already reads — never a
fifth token list.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import scripts.adversarial_verify as av  # noqa: E402


def _flags(d: dict) -> list:
    flags: list = []
    av.check_verdict_class_consistency(d, flags)
    return [f for f in flags if f.kind == "VERDICT_CLASS_MISMATCH"]


def test_absent_declaration_draws_no_flag() -> None:
    # Adoption is forward-only; the historical corpus has no verdict_class.
    assert _flags({"honest_verdict": "complete: fine"}) == []


def test_unknown_class_is_critical() -> None:
    # SCENARIO-CONDUCTOR-VERDICT-3: the enum is closed on purpose.
    out = _flags({"verdict_class": "triumphant"})
    assert len(out) == 1
    assert out[0].severity == "critical"
    assert "closed enum" in out[0].detail


def test_positive_with_oracle_verifier_is_critical() -> None:
    # SCENARIO-CONDUCTOR-VERDICT-1: the exp6478 shape — an honest circular
    # result whose class would connote a research win downstream.
    out = _flags({"verdict_class": "positive", "verifier_is_oracle": True})
    assert len(out) == 1
    assert out[0].severity == "critical"
    assert "circular_positive" in out[0].detail


def test_circular_positive_with_oracle_verifier_is_clean() -> None:
    # SCENARIO-CONDUCTOR-VERDICT-2
    assert _flags({"verdict_class": "circular_positive", "verifier_is_oracle": True}) == []


def test_positive_with_failed_gate_is_critical() -> None:
    out = _flags({"verdict_class": "positive", "acceptance_gate_passed": False})
    assert len(out) == 1
    assert out[0].severity == "critical"
    assert "acceptance_gate" in out[0].detail


def test_positive_with_passed_gate_and_distinct_verifier_is_clean() -> None:
    assert (
        _flags(
            {
                "verdict_class": "positive",
                "verifier_is_oracle": False,
                "acceptance_gate_passed": True,
            }
        )
        == []
    )


def test_non_positive_classes_are_structurally_clean() -> None:
    for vc in ("null", "blocked", "disqualified", "partial"):
        assert _flags({"verdict_class": vc, "verifier_is_oracle": True}) == []
