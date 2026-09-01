"""Spec: REQ-VERIFY-6802, SCENARIO-VERIFY-6802-A, SCENARIO-VERIFY-6802-B, SCENARIO-VERIFY-6802-C

QA-layer audit 2026-08-31: three guards were narrower than the concepts they named.

Each test below IS the audit's own MISSED INPUT — the input that falls inside a guard's stated
concept and got through anyway. Per the QA-Layer Authenticity Discipline, a SILENT_NON_FIRING
finding is actioned by WIDENING the pattern and adding that exact input as a regression test, not
by rewriting the check's logic.

Triage note: the audit reported SIX findings for these files. Two were REFUTED by running the
reviewer's own inputs against the live guards -- `_legitimate_pair` does not exempt the cited
pair, and `_inference_substrate_text` does read through a principle wrapper. A hostile reviewer's
hit rate here was 4/6, which is why the findings are tested rather than believed.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import adversarial_verify as av  # noqa: E402


def test_roc_auc_spelling_is_not_invisible() -> None:
    """`roc_auc` does NOT contain "auroc" as a substring.

    The most common sklearn spelling of the project's headline metric was invisible to the
    fabrication gate: a perfect score on 400 samples drew nothing at all.
    """
    flags: list = []
    av.check_implausible_perfect({"roc_auc": 1.0, "n_samples": 400}, flags)
    assert any(f.kind == "IMPLAUSIBLE_PERFECT" for f in flags)


def test_the_original_auroc_spelling_still_fires() -> None:
    """Widening must not trade one spelling for another."""
    flags: list = []
    av.check_implausible_perfect({"auroc": 1.0, "n_samples": 400}, flags)
    assert any(f.kind == "IMPLAUSIBLE_PERFECT" for f in flags)


def test_a_falling_pass_rate_is_a_sign_anomaly() -> None:
    """The concept is "metrics that should go UP" and the list omitted rates entirely, so a pass
    rate falling 0.80 -> 0.60 -- a real regression -- drew nothing."""
    flags: list = []
    av.check_sign_anomaly({"initial_pass_rate": 0.80, "final_pass_rate": 0.60}, flags)
    assert any(f.kind == "SIGN_ANOMALY" for f in flags)


def test_a_rising_pass_rate_is_not_an_anomaly() -> None:
    """The direction must be right, not merely present."""
    flags: list = []
    av.check_sign_anomaly({"initial_pass_rate": 0.60, "final_pass_rate": 0.80}, flags)
    assert not any(f.kind == "SIGN_ANOMALY" for f in flags)


def test_a_principle_wrapped_delta_decides_instead_of_raising() -> None:
    """`int({"principle":..., "value": 0})` RAISES TypeError.

    That REMOVES the guard rather than failing it closed, which is the worst of the three
    outcomes. The July origin incident was this same class in a sibling function; the fix had
    been applied there and not here.
    """
    d = {
        "solve_claimed": False,
        "offline_reproduced": False,
        "level_credit_delta": {"principle": "measured", "value": 0},
        "field_provenance": {"level_credit_delta": {}},
    }
    assert av._is_declared_honest_zero_delta("level_credit_delta", d) is True


def test_the_bare_form_still_works() -> None:
    d = {
        "solve_claimed": False,
        "offline_reproduced": False,
        "level_credit_delta": 0,
        "field_provenance": {"level_credit_delta": {}},
    }
    assert av._is_declared_honest_zero_delta("level_credit_delta", d) is True


def test_unwrap_leaves_a_bare_value_alone() -> None:
    assert av._unwrapped_scalar(3) == 3
    assert av._unwrapped_scalar({"principle": "p", "value": 7}) == 7
    assert av._unwrapped_scalar({"no_value_key": 1}) == {"no_value_key": 1}
