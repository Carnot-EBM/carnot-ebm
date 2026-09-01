"""Spec: REQ-VERIFY-6802, SCENARIO-VERIFY-6802-A, SCENARIO-VERIFY-6802-B, SCENARIO-VERIFY-6802-C, SCENARIO-VERIFY-6802-D

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


# --- SCENARIO-VERIFY-6802-D: a training claim with no tool named ---------------------------
# The fourth finding from the same audit, held back from the first commit because widening
# the compute-bound marker changes which artifacts face the duration floor at all. Measured
# before landing: 16 artifacts newly flagged across the 5,915-artifact corpus, every one a
# retrain claiming completion in 1.7 to 31 seconds.


def test_a_retrain_verdict_with_no_tool_named_is_compute_bound() -> None:
    """The origin artifact. It named no model, framework, or runner, so the marker missed it."""
    artifact = {
        "experiment": 746,
        "title": "DualGPU EORM+JEPA Retrain - production rollout and speedup validation",
        "honest_verdict": "dualgpu_retrain_validated",
        "duration_s": 1.73,
    }
    assert av._has_compute_bound_marker(artifact) is True
    flags: list[av.Flag] = []
    av.check_duration_vs_claim(artifact, flags)
    assert [f.kind for f in flags] == ["DURATION_TOO_SHORT"]


def test_a_retrospective_that_lists_a_retrain_is_not_compute_bound() -> None:
    """Why the scan is restricted to self-describing fields.

    A whole-blob scan made retrospectives and closeout artifacts compute-bound because they
    name the experiments they summarise. Measured: 13 such false positives.
    """
    artifact = {
        "experiment": "1215_milestone_retro_94",
        "honest_verdict": "milestone_94_clean_sweep_13_of_13",
        "tasks_completed": ["exp664 dualgpu retrain", "exp746 eorm retrain"],
        "duration_s": 0.1,
    }
    assert av._has_compute_bound_marker(artifact) is False


def test_an_honest_blocked_retrain_is_not_compute_bound() -> None:
    """A blocked verdict reports that nothing ran. Flagging it would punish the honesty."""
    artifact = {
        "honest_verdict": "complete: blocked_model_not_cached_retrain",
        "duration_s": 0.05,
    }
    assert av._has_compute_bound_marker(artifact) is False


def test_a_principle_wrapped_verdict_still_counts() -> None:
    """Any field may be principle-annotated, so read through the wrapper."""
    artifact = {
        "honest_verdict": {"principle": "self-declared terminal state", "value": "retrain_done"},
        "duration_s": 2.0,
    }
    assert av._has_compute_bound_marker(artifact) is True


def test_infrastructure_work_on_gpus_is_not_a_training_claim() -> None:
    """Why the token list holds one word.

    `dualgpu` and `gpu` were measured and rejected. A harness patch or an enforcement audit
    names the hardware without ever running a job; adding those words flagged 13 of them.
    """
    for verdict in ("all_patched", "harness_audit_complete", "zombie_detected"):
        artifact = {"title": f"dual GPU harness work", "honest_verdict": verdict, "duration_s": 2.0}
        assert av._has_compute_bound_marker(artifact) is False, verdict
