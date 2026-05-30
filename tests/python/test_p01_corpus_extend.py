"""Tests for the v2 resume-and-extend logic of the P0.1 generation corpus (exp3459).

Spec: REQ-KONA-3459, SCENARIO-KONA-3459, SCENARIO-KONA-3459-RESUME-MONOTONE.

These tests pin the GPU-free scientific decisions the extend run makes: the v2
terminal-verdict bands (complete / headline-eligible / extended-partial), the two
v2 acceptance gates (G1 corpus-not-regressed, G2 headline-eligible), and the
``added_this_run`` accounting that proves the resume did new work. No live model
is loaded — every assertion runs deterministically so a reviewer/CI can re-derive
the verdict the live exp3459 run reports without a 26B GGUF.
"""

from __future__ import annotations

from carnot.phase3.p01_corpus_extend import (
    DEFAULT_N_TARGET,
    EXP3448_CORPUS_FLOOR,
    HEADLINE_ELIGIBLE_FLOOR,
    added_this_run,
    derive_extend_verdict,
    extend_acceptance_gates,
)


# ---------------------------------------------------------------------------
# derive_extend_verdict (REQ-KONA-3459: three-band terminal verdict)
# ---------------------------------------------------------------------------


def test_verdict_complete_at_or_above_target() -> None:
    """REQ-KONA-3459: reaching the target yields the `complete` verdict."""
    v = derive_extend_verdict(120)
    assert v == "complete: p01_generation_corpus_complete_n=120"
    # Above the target also counts as complete.
    assert derive_extend_verdict(125).startswith(
        "complete: p01_generation_corpus_complete_n=125"
    )


def test_verdict_headline_eligible_band() -> None:
    """REQ-KONA-3459: 80 <= n < 120 is the HEADLINE-eligible band."""
    v = derive_extend_verdict(96)
    assert v == "complete: p01_generation_corpus_headline_eligible_n=96"
    # Exactly the floor is headline-eligible.
    assert derive_extend_verdict(HEADLINE_ELIGIBLE_FLOOR) == (
        "complete: p01_generation_corpus_headline_eligible_n=80"
    )
    # Just below the target is still headline-eligible, not complete.
    assert "headline_eligible" in derive_extend_verdict(119)


def test_verdict_extended_partial_below_headline_floor() -> None:
    """REQ-KONA-3459: below 80 is an extended-partial that resumes next milestone."""
    v = derive_extend_verdict(63)
    assert v == (
        "complete: p01_generation_corpus_extended_partial_n=63_resume_next_milestone"
    )
    # The exp3448 corpus floor (47) is still in the extended-partial band.
    assert "extended_partial" in derive_extend_verdict(EXP3448_CORPUS_FLOOR)
    # A degenerate fresh-build fallback (n=0) is also terminal, never crashes.
    assert derive_extend_verdict(0).startswith(
        "complete: p01_generation_corpus_extended_partial_n=0"
    )


def test_verdict_is_always_complete_prefixed() -> None:
    """REQ-KONA-3459: every band is `complete:`-prefixed (Verdict Terminal-Prefix)."""
    for n in (0, 47, 80, 119, 120, 200):
        assert derive_extend_verdict(n).startswith("complete:")


def test_verdict_honours_custom_target() -> None:
    """REQ-KONA-3459: the target is a parameter, so a smaller test corpus works."""
    assert derive_extend_verdict(10, n_target=10).startswith(
        "complete: p01_generation_corpus_complete_n=10"
    )
    assert DEFAULT_N_TARGET == 120


# ---------------------------------------------------------------------------
# extend_acceptance_gates (REQ-KONA-3459: G1 corpus-not-regressed, G2 headline)
# ---------------------------------------------------------------------------


def test_g1_requires_both_count_and_logprobs() -> None:
    """SCENARIO-KONA-3459-RESUME-MONOTONE: G1 needs the floor AND the logprobs."""
    # Healthy: above floor with logprobs captured.
    gates = extend_acceptance_gates(60, True)
    assert gates["g1_corpus_not_regressed"] is True
    # Regressed below the exp3448 floor -> G1 fails even with logprobs.
    assert extend_acceptance_gates(30, True)["g1_corpus_not_regressed"] is False
    # Logprobs dropped -> G1 fails even with enough rows.
    assert extend_acceptance_gates(90, False)["g1_corpus_not_regressed"] is False


def test_g2_headline_eligible_threshold() -> None:
    """REQ-KONA-3459: G2 is true only at or above the 80-problem headline floor."""
    assert extend_acceptance_gates(79, True)["g2_headline_eligible"] is False
    assert extend_acceptance_gates(80, True)["g2_headline_eligible"] is True
    assert extend_acceptance_gates(120, True)["g2_headline_eligible"] is True


def test_gates_return_both_named_booleans() -> None:
    """REQ-KONA-3459: the gate dict carries exactly the two named gates."""
    gates = extend_acceptance_gates(100, True)
    assert set(gates) == {"g1_corpus_not_regressed", "g2_headline_eligible"}
    assert all(isinstance(v, bool) for v in gates.values())


def test_gates_honour_custom_floors() -> None:
    """REQ-KONA-3459: floors are parameters so a small test corpus can drive them."""
    gates = extend_acceptance_gates(
        5, True, corpus_floor=3, headline_floor=4
    )
    assert gates == {"g1_corpus_not_regressed": True, "g2_headline_eligible": True}


# ---------------------------------------------------------------------------
# added_this_run (REQ-KONA-3459: proves the resume did new work)
# ---------------------------------------------------------------------------


def test_added_this_run_reports_new_problems() -> None:
    """REQ-KONA-3459: added = total - prior when the run generated new problems."""
    assert added_this_run(96, 47) == 49


def test_added_this_run_zero_when_nothing_new() -> None:
    """REQ-KONA-3459: a resume that found everything done adds zero, never negative."""
    assert added_this_run(47, 47) == 0
    # A re-count anomaly (total below prior) clamps to zero, not a negative count.
    assert added_this_run(45, 47) == 0
