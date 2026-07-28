"""Pin the nested-wall-clock carve-out in adversarial_verify.check_tautology (2026-07-28).

Spec: SCENARIO-NWCT-1 .. SCENARIO-NWCT-6 (same local-identifier convention as the sibling
carve-out pin tests/python/test_adversarial_verify_baseline_identity_tautology.py, which
declares SCENARIO-BIDT-1..8).

REQ: the TAUTOLOGY check must NOT emit CRITICAL on `duration_s` vs `measurement_wall_s`.
`duration_s` is the analyser PROCESS's total wall time; `measurement_wall_s` is the sum of
the per-row measurement spans INSIDE that same process. One physically CONTAINS the other,
so the gap between them is only the analyser's own bookkeeping overhead -- sub-millisecond
on a ~30s numpy run -- and the two therefore agree to 5+ significant figures BY
CONSTRUCTION. That is structural nesting, not the "two DISTINCT measured metrics agree to
>5 sig figs" fabrication signal TAUTOLOGY exists to catch.

ORIGIN INCIDENT (the exact case reproduced below). On 2026-07-28, rebuilding
experiment_6011 / 6012 / 6013 -- three honest artifacts that deliberately report BOTH
clocks, their own source saying "THE MEASUREMENT CLOCK IS NOT THE ANALYSER CLOCK ...
reporting one as the other is how an artifact ends up claiming a measurement it did not
make" -- made the two clocks land within 0.0002s of each other. TAUTOLOGY fired CRITICAL,
which via the fabrication gate would have stamped `flagged_adversarial` on all three and
excluded them from capstone aggregation. The concrete values that fired:

    exp6013 first rebuild : duration_s = 29.22       measurement_wall_s = 29.22
    exp6012 after 6dp fix : duration_s = 29.184199   measurement_wall_s = 29.184

Note the second row: raising the emitted precision from 3dp to 6dp did NOT fix it and
could not, because the values genuinely agree to that many figures. Re-running until the
timings happened to differ would be dodging the detector rather than satisfying it. So the
fix belongs in the detector, and it is the same shape as the identifier/seed carve-out
(CLAUDE.md, "TAUTOLOGY excludes identifiers/seeds").

SAFETY: the carve-out is anchored to the exact pair {duration_s, measurement_wall_s} by
name. It is deliberately NOT a generic "any two *_s duration fields" rule -- two UNRELATED
durations coinciding to 5 sig figs is still real evidence of a copy-paste bug and must stay
CRITICAL. NWCT-3/4/5 below guard that.
"""

import scripts.adversarial_verify as av


def _tautology_flags(artifact: dict) -> list:
    """Return only the CRITICAL TAUTOLOGY flags check_tautology raises for this artifact.

    CRITICAL is what matters here: it is the severity the fabrication gate turns into
    `flagged_adversarial`, which quarantines the artifact out of capstone aggregation.
    """
    flags: list = []
    av.check_tautology(artifact, flags)
    return [f for f in flags if f.kind == "TAUTOLOGY" and f.severity == "critical"]


def _base() -> dict:
    """Minimal artifact skeleton shaped like exp6012's real one."""
    return {
        "experiment": "experiment_6012_hidden_state_trust_gate_hole",
        "experiment_id": 6012,
        "honest_verdict": "complete_hidden_state_trust_gate_hole_confirmed",
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "random_seed": 11,
    }


# --- NWCT-1/2: the origin incident's exact values must NOT be flagged ----------------


def test_nwct1_exp6013_first_rebuild_values_not_flagged() -> None:
    """SCENARIO-NWCT-1: exp6013's rebuild -- both clocks rounded to the identical 29.22."""
    art = _base()
    art["duration_s"] = 29.22
    art["measurement_wall_s"] = 29.22
    assert _tautology_flags(art) == [], (
        "duration_s vs measurement_wall_s are NESTED spans (process wall time contains "
        "the summed measurement spans); their agreement is structural, not fabrication"
    )


def test_nwct2_exp6012_after_precision_fix_values_not_flagged() -> None:
    """SCENARIO-NWCT-2: exp6012 after 6dp -- 29.184199 vs 29.184, still 5 sig figs."""
    art = _base()
    art["duration_s"] = 29.184199
    art["measurement_wall_s"] = 29.184
    assert _tautology_flags(art) == [], (
        "raising precision cannot separate two genuinely-nested clocks; the detector, "
        "not the artifact, is what had to change"
    )


def test_nwct2b_legitimate_pair_is_symmetric() -> None:
    """SCENARIO-NWCT-3: order must not matter -- the pair may be yielded either way round."""
    assert av._legitimate_pair("duration_s", "measurement_wall_s") is True
    assert av._legitimate_pair("measurement_wall_s", "duration_s") is True


# --- NWCT-3/4/5: the carve-out must NOT become a blanket duration exemption ----------


def test_nwct3_two_unrelated_durations_still_flagged() -> None:
    """SCENARIO-NWCT-4 (GUARD): two unrelated coinciding durations STILL flag critical."""
    art = _base()
    art["train_duration_s"] = 29.184199
    art["eval_duration_s"] = 29.184199
    assert _tautology_flags(art), (
        "two UNRELATED durations agreeing to >5 sig figs must stay CRITICAL -- the "
        "carve-out is anchored to one named nested pair, not to duration fields at large"
    )


def test_nwct4_duration_vs_unrelated_wall_field_still_flagged() -> None:
    """SCENARIO-NWCT-5 (GUARD): duration_s + some OTHER wall field is not the exempted pair."""
    art = _base()
    art["duration_s"] = 29.184199
    art["server_startup_wall_s"] = 29.184199
    assert _tautology_flags(art), (
        "only {duration_s, measurement_wall_s} is structurally nested; any other "
        "partner for duration_s must still be flagged"
    )


def test_nwct5_two_distinct_outcome_metrics_still_flagged() -> None:
    """SCENARIO-NWCT-6 (GUARD): two distinct OUTCOME metrics still flag critical."""
    art = _base()
    art["heldout_accuracy"] = 0.8472913
    art["train_auroc"] = 0.8472913
    assert _tautology_flags(art), (
        "two distinct OUTCOME metrics coinciding to >5 sig figs is exactly what "
        "TAUTOLOGY is for and must remain CRITICAL"
    )
