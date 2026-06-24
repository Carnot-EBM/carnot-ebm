"""Pin the control-vs-treatment HONEST-NULL carve-out in adversarial_verify.check_tautology (2026-06-21).

REQ: the TAUTOLOGY check must NOT quarantine a declared honest-null ablation whose control arm equals
its treatment arm because the treatment legitimately added nothing (X_baseline == X_with_verifier).
The .421 HEADLINE generic_transfer null (exp4556) and the integration gate null (exp4560) were
spuriously flagged ~8x across .420/.421 on honest 0.04==0.04 / 2.0074==2.0074 equalities, excluding
the project's two most important ARC measurements from capstone aggregation. The carve-out is GATED on
the artifact's own honest-null verdict (a fabrication never self-labels a null) AND a control/treatment
qualifier key, so a genuine two-distinct-metric fabrication is still flagged.

adversarial_verify.py carries a documented history of TAUTOLOGY false-positive patches (exp3449,
exp4257, the identifier/seed exclusion); this pins the next recurrence so it cannot silently return.

SCENARIO-CTNULL-1: the real exp4556 (HEADLINE, honest-null verdict + control/treatment equality) yields
                   ZERO critical TAUTOLOGY flags.
SCENARIO-CTNULL-2: the real exp4560 (integration gate, honest-null) yields ZERO critical TAUTOLOGY flags.
SCENARIO-CTNULL-3 (REGRESSION GUARD): two genuinely-distinct metrics coinciding WITHOUT an honest-null
                   verdict STILL produce a critical TAUTOLOGY flag (the carve-out cannot whitelist a
                   fabrication).
SCENARIO-CTNULL-4 (SAFETY GATE): an honest-null verdict but NO control/treatment qualifier key STILL
                   flags — the carve-out requires BOTH conditions.
"""
import json
from pathlib import Path

import scripts.adversarial_verify as av

REPO = Path(__file__).resolve().parents[2]


def _critical_tautology(d: dict) -> list:
    flags: list = []
    av.check_tautology(d, flags)
    return [f for f in flags if f.kind == "TAUTOLOGY" and f.severity == "critical"]


def test_scenario_ctnull_1_headline_generic_transfer_null_not_flagged() -> None:
    """SCENARIO-CTNULL-1: the real headline generic_transfer null clears TAUTOLOGY."""
    d = json.loads((REPO / "results" / "experiment_4556_verifier_router_generic_transfer.json").read_text())
    assert "honest_null" in str(d.get("honest_verdict", "")).lower()
    assert _critical_tautology(d) == []


def test_scenario_ctnull_2_integration_gate_null_not_flagged() -> None:
    """SCENARIO-CTNULL-2: the real integration-gate null clears TAUTOLOGY (a separate DURATION flag,
    if any, is out of scope for check_tautology, which only emits TAUTOLOGY)."""
    d = json.loads((REPO / "results" / "experiment_4560_integration_8game_gate.json").read_text())
    assert _critical_tautology(d) == []


def test_scenario_ctnull_3_distinct_metrics_no_null_verdict_still_flags() -> None:
    """SCENARIO-CTNULL-3: a celebratory two-distinct-metric coincidence is STILL a critical flag."""
    fab = {
        "experiment": "fab", "honest_verdict": "success: headline_auroc_beats_peer",
        "auroc": 0.9131, "kl_divergence": 0.9131,
    }
    assert len(_critical_tautology(fab)) >= 1


def test_scenario_ctnull_4_null_verdict_without_qualifier_still_flags() -> None:
    """SCENARIO-CTNULL-4: honest-null verdict but NO control/treatment qualifier key -> still flags.
    The carve-out requires BOTH the honest-null verdict AND an ablation-arm qualifier, so it cannot
    be abused to whitelist a fabrication merely by stamping an honest-null verdict on it."""
    fab = {
        "experiment": "fab", "honest_verdict": "complete: honest_null",
        "auroc": 0.9131, "kl_divergence": 0.9131,  # distinct metrics, no _baseline/_with_verifier/etc.
    }
    assert len(_critical_tautology(fab)) >= 1


def test_scenario_ctnull_carveout_requires_both_conditions() -> None:
    """The carve-out fires only with BOTH an honest-null verdict AND a qualifier key (the .421 shape)."""
    base = {"metric_baseline": 0.04, "metric_with_verifier": 0.04}
    # honest-null + qualifier -> cleared
    assert _critical_tautology({**base, "honest_verdict": "complete: no_value_added_honest_null"}) == []
    # qualifier but celebratory verdict -> flagged
    assert len(_critical_tautology({**base, "honest_verdict": "success: verifier_wins"})) >= 1


def _warn_tautology(d: dict) -> list:
    flags: list = []
    av.check_tautology(d, flags)
    return [f for f in flags if f.kind == "TAUTOLOGY" and f.severity == "warn"]


# The integration gate writes a verdict (`integration_unchanged_both_levers_null`) that lacks a Path-1
# honest-null TOKEN, so it does not clear via the verdict-qualifier carve-out above. It instead carries the
# Path-2 DESCRIPTOR markers (null_delta_methodology_note + positive_control_passed + a covering zero delta),
# which DOWNGRADE the CRITICAL TAUTOLOGY to WARN (not quarantined). Origin: the .429/.430/.431 A6 integration
# phase recurringly logged FLAGGED on the honest `*_integrated == *_pre_integration` (delta=0 because the A1/A2
# levers nulled). Fixed by emitting the markers the carve-out reads -- NOT by weakening the gate.

def test_scenario_ctnull_5_integration_parity_with_markers_downgraded_not_quarantined() -> None:
    """SCENARIO-CTNULL-5: integrated==pre_integration (delta=0) carrying the descriptor markers is
    DOWNGRADED CRITICAL->WARN -- surfaced for audit but NOT quarantined."""
    d = {
        "experiment": "integration_gate",
        "honest_verdict": "complete: integration_unchanged_both_levers_null",
        "live_first_win_rate_integrated": 0.04,
        "live_first_win_rate_pre_integration": 0.04,
        "live_first_win_rate_delta_vs_pre_integration": 0.0,
        "no_regression_vs_pre_integration": True,
        "positive_control_passed": True,
        "null_delta_methodology_note": (
            "honest no-change: levers nulled; integrated==pre is the expected ablation equality; "
            "parity + no_regression are the passing positive control confirming the measurement is real."
        ),
    }
    assert _critical_tautology(d) == []           # not quarantined
    assert len(_warn_tautology(d)) >= 1           # still surfaced as WARN for audit


def test_scenario_ctnull_6_integration_parity_without_passing_control_still_critical() -> None:
    """SCENARIO-CTNULL-6 (SAFETY GATE): the SAME equality WITHOUT a passing positive control (parity failed
    or a regression) STILL flags CRITICAL -- an UNVALIDATED measurement is not excused. The positive control
    is the anti-fabrication gate: a fabricator cannot dodge by merely claiming 'no change'."""
    d = {
        "experiment": "integration_gate",
        "honest_verdict": "complete: integration_unchanged_both_levers_null",
        "live_first_win_rate_integrated": 0.04,
        "live_first_win_rate_pre_integration": 0.04,
        "live_first_win_rate_delta_vs_pre_integration": 0.0,
        "null_delta_methodology_note": "claims no change but the measurement is not validated",
        "positive_control_passed": False,   # measurement NOT validated -> equality NOT excused
    }
    assert len(_critical_tautology(d)) >= 1
