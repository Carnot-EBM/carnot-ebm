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
