"""Pin the baseline-identity carve-out in adversarial_verify.check_tautology (2026-06-22).

REQ: the TAUTOLOGY check must NOT quarantine (emit CRITICAL on) a coincidence between two
fields that are NOT independent measurements — a baseline/reference (a known prior value) or a
VERIFIED arithmetic delta (a value equal to the difference of two other present fields sharing
its metric stem). Two names for the same baseline, two related baselines, or a delta collapsing
onto its baseline (when treatment == 2*baseline) are structural arithmetic, not the "two DISTINCT
measured metrics agree to >5 sig figs" fabrication signal. Those pairs downgrade CRITICAL->WARN
(surfaced for audit, not quarantining).

Origin: exp4592 (.424 generation-completeness) is a GENUINE winner_generated 1/25->2/25 positive
(verdict `success:`), yet it was quarantined by 11 TAUTOLOGY flags — almost all the 0.04
baseline/delta arithmetic cascade (baseline=0.04=1/25, treatment=0.08=2/25, delta=0.04=baseline).
The pre-existing control-vs-treatment honest-null carve-out did NOT catch it (exp4592 declares a
`success:` verdict, not a null).

SAFETY (adversarial review 2026-06-22): the carve-out is SUFFIX-ANCHORED (no bare-substring
collisions with `n_referenced_artifacts` / `all_spins_different` / `ops_changelog_modified`) and a
`*_delta` field counts as derived ONLY when its value is VERIFIED to equal the difference of two
other present fields that SHARE its metric stem. A fabricator cannot escape quarantine by NAMING
two distinct copied outcomes `accuracy_delta` / `auroc_delta` — those have no backing arithmetic
and stay CRITICAL (guards BIDT-7/8). Two distinct OUTCOME metrics coinciding always stays CRITICAL
(guards BIDT-4/5/6).
"""
import json
from pathlib import Path

import scripts.adversarial_verify as av

REPO = Path(__file__).resolve().parents[2]


def _flags(d: dict) -> list:
    flags: list = []
    av.check_tautology(d, flags)
    return flags


def _critical(d: dict) -> list:
    return [f for f in _flags(d) if f.kind == "TAUTOLOGY" and f.severity == "critical"]


def _warn(d: dict) -> list:
    return [f for f in _flags(d) if f.kind == "TAUTOLOGY" and f.severity == "warn"]


def test_bidt_1_two_names_for_same_baseline_not_critical() -> None:
    """SCENARIO-BIDT-1: two baseline references coinciding -> WARN, not CRITICAL (reference-only;
    no arithmetic needed — a baseline is a prior reference, not a fresh measurement)."""
    d = {
        "experiment": "t", "honest_verdict": "success: x",
        "generic_transfer_baseline_reference": 0.04,
        "generic_transfer_rate_baseline": 0.04,
    }
    assert _critical(d) == []
    assert len(_warn(d)) >= 1


def test_bidt_2_baseline_vs_verified_delta_not_critical() -> None:
    """SCENARIO-BIDT-2: a baseline coinciding with a VERIFIED delta (delta == treatment-baseline,
    both present and sharing the stem) is arithmetic, not fabrication."""
    d = {
        "experiment": "t", "honest_verdict": "success: x",
        "generic_transfer_rate_baseline": 0.04,
        "generic_transfer_rate_with_wiring": 0.08,
        "transfer_delta": 0.04,  # 0.08 - 0.04, stem 'transfer'
    }
    assert _critical(d) == []


def test_bidt_3_two_verified_deltas_not_critical() -> None:
    """SCENARIO-BIDT-3: two VERIFIED deltas coinciding is arithmetic. Treatments differ so only the
    deltas coincide; each delta is backed by its own stem's treatment-baseline pair."""
    d = {
        "experiment": "t", "honest_verdict": "success: x",
        "m1_baseline": 0.10, "m1_treatment": 0.14, "m1_delta": 0.04,
        "m2_baseline": 0.50, "m2_treatment": 0.54, "m2_delta": 0.04,
    }
    assert _critical(d) == []
    assert len(_warn(d)) >= 1


def test_bidt_4_two_outcome_metrics_still_critical() -> None:
    """SCENARIO-BIDT-4 (GUARD): two distinct OUTCOME metrics coinciding STILL flags critical."""
    fab = {
        "experiment": "fab", "honest_verdict": "success: headline_auroc_beats_peer",
        "auroc": 0.9131, "kl_divergence": 0.9131,
    }
    assert len(_critical(fab)) >= 1


def test_bidt_5_baseline_vs_outcome_still_critical() -> None:
    """SCENARIO-BIDT-5 (GUARD): only ONE side reference/derived -> still flags."""
    d = {
        "experiment": "t", "honest_verdict": "success: x",
        "auroc_baseline": 0.9131,
        "kl_divergence": 0.9131,  # an independent OUTCOME metric
    }
    assert len(_critical(d)) >= 1


def test_bidt_6_real_exp4592_one_critical_remains() -> None:
    """SCENARIO-BIDT-6: the real exp4592 drops from 11 critical to exactly ONE — the legitimate
    two-treatment-outcome coincidence (generic_transfer_rate_with_wiring ==
    winner_generated_rate_with_wiring) — with the baseline/delta cascade downgraded to WARN."""
    d = json.loads((REPO / "results" / "experiment_4592_generation_completeness_wiring.json").read_text())
    assert "success:" in str(d.get("honest_verdict", ""))
    crit = _critical(d)
    assert len(crit) == 1, [f.detail for f in crit]
    assert "with_wiring" in crit[0].detail
    assert len(_warn(d)) >= 8


def test_bidt_7_named_deltas_without_backing_arithmetic_still_critical() -> None:
    """SCENARIO-BIDT-7 (GUARD — the adversarial-review hole): two distinct measured outcomes copied
    identical and merely NAMED `*_delta`, with NO backing treatment/baseline arithmetic present,
    STILL flag critical. Name alone does not establish derived-ness."""
    fab = {
        "experiment": "fab", "honest_verdict": "success: energy_beats_SC_on_both",
        "accuracy_delta_vs_self_consistency": 0.137,
        "auroc_delta": 0.137,
    }
    assert len(_critical(fab)) >= 1


def test_bidt_8_cross_metric_copied_delta_still_critical() -> None:
    """SCENARIO-BIDT-8 (GUARD): accuracy_delta is verifiable (its accuracy_* triple is present), but
    auroc_delta is copied with NO auroc_* backing — stem-binding refuses to verify it against the
    accuracy fields, so the accuracy_delta == auroc_delta pair stays critical."""
    fab = {
        "experiment": "fab", "honest_verdict": "success: x",
        "accuracy_baseline": 0.663, "accuracy_treatment": 0.80, "accuracy_delta": 0.137,
        "auroc_delta": 0.137,  # copied; no auroc_* operands to derive from
    }
    assert len(_critical(fab)) >= 1


def test_bidt_9_suffix_anchoring_rejects_substring_collisions() -> None:
    """SCENARIO-BIDT-9 (GUARD): suffix-anchoring must NOT classify measured fields whose names
    merely contain reference/diff/change as substrings."""
    assert av._is_reference_field("generic_transfer_rate_baseline") is True
    assert av._is_reference_field("generic_transfer_baseline_reference") is True
    assert av._is_reference_field("n_referenced_artifacts") is False
    assert av._is_reference_field("preference_score") is False
    assert av._delta_stem("transfer_delta") == "transfer"
    assert av._delta_stem("all_spins_different") is None
    assert av._delta_stem("no_full_discrete_diffusion") is None
    assert av._delta_stem("ops_changelog_modified") is None
