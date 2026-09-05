"""The moat-rigor vocabulary matches the concept, not one track's spelling.

Spec: REQ-VERIFY-7040 (openspec/capabilities/verification/spec.md).
SCENARIOs: SCENARIO-VERIFY-7040-1 (moat_survives in honest_verdict, in status, and
concatenated as moatMOAT_SURVIVES), SCENARIO-VERIFY-7040-2 (beats_vote is a beats-SC
claim and gets the full rigor contract), SCENARIO-VERIFY-7040-3 (the SC token is matched
on the full path), SCENARIO-VERIFY-7040-4 (MET as a leading gate-status token; METHOD is
not), SCENARIO-VERIFY-7040-5 (a negated claim is a null, never a win),
SCENARIO-VERIFY-7040-6 (right token boundaries: score_delta, beats_scissor,
moat_provenance do not match), SCENARIO-VERIFY-7040-7 (untuned / vanilla
self-consistency is a naive baseline), SCENARIO-VERIFY-7040-8 (the lint's own shipping
receipt is not a moat claim), SCENARIO-VERIFY-7040-9 (the ledger-named corpus artifacts
fire through the FULL verifier on a copy; results/** is read, never written).

Origin: nine SILENT_NON_FIRING rows dated 2026-09-04 in ops/audit-findings-ledger.md,
triaged 2026-09-05 to one shared root.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import scripts.adversarial_verify as av  # noqa: E402

KIND = "MOAT_CLAIM_RIGOR"


def _flags(d: dict) -> list[tuple[str, str]]:
    flags: list = []
    av.check_moat_claim_rigor(d, flags)
    return [(f.kind, f.severity) for f in flags if f.kind == KIND]


def test_moat_survives_is_a_moat_claim_wherever_it_is_spelled() -> None:
    # SCENARIO-VERIFY-7040-1: exp3916 (verdict), exp3827 (status only), exp3923 (concatenated).
    shapes = (
        {"honest_verdict": "complete: moat_scissor_MOAT_SURVIVES_residcatch_strong0.9143"},
        {"status": "complete: verifier_moat_survives_error_independent_residualcatch0.9000"},
        {
            "honest_verdict": "complete: capstone_v362_moatMOAT_SURVIVES_efficiencyCHEAPER_NOT_PARITY"
        },
    )
    for d in shapes:
        assert av._moat_rigor_claims_relevant(d), d
        # verifier_is_oracle is undeclared, so the circularity rule fires.
        assert _flags(d) == [(KIND, "critical")], d
    assert av._claims_moat({"status": "complete: verifier_moat_survives_x"})


def test_beats_vote_is_a_beats_sc_claim_with_the_full_rigor_contract() -> None:
    # SCENARIO-VERIFY-7040-2: exp4245's shape -- oracle-distinct, delta 0.44, and none of
    # headroom_present / paired_ci95 / mcnemar_p. Majority vote IS self-consistency.
    d = {
        "honest_verdict": "complete: arc_oracle_distinct_set_encoder_beats_vote",
        "verifier_is_oracle": False,
        "set_encoder_minus_vote_delta": 0.4423,
        "set_encoder_top1": 0.83,
        "vote_top1": 0.39,
    }
    assert av._moat_rigor_claims_relevant(d)
    assert av._moat_rigor_claims_win(d)
    assert _flags(d) == [(KIND, "critical"), (KIND, "critical")]
    clean = {
        **d,
        "headroom_present": True,
        "oracle_at_k": 0.83,
        "tuned_sc_accuracy": 0.39,
        "n_flips_possible": 12,
        "paired_ci95": [0.31, 0.60],
        "mcnemar_p": 0.001,
    }
    assert _flags(clean) == []


def test_a_claim_that_lives_only_in_status_reaches_the_win_branch() -> None:
    # SCENARIO-VERIFY-7040-1, the claim-key half. `beats_sc` is a relevance marker and NOT a
    # headline marker, so this reaches relevance and win only through the claim keys. A
    # mutation proof found `status` in the claim keys was otherwise double-covered.
    # `delta_vs_tuned_sc`, not `delta_vs_sc`: the latter is in the naive-SC leaf set and
    # correctly draws the naive warn as well (a first draft of this test got that wrong).
    d = {
        "status": "complete: hybrid_beats_sc_on_gsm8k",
        "verifier_is_oracle": False,
        "headroom_present": True,
        "delta_vs_tuned_sc": 0.05,
    }
    assert av._moat_rigor_claims_relevant(d)
    assert av._moat_rigor_claims_win(d)
    assert _flags(d) == [(KIND, "critical")]


def test_moat_survives_with_a_positive_delta_is_a_win_that_needs_significance() -> None:
    # SCENARIO-VERIFY-7040-1, the win-marker half: a survives claim with a measured lift
    # over SC is a win, and a win needs paired significance.
    d = {
        "honest_verdict": "complete: verifier_moat_survives_ablation",
        "verifier_is_oracle": False,
        "headroom_present": True,
        "verifier_minus_sc_delta": 0.04,
    }
    assert av._moat_rigor_claims_win(d)
    assert _flags(d) == [(KIND, "critical")]


def test_the_sc_token_is_matched_on_the_full_path() -> None:
    # SCENARIO-VERIFY-7040-3: exp3645 keeps its delta at verifier_over_sc_lift.delta.
    d = {
        "honest_verdict": "complete: verifier_beats_sc_on_headroom_corpus_hybrid_wins_under_budget",
        "verifier_is_oracle": False,
        "headroom_present": True,
        "verifier_over_sc_lift": {"delta": 0.0333, "ci95": [-0.067, 0.133]},
    }
    assert av._moat_rigor_positive_delta_items(d) == [("verifier_over_sc_lift.delta", 0.0333)]
    assert av._moat_rigor_claims_win(d)
    # The win branch now runs and demands paired significance.
    assert _flags(d) == [(KIND, "critical")]


def test_met_is_recognised_as_a_leading_token_and_method_is_not() -> None:
    # SCENARIO-VERIFY-7040-4: exp4346's status, with its verdict removed.
    assert av._flips_gate(
        {"diffusiongemma_gate_status": "MET_oracle_distinct_leak_robust_replicated"}
    )
    assert av._flips_gate({"diffusiongemma_gate_status": "MET"})
    assert av._flips_gate({"diffusiongemma_gate": {"status": "met_replicated"}})
    assert not av._flips_gate({"diffusiongemma_gate_status": "METHOD_pending"})
    assert not av._flips_gate({"diffusiongemma_gate_status": "STILL-PENDING"})
    assert not av._flips_gate(
        {"diffusiongemma_gate_status": "STILL_PENDING_second_corpus_scorer_leaky"}
    )


def test_a_negated_claim_is_a_null_not_a_win() -> None:
    # SCENARIO-VERIFY-7040-5: `does_not_beat_self_consistency` contains the win token
    # `beat_self_consistency`; null markers take precedence.
    d = {
        "honest_verdict": "complete: hybrid_does_not_beat_self_consistency",
        "delta_vs_sc": 0.02,
        "verifier_is_oracle": False,
    }
    assert av._moat_rigor_claims_null(d)
    assert not av._moat_rigor_claims_win(d)
    plural = {
        "honest_verdict": "complete: capstone_local_not_beats_vote",
        "x_minus_vote_delta": 0.01,
        "verifier_is_oracle": False,
    }
    assert av._moat_rigor_claims_null(plural)
    assert not av._moat_rigor_claims_win(plural)


def test_right_token_boundaries_reject_the_known_false_positives() -> None:
    # SCENARIO-VERIFY-7040-6
    assert av._moat_rigor_positive_delta_items({"score_delta": 0.5}) == []
    assert av._moat_rigor_positive_delta_items({"disc_delta": 0.5}) == []
    assert not av._moat_rigor_claims_relevant(
        {"honest_verdict": "complete: beats_scissor_baseline"}
    )
    assert not av._claims_moat({"honest_verdict": "complete: moat_provenance_audit"})
    assert av._moat_marker_present("x_beats_sc_y", "beats_sc")
    assert not av._moat_marker_present("beats_scissor", "beats_sc")


def test_untuned_and_vanilla_self_consistency_are_naive_baselines() -> None:
    # SCENARIO-VERIFY-7040-7: `untuned_self_consistency_accuracy` used to read as TUNED.
    assert av._moat_rigor_uses_naive_sc({"untuned_self_consistency_accuracy": 0.5})
    assert av._moat_rigor_uses_naive_sc({"vanilla_sc_accuracy": 0.5})
    assert not av._moat_rigor_uses_naive_sc({"tuned_sc_accuracy": 0.5, "naive_sc_accuracy": 0.4})


def test_the_lints_own_shipping_receipt_is_not_a_moat_claim() -> None:
    # SCENARIO-VERIFY-7040-8: exp5008, `success_moat_rigor_lint_shipped_fixtures_green.`,
    # quarantined its own receipt through the over-broad `success_moat` marker.
    d = {
        "honest_verdict": "success_moat_rigor_lint_shipped_fixtures_green.",
        "inference_substrate": "aggregation_from_upstream_artifacts",
    }
    assert not av._moat_rigor_claims_relevant(d)
    assert _flags(d) == []
    assert av._moat_rigor_claims_relevant({"honest_verdict": "success_verifier_moat_beats_sc_musr"})


LEDGER_NAMED = (
    "results/experiment_3916_moat_scissor_accuracy.json",
    "results/experiment_3827_verifier_error_independence_scissor.json",
    "results/experiment_3923_capstone_v362.json",
    "results/experiment_4245_arc_set_encoder_beats_vote.json",
    "results/experiment_3645_headroom_hybrid_verifier_vs_sc_v3.json",
    "results/experiment_4346_capstone_v401.json",
)


def test_the_ledger_named_artifacts_fire_through_the_full_verifier_on_a_copy(
    tmp_path: Path,
) -> None:
    # SCENARIO-VERIFY-7040-9. Copies only: results/** is evidence and is never written.
    for rel in LEDGER_NAMED:
        src = REPO_ROOT / rel
        copy = tmp_path / src.name
        copy.write_bytes(src.read_bytes())
        report = av.verify_artifact(copy)
        critical = [f["kind"] for f in report["flags"] if f["severity"] == "critical"]
        assert KIND in critical, rel
    src = REPO_ROOT / "results/experiment_5008_moat_oracle_distinct_lint.json"
    copy = tmp_path / src.name
    copy.write_bytes(src.read_bytes())
    report = av.verify_artifact(copy)
    assert KIND not in [f["kind"] for f in report["flags"]]
