"""Regression guards for adversarial_verify.py false-positive / false-negative
hardening (2026-05-31).

Covers three behaviours the outer-loop session added:
  1. TAUTOLOGY no longer flags identifier/seed/metadata fields (the
     exp3505/3506/3496/3481 false positives — experiment_id == random_seed ==
     experiment number is good reproducibility, not a coincidence).
  2. TAUTOLOGY STILL flags two genuinely-distinct metrics that coincide
     (regression: the hardening must not blind the real check).
  3. FALSE_NEGATIVE_RISK fires on a NULL claim that lacks a positive control
     (flip_count==0 / oracle<=baseline / self-reported g2=False) — the exp3507
     degenerate-reranker trap — and does NOT fire on a positive claim.

These trace to the CLAUDE.md "Adversarial Artifact Verification" rule:
FALSE_NEGATIVE_RISK + TAUTOLOGY-excludes-identifiers (2026-05-31).
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import adversarial_verify as av  # noqa: E402


def _kinds(flags):
    return [f.kind if hasattr(f, "kind") else f.get("kind") for f in flags]


# --- 1. TAUTOLOGY excludes identifier/seed fields --------------------------

@pytest.mark.parametrize(
    "d",
    [
        {"experiment_id": 3506.0, "experiment": 3506.0, "random_seed": 3506.0},
        {"experiment": 3505.0, "random_seed": 3505.0},
        {"experiment_id": 3496.0, "rng_seed": 3496.0},
        {"milestone": 324.0, "torch_seed": 324.0},
    ],
)
def test_tautology_excludes_identifier_seed_pairs(d):
    """REQ: identifier/seed fields equal to the experiment number must NOT
    raise TAUTOLOGY. Seeding off the experiment id is good practice."""
    flags = []
    av.check_tautology(d, flags)
    assert "TAUTOLOGY" not in _kinds(flags), f"false positive on {d}"


def test_is_identifier_field():
    for k in ("experiment_id", "experiment", "random_seed", "rng_seed",
              "np_seed", "milestone", "run_id", "task_id", "gpu_id"):
        assert av._is_identifier_field(k), k
    for k in ("final_loss", "auroc", "v1_recall", "gradient_norm", "accuracy"):
        assert not av._is_identifier_field(k), k


# --- 2. TAUTOLOGY still catches genuine distinct-metric coincidence ---------

def test_tautology_still_fires_on_distinct_float_metrics():
    """REQ: two distinct measured metrics agreeing to >5 sig figs is still a
    critical signal. The identifier hardening must not blind this."""
    flags = []
    av.check_tautology(
        {"nrgpt_grad_norm": 9.762872695922852, "ce_grad_norm": 9.762872695922852},
        flags,
    )
    assert "TAUTOLOGY" in _kinds(flags)


def test_tautology_real_metric_kept_even_with_identifiers_present():
    """REQ: excluding id/seed must not suppress a real metric tautology in the
    same artifact."""
    flags = []
    av.check_tautology(
        {
            "experiment_id": 3500.0,
            "random_seed": 3500.0,
            "final_loss": 0.123456789,
            "val_loss": 0.123456789,
        },
        flags,
    )
    assert "TAUTOLOGY" in _kinds(flags)


def test_tautology_allows_same_corpus_selector_cross_family_delta_tie():
    """REQ-VERIFY-4283: selector arms may legitimately tie on the same corpus."""
    flags = []
    av.check_tautology(
        {
            "static_cross_family_delta": 0.5,
            "tier2_cross_family_delta": 0.5,
        },
        flags,
    )
    assert "TAUTOLOGY" not in _kinds(flags)


# --- 3. FALSE_NEGATIVE_RISK on degenerate null claims ----------------------

def test_fnr_flip_count_zero():
    """REQ: a null claim with flip_count==0 means the method never acted →
    cannot conclude it fails (exp3507)."""
    flags = []
    av.check_false_negative_risk(
        {
            "honest_verdict": "complete: process_energy_does_not_change_selections",
            "flip_count_optimal_vs_sc": 0,
        },
        flags,
    )
    assert "FALSE_NEGATIVE_RISK" in _kinds(flags)


def test_fnr_oracle_not_above_baseline():
    """REQ: a null claim where the oracle upper bound does not exceed the
    baseline means the corpus has no headroom — no method could win."""
    flags = []
    av.check_false_negative_risk(
        {
            "honest_verdict": "complete: no_improvement_over_self_consistency",
            "optimal_aggregation_accuracy": 0.653061,
            "self_consistency_accuracy": 0.653061,
        },
        flags,
    )
    assert "FALSE_NEGATIVE_RISK" in _kinds(flags)


def test_fnr_self_reported_degenerate_gate():
    """REQ: a null claim whose own non-degeneracy/g2 gate is False is
    self-reporting a degenerate test."""
    flags = []
    av.check_false_negative_risk(
        {
            "honest_verdict": "complete: selection_premise_refuted",
            "acceptance_gate_g2_non_degenerate_flips": False,
        },
        flags,
    )
    assert "FALSE_NEGATIVE_RISK" in _kinds(flags)


def test_fnr_does_not_fire_on_positive_claim():
    """REQ: a POSITIVE claim (method beat baseline, flips happened) must not be
    false-flagged as a false-negative risk."""
    flags = []
    av.check_false_negative_risk(
        {
            "honest_verdict": "complete: energy_reranker_beats_self_consistency",
            "flip_count_optimal_vs_sc": 12,
            "optimal_aggregation_accuracy": 0.71,
            "self_consistency_accuracy": 0.65,
        },
        flags,
    )
    assert "FALSE_NEGATIVE_RISK" not in _kinds(flags)


def test_fnr_does_not_fire_when_oracle_exceeds_baseline_even_if_null():
    """REQ: if there IS headroom (oracle>baseline) the null is informative —
    the signal-2 false-negative guard must not fire."""
    flags = []
    av.check_false_negative_risk(
        {
            "honest_verdict": "complete: method_no_gain_despite_headroom",
            "optimal_aggregation_accuracy": 0.80,
            "self_consistency_accuracy": 0.65,
        },
        flags,
    )
    # signal-2 must not fire (oracle>baseline); no flip/gate fields present
    assert "FALSE_NEGATIVE_RISK" not in _kinds(flags)


def test_req_capstone_4563_positive_control_failed_efficiency_null_fires():
    """REQ-CAPSTONE-4563: exp4544 is a broken positive-control null, not evidence."""
    fixture = (
        Path(__file__).resolve().parents[2]
        / "results"
        / "experiment_4544_llm_proposer_reinduction.json"
    )
    payload = json.loads(fixture.read_text(encoding="utf-8"))

    assert payload["positive_control_passed"] is False
    assert payload["false_negative_risk_checked"] is False

    flags = []
    av.check_false_negative_risk(payload, flags)
    matching = [
        flag
        for flag in flags
        if flag.kind == "FALSE_NEGATIVE_RISK" and "false_negative_risk_open" in flag.detail
    ]
    assert matching, "positive-control-failed efficiency null must open false-negative risk"


def test_req_capstone_4563_passed_positive_control_null_does_not_fire():
    """REQ-CAPSTONE-4563: a null with a passed positive control remains a valid null."""
    flags = []
    av.check_false_negative_risk(
        {
            "honest_verdict": "complete: llm_proposer_no_deeper_level_honest_null",
            "positive_control_passed": True,
            "false_negative_risk_checked": True,
            "core_efficiency_baseline": 2.0074,
            "core_efficiency_best": 2.0074,
            "efficiency_delta": 0.0,
            "null_delta_methodology_note": "matched measurement; no deeper level reached.",
        },
        flags,
    )
    assert "FALSE_NEGATIVE_RISK" not in _kinds(flags)


# --- 4. DEGENERATE_SEPARATION catches wrong-majority synthetic wins --------

def test_degenerate_separation_flags_vote_zero_delta_one_arcgen_win():
    """REQ-VERIFY-4291: a +1 selector-vs-vote win with vote@1=0 and oracle@K=1
    is a degenerate ARC-GEN pool construction, not transfer evidence."""
    flags = []
    av.check_degenerate_separation(
        {
            "honest_verdict": "complete: arcgen_cross_generator_generalizes",
            "cross_generator_delta": 1.0,
            "vote_at_1": 0.0,
            "oracle_at_k": 1.0,
            "verifier_is_oracle": False,
        },
        flags,
    )
    assert "DEGENERATE_SEPARATION" in _kinds(flags)


def test_degenerate_separation_allows_nondegenerate_arcgen_read():
    """SCENARIO-VERIFY-4291: non-zero vote, sub-ceiling oracle, and delta<0.95
    must pass the ARC-GEN non-degeneracy guard."""
    flags = []
    av.check_degenerate_separation(
        {
            "honest_verdict": "complete: arcgen_cross_generator_generalizes",
            "cross_generator_delta": 0.42,
            "vote_at_1": 0.25,
            "oracle_at_k": 0.88,
            "verifier_is_oracle": False,
        },
        flags,
    )
    assert "DEGENERATE_SEPARATION" not in _kinds(flags)


# --- 4. Backfill backstop precision (DURATION live-claim guard) -------------

def test_claims_live_model():
    assert av._claims_live_model({"model_specs": [{"name": "Qwen"}]})
    assert av._claims_live_model({"target_model": "gemma-4-31B"})
    assert av._claims_live_model({"inference_substrate": "live_llm_inference"})
    # an aggregation/audit artifact that names no model does NOT claim a live run
    assert not av._claims_live_model({"roce_success_rate": 0.8})
    assert not av._claims_live_model(
        {"inference_substrate": "aggregation_from_upstream_artifacts"}
    )


def test_backfill_dryrun_does_not_mutate(tmp_path):
    """REQ: dry-run reports scope without writing flagged_adversarial."""
    art = {
        "experiment": 9001,
        "duration_s": 0.5,
        "model_specs": [{"name": "Qwen3.6-35B-A3B-GGUF"}],
        "honest_verdict": "complete: live_eval_finished",
    }
    p = tmp_path / "experiment_9001_x.json"
    p.write_text(json.dumps(art))
    recs = av.backfill_stamps([p], apply=False)
    assert len(recs) == 1 and recs[0]["written"] is False
    assert "flagged_adversarial" not in json.loads(p.read_text())


def test_backfill_apply_stamps_real_live_claim(tmp_path):
    """REQ: a sub-floor-duration artifact that DECLARES a model gets stamped."""
    art = {
        "experiment": 9002,
        "duration_s": 0.5,
        "model_specs": [{"name": "Qwen3.6-35B-A3B-GGUF"}],
        "honest_verdict": "complete: live_eval_finished",
    }
    p = tmp_path / "experiment_9002_x.json"
    p.write_text(json.dumps(art))
    recs = av.backfill_stamps([p], apply=True)
    assert len(recs) == 1 and recs[0]["written"] is True
    out = json.loads(p.read_text())
    assert out["flagged_adversarial"] is True
    assert out["corrigendum_pending"] and out.get("corrigendum_note")


def test_backfill_skips_aggregation_duration_false_positive(tmp_path):
    """REQ: a fast aggregation/audit artifact that names NO model must NOT be
    stamped even though it references compute markers in prose (the operator's
    explicit false-positive concern — exp1877/1498/1459)."""
    art = {
        "experiment": 9003,
        "duration_s": 0.001,
        "notes": "scored candidates from the GGUF run upstream",  # prose marker
        "honest_verdict": "complete: reachability_audit_done",
    }
    p = tmp_path / "experiment_9003_x.json"
    p.write_text(json.dumps(art))
    recs = av.backfill_stamps([p], apply=True)
    assert recs == [] or all(r["written"] is False for r in recs)
    assert "flagged_adversarial" not in json.loads(p.read_text())


def test_backfill_idempotent_skips_already_stamped(tmp_path):
    art = {
        "experiment": 9004,
        "duration_s": 0.5,
        "model_specs": [{"name": "x"}],
        "flagged_adversarial": True,
    }
    p = tmp_path / "experiment_9004_x.json"
    p.write_text(json.dumps(art))
    assert av.backfill_stamps([p], apply=True) == []


# --- 5. CEILING_SATURATION (positive-claim no-headroom partner) -------------

def test_ceiling_saturation_trivial_baseline_ties():
    """REQ: a superiority claim where a trivial baseline also saturates the
    ceiling is uninformative (exp3518: vanilla_descent also solved 100%)."""
    flags = []
    av.check_ceiling_saturation(
        {
            "honest_verdict": "complete: energy_vs_ar_generalizes_to_graph_coloring",
            "ar_baseline_solve_rate": 0.5,
            "solve_rate": 1.0,
            "solve_rate_by_optimizer_variant": {
                "vanilla_descent": 1.0,
                "parallel_tempering": 1.0,
                "exact_backtracking": 1.0,
            },
        },
        flags,
    )
    assert "CEILING_SATURATION" in _kinds(flags)


def test_ceiling_saturation_difficulty_inert():
    """REQ: a comparative claim where every difficulty tier saturates means the
    difficulty axis is inert — a 'solves hard instances' claim is unsupported."""
    flags = []
    av.check_ceiling_saturation(
        {
            "honest_verdict": "complete: method_beats_baseline",
            "baseline_solve_rate": 0.4,
            "solve_rate": 1.0,
            "solve_rate_by_difficulty": {
                "easy": 1.0, "medium": 1.0, "hard": 1.0, "extreme": 1.0
            },
        },
        flags,
    )
    assert "CEILING_SATURATION" in _kinds(flags)


def test_ceiling_saturation_silent_on_noncomparative():
    """REQ: a pure sanity check (no superiority claim) that happens to all-pass
    must NOT be flagged — the gate is only meaningful for comparative claims."""
    flags = []
    av.check_ceiling_saturation(
        {
            "honest_verdict": "complete: sanity_smoke_all_pass",
            "solve_rate_by_difficulty": {"easy": 1.0, "hard": 1.0},
        },
        flags,
    )
    assert "CEILING_SATURATION" not in _kinds(flags)


def test_ceiling_saturation_silent_with_real_headroom():
    """REQ: a comparative claim where the trivial baseline does NOT saturate
    (real headroom) is a valid test — must not fire."""
    flags = []
    av.check_ceiling_saturation(
        {
            "honest_verdict": "complete: energy_beats_ar",
            "ar_baseline_solve_rate": 0.3,
            "solve_rate": 0.9,
            "solve_rate_by_optimizer_variant": {
                "vanilla_descent": 0.4, "parallel_tempering": 0.9
            },
            "solve_rate_by_difficulty": {"easy": 1.0, "hard": 0.6},
        },
        flags,
    )
    assert "CEILING_SATURATION" not in _kinds(flags)
