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
import summarize_artifact as summary_reader  # noqa: E402


def _kinds(flags):
    return [f.kind if hasattr(f, "kind") else f.get("kind") for f in flags]


def _report_for_payload(tmp_path: Path, payload: dict) -> dict:
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return av.verify_artifact(path)


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


def test_req_report_5224_row_count_fields_are_not_tautology_metrics():
    """REQ-REPORT-5224: canonical_pool_n may equal regenerated_rows by construction."""

    flags = []
    av.check_tautology(
        {
            "canonical_pool_n": 120,
            "regenerated_rows": 120,
            "repaired_rows": 0,
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


# --- 4b. Inference-substrate floor for learned CNN action models ------------

def test_req_arc_fcp_4575_learned_cnn_torch_marker_uses_offline_floor(
    tmp_path: Path,
) -> None:
    """REQ-ARC-FCP-4575: a fast learned-CNN action-model artifact with a torch
    marker and cached-candidate substrate must use the 1s floor, not the 60s
    live-model floor."""
    artifact = {
        "experiment": "experiment_4575_cnn_fixture",
        "honest_verdict": "complete: learned_cnn_fixture_real_fast_forward",
        "inference_substrate": av.VERIFIER_SCORING_SUBSTRATE,
        "duration_s": 5.0,
        "model_specs": {
            "architecture": "learned frame-action CNN",
            "framework": "torch",
            "device": "cpu",
        },
        "random_seed": 4575,
        "reproducibility_checksum": "sha256:" + "a" * 64,
    }

    floor = av.duration_floor_for_artifact(artifact)
    report = _report_for_payload(tmp_path, artifact)

    assert floor == {
        "substrate": av.VERIFIER_SCORING_SUBSTRATE,
        "min_duration_s": av.VERIFIER_SCORING_MIN_DURATION_S,
        "reason": "verifier_scoring",
    }
    assert "DURATION_TOO_SHORT" not in _kinds(report["flags"])


def test_req_arc_fcp_4575_live_llm_gguf_still_uses_60s_floor(
    tmp_path: Path,
) -> None:
    """REQ-ARC-FCP-4575: the CNN carve-out must not weaken the real live-LLM
    fabrication check."""
    artifact = {
        "experiment": "experiment_4575_fake_llm_fixture",
        "honest_verdict": "complete: fake_llm_claim_finished_too_fast",
        "inference_substrate": "live_llm_inference",
        "duration_s": 5.0,
        "model_specs": [{"name": "unsloth/Qwen3.6-35B-A3B-GGUF"}],
        "random_seed": 4575,
        "reproducibility_checksum": "sha256:" + "b" * 64,
    }

    floor = av.duration_floor_for_artifact(artifact)
    report = _report_for_payload(tmp_path, artifact)
    duration_flags = [flag for flag in report["flags"] if flag["kind"] == "DURATION_TOO_SHORT"]

    assert floor == {
        "substrate": "live_llm_inference",
        "min_duration_s": av.COMPUTE_BOUND_MIN_DURATION_S,
        "reason": "live_model",
    }
    assert duration_flags
    assert duration_flags[0]["severity"] == "critical"


def test_req_arc_fcp_4575_summary_surfaces_applied_floor(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-ARC-FCP-4575: summarize_artifact prints the floor chosen by the
    inference-substrate declaration before headline metrics."""
    artifact = {
        "experiment": "experiment_4575_cnn_fixture",
        "honest_verdict": "complete: learned_cnn_fixture_real_fast_forward",
        "inference_substrate": (
            "verifier_ensemble_against_cached_candidates -- learned CNN CPU forward pass"
        ),
        "duration_s": 5.0,
        "model_specs": {
            "architecture": "learned frame-action CNN",
            "framework": "torch",
            "device": "cpu",
        },
        "random_seed": 4575,
        "reproducibility_checksum": "sha256:" + "c" * 64,
    }
    path = tmp_path / "experiment_4575_cnn_fixture.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    assert summary_reader.summarize(path) == 0
    out = capsys.readouterr().out

    assert "duration floor" in out
    assert "verifier_ensemble_against_cached_candidates" in out
    assert ">=1.0s" in out
    assert "adversarial flags: none" in out


# --- 4c. Offline ARC methodology descriptor suppresses false warn ----------

def _offline_arc_methodology_fixture() -> dict:
    return {
        "experiment": "experiment_4587_offline_arc_fixture",
        "honest_verdict": "complete: offline_arc_fixture_methodology_cited",
        "inference_substrate": (
            "verifier_ensemble_against_cached_candidates -- offline ARC solve, no LLM load"
        ),
        "duration_s": 5.0,
        "solver_module": "python/carnot/agentic/arc_solver_kit.py",
        "reproduction_gate": {
            "entrypoint": "arc_solver_kit.reproduce()",
            "checksum": "sha256:" + "d" * 64,
        },
        "verifier_checkpoint": "models/arc_verifier_ar25.json",
        "random_seed": 4587,
        "reproducibility_checksum": "sha256:" + "e" * 64,
        "preconditions_checked": {
            "torch_import_required": False,
            "offline_arcade_import_smoke": True,
        },
    }


def test_req_arc_wmte_4587_offline_arc_descriptor_suppresses_methodology_warn(
    tmp_path: Path,
) -> None:
    """REQ-ARC-WMTE-4587: offline ARC solver/checksum/checkpoint methodology is
    sufficient without model_specs."""
    report = _report_for_payload(tmp_path, _offline_arc_methodology_fixture())

    methodology_flags = [
        flag for flag in report["flags"] if flag["kind"] == "METHODOLOGY_MISSING"
    ]

    assert methodology_flags == []


@pytest.mark.parametrize(
    "artifact_name",
    [
        "experiment_4568_clickability_action_effect_predictor.json",
        "experiment_4572_integration_gate.json",
        "experiment_4573_primitive_persist_transfer.json",
    ],
)
def test_req_arc_wmte_4587_dot422_arc_artifacts_do_not_methodology_warn(
    artifact_name: str,
) -> None:
    """REQ-ARC-WMTE-4587: the real .422 offline ARC artifacts are covered."""
    fixture = Path(__file__).resolve().parents[2] / "results" / artifact_name
    report = av.verify_artifact(fixture)

    methodology_flags = [
        flag for flag in report["flags"] if flag["kind"] == "METHODOLOGY_MISSING"
    ]

    assert methodology_flags == []


def test_req_arc_wmte_4587_live_llm_missing_model_specs_still_warns(
    tmp_path: Path,
) -> None:
    """REQ-ARC-WMTE-4587: live_llm_inference still requires model_specs."""
    artifact = {
        "experiment": "experiment_4587_live_llm_fixture",
        "honest_verdict": "complete: live_llm_fixture_missing_model_specs",
        "inference_substrate": "live_llm_inference",
        "duration_s": 120.0,
        "random_seed": 4587,
        "reproducibility_checksum": "sha256:" + "f" * 64,
    }

    report = _report_for_payload(tmp_path, artifact)
    methodology_flags = [
        flag for flag in report["flags"] if flag["kind"] == "METHODOLOGY_MISSING"
    ]

    assert methodology_flags
    assert "model_specs/target_model" in methodology_flags[0]["detail"]


def test_req_arc_wmte_4587_summary_surfaces_offline_methodology_descriptor(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SCENARIO-ARC-WMTE-4587: summary shows the recognized offline descriptor."""
    path = tmp_path / "experiment_4587_offline_arc_fixture.json"
    path.write_text(json.dumps(_offline_arc_methodology_fixture()), encoding="utf-8")

    assert summary_reader.summarize(path) == 0
    out = capsys.readouterr().out

    assert "methodology" in out
    assert "offline_arc_methodology_descriptor" in out
    assert "solver_module" in out


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


# --------------------------------------------------------------------------- #
# Separator-tolerant inference_substrate matching (2026-06-26 outer-loop fix)  #
# Origin: .437 B2 exp4756 declared "aggregation_from_upstream_artifacts; 100us  #
# floor." -- the `;` separator (a human note appended to the canonical value)  #
# was not recognized, so the aggregation duration floor was NOT applied and a   #
# legit 2.9s submission-package aggregation got DURATION_TOO_SHORT-flagged. The #
# matcher must tolerate ANY separator after the canonical token, while NOT      #
# matching a longer different enum (`<canonical>_v2`).                          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("aggregation_from_upstream_artifacts", True),  # bare
        ("aggregation_from_upstream_artifacts; 100us floor.", True),  # the exp4756 `;` case
        ("aggregation_from_upstream_artifacts -- reads upstream JSON", True),  # legacy `--`
        ("aggregation_from_upstream_artifacts (0.0001s floor)", True),  # space separator
        ("aggregation_from_upstream_artifacts, cites exp4761", True),  # comma
        ("aggregation_from_upstream_artifacts: floor 100us", True),  # colon
        ("aggregation_from_upstream_artifacts_v2", False),  # DIFFERENT enum -- must NOT match
        ("live_llm_inference", False),  # different substrate
    ],
)
def test_inference_substrate_matches_tolerates_any_separator(raw, expected):
    d = {"inference_substrate": raw}
    assert av._inference_substrate_matches(d, av.AGGREGATION_SUBSTRATE) is expected


def test_aggregation_with_semicolon_note_uses_aggregation_floor():
    """The exp4756 regression: `;`-separated note must still select the aggregation
    floor (not the 60s compute-bound floor), so a fast aggregation is not flagged."""
    d = {"inference_substrate": "aggregation_from_upstream_artifacts; 100us floor."}
    assert av._is_aggregation_only(d) is True
    floor = av.duration_floor_for_artifact(d)
    assert floor is not None
    assert floor["reason"] == "aggregation"
    assert floor["min_duration_s"] <= 0.001


def test_blocked_precondition_check_only_has_no_live_model_floor(tmp_path: Path):
    """REQ-VERIFY-5017: blocked precondition artifacts may name the missing model."""

    artifact = {
        "schema": "carnot.experiment_5017_lora_ebm_scorer_musr_v2.v1",
        "honest_verdict": "blocked_trainable_qwen_base",
        "inference_substrate": "precondition_check_only",
        "duration_s": 1.0,
        "model_specs": {"base_model": "Qwen/Qwen3.5-1.7B"},
        "random_seed": 20260630,
        "reproducibility_checksum": "sha256:fixture",
    }
    flags = []
    av.check_duration_vs_claim(artifact, flags)
    report = _report_for_payload(tmp_path, artifact)

    assert av.duration_floor_for_artifact(artifact) is None
    assert "DURATION_TOO_SHORT" not in _kinds(flags)
    assert "DURATION_TOO_SHORT" not in [flag["kind"] for flag in report["flags"]]


def test_artifact_provenance_audit_quotes_upstream_compute_markers_without_live_floor(
    tmp_path: Path,
):
    """REQ-REPORT-5223: provenance audits may quote quarantined live-model artifacts."""

    artifact = {
        "experiment": "experiment_5223_gap4_flagged_pool_authenticity_audit_v478",
        "schema": "carnot.gap4_flagged_pool_authenticity_audit_5223.v1",
        "honest_verdict": "complete: old GAP-4 pool must be regenerated",
        "inference_substrate": "artifact_provenance_audit",
        "duration_s": 0.01,
        "reproducibility_checksum": "sha256:fixture",
        "artifact_findings": {
            "upstream": {
                "detail": (
                    "quoted upstream GGUF / CUDA / unsloth/Qwen3.6-35B-A3B-GGUF "
                    "duration_s=48.6 blocker"
                )
            }
        },
    }
    flags = []
    av.check_duration_vs_claim(artifact, flags)
    av.check_methodology_present(artifact, flags)
    report = _report_for_payload(tmp_path, artifact)

    floor = av.duration_floor_for_artifact(artifact)
    assert floor is not None
    assert floor["reason"] == "deterministic_verifier"
    assert "DURATION_TOO_SHORT" not in _kinds(flags)
    assert "METHODOLOGY_MISSING" not in _kinds(flags)
    assert "DURATION_TOO_SHORT" not in [flag["kind"] for flag in report["flags"]]
    assert "METHODOLOGY_MISSING" not in [flag["kind"] for flag in report["flags"]]


# --------------------------------------------------------------------------- #
# TAUTOLOGY excludes wall-clock TIMESTAMP fields (2026-06-26 outer-loop fix)   #
# Origin: .438 A3 self-play (exp4763) -- checkpoint_mtime_before_ns vs         #
# checkpoint_mtime_after_ns was a GENUINE ~2s checkpoint advance (verdict      #
# success_..._checkpoint_refreshed) but flagged TAUTOLOGY because two ns epoch #
# timestamps share leading sig figs (both 1.78244...). Same class as the       #
# identifier/seed carve-out. Must NOT exclude real durations (latency_ns).     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "field,is_excluded",
    [
        ("checkpoint_mtime_before_ns", True),  # the exp4763 case
        ("checkpoint_mtime_after_ns", True),
        ("created_timestamp", True),
        ("run_started_ts", True),
        ("build_epoch", True),
        ("wall_clock_ns", True),  # has time-ish token 'clock'
        # safety: real measured metrics / durations must stay IN the check
        ("latency_ns", False),
        ("duration_ns", False),
        ("loo_auroc_structural", False),
        ("heldout_first_win_rate", False),
    ],
)
def test_tautology_excludes_timestamp_fields(field, is_excluded):
    assert av._is_identifier_field(field) is is_excluded


def test_tautology_does_not_fire_on_checkpoint_mtime_pair():
    """The exp4763 regression: a genuine before/after mtime advance must not flag."""
    d = {
        "experiment": 4763,
        "honest_verdict": "success_re86_L2_checkpoint_refreshed",
        "checkpoint_mtime_before_ns": 1.7824478245866583e18,
        "checkpoint_mtime_after_ns": 1.7824497419434204e18,
    }
    flags = []
    av.check_tautology(d, flags)
    assert not [f for f in flags if _kinds([f])[0] == "TAUTOLOGY"]


# --------------------------------------------------------------------------- #
# TAUTOLOGY skips two SCORE metrics both at the 0.5 chance floor (2026-06-26)   #
# Origin: exp4771 (S0' origin-matched) success_..._reopens_s1 was flagged       #
# because loo_auroc_majority_control=0.5 (definitional floor) ==                #
# origin_probe_auroc=0.5 (the SUCCESS signal origin-matching removed the leak). #
# --------------------------------------------------------------------------- #
def test_tautology_skips_two_chance_floor_auroc_controls():
    d = {
        "honest_verdict": "success_structural_energy_s0prime_reopens_s1",
        "loo_auroc_majority_control": 0.5,
        "origin_probe_auroc": 0.5,
        "loo_auroc_structural": 0.7386642861889572,
    }
    flags = []
    av.check_tautology(d, flags)
    assert not [f for f in flags if _kinds([f])[0] == "TAUTOLOGY"]


def test_tautology_still_fires_on_real_distinct_metric_coincidence():
    """Safety: two genuinely-distinct high-precision metrics coinciding (NOT at
    the 0.5 floor) must still flag -- the carve-out must not blind the check."""
    d = {"loo_auroc_structural": 0.738664, "unrelated_energy_gap": 0.738664}
    flags = []
    av.check_tautology(d, flags)
    assert [f for f in flags if _kinds([f])[0] == "TAUTOLOGY"]


def test_chance_floor_score_recognizes_auroc_probe_control_but_not_metrics():
    assert av._is_chance_floor_score("origin_probe_auroc") is True
    assert av._is_chance_floor_score("loo_auroc_majority_control") is True
    assert av._is_chance_floor_score("shuffled_label_control_auroc") is True
    assert av._is_chance_floor_score("heldout_first_win_rate") is False
    assert av._is_chance_floor_score("loo_auroc_structural") is True  # auroc-named (ok; needs ==0.5 to skip)


# --------------------------------------------------------------------------- #
# DEGENERATE_CANDIDATE_POOL on S2-style engine-selection (2026-06-26)          #
# Origin: exp4791 (S2 off-path trust gate) reported energy_minus_accuracy_delta #
# == 0.0 + a BOUNDED 'no_live_trust_value' verdict, but 2/5 games had           #
# behaviorally-identical candidate engines -> a NON-TEST, not a genuine null.    #
# --------------------------------------------------------------------------- #
def _s2_artifact(
    per_game_recalls,
    verdict="complete_structural_energy_s2_no_live_trust_value",
    min_games=5,
    field="heldout_cell_recall",
):
    return {
        "experiment": "structural_energy_s2_offpath_trust_gate",
        "honest_verdict": verdict,
        "energy_minus_accuracy_delta": 0.0,
        "min_heldout_games": min_games,
        "game_results": [
            {"candidate_rows": [{field: r} for r in recalls]} for recalls in per_game_recalls
        ],
    }


def _has_degen(flags):
    return bool([f for f in flags if _kinds([f])[0] == "DEGENERATE_CANDIDATE_POOL"])


def _run_degen(d):
    flags = []
    av.check_engine_selection_candidate_diversity(d, flags)
    return _has_degen(flags)


def test_degenerate_candidate_pool_fires_on_low_effective_games():
    # exp4791 shape: 2 of 5 games behaviorally diverse, required floor 5
    assert _run_degen(_s2_artifact([[1.0, 1.0], [1.0, 1.0], [0.3, 0.3], [0.3, 0.7], [0.4, 0.9]]))


def test_degenerate_candidate_pool_does_not_fire_when_pool_is_diverse():
    # all 5 games behaviorally diverse, required 5 -> a genuine test, no flag
    assert not _run_degen(_s2_artifact([[0.3, 0.7]] * 12))


def test_degenerate_candidate_pool_does_not_fire_on_non_selection_artifact():
    flags = []
    av.check_engine_selection_candidate_diversity({"honest_verdict": "success_foo", "loo_auroc": 0.7}, flags)
    assert not flags


# --- hardening regressions (adversarial verify wf_3c4337f4) ------------------ #
def test_degen_self_declared_min_games_cannot_dodge():
    # FN-3: an artifact declaring min_heldout_games=1 (or 0) at effective=2/5 must STILL flag
    assert _run_degen(_s2_artifact([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [0.3, 0.7], [0.4, 0.9]], min_games=1))
    assert _run_degen(_s2_artifact([[1.0, 1.0]] * 5, min_games=0))


def test_degen_negative_delta_null_flags():
    # FN-2: a negative delta (energy strictly LOSES) on a degenerate pool is a no-value read
    d = _s2_artifact([[1.0, 1.0]] * 5, verdict="complete_s2_energy_below_control")
    d["energy_minus_accuracy_delta"] = -0.2
    assert _run_degen(d)


def test_degen_noise_manufactured_diversity_flags():
    # FN-4: 1e-8 float-noise "diversity" is not a real spread -> still degenerate
    assert _run_degen(_s2_artifact([[0.5, 0.5 + 1e-8]] * 5))


def test_degen_pass_verdict_on_degenerate_pool_now_flags():
    # attack #3: a PASS off a degenerate pool is NOT exempt (the old test codified this hole)
    d = _s2_artifact([[1.0, 1.0]] * 5, verdict="success_structural_energy_s2_trust_gate_authorizes_s3")
    d["energy_minus_accuracy_delta"] = 0.21
    assert _run_degen(d)


def test_degen_genuinely_diverse_pass_does_not_flag():
    # a PASS on a genuinely diverse pool (5 effective) must NOT flag
    d = _s2_artifact([[0.3, 0.9]] * 12, verdict="success_structural_energy_s2_trust_gate_authorizes_s3")
    d["energy_minus_accuracy_delta"] = 0.21
    assert not _run_degen(d)


def test_degen_renamed_delta_field_still_recognized():
    # FN-1: a degenerate S2 whose delta is renamed 'trust_gate_margin' is still recognized via the S2 schema
    d = _s2_artifact([[1.0, 1.0]] * 5, verdict="complete_s2_no_value")
    del d["energy_minus_accuracy_delta"]
    d["trust_gate_margin"] = 0.0
    assert _run_degen(d)


def test_degen_field_agnostic_diverse_pool_not_flagged():
    # FP-2: a diverse pool logged under offpath_structural_energy must NOT be mis-flagged
    assert not _run_degen(_s2_artifact([[10.0, 200.0]] * 12, field="offpath_structural_energy"))


def test_degen_non_s2_incidental_energy_delta_not_flagged():
    # FP-1: a non-S2 artifact with an incidental 'energy_delta' + candidate_rows is NOT recognized
    d = {
        "experiment": "some_other_unrelated_experiment",
        "honest_verdict": "complete_no_value",
        "game_results": [{"candidate_rows": [{"heldout_cell_recall": 1.0}, {"heldout_cell_recall": 1.0}]}],
    }
    # no delta/margin key, no S2 schema token -> not recognized
    flags = []
    av.check_engine_selection_candidate_diversity(d, flags)
    assert not flags


def test_real_exp4791_still_flags():
    # the original true positive must survive all the broadening
    import json as _json
    from pathlib import Path as _Path

    p = _Path(__file__).resolve().parents[2] / "results" / "experiment_4791_structural_energy_s2_offpath_trust_gate.json"
    if p.exists():
        assert _run_degen(_json.load(open(p)))


def test_degen_thin_5_game_pool_now_flags():
    # operator 2026-06-26: a thin 5-game test (even fully diverse) is under-covered -> flags.
    # This is the S2-v2 (exp4801) reclassification: 5/25 games is too narrow for a verdict.
    assert _run_degen(_s2_artifact([[0.3, 0.7]] * 5))


def test_degen_corpus_coverage_under_declaration_cannot_dodge():
    # 12 effective games but the declared corpus is 25 -> required = max(10, 15) = 15 -> still flags.
    d = _s2_artifact([[0.3, 0.7]] * 12)
    d["n_available_games"] = 25
    assert _run_degen(d)
    # and a genuinely corpus-wide test (16 of 25 effective) does NOT flag
    d2 = _s2_artifact([[0.3, 0.7]] * 16)
    d2["n_available_games"] = 25
    assert not _run_degen(d2)


def test_effective_selection_games_count():
    gr = [
        {"candidate_rows": [{"heldout_cell_recall": 1.0}, {"heldout_cell_recall": 1.0}]},  # degenerate (spread 0)
        {"candidate_rows": [{"heldout_cell_recall": 0.3}, {"heldout_cell_recall": 0.7}]},  # diverse
        {"candidate_rows": [{"heldout_cell_recall": 0.5}]},  # single candidate -> not effective
    ]
    eff, tot = av._count_effective_selection_games(gr)
    assert (eff, tot) == (1, 3)
