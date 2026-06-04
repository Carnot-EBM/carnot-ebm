"""Tests for Exp 3796 capstone v347.

Spec refs: REQ-REPORT-3796, SCENARIO-REPORT-3796,
SCENARIO-REPORT-3796-P1-GUARD, SCENARIO-REPORT-3796-FUNDAMENTAL-GUARD,
SCENARIO-REPORT-3796-MISSING-BLOCKED, SCENARIO-REPORT-3796-FLAGGED.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.reporting import capstone_v347_post_convergence_3796 as mod


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _gate_data(*, paper_ready: bool = True) -> dict[str, object]:
    return {
        "paper_ready": paper_ready,
        "gates": {
            "G1": {"pass": paper_ready},
            "G2": {"pass": paper_ready},
            "G3": {"pass": paper_ready},
            "G4": {"pass": paper_ready},
        },
        "unmet_gates": [] if paper_ready else ["G3"],
    }


def _clean_reports() -> dict[int, dict[str, object]]:
    return {experiment_id: {"flags": []} for experiment_id in mod.UPSTREAM_IDS}


def _summary_records() -> list[dict[str, object]]:
    return [
        {
            "experiment_id": experiment_id,
            "returncode": 0,
            "stdout_sha256": f"{experiment_id:064x}"[-64:],
            "stderr_sha256": "0" * 64,
        }
        for experiment_id in mod.UPSTREAM_IDS
    ]


def _payloads(
    *,
    p1_adjudication: str = "blocked_no_free_gpu",
    p1_positive_control_passed: bool = False,
    p1_blocked: bool = True,
    product_headline_restorable: str = "not_yet_headline_eligible",
) -> dict[int, dict[str, object]]:
    p1_verdict = (
        "blocked_no_free_gpu"
        if p1_blocked
        else f"complete: thesis_a_p1_v3_{p1_adjudication}_positive_control_passed"
    )
    return {
        3786: {
            "honest_verdict": (
                "complete: archived_v346_landed_10_of_11_exp3777_blocked_no_free_gpu_"
                "v347_post_convergence_active_paper_ready_true_both_energy_routes_"
                "bounded_frozen_headline_unchanged"
            ),
            "both_energy_routes_still_bounded": True,
            "paper_ready_preserved": True,
            "paper_ready_evidence": {
                "paper_ready": True,
                "g1": True,
                "g2": True,
                "g3": True,
                "g4": True,
                "frozen_headline_unchanged": True,
                "frozen_headline_auroc": 0.9131,
            },
            "v347_focus_recorded": (
                "retry_p1_v3_harden_banked_product_tier3_self_learning_validate_"
                "anomaly_edlm_preflight_regrind_nothing_bounded"
            ),
            "random_seed": 3786,
            "reproducibility_checksum": "6" * 64,
            "duration_s": 0.1,
        },
        3787: {
            "honest_verdict": p1_verdict,
            "adjudication": (
                "inconclusive_positive_control_failed" if p1_blocked else p1_adjudication
            ),
            "positive_control_passed": p1_positive_control_passed,
            "energy_as_generator_still_bounded": True,
            "handoff_to_operator": p1_blocked,
            "ar_best": 0.0 if not p1_positive_control_passed else 0.42,
            "ebt_best": 0.0,
            "random_seed": 30603,
            "reproducibility_checksum": "7" * 64,
            "duration_s": 0.1,
        },
        3788: {
            "honest_verdict": (
                "complete: fr11_v19_tier3_predictive_verifier_trained_predictive_"
                "auroc_0.9715_headline_ensemble_unchanged_memory_contribution_"
                "preserved_state_persisted"
            ),
            "continuous_self_learning_task": True,
            "is_tier3_not_tier1_or_tier2": True,
            "predictive_auroc": 0.9715,
            "headline_ensemble_unchanged": True,
            "frozen_headline_ensemble_auroc": 0.9131336,
            "memory_contribution_preserved": True,
            "tracker_state_persisted": True,
            "acceptance_gate": {"passed": True},
            "random_seed": 3788,
            "reproducibility_checksum": "8" * 64,
            "duration_s": 0.1,
        },
        3789: {
            "honest_verdict": (
                "complete: abstention_cli_batch_surface_added_default_off_e2e_"
                "passed_doc_proposal_emitted_not_curated_edit"
            ),
            "cli_abstention_surface_added": True,
            "batch_path_works": True,
            "default_off_preserves_prior_behavior": True,
            "e2e_cli_abstention_passed": True,
            "tests_assert_real_behavior": True,
            "scripts_research_conductor_modified": False,
            "random_seed": 3789,
            "reproducibility_checksum": "9" * 64,
            "duration_s": 0.1,
        },
        3790: {
            "honest_verdict": (
                "complete: verifier_gaming_resistance_characterized_degradation_"
                "curve_n240_holds_and_degrades_documented_not_a_moat_reopen_"
                "headline_unchanged"
            ),
            "gaming_degradation_curve": {"clean": {"auroc": 1.0}},
            "headline_unchanged": True,
            "n_samples": 240,
            "not_a_moat_reopen": True,
            "perturbations_tested": ["arithmetic_result_plus_one", "context_compaction"],
            "verifier_degrades_where": ["context_compaction"],
            "random_seed": 3790,
            "reproducibility_checksum": "a" * 64,
            "duration_s": 0.1,
        },
        3791: {
            "honest_verdict": (
                "complete: anomaly_escalation_validated_n32_false_escalation_rate_"
                "0.833333_frame_violating_recall_1.000000_never_relaxes_"
                "verification_conductor_unmodified"
            ),
            "false_escalation_rate": 0.833333,
            "frame_violating_recall": 1.0,
            "never_relaxes_verification": True,
            "conductor_unmodified": True,
            "tests_assert_real_behavior": True,
            "supports_wiring_in": False,
            "random_seed": 3791,
            "reproducibility_checksum": "b" * 64,
            "duration_s": 0.1,
        },
        3792: {
            "honest_verdict": (
                "complete: product_headline_provenance_confirmed_exp1999_g4_false_"
                "exp2090_g4_true_headline_not_yet_eligible_operator_curated_doc_"
                "unedited"
            ),
            "product_headline_restorable": product_headline_restorable,
            "exp1999_g4_pass": False,
            "exp2090_g4_pass": True,
            "operator_curated_doc_unedited": True,
            "provenance_table": [{"number": "fixture", "g4_pass": True}],
            "random_seed": {"exp1999": None, "exp2090": 42},
            "reproducibility_checksum": {"exp1999": None, "exp2090": "bfb0acdb53773a49"},
            "duration_s": 0.1,
        },
        3793: {
            "honest_verdict": (
                "complete: edlm_no_train_preflight_go_reference_impl_fetchable_true_"
                "minimal_kill_gate_sound_operator_seed_command_emitted_loop_does_not_"
                "commit"
            ),
            "readiness_verdict": "go",
            "reference_impl_fetchable": True,
            "minimal_kill_gate_sound": True,
            "operator_seed_command": (
                "git clone https://github.com/MinkaiXu/Energy-Diffusion-LLM.git && "
                "cd Energy-Diffusion-LLM && git checkout main && echo 'Seed ready'"
            ),
            "loop_does_not_commit": True,
            "random_seed": 3793,
            "reproducibility_checksum": "c" * 64,
            "duration_s": 0.1,
        },
        3794: {
            "honest_verdict": (
                "complete: external_research_refresh_347_filed_references_section_"
                "appended_numbers_as_reported"
            ),
            "references_added": ["arXiv:2605.04291", "arXiv:2601.21484"],
            "n_references_added": 6,
            "section_appended_not_replaced": True,
            "numbers_are_as_reported": True,
            "random_seed": 3794,
            "reproducibility_checksum": "d" * 64,
            "duration_s": 0.1,
        },
        3795: {
            "honest_verdict": (
                "complete: kv260_terminal_state_holds_ssh_reachable_accelerator_"
                "loadable_opportunistic_audit"
            ),
            "terminal_state_holds": True,
            "kv260_ssh_reachable": True,
            "kv260_overlay_loadable": True,
            "speedup_claim_made": False,
            "random_seed": 3795,
            "reproducibility_checksum": "e" * 64,
            "duration_s": 0.1,
        },
    }


def _seed_upstreams(
    root: Path,
    *,
    missing: set[int] | None = None,
    flagged: set[int] | None = None,
    p1_adjudication: str = "blocked_no_free_gpu",
    p1_positive_control_passed: bool = False,
    p1_blocked: bool = True,
    product_headline_restorable: str = "not_yet_headline_eligible",
) -> None:
    missing = missing or set()
    flagged = flagged or set()
    for experiment_id, payload in _payloads(
        p1_adjudication=p1_adjudication,
        p1_positive_control_passed=p1_positive_control_passed,
        p1_blocked=p1_blocked,
        product_headline_restorable=product_headline_restorable,
    ).items():
        if experiment_id in missing:
            continue
        if experiment_id in flagged:
            payload["flagged_adversarial"] = True
        _write_json(root / mod.DEFAULT_UPSTREAM_PATHS[experiment_id], payload)


def test_req_report_3796_spec_anchor_exists() -> None:
    """REQ-REPORT-3796: OpenSpec declares the v347 capstone contract."""
    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-3796" in spec
    assert "SCENARIO-REPORT-3796" in spec
    assert "SCENARIO-REPORT-3796-P1-GUARD" in spec
    assert "SCENARIO-REPORT-3796-FUNDAMENTAL-GUARD" in spec
    assert "SCENARIO-REPORT-3796-MISSING-BLOCKED" in spec
    assert "SCENARIO-REPORT-3796-FLAGGED" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec


def test_scenario_report_3796_current_capstone_records_post_convergence(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3796: current upstreams produce the lean capstone."""
    _seed_upstreams(tmp_path)

    artifact = mod.build_artifact(
        tmp_path,
        summary_records=_summary_records(),
        adversarial_reports=_clean_reports(),
        publication_gate_data=_gate_data(),
        capstone_adversarial_verify_clean=True,
        started_s=1.0,
        now_s=1.5,
    )

    assert mod.validate_artifact(artifact) == []
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["honest_verdict"] == (
        "complete: capstone_v347_p1_blocked_no_free_gpu_"
        "energy_as_generator_still_bounded_verifier_product_hardened_"
        "fr11_v19_tier3_anomaly_validated_edlm_preflighted_"
        "paper_ready_true_frozen_headline_unchanged"
    )
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["p1_adjudication"] == "blocked_no_free_gpu"
    assert artifact["p1_positive_control_passed"] is False
    assert artifact["p1_mechanism_status"] == "open_for_operator: blocked_no_free_gpu"
    assert artifact["energy_as_generator_still_bounded"] is True
    assert artifact["energy_as_selector_status"] == "honest-negative-bounded"
    assert artifact["energy_as_generator_status"] == "honest-negative-bounded"
    assert artifact["verifier_product_hardened"]["hardened"] is True
    assert artifact["verifier_product_hardened"]["abstention_cli_batch_surface"] is True
    assert artifact["verifier_product_hardened"]["gaming_resistance_curve"] is True
    assert artifact["verifier_product_hardened"]["product_headline_provenance_confirmed"] is True
    assert artifact["product_headline_restorable"] == "not_yet_eligible"
    assert artifact["fr11_v19_tier3_self_learning"]["validated"] is True
    assert artifact["fr11_v19_tier3_self_learning"]["predictive_auroc"] == pytest.approx(
        0.9715
    )
    assert artifact["fr11_v19_tier3_self_learning"]["headline_ensemble_unchanged"] is True
    assert artifact["fr11_v19_tier3_self_learning"]["memory_contribution_preserved"] is True
    assert artifact["anomaly_escalation_validated"] is True
    assert artifact["anomaly_escalation_validation"]["false_escalation_rate"] == pytest.approx(
        0.833333
    )
    assert artifact["anomaly_escalation_validation"]["frame_violating_recall"] == pytest.approx(
        1.0
    )
    assert artifact["anomaly_escalation_validation"]["never_relaxes_verification"] is True
    assert artifact["anomaly_escalation_validation"]["conductor_unmodified"] is True
    assert artifact["edlm_seed_preflighted"]["readiness_verdict"] == "go"
    assert "git clone" in artifact["edlm_seed_preflighted"]["operator_seed_command"]
    assert artifact["edlm_seed_preflighted"]["loop_does_not_commit"] is True
    assert artifact["paper_ready_preserved"] is True
    assert artifact["publication_gate_state"]["paper_ready"] is True
    assert artifact["frozen_headline_unchanged"] is True
    assert artifact["frozen_fover_auroc"] == pytest.approx(0.9131)
    assert artifact["next_thesis_remains_operator_surface"] is True
    assert artifact["no_new_existential_claim"] is True
    assert artifact["regrinds_nothing_already_bounded"] is True
    assert artifact["references_refreshed"] is True
    assert artifact["kv260_terminal_confirmed"] is True
    assert artifact["flagged_artifacts_excluded"] == []
    assert artifact["not_landed_or_blocked_recorded_honestly"] == [
        {
            "experiment_id": 3787,
            "path": str(tmp_path / mod.DEFAULT_UPSTREAM_PATHS[3787]),
            "status": "blocked",
            "reason": "blocked_no_free_gpu",
        }
    ]
    assert {item["experiment_id"] for item in artifact["cited_upstream_artifacts"]} == set(
        mod.UPSTREAM_IDS
    )
    assert {item["experiment_id"] for item in artifact["summarized_upstream_artifacts"]} == set(
        mod.UPSTREAM_IDS
    )
    assert "live_llm_inference" not in json.dumps(artifact, sort_keys=True)
    assert "model_specs" not in json.dumps(artifact, sort_keys=True)
    assert "newly disproven" not in artifact["milestone_outcome_plain"]
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)


def test_scenario_report_3796_missing_p1_is_open_not_negative(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3796-MISSING-BLOCKED: missing Exp 3787 stays open."""
    _seed_upstreams(tmp_path, missing={3787})

    artifact = mod.build_artifact(
        tmp_path,
        summary_records=_summary_records(),
        adversarial_reports=_clean_reports(),
        publication_gate_data=_gate_data(),
        capstone_adversarial_verify_clean=True,
        started_s=2.0,
        now_s=2.25,
    )

    assert artifact["p1_adjudication"] == "blocked_missing_upstream_artifact"
    assert artifact["p1_positive_control_passed"] is False
    assert artifact["p1_mechanism_status"] == "open_for_operator: blocked_missing_upstream_artifact"
    assert artifact["energy_as_generator_still_bounded"] is True
    assert artifact["not_landed_or_blocked_recorded_honestly"] == [
        {
            "experiment_id": 3787,
            "path": str(tmp_path / mod.DEFAULT_UPSTREAM_PATHS[3787]),
            "status": "not-landed",
            "reason": "artifact_missing",
        }
    ]
    assert "newly failed" not in artifact["milestone_outcome_plain"]


def test_scenario_report_3796_flagged_upstream_is_excluded(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3796-FLAGGED: flagged artifacts are quarantined."""
    _seed_upstreams(tmp_path, flagged={3790})

    artifact = mod.build_artifact(
        tmp_path,
        summary_records=_summary_records(),
        adversarial_reports=_clean_reports(),
        publication_gate_data=_gate_data(),
        capstone_adversarial_verify_clean=True,
        started_s=3.0,
        now_s=3.25,
    )

    assert artifact["verifier_product_hardened"]["hardened"] is False
    assert artifact["verifier_product_hardened"]["gaming_resistance_curve"] is False
    assert artifact["flagged_artifacts_excluded"] == [
        {
            "experiment_id": 3790,
            "path": str(tmp_path / mod.DEFAULT_UPSTREAM_PATHS[3790]),
            "reason": "flagged_adversarial=true",
        }
    ]
    assert 3790 not in {item["experiment_id"] for item in artifact["cited_upstream_artifacts"]}
    assert 3790 not in artifact["headline_aggregation_experiment_ids"]


def test_scenario_report_3796_fundamental_requires_positive_control(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3796-FUNDAMENTAL-GUARD: null FUNDAMENTAL needs control."""
    _seed_upstreams(
        tmp_path,
        p1_adjudication="fundamental_causal_inductive_bias_gap",
        p1_positive_control_passed=False,
        p1_blocked=False,
    )

    with pytest.raises(ValueError, match="fundamental P1 adjudication requires positive control"):
        mod.build_artifact(
            tmp_path,
            summary_records=_summary_records(),
            adversarial_reports=_clean_reports(),
            publication_gate_data=_gate_data(),
            capstone_adversarial_verify_clean=True,
            started_s=4.0,
            now_s=4.25,
        )


def test_validate_artifact_reports_schema_hygiene_and_checksum_errors() -> None:
    """REQ-REPORT-3796: malformed capstones fail closed before reporting."""
    errors = mod.validate_artifact({})

    assert any(error.startswith("missing required artifact fields:") for error in errors)
    assert "honest_verdict must be a terminal Exp 3796 verdict" in errors
    assert "inference_substrate must declare the v347 aggregation-only substrate" in errors
    assert "p1_adjudication must be a valid adjudication or blocked_* value" in errors
    assert "p1_positive_control_passed must be a bare bool" in errors
    assert "energy_as_generator_still_bounded must be true" in errors
    assert "paper_ready_preserved must be true" in errors
    assert "frozen_headline_unchanged must be true" in errors
    assert "cited_upstream_artifacts must be a list" in errors

    valid = {
        "honest_verdict": mod.terminal_verdict(
            p1_adjudication="decode_artifact_bounded",
            energy_bounded=True,
            verifier_hardened=True,
            fr11=True,
            anomaly=True,
            edlm=True,
            paper_ready=True,
            frozen=True,
        ),
        "inference_substrate": mod.INFERENCE_SUBSTRATE,
        "p1_adjudication": "decode_artifact_bounded",
        "p1_positive_control_passed": True,
        "energy_as_generator_still_bounded": True,
        "verifier_product_hardened": {
            "hardened": True,
            "abstention_cli_batch_surface": True,
            "gaming_resistance_curve": True,
            "product_headline_provenance_confirmed": True,
        },
        "product_headline_restorable": "restorable",
        "fr11_v19_tier3_self_learning": {"validated": True},
        "anomaly_escalation_validated": True,
        "edlm_seed_preflighted": {"preflighted": True, "loop_does_not_commit": True},
        "paper_ready_preserved": True,
        "frozen_headline_unchanged": True,
        "next_thesis_remains_operator_surface": True,
        "flagged_artifacts_excluded": [],
        "not_landed_or_blocked_recorded_honestly": [],
        "cited_upstream_artifacts": [
            {
                "experiment_id": 3786,
                "path": "results/experiment_3786.json",
                "fields_imported": ["honest_verdict"],
                "sha256": "1" * 64,
            }
        ],
        "field_principles": dict(mod.FIELD_PRINCIPLES),
        "adversarial_verify_report": {"flags": []},
        "random_seed": mod.RANDOM_SEED,
        "duration_s": 0.1,
        "reproducibility_checksum": "",
    }
    valid["reproducibility_checksum"] = mod.payload_checksum(valid)
    assert mod.validate_artifact(valid) == []

    bad_live_marker = dict(valid)
    bad_live_marker["copied_substrate"] = "live_llm_inference"
    bad_live_marker["reproducibility_checksum"] = mod.payload_checksum(bad_live_marker)
    assert "artifact must not copy live-model substrate markers" in mod.validate_artifact(
        bad_live_marker
    )

    bad_checksum = dict(valid)
    bad_checksum["reproducibility_checksum"] = "2" * 64
    assert "reproducibility_checksum does not match artifact content" in mod.validate_artifact(
        bad_checksum
    )

    bad_citation = dict(valid)
    bad_citation["cited_upstream_artifacts"] = [123, {"experiment_id": 3786}]
    bad_citation["reproducibility_checksum"] = mod.payload_checksum(bad_citation)
    citation_errors = mod.validate_artifact(bad_citation)
    assert "each citation must be an object" in citation_errors
    assert "each citation must include fields_imported" in citation_errors

    critical_report = dict(valid)
    critical_report["adversarial_verify_report"] = {
        "flags": [{"severity": "critical", "kind": "TEST", "detail": "blocked"}]
    }
    critical_report["reproducibility_checksum"] = mod.payload_checksum(critical_report)
    assert "adversarial verifier must report no critical flag" in mod.validate_artifact(
        critical_report
    )


def test_run_writes_artifact_and_helper_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-3796: CLI path writes stable JSON and helpers stay honest."""
    _seed_upstreams(tmp_path, product_headline_restorable="restorable_with_caveat")

    out_path = mod.run(
        tmp_path,
        summary_records=_summary_records(),
        adversarial_reports={
            **_clean_reports(),
            3795: {"flags": [{"severity": "critical", "kind": "X", "detail": "bad"}]},
        },
        publication_gate_data=_gate_data(),
        started_s=5.0,
        now_s=5.25,
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))

    assert out_path == tmp_path / mod.OUTPUT_REL_PATH
    assert payload["product_headline_restorable"] == "restorable_with_caveat"
    assert payload["upstream_adversarial_critical_flags"] == [
        {"experiment_id": 3795, "kind": "X", "detail": "bad"}
    ]
    assert payload["adversarial_verify_clean"] is True
    assert payload["adversarial_verify_report"]["flags"] == []
    assert payload["reproducibility_checksum"] == mod.payload_checksum(payload)
    assert mod.normalize_p1_adjudication({"adjudication": "fundamental"}, default="blocked_x") == (
        "fundamental_causal_inductive_bias_gap"
    )
    assert mod.normalize_p1_adjudication({"adjudication": "inconclusive"}, default="blocked_x") == (
        "inconclusive_positive_control_failed"
    )
    assert mod.normalize_p1_adjudication(
        {"honest_verdict": "blocked_no_free_gpu", "adjudication": "inconclusive"},
        default="blocked_x",
    ) == "blocked_no_free_gpu"
    assert mod.normalize_p1_adjudication(
        {"honest_verdict": "complete: decode_artifact_bounded"},
        default="blocked_x",
    ) == "decode_artifact_bounded"
    assert mod.normalize_product_headline_status("not_yet_headline_eligible") == "not_yet_eligible"
    assert mod.normalize_product_headline_status("unknown") == "not_yet_eligible"
    assert mod.report_is_clean(None) is True
    assert mod.numeric(True) is None

    array_path = tmp_path / "array.json"
    array_path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a JSON object"):
        mod.read_json_object(array_path)

    monkeypatch.setattr(
        mod.adversarial_verify,
        "verify_artifact",
        lambda _path: {"flags": [{"severity": "critical", "kind": "X", "detail": "bad"}]},
    )
    with pytest.raises(ValueError, match="adversarial verifier must report no critical flag"):
        mod.run(
            tmp_path,
            summary_records=_summary_records(),
            adversarial_reports=_clean_reports(),
            publication_gate_data=_gate_data(),
            started_s=6.0,
            now_s=6.25,
        )
