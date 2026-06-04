"""Tests for Exp 3807 capstone v348.

Spec refs: REQ-REPORT-3807, SCENARIO-REPORT-3807,
SCENARIO-REPORT-3807-RERUN-GUARD,
SCENARIO-REPORT-3807-POSITIVE-CONTROL-GUARD,
SCENARIO-REPORT-3807-MISSING-BLOCKED, SCENARIO-REPORT-3807-FLAGGED.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.reporting import capstone_v348_product_headline_3807 as mod


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
        "unmet_gates": [] if paper_ready else ["G4"],
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
    rerun_flagged: bool = True,
    baseline_pass1: float = 0.13333333333333333,
    repair_pass1: float = 0.13333333333333333,
    repair_delta_pp: float = 0.0,
    positive_control_passed: bool = True,
    exp3799_status: str = "not_yet_headline_eligible",
    http_complete: bool = False,
) -> dict[int, dict[str, object]]:
    http_verdict = (
        "complete: abstention_http_rest_surface_added_default_off_e2e_passed"
        if http_complete
        else "blocked_http_abstention_e2e_failed"
    )
    return {
        3797: {
            "experiment_id": "exp3797",
            "honest_verdict": (
                "complete: archived_v347_landed_all_p1_blocked_no_free_gpu_"
                "handed_to_operator_v348_headline_advancement_active_paper_ready_"
                "true_both_energy_routes_bounded_frozen_headline_unchanged"
            ),
            "paper_ready_preserved": True,
            "both_energy_routes_still_bounded": True,
            "paper_ready_evidence": {
                "paper_ready": True,
                "g1": True,
                "g2": True,
                "g3": True,
                "g4": True,
                "frozen_headline_unchanged": True,
                "frozen_headline_auroc": 0.9131,
            },
            "v348_active_confirmed": True,
            "v348_focus_recorded": (
                "g4_headline_restoration_product_harden_repair_classifier_tuning_"
                "tier3_fast_path_no_bounded_regrind_no_paradigm_self_seed"
            ),
            "random_seed": 3797,
            "reproducibility_checksum": "7" * 64,
            "duration_s": 0.1,
        },
        3798: {
            "honest_verdict": (
                "complete: g4_product_headline_restoration_baseline_"
                f"{baseline_pass1:.2f}_repair_{repair_pass1:.2f}_"
                f"delta_{repair_delta_pp:.1f}pp_g4_provenance_complete_headline_stays_demoted"
            ),
            "baseline_pass1": baseline_pass1,
            "repair_pass1": repair_pass1,
            "repair_delta_pp": repair_delta_pp,
            "g4_provenance_complete": True,
            "positive_control_passed": positive_control_passed,
            "inference_path": "gpu_live",
            "product_headline_restorable": "stays_demoted",
            "n": 30,
            "n_broken": 26,
            "n_repaired": 0 if repair_delta_pp <= 0 else 9,
            "random_seed": 3798,
            "reproducibility_checksum": "8" * 64,
            "duration_s": 2.0,
        },
        3799: {
            "honest_verdict": (
                "complete: product_headline_provenance_reconfirmed_rerun_g4_true_"
                "exp2090_g4_true_headline_not_yet_eligible_operator_curated_doc_unedited"
            ),
            "product_headline_restorable": exp3799_status,
            "rerun_code_repair_g4_pass": True,
            "exp2090_g4_pass": True,
            "operator_curated_doc_unedited": True,
            "provenance_table": [{"number": "fixture", "g4_pass": True}],
            "random_seed": 3799,
            "reproducibility_checksum": "9" * 64,
            "duration_s": 0.1,
        },
        3800: {
            "honest_verdict": (
                "complete: gaming_resistance_mitigation_v2_context_compaction_"
                "closed_clean_auroc_preserved_n240_not_a_moat_reopen_headline_unchanged"
            ),
            "evasion_status": "closed",
            "clean_auroc_preserved": True,
            "n_samples": 240,
            "not_a_moat_reopen": True,
            "headline_unchanged": True,
            "tests_assert_real_behavior": True,
            "random_seed": 3800,
            "reproducibility_checksum": "a" * 64,
            "duration_s": 0.1,
        },
        3801: {
            "honest_verdict": http_verdict,
            "http_rest_surface_added": http_complete,
            "default_off_preserves_prior_behavior": http_complete,
            "e2e_http_abstention_passed": http_complete,
            "batch_post_works": http_complete,
            "tests_assert_real_behavior": http_complete,
            "scripts_research_conductor_modified": False,
            "no_heavy_new_dependency": True,
            "random_seed": 3801,
            "reproducibility_checksum": "b" * 64,
            "duration_s": 0.1,
        },
        3802: {
            "honest_verdict": (
                "complete: anomaly_escalation_v2_tuned_false_escalation_"
                "0.833333_to_0.000000_frame_violating_recall_1.0_"
                "never_relaxes_verification_supports_wiring_in_true_conductor_unmodified"
            ),
            "false_escalation_rate_before": 0.833333,
            "false_escalation_rate_after": 0.0,
            "frame_violating_recall_after": 1.0,
            "supports_wiring_in": True,
            "conductor_unmodified": True,
            "never_relaxes_verification": True,
            "tests_assert_real_behavior": True,
            "n_validation_artifacts": 32,
            "random_seed": 3802,
            "reproducibility_checksum": "c" * 64,
            "duration_s": 0.1,
        },
        3803: {
            "honest_verdict": (
                "complete: fr11_v20_tier3_fast_path_gate_skip_rate_0.5600_"
                "effective_auroc_0.9227_in_frozen_ci_no_accuracy_regression_"
                "headline_ensemble_unchanged_operating_point_persisted"
            ),
            "skip_rate_at_no_regression": 0.56,
            "effective_auroc_at_operating_point": 0.92275,
            "frozen_ci95": {"low": 0.9027316334533082, "high": 0.9235355665466916},
            "headline_ensemble_unchanged": True,
            "accuracy_regression": False,
            "is_tier3_application_not_retrain": True,
            "operating_point_persisted": True,
            "acceptance_gate": {"passed": True},
            "random_seed": 3803,
            "reproducibility_checksum": "d" * 64,
            "duration_s": 0.1,
        },
        3805: {
            "honest_verdict": (
                "complete: external_research_refresh_348_filed_references_section_"
                "appended_numbers_as_reported"
            ),
            "references_added": ["arXiv:2510.08146", "arXiv:2603.23701"],
            "n_references_added": 12,
            "section_appended_not_replaced": True,
            "section_confirmed_intact": True,
            "numbers_are_as_reported": True,
            "random_seed": 3805,
            "reproducibility_checksum": "e" * 64,
            "duration_s": 0.1,
        },
        3806: {
            "experiment_id": "exp3806",
            "honest_verdict": (
                "complete: kv260_terminal_state_holds_ssh_reachable_accelerator_"
                "loadable_opportunistic_audit"
            ),
            "kv260_ssh_reachable": True,
            "kv260_overlay_loadable": True,
            "terminal_state_holds": True,
            "speedup_claim_made": False,
            "random_seed": 3806,
            "reproducibility_checksum": "f" * 64,
            "duration_s": 0.1,
        },
    } | (
        {3798: {**_payloads(rerun_flagged=False)[3798], "flagged_adversarial": True}}
        if rerun_flagged
        else {}
    )


def _seed_upstreams(
    root: Path,
    *,
    missing: set[int] | None = None,
    rerun_flagged: bool = True,
    baseline_pass1: float = 0.13333333333333333,
    repair_pass1: float = 0.13333333333333333,
    repair_delta_pp: float = 0.0,
    positive_control_passed: bool = True,
    exp3799_status: str = "not_yet_headline_eligible",
    http_complete: bool = False,
) -> None:
    missing = missing or set()
    for experiment_id, payload in _payloads(
        rerun_flagged=rerun_flagged,
        baseline_pass1=baseline_pass1,
        repair_pass1=repair_pass1,
        repair_delta_pp=repair_delta_pp,
        positive_control_passed=positive_control_passed,
        exp3799_status=exp3799_status,
        http_complete=http_complete,
    ).items():
        if experiment_id not in missing:
            _write_json(root / mod.DEFAULT_UPSTREAM_PATHS[experiment_id], payload)


def test_req_report_3807_spec_anchor_exists() -> None:
    """REQ-REPORT-3807: OpenSpec declares the v348 capstone contract."""
    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-3807" in spec
    assert "SCENARIO-REPORT-3807" in spec
    assert "SCENARIO-REPORT-3807-RERUN-GUARD" in spec
    assert "SCENARIO-REPORT-3807-POSITIVE-CONTROL-GUARD" in spec
    assert "SCENARIO-REPORT-3807-MISSING-BLOCKED" in spec
    assert "SCENARIO-REPORT-3807-FLAGGED" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec


def test_scenario_report_3807_current_capstone_records_flagged_rerun_and_blocked_http(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3807: current upstreams produce the lean capstone."""
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
        "complete: capstone_v348_product_headline_demoted_"
        "verifier_product_hardened_http_rest_blocked_"
        "anomaly_classifier_repaired_fr11_v20_tier3_fast_path_"
        "paper_ready_true_frozen_headline_unchanged_both_energy_routes_bounded"
    )
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["product_headline_restorable"] == "not_yet_eligible"
    rerun = artifact["product_headline_advanced"]["rerun"]
    assert rerun["headline_aggregation_status"] == "excluded_flagged_adversarial"
    assert rerun["baseline_pass1"] == pytest.approx(0.133333)
    assert rerun["repair_pass1"] == pytest.approx(0.133333)
    assert rerun["repair_delta_pp"] == pytest.approx(0.0)
    assert rerun["positive_control_passed"] is True
    assert rerun["delta_valid_for_headline"] is False
    assert rerun["historical_plus18pp_disproven"] is False
    assert artifact["product_headline_advanced"]["provenance_reconfirmation"][
        "product_headline_restorable"
    ] == "not_yet_eligible"
    assert artifact["verifier_product_hardened"]["context_compaction_mitigation"] == {
        "evasion_status": "closed",
        "clean_auroc_preserved": True,
        "mitigated": True,
    }
    assert artifact["verifier_product_hardened"]["http_rest_surface"]["status"] == "blocked"
    assert artifact["verifier_product_hardened"]["http_rest_surface"]["default_off"] is False
    assert artifact["verifier_product_hardened"]["hardened"] is False
    assert artifact["anomaly_classifier_repaired"]["repaired"] is True
    assert artifact["anomaly_classifier_repaired"]["false_escalation_rate_before"] == pytest.approx(
        0.833333
    )
    assert artifact["anomaly_classifier_repaired"]["false_escalation_rate_after"] == pytest.approx(
        0.0
    )
    assert artifact["anomaly_classifier_repaired"]["frame_violating_recall"] == pytest.approx(1.0)
    assert artifact["fr11_v20_tier3_fast_path"]["validated"] is True
    assert artifact["fr11_v20_tier3_fast_path"]["skip_rate_at_no_regression"] == pytest.approx(
        0.56
    )
    assert artifact["fr11_v20_tier3_fast_path"]["effective_auroc_in_frozen_ci"] is True
    assert artifact["references_refreshed"] is True
    assert artifact["kv260_terminal_confirmed"] is True
    assert artifact["energy_as_generator_still_bounded"] is True
    assert artifact["energy_as_selector_status"] == "honest-negative-bounded"
    assert artifact["energy_as_generator_status"] == "honest-negative-bounded"
    assert artifact["paper_ready_preserved"] is True
    assert artifact["publication_gate_state"]["paper_ready"] is True
    assert artifact["frozen_headline_unchanged"] is True
    assert artifact["frozen_fover_auroc"] == pytest.approx(0.9131)
    assert artifact["next_thesis_remains_operator_surface"] is True
    assert artifact["operator_flag_carried_forward"].startswith("project_converged")
    assert artifact["no_new_existential_claim"] is True
    assert artifact["regrinds_nothing_already_bounded"] is True
    assert artifact["flagged_artifacts_excluded"] == [
        {
            "experiment_id": 3798,
            "path": str(tmp_path / mod.DEFAULT_UPSTREAM_PATHS[3798]),
            "reason": "flagged_adversarial=true",
        }
    ]
    assert artifact["not_landed_or_blocked_recorded_honestly"] == [
        {
            "experiment_id": 3801,
            "path": str(tmp_path / mod.DEFAULT_UPSTREAM_PATHS[3801]),
            "status": "blocked",
            "reason": "blocked_http_abstention_e2e_failed",
        }
    ]
    assert 3798 not in {item["experiment_id"] for item in artifact["cited_upstream_artifacts"]}
    assert {item["experiment_id"] for item in artifact["summarized_upstream_artifacts"]} == set(
        mod.UPSTREAM_IDS
    )
    encoded = json.dumps(artifact, sort_keys=True)
    assert "live_llm_inference" not in encoded
    assert "model_specs" not in encoded
    assert "GGUF" not in encoded
    assert "CUDA" not in encoded
    assert "disproven" not in artifact["milestone_outcome_plain"].lower()
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)


def test_scenario_report_3807_clean_restorable_path_is_headline_eligible(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3807: clean re-run plus provenance can restore headline."""
    _seed_upstreams(
        tmp_path,
        rerun_flagged=False,
        baseline_pass1=0.66,
        repair_pass1=0.84,
        repair_delta_pp=18.0,
        exp3799_status="restorable",
        http_complete=True,
    )

    artifact = mod.build_artifact(
        tmp_path,
        summary_records=_summary_records(),
        adversarial_reports=_clean_reports(),
        publication_gate_data=_gate_data(),
        capstone_adversarial_verify_clean=True,
        started_s=2.0,
        now_s=2.5,
    )

    assert artifact["honest_verdict"] == (
        "complete: capstone_v348_product_headline_restorable_"
        "verifier_product_hardened_anomaly_classifier_repaired_"
        "fr11_v20_tier3_fast_path_paper_ready_true_"
        "frozen_headline_unchanged_both_energy_routes_bounded"
    )
    assert artifact["product_headline_restorable"] == "restorable"
    assert artifact["product_headline_advanced"]["rerun"]["headline_aggregation_status"] == "used"
    assert artifact["product_headline_advanced"]["rerun"]["delta_valid_for_headline"] is True
    assert artifact["product_headline_advanced"]["rerun"]["historical_plus18pp_disproven"] is False
    assert artifact["verifier_product_hardened"]["hardened"] is True
    assert artifact["verifier_product_hardened"]["http_rest_surface"]["status"] == "complete"
    assert artifact["flagged_artifacts_excluded"] == []
    assert artifact["not_landed_or_blocked_recorded_honestly"] == []


def test_scenario_report_3807_blocked_rerun_stays_operator_handoff(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3807-MISSING-BLOCKED: missing headline inputs stay blocked."""
    _seed_upstreams(tmp_path, missing={3798, 3799})

    artifact = mod.build_artifact(
        tmp_path,
        summary_records=_summary_records(),
        adversarial_reports=_clean_reports(),
        publication_gate_data=_gate_data(),
        capstone_adversarial_verify_clean=True,
        started_s=3.0,
        now_s=3.5,
    )

    assert artifact["product_headline_restorable"] == "blocked_rerun"
    assert artifact["product_headline_advanced"]["rerun"]["headline_aggregation_status"] == "missing"
    assert artifact["product_headline_advanced"]["operator_handoff"] is True
    assert artifact["product_headline_advanced"]["historical_plus18pp_disproven"] is False
    assert artifact["honest_verdict"].startswith(
        "complete: capstone_v348_product_headline_blocked_"
    )
    assert artifact["not_landed_or_blocked_recorded_honestly"][:2] == [
        {
            "experiment_id": 3798,
            "path": str(tmp_path / mod.DEFAULT_UPSTREAM_PATHS[3798]),
            "status": "not-landed",
            "reason": "artifact_missing",
        },
        {
            "experiment_id": 3799,
            "path": str(tmp_path / mod.DEFAULT_UPSTREAM_PATHS[3799]),
            "status": "not-landed",
            "reason": "artifact_missing",
        },
    ]


def test_scenario_report_3807_nonpositive_delta_requires_positive_control(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3807-POSITIVE-CONTROL-GUARD: null deltas need controls."""
    _seed_upstreams(
        tmp_path,
        rerun_flagged=False,
        repair_delta_pp=0.0,
        positive_control_passed=False,
    )

    with pytest.raises(ValueError, match="non-positive product-headline delta requires positive control"):
        mod.build_artifact(
            tmp_path,
            summary_records=_summary_records(),
            adversarial_reports=_clean_reports(),
            publication_gate_data=_gate_data(),
            capstone_adversarial_verify_clean=True,
            started_s=4.0,
            now_s=4.5,
        )


def test_validate_artifact_reports_schema_and_checksum_errors(tmp_path: Path) -> None:
    """REQ-REPORT-3807: malformed capstones fail closed before reporting."""
    errors = mod.validate_artifact({})

    assert any(error.startswith("missing required artifact fields:") for error in errors)
    assert "honest_verdict must be a terminal Exp 3807 verdict" in errors
    assert "inference_substrate must declare the v348 aggregation-only substrate" in errors
    assert "product_headline_restorable must be one of the allowed terminal values" in errors
    assert "energy_as_generator_still_bounded must be true" in errors
    assert "paper_ready_preserved must be true" in errors
    assert "cited_upstream_artifacts must be a list" in errors

    _seed_upstreams(tmp_path, rerun_flagged=False, http_complete=True, exp3799_status="restorable")
    valid = mod.build_artifact(
        tmp_path,
        summary_records=_summary_records(),
        adversarial_reports=_clean_reports(),
        publication_gate_data=_gate_data(),
        capstone_adversarial_verify_clean=True,
        started_s=5.0,
        now_s=5.5,
    )
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

    critical_report = dict(valid)
    critical_report["adversarial_verify_report"] = {
        "flags": [{"severity": "critical", "kind": "X", "detail": "bad"}]
    }
    critical_report["reproducibility_checksum"] = mod.payload_checksum(critical_report)
    assert "adversarial verifier must report no critical flag" in mod.validate_artifact(
        critical_report
    )


def test_run_writes_artifact_and_helper_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-3807: CLI path writes stable JSON and helper branches stay honest."""
    _seed_upstreams(tmp_path, rerun_flagged=False, http_complete=True, exp3799_status="restorable")

    out_path = mod.run(
        tmp_path,
        summary_records=_summary_records(),
        adversarial_reports={
            **_clean_reports(),
            3806: {"flags": [{"severity": "critical", "kind": "X", "detail": "bad"}]},
        },
        publication_gate_data=_gate_data(),
        started_s=6.0,
        now_s=6.5,
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))

    assert out_path == tmp_path / mod.OUTPUT_REL_PATH
    assert payload["upstream_adversarial_critical_flags"] == [
        {"experiment_id": 3806, "kind": "X", "detail": "bad"}
    ]
    assert payload["adversarial_verify_clean"] is True
    assert payload["adversarial_verify_report"]["flags"] == []
    assert payload["reproducibility_checksum"] == mod.payload_checksum(payload)
    assert mod.normalize_product_headline_status("not_yet_headline_eligible") == "not_yet_eligible"
    assert mod.normalize_product_headline_status("stays_demoted") == "not_yet_eligible"
    assert mod.normalize_product_headline_status("unknown") == "blocked_rerun"
    assert mod.terminal_headline_segment("restorable_with_caveat") == "with_caveat"
    assert mod.terminal_headline_segment("blocked_rerun") == "blocked"
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
            started_s=7.0,
            now_s=7.5,
        )
