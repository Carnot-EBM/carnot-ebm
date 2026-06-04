"""Tests for Exp 3818 capstone v349.

Spec refs: REQ-REPORT-3818, SCENARIO-REPORT-3818,
SCENARIO-REPORT-3818-BLOCKED-PARITY, SCENARIO-REPORT-3818-FLAGGED.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.reporting import capstone_v349_3818 as mod


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _gate_data() -> dict[str, object]:
    return {
        "paper_ready": True,
        "gates": {
            "G1": {"pass": True},
            "G2": {"pass": True},
            "G3": {"pass": True},
            "G4": {"pass": True},
        },
        "unmet_gates": [],
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


def _payloads() -> dict[int, dict[str, object]]:
    return {
        3808: {
            "honest_verdict": "complete: archived_v348_landed_all_product_headline_demoted",
            "paper_ready_preserved": True,
            "both_energy_routes_still_bounded": True,
            "edlm_remains_operator_seed_surface": True,
            "v349_active_confirmed": True,
            "random_seed": 3808,
            "reproducibility_checksum": "8" * 64,
            "duration_s": 0.1,
        },
        3809: {
            "honest_verdict": (
                "complete: anomaly_escalation_advisory_hook_wired_recommend_only_"
                "replay_false_escalation_0.000000_frame_violating_recall_1.0_"
                "never_relaxes_verification_conductor_unmodified_integration_proposal_emitted"
            ),
            "advisory_module_added": True,
            "offline_replay_false_escalation_rate": 0.0,
            "offline_replay_frame_violating_recall": 1.0,
            "never_relaxes_verification": True,
            "conductor_unmodified": True,
            "integration_proposal_emitted": True,
            "tests_assert_real_behavior": True,
            "random_seed": 3809,
            "reproducibility_checksum": "9" * 64,
            "duration_s": 0.1,
        },
        3810: {
            "honest_verdict": (
                "complete: abstention_http_rest_surface_v2_repaired_e2e_passed_"
                "default_off_batch_post_works_doc_proposal_emitted_not_curated_edit"
            ),
            "http_rest_surface_added": True,
            "default_off_preserves_prior_behavior": True,
            "e2e_http_abstention_passed": True,
            "batch_post_works": True,
            "doc_proposal_emitted_not_curated_edit": True,
            "tests_assert_real_behavior": True,
            "scripts_research_conductor_modified": False,
            "random_seed": 3810,
            "reproducibility_checksum": "a" * 64,
            "duration_s": 0.1,
        },
        3811: {
            "honest_verdict": (
                "complete: abstention_cross_surface_parity_smoke_all_surfaces_"
                "agree_true_n10_verify_api_cli_http_rest_no_surface_drift"
            ),
            "all_surfaces_agree": True,
            "surfaces_compared": ["verify_api", "cli", "http_rest"],
            "n_candidates_compared": 10,
            "mismatches": [],
            "tests_assert_real_behavior": True,
            "scripts_research_conductor_modified": False,
            "random_seed": 3811,
            "reproducibility_checksum": "b" * 64,
            "duration_s": 0.1,
        },
        3812: {
            "honest_verdict": (
                "complete: product_headline_status_recorded_code_repair_false_"
                "crane_false_sole_defensible_fover_0.9131_stays_demoted_"
                "doc_proposal_emitted_operator_curated_doc_unedited"
            ),
            "product_headline_recommendation": "stays_demoted",
            "code_repair_supports_headline": False,
            "crane_supports_headline": False,
            "sole_defensible_headline": "FoVer methods headline 0.9131",
            "operator_restore_path": "clean GPU rerun for the operator",
            "doc_proposal_emitted_not_curated_edit": True,
            "operator_curated_doc_unedited": True,
            "publication_gate_state": _gate_data(),
            "random_seed": 3812,
            "reproducibility_checksum": "c" * 64,
            "duration_s": 0.1,
        },
        3813: {
            "honest_verdict": (
                "complete: fr11_v21_fast_path_robustness_skip_0.5950_"
                "effective_auroc_0.9594_operating_point_generalizes_false_"
                "headline_ensemble_unchanged_measurement_not_retrain"
            ),
            "continuous_self_learning_task": True,
            "skip_rate_second_split": 0.595,
            "effective_auroc_second_split": 0.9594,
            "operating_point_generalizes": False,
            "headline_ensemble_unchanged": True,
            "acceptance_gate": {"passed": False},
            "is_measurement_not_retrain": True,
            "random_seed": 3813,
            "reproducibility_checksum": "d" * 64,
            "duration_s": 0.1,
        },
        3814: {
            "honest_verdict": (
                "complete: publication_gate_regression_confirmed_g1_g2_g3_g4_pass_"
                "paper_ready_true_frozen_fover_0.9131_unchanged_no_gate_redefined"
            ),
            "g1_pass": True,
            "g2_pass": True,
            "g3_pass": True,
            "g4_pass": True,
            "paper_ready": True,
            "frozen_fover_auroc": 0.9131,
            "frozen_fover_auroc_unchanged": True,
            "gate_definitions_unchanged": True,
            "any_gate_regressed": False,
            "random_seed": 3814,
            "reproducibility_checksum": "e" * 64,
            "duration_s": 0.1,
        },
        3815: {
            "honest_verdict": (
                "complete: edlm_operator_seed_staged_one_command_seed_packaged_"
                "kill_gate_design_documented_loop_does_not_seed_operator_gated_"
                "operator_curated_doc_unedited"
            ),
            "staging_note_written": True,
            "operator_seed_command": "git clone https://example.invalid/edlm && echo ready",
            "kill_gate_design_documented": True,
            "loop_does_not_seed": True,
            "edlm_remains_operator_gated": True,
            "operator_curated_doc_unedited": True,
            "random_seed": 3815,
            "reproducibility_checksum": "f" * 64,
            "duration_s": 0.1,
        },
        3816: {
            "honest_verdict": (
                "complete: external_research_refresh_349_section_intact_"
                "references_appended_numbers_as_reported"
            ),
            "references_section_intact": True,
            "references_added": ["arXiv:2605.28920"],
            "n_references_added": 10,
            "section_appended_not_replaced": True,
            "numbers_are_as_reported": True,
            "random_seed": 3816,
            "reproducibility_checksum": "1" * 64,
            "duration_s": 0.1,
        },
        3817: {
            "honest_verdict": (
                "complete: kv260_terminal_state_holds_ssh_reachable_"
                "accelerator_loadable_opportunistic_audit"
            ),
            "terminal_state_holds": True,
            "kv260_ssh_reachable": True,
            "kv260_overlay_loadable": True,
            "speedup_claim_made": False,
            "random_seed": 3817,
            "reproducibility_checksum": "2" * 64,
            "duration_s": 0.1,
        },
    }


def _seed_upstreams(
    root: Path,
    *,
    missing: set[int] | None = None,
    blocked: set[int] | None = None,
    flagged: set[int] | None = None,
) -> None:
    missing = missing or set()
    blocked = blocked or set()
    flagged = flagged or set()
    for experiment_id, payload in _payloads().items():
        if experiment_id in missing:
            continue
        row = dict(payload)
        if experiment_id in blocked:
            row["honest_verdict"] = "blocked_http_surface_precondition_missing"
        if experiment_id in flagged:
            row["flagged_adversarial"] = True
        _write_json(root / mod.DEFAULT_UPSTREAM_PATHS[experiment_id], row)


def test_req_report_3818_spec_anchor_exists() -> None:
    """REQ-REPORT-3818: OpenSpec declares the v349 capstone contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-3818" in spec
    assert "SCENARIO-REPORT-3818" in spec
    assert "SCENARIO-REPORT-3818-BLOCKED-PARITY" in spec
    assert "SCENARIO-REPORT-3818-FLAGGED" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.TERMINAL_VERDICT in spec


def test_scenario_report_3818_current_capstone_records_v349_lean_maintenance(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3818: current upstreams produce the lean capstone."""

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
    assert artifact["honest_verdict"] == mod.TERMINAL_VERDICT
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["anomaly_advisory_hook_wired"]["wired"] is True
    assert artifact["anomaly_advisory_hook_wired"]["recommend_only"] is True
    assert artifact["anomaly_advisory_hook_wired"]["offline_replay_false_escalation_rate"] == 0.0
    assert artifact["anomaly_advisory_hook_wired"]["offline_replay_frame_violating_recall"] == 1.0
    assert artifact["anomaly_advisory_hook_wired"]["never_relaxes_verification"] is True
    assert artifact["verifier_product_repaired"]["http_rest_surface"]["status"] == "repaired"
    assert artifact["verifier_product_repaired"]["cross_surface_parity"]["status"] == "confirmed"
    assert artifact["verifier_product_repaired"]["product_repaired"] is True
    assert artifact["product_headline_status"] == "stays_demoted"
    assert artifact["product_headline_evidence"]["code_repair_supports_headline"] is False
    assert artifact["product_headline_evidence"]["crane_supports_headline"] is False
    assert artifact["product_headline_evidence"]["sole_defensible_headline"].endswith("0.9131")
    assert artifact["product_headline_evidence"]["operator_gpu_rerun_handoff"] is True
    assert artifact["fr11_v21_self_learning"]["skip_rate_second_split"] == pytest.approx(0.595)
    assert artifact["fr11_v21_self_learning"]["effective_auroc_second_split"] == pytest.approx(
        0.9594
    )
    assert artifact["fr11_v21_self_learning"]["operating_point_generalizes"] is False
    assert artifact["fr11_v21_self_learning"]["acceptance_gate_passed"] is False
    assert artifact["fr11_v21_self_learning"]["headline_ensemble_unchanged"] is True
    assert artifact["publication_gate_confirmed"]["paper_ready"] is True
    assert artifact["publication_gate_confirmed"]["g1_g4_pass"] is True
    assert artifact["publication_gate_confirmed"]["frozen_fover_auroc"] == pytest.approx(0.9131)
    assert artifact["edlm_seed_staged"]["staged"] is True
    assert artifact["edlm_seed_staged"]["loop_does_not_seed"] is True
    assert artifact["references_refreshed"] is True
    assert artifact["kv260_terminal_confirmed"] is True
    assert artifact["energy_routes_still_bounded"] is True
    assert artifact["paper_ready_preserved"] is True
    assert artifact["frozen_headline_unchanged"] is True
    assert artifact["next_thesis_remains_operator_surface"] is True
    assert artifact["operator_flag_carried_forward"].endswith("not_a_fifth_deferral")
    assert artifact["no_new_existential_claim"] is True
    assert artifact["regrinds_nothing_bounded"] is True
    assert artifact["flagged_artifacts_excluded"] == []
    assert artifact["not_landed_or_blocked_recorded_honestly"] == []
    assert {item["experiment_id"] for item in artifact["cited_upstream_artifacts"]} == set(
        mod.UPSTREAM_IDS
    )
    assert {item["experiment_id"] for item in artifact["summarized_upstream_artifacts"]} == set(
        mod.UPSTREAM_IDS
    )
    encoded = json.dumps(artifact, sort_keys=True)
    assert "model_specs" not in artifact
    assert "target_model" not in artifact
    assert "GGUF" not in encoded
    assert "CUDA" not in encoded
    assert "live-model" not in encoded
    assert "live_llm_inference" not in encoded
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)


def test_scenario_report_3818_blocked_parity_is_not_a_research_negative(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3818-BLOCKED-PARITY: blocked parity is recorded honestly."""

    _seed_upstreams(tmp_path, missing={3811})

    artifact = mod.build_artifact(
        tmp_path,
        summary_records=_summary_records(),
        adversarial_reports=_clean_reports(),
        publication_gate_data=_gate_data(),
        capstone_adversarial_verify_clean=True,
        started_s=2.0,
        now_s=2.5,
    )

    assert mod.validate_artifact(artifact) == []
    assert artifact["verifier_product_repaired"]["product_repaired"] is False
    assert artifact["verifier_product_repaired"]["cross_surface_parity"]["status"] == "not-landed"
    assert artifact["verifier_product_repaired"]["not_a_research_negative"] is True
    assert artifact["honest_verdict"].startswith(
        "complete: capstone_v349_anomaly_advisory_wired_http_rest_repaired_parity_blocked"
    )
    assert artifact["not_landed_or_blocked_recorded_honestly"] == [
        {
            "experiment_id": 3811,
            "path": str(tmp_path / mod.DEFAULT_UPSTREAM_PATHS[3811]),
            "status": "not-landed",
            "reason": "artifact_missing",
        }
    ]
    assert "research negative" not in artifact["milestone_outcome_plain"].lower()


def test_scenario_report_3818_flagged_upstreams_are_excluded(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3818-FLAGGED: flagged artifacts are excluded from citations."""

    _seed_upstreams(tmp_path, flagged={3816})

    artifact = mod.build_artifact(
        tmp_path,
        summary_records=_summary_records(),
        adversarial_reports=_clean_reports(),
        publication_gate_data=_gate_data(),
        capstone_adversarial_verify_clean=True,
        started_s=3.0,
        now_s=3.5,
    )

    assert artifact["references_refreshed"] is False
    assert artifact["flagged_artifacts_excluded"] == [
        {
            "experiment_id": 3816,
            "path": str(tmp_path / mod.DEFAULT_UPSTREAM_PATHS[3816]),
            "reason": "flagged_adversarial=true",
        }
    ]
    assert 3816 not in {item["experiment_id"] for item in artifact["cited_upstream_artifacts"]}
    assert 3816 not in artifact["headline_aggregation_experiment_ids"]


def test_run_writes_artifact_and_validation_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-3818: CLI path writes stable JSON and malformed artifacts fail."""

    _seed_upstreams(tmp_path)
    out_path = mod.run(
        tmp_path,
        summary_records=_summary_records(),
        adversarial_reports={**_clean_reports(), 3817: {"flags": [{"severity": "critical", "kind": "X"}]}},
        publication_gate_data=_gate_data(),
        started_s=4.0,
        now_s=4.5,
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))

    assert out_path == tmp_path / mod.OUTPUT_REL_PATH
    assert payload["upstream_adversarial_critical_flags"] == [
        {"experiment_id": 3817, "kind": "X", "detail": None}
    ]
    assert payload["adversarial_verify_clean"] is True
    assert payload["adversarial_verify_report"]["flags"] == []
    assert payload["reproducibility_checksum"] == mod.payload_checksum(payload)
    assert mod.numeric(True) is None

    errors = mod.validate_artifact({})
    assert any(error.startswith("missing required artifact fields:") for error in errors)
    assert "honest_verdict must be a terminal Exp 3818 verdict" in errors
    assert "inference_substrate must declare the v349 aggregation-only substrate" in errors
    assert "product_headline_status must be one of the allowed terminal values" in errors
    assert "paper_ready_preserved must be true" in errors
    assert "cited_upstream_artifacts must be a list" in errors

    bad_checksum = dict(payload)
    bad_checksum["reproducibility_checksum"] = "3" * 64
    assert "reproducibility_checksum does not match artifact content" in mod.validate_artifact(
        bad_checksum
    )

    bad_live_marker = dict(payload)
    bad_live_marker["copied_substrate"] = "live_llm_inference"
    bad_live_marker["reproducibility_checksum"] = mod.payload_checksum(bad_live_marker)
    assert "artifact must not copy live-model substrate markers" in mod.validate_artifact(
        bad_live_marker
    )

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
            started_s=5.0,
            now_s=5.5,
        )


def test_req_report_3818_defensive_branches_remain_closed(tmp_path: Path) -> None:
    """REQ-REPORT-3818: defensive helper branches reject malformed evidence."""

    _seed_upstreams(tmp_path, blocked={3810})

    artifact = mod.build_artifact(
        tmp_path,
        summary_records=_summary_records(),
        adversarial_reports=_clean_reports(),
        publication_gate_data=_gate_data(),
        capstone_adversarial_verify_clean=True,
        started_s=6.0,
        now_s=6.5,
    )

    assert artifact["verifier_product_repaired"]["http_rest_surface"]["status"] == "blocked"
    assert artifact["not_landed_or_blocked_recorded_honestly"] == [
        {
            "experiment_id": 3810,
            "path": str(tmp_path / mod.DEFAULT_UPSTREAM_PATHS[3810]),
            "status": "blocked",
            "reason": "blocked_http_surface_precondition_missing",
        }
    ]
    assert mod.status_for({3810: {}}, {3810}, 3810) == "flagged"
    assert mod.product_headline_status({}) == "stays_demoted"

    citation_errors: list[str] = []
    mod.validate_citations([42, {"sha256": "bad"}], citation_errors)
    assert "each citation must be an object" in citation_errors
    assert "each citation must include fields_imported" in citation_errors
    assert "each citation must include a sha256 hex string" in citation_errors

    bad_gate = _gate_data()
    bad_gate["paper_ready"] = False
    bad_gate["gates"]["G4"]["pass"] = False  # type: ignore[index]
    with pytest.raises(ValueError, match="paper_ready_preserved must be true"):
        mod.build_artifact(
            tmp_path,
            summary_records=_summary_records(),
            adversarial_reports=_clean_reports(),
            publication_gate_data=bad_gate,
            capstone_adversarial_verify_clean=True,
            started_s=7.0,
            now_s=7.5,
        )
