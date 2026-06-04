"""Tests for Exp 3785 capstone v346.

Spec refs: REQ-REPORT-3785, SCENARIO-REPORT-3785,
SCENARIO-REPORT-3785-P1-GUARD, SCENARIO-REPORT-3785-MISSING-BLOCKED,
SCENARIO-REPORT-3785-FLAGGED.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.reporting import capstone_v346_convergence_3785 as mod


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
        "unmet_gates": [] if paper_ready else ["G2"],
    }


def _clean_reports() -> dict[int, dict[str, object]]:
    return {experiment_id: {"flags": []} for experiment_id in mod.UPSTREAM_IDS}


def _summary_records() -> list[dict[str, object]]:
    return [
        {
            "experiment_id": experiment_id,
            "returncode": 0 if experiment_id != 3777 else 1,
            "stdout_sha256": f"{experiment_id:064x}"[-64:],
            "stderr_sha256": "0" * 64,
        }
        for experiment_id in mod.UPSTREAM_IDS
    ]


def _payloads(
    *,
    p1_adjudication: str = "decode_artifact_bounded",
    p1_positive_control_passed: bool = True,
    p1_blocked: bool = False,
) -> dict[int, dict[str, object]]:
    p1_verdict = (
        "blocked: thesis_a_p1_v3_no_free_gpu"
        if p1_blocked
        else f"complete: thesis_a_p1_v3_{p1_adjudication}_positive_control_passed"
    )
    return {
        3776: {
            "honest_verdict": (
                "complete: archived_v345_fully_landed_v346_convergence_active_"
                "paper_ready_true_both_energy_routes_bounded_frozen_headline_unchanged"
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
            "v346_focus_recorded": (
                "settle_p1_discrete_search_v3_bank_verifier_product_build_"
                "anomaly_escalation_scaffold_edlm_continue_self_learning_regrind_"
                "nothing_bounded"
            ),
            "random_seed": 3776,
            "reproducibility_checksum": "6" * 64,
            "duration_s": 0.1,
        },
        3777: {
            "honest_verdict": p1_verdict,
            "adjudication": p1_adjudication if not p1_blocked else "blocked_no_free_gpu",
            "positive_control_passed": p1_positive_control_passed,
            "energy_as_generator_still_bounded": True,
            "ar_best": 0.42 if p1_positive_control_passed else 0.0,
            "ebt_best": 0.02,
            "random_seed": 3777,
            "reproducibility_checksum": "7" * 64,
            "duration_s": 0.1,
        },
        3778: {
            "honest_verdict": (
                "complete: fr11_v18_tier2_constraint_memory_consolidated_"
                "auroc_within_frozen_ci_memory_contribution_preserved_state_persisted"
            ),
            "continuous_self_learning_task": True,
            "is_tier2_not_tier1": True,
            "memory_contribution_preserved": True,
            "tracker_state_persisted": True,
            "auroc_within_frozen_ci": True,
            "acceptance_gate": {"passed": True},
            "random_seed": 3778,
            "reproducibility_checksum": "8" * 64,
            "duration_s": 0.1,
        },
        3779: {
            "honest_verdict": (
                "complete: abstention_mode_wired_into_verify_api_default_off_"
                "e2e_passed_doc_proposal_emitted_not_curated_edit"
            ),
            "abstention_mode_wired": True,
            "default_off_preserves_prior_behavior": True,
            "e2e_abstention_passed": True,
            "mcp_surface_confirmed": True,
            "tests_assert_real_behavior": True,
            "random_seed": 3779,
            "reproducibility_checksum": "9" * 64,
            "duration_s": 0.1,
        },
        3780: {
            "honest_verdict": (
                "complete: anomaly_escalation_classifier_prototyped_recommend_only_"
                "change_proposal_written_never_relaxes_verification_conductor_unmodified"
            ),
            "classifier_shipped": True,
            "classifier_only_recommends": True,
            "never_relaxes_verification": True,
            "change_proposal_written": True,
            "random_seed": 3780,
            "reproducibility_checksum": "a" * 64,
            "duration_s": 0.1,
        },
        3781: {
            "honest_verdict": (
                "complete: edlm_feasibility_scoped_residual_corrector_not_blocked_"
                "by_either_negative_minimal_kill_gate_designed_operator_decision_"
                "surface_loop_does_not_commit"
            ),
            "minimal_kill_gate_design": "matched-compute PPL kill gate",
            "operator_decision_framing": "seed vs do not seed",
            "loop_does_not_commit": True,
            "random_seed": 3781,
            "reproducibility_checksum": "b" * 64,
            "duration_s": 0.1,
        },
        3782: {
            "honest_verdict": (
                "complete: g4_correction_prepped_unsupported_numbers_identified_"
                "real_numbers_confirmed_proposal_written_operator_curated_doc_unedited"
            ),
            "proposed_correction_written": True,
            "operator_curated_doc_unedited": True,
            "unsupported_numbers_identified": {"exp227": {"delta_pp": 0.0}},
            "real_numbers_confirmed": [{"experiment_id": 1999}],
            "random_seed": 3782,
            "reproducibility_checksum": "c" * 64,
            "duration_s": 0.1,
        },
        3783: {
            "honest_verdict": (
                "complete: external_research_refresh_346_filed_5_references_"
                "section_appended_numbers_as_reported"
            ),
            "references_added": [
                "arXiv:2604.07650",
                "arXiv:2506.07962",
                "arXiv:2601.17223",
                "arXiv:2604.15149",
                "arXiv:2502.11157",
            ],
            "n_references_added": 5,
            "section_appended_not_replaced": True,
            "numbers_are_as_reported": True,
            "random_seed": 3783,
            "reproducibility_checksum": "d" * 64,
            "duration_s": 0.1,
        },
        3784: {
            "honest_verdict": (
                "complete: kv260_terminal_state_holds_ssh_reachable_"
                "accelerator_loadable_opportunistic_audit"
            ),
            "terminal_state_holds": True,
            "kv260_ssh_reachable": True,
            "kv260_overlay_loadable": True,
            "speedup_claim_made": False,
            "random_seed": 3784,
            "reproducibility_checksum": "e" * 64,
            "duration_s": 0.1,
        },
    }


def _seed_upstreams(
    root: Path,
    *,
    missing: set[int] | None = None,
    flagged: set[int] | None = None,
    p1_adjudication: str = "decode_artifact_bounded",
    p1_positive_control_passed: bool = True,
    p1_blocked: bool = False,
) -> None:
    missing = missing or set()
    flagged = flagged or set()
    for experiment_id, payload in _payloads(
        p1_adjudication=p1_adjudication,
        p1_positive_control_passed=p1_positive_control_passed,
        p1_blocked=p1_blocked,
    ).items():
        if experiment_id in missing:
            continue
        if experiment_id in flagged:
            payload["flagged_adversarial"] = True
        _write_json(root / mod.DEFAULT_UPSTREAM_PATHS[experiment_id], payload)


def test_req_report_3785_spec_anchor_exists() -> None:
    """REQ-REPORT-3785: OpenSpec declares the v346 capstone contract."""
    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-3785" in spec
    assert "SCENARIO-REPORT-3785" in spec
    assert "SCENARIO-REPORT-3785-P1-GUARD" in spec
    assert "SCENARIO-REPORT-3785-MISSING-BLOCKED" in spec
    assert "SCENARIO-REPORT-3785-FLAGGED" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec


def test_scenario_report_3785_clean_capstone_records_convergence(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3785: clean upstreams produce the convergence capstone."""
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
        "complete: capstone_v346_p1_decode_artifact_bounded_"
        "energy_as_generator_still_bounded_verifier_product_banked_"
        "anomaly_escalation_prototyped_edlm_scaffolded_fr11_v18_"
        "paper_ready_true_frozen_headline_unchanged"
    )
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["p1_adjudication"] == "decode_artifact_bounded"
    assert artifact["p1_positive_control_passed"] is True
    assert artifact["p1_mechanism_status"] == "settled_or_sharpened: decode_artifact_bounded"
    assert artifact["energy_as_generator_still_bounded"] is True
    assert artifact["energy_as_selector_status"] == "honest-negative-bounded"
    assert artifact["energy_as_generator_status"] == "honest-negative-bounded"
    assert artifact["verifier_product_banked"] is True
    assert artifact["anomaly_escalation_prototyped"] is True
    assert artifact["edlm_seed_scaffolded"] is True
    assert artifact["fr11_v18_self_learning"] is True
    assert artifact["paper_ready_preserved"] is True
    assert artifact["publication_gate_state"]["paper_ready"] is True
    assert artifact["frozen_headline_unchanged"] is True
    assert artifact["frozen_fover_auroc"] == pytest.approx(0.9131)
    assert artifact["next_thesis_remains_operator_surface"] is True
    assert artifact["no_new_existential_claim"] is True
    assert artifact["regrinds_nothing_already_bounded"] is True
    assert artifact["flagged_artifacts_excluded"] == []
    assert artifact["not_landed_or_blocked_recorded_honestly"] == []
    assert {item["experiment_id"] for item in artifact["cited_upstream_artifacts"]} == set(
        mod.UPSTREAM_IDS
    )
    assert {item["experiment_id"] for item in artifact["summarized_upstream_artifacts"]} == set(
        mod.UPSTREAM_IDS
    )
    assert "live_llm_inference" not in json.dumps(artifact, sort_keys=True)
    assert "model_specs" not in json.dumps(artifact, sort_keys=True)
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)


def test_scenario_report_3785_missing_p1_is_blocked_not_negative(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3785-MISSING-BLOCKED: missing Exp 3777 stays open."""
    _seed_upstreams(tmp_path, missing={3777})

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
            "experiment_id": 3777,
            "path": str(tmp_path / mod.DEFAULT_UPSTREAM_PATHS[3777]),
            "status": "not-landed",
            "reason": "artifact_missing",
        }
    ]
    assert "newly disproven" not in artifact["milestone_outcome_plain"]


def test_scenario_report_3785_blocked_p1_is_not_fundamental(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3785-MISSING-BLOCKED: blocked P1 is not a research negative."""
    _seed_upstreams(tmp_path, p1_blocked=True, p1_positive_control_passed=False)

    artifact = mod.build_artifact(
        tmp_path,
        summary_records=_summary_records(),
        adversarial_reports=_clean_reports(),
        publication_gate_data=_gate_data(),
        capstone_adversarial_verify_clean=True,
        started_s=3.0,
        now_s=3.25,
    )

    assert artifact["p1_adjudication"] == "blocked_no_free_gpu"
    assert artifact["p1_positive_control_passed"] is False
    assert artifact["p1_mechanism_status"] == "open_for_operator: blocked_no_free_gpu"
    assert artifact["not_landed_or_blocked_recorded_honestly"] == [
        {
            "experiment_id": 3777,
            "path": str(tmp_path / mod.DEFAULT_UPSTREAM_PATHS[3777]),
            "status": "blocked",
            "reason": "blocked: thesis_a_p1_v3_no_free_gpu",
        }
    ]
    assert 3777 in {item["experiment_id"] for item in artifact["cited_upstream_artifacts"]}


def test_scenario_report_3785_flagged_upstream_is_excluded(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3785-FLAGGED: flagged artifacts are quarantined."""
    _seed_upstreams(tmp_path, flagged={3779})

    artifact = mod.build_artifact(
        tmp_path,
        summary_records=_summary_records(),
        adversarial_reports=_clean_reports(),
        publication_gate_data=_gate_data(),
        capstone_adversarial_verify_clean=True,
        started_s=4.0,
        now_s=4.25,
    )

    assert artifact["verifier_product_banked"] is False
    assert artifact["flagged_artifacts_excluded"] == [
        {
            "experiment_id": 3779,
            "path": str(tmp_path / mod.DEFAULT_UPSTREAM_PATHS[3779]),
            "reason": "flagged_adversarial=true",
        }
    ]
    assert 3779 not in {item["experiment_id"] for item in artifact["cited_upstream_artifacts"]}
    assert 3779 not in artifact["headline_aggregation_experiment_ids"]


def test_scenario_report_3785_p1_fundamental_requires_positive_control(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3785-P1-GUARD: null FUNDAMENTAL needs the AR control."""
    _seed_upstreams(
        tmp_path,
        p1_adjudication="fundamental_causal_inductive_bias_gap",
        p1_positive_control_passed=False,
    )

    with pytest.raises(ValueError, match="fundamental P1 adjudication requires positive control"):
        mod.build_artifact(
            tmp_path,
            summary_records=_summary_records(),
            adversarial_reports=_clean_reports(),
            publication_gate_data=_gate_data(),
            capstone_adversarial_verify_clean=True,
            started_s=5.0,
            now_s=5.25,
        )


def test_validate_artifact_reports_schema_hygiene_and_checksum_errors() -> None:
    """REQ-REPORT-3785: malformed capstones fail closed before reporting."""
    errors = mod.validate_artifact({})

    assert any(error.startswith("missing required artifact fields:") for error in errors)
    assert "honest_verdict must be a terminal Exp 3785 verdict" in errors
    assert "inference_substrate must declare the v346 aggregation-only substrate" in errors
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
            verifier_product=True,
            anomaly=True,
            edlm=True,
            fr11=True,
            paper_ready=True,
            frozen=True,
        ),
        "inference_substrate": mod.INFERENCE_SUBSTRATE,
        "p1_adjudication": "decode_artifact_bounded",
        "p1_positive_control_passed": True,
        "energy_as_generator_still_bounded": True,
        "verifier_product_banked": True,
        "anomaly_escalation_prototyped": True,
        "edlm_seed_scaffolded": True,
        "fr11_v18_self_learning": True,
        "paper_ready_preserved": True,
        "frozen_headline_unchanged": True,
        "next_thesis_remains_operator_surface": True,
        "flagged_artifacts_excluded": [],
        "not_landed_or_blocked_recorded_honestly": [],
        "cited_upstream_artifacts": [
            {
                "experiment_id": 3776,
                "path": "results/experiment_3776.json",
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
    bad_citation["cited_upstream_artifacts"] = [123, {"experiment_id": 3776}]
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


def test_run_writes_artifact_and_rejects_array_sources(tmp_path: Path) -> None:
    """REQ-REPORT-3785: CLI path writes stable JSON and source JSON objects only."""
    _seed_upstreams(tmp_path)

    out_path = mod.run(
        tmp_path,
        summary_records=_summary_records(),
        adversarial_reports=_clean_reports(),
        publication_gate_data=_gate_data(),
        started_s=6.0,
        now_s=6.25,
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))

    assert out_path == tmp_path / mod.OUTPUT_REL_PATH
    assert payload["adversarial_verify_clean"] is True
    assert payload["adversarial_verify_report"]["flags"] == []
    assert payload["reproducibility_checksum"] == mod.payload_checksum(payload)

    array_path = tmp_path / "array.json"
    array_path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a JSON object"):
        mod.read_json_object(array_path)


def test_helper_branches_keep_p1_and_verifier_failures_honest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-3785: helper fallbacks preserve blocked/flagged distinctions."""
    _seed_upstreams(tmp_path, flagged={3777})

    artifact = mod.build_artifact(
        tmp_path,
        summary_records=_summary_records(),
        adversarial_reports={
            **_clean_reports(),
            3779: {"flags": [{"severity": "critical", "kind": "X", "detail": "bad"}]},
        },
        publication_gate_data=_gate_data(),
        capstone_adversarial_verify_clean=True,
        started_s=7.0,
        now_s=7.25,
    )

    assert artifact["p1_adjudication"] == "blocked_flagged_adversarial"
    assert artifact["p1_positive_control_passed"] is False
    assert artifact["upstream_adversarial_critical_flags"] == [
        {"experiment_id": 3779, "kind": "X", "detail": "bad"}
    ]
    assert mod.normalize_p1_adjudication({"adjudication": "fundamental"}, default="blocked_x") == (
        "fundamental_causal_inductive_bias_gap"
    )
    assert mod.normalize_p1_adjudication({"adjudication": "inconclusive"}, default="blocked_x") == (
        "inconclusive_positive_control_failed"
    )
    assert mod.normalize_p1_adjudication(
        {"honest_verdict": "complete: decode_artifact_bounded"},
        default="blocked_x",
    ) == "decode_artifact_bounded"
    assert mod.normalize_p1_adjudication(
        {"honest_verdict": "complete: FUNDAMENTAL causal gap"},
        default="blocked_x",
    ) == "fundamental_causal_inductive_bias_gap"
    assert mod.normalize_p1_adjudication(
        {"honest_verdict": "inconclusive: positive_control_failed"},
        default="blocked_x",
    ) == "inconclusive_positive_control_failed"
    assert mod.normalize_p1_adjudication(
        {"honest_verdict": "blocked: no free GPU"},
        default="blocked_x",
    ) == "blocked_no_free_gpu"
    assert mod.normalize_p1_adjudication({}, default="blocked_x") == "blocked_x"
    assert mod.report_is_clean(None) is True
    assert mod.numeric(True) is None

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
            started_s=8.0,
            now_s=8.25,
        )
