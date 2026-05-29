"""Tests for Exp 3313 repair-substrate root-cause autopsy.

Spec refs: REQ-REPORT-3313, SCENARIO-REPORT-3313.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from carnot.reporting import repair_substrate_root_cause_autopsy_3313 as mod


MODEL_ID = "unsloth/gemma-4-26B-A4B-it-GGUF"
QWEN_ID = "unsloth/Qwen3.6-35B-A3B-GGUF"
DENSE_ID = "unsloth/gemma-4-31B-it-GGUF"


def _write_json(root: Path, rel_path: Path, payload: Mapping[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _candidate_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(30):
        clean_reject = index in {7, 11, 23}
        rows.append(
            {
                "case_id": f"case-{index:02d}",
                "family": "arithmetic_exact_rows" if clean_reject else "symbolic_aliases",
                "exact_check_passed": True,
                "exact_checker_type": "exact_integer_string" if clean_reject else "exact_alias_string",
                "verified_success": not clean_reject,
                "false_accept": False,
                "calibrated_clean_verifier_decision": "reject" if clean_reject else "accept",
                "failure_class": "clean_verifier_rejected_exact_success" if clean_reject else "unknown_verifier_decision",
                "model_id": MODEL_ID,
            }
        )
    return rows


def _exp3302_payload() -> dict[str, Any]:
    return {
        "artifact": "experiment_3302_headline_sota_repair_panel_v11",
        "experiment_id": "exp3302",
        "task_id": "exp3302-headline-sota-repair-panel-v11",
        "schema": "carnot.headline_sota_repair_panel.v11",
        "schema_version": "carnot.headline_sota_repair_panel.v11",
        "milestone": "2026.05.305",
        "run_date": "20260529",
        "headline_repair_panel_ready": True,
        "repair_panel_ran": True,
        "headline_claim_allowed": False,
        "provenance_clean": False,
        "flagged_adversarial": True,
        "corrigendum_pending": [
            {
                "kind": "DURATION_TOO_SHORT",
                "severity": "critical",
                "detail": "duration_s=15.496424 but artifact references compute-bound markers.",
            }
        ],
        "duration_s": 15.496424,
        "inference_substrate": "live_local_sota_gguf_repair_plus_calibrated_clean_verifier",
        "panel_case_count": 30,
        "verified_success_count": 27,
        "false_accept_count": 0,
        "abstention_count": 0,
        "repair_success_rate": 0.9,
        "repair_success_ci95": [0.743789, 0.9654],
        "false_accept_rate_ci95": [0.0, 0.113513],
        "manifest_case_hashes": [f"hash-{index:02d}" for index in range(30)],
        "model_specs": {
            "runtime": "llama_cpp_local_gguf_only",
            "generation_runtime": "llama_cpp_local_generation",
            "mandated_model_ids": [QWEN_ID, DENSE_ID, MODEL_ID],
            "mandated_models": {
                MODEL_ID: {
                    "model_id": MODEL_ID,
                    "cached": True,
                    "size_bytes": 16_947_539_744,
                    "expected_quantization": "Q4_K_M",
                }
            },
        },
        "models_used": [
            {
                "model_id": MODEL_ID,
                "hf_id": MODEL_ID,
                "model_path": "/cache/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
                "legacy_small_model": False,
            }
        ],
        "missing_model_specs": [
            {"model_id": QWEN_ID, "reason": "not_cached"},
            {"model_id": DENSE_ID, "reason": "not_cached"},
        ],
        "candidate_results": _candidate_rows(),
        "source_artifacts": [
            {
                "label": "exp3301_fixed_manifest",
                "path": "results/experiment_3301_exact_repair_panel_manifest_v11.json",
                "sha256": "sha-exp3301",
            }
        ],
        "random_seed": 3302,
        "reproducibility_checksum": "sha-exp3302",
        "honest_verdict": "complete: fixture",
    }


def _exp3303_payload() -> dict[str, Any]:
    return {
        "artifact": "experiment_3303_repair_headline_evidence_audit_v1",
        "experiment_id": "exp3303",
        "task_id": "exp3303-repair-headline-evidence-audit-v1",
        "schema": "carnot.repair_headline_evidence_audit.v1",
        "schema_version": "carnot.repair_headline_evidence_audit.v1",
        "milestone": "2026.05.305",
        "run_date": "20260529",
        "repair_headline_evidence_audit_ready": True,
        "headline_claim_allowed_after_audit": False,
        "source_headline_claim_allowed": False,
        "source_provenance_clean": False,
        "substrate_consistency_passed": False,
        "no_new_model_execution": True,
        "no_new_repair_generation": True,
        "no_llm_judge_used_by_audit": True,
        "duration_s": 0.001598,
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "panel_case_count": 30,
        "exact_successes_audited": 27,
        "false_accept_count": 0,
        "llm_judge_dependency_count": 0,
        "adversarial_verify_flags": [
            {
                "kind": "DURATION_TOO_SHORT",
                "severity": "critical",
                "detail": "duration_s=15.496424 but artifact references compute-bound markers.",
            }
        ],
        "exact_check_provenance": {
            "all_claimed_successes_exact_checked": True,
            "reported_success_count_matches_rows": True,
            "reported_verified_success_count": 27,
            "claimed_success_count": 27,
            "candidate_result_count": 30,
            "llm_judge_dependency_count": 0,
            "exact_checker_types_for_successes": ["exact_alias_string", "exact_integer_string"],
        },
        "manifest_consistency": {
            "hashes_match_exp3301": True,
            "panel_manifest_hash_count": 30,
            "exp3301_manifest_hash_count": 30,
        },
        "model_invocation_summary": {
            "actual_model_declarations_present": True,
            "legacy_small_model_used": False,
            "used_model_count": 1,
            "used_model_ids": [MODEL_ID],
            "missing_model_ids": [QWEN_ID, DENSE_ID],
            "mandated_model_ids": [QWEN_ID, MODEL_ID, DENSE_ID],
        },
        "source_artifacts": {
            "exp3302": {
                "path": mod.EXP3302_REL_PATH.as_posix(),
                "present": True,
                "readable": True,
                "sha256": "sha-filled-by-test",
            }
        },
        "random_seed": 3303,
        "reproducibility_checksum": "sha-exp3303",
        "honest_verdict": "complete: fixture",
    }


def _stage_sources(root: Path) -> None:
    exp3302 = _exp3302_payload()
    exp3303 = _exp3303_payload()
    _write_json(root, mod.EXP3302_REL_PATH, exp3302)
    exp3302_sha = mod.sha256_file(root / mod.EXP3302_REL_PATH)
    exp3303["source_artifacts"]["exp3302"]["sha256"] = exp3302_sha
    _write_json(root, mod.EXP3303_REL_PATH, exp3303)
    _write_json(
        root,
        mod.EXP3308_REL_PATH,
        {
            "artifact": "experiment_3308_quality_flag_root_cause_autopsy_v1",
            "experiment_id": "exp3308",
            "quality_flag_autopsy_ready": True,
            "root_cause_hypotheses": [{"id": "repair_substrate_provenance_blocker"}],
            "honest_verdict": "complete: fixture",
        },
    )
    _write_json(
        root,
        mod.EXP3309_REL_PATH,
        {
            "artifact": "experiment_3309_live_runtime_provenance_contract_v1",
            "experiment_id": "exp3309",
            "runtime_contract_ready": True,
            "minimum_live_duration_s": 60.0,
            "repair_substrate_rules": {"shared_fields": ["source_artifact_hashes"]},
            "honest_verdict": "complete: fixture",
        },
    )
    conductor = root / mod.CONDUCTOR_LOG_REL_PATH
    conductor.parent.mkdir(parents=True, exist_ok=True)
    conductor.write_text(
        "| 2026-05-29 00:36 UTC | Headline SOTA repair panel v11 | OK | 81 passed in 3.12s |\n"
        "| 2026-05-29 00:49 UTC | Repair headline evidence audit v1 | OK | 81 passed in 3.35s |\n",
        encoding="utf-8",
    )


def test_req_report_3313_spec_anchor_declares_repair_substrate_autopsy() -> None:
    """REQ-REPORT-3313: OpenSpec names the autopsy before implementation."""

    spec = (mod.REPO_ROOT / mod.SPEC_REL_PATH).read_text(encoding="utf-8")

    assert "REQ-REPORT-3313" in spec
    assert "SCENARIO-REPORT-3313" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.EXP3302_REL_PATH.as_posix() in spec
    assert mod.EXP3303_REL_PATH.as_posix() in spec
    assert mod.EXP3308_REL_PATH.as_posix() in spec
    assert mod.EXP3309_REL_PATH.as_posix() in spec
    assert "exp3314" in spec
    assert "exp3315" in spec
    assert "exp3316" in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_scenario_report_3313_writes_repair_substrate_autopsy(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3313: audit blockers become exact rerun requirements."""

    _stage_sources(tmp_path)

    output = mod.write_artifact(
        tmp_path,
        output_path=Path("results/out.json"),
        started_s=10.0,
        now_s=13.5,
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / "results/out.json"
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["repair_substrate_autopsy_ready"] is True
    assert artifact["no_new_model_execution"] is True
    assert artifact["no_new_repair_generation"] is True
    assert artifact["no_conductor_execution"] is True
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["duration_s"] == pytest.approx(3.5)
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["honest_verdict"].startswith("complete:")

    comparisons = {row["field"]: row for row in artifact["panel_audit_field_comparison"]}
    assert comparisons["panel_case_count"]["status"] == "match"
    assert comparisons["verified_success_count_vs_exact_successes_audited"]["status"] == "match"
    assert comparisons["false_accept_count"]["status"] == "match"
    assert comparisons["source_provenance_clean"]["status"] == "blocking_false_match"
    assert comparisons["inference_substrate"]["status"] == "expected_boundary_difference"
    assert comparisons["duration_s"]["status"] == "expected_audit_elapsed_time_difference"
    assert comparisons["mandated_model_ids"]["status"] == "set_match_order_differs"
    assert comparisons["used_model_ids"]["status"] == "match"
    assert comparisons["missing_model_ids"]["status"] == "match"

    source_modes = {row["id"]: row for row in artifact["source_provenance_failure_modes"]}
    assert source_modes["source_panel_provenance_dirty"]["classification"] == "evidence_hygiene_failure"
    assert source_modes["critical_duration_flag_carried_forward"]["classification"] == "evidence_hygiene_failure"
    assert source_modes["runtime_contract_fields_missing"]["missing_fields"] == [
        "runtime_provenance",
        "checker_versions",
        "duration_contract_passed",
        "runtime_provenance_clean",
    ]
    assert source_modes["incomplete_mandated_model_coverage"]["missing_model_ids"] == [QWEN_ID, DENSE_ID]

    substrate_modes = {row["id"]: row for row in artifact["substrate_consistency_failure_modes"]}
    assert substrate_modes["source_duration_below_live_floor"]["minimum_live_duration_s"] == pytest.approx(60.0)
    assert substrate_modes["source_panel_runtime_contract_absent"]["is_blocker"] is True
    assert substrate_modes["audit_aggregation_boundary_expected"]["is_blocker"] is False
    assert substrate_modes["critical_adversarial_flag_present"]["is_blocker"] is True

    classifications = {row["id"]: row for row in artifact["blocker_classification"]}
    assert classifications["false_accept_and_exact_check_status"]["classification"] == "not_true_repair_failure"
    assert classifications["verified_success_shortfall"]["classification"] == "bounded_repair_performance_limitation"
    assert classifications["headline_blocker_status"]["classification"] == "evidence_hygiene_failure"
    assert classifications["verified_success_shortfall"]["clean_verifier_rejected_exact_success_count"] == 3

    reruns = artifact["rerun_contract"]
    assert set(reruns) == {"exp3314", "exp3315", "exp3316"}
    assert "distributional_repair_audit_ready=true" in reruns["exp3314"]["acceptance_requirements"]
    assert "exact_acceptance_remains_final_authority" in reruns["exp3314"]["acceptance_requirements"]
    assert "vgb_repair_policy_ready=true" in reruns["exp3315"]["acceptance_requirements"]
    assert "exact_verifiers_not_llm_judges_are_final" in reruns["exp3315"]["acceptance_requirements"]
    assert "duration_contract_passed=true" in reruns["exp3316"]["acceptance_requirements"]
    assert "substrate_consistency_passed=true" in reruns["exp3316"]["acceptance_requirements"]
    assert "headline_claim_allowed_true_only_if_all_gates_pass_else_honestly_blocked" in reruns["exp3316"][
        "acceptance_requirements"
    ]

    analyzed = {row["experiment_id"]: row for row in artifact["analyzed_artifacts"]}
    assert analyzed["exp3302"]["ready"] is True
    assert analyzed["exp3303"]["ready"] is True
    assert analyzed["exp3308"]["ready"] is True
    assert analyzed["exp3309"]["ready"] is True
    assert analyzed["ops_conductor_log"]["readable_text"] is True
    mod.validate_artifact(artifact)


def test_req_report_3313_validate_rejects_incomplete_artifacts(tmp_path: Path) -> None:
    """REQ-REPORT-3313: malformed autopsies cannot masquerade as complete."""

    _stage_sources(tmp_path)
    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.5)

    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({})
    with pytest.raises(ValueError, match="repair_substrate_autopsy_ready"):
        mod.validate_artifact(artifact | {"repair_substrate_autopsy_ready": "true"})
    with pytest.raises(ValueError, match="analyzed_artifacts"):
        mod.validate_artifact(artifact | {"analyzed_artifacts": []})
    with pytest.raises(ValueError, match="source_provenance_failure_modes"):
        mod.validate_artifact(artifact | {"source_provenance_failure_modes": []})
    with pytest.raises(ValueError, match="substrate_consistency_failure_modes"):
        mod.validate_artifact(artifact | {"substrate_consistency_failure_modes": []})
    with pytest.raises(ValueError, match="rerun_contract"):
        mod.validate_artifact(artifact | {"rerun_contract": {}})
    with pytest.raises(ValueError, match="no_new_model_execution"):
        mod.validate_artifact(artifact | {"no_new_model_execution": False})
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(artifact | {"honest_verdict": "blocked"})


def test_req_report_3313_defensive_helpers_summarize_panel_and_audit() -> None:
    """REQ-REPORT-3313: helper outputs keep repair evidence classification stable."""

    panel = _exp3302_payload()
    audit = _exp3303_payload()
    runtime = mod.missing_runtime_contract_fields(panel)
    outcomes = mod.candidate_outcome_summary(panel)
    comparison = {row["field"]: row for row in mod.panel_audit_field_comparison(panel, audit, "sha-panel")}

    assert runtime == [
        "runtime_provenance",
        "checker_versions",
        "duration_contract_passed",
        "runtime_provenance_clean",
    ]
    assert outcomes["candidate_result_count"] == 30
    assert outcomes["exact_check_passed_count"] == 30
    assert outcomes["verified_success_count"] == 27
    assert outcomes["false_accept_count"] == 0
    assert outcomes["clean_verifier_rejected_exact_success_count"] == 3
    assert comparison["audited_artifact_sha256"]["status"] == "mismatch"
    assert mod.audit_source_exp3302_sha({"source_artifacts": [{"experiment_id": "exp3302", "sha256": "sha-list"}]}) == "sha-list"
    assert mod.audit_source_exp3302_sha({"source_artifacts": [{"label": "exp3302", "sha256": "sha-label"}]}) == "sha-label"
    assert mod.audit_source_exp3302_sha({"source_artifacts": [{"experiment_id": "other", "sha256": "sha-other"}]}) == ""
    assert mod.string_list(None) == []
    assert mod.string_list("not-a-list") == []
    assert mod.string_list(42) == []
    assert mod.numeric("not-a-number") == 0.0
