"""Tests for Exp 3305 evidence matrix v37.

Spec refs: REQ-REPORT-3305, SCENARIO-REPORT-3305.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from carnot.reporting import evidence_matrix_v37_3305 as mod


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec/capabilities/research-reporting/spec.md"


def _write_json(root: Path, rel_path: Path, payload: Mapping[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _flag(kind: str, *, severity: str = "critical") -> dict[str, str]:
    return {
        "kind": kind,
        "severity": severity,
        "detail": f"{kind} carried forward for matrix v37",
    }


def _write_v305_sources(root: Path) -> None:
    _write_json(
        root,
        mod.EXP3293_REL_PATH,
        {
            "artifact": "experiment_3293_capstone_v304",
            "experiment_id": "exp3293",
            "capstone_v304_ready": True,
            "paper_ready": False,
            "publication_blocker_count": 10,
            "garak_gate_passed": False,
            "next_top_gap": "pass_garak_redteam_gate",
            "inference_substrate": "artifact_aggregation_only",
            "honest_verdict": "complete: capstone v304 closed without paper readiness",
        },
    )
    _write_json(
        root,
        mod.EXP3294_REL_PATH,
        {
            "artifact": "experiment_3294_archive_v304_activate_v305",
            "experiment_id": "exp3294",
            "v304_closed_v305_opened": True,
            "prior_garak_gate_passed": False,
            "blocked_reasons": [],
            "inference_substrate": "artifact_aggregation_only",
            "honest_verdict": "complete: v305 opened with garak gate as top gap",
        },
    )
    _write_json(
        root,
        mod.EXP3295_REL_PATH,
        {
            "artifact": "experiment_3295_garak_failure_mode_autopsy_v1",
            "experiment_id": "exp3295",
            "garak_failure_autopsy_ready": True,
            "prior_garak_gate_passed": False,
            "prior_attack_success_rate": 0.311111,
            "headline_claim_made": False,
            "flagged_adversarial": True,
            "corrigendum_pending": [
                _flag("DURATION_TOO_SHORT"),
                _flag("METHODOLOGY_MISSING", severity="warn"),
            ],
            "inference_substrate": "artifact_aggregation_only",
            "honest_verdict": "complete: historical Garak failure autopsy bounded",
        },
    )
    _write_json(
        root,
        mod.EXP3296_REL_PATH,
        {
            "artifact": "experiment_3296_substrate_corrigendum_kan_no_retry_v1",
            "experiment_id": "exp3296",
            "substrate_corrigendum_ready": True,
            "kan_no_retry_ledger_ready": True,
            "kan_prompt_injection_headline_retired": True,
            "headline_eligible_prior_metrics": [{"metric_id": "negative-boundary"}],
            "non_headline_prior_metrics": [{"metric_id": "kan-sidecar"}],
            "inference_substrate": "aggregation_from_upstream_artifacts",
            "honest_verdict": "complete: KAN sidecar remains retired",
        },
    )
    _write_json(
        root,
        mod.EXP3297_REL_PATH,
        {
            "artifact": "experiment_3297_prefix_closed_garak_guard_v1",
            "experiment_id": "exp3297",
            "prefix_guard_policy_ready": True,
            "live_benchmark_claim": False,
            "guard_kind": "normalized_text_prefix_and_exact_phrase_guard_not_probability_bound",
            "methodology_gap": "cached response-preview text guard only",
            "inference_substrate": "cached_response_previews_only_no_live_inference",
            "honest_verdict": "complete: cached prefix guard only",
        },
    )
    _write_json(
        root,
        mod.EXP3298_REL_PATH,
        {
            "artifact": "experiment_3298_redteam_energy_telemetry_router_v1",
            "experiment_id": "exp3298",
            "redteam_telemetry_policy_ready": True,
            "live_probe_count": 36,
            "blocked_reasons": [],
            "inference_substrate": "llama_cpp_gpu",
            "honest_verdict": "complete: live telemetry policy ready",
        },
    )
    _write_json(
        root,
        mod.EXP3299_REL_PATH,
        {
            "artifact": "experiment_3299_garak_defense_ablation_v1",
            "experiment_id": "exp3299",
            "garak_defense_ablation_ready": True,
            "selected_defense_config_ready": True,
            "blocked_reasons": [],
            "inference_substrate": "llama_cpp_gpu",
            "honest_verdict": "complete: selected defense ready",
        },
    )
    _write_json(
        root,
        mod.EXP3300_REL_PATH,
        {
            "artifact": "experiment_3300_full_garak_dataflip_gate_rerun_v3",
            "experiment_id": "exp3300",
            "garak_redteam_eval_v3_ready": True,
            "garak_gate_passed": True,
            "dataflip_gate_passed": False,
            "attack_success_rate": 0.0,
            "error_count": 0,
            "blocked_reasons": ["dataflip_gate_failed"],
            "flagged_adversarial": True,
            "corrigendum_pending": [
                _flag("TAUTOLOGY"),
                _flag("IMPLAUSIBLE_PERFECT", severity="info"),
                _flag("DURATION_TOO_SHORT"),
            ],
            "inference_substrate": "llama_cpp_gpu_openai_adapter",
            "honest_verdict": "complete: garak passed; dataflip failed",
        },
    )
    _write_json(
        root,
        mod.EXP3301_REL_PATH,
        {
            "artifact": "experiment_3301_exact_repair_panel_manifest_v11",
            "experiment_id": "exp3301",
            "repair_panel_manifest_ready": True,
            "panel_case_count": 30,
            "llm_judge_required_count": 0,
            "inference_substrate": "deterministic_exact_manifest_no_live_inference",
            "honest_verdict": "complete: exact repair manifest ready",
        },
    )
    _write_json(
        root,
        mod.EXP3302_REL_PATH,
        {
            "artifact": "experiment_3302_headline_sota_repair_panel_v11",
            "experiment_id": "exp3302",
            "headline_repair_panel_ready": True,
            "repair_panel_ran": True,
            "headline_claim_allowed": False,
            "provenance_clean": False,
            "panel_case_count": 30,
            "verified_success_count": 27,
            "false_accept_count": 0,
            "flagged_adversarial": True,
            "corrigendum_pending": [_flag("DURATION_TOO_SHORT")],
            "inference_substrate": "live_local_sota_gguf_repair_plus_calibrated_clean_verifier",
            "honest_verdict": "complete: repair evidence exists but headline claim blocked",
        },
    )
    _write_json(
        root,
        mod.EXP3303_REL_PATH,
        {
            "artifact": "experiment_3303_repair_headline_evidence_audit_v1",
            "experiment_id": "exp3303",
            "repair_headline_evidence_audit_ready": True,
            "headline_claim_allowed_after_audit": False,
            "source_headline_claim_allowed": False,
            "source_provenance_clean": False,
            "substrate_consistency_passed": False,
            "adversarial_verify_flags": [_flag("DURATION_TOO_SHORT")],
            "inference_substrate": "aggregation_from_upstream_artifacts",
            "honest_verdict": "complete: repair headline audit keeps promotion blocked",
        },
    )
    _write_json(
        root,
        mod.EXP3304_REL_PATH,
        {
            "artifact": "experiment_3304_fr11_redteam_repair_memory_replay_v2",
            "experiment_id": "exp3304",
            "fr11_redteam_repair_memory_replay_ready": True,
            "continuous_self_learning_task": True,
            "controller_memory_only": True,
            "foundation_weight_updates_performed": False,
            "consolidation_gate_passed": True,
            "retention_score": 0.982143,
            "negative_transfer_rate": 0.033333,
            "blocked_reason": "",
            "inference_substrate": "artifact_only_controller_memory_replay",
            "honest_verdict": "complete: controller-memory replay safe",
        },
    )


def _row_by_id(artifact: Mapping[str, Any], experiment_id: str) -> Mapping[str, Any]:
    return next(row for row in artifact["rows"] if row["experiment_id"] == experiment_id)


def test_req_report_3305_spec_anchor_declares_matrix_schema() -> None:
    """REQ-REPORT-3305: OpenSpec declares matrix v37 before implementation."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-3305" in spec
    assert "SCENARIO-REPORT-3305" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_scenario_report_3305_builds_v37_claim_eligibility_matrix(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3305: v37 preserves .305 gates, flags, and claim boundaries."""

    _write_v305_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=10.0, now_s=12.5)
    rows_by_id = {row["experiment_id"]: row for row in artifact["rows"]}

    assert mod.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert artifact["experiment_id"] == "exp3305"
    assert artifact["task_id"] == "exp3305-evidence-matrix-v37"
    assert artifact["run_date"] == "20260529"
    assert artifact["milestone"] == "2026.05.305"
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["matrix_v37_ready"] is True
    assert artifact["artifact_count_scanned"] == 12
    assert artifact["artifacts_missing"] == []
    assert artifact["clean_evidence_count"] == 2
    assert artifact["blocked_evidence_count"] == 3
    assert artifact["flagged_evidence_count"] == 4
    assert artifact["sidecar_only_evidence_count"] == 2
    assert artifact["garak_gate_passed"] is True
    assert artifact["repair_headline_claim_allowed"] is False
    assert artifact["fr11_replay_safe"] is True
    assert artifact["paper_ready"] is False
    assert artifact["paper_blocker_count"] == 8
    assert artifact["top_gap"] == "clear_garak_dataflip_and_quality_flags"
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["honest_verdict"].startswith("complete:")

    assert rows_by_id["exp3298"]["evidence_class"] == "clean-live"
    assert rows_by_id["exp3299"]["evidence_class"] == "clean-live"
    assert rows_by_id["exp3295"]["evidence_kind"] == "historical-corrigendum"
    assert rows_by_id["exp3296"]["evidence_class"] == "sidecar-only"
    assert rows_by_id["exp3297"]["evidence_class"] == "gated-skipped"
    assert rows_by_id["exp3301"]["evidence_class"] == "gated-skipped"
    assert rows_by_id["exp3302"]["evidence_kind"] == "headline-repair"
    assert rows_by_id["exp3302"]["evidence_class"] == "blocked"
    assert rows_by_id["exp3300"]["blocker_reasons"] == [
        "dataflip_gate_failed",
        "dataflip_gate_passed=false",
    ]
    assert rows_by_id["exp3300"]["quality_flags"] == [
        {"kind": "TAUTOLOGY", "severity": "critical", "detail": "TAUTOLOGY carried forward for matrix v37"},
        {
            "kind": "IMPLAUSIBLE_PERFECT",
            "severity": "info",
            "detail": "IMPLAUSIBLE_PERFECT carried forward for matrix v37",
        },
        {
            "kind": "DURATION_TOO_SHORT",
            "severity": "critical",
            "detail": "DURATION_TOO_SHORT carried forward for matrix v37",
        },
    ]
    assert artifact["historical_flagged_evidence_bounded"] is True
    assert set(artifact["cited_upstream_artifacts"]) == {
        spec.path.as_posix() for spec in mod.EXPECTED_SOURCES
    }
    mod.validate_artifact(artifact)


def test_req_report_3305_missing_and_gated_sources_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-3305: missing evidence blocks paper readiness, not matrix rows."""

    _write_v305_sources(tmp_path)
    (tmp_path / mod.EXP3303_REL_PATH).unlink()
    _write_json(
        tmp_path,
        mod.EXP3302_REL_PATH,
        {
            "artifact": "experiment_3302_headline_sota_repair_panel_v11",
            "experiment_id": "exp3302",
            "headline_repair_panel_ready": False,
            "repair_panel_ran": False,
            "blocked_reasons": ["exp3300_garak_gate_missing"],
            "inference_substrate": "live_local_sota_gguf_repair_plus_calibrated_clean_verifier",
            "honest_verdict": "complete: repair panel gated and skipped",
        },
    )

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.0)

    assert artifact["matrix_v37_ready"] is True
    assert artifact["artifact_count_scanned"] == 11
    assert artifact["artifacts_missing"] == [mod.EXP3303_REL_PATH.as_posix()]
    assert artifact["paper_ready"] is False
    assert artifact["top_gap"] == "restore_missing_v305_artifacts"
    assert _row_by_id(artifact, "exp3302")["present"] is True
    assert _row_by_id(artifact, "exp3302")["readable_json_object"] is True
    assert _row_by_id(artifact, "exp3302")["evidence_class"] == "blocked"
    assert "headline_repair_panel_ready=false" in _row_by_id(artifact, "exp3302")[
        "blocker_reasons"
    ]
    assert _row_by_id(artifact, "exp3303")["evidence_class"] == "missing"


def test_req_report_3305_writer_helpers_and_validation_edges(tmp_path: Path) -> None:
    """REQ-REPORT-3305: helper and validation paths preserve conservative behavior."""

    _write_v305_sources(tmp_path)
    output = mod.write_artifact(tmp_path, started_s=2.0, now_s=5.25)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["duration_s"] == pytest.approx(3.25)
    assert len(saved["reproducibility_checksum"]) == 64

    malformed = tmp_path / "bad.json"
    malformed.write_text("{bad", encoding="utf-8")
    assert mod.read_json_object(malformed) == {}
    non_object = tmp_path / "list.json"
    non_object.write_text("[]", encoding="utf-8")
    assert mod.read_json_object(non_object) == {}
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    assert mod.sha256_file(tmp_path / "missing.json") is None
    assert mod._duration(5.0, 4.0) == 0.0
    assert mod._as_mapping({"a": 1}) == {"a": 1}
    assert mod._as_mapping([]) == {}
    assert mod._as_list(["a"]) == ["a"]
    assert mod._as_list("a") == []
    assert mod._list_of_strings(["a", 2, None]) == ["a", "2", "None"]
    assert mod._bool_is_false({"x": False}, "x") is True
    assert mod._bool_is_false({"x": 0}, "x") is False
    assert mod._quality_flags({"flagged_adversarial": True}) == [
        {"kind": "flagged_adversarial", "severity": "unknown", "detail": "flagged_adversarial=true"}
    ]
    assert mod._is_live_substrate("llama_cpp_gpu_openai_adapter") is True
    assert mod._is_live_substrate("artifact_aggregation_only") is False
    assert mod._explicit_blockers(
        {
            "blocked_reasons": ["blocked_a"],
            "gate_reasons": ["blocked_b"],
            "blocked_reason": "blocked_c",
            "gate_check_summary": "blocked_d",
            "runner_error": "blocked_e",
        }
    ) == ["blocked_a", "blocked_b", "blocked_c", "blocked_d", "blocked_e"]
    assert (
        mod._evidence_class(
            {"present": True, "readable_json_object": True, "evidence_kind": "headline-repair"},
            {"headline_claim_allowed": True},
            [],
        )
        == "headline-repair"
    )
    assert (
        mod._evidence_class(
            {"present": True, "readable_json_object": True, "evidence_kind": "novel-kind"},
            {},
            [],
        )
        == "novel-kind"
    )
    assert (
        mod._evidence_class(
            {"present": True, "readable_json_object": True, "evidence_kind": ""},
            {},
            [],
        )
        == "gated-skipped"
    )
    assert mod._blocker_reasons({"present": True, "readable_json_object": False, "path": "bad"}, {}) == [
        "artifact_unreadable_or_not_json_object: bad"
    ]
    assert mod._blocker_reasons(
        {"present": True, "readable_json_object": True, "experiment_id": "exp3300"},
        {"garak_gate_passed": False, "dataflip_gate_passed": True, "error_count": 1},
    ) == ["garak_gate_passed=false", "error_count>0"]
    assert mod._blocker_reasons(
        {"present": True, "readable_json_object": True, "experiment_id": "exp3304"},
        {"fr11_redteam_repair_memory_replay_ready": True},
    ) == ["fr11_replay_safe=false"]
    assert mod._top_gap(
        [],
        [],
        garak_gate_passed=False,
        repair_claim_allowed=True,
        repair_audit_required=True,
        fr11_safe=True,
        historical_bounded=True,
    ) == "pass_garak_redteam_gate"
    assert mod._top_gap(
        [],
        [],
        garak_gate_passed=True,
        repair_claim_allowed=False,
        repair_audit_required=True,
        fr11_safe=True,
        historical_bounded=True,
    ) == "clear_repair_headline_evidence_audit"
    assert mod._top_gap(
        [],
        [],
        garak_gate_passed=True,
        repair_claim_allowed=True,
        repair_audit_required=True,
        fr11_safe=False,
        historical_bounded=True,
    ) == "repair_fr11_controller_memory_replay_safety"
    assert mod._top_gap(
        [],
        [],
        garak_gate_passed=True,
        repair_claim_allowed=True,
        repair_audit_required=True,
        fr11_safe=True,
        historical_bounded=False,
    ) == "bound_historical_flagged_evidence"
    assert mod._top_gap(
        [],
        [],
        garak_gate_passed=True,
        repair_claim_allowed=True,
        repair_audit_required=True,
        fr11_safe=True,
        historical_bounded=True,
    ) == "ready_for_v305_capstone"

    rows = [
        {
            "experiment_id": "exp3300",
            "path": "garak.json",
            "evidence_kind": "clean-live",
            "evidence_class": "blocked",
            "quality_flags": [],
            "claim_boundaries": [],
        },
        {
            "experiment_id": "exp3295",
            "path": "historical.json",
            "evidence_kind": "historical-corrigendum",
            "evidence_class": "historical-corrigendum",
            "quality_flags": [_flag("DURATION_TOO_SHORT")],
            "claim_boundaries": [],
        },
        {
            "experiment_id": "exp3304",
            "path": "fr11.json",
            "evidence_kind": "sidecar-only",
            "evidence_class": "sidecar-only",
            "quality_flags": [],
            "claim_boundaries": [],
        },
    ]
    blockers = mod._paper_blocker_records(
        rows,
        {
            "exp3300": {"garak_gate_passed": False, "dataflip_gate_passed": True},
            "exp3304": {"fr11_redteam_repair_memory_replay_ready": True},
        },
    )
    assert {record["reason"] for record in blockers} == {
        "garak_gate_not_passed",
        "fr11_replay_not_safe",
        "historical_flagged_evidence_unbounded",
    }

    artifact = mod.build_artifact(tmp_path, started_s=0.0, now_s=0.0)
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({})
    with pytest.raises(ValueError, match="experiment_id"):
        mod.validate_artifact(artifact | {"experiment_id": "bad"})
    with pytest.raises(ValueError, match="task_id"):
        mod.validate_artifact(artifact | {"task_id": "bad"})
    with pytest.raises(ValueError, match="milestone"):
        mod.validate_artifact(artifact | {"milestone": "bad"})
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(artifact | {"inference_substrate": "bad"})
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(artifact | {"honest_verdict": "blocked"})
    with pytest.raises(ValueError, match="paper_blocker_count"):
        mod.validate_artifact(artifact | {"paper_blocker_count": -1})
    with pytest.raises(ValueError, match="paper_ready cannot be true"):
        mod.validate_artifact(artifact | {"paper_ready": True, "paper_blocker_count": 1})
