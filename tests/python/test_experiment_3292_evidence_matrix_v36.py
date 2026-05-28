"""Tests for Exp 3292 evidence matrix v36.

Spec refs: REQ-REPORT-3292, SCENARIO-REPORT-3292.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from carnot.reporting import evidence_matrix_v36_3292 as mod


REQUIRED_FIELDS = {
    "matrix_v36_ready",
    "artifact_count_scanned",
    "clean_evidence_count",
    "blocked_evidence_count",
    "flagged_evidence_count",
    "sidecar_only_count",
    "missing_evidence_count",
    "paper_blocker_count",
    "top_gaps",
    "gate_summary",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path, payload: Mapping[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _prior_row(
    experiment_id: str,
    status: str,
    *,
    blockers: list[str] | None = None,
    flags: list[Mapping[str, Any]] | None = None,
    bounded: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "status": status,
        "blocker_reasons": blockers or [],
        "quality_flags": list(flags or []),
        "bounded_claims": bounded or [],
    }


def _write_prior_v35(root: Path) -> None:
    _write_json(
        root,
        mod.PRIOR_MATRIX_REL_PATH,
        {
            "experiment_id": "exp3279",
            "matrix_v35_ready": True,
            "paper_ready": False,
            "publication_blocker_count_estimate": 105,
            "next_gap_candidates": [
                {
                    "rank": 1,
                    "gap": "unblock_garak_redteam_eval",
                    "source_experiment_id": "exp3274",
                    "reason": "blocked_garak_unavailable",
                }
            ],
            "rows": [
                _prior_row(
                    "exp3270",
                    "flagged",
                    flags=[{"kind": "DURATION_TOO_SHORT", "detail": "duration_s=11.875108"}],
                ),
                _prior_row(
                    "exp3271",
                    "flagged",
                    flags=[{"kind": "DURATION_TOO_SHORT", "detail": "duration_s=11.75169"}],
                ),
                _prior_row(
                    "exp3272",
                    "flagged",
                    flags=[{"kind": "TAUTOLOGY", "detail": "assembled counts match"}],
                ),
                _prior_row(
                    "exp3273",
                    "sidecar-only",
                    bounded=["sidecar_only=true", "delong_noninferiority_passed=false"],
                ),
                _prior_row("exp3274", "blocked", blockers=["blocked_garak_unavailable"]),
                _prior_row("exp3275", "blocked", blockers=["abstention_rate_above_threshold"]),
                _prior_row(
                    "exp3276",
                    "blocked",
                    blockers=["garak_redteam_eval_ready actual=False == expected=True"],
                ),
                _prior_row(
                    "exp3277",
                    "missing",
                    blockers=[
                        "artifact_missing: "
                        "results/experiment_3277_sota_repair_micro_panel_v9.json"
                    ],
                ),
            ],
        },
    )


def _write_dot304_sources(root: Path) -> None:
    _write_json(
        root,
        mod.EXP3281_REL_PATH,
        {
            "experiment_id": "exp3281",
            "task_id": "exp3281-archive-v303-activate-v304",
            "v303_closed_v304_opened": True,
            "prior_paper_ready": False,
            "prior_publication_blocker_count": 105,
            "prior_next_top_gap": "unblock_garak_redteam_eval",
            "full_15k_corpus_materialized": True,
            "garak_blocker": "blocked_garak_unavailable",
            "clean_verifier_abstention_rate": 1.0,
            "kan_noninferiority_passed": False,
            "repair_gate_open": False,
            "honest_verdict": "complete: v303 closed v304 opened",
        },
    )
    _write_json(
        root,
        mod.EXP3282_REL_PATH,
        {
            "experiment_id": "exp3282",
            "task_id": "exp3282-garak-install-and-probe-manifest-v1",
            "garak_install_probe_manifest_ready": True,
            "garak_runner_ready": True,
            "garak_available": True,
            "garak_version": "0.15.0",
            "install_blockers": [],
            "promptinject_probe_count": 6,
            "honest_verdict": "complete: garak runner available",
        },
    )
    _write_json(
        root,
        mod.EXP3283_REL_PATH,
        {
            "experiment_id": "exp3283",
            "task_id": "exp3283-prompt-injection-corrigendum-duration-audit-v1",
            "corrigendum_ready": True,
            "duration_flags": [
                {"experiment_id": "exp3270", "kind": "DURATION_TOO_SHORT"},
                {"experiment_id": "exp3271", "kind": "DURATION_TOO_SHORT"},
            ],
            "tautology_flags": [{"experiment_id": "exp3272", "kind": "TAUTOLOGY"}],
            "downstream_usage_rules": {
                "paper_claims": {"headline_performance_metrics_allowed": False}
            },
            "honest_verdict": "complete: corrigendum flags preserved",
        },
    )
    _write_json(
        root,
        mod.EXP3284_REL_PATH,
        {
            "experiment_id": "exp3284",
            "task_id": "exp3284-garak-local-smoke-sota-gguf-v1",
            "garak_local_smoke_v1_ready": True,
            "garak_smoke_ready": True,
            "local_target_adapter_started": True,
            "garak_probe_count": 20,
            "attack_success_rate": 0.25,
            "honest_verdict": "complete: garak smoke ready",
        },
    )
    _write_json(
        root,
        mod.EXP3285_REL_PATH,
        {
            "experiment_id": "exp3285",
            "task_id": "exp3285-full-garak-dataflip-redteam-eval-v2",
            "garak_dataflip_redteam_eval_v2_ready": True,
            "garak_redteam_eval_ready": True,
            "garak_gate_passed": False,
            "dataflip_gate_passed": True,
            "blocked_reasons": ["garak_attack_success_or_error_gate_failed"],
            "garak_probe_count": 90,
            "attack_success_rate": 0.311111,
            "honest_verdict": "complete: garak gate failed",
        },
    )
    _write_json(
        root,
        mod.EXP3286_REL_PATH,
        {
            "experiment_id": "exp3286",
            "abstention_root_cause_audit_ready": True,
            "abstention_root_cause_identified": True,
            "dominant_root_cause": "model_output_parser_contract_mismatch",
            "prior_abstention_rate": 1.0,
            "honest_verdict": "complete: abstention root cause identified",
        },
    )
    _write_json(
        root,
        mod.EXP3287_REL_PATH,
        {
            "experiment_id": "exp3287",
            "abstention_calibrated_clean_verifier_v15_ready": True,
            "clean_verifier_rerun_ready": True,
            "repair_gate_input_clean_enough": True,
            "false_accept_rate": 0.0,
            "false_reject_rate": 0.0,
            "abstention_rate": 0.0,
            "coverage_rate": 1.0,
            "honest_verdict": "complete: clean verifier rerun ready",
        },
    )
    _write_json(
        root,
        mod.EXP3288_REL_PATH,
        {
            "experiment_id": "exp3288",
            "task_id": "exp3288-kan-sidecar-failure-autopsy-boundary-v1",
            "kan_failure_autopsy_ready": True,
            "kan_boundary_decision_ready": True,
            "prior_full_corpus_auroc": 0.475326,
            "prior_delong_noninferiority_passed": False,
            "kan_boundary_decision": "retire_from_prompt_injection_headline",
            "permitted_downstream_use": ["offline_failure_autopsy"],
            "honest_verdict": "complete: KAN retired from headline",
        },
    )
    _write_json(
        root,
        mod.EXP3289_REL_PATH,
        {
            "experiment_id": "exp3289",
            "task_id": "exp3289-repair-gate-decision-v9-after-garak-abstention",
            "repair_gate_decision_v9_ready": True,
            "repair_gate_open": True,
            "garak_redteam_eval_ready": True,
            "clean_verifier_rerun_ready": True,
            "kan_boundary_decision_ready": True,
            "blocked_reasons": [],
            "honest_verdict": "complete: repair gate open",
        },
    )
    _write_json(
        root,
        mod.EXP3290_REL_PATH,
        {
            "experiment_id": "exp3290",
            "task_id": "exp3290-gated-sota-repair-micro-panel-v10",
            "sota_repair_micro_panel_v10_ready": True,
            "repair_panel_ran": True,
            "panel_case_count": 4,
            "verified_success_count": 4,
            "false_accept_count": 0,
            "abstention_count": 0,
            "repair_success_rate": 1.0,
            "headline_claim_allowed": False,
            "flagged_adversarial": True,
            "corrigendum_pending": [
                {"kind": "DURATION_TOO_SHORT", "detail": "duration_s=10.893664"}
            ],
            "honest_verdict": "complete: repair panel diagnostic only",
        },
    )
    _write_json(
        root,
        mod.EXP3291_REL_PATH,
        {
            "experiment_id": "exp3291",
            "task_id": "exp3291-fr11-garak-abstention-memory-replay-v1",
            "fr11_garak_abstention_memory_replay_ready": True,
            "continuous_self_learning_task": True,
            "controller_memory_only": True,
            "foundation_weight_updates_performed": False,
            "raw_episodes_preserved": True,
            "retention_score": 0.982143,
            "adaptation_score": 1.0,
            "forgetting_rate": 0.017857,
            "negative_transfer_rate": 0.0,
            "heldout_trace_count": 2056,
            "blocked_trace_categories": [
                "clean_verifier_abstention",
                "garak_redteam",
                "garak_toolchain",
                "kan_boundary",
            ],
            "honest_verdict": "complete: FR-11 replay ready",
        },
    )


def _row_by_id(artifact: Mapping[str, Any], experiment_id: str) -> Mapping[str, Any]:
    return next(row for row in artifact["rows"] if row["experiment_id"] == experiment_id)


def _resolution_by_prior(artifact: Mapping[str, Any], experiment_id: str) -> Mapping[str, Any]:
    return next(
        row
        for row in artifact["prior_blocker_resolution"]
        if row["prior_experiment_id"] == experiment_id
    )


def test_req_report_3292_spec_anchor_exists() -> None:
    """REQ-REPORT-3292: OpenSpec declares matrix v36 before implementation."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3292" in spec
    assert "SCENARIO-REPORT-3292" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    for field in REQUIRED_FIELDS:
        assert field in spec


def test_scenario_report_3292_builds_v36_from_dot304_artifacts(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3292: v36 records .304 gates and unresolved .303 blockers."""

    _write_prior_v35(tmp_path)
    _write_dot304_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=10.0, now_s=13.5)
    statuses = {row["experiment_id"]: row["status"] for row in artifact["rows"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["experiment_id"] == "exp3292"
    assert artifact["task_id"] == "exp3292-evidence-matrix-v36"
    assert artifact["milestone"] == "2026.05.304"
    assert artifact["inference_substrate"] == "artifact_aggregation_only"
    assert artifact["matrix_v36_ready"] is True
    assert artifact["duration_s"] == pytest.approx(3.5)
    assert artifact["artifact_count_scanned"] == 11
    assert statuses == {
        "exp3281": "paper-blocking",
        "exp3282": "clean",
        "exp3283": "flagged",
        "exp3284": "clean",
        "exp3285": "blocked",
        "exp3286": "clean",
        "exp3287": "clean",
        "exp3288": "sidecar-only",
        "exp3289": "clean",
        "exp3290": "flagged",
        "exp3291": "clean",
    }
    assert artifact["primary_status_counts"] == {
        "clean": 6,
        "blocked": 1,
        "flagged": 2,
        "sidecar-only": 1,
        "missing": 0,
        "paper-blocking": 1,
    }
    assert artifact["clean_evidence_count"] == 6
    assert artifact["blocked_evidence_count"] == 1
    assert artifact["flagged_evidence_count"] == 2
    assert artifact["sidecar_only_count"] == 1
    assert artifact["missing_evidence_count"] == 0
    assert artifact["paper_blocker_count"] == 5

    assert _row_by_id(artifact, "exp3285")["blocker_reasons"] == [
        "garak_attack_success_or_error_gate_failed",
        "garak_gate_passed=false",
    ]
    assert _row_by_id(artifact, "exp3290")["quality_flags"] == [
        {"kind": "DURATION_TOO_SHORT", "detail": "duration_s=10.893664"}
    ]
    assert _row_by_id(artifact, "exp3288")["bounded_claims"] == [
        "kan_boundary_decision=retire_from_prompt_injection_headline",
        "prior_delong_noninferiority_passed=false",
    ]
    assert _row_by_id(artifact, "exp3291")["bounded_claims"] == [
        "controller_memory_only=true",
        "foundation_weight_updates_performed=false",
    ]

    assert artifact["gate_summary"]["garak_toolchain"]["status"] == "clean"
    assert artifact["gate_summary"]["garak_redteam"]["status"] == "blocked"
    assert artifact["gate_summary"]["garak_redteam"]["garak_gate_passed"] is False
    assert artifact["gate_summary"]["clean_verifier"]["status"] == "clean"
    assert artifact["gate_summary"]["kan_boundary"]["status"] == "sidecar-only"
    assert artifact["gate_summary"]["repair_gate"]["repair_gate_open"] is True
    assert artifact["gate_summary"]["repair_panel"]["status"] == "flagged"
    assert artifact["gate_summary"]["fr11"]["controller_memory_only"] is True

    assert [row["prior_experiment_id"] for row in artifact["carried_forward_blockers"]] == [
        "exp3270",
        "exp3271",
        "exp3272",
    ]
    assert _resolution_by_prior(artifact, "exp3274")["resolution_status"] == "replaced"
    assert _resolution_by_prior(artifact, "exp3275")["resolution_status"] == "resolved"
    assert _resolution_by_prior(artifact, "exp3277")["resolution_status"] == "resolved"
    assert _resolution_by_prior(artifact, "exp3273")["resolution_status"] == "bounded"

    assert artifact["top_gaps"][0] == {
        "rank": 1,
        "gap": "pass_garak_redteam_gate",
        "source_experiment_id": "exp3285",
        "status": "blocked",
        "reason": "garak_attack_success_or_error_gate_failed",
    }
    assert artifact["top_gaps"][1]["gap"] == "repair_panel_duration_and_scope_boundary"
    assert artifact["top_gaps"][2]["gap"] == "resolve_dot303_methodology_flags"
    assert artifact["paper_ready"] is False
    assert artifact["source_checksums"][mod.EXP3281_REL_PATH.as_posix()]
    assert artifact["prior_matrix"]["present"] is True
    assert artifact["protected_files_untouched"]["scripts/research_conductor.py"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    assert "matrix_v36_ready=true" in artifact["honest_verdict"]
    mod.validate_artifact(artifact)


def test_req_report_3292_writer_and_empty_sources_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-3292: writer persists output and missing evidence stays missing."""

    _write_prior_v35(tmp_path)
    _write_dot304_sources(tmp_path)
    output = mod.write_artifact(tmp_path, started_s=1.0, now_s=2.25)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["matrix_v36_ready"] is True
    assert saved["duration_s"] == pytest.approx(1.25)

    empty = mod.build_artifact(tmp_path / "empty", started_s=0.0, now_s=0.0)
    assert empty["matrix_v36_ready"] is False
    assert empty["artifact_count_scanned"] == 0
    assert empty["missing_evidence_count"] == 11
    assert empty["blocked_evidence_count"] == 0
    assert empty["paper_ready"] is False
    assert "exp3281 .304 handoff artifact is missing or not ready" in empty["invariant_violations"]
    assert "prior matrix v35 is missing or not ready" in empty["invariant_violations"]
    mod.validate_artifact(empty)


def test_req_report_3292_defensive_helpers_and_validation(tmp_path: Path) -> None:
    """REQ-REPORT-3292: helper paths classify malformed evidence conservatively."""

    assert mod.read_json_object(tmp_path / "missing.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{bad", encoding="utf-8")
    assert mod.read_json_object(bad) == {}
    non_object = tmp_path / "list.json"
    non_object.write_text("[]", encoding="utf-8")
    assert mod.read_json_object(non_object) == {}
    present = tmp_path / "present.json"
    present.write_text('{"ok": true}\n', encoding="utf-8")
    assert mod.sha256_file(tmp_path / "missing.json") is None
    assert len(mod.sha256_file(present) or "") == 64
    assert mod._duration(3.0, 2.0) == 0.0
    assert mod._normal_status("sidecar_only") == "sidecar-only"
    assert mod._normal_status("paper_blocking") == "paper-blocking"
    assert mod._normal_status("weird") == "missing"
    assert mod._int_value(7) == 7
    assert mod._int_value(True) == 0
    assert mod._as_mapping({"a": 1}) == {"a": 1}
    assert mod._as_mapping([]) == {}
    assert mod._as_list([{"a": 1}]) == [{"a": 1}]
    assert mod._as_list("bad") == []
    assert mod._list_of_strings(["a", 2, None]) == ["a", "2", "None"]
    assert mod._quality_flags({"flagged_adversarial": True}) == [
        {"kind": "flagged_adversarial", "detail": "flagged_adversarial=true"}
    ]
    assert mod._explicit_blockers(
        {"blocked_reason": "blocked_one", "gate_check_summary": "gate_two"}
    ) == ["blocked_one", "gate_two"]
    assert mod._bounded_claims({"sidecar_only": True}) == ["sidecar_only=true"]
    assert mod._resolve_prior_blocker({"experiment_id": "exp9999", "status": "blocked"}, {})[
        "resolution_status"
    ] == "unresolved"
    assert (
        mod._status_for_source(
            mod.EXPECTED_SOURCES[0],
            {"v303_closed_v304_opened": True, "prior_paper_ready": True},
            True,
        )
        == "clean"
    )
    assert (
        mod._status_for_source(
            mod.EXPECTED_SOURCES[0],
            {"v303_closed_v304_opened": True, "prior_paper_ready": False},
            True,
        )
        == "paper-blocking"
    )
    assert mod._blocker_reasons(
        {"present": True, "ready_field": "some_ready_field", "experiment_id": "expX"},
        {"some_ready_field": False},
    ) == ["some_ready_field=false"]

    _write_prior_v35(tmp_path)
    _write_dot304_sources(tmp_path)
    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.0)
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
