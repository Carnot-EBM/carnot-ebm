"""Tests for Exp 3122 archive .290 and .291 handoff.

Spec refs: REQ-REPORT-3122, SCENARIO-REPORT-3122.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import archive_v290_activate_v291_3122 as mod


REQUIRED_FIELDS = {
    "archive_v290_activate_v291_ready",
    "prior_capstone_ready",
    "prior_paper_ready",
    "prior_publication_blocker_count",
    "carry_forward_blockers",
    "next_milestone",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(root: Path, rel_path: Path, text: str) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _matrix_payload() -> dict[str, Any]:
    return {
        "artifact": "experiment_3120_cross_corpus_matrix_v24",
        "milestone": "2026.05.290",
        "matrix_v24_ready": True,
        "publication_blocker_count": 36,
        "blocker_delta_from_v23": 0,
        "headline_model_spec_gaps": [
            {
                "row_id": "dot290:exp3110_sota_model_spec_cache_manifest",
                "source_artifact": (
                    "results/experiment_3110_sota_model_spec_cache_manifest_corrigendum_v1.json"
                ),
                "present_model_ids": ["unsloth/gemma-4-26B-A4B-it-GGUF"],
                "missing_model_ids": [
                    "unsloth/Qwen3.6-35B-A3B-GGUF",
                    "unsloth/gemma-4-31B-it-GGUF",
                ],
                "reason": "mandated headline model cache coverage incomplete",
            }
        ],
        "verifier_repair_status": {
            "diagnostic_calibration_status": "diagnostic_only",
            "repair_micro_panel_status": "bounded",
            "repair_success_delta": 0.0,
            "intent_preservation_rate": 0.0,
        },
        "fr11_status": {
            "controller_only": True,
            "promotion_decision": "controller_only",
            "no_weight_update_claim": True,
            "soundness_mistakes": 0,
            "completeness_mistakes": 0,
        },
        "architecture_boundary_status": {
            "ebt_arm_status": "projection_only_no_live_model_integration",
            "clut_status": "bounded_cpu_only_no_hardware_speedup",
            "gatemate_status": "blocked_operator_evidence_incomplete",
            "ssqa_status": "gated_skipped_host_visible_readback_missing",
        },
        "source_artifacts": [
            {
                "path": "results/experiment_3110_sota_model_spec_cache_manifest_corrigendum_v1.json",
                "role": "sota_model_spec_cache_manifest",
            },
            {
                "path": "results/experiment_3115_explicit_repair_gate_micro_panel_v4.json",
                "role": "explicit_repair_gate_micro_panel",
            },
        ],
        "honest_verdict": (
            "complete: matrix_v24_ready=true; rows_total=102; "
            "publication_blocker_count=36; blocker_delta_from_v23=0"
        ),
    }


def _capstone_payload(
    *,
    capstone_ready: bool = True,
    paper_ready: bool = False,
    publication_blocker_count: int | None = 36,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "artifact": "experiment_3121_capstone_v290",
        "milestone": "2026.05.290",
        "capstone_ready": capstone_ready,
        "paper_ready": paper_ready,
        "blocker_delta_from_v23": 0,
        "verifier_gain_status": (
            "diagnostic_gain_recovered_but_headline_bounded_by_cache_and_prior_flags"
        ),
        "formal_feedback_status": "solver_certified_feedback_ready_no_live_sota_lift",
        "repair_claim_status": "bounded_micro_panel_executed_zero_delta_no_promotion",
        "fr11_self_learning_status": (
            "bounded_controller_only_soundness_zero_completeness_zero_no_weight_update"
        ),
        "ebt_arm_status": "projection_only_sidecar_correlation_no_live_model_integration",
        "sampler_hardware_status": "bounded_clut_cpu_only_no_hardware_speedup",
        "gatemate_status": "blocked_operator_evidence_incomplete_no_hardware_run",
        "ssqa_status": "gated_skipped_host_visible_readback_missing",
        "remaining_top_gaps": [
            "publishable_verifier_repair_headline_evidence",
            "operator_owned_gatemate_ssqa_host_visible_evidence",
            "live_model_or_authenticated_hardware_architecture_integration",
        ],
        "headline_model_spec_gaps": _matrix_payload()["headline_model_spec_gaps"],
        "source_artifacts": [
            {"path": mod.MATRIX_V24_REL_PATH.as_posix(), "role": "matrix_v24_authority"},
            {
                "path": "results/experiment_3115_explicit_repair_gate_micro_panel_v4.json",
                "role": "explicit_repair_gate_micro_panel",
            },
        ],
        "matrix_v24_summary": {
            "matrix_v24_ready": True,
            "publication_blocker_count": 36,
            "blocker_delta_from_v23": 0,
            "rows_total": 102,
        },
        "honest_verdict": (
            "complete: capstone_ready=true; paper_ready=false; "
            "publication_blocker_count=36; blocker_delta_from_v23=0; "
            "next_top_gap=publishable_verifier_repair_headline_evidence"
        ),
    }
    if publication_blocker_count is not None:
        payload["publication_blocker_count"] = publication_blocker_count
    return payload


def _roadmap_yaml(
    *,
    milestone: str = "2026.05.291",
    milestone_doc: str = "openspec/change-proposals/research-roadmap-vNEXT.md",
    include_task: bool = True,
) -> str:
    tasks = (
        "tasks:\n"
        "  - id: exp3122-archive-v290-activate-v291\n"
        "    deliverable: results/experiment_3122_archive_v290_activate_v291.json\n"
        if include_task
        else "tasks: []\n"
    )
    return (
        f'milestone: "{milestone}"\n'
        'milestone_title: "Live SOTA Verifier Repair"\n'
        f'milestone_doc: "{milestone_doc}"\n'
        f"{tasks}"
    )


def _write_common_sources(
    root: Path,
    *,
    staged_roadmap: bool = True,
    active_milestone: str = "2026.05.291",
    capstone_ready: bool = True,
    paper_ready: bool = False,
) -> None:
    _write_json(root, mod.MATRIX_V24_REL_PATH, _matrix_payload())
    _write_json(
        root,
        mod.CAPSTONE_V290_REL_PATH,
        _capstone_payload(capstone_ready=capstone_ready, paper_ready=paper_ready),
    )
    _write_json(
        root,
        Path("results/experiment_3110_sota_model_spec_cache_manifest_corrigendum_v1.json"),
        {"artifact": "experiment_3110_sota_model_spec_cache_manifest_corrigendum_v1"},
    )
    _write_json(
        root,
        Path("results/experiment_3115_explicit_repair_gate_micro_panel_v4.json"),
        {"artifact": "experiment_3115_explicit_repair_gate_micro_panel_v4"},
    )
    if staged_roadmap:
        _write_text(root, mod.STAGED_ROADMAP_REL_PATH, _roadmap_yaml())
    _write_text(root, mod.ACTIVE_ROADMAP_REL_PATH, _roadmap_yaml(milestone=active_milestone))
    _write_text(root, mod.VNEXT_DOC_REL_PATH, "# Research Roadmap vNEXT - Milestone 2026.05.291\n")
    _write_text(root, mod.AGENTS_REL_PATH, "# repo instructions\n")
    _write_text(root, mod.CODEX_REL_PATH, "# codex instructions\n")
    _write_text(root, mod.CLAUDE_REL_PATH, "# claude instructions\n")
    _write_text(root, mod.CONDUCTOR_REL_PATH, "# protected conductor\n")
    _write_text(root, mod.OPS_STATUS_REL_PATH, "status mentions exp3121 and .291\n")
    _write_text(root, mod.OPS_CHANGELOG_REL_PATH, "changelog mentions exp3121 and .291\n")
    _write_text(root, mod.TRACEABILITY_REL_PATH, "traceability mentions REQ-REPORT-3121\n")


def test_req_report_3122_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3122: OpenSpec declares the archive/handoff contract first."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3122" in spec
    assert "SCENARIO-REPORT-3122" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_report_3122_builds_ready_archive_from_staged_roadmap(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3122: staged .291 roadmap can drive a ready handoff."""

    _write_common_sources(tmp_path, staged_roadmap=True, active_milestone="2026.05.290")

    artifact = mod.build_artifact(tmp_path, started_s=5.0, now_s=6.25)
    blockers = {row["blocker_id"]: row for row in artifact["carry_forward_blockers"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["archive_v290_activate_v291_ready"] is True
    assert artifact["prior_capstone_ready"] is True
    assert artifact["prior_paper_ready"] is False
    assert artifact["prior_publication_blocker_count"] == 36
    assert artifact["next_milestone"] == "2026.05.291"
    assert artifact["duration_s"] == pytest.approx(1.25)
    assert artifact["honest_verdict"].startswith("complete: archive_v290_activate_v291_ready=true")
    assert artifact["status_summary_290"] == {
        "paper_ready": False,
        "capstone_ready": True,
        "matrix_v24_ready": True,
        "publication_blocker_count": 36,
        "publication_blocker_count_source": "capstone_publication_blocker_count",
        "next_top_gap": "publishable_verifier_repair_headline_evidence",
        "model_cache_status": {
            "present_model_ids": ["unsloth/gemma-4-26B-A4B-it-GGUF"],
            "missing_model_ids": [
                "unsloth/Qwen3.6-35B-A3B-GGUF",
                "unsloth/gemma-4-31B-it-GGUF",
            ],
            "headline_model_spec_gaps": _matrix_payload()["headline_model_spec_gaps"],
        },
        "verifier_gain_status": (
            "diagnostic_gain_recovered_but_headline_bounded_by_cache_and_prior_flags"
        ),
        "formal_feedback_status": "solver_certified_feedback_ready_no_live_sota_lift",
        "repair_claim_status": "bounded_micro_panel_executed_zero_delta_no_promotion",
        "fr11_self_learning_status": (
            "bounded_controller_only_soundness_zero_completeness_zero_no_weight_update"
        ),
        "ebt_arm_status": "projection_only_sidecar_correlation_no_live_model_integration",
        "sampler_hardware_status": "bounded_clut_cpu_only_no_hardware_speedup",
        "gatemate_status": "blocked_operator_evidence_incomplete_no_hardware_run",
        "ssqa_status": "gated_skipped_host_visible_readback_missing",
        "source_artifacts": _capstone_payload()["source_artifacts"],
    }
    assert set(blockers) == {
        "publication_blockers_36",
        "missing_headline_cache_coverage",
        "diagnostic_only_verifier_lift",
        "zero_repair_delta",
        "fr11_controller_only_learning",
        "ebt_arm_projection_only",
        "cpu_only_clut",
        "missing_operator_visible_hardware_evidence",
    }
    assert blockers["publication_blockers_36"]["value"] == 36
    assert blockers["missing_headline_cache_coverage"]["value"] == {
        "missing_model_ids": [
            "unsloth/Qwen3.6-35B-A3B-GGUF",
            "unsloth/gemma-4-31B-it-GGUF",
        ],
        "present_model_ids": ["unsloth/gemma-4-26B-A4B-it-GGUF"],
    }
    assert blockers["diagnostic_only_verifier_lift"]["value"] == (
        "diagnostic_gain_recovered_but_headline_bounded_by_cache_and_prior_flags"
    )
    assert blockers["zero_repair_delta"]["value"] == {
        "repair_claim_status": "bounded_micro_panel_executed_zero_delta_no_promotion",
        "repair_success_delta": 0.0,
    }
    assert blockers["fr11_controller_only_learning"]["value"] == (
        "bounded_controller_only_soundness_zero_completeness_zero_no_weight_update"
    )
    assert blockers["ebt_arm_projection_only"]["value"] == (
        "projection_only_sidecar_correlation_no_live_model_integration"
    )
    assert blockers["cpu_only_clut"]["value"] == "bounded_clut_cpu_only_no_hardware_speedup"
    assert blockers["missing_operator_visible_hardware_evidence"]["value"] == {
        "gatemate_status": "blocked_operator_evidence_incomplete_no_hardware_run",
        "ssqa_status": "gated_skipped_host_visible_readback_missing",
    }
    assert all(row["matches_expected"] is True for row in blockers.values())
    assert artifact["roadmap_handoff"]["source_path"] == mod.STAGED_ROADMAP_REL_PATH.as_posix()
    assert artifact["roadmap_handoff"]["requested_staged_roadmap_present"] is True
    assert artifact["roadmap_handoff"]["used_active_roadmap_fallback"] is False
    assert artifact["roadmap_handoff"]["task_ids"] == ["exp3122-archive-v290-activate-v291"]
    assert artifact["activation_performed_by_this_task"] is False
    assert artifact["research_roadmap_yaml_modified"] is False
    assert artifact["research_roadmap_next_yaml_modified"] is False
    assert artifact["scripts_research_conductor_modified"] is False
    assert artifact["ops_status_updated"] is False
    assert artifact["ops_changelog_updated"] is False
    assert artifact["traceability_updated"] is False
    assert artifact["inference_substrate"] == {
        "kind": "aggregation_from_upstream_artifacts",
        "executes_models": False,
        "executes_verifiers": False,
        "executes_repairs": False,
        "executes_solvers": False,
        "executes_hardware": False,
        "executes_conductor": False,
        "local_repo_only": True,
        "no_live_llm_inference": True,
        "source": "checked_in_artifacts",
        "live_model_calls": 0,
        "hardware_commands_run": [],
    }


def test_req_report_3122_uses_active_roadmap_fallback_after_activation(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-3122: post-activation active roadmap is audited read-only."""

    _write_common_sources(tmp_path, staged_roadmap=False)

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.5)

    assert artifact["archive_v290_activate_v291_ready"] is True
    assert artifact["roadmap_handoff"]["source_path"] == mod.ACTIVE_ROADMAP_REL_PATH.as_posix()
    assert artifact["roadmap_handoff"]["requested_staged_roadmap_present"] is False
    assert artifact["roadmap_handoff"]["used_active_roadmap_fallback"] is True
    assert artifact["roadmap_handoff"]["active_roadmap_milestone"] == "2026.05.291"
    assert mod.STAGED_ROADMAP_REL_PATH.as_posix() in artifact["missing_source_artifacts"]


def test_req_report_3122_blocks_when_capstone_is_not_ready(tmp_path: Path) -> None:
    """REQ-REPORT-3122: prior capstone readiness is a hard precondition."""

    _write_common_sources(tmp_path, capstone_ready=False)

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.25)

    assert artifact["archive_v290_activate_v291_ready"] is False
    assert artifact["prior_capstone_ready"] is False
    assert artifact["honest_verdict"].startswith("blocked_prior_capstone_not_ready:")
    assert "prior capstone is not capstone_ready=true" in artifact["blocked_reasons"]


def test_req_report_3122_blocks_when_roadmap_handoff_is_invalid(tmp_path: Path) -> None:
    """REQ-REPORT-3122: roadmap milestone/doc/tasks must match the .291 handoff."""

    _write_common_sources(tmp_path, staged_roadmap=False, active_milestone="2026.05.290")

    artifact = mod.build_artifact(tmp_path, started_s=2.0, now_s=2.25)

    assert artifact["archive_v290_activate_v291_ready"] is False
    assert artifact["honest_verdict"].startswith("blocked_roadmap_handoff_not_ready:")
    assert "roadmap milestone is not 2026.05.291" in artifact["blocked_reasons"]


def test_req_report_3122_write_artifact_persists_stable_json(tmp_path: Path) -> None:
    """REQ-REPORT-3122: write_artifact emits the deliverable JSON."""

    _write_common_sources(tmp_path)

    output = mod.write_artifact(tmp_path, started_s=3.0, now_s=3.125)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["archive_v290_activate_v291_ready"] is True
    assert saved["duration_s"] == pytest.approx(0.125)
    checksums = {row["path"]: row["sha256"] for row in saved["source_artifacts"]}
    assert checksums[mod.CAPSTONE_V290_REL_PATH.as_posix()] == _sha256(
        tmp_path / mod.CAPSTONE_V290_REL_PATH
    )


def test_req_report_3122_helper_edges_remain_honest(tmp_path: Path) -> None:
    """REQ-REPORT-3122: helper edges preserve malformed and unusual states."""

    malformed_json = tmp_path / "bad.json"
    list_json = tmp_path / "list.json"
    malformed_yaml = tmp_path / "bad.yaml"
    list_yaml = tmp_path / "list.yaml"
    malformed_json.write_text("{bad-json}\n", encoding="utf-8")
    list_json.write_text("[1, 2, 3]\n", encoding="utf-8")
    malformed_yaml.write_text(": bad\n", encoding="utf-8")
    list_yaml.write_text("- x\n", encoding="utf-8")

    assert mod.read_json_object(tmp_path / "missing.json") == {}
    assert mod.read_json_object(malformed_json) == {}
    assert mod.read_json_object(list_json) == {}
    assert mod.read_yaml_mapping(tmp_path / "missing.yaml") == {}
    assert mod.read_yaml_mapping(malformed_yaml) == {}
    assert mod.read_yaml_mapping(list_yaml) == {}
    assert mod.sha256_file(tmp_path / "missing.txt") is None
    assert mod._as_mapping({"x": 1}) == {"x": 1}
    assert mod._as_mapping(None) == {}
    assert mod._as_list(["x"]) == ["x"]
    assert mod._as_list("x") == []
    assert mod._int_or(True, 7) == 7
    assert mod._int_or("5", 7) == 5
    assert mod._int_or("x", 7) == 7
    assert mod._first_int_from_text("publication_blocker_count=12") == 12
    assert mod._first_int_from_text("no count") is None
    assert mod._next_top_gap({"remaining_top_gaps": ["gap-a"]}) == "gap-a"
    assert mod._next_top_gap({"honest_verdict": "complete: next_top_gap=gap-b"}) == "gap-b"
    assert mod._next_top_gap({}) == ""
    assert mod._publication_blocker_count(
        {"publication_blocker_count": 9},
        {},
    ) == (9, "capstone_publication_blocker_count")
    assert mod._publication_blocker_count(
        {"publication_blockers": [1, 2]},
        {},
    ) == (2, "capstone_publication_blockers_length")
    assert mod._publication_blocker_count(
        {"matrix_v24_summary": {"publication_blocker_count": 8}},
        {},
    ) == (8, "capstone_matrix_v24_summary")
    assert mod._publication_blocker_count(
        {"honest_verdict": "complete: publication_blocker_count=4"},
        {},
    ) == (4, "capstone_honest_verdict")
    assert mod._publication_blocker_count(
        {},
        {"honest_verdict": "complete: publication_blocker_count=3"},
    ) == (3, "matrix_honest_verdict")
    assert mod._publication_blocker_count({}, {}) == (0, "missing")
    assert mod._status_value("repair_claim_status", {"repair_claim_status": "bounded"}, {}) == (
        "bounded"
    )
    assert mod._task_ids({"tasks": [{"id": "a"}, {"no": "id"}, "bad"]}) == ["a"]
    assert mod._task_ids({"tasks": "bad"}) == []
    assert mod._source_artifact(tmp_path, "missing", Path("missing.txt"))["present"] is False
    assert mod._repair_success_delta({}, {"verifier_repair_status": {"repair_success_delta": 0.5}})
    assert mod._repair_success_delta({"repair_success_delta": 0.25}, {}) == 0.25
    assert mod._repair_success_delta({}, {}) == 0.0
    assert mod._blocked_reasons(
        capstone_present=False,
        prior_capstone_ready=False,
        roadmap_handoff={
            "source_present": False,
            "milestone_matches": False,
            "milestone_doc_matches": False,
            "non_empty_tasks": False,
        },
        vnext_doc_present=False,
    ) == [
        "prior capstone artifact missing or malformed",
        "roadmap handoff source is missing",
        "roadmap milestone is not 2026.05.291",
        "roadmap milestone_doc is not openspec/change-proposals/research-roadmap-vNEXT.md",
        "roadmap has no tasks",
        "openspec/change-proposals/research-roadmap-vNEXT.md is missing",
    ]
