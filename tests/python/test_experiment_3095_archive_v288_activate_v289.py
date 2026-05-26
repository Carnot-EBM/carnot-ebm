"""Tests for Exp 3095 archive .288 and .289 handoff.

Spec refs: REQ-REPORT-3095, SCENARIO-REPORT-3095.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import archive_v288_activate_v289_3095 as mod


REQUIRED_FIELDS = {
    "archive_v288_activate_v289_ready",
    "prior_capstone_ready",
    "prior_paper_ready",
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
        "artifact": "experiment_3093_cross_corpus_matrix_v22",
        "milestone": "2026.05.288",
        "matrix_v22_ready": True,
        "publication_blocker_count": 36,
        "headline_model_spec_gaps": [
            {
                "row_id": "dot288:exp3085_icalm_task_abstention_sota_panel",
                "source_artifact": "results/experiment_3085.json",
                "missing_model_ids": [],
                "present_model_ids": ["unsloth/gemma-4-26B-A4B-it-GGUF"],
                "reason": "mandatory_headline_model_ids missing for live LLM artifact",
            }
        ],
        "honest_verdict": (
            "complete: matrix_v22_ready=true; rows_total=75; "
            "publication_blocker_count=36; missing_artifacts=1"
        ),
    }


def _capstone_payload(
    *,
    capstone_ready: bool = True,
    paper_ready: bool = False,
    publication_blocker_count: int | None = 36,
    verifier_repair_count: int = 17,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "artifact": "experiment_3094_capstone_v288",
        "milestone": "2026.05.288",
        "capstone_ready": capstone_ready,
        "paper_ready": paper_ready,
        "verifier_gain_status": "flagged_or_gated_verifier_gain_recovery_incomplete",
        "repair_claim_status": "bounded_flagged_gated_missing_verifier_gated",
        "fr11_self_learning_status": "clean_controller_only_zero_mistake_budget",
        "ebt_arm_status": "projection_only_sidecar_schema_no_model_integration",
        "gatemate_status": "blocked_no_rerun_operator_actions_required",
        "ssqa_status": "gated_skipped_host_visible_smoke_missing",
        "missing_capstone_input_artifacts": [
            {
                "path": "results/experiment_3089_gated_xgrammar_sota_repair_micro_panel_v2.json",
                "reason": "named by matrix v22 capstone_input_artifacts but not readable",
            }
        ],
        "headline_model_spec_gaps": [
            {
                "row_id": "dot288:exp3085_icalm_task_abstention_sota_panel",
                "source_artifact": "results/experiment_3085_icalm_task_abstention_sota_panel_v2.json",
                "missing_model_ids": [],
                "present_model_ids": ["unsloth/gemma-4-26B-A4B-it-GGUF"],
                "reason": "mandatory_headline_model_ids missing for live LLM artifact",
            },
            {
                "row_id": "dot288:exp3086_dafny_z3_formal_feedback_pilot",
                "source_artifact": "results/experiment_3086_dafny_z3_formal_feedback_pilot_v1.json",
                "missing_model_ids": [
                    "unsloth/Qwen3.6-35B-A3B-GGUF",
                    "unsloth/gemma-4-31B-it-GGUF",
                ],
                "present_model_ids": ["unsloth/gemma-4-26B-A4B-it-GGUF"],
                "reason": "mandated model_specs missing for live LLM artifact",
            },
        ],
        "prd_gap_summary": {
            "verifier_repair": {
                "publication_blocker_count": verifier_repair_count,
                "publication_blocker_row_ids": ["vr-1", "vr-2"],
            },
            "ebt_arm_bridge": {"publication_blocker_count": 1},
            "hardware_evidence": {"publication_blocker_count": 11},
        },
        "next_milestone_recommendation": (
            "2026.05.289: clear verifier/repair first (17 blocker rows: verifier "
            "gain, formal feedback, and missing repair micro-panel), because repair "
            "promotion remains gated by verifier evidence; then collect operator-owned "
            "GateMate/SSQA host-visible evidence (11 blocker rows) without claiming "
            "speedup; keep EBT/ARM projection-only until live model integration has "
            "tests."
        ),
        "honest_verdict": (
            "complete: capstone_ready=true; paper_ready=false; matrix_v22_ready=true; "
            "publication_blocker_count=36; "
            "verifier_gain_status=flagged_or_gated_verifier_gain_recovery_incomplete; "
            "repair_claim_status=bounded_flagged_gated_missing_verifier_gated; "
            "fr11_self_learning_status=clean_controller_only_zero_mistake_budget; "
            "ebt_arm_status=projection_only_sidecar_schema_no_model_integration; "
            "gatemate_status=blocked_no_rerun_operator_actions_required; "
            "ssqa_status=gated_skipped_host_visible_smoke_missing"
        ),
    }
    if publication_blocker_count is not None:
        payload["publication_blocker_count"] = publication_blocker_count
    return payload


def _roadmap_yaml(
    *,
    milestone: str = "2026.05.289",
    milestone_doc: str = "openspec/change-proposals/research-roadmap-vNEXT.md",
    include_task: bool = True,
) -> str:
    tasks = (
        "tasks:\n"
        "  - id: exp3095-archive-v288-activate-v289\n"
        "    deliverable: results/experiment_3095_archive_v288_activate_v289.json\n"
        if include_task
        else "tasks: []\n"
    )
    return (
        f'milestone: "{milestone}"\n'
        'milestone_title: "Verifier/Repair Recovery + MaxSAT Routing"\n'
        f'milestone_doc: "{milestone_doc}"\n'
        f"{tasks}"
    )


def _write_common_sources(
    root: Path,
    *,
    staged_roadmap: bool = True,
    active_milestone: str = "2026.05.289",
    capstone_ready: bool = True,
    paper_ready: bool = False,
) -> None:
    _write_json(root, mod.MATRIX_V22_REL_PATH, _matrix_payload())
    _write_json(
        root,
        mod.CAPSTONE_V288_REL_PATH,
        _capstone_payload(capstone_ready=capstone_ready, paper_ready=paper_ready),
    )
    if staged_roadmap:
        _write_text(root, mod.STAGED_ROADMAP_REL_PATH, _roadmap_yaml())
    _write_text(root, mod.ACTIVE_ROADMAP_REL_PATH, _roadmap_yaml(milestone=active_milestone))
    _write_text(root, mod.VNEXT_DOC_REL_PATH, "# Research Roadmap vNEXT - Milestone 2026.05.289\n")
    _write_text(root, mod.CONDUCTOR_REL_PATH, "# protected conductor\n")
    _write_text(root, mod.CODEX_REL_PATH, "# codex instructions\n")
    _write_text(root, mod.CLAUDE_REL_PATH, "# claude instructions\n")
    _write_text(root, mod.OPS_STATUS_REL_PATH, "status mentions exp3094 and .289\n")
    _write_text(root, mod.OPS_CHANGELOG_REL_PATH, "changelog mentions exp3094 and .289\n")
    _write_text(root, mod.TRACEABILITY_REL_PATH, "traceability mentions REQ-REPORT-3094\n")


def test_req_report_3095_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3095: OpenSpec declares the archive/handoff contract first."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3095" in spec
    assert "SCENARIO-REPORT-3095" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_report_3095_builds_ready_archive_from_staged_roadmap(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3095: staged .289 roadmap can drive a ready handoff."""

    _write_common_sources(tmp_path, staged_roadmap=True, active_milestone="2026.05.288")

    artifact = mod.build_artifact(tmp_path, started_s=5.0, now_s=6.25)
    blockers = {row["blocker_id"]: row for row in artifact["carry_forward_blockers"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["archive_v288_activate_v289_ready"] is True
    assert artifact["prior_capstone_ready"] is True
    assert artifact["prior_paper_ready"] is False
    assert artifact["next_milestone"] == "2026.05.289"
    assert artifact["duration_s"] == pytest.approx(1.25)
    assert artifact["honest_verdict"].startswith("complete: archive_v288_activate_v289_ready=true")

    assert artifact["status_summary_288"] == {
        "paper_ready": False,
        "capstone_ready": True,
        "matrix_v22_ready": True,
        "verifier_gain_status": "flagged_or_gated_verifier_gain_recovery_incomplete",
        "repair_claim_status": "bounded_flagged_gated_missing_verifier_gated",
        "fr11_self_learning_status": "clean_controller_only_zero_mistake_budget",
        "ebt_arm_status": "projection_only_sidecar_schema_no_model_integration",
        "gatemate_status": "blocked_no_rerun_operator_actions_required",
        "ssqa_status": "gated_skipped_host_visible_smoke_missing",
        "publication_blocker_count": 36,
        "publication_blocker_count_source": "capstone_publication_blocker_count",
        "missing_capstone_input_artifacts": [
            {
                "path": "results/experiment_3089_gated_xgrammar_sota_repair_micro_panel_v2.json",
                "reason": "named by matrix v22 capstone_input_artifacts but not readable",
            }
        ],
        "headline_model_spec_gaps": _capstone_payload()["headline_model_spec_gaps"],
    }
    assert set(blockers) == {
        "publication_blockers_36",
        "verifier_repair_blockers_17",
        "missing_repair_micro_panel",
        "local_sota_model_spec_gaps",
        "ebt_arm_projection_only",
        "gatemate_ssqa_missing_operator_evidence",
    }
    assert blockers["publication_blockers_36"]["value"] == 36
    assert blockers["verifier_repair_blockers_17"]["value"] == 17
    assert blockers["missing_repair_micro_panel"]["value"] == [
        "results/experiment_3089_gated_xgrammar_sota_repair_micro_panel_v2.json"
    ]
    assert blockers["local_sota_model_spec_gaps"]["value"] == [
        "dot288:exp3085_icalm_task_abstention_sota_panel",
        "dot288:exp3086_dafny_z3_formal_feedback_pilot",
    ]
    assert blockers["ebt_arm_projection_only"]["value"] == (
        "projection_only_sidecar_schema_no_model_integration"
    )
    assert blockers["gatemate_ssqa_missing_operator_evidence"]["value"] == {
        "gatemate_status": "blocked_no_rerun_operator_actions_required",
        "ssqa_status": "gated_skipped_host_visible_smoke_missing",
    }
    assert all(row["matches_expected"] is True for row in blockers.values())
    assert artifact["roadmap_handoff"]["source_path"] == mod.STAGED_ROADMAP_REL_PATH.as_posix()
    assert artifact["roadmap_handoff"]["requested_staged_roadmap_present"] is True
    assert artifact["roadmap_handoff"]["used_active_roadmap_fallback"] is False
    assert artifact["roadmap_handoff"]["milestone_doc_matches"] is True
    assert artifact["roadmap_handoff"]["task_ids"] == ["exp3095-archive-v288-activate-v289"]
    assert artifact["activation_performed_by_this_task"] is False
    assert artifact["research_roadmap_yaml_modified"] is False
    assert artifact["scripts_research_conductor_modified"] is False
    assert artifact["ops_status_updated"] is False
    assert artifact["ops_changelog_updated"] is False
    assert artifact["traceability_updated"] is False
    assert artifact["inference_substrate"] == {
        "kind": "aggregation_from_upstream_artifacts",
        "executes_models": False,
        "executes_hardware": False,
        "executes_conductor": False,
        "executes_live_repair": False,
        "local_repo_only": True,
        "no_live_llm_inference": True,
        "source": "checked_in_artifacts",
    }


def test_req_report_3095_uses_active_roadmap_fallback_after_activation(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-3095: post-activation active roadmap is audited read-only."""

    _write_common_sources(tmp_path, staged_roadmap=False)

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.5)

    assert artifact["archive_v288_activate_v289_ready"] is True
    assert artifact["roadmap_handoff"]["source_path"] == mod.ACTIVE_ROADMAP_REL_PATH.as_posix()
    assert artifact["roadmap_handoff"]["requested_staged_roadmap_present"] is False
    assert artifact["roadmap_handoff"]["used_active_roadmap_fallback"] is True
    assert artifact["roadmap_handoff"]["active_roadmap_milestone"] == "2026.05.289"
    assert mod.STAGED_ROADMAP_REL_PATH.as_posix() in artifact["missing_source_artifacts"]


def test_req_report_3095_blocks_when_capstone_is_not_ready(tmp_path: Path) -> None:
    """REQ-REPORT-3095: prior capstone readiness is a hard precondition."""

    _write_common_sources(tmp_path, capstone_ready=False)

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.25)

    assert artifact["archive_v288_activate_v289_ready"] is False
    assert artifact["prior_capstone_ready"] is False
    assert artifact["honest_verdict"].startswith("blocked_prior_capstone_not_ready:")
    assert "prior capstone is not capstone_ready=true" in artifact["blocked_reasons"]


def test_req_report_3095_blocks_when_roadmap_handoff_is_invalid(tmp_path: Path) -> None:
    """REQ-REPORT-3095: roadmap milestone/doc/tasks must match the .289 handoff."""

    _write_common_sources(tmp_path, staged_roadmap=False, active_milestone="2026.05.288")

    artifact = mod.build_artifact(tmp_path, started_s=2.0, now_s=2.25)

    assert artifact["archive_v288_activate_v289_ready"] is False
    assert artifact["honest_verdict"].startswith("blocked_roadmap_handoff_not_ready:")
    assert "roadmap milestone is not 2026.05.289" in artifact["blocked_reasons"]


def test_req_report_3095_write_artifact_persists_stable_json(tmp_path: Path) -> None:
    """REQ-REPORT-3095: write_artifact emits the deliverable JSON."""

    _write_common_sources(tmp_path)

    output = mod.write_artifact(tmp_path, started_s=3.0, now_s=3.125)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["archive_v288_activate_v289_ready"] is True
    assert saved["duration_s"] == pytest.approx(0.125)
    checksums = {row["path"]: row["sha256"] for row in saved["source_artifacts"]}
    assert checksums[mod.CAPSTONE_V288_REL_PATH.as_posix()] == _sha256(
        tmp_path / mod.CAPSTONE_V288_REL_PATH
    )


def test_req_report_3095_helper_edges_remain_honest(tmp_path: Path) -> None:
    """REQ-REPORT-3095: helper edges preserve malformed and unusual states."""

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
    assert mod._publication_blocker_count(
        {"publication_blocker_count": 9},
        {},
    ) == (9, "capstone_publication_blocker_count")
    assert mod._publication_blocker_count(
        {"publication_blockers": [1, 2]},
        {},
    ) == (2, "capstone_publication_blockers_length")
    assert mod._publication_blocker_count(
        {"matrix_v22_summary": {"publication_blocker_count": 8}},
        {},
    ) == (8, "capstone_matrix_v22_summary")
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
    assert mod._verifier_repair_blocker_count(
        {"prd_gap_summary": {"verifier_repair": {"publication_blocker_count": 6}}},
    ) == 6
    assert mod._verifier_repair_blocker_count(
        {"next_milestone_recommendation": "17 blocker rows remain"},
    ) == 17
    assert mod._verifier_repair_blocker_count({}) == 0
    assert mod._task_ids({"tasks": [{"id": "a"}, {"no": "id"}, "bad"]}) == ["a"]
    assert mod._task_ids({"tasks": "bad"}) == []
    assert mod._source_artifact(tmp_path, "missing", Path("missing.txt"))["present"] is False
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
        "roadmap milestone is not 2026.05.289",
        "roadmap milestone_doc is not openspec/change-proposals/research-roadmap-vNEXT.md",
        "roadmap has no tasks",
        "openspec/change-proposals/research-roadmap-vNEXT.md is missing",
    ]
