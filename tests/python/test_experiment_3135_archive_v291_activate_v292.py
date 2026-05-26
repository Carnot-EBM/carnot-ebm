"""Tests for Exp 3135 archive .291 and .292 handoff.

Spec refs: REQ-REPORT-3135, SCENARIO-REPORT-3135.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import archive_v291_activate_v292_3135 as mod


REQUIRED_FIELDS = {
    "archive_v291_activate_v292_ready",
    "prior_capstone_ready",
    "prior_paper_ready",
    "prior_publication_blocker_count",
    "blocker_delta_from_v24",
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
        "artifact": "experiment_3133_cross_corpus_matrix_v25",
        "milestone": "2026.05.291",
        "matrix_v25_ready": True,
        "publication_blocker_count": 46,
        "blocker_delta_from_v24": 10,
        "headline_claim_allowance_summary": {
            "cached_sota_pair_available": False,
            "false_accept_rate": 0.5,
            "missing_model_ids": [
                "unsloth/Qwen3.6-35B-A3B-GGUF",
                "unsloth/gemma-4-31B-it-GGUF",
            ],
            "present_model_ids": ["unsloth/gemma-4-26B-A4B-it-GGUF"],
            "repair_gate_state": "blocked_false_accept",
        },
        "verifier_repair_summary": {
            "false_accept_rate": 0.5,
            "repair_gate_state": "blocked_false_accept",
            "verifier_gain_delta": 0.0,
        },
        "fr11_summary": {
            "ledger_consistency_rate": 0.666667,
            "model_weight_learning_allowed": False,
        },
        "architecture_boundary_summary": {
            "arm_ebt_status": "projection_only",
            "kan_pwa_milp_status": "bounded",
            "hardware_sampler_status": "blocked",
            "speedup_claim_allowed": False,
        },
        "source_artifacts": [
            {"path": "results/experiment_3124_difficulty_stratified_live_sota_verifier_panel_v6.json"},
            {"path": "results/experiment_3134_capstone_v291.json"},
        ],
        "honest_verdict": (
            "complete: matrix_v25_ready=true; rows_total=113; "
            "publication_blocker_count=46; blocker_delta_from_v24=10"
        ),
    }


def _capstone_payload(
    *,
    capstone_ready: bool = True,
    paper_ready: bool = False,
    publication_blocker_count: int | None = 46,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "artifact": "experiment_3134_capstone_v291",
        "milestone": "2026.05.291",
        "capstone_ready": capstone_ready,
        "paper_ready": paper_ready,
        "blocker_delta_from_v24": 10,
        "sota_cache_status": "bounded_missing_comparative_sota_pair",
        "live_verifier_status": "blocked",
        "verifier_claim_status": "blocked_false_accept_rate_0.5_no_headline_lift",
        "repair_claim_status": "blocked_repair_ladder_gate_failed_by_live_verifier_gate",
        "repair_ladder_status": "blocked",
        "fr11_self_learning_status": (
            "bounded_controller_environment_memory_only_no_weight_update_ledger_0.666667"
        ),
        "ebt_arm_status": "projection_only_sidecar_diagnostic_no_live_integration",
        "kan_status": "bounded_pwa_milp_abstraction_no_deployed_verifier_claim",
        "sampler_hardware_status": "blocked_hardware_sampler_boundary_no_speedup_claim",
        "next_top_gap": "live_verifier_false_accept_repair_gate",
        "claim_allowance_summary": {
            "cached_sota_pair_available": False,
            "false_accept_rate": 0.5,
            "missing_model_ids": [
                "unsloth/Qwen3.6-35B-A3B-GGUF",
                "unsloth/gemma-4-31B-it-GGUF",
            ],
            "present_model_ids": ["unsloth/gemma-4-26B-A4B-it-GGUF"],
            "repair_gate_state": "blocked_false_accept",
        },
        "matrix_v25_summary": {
            "matrix_v25_ready": True,
            "publication_blocker_count": 46,
            "blocker_delta_from_v24": 10,
            "rows_total": 113,
        },
        "source_artifacts": [
            {"path": mod.MATRIX_V25_REL_PATH.as_posix(), "role": "matrix_v25_authority"},
            {"path": "results/experiment_3124_difficulty_stratified_live_sota_verifier_panel_v6.json"},
        ],
        "honest_verdict": (
            "complete: capstone_ready=true; paper_ready=false; "
            "publication_blocker_count=46; blocker_delta_from_v24=10; "
            "next_top_gap=live_verifier_false_accept_repair_gate"
        ),
    }
    if publication_blocker_count is not None:
        payload["publication_blocker_count"] = publication_blocker_count
    return payload


def _roadmap_yaml(
    *,
    milestone: str = "2026.05.292",
    milestone_doc: str = "openspec/change-proposals/research-roadmap-vNEXT.md",
    include_task: bool = True,
) -> str:
    tasks = (
        "tasks:\n"
        "  - id: exp3135-archive-v291-activate-v292\n"
        "    deliverable: results/experiment_3135_archive_v291_activate_v292.json\n"
        if include_task
        else "tasks: []\n"
    )
    return (
        f'milestone: "{milestone}"\n'
        'milestone_title: "False-Accept Verifier Recovery"\n'
        f'milestone_doc: "{milestone_doc}"\n'
        f"{tasks}"
    )


def _write_common_sources(
    root: Path,
    *,
    staged_roadmap: bool = True,
    active_milestone: str = "2026.05.292",
    capstone_ready: bool = True,
    paper_ready: bool = False,
) -> None:
    _write_json(root, mod.MATRIX_V25_REL_PATH, _matrix_payload())
    _write_json(
        root,
        mod.CAPSTONE_V291_REL_PATH,
        _capstone_payload(capstone_ready=capstone_ready, paper_ready=paper_ready),
    )
    if staged_roadmap:
        _write_text(root, mod.STAGED_ROADMAP_REL_PATH, _roadmap_yaml())
    _write_text(root, mod.ACTIVE_ROADMAP_REL_PATH, _roadmap_yaml(milestone=active_milestone))
    _write_text(root, mod.VNEXT_DOC_REL_PATH, "# Research Roadmap vNEXT - Milestone 2026.05.292\n")
    _write_text(root, mod.AGENTS_REL_PATH, "# repo instructions\n")
    _write_text(root, mod.CODEX_REL_PATH, "# codex instructions\n")
    _write_text(root, mod.CLAUDE_REL_PATH, "# claude instructions\n")
    _write_text(root, mod.CONDUCTOR_REL_PATH, "# protected conductor\n")
    _write_text(root, mod.OPS_STATUS_REL_PATH, "status mentions exp3134 and .292\n")
    _write_text(root, mod.OPS_CHANGELOG_REL_PATH, "changelog mentions exp3134 and .292\n")
    _write_text(root, mod.TRACEABILITY_REL_PATH, "traceability mentions REQ-REPORT-3134\n")


def test_req_report_3135_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3135: OpenSpec declares the archive/handoff contract first."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3135" in spec
    assert "SCENARIO-REPORT-3135" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_report_3135_builds_ready_archive_from_staged_roadmap(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3135: staged .292 roadmap can drive a ready handoff."""

    _write_common_sources(tmp_path, staged_roadmap=True, active_milestone="2026.05.291")

    artifact = mod.build_artifact(tmp_path, started_s=5.0, now_s=6.25)
    blockers = {row["blocker_id"]: row for row in artifact["carry_forward_blockers"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["archive_v291_activate_v292_ready"] is True
    assert artifact["prior_capstone_ready"] is True
    assert artifact["prior_paper_ready"] is False
    assert artifact["prior_publication_blocker_count"] == 46
    assert artifact["blocker_delta_from_v24"] == 10
    assert artifact["next_milestone"] == "2026.05.292"
    assert artifact["duration_s"] == pytest.approx(1.25)
    assert artifact["honest_verdict"].startswith("complete: archive_v291_activate_v292_ready=true")
    assert artifact["status_summary_291"] == {
        "paper_ready": False,
        "capstone_ready": True,
        "matrix_v25_ready": True,
        "publication_blocker_count": 46,
        "publication_blocker_count_source": "capstone_publication_blocker_count",
        "blocker_delta_from_v24": 10,
        "blocker_delta_source": "capstone_blocker_delta_from_v24",
        "next_top_gap": "live_verifier_false_accept_repair_gate",
        "sota_cache_status": "bounded_missing_comparative_sota_pair",
        "live_verifier_status": "blocked",
        "verifier_claim_status": "blocked_false_accept_rate_0.5_no_headline_lift",
        "false_accept_rate": 0.5,
        "verifier_gain_delta": 0.0,
        "repair_gate_status": "blocked_false_accept",
        "repair_claim_status": "blocked_repair_ladder_gate_failed_by_live_verifier_gate",
        "fr11_self_learning_status": (
            "bounded_controller_environment_memory_only_no_weight_update_ledger_0.666667"
        ),
        "fr11_ledger_consistency_rate": 0.666667,
        "ebt_arm_status": "projection_only_sidecar_diagnostic_no_live_integration",
        "kan_status": "bounded_pwa_milp_abstraction_no_deployed_verifier_claim",
        "sampler_hardware_status": "blocked_hardware_sampler_boundary_no_speedup_claim",
        "source_artifacts": _capstone_payload()["source_artifacts"],
    }
    assert set(blockers) == {
        "publication_blockers_46",
        "missing_comparative_sota_pair",
        "false_accept_rate_0_5",
        "zero_verifier_gain",
        "repair_gate_blocked",
        "fr11_ledger_consistency_0_666667",
        "ebt_arm_projection_only",
        "kan_bounded_only",
        "no_authenticated_hardware_speedup",
    }
    assert blockers["publication_blockers_46"]["value"] == 46
    assert blockers["missing_comparative_sota_pair"]["value"] == {
        "missing_model_ids": [
            "unsloth/Qwen3.6-35B-A3B-GGUF",
            "unsloth/gemma-4-31B-it-GGUF",
        ],
        "present_model_ids": ["unsloth/gemma-4-26B-A4B-it-GGUF"],
    }
    assert blockers["false_accept_rate_0_5"]["value"] == 0.5
    assert blockers["zero_verifier_gain"]["value"] == 0.0
    assert blockers["repair_gate_blocked"]["value"] == "blocked_false_accept"
    assert blockers["fr11_ledger_consistency_0_666667"]["value"] == 0.666667
    assert blockers["ebt_arm_projection_only"]["value"] == (
        "projection_only_sidecar_diagnostic_no_live_integration"
    )
    assert blockers["kan_bounded_only"]["value"] == (
        "bounded_pwa_milp_abstraction_no_deployed_verifier_claim"
    )
    assert blockers["no_authenticated_hardware_speedup"]["value"] == (
        "blocked_hardware_sampler_boundary_no_speedup_claim"
    )
    assert all(row["matches_expected"] is True for row in blockers.values())
    assert artifact["roadmap_handoff"]["source_path"] == mod.STAGED_ROADMAP_REL_PATH.as_posix()
    assert artifact["roadmap_handoff"]["requested_staged_roadmap_present"] is True
    assert artifact["roadmap_handoff"]["used_active_roadmap_fallback"] is False
    assert artifact["roadmap_handoff"]["task_ids"] == ["exp3135-archive-v291-activate-v292"]
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


def test_req_report_3135_uses_active_roadmap_fallback_after_activation(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-3135: post-activation active roadmap is audited read-only."""

    _write_common_sources(tmp_path, staged_roadmap=False)

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.5)

    assert artifact["archive_v291_activate_v292_ready"] is True
    assert artifact["roadmap_handoff"]["source_path"] == mod.ACTIVE_ROADMAP_REL_PATH.as_posix()
    assert artifact["roadmap_handoff"]["requested_staged_roadmap_present"] is False
    assert artifact["roadmap_handoff"]["used_active_roadmap_fallback"] is True
    assert artifact["roadmap_handoff"]["active_roadmap_milestone"] == "2026.05.292"
    assert mod.STAGED_ROADMAP_REL_PATH.as_posix() in artifact["missing_source_artifacts"]


def test_req_report_3135_blocks_when_capstone_is_not_ready(tmp_path: Path) -> None:
    """REQ-REPORT-3135: prior capstone readiness is a hard precondition."""

    _write_common_sources(tmp_path, capstone_ready=False)

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.25)

    assert artifact["archive_v291_activate_v292_ready"] is False
    assert artifact["prior_capstone_ready"] is False
    assert artifact["honest_verdict"].startswith("blocked_prior_capstone_not_ready:")
    assert "prior capstone is not capstone_ready=true" in artifact["blocked_reasons"]


def test_req_report_3135_blocks_when_roadmap_handoff_is_invalid(tmp_path: Path) -> None:
    """REQ-REPORT-3135: roadmap milestone/doc/tasks must match the .292 handoff."""

    _write_common_sources(tmp_path, staged_roadmap=False, active_milestone="2026.05.291")

    artifact = mod.build_artifact(tmp_path, started_s=2.0, now_s=2.25)

    assert artifact["archive_v291_activate_v292_ready"] is False
    assert artifact["honest_verdict"].startswith("blocked_roadmap_handoff_not_ready:")
    assert "roadmap milestone is not 2026.05.292" in artifact["blocked_reasons"]


def test_req_report_3135_write_artifact_persists_stable_json(tmp_path: Path) -> None:
    """REQ-REPORT-3135: write_artifact emits the deliverable JSON."""

    _write_common_sources(tmp_path)

    output = mod.write_artifact(tmp_path, started_s=3.0, now_s=3.125)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["archive_v291_activate_v292_ready"] is True
    assert saved["duration_s"] == pytest.approx(0.125)
    checksums = {row["path"]: row["sha256"] for row in saved["source_artifacts"]}
    assert checksums[mod.CAPSTONE_V291_REL_PATH.as_posix()] == _sha256(
        tmp_path / mod.CAPSTONE_V291_REL_PATH
    )


def test_req_report_3135_helper_edges_remain_honest(tmp_path: Path) -> None:
    """REQ-REPORT-3135: helper edges preserve malformed and unusual states."""

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
    assert mod._float_or(True, 0.25) == 0.25
    assert mod._float_or("0.5", 0.25) == 0.5
    assert mod._float_or("x", 0.25) == 0.25
    assert mod._first_int_from_text("publication_blocker_count=12") == 12
    assert mod._first_int_from_text("no count") is None
    assert mod._first_float_from_text("false_accept_rate=0.5") == 0.5
    assert mod._first_float_from_text("no rate") is None
    assert mod._next_top_gap({"next_top_gap": "gap-a"}) == "gap-a"
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
        {"matrix_v25_summary": {"publication_blocker_count": 8}},
        {},
    ) == (8, "capstone_matrix_v25_summary")
    assert mod._publication_blocker_count(
        {"honest_verdict": "complete: publication_blocker_count=4"},
        {},
    ) == (4, "capstone_honest_verdict")
    assert mod._publication_blocker_count(
        {},
        {"honest_verdict": "complete: publication_blocker_count=3"},
    ) == (3, "matrix_honest_verdict")
    assert mod._publication_blocker_count({}, {}) == (0, "missing")
    assert mod._blocker_delta_from_v24({"blocker_delta_from_v24": 2}, {}) == (
        2,
        "capstone_blocker_delta_from_v24",
    )
    assert mod._blocker_delta_from_v24(
        {"matrix_v25_summary": {"blocker_delta_from_v24": 1}},
        {},
    ) == (1, "capstone_matrix_v25_summary")
    assert mod._blocker_delta_from_v24(
        {"honest_verdict": "complete: blocker_delta_from_v24=4"},
        {},
    ) == (4, "capstone_honest_verdict")
    assert mod._blocker_delta_from_v24(
        {},
        {"honest_verdict": "complete: blocker_delta_from_v24=3"},
    ) == (3, "matrix_honest_verdict")
    assert mod._blocker_delta_from_v24({}, {}) == (0, "missing")
    assert mod._status_value("kan_status", {"kan_status": "bounded"}, {}) == "bounded"
    assert mod._task_ids({"tasks": [{"id": "a"}, {"no": "id"}, "bad"]}) == ["a"]
    assert mod._task_ids({"tasks": "bad"}) == []
    assert mod._source_artifact(tmp_path, "missing", Path("missing.txt"))["present"] is False
    assert mod._field_from_summary_or_matrix(
        "ledger_consistency_rate",
        "fr11_summary",
        {},
        {"fr11_summary": {"ledger_consistency_rate": 0.6}},
        0.0,
    ) == 0.6
    assert mod._field_from_summary_or_matrix("x", "summary", {"summary": {"x": 1}}, {}, 0) == 1
    assert mod._field_from_summary_or_matrix("x", "summary", {}, {}, 0) == 0
    assert mod._repair_gate_status({}, {}, {"repair_gate_state": "blocked_h"}, {}) == "blocked_h"
    assert mod._repair_gate_status({}, {}, {}, {"repair_gate_state": "blocked_v"}) == "blocked_v"
    assert mod._repair_gate_status({"repair_ladder_status": "blocked_l"}, {}, {}, {}) == "blocked_l"
    assert mod._model_ids(
        {},
        {
            "headline_claim_allowance_summary": {
                "missing_model_ids": ["missing-a"],
                "present_model_ids": ["present-a"],
            }
        },
    ) == {"missing_model_ids": ["missing-a"], "present_model_ids": ["present-a"]}
    assert mod._blocked_reasons(
        capstone_present=False,
        prior_capstone_ready=False,
        roadmap_handoff={"source_present": False, "milestone_matches": False},
        vnext_doc_present=False,
    ) == [
        "prior capstone artifact missing or malformed",
        "roadmap handoff source is missing",
        "roadmap milestone is not 2026.05.292",
        "roadmap milestone_doc is not openspec/change-proposals/research-roadmap-vNEXT.md",
        "roadmap has no tasks",
        "openspec/change-proposals/research-roadmap-vNEXT.md is missing",
    ]
