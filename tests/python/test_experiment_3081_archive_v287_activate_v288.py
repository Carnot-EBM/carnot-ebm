"""Tests for Exp 3081 archive .287 and .288 handoff.

Spec refs: REQ-REPORT-3081, SCENARIO-REPORT-3081.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import archive_v287_activate_v288_3081 as mod


REQUIRED_FIELDS = {
    "archive_v287_activate_v288_ready",
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
    blockers = [
        {
            "row_id": f"blocker-{idx}",
            "status": "flagged",
            "source_artifact": "results/source.json",
            "source_field": f"rows[{idx}]",
            "blocker_class": "test_blocker",
            "claim_scope": "test",
        }
        for idx in range(42)
    ]
    return {
        "schema": "carnot.cross_corpus_matrix.v21_287_claim_aggregation.v1",
        "milestone": "2026.05.287",
        "matrix_v21_ready": True,
        "rows_total": 61,
        "clean_rows": 14,
        "flagged_rows": 12,
        "bounded_rows": 12,
        "blocked_rows": 5,
        "gated_skipped_rows": 9,
        "projection_only_rows": 2,
        "missing_rows": 2,
        "retired_rows": 5,
        "publication_blocker_count": 42,
        "publication_blockers": blockers,
        "honest_verdict": (
            "complete: matrix_v21_ready=true; rows_total=61; "
            "publication_blocker_count=42; paper_ready=false"
        ),
    }


def _capstone_payload(
    *,
    capstone_ready: bool = True,
    paper_ready: bool = False,
    publication_blocker_count: int | None = 42,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "carnot.milestone_capstone.v287_matrix_v21_aggregation.v1",
        "milestone": "2026.05.287",
        "capstone_ready": capstone_ready,
        "paper_ready": paper_ready,
        "verifier_gain_status": "flagged_or_gated_verifier_gain_recovery_incomplete",
        "repair_claim_status": "bounded_and_gated_skipped",
        "fr11_self_learning_status": "flagged_controller_only_budget_exceeded",
        "gatemate_status": "blocked_no_rerun_operator_actions_required",
        "ssqa_status": "gated_skipped_host_visible_smoke_missing",
        "ebt_arm_status": "projection_only_feasible_no_implementation",
        "publication_blockers": [{"row_id": f"blocker-{idx}"} for idx in range(42)],
        "matrix_v21_summary": {
            "matrix_v21_ready": True,
            "rows_total": 61,
            "publication_blocker_count": 42,
            "status_counts": {
                "clean": 14,
                "flagged": 12,
                "bounded": 12,
                "blocked": 5,
                "gated_skipped": 9,
                "projection_only": 2,
                "missing": 2,
                "retired": 5,
            },
        },
        "next_milestone_recommendation": (
            "2026.05.288: raise abstention_precision to the gate; run Exp3072 "
            "calibration and Exp3075 repair only after gates pass; keep FR-11 "
            "controller-only until the completeness budget is zero; keep "
            "EBT/ARM-EBM projection-only until an adapter implementation has tests; "
            "commit GateMate output contract and host-visible smoke transcript."
        ),
        "honest_verdict": (
            "complete: capstone_ready=true; paper_ready=false; matrix_v21_ready=true; "
            "publication_blocker_count=42; "
            "verifier_gain_status=flagged_or_gated_verifier_gain_recovery_incomplete; "
            "repair_claim_status=bounded_and_gated_skipped; "
            "fr11_self_learning_status=flagged_controller_only_budget_exceeded; "
            "gatemate_status=blocked_no_rerun_operator_actions_required; "
            "ssqa_status=gated_skipped_host_visible_smoke_missing"
        ),
    }
    if publication_blocker_count is not None:
        payload["publication_blocker_count"] = publication_blocker_count
    return payload


def _roadmap_yaml(
    *,
    milestone: str = "2026.05.288",
    milestone_doc: str = "openspec/change-proposals/research-roadmap-vNEXT.md",
    include_task: bool = True,
) -> str:
    tasks = (
        "tasks:\n"
        "  - id: exp3081-archive-v287-activate-v288\n"
        "    deliverable: results/experiment_3081_archive_v287_activate_v288.json\n"
        if include_task
        else "tasks: []\n"
    )
    return (
        f'milestone: "{milestone}"\n'
        'milestone_title: "Abstention-Calibrated Verifier Recovery"\n'
        f'milestone_doc: "{milestone_doc}"\n'
        f"{tasks}"
    )


def _write_common_sources(
    root: Path,
    *,
    staged_roadmap: bool = True,
    active_milestone: str = "2026.05.288",
    capstone_ready: bool = True,
    paper_ready: bool = False,
) -> None:
    _write_json(root, mod.MATRIX_V21_REL_PATH, _matrix_payload())
    _write_json(
        root,
        mod.CAPSTONE_V287_REL_PATH,
        _capstone_payload(capstone_ready=capstone_ready, paper_ready=paper_ready),
    )
    if staged_roadmap:
        _write_text(root, mod.STAGED_ROADMAP_REL_PATH, _roadmap_yaml())
    _write_text(root, mod.ACTIVE_ROADMAP_REL_PATH, _roadmap_yaml(milestone=active_milestone))
    _write_text(root, mod.VNEXT_DOC_REL_PATH, "# Research Roadmap vNEXT - Milestone 2026.05.288\n")
    _write_text(root, mod.CONDUCTOR_REL_PATH, "# protected conductor\n")
    _write_text(root, mod.OPS_STATUS_REL_PATH, "status mentions exp3080 and .288\n")
    _write_text(root, mod.OPS_CHANGELOG_REL_PATH, "changelog mentions exp3080 and .288\n")


def test_req_report_3081_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3081: OpenSpec declares the archive/handoff contract first."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3081" in spec
    assert "SCENARIO-REPORT-3081" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_report_3081_builds_ready_archive_from_staged_roadmap(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3081: staged .288 roadmap can drive a ready handoff."""

    _write_common_sources(tmp_path, staged_roadmap=True, active_milestone="2026.05.287")

    artifact = mod.build_artifact(tmp_path, started_s=5.0, now_s=6.5)
    blockers = {row["blocker_id"]: row for row in artifact["carry_forward_blockers"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["archive_v287_activate_v288_ready"] is True
    assert artifact["prior_capstone_ready"] is True
    assert artifact["prior_paper_ready"] is False
    assert artifact["next_milestone"] == "2026.05.288"
    assert artifact["duration_s"] == pytest.approx(1.5)
    assert artifact["honest_verdict"].startswith("complete: archive_v287_activate_v288_ready=true")

    assert artifact["status_summary_287"] == {
        "paper_ready": False,
        "capstone_ready": True,
        "matrix_v21_ready": True,
        "verifier_gain_status": "flagged_or_gated_verifier_gain_recovery_incomplete",
        "repair_claim_status": "bounded_and_gated_skipped",
        "fr11_self_learning_status": "flagged_controller_only_budget_exceeded",
        "gatemate_status": "blocked_no_rerun_operator_actions_required",
        "ssqa_status": "gated_skipped_host_visible_smoke_missing",
        "ebt_arm_status": "projection_only_feasible_no_implementation",
        "publication_blocker_count": 42,
        "publication_blocker_count_source": "capstone_publication_blocker_count",
        "counts": {
            "clean": 14,
            "flagged": 12,
            "bounded": 12,
            "blocked": 5,
            "gated_skipped": 9,
            "projection_only": 2,
            "missing": 2,
            "retired": 5,
        },
    }
    assert set(blockers) == {
        "low_abstention_precision",
        "gated_calibration_repair",
        "fr11_completeness_mistake",
        "ebt_arm_projection_only",
        "gatemate_ssqa_missing_operator_evidence",
        "publication_blockers_42",
    }
    assert blockers["low_abstention_precision"]["source_field"] == "next_milestone_recommendation"
    assert blockers["gated_calibration_repair"]["value"] == "bounded_and_gated_skipped"
    assert blockers["fr11_completeness_mistake"]["value"] == (
        "flagged_controller_only_budget_exceeded"
    )
    assert blockers["ebt_arm_projection_only"]["value"] == (
        "projection_only_feasible_no_implementation"
    )
    assert blockers["gatemate_ssqa_missing_operator_evidence"]["value"] == {
        "gatemate_status": "blocked_no_rerun_operator_actions_required",
        "ssqa_status": "gated_skipped_host_visible_smoke_missing",
    }
    assert blockers["publication_blockers_42"]["value"] == 42
    assert all(row["matches_expected"] is True for row in blockers.values())
    assert artifact["roadmap_handoff"]["source_path"] == mod.STAGED_ROADMAP_REL_PATH.as_posix()
    assert artifact["roadmap_handoff"]["requested_staged_roadmap_present"] is True
    assert artifact["roadmap_handoff"]["used_active_roadmap_fallback"] is False
    assert artifact["roadmap_handoff"]["milestone_doc_matches"] is True
    assert artifact["roadmap_handoff"]["task_ids"] == ["exp3081-archive-v287-activate-v288"]
    assert artifact["activation_performed_by_this_task"] is False
    assert artifact["research_roadmap_yaml_modified"] is False
    assert artifact["scripts_research_conductor_modified"] is False
    assert artifact["inference_substrate"] == {
        "kind": "aggregation_from_upstream_artifacts",
        "executes_models": False,
        "executes_hardware": False,
        "executes_conductor": False,
        "no_live_llm_inference": True,
        "source": "checked_in_artifacts",
    }


def test_req_report_3081_uses_active_roadmap_fallback_after_activation(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-3081: post-activation active roadmap is audited read-only."""

    _write_common_sources(tmp_path, staged_roadmap=False)

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.5)

    assert artifact["archive_v287_activate_v288_ready"] is True
    assert artifact["roadmap_handoff"]["source_path"] == mod.ACTIVE_ROADMAP_REL_PATH.as_posix()
    assert artifact["roadmap_handoff"]["requested_staged_roadmap_present"] is False
    assert artifact["roadmap_handoff"]["used_active_roadmap_fallback"] is True
    assert artifact["roadmap_handoff"]["active_roadmap_milestone"] == "2026.05.288"
    assert mod.STAGED_ROADMAP_REL_PATH.as_posix() in artifact["missing_source_artifacts"]


def test_req_report_3081_blocks_when_capstone_is_not_ready(tmp_path: Path) -> None:
    """REQ-REPORT-3081: prior capstone readiness is a hard precondition."""

    _write_common_sources(tmp_path, capstone_ready=False)

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.25)

    assert artifact["archive_v287_activate_v288_ready"] is False
    assert artifact["prior_capstone_ready"] is False
    assert artifact["honest_verdict"].startswith("blocked_prior_capstone_not_ready:")
    assert "prior capstone is not capstone_ready=true" in artifact["blocked_reasons"]


def test_req_report_3081_blocks_when_roadmap_handoff_is_invalid(tmp_path: Path) -> None:
    """REQ-REPORT-3081: roadmap milestone/doc/tasks must match the .288 handoff."""

    _write_common_sources(tmp_path, staged_roadmap=False, active_milestone="2026.05.287")

    artifact = mod.build_artifact(tmp_path, started_s=2.0, now_s=2.25)

    assert artifact["archive_v287_activate_v288_ready"] is False
    assert artifact["honest_verdict"].startswith("blocked_roadmap_handoff_not_ready:")
    assert "roadmap milestone is not 2026.05.288" in artifact["blocked_reasons"]


def test_req_report_3081_write_artifact_persists_stable_json(tmp_path: Path) -> None:
    """REQ-REPORT-3081: write_artifact emits the deliverable JSON."""

    _write_common_sources(tmp_path)

    output = mod.write_artifact(tmp_path, started_s=3.0, now_s=3.125)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["archive_v287_activate_v288_ready"] is True
    assert saved["duration_s"] == pytest.approx(0.125)
    checksums = {row["path"]: row["sha256"] for row in saved["source_artifacts"]}
    assert checksums[mod.CAPSTONE_V287_REL_PATH.as_posix()] == _sha256(
        tmp_path / mod.CAPSTONE_V287_REL_PATH
    )


def test_req_report_3081_helper_edges_remain_honest(tmp_path: Path) -> None:
    """REQ-REPORT-3081: helper edges preserve malformed and unusual states."""

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
        {"matrix_v21_summary": {"publication_blocker_count": 8}},
        {},
    ) == (8, "capstone_matrix_v21_summary")
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
    assert mod._count_summary({}, {}) == {status: 0 for status in mod.COUNT_FIELDS}
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
        "roadmap milestone is not 2026.05.288",
        "roadmap milestone_doc is not openspec/change-proposals/research-roadmap-vNEXT.md",
        "roadmap has no tasks",
        "openspec/change-proposals/research-roadmap-vNEXT.md is missing",
    ]
