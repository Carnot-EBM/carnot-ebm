"""Tests for Exp 3096 publication blocker triage and retirement ledger.

Spec refs: REQ-REPORT-3096, SCENARIO-REPORT-3096.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import publication_blocker_triage_retirement_ledger_3096 as mod


REQUIRED_FIELDS = {
    "blocker_triage_ready",
    "publication_blocker_count_before",
    "blocker_categories",
    "reducible_in_v289",
    "operator_evidence_required",
    "retire_or_promote_criteria",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}
CATEGORY_NAMES = {
    "verifier_repair",
    "formal_feedback",
    "repair_missing_artifact",
    "fr11_boundary",
    "ebt_arm_projection",
    "hardware_evidence",
    "publication_readiness",
    "model_spec_gap",
    "missing_artifact",
    "bounded_status",
    "retired_status",
}


def _write_json(root: Path, rel_path: Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row(row_id: str, status: str, claim_scope: str, evidence_class: str) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "status": status,
        "source_artifact": f"results/{row_id.replace(':', '_')}.json",
        "source_field": "status",
        "evidence_class": evidence_class,
        "blocker_class": mod.blocker_class(status),
        "claim_scope": claim_scope,
        "summary": {"claim_scope": claim_scope, "evidence_class": evidence_class},
    }


def _matrix_payload() -> dict[str, Any]:
    rows = [
        _row("clean-row", "clean", "milestone_activation", "archive"),
        _row("verifier-row", "flagged", "local_sota_solution_verifier_gain", "panel"),
        _row("formal-row", "flagged", "solver_grounded_repair_feedback", "formal_feedback"),
        _row("repair-missing-row", "missing", "repair_live_rerun", "repair_micro_panel"),
        _row("fr11-row", "flagged", "controller_only_online_learning_budget", "fr11"),
        _row("projection-row", "projection_only", "future_adapter_context", "ebt_arm"),
        _row("hardware-row", "blocked", "hardware_rerun_gate", "gatemate"),
        _row("publication-row", "bounded", "paper_readiness", "capstone"),
        _row("model-gap-row", "flagged", "local_sota_solution_verifier_gain", "live_llm"),
        _row("missing-row", "missing", "prior_v18_carry_forward", "capstone"),
        _row("bounded-row", "bounded", "repair_headline_boundary", "repair_bounded_claim"),
        _row("retired-row", "retired", "retired_repair_claim", "repair"),
    ]
    blockers = [
        {
            "row_id": row["row_id"],
            "status": row["status"],
            "source_artifact": row["source_artifact"],
            "source_field": row["source_field"],
            "claim_scope": row["claim_scope"],
            "blocker_class": row["blocker_class"],
        }
        for row in rows
        if row["status"] not in {"clean", "retired"}
    ]
    return {
        "artifact": "experiment_3093_cross_corpus_matrix_v22",
        "matrix_v22_ready": True,
        "rows_total": len(rows),
        "publication_blocker_count": len(blockers),
        "publication_blockers": blockers,
        "headline_model_spec_gaps": [
            {
                "row_id": "model-gap-row",
                "source_artifact": "results/model-gap-row.json",
                "missing_model_ids": ["model-b"],
                "present_model_ids": ["model-a"],
                "reason": "mandated model_specs missing for live LLM artifact",
            }
        ],
        "missing_artifacts": [
            {
                "path": "results/repair-missing-row.json",
                "reason": "expected repair micro-panel artifact is absent",
            }
        ],
        "rows": rows,
        "honest_verdict": "complete: matrix_v22_ready=true",
    }


def _capstone_payload(blocker_count: int) -> dict[str, Any]:
    return {
        "artifact": "experiment_3094_capstone_v288",
        "capstone_ready": True,
        "paper_ready": False,
        "publication_blocker_count": blocker_count,
        "headline_model_spec_gaps": [
            {
                "row_id": "model-gap-row",
                "source_artifact": "results/model-gap-row.json",
                "missing_model_ids": ["model-b"],
                "present_model_ids": ["model-a"],
                "reason": "mandated model_specs missing for live LLM artifact",
            }
        ],
        "honest_verdict": "complete: capstone_ready=true; paper_ready=false",
    }


def _ids(rows: list[dict[str, Any]]) -> list[str]:
    return [str(row["row_id"]) for row in rows]


def test_req_report_3096_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3096: OpenSpec declares the triage ledger contract first."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3096" in spec
    assert "SCENARIO-REPORT-3096" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_report_3096_builds_v23_consumable_triage_ledger(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3096: .289 actions are separated from operator evidence."""

    matrix = _matrix_payload()
    _write_json(tmp_path, mod.MATRIX_V22_REL_PATH, matrix)
    _write_json(tmp_path, mod.CAPSTONE_V288_REL_PATH, _capstone_payload(10))

    artifact = mod.build_artifact(tmp_path, started_s=2.0, now_s=5.5)
    categories = artifact["blocker_categories"]

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["blocker_triage_ready"] is True
    assert artifact["publication_blocker_count_before"] == 10
    assert artifact["duration_s"] == pytest.approx(3.5)
    assert artifact["honest_verdict"].startswith("complete:")
    assert set(categories) == CATEGORY_NAMES
    assert _ids(categories["verifier_repair"]) == ["verifier-row"]
    assert _ids(categories["formal_feedback"]) == ["formal-row"]
    assert _ids(categories["repair_missing_artifact"]) == ["repair-missing-row"]
    assert _ids(categories["fr11_boundary"]) == ["fr11-row"]
    assert _ids(categories["ebt_arm_projection"]) == ["projection-row"]
    assert _ids(categories["hardware_evidence"]) == ["hardware-row"]
    assert _ids(categories["publication_readiness"]) == ["publication-row"]
    assert _ids(categories["model_spec_gap"]) == ["model-gap-row"]
    assert _ids(categories["missing_artifact"]) == ["missing-row"]
    assert _ids(categories["bounded_status"]) == ["bounded-row"]
    assert _ids(categories["retired_status"]) == ["retired-row"]

    coverage = artifact["blocker_coverage"]
    assert coverage["covered_publication_blocker_count"] == 10
    assert coverage["uncategorized_publication_blocker_ids"] == []
    assert coverage["duplicate_publication_blocker_ids"] == []
    assert coverage["retired_row_count"] == 1

    reducible_ids = _ids(artifact["reducible_in_v289"])
    operator_ids = _ids(artifact["operator_evidence_required"])
    assert "verifier-row" in reducible_ids
    assert "formal-row" in reducible_ids
    assert "repair-missing-row" in reducible_ids
    assert "model-gap-row" in reducible_ids
    assert "hardware-row" not in reducible_ids
    assert operator_ids == ["hardware-row"]

    assert CATEGORY_NAMES <= set(artifact["retire_or_promote_criteria"])
    assert "operator evidence" in artifact["retire_or_promote_criteria"]["hardware_evidence"]
    assert "model_specs" in artifact["retire_or_promote_criteria"]["model_spec_gap"]
    assert artifact["matrix_v23_consumption"]["publication_blocker_count_authority"] == (
        "matrix_v22.publication_blocker_count"
    )

    source_by_path = {row["path"]: row for row in artifact["source_artifacts"]}
    assert source_by_path[mod.MATRIX_V22_REL_PATH.as_posix()]["sha256"] == _sha256(
        tmp_path / mod.MATRIX_V22_REL_PATH
    )
    assert source_by_path[mod.CAPSTONE_V288_REL_PATH.as_posix()]["sha256"] == _sha256(
        tmp_path / mod.CAPSTONE_V288_REL_PATH
    )
    assert artifact["inference_substrate"] == {
        "kind": "aggregation_from_checked_in_artifacts",
        "source": "matrix_v22_and_capstone_v288",
        "executes_models": False,
        "executes_hardware": False,
        "executes_conductor": False,
        "executes_live_repair": False,
        "no_live_llm_inference": True,
        "local_repo_only": True,
    }
    assert artifact["no_new_model_execution"] is True
    assert artifact["scripts_research_conductor_modified"] is False
    assert artifact["status_updates_written"] is False


def test_req_report_3096_blocks_missing_or_inconsistent_authorities(tmp_path: Path) -> None:
    """REQ-REPORT-3096: missing or inconsistent authorities fail closed."""

    missing = mod.build_artifact(tmp_path)
    assert missing["blocker_triage_ready"] is False
    assert missing["honest_verdict"] == "blocked_required_matrix_v22_missing"

    matrix = _matrix_payload()
    _write_json(tmp_path, mod.MATRIX_V22_REL_PATH, matrix)
    _write_json(tmp_path, mod.CAPSTONE_V288_REL_PATH, _capstone_payload(99))

    inconsistent = mod.build_artifact(tmp_path)
    assert inconsistent["blocker_triage_ready"] is False
    assert inconsistent["blocked_reasons"] == ["matrix v22 and capstone .288 blocker counts disagree"]
    assert inconsistent["honest_verdict"].startswith("blocked_triage_preconditions")


def test_req_report_3096_write_artifact_and_helper_edges(tmp_path: Path) -> None:
    """REQ-REPORT-3096: helper behavior is deterministic at triage edges."""

    matrix = _matrix_payload()
    _write_json(tmp_path, mod.MATRIX_V22_REL_PATH, matrix)
    _write_json(tmp_path, mod.CAPSTONE_V288_REL_PATH, _capstone_payload(10))
    malformed = tmp_path / "bad.json"
    list_json = tmp_path / "list.json"
    malformed.write_text("{bad-json}\n", encoding="utf-8")
    list_json.write_text("[1, 2]\n", encoding="utf-8")

    output = mod.write_artifact(tmp_path, started_s=1.0, now_s=1.75)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["blocker_triage_ready"] is True
    assert saved["duration_s"] == pytest.approx(0.75)
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    assert mod.read_json_object(malformed) == {}
    assert mod.read_json_object(list_json) == {}
    assert mod.sha256_file(tmp_path / "missing.txt") is None
    assert mod.normal_status("gate-skipped") == "gated_skipped"
    assert mod.normal_status("gate_skipped") == "gated_skipped"
    assert mod.normal_status("pilot_only") == "bounded"
    assert mod.normal_status("bad") == "missing"
    assert mod.blocker_class("projection_only") == "projection_only"
    assert mod._as_mapping({"x": 1}) == {"x": 1}
    assert mod._as_mapping(None) == {}
    assert mod._as_list([1]) == [1]
    assert mod._as_list("x") == []
    assert mod._int_or_none(True) is None
    assert mod._int_or_none("bad") is None
    assert mod._row_text({"summary": {"non-json": {1}}}).find("non-json") >= 0

    assert mod._category_for_row(
        {"status": "retired", "row_id": "old"}, set()
    ) == "retired_status"
    assert mod._category_for_row(
        {"status": "projection_only", "row_id": "projection"}, set()
    ) == "ebt_arm_projection"
    assert mod._category_for_row(
        {"status": "missing", "claim_scope": "repair_live_rerun"}, set()
    ) == "repair_missing_artifact"
    assert mod._category_for_row(
        {"status": "missing", "claim_scope": "paper_readiness"}, set()
    ) == "publication_readiness"
    assert mod._category_for_row(
        {"status": "missing", "claim_scope": "host_visible_hardware_transcript"}, set()
    ) == "hardware_evidence"
    assert mod._category_for_row(
        {"status": "missing", "row_id": "generic"}, set()
    ) == "missing_artifact"
    assert mod._category_for_row(
        {
            "status": "bounded",
            "claim_scope": "paper_readiness",
            "summary": {"gatemate_status": "blocked"},
        },
        set(),
    ) == "publication_readiness"
    assert mod._row_primary_text({"row_id": "x", "summary": {"gatemate": "blocked"}}) == (
        "x     "
    )

    reasons = mod._blocked_reasons(
        matrix={"matrix_v22_ready": False},
        capstone={"capstone_ready": False},
        before_count=1,
        capstone_count=1,
        coverage={
            "uncategorized_publication_blocker_ids": ["x"],
            "duplicate_publication_blocker_ids": ["y"],
            "retired_row_count": 0,
        },
        criteria={},
    )
    assert reasons == [
        "matrix v22 is not ready",
        "capstone .288 is not ready",
        "one or more matrix v22 blockers were not categorized",
        "one or more matrix v22 blockers were categorized more than once",
        "one or more categories lack retire-or-promote criteria",
    ]
