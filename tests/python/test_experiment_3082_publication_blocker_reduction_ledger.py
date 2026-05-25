"""Tests for Exp 3082 publication blocker reduction ledger.

Spec refs: REQ-REPORT-3082, SCENARIO-REPORT-3082.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import publication_blocker_reduction_ledger_3082 as mod


REQUIRED_FIELDS = {
    "blocker_ledger_ready",
    "publication_blocker_count_before",
    "blocker_categories",
    "reducible_in_v288",
    "operator_evidence_required",
    "retire_or_promote_criteria",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}
CATEGORY_NAMES = {
    "verifier_gain",
    "repair_gate",
    "fr11_budget",
    "hardware_evidence",
    "adapter_projection",
    "missing_artifact",
    "bounded_status",
    "retired_status",
    "documentation_hygiene",
}


def _write_json(root: Path, rel_path: Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row(
    row_id: str,
    status: str,
    claim_scope: str,
    evidence_class: str,
    source_artifact: str | None = None,
) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "status": status,
        "source_artifact": source_artifact or f"results/{row_id}.json",
        "source_field": "status",
        "evidence_class": evidence_class,
        "blocker_class": mod.blocker_class(status),
        "claim_scope": claim_scope,
        "summary": {"status": status, "claim_scope": claim_scope},
    }


def _matrix_payload() -> dict[str, Any]:
    rows = [
        _row("clean-row", "clean", "milestone_activation", "archive_activation"),
        _row("verifier-row", "flagged", "local_sota_solution_verifier_gain", "panel"),
        _row("repair-row", "gated_skipped", "repair_live_rerun", "repair_gate"),
        _row("fr11-row", "flagged", "controller_only_online_learning_budget", "fr11"),
        _row("hardware-row", "blocked", "hardware_rerun_gate", "gatemate"),
        _row("adapter-row", "projection_only", "future_adapter_context", "ebt_arm"),
        _row("missing-row", "missing", "prior_v18_carry_forward", "capstone"),
        _row("bounded-row", "bounded", "paper_readiness", "capstone"),
        _row("doc-row", "missing", "source_artifact_accounting", "artifact_alias"),
        _row("retired-row", "retired", "retired_repair_headline_wording", "repair"),
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
        "artifact": "experiment_3079_cross_corpus_matrix_v21",
        "matrix_v21_ready": True,
        "rows_total": len(rows),
        "publication_blocker_count": len(blockers),
        "publication_blockers": blockers,
        "rows": rows,
        "source_artifacts": [],
        "honest_verdict": "complete: matrix_v21_ready=true",
    }


def _capstone_payload(blocker_count: int) -> dict[str, Any]:
    return {
        "artifact": "experiment_3080_capstone_v287",
        "capstone_ready": True,
        "paper_ready": False,
        "publication_blocker_count": blocker_count,
        "honest_verdict": "complete: capstone_ready=true; paper_ready=false",
    }


def _ids(rows: list[dict[str, Any]]) -> list[str]:
    return [str(row["row_id"]) for row in rows]


def test_req_report_3082_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3082: OpenSpec declares the blocker ledger contract first."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3082" in spec
    assert "SCENARIO-REPORT-3082" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_report_3082_builds_reduction_ledger(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3082: blockers are categorized for matrix v22 consumption."""

    matrix = _matrix_payload()
    _write_json(tmp_path, mod.MATRIX_V21_REL_PATH, matrix)
    _write_json(tmp_path, mod.CAPSTONE_V287_REL_PATH, _capstone_payload(8))

    artifact = mod.build_artifact(tmp_path, started_s=2.0, now_s=4.5)
    categories = artifact["blocker_categories"]

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["blocker_ledger_ready"] is True
    assert artifact["publication_blocker_count_before"] == 8
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["honest_verdict"].startswith("complete:")
    assert set(categories) == CATEGORY_NAMES
    assert _ids(categories["verifier_gain"]) == ["verifier-row"]
    assert _ids(categories["repair_gate"]) == ["repair-row"]
    assert _ids(categories["fr11_budget"]) == ["fr11-row"]
    assert _ids(categories["hardware_evidence"]) == ["hardware-row"]
    assert _ids(categories["adapter_projection"]) == ["adapter-row"]
    assert _ids(categories["missing_artifact"]) == ["missing-row"]
    assert _ids(categories["bounded_status"]) == ["bounded-row"]
    assert _ids(categories["documentation_hygiene"]) == ["doc-row"]
    assert _ids(categories["retired_status"]) == ["retired-row"]

    coverage = artifact["blocker_coverage"]
    assert coverage["covered_publication_blocker_count"] == 8
    assert coverage["uncategorized_publication_blocker_ids"] == []
    assert coverage["duplicate_publication_blocker_ids"] == []
    assert coverage["retired_row_count"] == 1

    reducible_ids = _ids(artifact["reducible_in_v288"])
    operator_ids = _ids(artifact["operator_evidence_required"])
    assert "verifier-row" in reducible_ids
    assert "repair-row" in reducible_ids
    assert "fr11-row" in reducible_ids
    assert "hardware-row" not in reducible_ids
    assert operator_ids == ["hardware-row"]

    assert CATEGORY_NAMES <= set(artifact["retire_or_promote_criteria"])
    assert "operator evidence" in artifact["retire_or_promote_criteria"]["hardware_evidence"]
    assert "checked-in adapter" in artifact["retire_or_promote_criteria"]["adapter_projection"]

    source_by_path = {row["path"]: row for row in artifact["source_artifacts"]}
    assert source_by_path[mod.MATRIX_V21_REL_PATH.as_posix()]["sha256"] == _sha256(
        tmp_path / mod.MATRIX_V21_REL_PATH
    )
    assert source_by_path[mod.CAPSTONE_V287_REL_PATH.as_posix()]["sha256"] == _sha256(
        tmp_path / mod.CAPSTONE_V287_REL_PATH
    )
    assert artifact["inference_substrate"] == {
        "kind": "aggregation_from_upstream_artifacts",
        "source": "matrix_v21_and_capstone_v287",
        "executes_models": False,
        "executes_hardware": False,
        "executes_conductor": False,
        "executes_live_repair": False,
        "no_live_llm_inference": True,
    }


def test_req_report_3082_blocks_missing_or_inconsistent_authorities(tmp_path: Path) -> None:
    """REQ-REPORT-3082: missing or inconsistent authority artifacts fail closed."""

    missing = mod.build_artifact(tmp_path)
    assert missing["blocker_ledger_ready"] is False
    assert missing["honest_verdict"] == "blocked_required_matrix_v21_missing"

    matrix = _matrix_payload()
    _write_json(tmp_path, mod.MATRIX_V21_REL_PATH, matrix)
    _write_json(tmp_path, mod.CAPSTONE_V287_REL_PATH, _capstone_payload(99))

    inconsistent = mod.build_artifact(tmp_path)
    assert inconsistent["blocker_ledger_ready"] is False
    assert inconsistent["blocked_reasons"] == ["matrix v21 and capstone .287 blocker counts disagree"]
    assert inconsistent["honest_verdict"].startswith("blocked_ledger_preconditions")


def test_req_report_3082_write_artifact_and_helper_edges(tmp_path: Path) -> None:
    """REQ-REPORT-3082: helper behavior is deterministic at matrix edges."""

    matrix = _matrix_payload()
    _write_json(tmp_path, mod.MATRIX_V21_REL_PATH, matrix)
    _write_json(tmp_path, mod.CAPSTONE_V287_REL_PATH, _capstone_payload(8))
    malformed = tmp_path / "bad.json"
    list_json = tmp_path / "list.json"
    malformed.write_text("{bad-json}\n", encoding="utf-8")
    list_json.write_text("[1, 2]\n", encoding="utf-8")

    output = mod.write_artifact(tmp_path, started_s=1.0, now_s=1.75)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["blocker_ledger_ready"] is True
    assert saved["duration_s"] == pytest.approx(0.75)
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    assert mod.read_json_object(malformed) == {}
    assert mod.read_json_object(list_json) == {}
    assert mod.sha256_file(tmp_path / "missing.txt") is None
    assert mod.normal_status("gate-skipped") == "gated_skipped"
    assert mod.normal_status("gate_skipped") == "gated_skipped"
    assert mod.normal_status("pilot_only") == "bounded"
    assert mod.normal_status("unknown") == "missing"
    assert mod.blocker_class("retired") == "retired_claim"
    assert mod._as_mapping({"x": 1}) == {"x": 1}
    assert mod._as_mapping(None) == {}
    assert mod._as_list([1]) == [1]
    assert mod._as_list("x") == []
    assert mod._int_or_none(True) is None
    assert mod._int_or_none("bad") is None
    assert mod._category_for_row({"status": "clean", "row_id": ""}) == "documentation_hygiene"
    assert mod._category_for_row({"status": "bounded", "claim_scope": "generic"}) == "bounded_status"
    assert "non-json" in mod._row_text({"summary": {"non-json": {1}}})

    retired_blocker = _row("retired-blocker", "retired", "retired_claim", "repair")
    categories = mod._blocker_categories([retired_blocker], [])
    assert _ids(categories["documentation_hygiene"]) == ["retired-blocker"]

    reasons = mod._blocked_reasons(
        matrix={"matrix_v21_ready": False},
        capstone={"capstone_ready": False},
        before_count=1,
        capstone_count=1,
        coverage={
            "uncategorized_publication_blocker_ids": ["x"],
            "duplicate_publication_blocker_ids": ["y"],
        },
    )
    assert reasons == [
        "matrix v21 is not ready",
        "capstone .287 is not ready",
        "one or more matrix v21 blockers were not categorized",
        "one or more matrix v21 blockers were categorized more than once",
    ]
