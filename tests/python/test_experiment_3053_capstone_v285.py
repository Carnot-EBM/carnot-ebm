"""Tests for Exp 3053 milestone .285 capstone.

Spec refs: REQ-REPORT-3053, SCENARIO-REPORT-3053.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v285_3053 as mod


REQUIRED_FIELDS = {
    "capstone_ready",
    "paper_ready",
    "repair_claim_status",
    "fr11_self_learning_status",
    "gatemate_status",
    "ssqa_status",
    "matrix_v19_summary",
    "promoted_claims",
    "bounded_claims",
    "blocked_claims",
    "gated_skipped_claims",
    "next_milestone_recommendation",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}

FORBIDDEN_TOP_LEVEL = {
    "model_specs",
    "target_model",
    "cuda",
    "CUDA",
    "gguf",
    "GGUF",
    "gpu_inventory",
    "headline_models_used",
    "live_model_metadata",
}


def _write_json(root: Path, rel_path: Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_artifact(exp_id: str, path: Path, *, required: bool = False) -> dict[str, Any]:
    return {
        "experiment_id": exp_id,
        "path": path.as_posix(),
        "role": f"{exp_id}_role",
        "required": required,
        "present": True,
        "readable_json_object": True,
        "sha256": f"upstream-{exp_id}",
    }


def _row(row_id: str, status: str, evidence_class: str, blocker_class: str) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "status": status,
        "source_artifact": f"results/{row_id.replace(':', '_')}.json",
        "source_field": "status",
        "evidence_class": evidence_class,
        "blocker_class": blocker_class,
        "claim_scope": evidence_class,
        "summary": {"claim": row_id, "status": status},
    }


def _matrix_v19(
    *,
    clean: bool = False,
    source_artifacts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if clean:
        rows = [
            _row("repair:headline_status", "clean", "repair_reconciliation", "none"),
            _row("fr11:solver_feedback", "clean", "controller_solver_feedback", "none"),
            _row("fr11:kan_locality", "clean", "controller_locality_probe", "none"),
            _row("gatemate:host_visible_smoke", "clean", "host_visible_transcript", "none"),
            _row("ssqa:readback_gate", "clean", "ssqa_gate", "none"),
        ]
        counts = {
            "clean_count": 5,
            "flagged_count": 0,
            "bounded_count": 0,
            "blocked_count": 0,
            "gated_skipped_count": 0,
            "projection_only_count": 0,
            "missing_count": 0,
            "retired_count": 0,
        }
        return {
            "artifact": "experiment_3052_cross_corpus_matrix_v19",
            "matrix_v19_ready": True,
            "rows_total": len(rows),
            "repair_claim_status": "clean_candidate",
            "fr11_self_learning_status": "controller_only_solver_feedback_and_locality_ready",
            "gatemate_status": "host_visible_transcript_ready",
            "ssqa_status": "eligible_for_micro_panel",
            "rows": rows,
            "source_artifacts": source_artifacts or _required_sources(),
            "honest_verdict": "complete: matrix_v19_ready=true",
            **counts,
        }

    rows = [
        _row("fingerprint:verified_speculation", "clean", "transcript_fingerprint", "none"),
        _row("v18:exp3028", "flagged", "repair_rerun", "adversarial_or_methodology_flag"),
        _row("repair:headline_status", "bounded", "repair_reconciliation", "bounded_claim"),
        _row("fr11:solver_feedback", "bounded", "controller_solver_feedback", "controller_only_scope"),
        _row("fr11:kan_locality", "bounded", "controller_locality_probe", "controller_only_scope"),
        _row("gatemate:output_contract", "blocked", "gatemate_output_contract", "required_blocker"),
        _row("ssqa:readback_gate", "gated_skipped", "ssqa_gate", "structured_gate_skip"),
        _row("archive:v284", "projection_only", "archive_activation", "projection_only"),
        _row("gatemate:host_visible_smoke", "missing", "host_visible_transcript", "missing_artifact"),
        _row("repair:unsupported_exp3016", "retired", "repair_retirement_boundary", "retired_claim"),
    ]
    return {
        "artifact": "experiment_3052_cross_corpus_matrix_v19",
        "matrix_v19_ready": True,
        "rows_total": len(rows),
        "clean_count": 1,
        "flagged_count": 1,
        "bounded_count": 3,
        "blocked_count": 1,
        "gated_skipped_count": 1,
        "projection_only_count": 1,
        "missing_count": 1,
        "retired_count": 1,
        "repair_claim_status": "bounded",
        "fr11_self_learning_status": "controller_only_solver_feedback_and_locality_ready",
        "gatemate_status": "blocked_output_contract",
        "ssqa_status": "gated_skipped_host_visible_smoke_missing",
        "rows": rows,
        "source_artifacts": source_artifacts or _required_sources(),
        "honest_verdict": "complete: matrix_v19_ready=true",
    }


def _required_sources() -> list[dict[str, Any]]:
    return [
        _source_artifact("exp3038", Path("results/experiment_3038_cross_corpus_matrix_v18.json"), required=True),
        _source_artifact("exp3039", Path("results/experiment_3039_capstone_v284.json"), required=True),
        _source_artifact("exp3041", Path("results/experiment_3041_matrix_capstone_adversarial_flag_hygiene_v1.json"), required=True),
        _source_artifact("exp3042", Path("results/experiment_3042_repair_promotion_reconciliation_v3.json"), required=True),
        _source_artifact("exp3050", Path("results/experiment_3050_gatemate_host_visible_flash_smoke_v5.json")),
    ]


def _write_sources(root: Path, *, clean: bool = False) -> dict[str, Any]:
    matrix = _matrix_v19(clean=clean)
    _write_json(root, mod.MATRIX_V19_REL_PATH, matrix)
    for source in matrix["source_artifacts"]:
        _write_json(
            root,
            Path(source["path"]),
            {
                "artifact": source["experiment_id"],
                "honest_verdict": f"complete: {source['experiment_id']}",
            },
        )
    return matrix


def _claims_by_id(claims: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(claim["row_id"]): claim for claim in claims}


def test_req_report_3053_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3053: OpenSpec declares the .285 capstone contract first."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3053" in spec
    assert "SCENARIO-REPORT-3053" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_report_3053_builds_capstone_without_paper_overclaim(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3053: bounded, blocked, and gate-skipped rows stay visible."""

    _write_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=2.0, now_s=5.25)
    summary = artifact["matrix_v19_summary"]
    promoted = _claims_by_id(artifact["promoted_claims"])
    bounded = _claims_by_id(artifact["bounded_claims"])
    blocked = _claims_by_id(artifact["blocked_claims"])
    skipped = _claims_by_id(artifact["gated_skipped_claims"])

    assert REQUIRED_FIELDS <= artifact.keys()
    assert FORBIDDEN_TOP_LEVEL.isdisjoint(artifact)
    assert artifact["capstone_ready"] is True
    assert artifact["paper_ready"] is False
    assert artifact["duration_s"] == pytest.approx(3.25)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["repair_claim_status"] == "bounded"
    assert artifact["fr11_self_learning_status"] == (
        "controller_only_solver_feedback_and_locality_ready"
    )
    assert artifact["gatemate_status"] == "blocked_output_contract"
    assert artifact["ssqa_status"] == "gated_skipped_host_visible_smoke_missing"

    assert summary["matrix_v19_ready"] is True
    assert summary["rows_total"] == 10
    assert summary["counts_match_rows"] is True
    assert summary["required_source_artifacts_readable"] is True
    assert summary["nonclean_publication_blockers"] == 8
    assert summary["status_by_row"]["repair:headline_status"] == "bounded"

    assert set(promoted) == {"fingerprint:verified_speculation"}
    assert set(bounded) == {
        "repair:headline_status",
        "fr11:solver_feedback",
        "fr11:kan_locality",
    }
    assert {"v18:exp3028", "gatemate:output_contract", "gatemate:host_visible_smoke"} <= set(
        blocked
    )
    assert set(skipped) == {"ssqa:readback_gate"}
    assert artifact["retired_claims"][0]["row_id"] == "repair:unsupported_exp3016"
    assert artifact["gate_skipped_claims"] == artifact["gated_skipped_claims"]
    assert "retire" in artifact["next_milestone_recommendation"]
    assert "gate" in artifact["next_milestone_recommendation"]
    assert "rerun" in artifact["next_milestone_recommendation"]

    checks = {row["check"]: row for row in artifact["paper_ready_checks"]}
    assert checks["capstone_ready"]["passed"] is True
    assert checks["repair_promotable"]["passed"] is False
    assert checks["fr11_prd_scope"]["passed"] is True
    assert checks["gatemate_host_visible_transcript"]["passed"] is False
    assert checks["ssqa_readback_eligible"]["passed"] is False
    assert checks["matrix_has_no_publication_blockers"]["passed"] is False
    assert artifact["inference_substrate"] == {
        "kind": "aggregation_from_upstream_artifacts",
        "source": "checked_in_artifacts",
        "executes_models": False,
        "executes_hardware": False,
        "executes_conductor": False,
        "no_live_llm_inference": True,
    }
    source_by_path = {row["path"]: row for row in artifact["source_artifacts"]}
    assert source_by_path[mod.MATRIX_V19_REL_PATH.as_posix()]["sha256"] == _sha256(
        tmp_path / mod.MATRIX_V19_REL_PATH
    )


def test_req_report_3053_sets_paper_ready_only_when_all_gates_pass(tmp_path: Path) -> None:
    """REQ-REPORT-3053: paper readiness requires clean repair, FR-11, GateMate, SSQA, and matrix evidence."""

    _write_sources(tmp_path, clean=True)

    artifact = mod.build_artifact(tmp_path)
    checks = {row["check"]: row for row in artifact["paper_ready_checks"]}

    assert artifact["capstone_ready"] is True
    assert artifact["paper_ready"] is True
    assert artifact["repair_claim_status"] == "clean_candidate"
    assert artifact["gatemate_status"] == "host_visible_transcript_ready"
    assert artifact["ssqa_status"] == "eligible_for_micro_panel"
    assert all(row["passed"] is True for row in checks.values())
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_report_3053_blocks_fr11_model_weight_scope_violation(tmp_path: Path) -> None:
    """REQ-REPORT-3053: controller-only FR-11 evidence cannot become model-weight learning."""

    matrix = _matrix_v19(clean=True)
    matrix["fr11_self_learning_status"] = "controller_only_solver_feedback_and_locality_ready"
    matrix["rows"][1]["blocker_class"] = "model_weight_scope_violation"
    _write_json(tmp_path, mod.MATRIX_V19_REL_PATH, matrix)
    for source in matrix["source_artifacts"]:
        _write_json(tmp_path, Path(source["path"]), {"artifact": source["experiment_id"]})

    artifact = mod.build_artifact(tmp_path)
    checks = {row["check"]: row for row in artifact["paper_ready_checks"]}

    assert artifact["capstone_ready"] is True
    assert artifact["paper_ready"] is False
    assert checks["fr11_prd_scope"]["passed"] is False


def test_req_report_3053_write_artifact_persists_stable_json(tmp_path: Path) -> None:
    """REQ-REPORT-3053: write_artifact emits the deliverable JSON."""

    _write_sources(tmp_path)

    output = mod.write_artifact(tmp_path, started_s=10.0, now_s=11.5)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["capstone_ready"] is True
    assert saved["paper_ready"] is False
    assert saved["duration_s"] == pytest.approx(1.5)
    assert saved["source_checksums"][mod.MATRIX_V19_REL_PATH.as_posix()] == _sha256(
        tmp_path / mod.MATRIX_V19_REL_PATH
    )


def test_req_report_3053_missing_required_sources_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-3053: absent matrix or required source artifacts block capstone readiness."""

    blocked_without_matrix = mod.build_artifact(tmp_path)
    assert blocked_without_matrix["capstone_ready"] is False
    assert blocked_without_matrix["paper_ready"] is False
    assert blocked_without_matrix["honest_verdict"] == "blocked_required_matrix_v19_missing"

    matrix = _matrix_v19()
    _write_json(tmp_path, mod.MATRIX_V19_REL_PATH, matrix)
    for source in matrix["source_artifacts"][1:]:
        _write_json(tmp_path, Path(source["path"]), {"artifact": source["experiment_id"]})

    blocked_missing_required_source = mod.build_artifact(tmp_path)
    assert blocked_missing_required_source["capstone_ready"] is False
    assert blocked_missing_required_source["paper_ready"] is False
    assert blocked_missing_required_source["required_source_errors"] == [
        {"experiment_id": "exp3038", "reason": "missing_or_malformed_required_artifact"}
    ]
    assert blocked_missing_required_source["honest_verdict"].startswith(
        "blocked_capstone_preconditions"
    )


def test_req_report_3053_helper_edges_remain_honest(tmp_path: Path) -> None:
    """REQ-REPORT-3053: malformed inputs, count mismatches, and unknown statuses fail closed."""

    malformed = tmp_path / "bad.json"
    list_json = tmp_path / "list.json"
    malformed.write_text("{bad-json}\n", encoding="utf-8")
    list_json.write_text("[1, 2, 3]\n", encoding="utf-8")

    assert mod.read_json_object(tmp_path / "missing.json") == {}
    assert mod.read_json_object(malformed) == {}
    assert mod.read_json_object(list_json) == {}
    assert mod.sha256_file(tmp_path / "missing.txt") is None
    assert mod._normal_status("pilot_only") == "bounded"
    assert mod._normal_status("gate-skipped") == "gated_skipped"
    assert mod._normal_status("not-a-status") == "missing"
    assert mod._as_mapping({"x": 1}) == {"x": 1}
    assert mod._as_mapping(None) == {}
    assert mod._as_list([1]) == [1]
    assert mod._as_list("x") == []
    assert mod._int_or_none(None) is None
    assert mod._int_or_none(True) is None
    assert mod._int_or_none("bad") is None

    mismatched = _matrix_v19(clean=True)
    mismatched["clean_count"] = 99
    summary = mod._matrix_v19_summary(mismatched, mod._matrix_rows(mismatched), [])
    assert summary["counts_match_rows"] is False
    assert mod._capstone_ready(mismatched, summary, []) is False
    assert mod._paper_ready_checks(
        capstone_ready=False,
        matrix_summary=summary,
        repair_status="clean_candidate",
        fr11_status="controller_only_solver_feedback_and_locality_ready",
        gatemate_status="host_visible_transcript_ready",
        ssqa_status="eligible_for_micro_panel",
        rows_by_id={},
    )[0]["passed"] is False

    unknown_row = _row("unknown:row", "not-a-status", "unknown", "unknown")
    assert mod._claim_entry(unknown_row)["status"] == "missing"
    assert mod._claims_with_status([unknown_row], {"missing"})[0]["row_id"] == "unknown:row"
