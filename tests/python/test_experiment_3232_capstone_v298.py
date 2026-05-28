"""Tests for Exp 3232 milestone .298 capstone.

Spec refs: REQ-REPORT-3232, SCENARIO-REPORT-3232.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v298_3232 as mod


REQUIRED_FIELDS = {
    "schema_version",
    "experiment_id",
    "milestone",
    "matrix_artifact",
    "prior_capstone_artifact",
    "capstone_ready",
    "paper_ready",
    "publication_blocker_count",
    "blocker_delta_from_v31",
    "local_sota_receipt_status",
    "clean_verifier_status",
    "repair_gate_status",
    "repair_ladder_status",
    "continuous_self_learning_status",
    "hardware_claim_status",
    "what_this_milestone_proved",
    "next_top_gap",
    "recommended_next_milestone_theme",
    "inference_substrate",
    "conductor_file_modified",
    "active_roadmap_modified",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _input_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    statuses = {
        "exp3219": ("complete", mod.EXP3219_REL_PATH, "archive_v297_activate_v298"),
        "exp3220": ("blocked", mod.EXP3220_REL_PATH, "hermetic_cuda_runtime_repair"),
        "exp3221": ("gate_blocked", mod.EXP3221_REL_PATH, "llama_cpp_cuda_offload_receipt"),
        "exp3222": ("missing", mod.EXP3222_REL_PATH, "full_local_sota_receipt_v6"),
        "exp3223": ("complete", mod.EXP3223_REL_PATH, "exact_row_uncertainty_sidecar"),
        "exp3224": ("partial", mod.EXP3224_REL_PATH, "partial_smt_context_coverage"),
        "exp3225": ("gate_blocked", mod.EXP3225_REL_PATH, "clean_live_sota_verifier_v13"),
        "exp3226": ("missing", mod.EXP3226_REL_PATH, "structured_repair_preflight_v2"),
        "exp3227": ("blocked", mod.EXP3227_REL_PATH, "repair_gate_decision_v7"),
        "exp3228": ("gate_blocked", mod.EXP3228_REL_PATH, "repair_ladder_v8"),
        "exp3229": ("complete", mod.EXP3229_REL_PATH, "fr11_controller_promotion"),
        "exp3230": ("complete", mod.EXP3230_REL_PATH, "kan_cl_certificate_boundary"),
    }
    for experiment_id, (status, path, role) in statuses.items():
        rows.append(
            {
                "experiment_id": experiment_id,
                "path": path.as_posix(),
                "role": role,
                "status": status,
                "present": status != "missing",
                "readable_json_object": status != "missing",
                "status_rationale": f"{role} status is {status}",
                "honest_verdict": "blocked_gate_check_failed"
                if status == "gate_blocked"
                else f"{status}: {role}",
            }
        )
    return rows


def _matrix_v32(*, paper_ready: bool = False, blockers: int = 100) -> dict[str, Any]:
    return {
        "schema_version": "carnot.cross_corpus_matrix.v32_298_artifact_aggregation.v1",
        "experiment_id": "exp3231",
        "milestone": "2026.05.298",
        "cross_corpus_matrix_v32_ready": True,
        "paper_ready": paper_ready,
        "publication_blocker_count": blockers,
        "blocker_delta_from_v31": blockers - 92,
        "local_sota_receipt_state": (
            "missing_full_local_sota_receipt_v6_after_exp3221_gate_blocked"
        ),
        "clean_verifier_state": (
            "gate_blocked_on_missing_full_local_sota_receipt_v6_no_clean_verifier_evidence"
        ),
        "repair_gate_state": "blocked_v7_blocker_count_9",
        "repair_ladder_state": "gate_blocked_repair_gate_v7_blocked",
        "continuous_self_learning_state": (
            "controller_memory_promotion_allowed_28_accepted_no_model_weight_update_"
            "kan_sidecar_blocked_missing_certificates_4"
        ),
        "hardware_claim_boundary": (
            "cuda_runtime_visible_but_not_usable_no_llama_cpp_offload_receipt_"
            "no_hardware_speedup_tsu_or_kona_claim_allowed"
        ),
        "paper_ready_criteria": {
            "local_sota_receipt": False,
            "clean_verifier": False,
            "repair": False,
            "fr11": True,
            "claim_boundary": False,
        },
        "input_artifacts": _input_rows(),
        "blocker_delta_explanation": [
            {"experiment_id": exp_id, "status": status}
            for exp_id, status in {
                "exp3220": "blocked",
                "exp3221": "gate_blocked",
                "exp3222": "missing",
                "exp3225": "gate_blocked",
                "exp3226": "missing",
                "exp3227": "blocked",
                "exp3228": "gate_blocked",
                "exp3230": "complete_certificate_boundary_blocked",
            }.items()
        ],
        "next_top_gap": mod.NEXT_TOP_GAP,
        "conductor_file_modified": False,
        "active_roadmap_modified": False,
        "honest_verdict": (
            "complete: cross_corpus_matrix_v32_ready=true; "
            f"paper_ready={str(paper_ready).lower()}; "
            f"publication_blocker_count={blockers}"
        ),
    }


def _paper_ready_matrix_v32() -> dict[str, Any]:
    matrix = _matrix_v32(paper_ready=True, blockers=0)
    matrix.update(
        {
            "local_sota_receipt_state": "clean_rerun_allowed_full_local_sota_receipt_v6",
            "clean_verifier_state": "clean_live_sota_verifier_v13_ready",
            "repair_gate_state": "unblocked_v7",
            "repair_ladder_state": "complete_repair_ladder_v8",
            "continuous_self_learning_state": (
                "controller_memory_promotion_allowed_no_model_weight_update_"
                "certificate_boundary_ready"
            ),
            "hardware_claim_boundary": "authenticated_hardware_claim_allowed",
            "paper_ready_criteria": {
                "local_sota_receipt": True,
                "clean_verifier": True,
                "repair": True,
                "fr11": True,
                "claim_boundary": True,
            },
            "blocker_delta_explanation": [],
            "next_top_gap": "publication_blocker_retirement_review",
        }
    )
    return matrix


def _prior_capstone(*, ready: bool = True, blockers: int = 92) -> dict[str, Any]:
    return {
        "schema_version": "carnot.milestone_capstone.v297_matrix_v31_terminal_aggregation.v1",
        "experiment_id": "exp3218",
        "milestone": "2026.05.297",
        "capstone_v297_ready": ready,
        "paper_ready": False,
        "publication_blocker_count": blockers,
        "next_top_gap": (
            "cuda_offload_full_local_sota_receipt_clean_rerun_allowed_repair_gate_unblock"
        ),
        "honest_verdict": "complete: capstone_v297_ready=true",
    }


def _write_sources(root: Path, *, matrix: dict[str, Any] | None = None) -> None:
    matrix_payload = _matrix_v32() if matrix is None else matrix
    _write_json(root, mod.MATRIX_V32_REL_PATH, matrix_payload)
    _write_json(root, mod.PRIOR_CAPSTONE_REL_PATH, _prior_capstone())
    for row in matrix_payload["input_artifacts"]:
        if row["present"]:
            _write_json(
                root,
                Path(row["path"]),
                {
                    "schema_version": f"schema.{row['experiment_id']}",
                    "experiment_id": row["experiment_id"],
                    "milestone": "2026.05.298",
                    "honest_verdict": row["honest_verdict"],
                },
            )


def test_req_report_3232_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3232: OpenSpec declares the capstone before implementation."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3232" in spec
    assert "SCENARIO-REPORT-3232" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_report_3232_builds_capstone_from_matrix_v32(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3232: .298 capstone preserves matrix v32 boundaries."""

    _write_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=3.5)
    sources = {row["experiment_id"]: row for row in artifact["source_artifacts"]}
    proof_domains = {row["domain"]: row for row in artifact["what_this_milestone_proved"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["schema_version"] == mod.SCHEMA_VERSION
    assert artifact["experiment_id"] == "exp3232"
    assert artifact["milestone"] == "2026.05.298"
    assert artifact["matrix_artifact"] == mod.MATRIX_V32_REL_PATH.as_posix()
    assert artifact["prior_capstone_artifact"] == mod.PRIOR_CAPSTONE_REL_PATH.as_posix()
    assert artifact["capstone_ready"] is True
    assert artifact["paper_ready"] is False
    assert artifact["publication_blocker_count"] == 100
    assert artifact["blocker_delta_from_v31"] == 8
    assert artifact["local_sota_receipt_status"] == (
        "missing_full_local_sota_receipt_v6_after_exp3221_gate_blocked"
    )
    assert artifact["clean_verifier_status"] == (
        "gate_blocked_on_missing_full_local_sota_receipt_v6_no_clean_verifier_evidence"
    )
    assert artifact["repair_gate_status"] == "blocked_v7_blocker_count_9"
    assert artifact["repair_ladder_status"] == "gate_blocked_repair_gate_v7_blocked"
    assert artifact["continuous_self_learning_status"] == (
        "controller_memory_promotion_allowed_28_accepted_no_model_weight_update_"
        "kan_sidecar_blocked_missing_certificates_4"
    )
    assert artifact["hardware_claim_status"] == (
        "cuda_runtime_visible_but_not_usable_no_llama_cpp_offload_receipt_"
        "no_hardware_speedup_tsu_or_kona_claim_allowed"
    )
    assert artifact["next_top_gap"] == mod.NEXT_TOP_GAP
    assert artifact["recommended_next_milestone_theme"] == (
        "hermetic_cuda_offload_receipt_repair_for_clean_local_sota"
    )
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["conductor_file_modified"] is False
    assert artifact["active_roadmap_modified"] is False
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["honest_verdict"].startswith("complete:")
    assert len(artifact["source_artifacts"]) == 14
    assert sources["matrix_v32"]["sha256"] == _sha256(tmp_path / mod.MATRIX_V32_REL_PATH)
    assert sources["prior_capstone_v297"]["sha256"] == _sha256(
        tmp_path / mod.PRIOR_CAPSTONE_REL_PATH
    )
    assert sources["exp3222"]["present"] is False
    assert sources["exp3226"]["present"] is False
    assert artifact["publication_blockers"][0]["experiment_id"] == "exp3220"

    assert proof_domains["cuda_receipt"]["status"] == "blocked"
    assert "exp3220" in proof_domains["cuda_receipt"]["evidence"]
    assert proof_domains["live_sota_verification"]["status"] == "gate_blocked"
    assert proof_domains["repair"]["status"] == "blocked"
    assert proof_domains["repair_ladder"]["status"] == "gate_blocked"
    assert proof_domains["continuous_self_learning"]["status"] == "controller_only"
    assert proof_domains["hardware_boundary"]["status"] == "claim_denied"
    assert artifact["claim_boundaries_preserved"] == {
        "paper_ready_claim_allowed": False,
        "repair_success_claim_allowed": False,
        "hardware_speedup_claim_allowed": False,
        "tsu_or_kona_claim_allowed": False,
        "model_weight_learning_claim_allowed": False,
    }


def test_req_report_3232_paper_ready_requires_matrix_evidence_and_zero_blockers(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-3232: paper readiness is evidence-gated, not inferred."""

    _write_sources(tmp_path, matrix=_paper_ready_matrix_v32())

    ready = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.0)

    assert ready["capstone_ready"] is True
    assert ready["paper_ready"] is True
    assert ready["publication_blocker_count"] == 0
    assert ready["recommended_next_milestone_theme"] == "publication_blocker_retirement_review"

    contradicted_matrix = _paper_ready_matrix_v32()
    contradicted_matrix["publication_blocker_count"] = 1
    contradicted_matrix["blocker_delta_from_v31"] = -91
    _write_json(tmp_path, mod.MATRIX_V32_REL_PATH, contradicted_matrix)

    contradicted = mod.build_artifact(tmp_path)

    assert contradicted["capstone_ready"] is False
    assert contradicted["paper_ready"] is False
    assert (
        "matrix_v32 paper_ready=true while publication blockers remain"
        in contradicted["invariant_violations"]
    )


def test_req_report_3232_write_artifact_and_fail_closed_edges(tmp_path: Path) -> None:
    """REQ-REPORT-3232: writer and helper branches keep the capstone bounded."""

    _write_sources(tmp_path)

    output = mod.write_artifact(tmp_path, started_s=4.0, now_s=5.0)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["capstone_ready"] is True
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    assert mod.read_json_object(bad_json) == {}
    assert mod.sha256_file(tmp_path / "missing.json") is None

    empty = mod.build_artifact(tmp_path / "empty", started_s=0.0, now_s=0.0)

    assert empty["capstone_ready"] is False
    assert empty["paper_ready"] is False
    assert empty["publication_blocker_count"] == 0
    assert empty["blocker_delta_from_v31"] == 0
    assert empty["local_sota_receipt_status"] == "missing_local_sota_receipt_status"
    assert empty["honest_verdict"].startswith("blocked:")

    assert mod._field_str({}, "x", "fallback") == "fallback"
    assert mod._as_mapping([]) == {}
    assert mod._as_list({}) == []
    assert mod._int_value(3) == 3
    assert mod._int_value(True) == 0
    assert mod._int_value("3") == 0
    assert mod._source_experiment_id({"experiment_id": "exp1"}, "fallback") == "exp1"
    assert mod._source_experiment_id({"experiment": 7}, "fallback") == "exp7"
    assert mod._source_experiment_id({}, "fallback") == "fallback"
    assert mod._proof_status("clean_rerun_allowed_full_local_sota_receipt_v6") == "ready"
    assert mod._proof_status("gate_blocked_repair_gate_v7_blocked") == "gate_blocked"
    assert mod._proof_status("missing_full_local_sota_receipt_v6") == "missing"
    assert mod._proof_status("blocked_v7") == "blocked"
    assert mod._proof_status("controller_memory_promotion_allowed_no_model_weight_update") == (
        "controller_only"
    )
    assert mod._proof_status("cuda_runtime_visible_but_not_usable") == "claim_denied"
    assert mod._proof_status("unknown") == "blocked"
    assert mod._recommended_next_milestone_theme(mod.NEXT_TOP_GAP) == (
        "hermetic_cuda_offload_receipt_repair_for_clean_local_sota"
    )
    assert mod._recommended_next_milestone_theme("clean_live_verifier_v13_gate_clearance") == (
        "clean_live_verifier_v13_gate_clearance_after_receipt"
    )
    assert mod._recommended_next_milestone_theme("repair_gate_v7_unblock") == (
        "structured_repair_gate_v7_unblock_and_ladder_v8_execution"
    )
    assert mod._recommended_next_milestone_theme("fr11_certificate_boundary") == (
        "fr11_certificate_boundary_for_sidecar_promotion"
    )
    assert mod._recommended_next_milestone_theme("hardware_claim_boundary") == (
        "authenticated_hardware_boundary_or_explicit_no_speedup_disclosure"
    )
    assert mod._recommended_next_milestone_theme("other") == "publication_blocker_retirement_review"

    assert mod._required_fields_are_typed({"schema_version": "x"}) == [
        "experiment_id missing_or_wrong_type",
        "milestone missing_or_wrong_type",
        "matrix_artifact missing_or_wrong_type",
        "prior_capstone_artifact missing_or_wrong_type",
        "capstone_ready missing_or_wrong_type",
        "paper_ready missing_or_wrong_type",
        "publication_blocker_count missing_or_wrong_type",
        "blocker_delta_from_v31 missing_or_wrong_type",
        "local_sota_receipt_status missing_or_wrong_type",
        "clean_verifier_status missing_or_wrong_type",
        "repair_gate_status missing_or_wrong_type",
        "repair_ladder_status missing_or_wrong_type",
        "continuous_self_learning_status missing_or_wrong_type",
        "hardware_claim_status missing_or_wrong_type",
        "what_this_milestone_proved missing_or_wrong_type",
        "next_top_gap missing_or_wrong_type",
        "recommended_next_milestone_theme missing_or_wrong_type",
        "inference_substrate missing_or_wrong_type",
        "conductor_file_modified missing_or_wrong_type",
        "active_roadmap_modified missing_or_wrong_type",
        "honest_verdict missing_or_wrong_type",
    ]
    assert mod._invariant_violations(
        matrix={
            "cross_corpus_matrix_v32_ready": True,
            "paper_ready": True,
            "conductor_file_modified": True,
            "active_roadmap_modified": True,
            "input_artifacts": [{}, {}],
        },
        prior_capstone={"capstone_v297_ready": True, "publication_blocker_count": 92},
        publication_blocker_count=0,
        blocker_delta_from_v31=99,
        paper_ready_candidate=True,
        readiness_criteria={"a": True, "b": False},
        source_artifacts=[{}],
    ) == [
        "matrix_v32 paper_ready=true while readiness criteria are incomplete",
        "blocker_delta_from_v31 does not reconcile with prior capstone count",
        "matrix_v32 reports conductor file modification",
        "matrix_v32 reports active roadmap modification",
        "not all matrix-referenced .298 artifacts are accounted for",
    ]
    assert mod._invariant_violations(
        matrix={"cross_corpus_matrix_v32_ready": True},
        prior_capstone={"capstone_v297_ready": True, "publication_blocker_count": 0},
        publication_blocker_count=1,
        blocker_delta_from_v31=1,
        paper_ready_candidate=True,
        readiness_criteria={"a": True},
        source_artifacts=[{}, {}],
    ) == ["paper_ready candidate has nonzero blockers"]
