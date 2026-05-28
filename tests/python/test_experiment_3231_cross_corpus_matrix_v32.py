"""Tests for Exp 3231 cross-corpus matrix v32.

Spec refs: REQ-REPORT-3231, SCENARIO-REPORT-3231.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import cross_corpus_matrix_v32_3231 as mod


REQUIRED_FIELDS = {
    "schema_version",
    "experiment_id",
    "milestone",
    "matrix_version",
    "input_artifacts",
    "missing_artifacts",
    "gate_blocked_artifacts",
    "local_sota_receipt_state",
    "clean_verifier_state",
    "repair_gate_state",
    "repair_ladder_state",
    "continuous_self_learning_state",
    "hardware_claim_boundary",
    "paper_ready",
    "publication_blocker_count",
    "blocker_delta_from_v31",
    "next_top_gap",
    "inference_substrate",
    "conductor_file_modified",
    "active_roadmap_modified",
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


def _write_v31(root: Path, *, ready: bool = True) -> None:
    _write_json(
        root,
        mod.PREVIOUS_MATRIX_REL_PATH,
        {
            "schema_version": "carnot.cross_corpus_matrix.v31_297_artifact_aggregation.v1",
            "experiment_id": "exp3217",
            "cross_corpus_matrix_v31_ready": ready,
            "paper_ready": False,
            "publication_blocker_count": 92,
            "blocker_delta_from_v30": 7,
            "next_top_gap": mod.PREVIOUS_NEXT_TOP_GAP,
            "honest_verdict": "complete: cross_corpus_matrix_v31_ready=true",
        },
    )


def _write_dot298_sources(root: Path) -> None:
    _write_json(
        root,
        mod.EXP3219_REL_PATH,
        {
            "schema_version": "carnot.archive_activation.v297_to_v298.v1",
            "experiment_id": "exp3219",
            "milestone": "2026.05.298",
            "activation_ready": True,
            "conductor_file_modified": False,
            "active_roadmap_modified": False,
            "honest_verdict": "complete: archive_v297_activate_v298_ready=true",
        },
    )
    _write_json(
        root,
        mod.EXP3220_REL_PATH,
        {
            "schema_version": "carnot.hermetic_cuda_runtime_repair_ledger.v1",
            "experiment_id": "exp3220",
            "milestone": "2026.05.298",
            "cuda_receipt_ready_candidate": False,
            "selected_python_cuda_ok_after": False,
            "isolated_cuda_venv_cuda_ok": False,
            "recommended_next_action": "repair_system_driver_cuda_runtime_boundary",
            "nvidia_smi_available": True,
            "conductor_file_modified": False,
            "active_roadmap_modified": False,
            "honest_verdict": "blocked_cuda_runtime: cuda_receipt_ready_candidate=false",
        },
    )
    _write_json(
        root,
        mod.EXP3221_REL_PATH,
        {
            "schema": "blocked_gate_check_v1",
            "experiment": 3221,
            "status": "blocked",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": (
                "1 of 1 gate(s) failed; first failure: "
                "exp3220-hermetic-cuda-runtime-repair-ledger-v1.cuda_receipt_ready_candidate"
            ),
            "honest_verdict": "blocked_gate_check_failed",
        },
    )
    _write_json(
        root,
        mod.EXP3223_REL_PATH,
        {
            "schema_version": "carnot.distributional_ebm_exact_row_uncertainty_sidecar.v2",
            "experiment_id": "exp3223",
            "milestone": "2026.05.298",
            "uncertainty_sidecar_ready": True,
            "exact_verifier_authority_preserved": True,
            "exact_row_count": 45,
            "conductor_file_modified": False,
            "active_roadmap_modified": False,
            "honest_verdict": (
                "complete: exact-row uncertainty sidecar materialized as triage metadata"
            ),
        },
    )
    _write_json(
        root,
        mod.EXP3224_REL_PATH,
        {
            "schema_version": "carnot.logitext_partial_smt_context_coverage_pilot.v1",
            "experiment_id": "exp3224",
            "milestone": "2026.05.298",
            "coverage_ready": True,
            "fixture_row_count": 45,
            "fully_formalizable_count": 35,
            "partially_formalizable_count": 10,
            "partial_smt_coverage": 1.0,
            "fully_smt_coverage": 0.777778,
            "conductor_file_modified": False,
            "active_roadmap_modified": False,
            "honest_verdict": "complete: partial SMT coverage ready; rows=45 full=35 partial=10",
        },
    )
    _write_json(
        root,
        mod.EXP3225_REL_PATH,
        {
            "schema": "blocked_gate_check_v1",
            "experiment": 3225,
            "status": "blocked",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": (
                "1 of 1 gate(s) failed; first failure: "
                "exp3222-full-local-sota-receipt-v6.clean_rerun_allowed"
            ),
            "honest_verdict": "blocked_gate_check_failed",
        },
    )
    _write_json(
        root,
        mod.EXP3227_REL_PATH,
        {
            "schema_version": "carnot.repair_gate_decision.v7",
            "experiment_id": "exp3227",
            "milestone": "2026.05.298",
            "receipt_ok": False,
            "clean_verifier_ok": False,
            "structured_preflight_ok": False,
            "repair_gate_state": "blocked",
            "repair_ladder_allowed": False,
            "blocker_count": 9,
            "conductor_file_modified": False,
            "active_roadmap_modified": False,
            "honest_verdict": "complete: repair_gate_state=blocked; repair_ladder_allowed=false",
        },
    )
    _write_json(
        root,
        mod.EXP3228_REL_PATH,
        {
            "schema": "blocked_gate_check_v1",
            "experiment": 3228,
            "status": "blocked",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": (
                "1 of 1 gate(s) failed; first failure: "
                "exp3227-repair-gate-decision-v7.repair_gate_state"
            ),
            "honest_verdict": "blocked_gate_check_failed",
        },
    )
    _write_json(
        root,
        mod.EXP3229_REL_PATH,
        {
            "schema": "carnot.fr11.nonforgetting_promotion_controller.v3",
            "schema_version": "1.0",
            "experiment_id": "experiment_3229_fr11_nonforgetting_promotion_controller_v3",
            "milestone": "2026.05.298",
            "continuous_self_learning_task": True,
            "promotion_allowed": True,
            "controller_memory_promotion_allowed": True,
            "accepted_trace_count": 28,
            "rejected_trace_count": 2,
            "deferred_trace_count": 0,
            "nonforgetting_budget_exceeded": False,
            "negative_control_regression_count": 0,
            "model_weight_update_claimed": False,
            "conductor_file_modified": False,
            "active_roadmap_modified": False,
            "honest_verdict": "complete: fr11 controller promotion allowed",
        },
    )
    _write_json(
        root,
        mod.EXP3230_REL_PATH,
        {
            "schema": "carnot.fr11.kan_cl_certificate_boundary_audit.v2",
            "schema_version": "1.0",
            "experiment_id": "experiment_3230_kan_cl_certificate_boundary_audit_v2",
            "milestone": "2026.05.298",
            "continuous_self_learning_task": True,
            "missing_certificate_count": 4,
            "per_knot_budget_defined": False,
            "pwa_milp_abstraction_ready": False,
            "certificate_boundary_ready": False,
            "kan_sidecar_promotion_allowed": False,
            "model_weight_update_claimed": False,
            "conductor_file_modified": False,
            "active_roadmap_modified": False,
            "honest_verdict": "complete: kan-cl certificate boundary audit",
        },
    )
    _write_text(
        root,
        mod.CONDUCTOR_LOG_REL_PATH,
        "\n".join(
            [
                "| 2026-05-27 23:56 UTC | llama.cpp CUDA offload receipt smoke gated on herm | GATE_BLOCK | 1 of 1 gate(s) failed |",
                "| 2026-05-28 00:02 UTC | Full local SOTA GGUF receipt v6 gated on llama.cpp | GATE_BLOCK | Pre-emptive skip: upstream retired (exp3221-llama-cpp-cuda-offload-receipt-smoke) |",
                "| 2026-05-28 00:31 UTC | Clean live SOTA verifier rerun v13 using exact-row | GATE_BLOCK | Pre-emptive skip: upstream retired (exp3222-full-local-sota-receipt-v6) |",
                "| 2026-05-28 00:31 UTC | Structured repair proposal preflight v2 with schem | GATE_BLOCK | Pre-emptive skip: upstream retired (exp3222-full-local-sota-receipt-v6) |",
                "| 2026-05-28 00:49 UTC | Multi-turn repair ladder v8 gated on repair gate u | GATE_BLOCK | 1 of 1 gate(s) failed |",
            ]
        )
        + "\n",
    )


def test_req_report_3231_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3231: OpenSpec declares matrix v32 before implementation."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3231" in spec
    assert "SCENARIO-REPORT-3231" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_report_3231_builds_v32_from_dot298_artifacts(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3231: .298 matrix preserves all claim boundaries."""

    _write_v31(tmp_path)
    _write_dot298_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=3.25)
    rows = {row["experiment_id"]: row for row in artifact["input_artifacts"]}
    missing = {row["experiment_id"]: row for row in artifact["missing_artifacts"]}
    gate_blocked = {row["experiment_id"]: row for row in artifact["gate_blocked_artifacts"]}
    partial = {row["experiment_id"]: row for row in artifact["partial_artifacts"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["cross_corpus_matrix_v32_ready"] is True
    assert artifact["duration_s"] == pytest.approx(2.25)
    assert len(artifact["input_artifacts"]) == 12
    assert artifact["status_counts"] == {
        "complete": 4,
        "blocked": 2,
        "gate_blocked": 3,
        "missing": 2,
        "partial": 1,
    }
    assert artifact["publication_blocker_count"] == 100
    assert artifact["blocker_delta_from_v31"] == 8
    assert len(artifact["blocker_delta_explanation"]) == 8
    assert artifact["paper_ready"] is False
    assert artifact["honest_verdict"].startswith("complete:")

    assert rows["exp3219"]["status"] == "complete"
    assert rows["exp3220"]["status"] == "blocked"
    assert rows["exp3221"]["status"] == "gate_blocked"
    assert rows["exp3222"]["status"] == "missing"
    assert rows["exp3223"]["status"] == "complete"
    assert rows["exp3224"]["status"] == "partial"
    assert rows["exp3225"]["status"] == "gate_blocked"
    assert rows["exp3226"]["status"] == "missing"
    assert rows["exp3227"]["status"] == "blocked"
    assert rows["exp3228"]["status"] == "gate_blocked"
    assert rows["exp3229"]["status"] == "complete"
    assert rows["exp3230"]["status"] == "complete"

    assert set(missing) == {"exp3222", "exp3226"}
    assert missing["exp3222"]["gated_skip_evidence"]["status"] == "gate_blocked"
    assert set(gate_blocked) == {"exp3221", "exp3222", "exp3225", "exp3226", "exp3228"}
    assert set(partial) == {"exp3224"}
    assert "missing_full_local_sota_receipt_v6" in artifact["local_sota_receipt_state"]
    assert artifact["clean_verifier_state"] == (
        "gate_blocked_on_missing_full_local_sota_receipt_v6_no_clean_verifier_evidence"
    )
    assert artifact["repair_gate_state"] == "blocked_v7_blocker_count_9"
    assert artifact["repair_ladder_state"] == "gate_blocked_repair_gate_v7_blocked"
    assert artifact["continuous_self_learning_state"] == (
        "controller_memory_promotion_allowed_28_accepted_no_model_weight_update_"
        "kan_sidecar_blocked_missing_certificates_4"
    )
    assert artifact["hardware_claim_boundary"] == (
        "cuda_runtime_visible_but_not_usable_no_llama_cpp_offload_receipt_"
        "no_hardware_speedup_tsu_or_kona_claim_allowed"
    )
    assert artifact["paper_ready_criteria"] == {
        "local_sota_receipt": False,
        "clean_verifier": False,
        "repair": False,
        "fr11": True,
        "claim_boundary": False,
    }
    assert artifact["next_top_gap"] == (
        "repair_system_driver_cuda_runtime_boundary_to_unblock_cuda_offload_receipt"
    )
    assert artifact["source_checksums"][mod.EXP3220_REL_PATH.as_posix()] == _sha256(
        tmp_path / mod.EXP3220_REL_PATH
    )


def test_req_report_3231_missing_previous_matrix_blocks(tmp_path: Path) -> None:
    """REQ-REPORT-3231: absent v31 authority blocks readiness claims."""

    _write_dot298_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=2.0, now_s=2.0)

    assert artifact["cross_corpus_matrix_v32_ready"] is False
    assert artifact["paper_ready"] is False
    assert artifact["prior_publication_blocker_count"] == 0
    assert artifact["publication_blocker_count"] == 8
    assert artifact["blocker_delta_from_v31"] == 8
    assert "previous matrix v31 is missing or not ready" in artifact["invariant_violations"]
    assert artifact["honest_verdict"].startswith("blocked_matrix_v32_preconditions")


def test_req_report_3231_write_artifact_and_helper_edges(tmp_path: Path) -> None:
    """REQ-REPORT-3231: writer and defensive helpers are deterministic."""

    _write_v31(tmp_path)
    _write_dot298_sources(tmp_path)

    output = mod.write_artifact(tmp_path, started_s=4.0, now_s=6.0)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["cross_corpus_matrix_v32_ready"] is True
    assert mod._normal_status("blocked") == "blocked"
    assert mod._normal_status("gate-blocked") == "gate_blocked"
    assert mod._normal_status("unknown") == "missing"
    assert mod._experiment_id({"experiment_id": "exp1"}, "fallback") == "exp1"
    assert mod._experiment_id({"experiment": 7}, "fallback") == "exp7"
    assert mod._experiment_id({}, "fallback") == "fallback"
    assert mod._bool_value(True) is True
    assert mod._bool_value("true") is False
    assert mod._int_value(5) == 5
    assert mod._int_value(True) == 0
    assert mod._int_value("5") == 0
    assert mod._as_mapping({"a": 1}) == {"a": 1}
    assert mod._as_mapping(None) == {}
    assert mod._read_text(tmp_path / "missing.md") == ""
    assert mod._gate_evidence("exp9999", "") == {"status": "absent"}
    assert mod._source_payload(tmp_path / "missing", mod.SOURCE_SPECS[0])["status"] == "missing"

    assert mod._classify(mod.SOURCE_SPECS[0], {"activation_ready": False})[0] == "blocked"
    assert mod._classify(mod.SOURCE_SPECS[1], {"cuda_receipt_ready_candidate": True})[0] == (
        "complete"
    )
    assert mod._classify(mod.SOURCE_SPECS[2], {"cuda_receipt_ready": True})[0] == "complete"
    assert mod._classify(mod.SOURCE_SPECS[3], {"clean_rerun_allowed": True})[0] == ("complete")
    assert mod._classify(mod.SOURCE_SPECS[4], {"uncertainty_sidecar_ready": False})[0] == (
        "blocked"
    )
    assert mod._classify(mod.SOURCE_SPECS[5], {"coverage_ready": False})[0] == "blocked"
    assert (
        mod._classify(
            mod.SOURCE_SPECS[5],
            {"coverage_ready": True, "partially_formalizable_count": 0},
        )[0]
        == "complete"
    )
    assert (
        mod._classify(
            mod.SOURCE_SPECS[6], {"clean_verifier_ready": True, "headline_claim_allowed": True}
        )[0]
        == "complete"
    )
    assert mod._classify(mod.SOURCE_SPECS[7], {"ready_for_repair_gate": True})[0] == ("complete")
    assert mod._classify(mod.SOURCE_SPECS[8], {"repair_gate_state": "unblocked"})[0] == ("complete")
    assert mod._classify(mod.SOURCE_SPECS[9], {"repair_ladder_complete": True})[0] == ("complete")
    assert (
        mod._classify(
            mod.SOURCE_SPECS[10],
            {"promotion_allowed": True, "model_weight_update_claimed": True},
        )[0]
        == "blocked"
    )
    assert (
        mod._classify(
            mod.SOURCE_SPECS[11],
            {
                "certificate_boundary_ready": True,
                "kan_sidecar_promotion_allowed": True,
                "model_weight_update_claimed": False,
            },
        )[0]
        == "complete"
    )
    assert (
        mod._classify(
            mod.SOURCE_SPECS[11],
            {"certificate_boundary_ready": True, "model_weight_update_claimed": True},
        )[0]
        == "blocked"
    )
    assert (
        mod._classify(
            mod.SourceSpec("exp9999", Path("results/missing.json"), "unknown", "unknown"),
            {"some": "payload"},
        )[0]
        == "blocked"
    )


def test_req_report_3231_fail_closed_state_branches(tmp_path: Path) -> None:
    """REQ-REPORT-3231: state helpers fail closed rather than inventing success."""

    rows: list[dict[str, Any]] = []
    payloads = {spec.experiment_id: {} for spec in mod.SOURCE_SPECS}

    assert mod._local_sota_receipt_state(payloads, rows) == ("missing_full_local_sota_receipt_v6")
    assert (
        mod._local_sota_receipt_state(
            {"exp3222": {"clean_rerun_allowed": True}},
            [{"experiment_id": "exp3222", "status": "complete"}],
        )
        == "clean_rerun_allowed_full_local_sota_receipt_v6"
    )
    assert (
        mod._local_sota_receipt_state(
            {"exp3222": {}},
            [{"experiment_id": "exp3222", "status": "gate_blocked"}],
        )
        == "gate_blocked_full_local_sota_receipt_v6_no_clean_rerun_allowed"
    )
    assert (
        mod._local_sota_receipt_state(
            {"exp3222": {}},
            [{"experiment_id": "exp3222", "status": "blocked"}],
        )
        == "blocked_full_local_sota_receipt_v6_no_clean_rerun_allowed"
    )
    assert mod._clean_verifier_state(rows) == "missing_clean_live_sota_verifier_v13"
    assert mod._clean_verifier_state([{"experiment_id": "exp3225", "status": "complete"}]) == (
        "clean_live_sota_verifier_v13_ready"
    )
    assert mod._clean_verifier_state([{"experiment_id": "exp3225", "status": "blocked"}]) == (
        "blocked_clean_live_sota_verifier_v13"
    )
    assert mod._repair_gate_state(payloads, rows) == "missing_repair_gate_v7"
    assert (
        mod._repair_gate_state(
            {"exp3227": {"repair_gate_state": "unblocked"}},
            [{"experiment_id": "exp3227", "status": "complete"}],
        )
        == "unblocked_v7"
    )
    assert mod._repair_ladder_state(rows) == "missing_repair_ladder_v8"
    assert mod._repair_ladder_state([{"experiment_id": "exp3228", "status": "complete"}]) == (
        "complete_repair_ladder_v8"
    )
    assert mod._repair_ladder_state([{"experiment_id": "exp3228", "status": "blocked"}]) == (
        "blocked_repair_ladder_v8"
    )
    assert mod._continuous_self_learning_state(payloads) == (
        "missing_fr11_nonforgetting_or_certificate_boundary_artifacts"
    )
    assert (
        mod._continuous_self_learning_state(
            {
                "exp3229": {"promotion_allowed": True, "model_weight_update_claimed": True},
                "exp3230": {"certificate_boundary_ready": False},
            }
        )
        == "blocked_model_weight_update_claimed"
    )
    assert (
        mod._continuous_self_learning_state(
            {
                "exp3229": {"promotion_allowed": True},
                "exp3230": {"certificate_boundary_ready": True},
            }
        )
        == "controller_memory_promotion_allowed_no_model_weight_update_certificate_boundary_ready"
    )
    assert (
        mod._continuous_self_learning_state(
            {
                "exp3229": {"promotion_allowed": False},
                "exp3230": {"certificate_boundary_ready": False},
            }
        )
        == "blocked_fr11_nonforgetting_promotion_controller"
    )
    assert mod._hardware_claim_boundary(payloads, rows) == (
        "no_authenticated_runtime_or_hardware_evidence_no_speedup_tsu_or_kona_claim_allowed"
    )
    assert mod._paper_ready_criteria(
        "ok",
        "ok",
        "unblocked",
        "complete",
        "controller_memory_promotion_allowed_no_model_weight_update_certificate_boundary_ready",
    ) == {
        "local_sota_receipt": False,
        "clean_verifier": False,
        "repair": True,
        "fr11": True,
        "claim_boundary": True,
    }
    assert mod._all_ready({"a": True, "b": True}) is True
    assert mod._all_ready({"a": True, "b": False}) is False
    ready_receipt = "clean_rerun_allowed_full_local_sota_receipt_v6"
    ready_clean = "clean_live_sota_verifier_v13_ready"
    ready_gate = "unblocked_v7"
    ready_ladder = "complete_repair_ladder_v8"
    ready_fr11 = (
        "controller_memory_promotion_allowed_no_model_weight_update_certificate_boundary_ready"
    )
    ready_hw = "authenticated_hardware_claim_allowed"

    assert mod._next_top_gap(
        ready_receipt, "blocked", ready_gate, ready_ladder, ready_fr11, ready_hw
    ) == ("clean_live_verifier_v13_gate_clearance")
    assert mod._next_top_gap(
        ready_receipt, ready_clean, "blocked", ready_ladder, ready_fr11, ready_hw
    ) == ("repair_gate_v7_unblock_and_ladder_v8_execution")
    assert mod._next_top_gap(
        ready_receipt, ready_clean, ready_gate, "blocked", ready_fr11, ready_hw
    ) == ("repair_gate_v7_unblock_and_ladder_v8_execution")
    assert mod._next_top_gap(
        ready_receipt, ready_clean, ready_gate, ready_ladder, "blocked", ready_hw
    ) == ("fr11_certificate_boundary_for_sidecar_promotion")
    assert mod._next_top_gap(
        ready_receipt, ready_clean, ready_gate, ready_ladder, ready_fr11, "blocked"
    ) == ("authenticated_hardware_claim_boundary_or_explicit_no_speedup_disclosure")
    assert mod._next_top_gap(
        ready_receipt, ready_clean, ready_gate, ready_ladder, ready_fr11, ready_hw
    ) == ("publication_blocker_retirement_review")
    assert mod._gate_evidence("exp3222", "no matching gate line") == {"status": "absent"}
    assert mod._model_weight_update_claimed({"model_weight_update_performed": True}) is True

    assert any(
        "status_counts keys" in item
        for item in mod._invariant_violations(
            {"cross_corpus_matrix_v31_ready": True}, [], {}, 0, 0, []
        )
    )
    assert any(
        "status_counts do not sum" in item
        for item in mod._invariant_violations(
            {"cross_corpus_matrix_v31_ready": True},
            [],
            {status: 1 for status in mod.STATUSES},
            0,
            0,
            [],
        )
    )
    assert any(
        "publication_blocker_count does not reconcile" in item
        for item in mod._invariant_violations(
            {"cross_corpus_matrix_v31_ready": True},
            [],
            {status: 0 for status in mod.STATUSES},
            5,
            1,
            [],
        )
    )
