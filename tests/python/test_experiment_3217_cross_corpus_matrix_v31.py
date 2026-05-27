"""Tests for Exp 3217 cross-corpus matrix v31.

Spec refs: REQ-REPORT-3217, SCENARIO-REPORT-3217.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import cross_corpus_matrix_v31_3217 as mod


REQUIRED_FIELDS = {
    "schema_version",
    "experiment_id",
    "milestone",
    "previous_matrix_artifact",
    "upstream_artifacts",
    "missing_artifacts",
    "status_counts",
    "paper_ready",
    "publication_blocker_count",
    "blocker_delta_from_v30",
    "local_sota_receipt_status",
    "clean_verifier_status",
    "repair_status",
    "context_fixture_status",
    "constraintbench_fixture_status",
    "fr11_self_learning_status",
    "hardware_sampler_status",
    "next_top_gap",
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


def _matrix_v30() -> dict[str, Any]:
    return {
        "schema_version": "carnot.cross_corpus_matrix.v30_296_artifact_aggregation.v1",
        "experiment_id": "exp3203",
        "cross_corpus_matrix_v30_ready": True,
        "paper_ready": False,
        "publication_blocker_count": 85,
        "blocker_delta_from_v29": 5,
        "local_sota_receipt_status": "blocked_cuda_unavailable_no_full_local_sota_receipt",
        "clean_verifier_status": "gated_skipped_clean_verifier_v11_waiting_on_clean_rerun_allowed",
        "repair_status": "blocked_clean_verifier_gate_repair_ladder_gated_skipped",
        "fr11_self_learning_status": (
            "controller_memory_trace_policy_promoted_no_model_weight_update_"
            "sidecar_promotion_blocked"
        ),
        "hardware_sampler_status": (
            "diagnostic_only_sparse_potts_thrml_factor_boundary_no_authenticated_speedup"
        ),
        "next_top_gap": mod.NEXT_TOP_GAP,
        "honest_verdict": "complete: cross_corpus_matrix_v30_ready=true",
    }


def _write_matrix_v30(root: Path) -> None:
    _write_json(root, mod.PREVIOUS_MATRIX_REL_PATH, _matrix_v30())


def _write_dot297_sources(root: Path) -> None:
    _write_json(
        root,
        mod.EXP3205_REL_PATH,
        {
            "schema_version": "carnot.archive_activation.v296_to_v297.v1",
            "schema": "carnot.archive_activation.v296_to_v297.v1",
            "experiment_id": "exp3205",
            "milestone": "2026.05.297",
            "activation_ready": True,
            "prior_publication_blocker_count": 85,
            "prior_next_top_gap": mod.NEXT_TOP_GAP,
            "conductor_file_modified": False,
            "active_roadmap_modified": False,
            "duration_s": 0.1,
            "honest_verdict": "complete: archive_v296_activate_v297_ready=true",
        },
    )
    _write_json(
        root,
        mod.EXP3206_REL_PATH,
        {
            "schema_version": "carnot.cuda_env_forensics_ledger.v1",
            "experiment_id": "exp3206",
            "milestone": "2026.05.297",
            "cuda_env_diagnosed": True,
            "cuda_init_clean": False,
            "torch_cuda_available_clean_subprocess": False,
            "recommended_next_action": "repair_selected_python_torch_cuda_before_full_receipt",
            "conductor_file_modified": False,
            "active_roadmap_modified": False,
            "duration_s": 0.1,
            "honest_verdict": (
                "blocked_selected_python_torch_cuda: cuda_env_diagnosed=true; "
                "cuda_init_clean=false"
            ),
        },
    )
    _write_json(
        root,
        mod.EXP3207_REL_PATH,
        {
            "schema_version": "carnot.llama_cpp_cuda_rebuild_clean_subprocess.v1",
            "experiment_id": "exp3207",
            "milestone": "2026.05.297",
            "cuda_receipt_ready": False,
            "clean_rerun_allowed_candidate": False,
            "torch_cuda_available_after": False,
            "llama_cpp_cuda_build_detected_after": False,
            "blocker": "selected_python_torch_cuda_unavailable",
            "conductor_file_modified": False,
            "active_roadmap_modified": False,
            "duration_s": 0.1,
            "honest_verdict": "blocked_selected_python_torch_cuda: selected_python_torch_cuda_unavailable",
        },
    )
    _write_json(
        root,
        mod.EXP3208_REL_PATH,
        {
            "schema": "blocked_gate_check_v1",
            "experiment": 3208,
            "status": "blocked",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": (
                "1 of 1 gate(s) failed; first failure: "
                "exp3207-llama-cpp-cuda-rebuild-clean-subprocess-v1.cuda_receipt_ready "
                "(actual=False == expected=True)"
            ),
            "gates_evaluated": [
                {
                    "upstream": "exp3207-llama-cpp-cuda-rebuild-clean-subprocess-v1",
                    "artifact_field": "cuda_receipt_ready",
                    "expected": True,
                    "actual": False,
                    "passed": False,
                }
            ],
            "duration_s": 0.0,
            "honest_verdict": "blocked_gate_check_failed",
        },
    )
    _write_json(
        root,
        mod.EXP3210_REL_PATH,
        {
            "schema_version": "carnot.context_cot_clbench_parametric_shortcut_fixtures.v1",
            "experiment_id": "exp3210",
            "milestone": "2026.05.297",
            "fixture_count": 30,
            "fixture_path": "data/research/context_cot_clbench_parametric_shortcut_v1.jsonl",
            "context_following_score_available": True,
            "ready_for_clean_verifier": True,
            "inference_substrate": "deterministic_exact_fixture_generation",
            "conductor_file_modified": False,
            "active_roadmap_modified": False,
            "duration_s": 0.1,
            "honest_verdict": (
                "complete: context_cot_clbench_parametric_shortcut_fixtures_v1 "
                "ready_for_clean_verifier=true; fixture_count=30"
            ),
        },
    )
    _write_json(
        root,
        mod.EXP3211_REL_PATH,
        {
            "schema_version": "carnot.constraintbench_feasibility_objective_pilot.v1",
            "experiment_id": "exp3211",
            "milestone": "2026.05.297",
            "fixture_count": 15,
            "fixture_path": "data/research/constraintbench_feasibility_objective_pilot_v1.jsonl",
            "ready_for_clean_verifier": True,
            "metric_summary": {"candidate_count": 15, "feasibility_pass_rate": 0.6},
            "conductor_file_modified": False,
            "active_roadmap_modified": False,
            "duration_s": 0.1,
            "honest_verdict": (
                "complete: exact ConstraintBench-style pilot only; "
                "no full ConstraintBench coverage claimed; fixture_count=15"
            ),
        },
    )
    _write_json(
        root,
        mod.EXP3213_REL_PATH,
        {
            "schema_version": "carnot.repair_gate_decision.v6",
            "experiment_id": "exp3213",
            "milestone": "2026.05.297",
            "repair_gate_state": "blocked",
            "repair_ladder_allowed": False,
            "blockers": [
                {"source_artifact": mod.EXP3209_REL_PATH.as_posix(), "code": "missing_mandatory_artifact"},
                {"source_artifact": mod.EXP3212_REL_PATH.as_posix(), "code": "missing_mandatory_artifact"},
            ],
            "conductor_file_modified": False,
            "active_roadmap_modified": False,
            "duration_s": 0.1,
            "honest_verdict": "complete: repair_gate_state=blocked; repair_ladder_allowed=false; blocker_count=6",
        },
    )
    _write_json(
        root,
        mod.EXP3214_REL_PATH,
        {
            "schema": "blocked_gate_check_v1",
            "experiment": 3214,
            "status": "blocked",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": (
                "1 of 1 gate(s) failed; first failure: "
                "exp3213-repair-gate-decision-v6.repair_gate_state "
                "(actual='blocked' == expected='unblocked')"
            ),
            "gates_evaluated": [
                {
                    "upstream": "exp3213-repair-gate-decision-v6",
                    "artifact_field": "repair_gate_state",
                    "expected": "unblocked",
                    "actual": "blocked",
                    "passed": False,
                }
            ],
            "duration_s": 0.0,
            "honest_verdict": "blocked_gate_check_failed",
        },
    )
    _write_json(
        root,
        mod.EXP3215_REL_PATH,
        {
            "schema": "carnot.fr11.evidence_gated_trace_replay_controller.v2",
            "schema_version": "1.0",
            "experiment_id": "experiment_3215_fr11_evidence_gated_trace_replay_controller_v2",
            "milestone": "2026.05.297",
            "promotion_allowed": True,
            "promotion_blockers": [],
            "replay_utility_label_count": 30,
            "routing_improvement_count": 21,
            "negative_control_regression_count": 0,
            "rollback_event_count": 0,
            "model_weight_update_claimed": False,
            "inference_substrate": {
                "controller_memory_replay_only": True,
                "model_weight_learning": False,
                "model_weight_training": False,
                "model_weight_mutation": False,
            },
            "conductor_file_modified": False,
            "active_roadmap_modified": False,
            "duration_s": 0.1,
            "honest_verdict": (
                "complete: fr11 evidence-gated trace replay controller v2 materialized; "
                "promotion_allowed=true; model_weight_update_claimed=false"
            ),
        },
    )
    _write_json(
        root,
        mod.EXP3216_REL_PATH,
        {
            "schema": "carnot.fr11.grounded_continuation_nonforgetting_queue.v1",
            "schema_version": "1.0",
            "experiment_id": "experiment_3216_fr11_grounded_continuation_nonforgetting_queue_v1",
            "milestone": "2026.05.297",
            "source_trace_artifact": mod.EXP3215_REL_PATH.as_posix(),
            "source_selection": {"source_kind": "exp3215", "fallback_used": False},
            "nonforgetting_queue_defined": True,
            "nonforgetting_budget_exceeded": False,
            "controller_memory_promotion_allowed": False,
            "model_weight_update_claimed": False,
            "inference_substrate": {
                "controller_memory_replay_only": True,
                "grounded_continuation_graph_only": True,
                "nonforgetting_queue_report_only": True,
                "model_weight_learning": False,
                "model_weight_training": False,
                "model_weight_mutation": False,
                "kan_sidecar_promotion_allowed": False,
            },
            "conductor_file_modified": False,
            "active_roadmap_modified": False,
            "duration_s": 0.1,
            "honest_verdict": (
                "complete: fr11 grounded-continuation trace graph and virtual "
                "nonforgetting queue materialized; model_weight_update_claimed=false; "
                "controller_memory_promotion_allowed=false"
            ),
        },
    )
    _write_text(
        root,
        mod.CONDUCTOR_LOG_REL_PATH,
        "\n".join(
            [
                "| 2026-05-27 21:04 UTC | Clean live SOTA verifier rerun v12 gated on exp320 | GATE_BLOCK | Pre-emptive skip: upstream retired (exp3208-full-local-sota-receipt-v5) |",
                "| 2026-05-27 21:30 UTC | Structured repair proposal preflight gated on exp3 | GATE_BLOCK | Pre-emptive skip: upstream retired (exp3208-full-local-sota-receipt-v5) |",
                "| 2026-05-27 21:49 UTC | Multi-turn repair ladder v7 gated on exp3213 repai | GATE_BLOCK | 1 of 1 gate(s) failed; first failure: exp3213-repair-gate-decision-v6.repair_gat |",
            ]
        )
        + "\n",
    )


def test_req_report_3217_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3217: OpenSpec declares matrix v31 before implementation."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3217" in spec
    assert "SCENARIO-REPORT-3217" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_report_3217_builds_v31_from_dot297_artifacts(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3217: .297 matrix preserves all claim boundaries."""

    _write_matrix_v30(tmp_path)
    _write_dot297_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=3.0, now_s=5.5)
    rows = {row["experiment_id"]: row for row in artifact["upstream_artifacts"]}
    missing = {row["experiment_id"]: row for row in artifact["missing_artifacts"]}
    gated = {row["experiment_id"]: row for row in artifact["gated_skipped_artifacts"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["cross_corpus_matrix_v31_ready"] is True
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert len(artifact["upstream_artifacts"]) == 12
    assert artifact["status_counts"] == {
        "clean": 4,
        "blocked": 3,
        "gated_skipped": 2,
        "diagnostic_only": 1,
        "retired": 0,
        "missing": 2,
    }
    assert artifact["publication_blocker_count"] == 92
    assert artifact["blocker_delta_from_v30"] == 7
    assert artifact["paper_ready"] is False
    assert artifact["honest_verdict"].startswith("complete:")

    assert rows["exp3205"]["status"] == "clean"
    assert rows["exp3206"]["status"] == "blocked"
    assert rows["exp3207"]["status"] == "blocked"
    assert rows["exp3208"]["status"] == "gated_skipped"
    assert rows["exp3209"]["status"] == "missing"
    assert rows["exp3210"]["status"] == "clean"
    assert rows["exp3211"]["status"] == "clean"
    assert rows["exp3212"]["status"] == "missing"
    assert rows["exp3213"]["status"] == "blocked"
    assert rows["exp3214"]["status"] == "gated_skipped"
    assert rows["exp3215"]["status"] == "clean"
    assert rows["exp3216"]["status"] == "diagnostic_only"

    assert set(missing) == {"exp3209", "exp3212"}
    assert missing["exp3209"]["gated_skip_evidence"]["status"] == "gated_skipped"
    assert missing["exp3212"]["gated_skip_evidence"]["status"] == "gated_skipped"
    assert set(gated) == {"exp3208", "exp3214"}
    assert artifact["local_sota_receipt_status"] == (
        "blocked_selected_python_torch_cuda_cuda_receipt_ready_false_full_receipt_gated_skipped"
    )
    assert artifact["clean_verifier_status"] == (
        "gated_skipped_missing_clean_verifier_v12_after_full_receipt_gate"
    )
    assert artifact["repair_status"] == "repair_gate_blocked_v6_ladder_gated_skipped_v7"
    assert artifact["repair_gate_status"] == "blocked"
    assert artifact["repair_ladder_status"] == "gated_skipped"
    assert artifact["context_fixture_status"] == (
        "available_ready_for_clean_verifier_fixture_count_30"
    )
    assert artifact["constraintbench_fixture_status"] == (
        "available_exact_pilot_fixture_count_15_no_full_coverage_claimed"
    )
    assert artifact["fr11_self_learning_status"] == (
        "controller_trace_replay_promoted_nonforgetting_queue_audit_only_"
        "no_model_weight_update_claimed"
    )
    assert artifact["fr11_claim_boundaries"] == {
        "controller_memory_promotion_allowed": True,
        "queue_promotion_allowed": False,
        "model_weight_update_claimed": False,
    }
    assert artifact["hardware_sampler_status"] == (
        "no_authenticated_hardware_transcript_no_speedup_tsu_kona_claim"
    )
    assert artifact["hardware_claim_boundaries"] == {
        "authenticated_hardware_transcript_present": False,
        "speedup_claim_allowed": False,
        "tsu_or_kona_claim_allowed": False,
    }
    assert artifact["next_top_gap"] == mod.NEXT_TOP_GAP
    assert artifact["conductor_file_modified"] is False
    assert artifact["active_roadmap_modified"] is False
    assert artifact["source_checksums"][mod.EXP3215_REL_PATH.as_posix()] == _sha256(
        tmp_path / mod.EXP3215_REL_PATH
    )


def test_req_report_3217_missing_previous_matrix_blocks(tmp_path: Path) -> None:
    """REQ-REPORT-3217: absent v30 authority blocks paper-readiness claims."""

    _write_dot297_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.0)

    assert artifact["cross_corpus_matrix_v31_ready"] is False
    assert artifact["paper_ready"] is False
    assert artifact["publication_blocker_count"] == 7
    assert artifact["blocker_delta_from_v30"] is None
    assert "previous matrix v30 is missing or not ready" in artifact["invariant_violations"]
    assert artifact["honest_verdict"].startswith("blocked_matrix_v31_preconditions")


def test_req_report_3217_write_artifact_and_helper_edges(tmp_path: Path) -> None:
    """REQ-REPORT-3217: writer and helper branches are deterministic."""

    _write_matrix_v30(tmp_path)
    _write_dot297_sources(tmp_path)

    output = mod.write_artifact(tmp_path, started_s=2.0, now_s=4.0)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["cross_corpus_matrix_v31_ready"] is True
    assert mod._status_counts([{"status": "bad"}])["missing"] == 1
    assert mod._normal_status("gated-skip") == "gated_skipped"
    assert mod._normal_status("unknown") == "missing"
    assert mod._experiment_id({"experiment_id": "exp1"}, "fallback") == "exp1"
    assert mod._experiment_id({"experiment": 7}, "fallback") == "exp7"
    assert mod._experiment_id({}, "fallback") == "fallback"
    assert mod._boolish(False) is False
    assert mod._boolish("true") is True
    assert mod._boolish("false") is False
    assert mod._int_value(True) == 0
    assert mod._int_value("5") == 0
    assert mod._int_value(5) == 5
    assert mod._gate_skip_evidence("exp9999", "") == {"status": "absent"}
    assert mod._source_payload(tmp_path / "missing", mod.SOURCE_SPECS[0])["status"] == "missing"
    assert mod._source_payload(tmp_path, mod.SOURCE_SPECS[0])["sha256"] == _sha256(
        tmp_path / mod.EXP3205_REL_PATH
    )

    assert mod._classification_status(
        "exp3206", {"cuda_init_clean": True, "cuda_env_diagnosed": True}
    )[0] == "clean"
    assert mod._classification_status("exp3206", {"cuda_env_diagnosed": False})[0] == "blocked"
    assert mod._classification_status("exp3207", {"cuda_receipt_ready": True})[0] == "clean"
    assert mod._classification_status("exp3208", {"clean_rerun_allowed": True})[0] == "clean"
    assert mod._classification_status(
        "exp3210", {"fixture_count": 1, "ready_for_clean_verifier": False}
    )[0] == "diagnostic_only"
    assert mod._classification_status(
        "exp3211", {"fixture_count": 1, "ready_for_clean_verifier": False}
    )[0] == "diagnostic_only"
    assert mod._classification_status("exp3213", {"repair_gate_state": "unblocked"})[0] == (
        "clean"
    )
    assert mod._classification_status("exp3214", {"repair_ladder_complete": True})[0] == "clean"
    assert mod._classification_status(
        "exp3215", {"promotion_allowed": True, "model_weight_update_claimed": True}
    )[0] == "blocked"
    assert mod._classification_status(
        "exp3215",
        {
            "promotion_allowed": False,
            "model_weight_update_claimed": False,
            "negative_control_regression_count": 0,
        },
    )[0] == "diagnostic_only"
    assert mod._classification_status("exp3216", {"model_weight_update_claimed": True})[0] == (
        "blocked"
    )
    assert mod._classification_status(
        "exp3216",
        {"controller_memory_promotion_allowed": True, "model_weight_update_claimed": False},
    )[0] == "clean"
    assert mod._classification_status("exp9999", {"artifact": "unknown"})[0] == "blocked"


def test_req_report_3217_fail_closed_branch_matrix(tmp_path: Path) -> None:
    """REQ-REPORT-3217: defensive branches keep future artifacts bounded."""

    assert mod._classification_status("exp3205", {"activation_ready": False})[0] == "blocked"
    assert mod._classification_status("exp3208", {"clean_rerun_allowed": False})[0] == "blocked"
    assert mod._classification_status("exp3209", {"schema": "blocked_gate_check_v1"})[0] == (
        "gated_skipped"
    )
    assert mod._classification_status("exp3209", {"clean_verifier_ready": True})[0] == "clean"
    assert mod._classification_status("exp3209", {"clean_verifier_ready": False})[0] == "blocked"
    assert mod._classification_status("exp3210", {"fixture_count": 0})[0] == "blocked"
    assert mod._classification_status("exp3211", {"fixture_count": 0})[0] == "blocked"
    assert mod._classification_status("exp3212", {"schema": "blocked_gate_check_v1"})[0] == (
        "gated_skipped"
    )
    assert mod._classification_status(
        "exp3212", {"ready_for_repair_gate": True, "repair_correctness_claimed": False}
    )[0] == "clean"
    assert mod._classification_status(
        "exp3212", {"ready_for_repair_gate": True, "repair_correctness_claimed": True}
    )[0] == "blocked"
    assert mod._classification_status("exp3214", {"repair_ladder_complete": False})[0] == (
        "blocked"
    )
    assert mod._classification_status(
        "exp3215", {"model_weight_update_claimed": False, "negative_control_regression_count": 2}
    )[0] == "blocked"
    assert mod._classification_status(
        "exp3216",
        {"model_weight_update_claimed": False, "nonforgetting_queue_defined": False},
    )[0] == "blocked"

    assert (
        mod._local_sota_receipt_status(
            {"exp3207": {"cuda_receipt_ready": True}},
            [{"experiment_id": "exp3208", "status": "clean"}],
        )
        == "passed_cuda_receipt_ready_full_local_sota_clean_rerun_allowed"
    )
    assert (
        mod._local_sota_receipt_status({"exp3207": {}}, [])
        == "missing_full_local_sota_receipt_v5"
    )
    assert (
        mod._local_sota_receipt_status(
            {"exp3207": {}}, [{"experiment_id": "exp3208", "status": "blocked"}]
        )
        == "blocked_no_full_local_sota_receipt"
    )

    assert (
        mod._clean_verifier_status([{"experiment_id": "exp3209", "status": "clean"}])
        == "clean_live_verifier_v12_ready"
    )
    assert (
        mod._clean_verifier_status([{"experiment_id": "exp3209", "status": "gated_skipped"}])
        == "gated_skipped_clean_verifier_v12_after_full_receipt_gate"
    )
    assert mod._clean_verifier_status([]) == "missing_clean_verifier_v12"
    assert (
        mod._clean_verifier_status([{"experiment_id": "exp3209", "status": "blocked"}])
        == "blocked_clean_verifier_v12"
    )

    assert (
        mod._repair_gate_status(
            {"exp3213": {}}, [{"experiment_id": "exp3213", "status": "clean"}]
        )
        == "unblocked"
    )
    assert mod._repair_gate_status({"exp3213": {}}, []) == "missing"
    assert mod._repair_status("unblocked", "clean") == "repair_ready"
    assert mod._repair_status("missing", "clean") == "missing_repair_gate_or_ladder"
    assert mod._repair_status("blocked", "blocked") == "blocked_repair"

    assert mod._context_fixture_status({"exp3210": {}}, []) == "missing_context_shortcut_fixtures"
    assert (
        mod._context_fixture_status(
            {"exp3210": {"fixture_count": 2, "ready_for_clean_verifier": False}},
            [{"experiment_id": "exp3210", "status": "diagnostic_only"}],
        )
        == "available_not_clean_verifier_ready_fixture_count_2"
    )
    assert (
        mod._context_fixture_status(
            {"exp3210": {"fixture_count": 0}},
            [{"experiment_id": "exp3210", "status": "blocked"}],
        )
        == "blocked_context_shortcut_fixtures"
    )
    assert (
        mod._constraintbench_fixture_status({"exp3211": {}}, [])
        == "missing_constraintbench_fixture_pilot"
    )
    assert (
        mod._constraintbench_fixture_status(
            {
                "exp3211": {
                    "fixture_count": 2,
                    "ready_for_clean_verifier": True,
                    "honest_verdict": "complete: exact pilot",
                }
            },
            [{"experiment_id": "exp3211", "status": "clean"}],
        )
        == "available_exact_pilot_fixture_count_2"
    )
    assert (
        mod._constraintbench_fixture_status(
            {"exp3211": {"fixture_count": 2, "ready_for_clean_verifier": False}},
            [{"experiment_id": "exp3211", "status": "diagnostic_only"}],
        )
        == "available_not_clean_verifier_ready_fixture_count_2"
    )
    assert (
        mod._constraintbench_fixture_status(
            {"exp3211": {"fixture_count": 0}},
            [{"experiment_id": "exp3211", "status": "blocked"}],
        )
        == "blocked_constraintbench_fixture_pilot"
    )

    assert (
        mod._fr11_self_learning_status(
            {"exp3215": {}, "exp3216": {}},
            [],
            {"model_weight_update_claimed": True},
        )
        == "blocked_fr11_model_weight_update_claimed"
    )
    assert (
        mod._fr11_self_learning_status(
            {"exp3215": {}, "exp3216": {}},
            [
                {"experiment_id": "exp3215", "status": "clean"},
                {"experiment_id": "exp3216", "status": "clean"},
            ],
            {"model_weight_update_claimed": False},
        )
        == "controller_trace_replay_and_queue_promoted_no_model_weight_update_claimed"
    )
    assert (
        mod._fr11_self_learning_status(
            {"exp3215": {}, "exp3216": {}},
            [{"experiment_id": "exp3215", "status": "clean"}],
            {"model_weight_update_claimed": False},
        )
        == "controller_trace_replay_promoted_queue_missing_no_model_weight_update_claimed"
    )
    assert (
        mod._fr11_self_learning_status(
            {"exp3215": {}, "exp3216": {}}, [], {"model_weight_update_claimed": False}
        )
        == "missing_fr11_self_learning_artifacts"
    )
    assert (
        mod._fr11_self_learning_status(
            {"exp3215": {"promotion_allowed": False}, "exp3216": {}},
            [{"experiment_id": "exp3215", "status": "blocked"}],
            {"model_weight_update_claimed": False},
        )
        == "blocked_fr11_self_learning"
    )

    assert mod._hardware_claim_boundaries(
        {"exp": {"authenticated_hardware_transcript_present": True, "speedup_claim_allowed": True}},
        {},
    ) == {
        "authenticated_hardware_transcript_present": True,
        "speedup_claim_allowed": True,
        "tsu_or_kona_claim_allowed": False,
    }
    assert mod._hardware_claim_boundaries({"exp": {"kona_execution_claimed": True}}, {}) == {
        "authenticated_hardware_transcript_present": False,
        "speedup_claim_allowed": False,
        "tsu_or_kona_claim_allowed": False,
    }
    assert (
        mod._hardware_sampler_status(
            {
                "authenticated_hardware_transcript_present": True,
                "speedup_claim_allowed": True,
                "tsu_or_kona_claim_allowed": False,
            }
        )
        == "authenticated_hardware_claim_allowed"
    )

    assert mod._required_evidence_blocked_or_missing(
        "blocked",
        "blocked",
        "blocked",
        "blocked_fr11",
        "blocked_hardware",
    ) == [
        "local_sota_receipt",
        "clean_verifier",
        "repair",
        "fr11_self_learning",
        "hardware_sampler",
    ]
    assert (
        mod._next_top_gap(
            "passed_cuda_receipt_ready_full_local_sota_clean_rerun_allowed",
            "blocked",
            "repair_ready",
            "controller_trace_replay_ok",
            "authenticated_hardware_claim_allowed",
        )
        == "clean_live_verifier_v12_gate_clearance"
    )
    assert (
        mod._next_top_gap(
            "passed_cuda_receipt_ready_full_local_sota_clean_rerun_allowed",
            "clean_live_verifier_v12_ready",
            "blocked",
            "controller_trace_replay_ok",
            "authenticated_hardware_claim_allowed",
        )
        == "repair_gate_v6_unblock_and_ladder_v7_execution"
    )
    assert (
        mod._next_top_gap(
            "passed_cuda_receipt_ready_full_local_sota_clean_rerun_allowed",
            "clean_live_verifier_v12_ready",
            "repair_ready",
            "blocked_fr11",
            "authenticated_hardware_claim_allowed",
        )
        == "fr11_controller_memory_nonforgetting_promotion"
    )
    assert (
        mod._next_top_gap(
            "passed_cuda_receipt_ready_full_local_sota_clean_rerun_allowed",
            "clean_live_verifier_v12_ready",
            "repair_ready",
            "controller_trace_replay_ok",
            "blocked_hardware",
        )
        == "authenticated_hardware_transcript_or_explicit_no_speedup_boundary"
    )
    assert (
        mod._next_top_gap(
            "passed_cuda_receipt_ready_full_local_sota_clean_rerun_allowed",
            "clean_live_verifier_v12_ready",
            "repair_ready",
            "controller_trace_replay_ok",
            "authenticated_hardware_claim_allowed",
        )
        == "publication_blocker_retirement_review"
    )

    assert any(
        "status_counts keys" in item
        for item in mod._invariant_violations(
            {"cross_corpus_matrix_v30_ready": True}, [], {}, 0, None, [], {}, {}
        )
    )
    assert any(
        "status_counts do not sum" in item
        for item in mod._invariant_violations(
            {"cross_corpus_matrix_v30_ready": True},
            [],
            {status: 1 for status in mod.STATUSES},
            0,
            None,
            [],
            {},
            {},
        )
    )
    assert any(
        "publication_blocker_count does not reconcile" in item
        for item in mod._invariant_violations(
            {"cross_corpus_matrix_v30_ready": True},
            [],
            {status: 0 for status in mod.STATUSES},
            5,
            1,
            [],
            {},
            {},
        )
    )
    assert any(
        "hardware speedup claim lacks authenticated transcript" in item
        for item in mod._invariant_violations(
            {"cross_corpus_matrix_v30_ready": True},
            [],
            {status: 0 for status in mod.STATUSES},
            0,
            None,
            [],
            {"speedup_claim_allowed": True, "authenticated_hardware_transcript_present": False},
            {},
        )
    )
    assert any(
        "FR-11 model-weight update claim" in item
        for item in mod._invariant_violations(
            {"cross_corpus_matrix_v30_ready": True},
            [],
            {status: 0 for status in mod.STATUSES},
            0,
            None,
            [],
            {},
            {"model_weight_update_claimed": True},
        )
    )

    assert mod._gate_skip_evidence("exp3209", "no matching gate line") == {"status": "absent"}
    assert mod._row([], "exp9999") == {}
    assert mod._prior_publication_blocker_count({"publication_blocker_count": True}) is None
    assert mod._model_weight_update_claimed({"model_weight_update_performed": True}) is True
    assert mod._read_text(tmp_path / "missing.txt") == ""
