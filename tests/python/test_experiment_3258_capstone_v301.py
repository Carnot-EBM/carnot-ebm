"""Tests for Exp 3258 milestone .301 capstone.

Spec refs: REQ-REPORT-3258, SCENARIO-REPORT-3258.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from carnot.reporting import capstone_v301_3258 as mod


REQUIRED_FIELDS = {
    "experiment_id",
    "task_id",
    "milestone",
    "inference_substrate",
    "principle_annotations",
    "capstone_v301_ready",
    "paper_ready",
    "publication_blocker_count",
    "blocker_delta_from_v300",
    "local_sota_receipt_status",
    "prompt_injection_v4_status",
    "dccd_severa_preflight_status",
    "fr11_lifelong_retention_status",
    "pdit_potts_status",
    "next_top_gap",
    "recommended_next_milestone_theme",
    "protected_files_untouched",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path, payload: Mapping[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(root: Path, rel_path: Path, text: str) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _gate_payload(experiment: int, summary: str) -> dict[str, Any]:
    return {
        "experiment": experiment,
        "schema": "blocked_gate_check_v1",
        "status": "blocked",
        "blocked_at_layer": "conductor_pre_gate",
        "gate_check_summary": summary,
        "honest_verdict": "blocked_gate_check_failed",
    }


def _matrix_v34(*, paper_ready: bool = False, blockers: int = 106) -> dict[str, Any]:
    runtime_ready = paper_ready and blockers == 0
    prompt_ready = paper_ready and blockers == 0
    complete_rows = [
        {"experiment_id": spec.experiment_id, "path": spec.path.as_posix(), "status": "complete"}
        for spec in mod.EXPECTED_DOT301_SOURCES
    ]
    blocked_rows = [
        {"experiment_id": "exp3246", "path": mod.EXP3246_REL_PATH.as_posix(), "status": "complete"},
        {"experiment_id": "exp3247", "path": mod.EXP3247_REL_PATH.as_posix(), "status": "blocked"},
        {"experiment_id": "exp3248", "path": mod.EXP3248_REL_PATH.as_posix(), "status": "gate_blocked"},
        {"experiment_id": "exp3250", "path": mod.EXP3250_REL_PATH.as_posix(), "status": "gate_blocked"},
        {"experiment_id": "exp3251", "path": mod.EXP3251_REL_PATH.as_posix(), "status": "complete"},
        {"experiment_id": "exp3253", "path": mod.EXP3253_REL_PATH.as_posix(), "status": "gate_blocked"},
        {"experiment_id": "exp3255", "path": mod.EXP3255_REL_PATH.as_posix(), "status": "complete"},
        {"experiment_id": "exp3256", "path": mod.EXP3256_REL_PATH.as_posix(), "status": "complete"},
    ]
    missing_rows = [
        {"experiment_id": "exp3249", "path": mod.EXP3249_REL_PATH.as_posix(), "status": "gate_blocked"},
        {"experiment_id": "exp3252", "path": mod.EXP3252_REL_PATH.as_posix(), "status": "gate_blocked"},
        {"experiment_id": "exp3254", "path": mod.EXP3254_REL_PATH.as_posix(), "status": "gate_blocked"},
    ]
    return {
        "artifact": "experiment_3257_cross_corpus_matrix_v34",
        "experiment_id": "exp3257",
        "task_id": "exp3257-cross-corpus-matrix-v34",
        "matrix_v34_ready": True,
        "paper_ready": paper_ready,
        "publication_blocker_count": blockers,
        "next_top_gap": (
            "publication_blocker_retirement_review"
            if paper_ready
            else "keep_exp3248_blocked_repair_cuda_runtime"
        ),
        "runtime_receipt_status": {
            "selected_python_cuda": {
                "artifact_status": "complete" if runtime_ready else "gate_blocked",
                "root_cause_class": "" if runtime_ready else "cuda_bindings_runtime_failure",
                "next_smoke_allowed": runtime_ready,
                "selected_python_cuda_repaired_candidate": runtime_ready,
                "cuda_python_smoke_passed": runtime_ready,
                "state": (
                    "selected_python_cuda_smoke_passed"
                    if runtime_ready
                    else "blocked_root_cause_surgery_next_smoke_not_allowed"
                ),
            },
            "llama_cpp_cuda": {
                "artifact_status": "complete" if runtime_ready else "gate_blocked",
                "llama_cpp_cuda_receipt_ready": runtime_ready,
            },
            "sota_gguf_receipt": {
                "artifact_status": "complete" if runtime_ready else "gate_blocked",
                "sota_gguf_receipt_ready": runtime_ready,
                "mandatory_model_receipt_count": 3 if runtime_ready else 0,
                "models_used": ["unsloth/Qwen3.6-35B-A3B-GGUF"] if runtime_ready else [],
            },
            "clean_rerun_allowed": runtime_ready,
            "receipt_chain_ready": runtime_ready,
            "next_action": "keep_exp3248_blocked_repair_cuda_runtime",
        },
        "prompt_injection_status": {
            "manifest_v2": {
                "artifact_status": "complete",
                "v4_manifest_v2_ready": True,
                "constraint_tax_control_plan_ready": True,
                "garak_config_ready": True,
                "no_llm_invoked": True,
            },
            "constraint_tax_diagnostic": {
                "status": "measured" if prompt_ready else "plan_ready_no_measurement",
                "constraint_tax_delta_accuracy_or_parse": 0.0 if prompt_ready else None,
            },
            "teacher_labels": {
                "artifact_status": "complete" if prompt_ready else "gate_blocked",
                "teacher_label_shard_ready": prompt_ready,
                "completed_free_reasoning_count": 4 if prompt_ready else 0,
                "completed_schema_constrained_count": 4 if prompt_ready else 0,
            },
            "kan_shard": {
                "artifact_status": "complete" if prompt_ready else "gate_blocked",
                "train_eval_completed": prompt_ready,
                "headline_claim_allowed": False,
            },
            "repair_proposal_preflight": {
                "artifact_status": "complete" if prompt_ready else "gate_blocked",
                "structured_proposal_preflight_ready": prompt_ready,
                "repair_acceptance_claimed": False,
            },
            "publication_evidence_ready": prompt_ready,
        },
        "fr11_lifelong_status": {
            "artifact_status": "complete",
            "continuous_self_learning_task": True,
            "fr11_controller_update_ready": True,
            "lifelong_eval_ready": True,
            "failure_trace_count": 44,
            "heldout_replay_count": 9,
            "retention_score": 1.0,
            "adaptation_score": 1.0,
            "forgetting_score": 1.0,
            "negative_control_regression_count": 0,
            "doomed_rerun_avoidance_count": 43,
            "model_weight_update_claimed": False,
            "no_new_llm_invoked": True,
        },
        "pdit_potts_status": {
            "artifact_status": "complete",
            "pdit_potts_mapping_ready": True,
            "candidate_verifier_row_type_count": 2,
            "q_state_energy_mapping_count": 2,
            "exact_fallback_preserved": True,
            "hardware_speedup_claim_allowed": False,
            "retired_pimi_scope_reopened": False,
            "thrml_scaling_sweep_reopened": False,
        },
        "artifacts_expected": [
            {
                "experiment_id": spec.experiment_id,
                "task_id": spec.task_id,
                "path": spec.path.as_posix(),
                "role": spec.role,
                "ready_field": spec.ready_field,
            }
            for spec in mod.EXPECTED_DOT301_SOURCES
        ],
        "artifacts_found": complete_rows if paper_ready else blocked_rows,
        "artifacts_missing": [] if paper_ready else missing_rows,
        "gate_blocked_artifacts": [] if paper_ready else [
            row for row in blocked_rows + missing_rows if row["status"] == "gate_blocked"
        ],
        "honest_verdict": "complete: matrix_v34_ready=true; paper_ready=false",
    }


def _write_sources(root: Path, *, paper_ready: bool = False, blockers: int = 106) -> None:
    _write_json(root, mod.MATRIX_V34_REL_PATH, _matrix_v34(paper_ready=paper_ready, blockers=blockers))
    _write_json(
        root,
        mod.CAPSTONE_V300_REL_PATH,
        {
            "experiment_id": "exp3245",
            "task_id": "exp3245-capstone-v300",
            "capstone_v300_ready": True,
            "paper_ready": False,
            "publication_blocker_count": 106,
            "honest_verdict": "complete: capstone_v300_ready=true; publication_blocker_count=106",
        },
    )
    _write_json(
        root,
        mod.EXP3246_REL_PATH,
        {
            "experiment_id": "exp3246",
            "archive_v300_activate_v301_ready": True,
            "prior_publication_blocker_count": 106,
            "honest_verdict": "complete: archive_v300_activate_v301_ready=true",
        },
    )
    _write_json(
        root,
        mod.EXP3247_REL_PATH,
        {
            "experiment_id": "exp3247",
            "cuda_root_cause_class": "" if paper_ready else "cuda_bindings_runtime_failure",
            "selected_python_cuda_repaired_candidate": paper_ready,
            "next_smoke_allowed": paper_ready,
            "recommended_next_task": (
                "publication_blocker_retirement_review"
                if paper_ready
                else "keep_exp3248_blocked_repair_cuda_runtime"
            ),
            "honest_verdict": f"complete: next_smoke_allowed={str(paper_ready).lower()}",
        },
    )
    if paper_ready:
        _write_json(root, mod.EXP3248_REL_PATH, {"experiment_id": "exp3248", "cuda_python_smoke_passed": True})
        _write_json(
            root,
            mod.EXP3249_REL_PATH,
            {"experiment_id": "exp3249", "llama_cpp_cuda_receipt_ready": True},
        )
        _write_json(
            root,
            mod.EXP3250_REL_PATH,
            {
                "experiment_id": "exp3250",
                "sota_gguf_receipt_ready": True,
                "clean_rerun_allowed": True,
                "mandatory_model_receipt_count": 3,
            },
        )
    else:
        _write_json(
            root, mod.EXP3248_REL_PATH, _gate_payload(3248, "exp3247.next_smoke_allowed failed")
        )
        _write_json(root, mod.EXP3250_REL_PATH, _gate_payload(3250, "exp3249 artifact not found"))
    _write_json(
        root,
        mod.EXP3251_REL_PATH,
        {
            "experiment_id": "exp3251",
            "v4_manifest_v2_ready": True,
            "constraint_tax_control_plan_ready": True,
            "no_llm_invoked": True,
            "honest_verdict": "complete: v4_manifest_v2_ready=true",
        },
    )
    if paper_ready:
        _write_json(
            root,
            mod.EXP3252_REL_PATH,
            {
                "experiment_id": "exp3252",
                "teacher_label_shard_ready": True,
                "constraint_tax_delta_accuracy_or_parse": 0.0,
            },
        )
        _write_json(root, mod.EXP3253_REL_PATH, {"experiment_id": "exp3253", "train_eval_completed": True})
        _write_json(
            root,
            mod.EXP3254_REL_PATH,
            {
                "experiment_id": "exp3254",
                "structured_proposal_preflight_ready": True,
                "repair_acceptance_claimed": False,
            },
        )
    else:
        _write_json(root, mod.EXP3253_REL_PATH, _gate_payload(3253, "exp3252 artifact not found"))
    _write_json(
        root,
        mod.EXP3255_REL_PATH,
        {
            "experiment_id": "exp3255",
            "continuous_self_learning_task": True,
            "fr11_controller_update_ready": True,
            "lifelong_eval_ready": True,
            "retention_score": 1.0,
            "adaptation_score": 1.0,
            "forgetting_score": 1.0,
            "negative_control_regression_count": 0,
            "model_weight_update_claimed": False,
            "no_new_llm_invoked": True,
            "honest_verdict": "complete: fr11 lifelong retention audit ready=true",
        },
    )
    _write_json(
        root,
        mod.EXP3256_REL_PATH,
        {
            "experiment_id": "exp3256",
            "pdit_potts_mapping_ready": True,
            "exact_fallback_preserved": True,
            "hardware_speedup_claim_allowed": False,
            "retired_pimi_scope_reopened": False,
            "thrml_scaling_sweep_reopened": False,
            "honest_verdict": "complete: p-dit/Potts diagnostic ready",
        },
    )
    _write_text(
        root,
        mod.CONDUCTOR_LOG_REL_PATH,
        "\n".join(
            [
                "| 2026-05-28 07:18 UTC | llama.cpp CUDA receipt smoke v3 gated on selected- | GATE_BLOCK | skip |",
                "| 2026-05-28 07:51 UTC | Prompt-injection teacher-label shard v2 gated on S | GATE_BLOCK | skip |",
                "| 2026-05-28 07:57 UTC | DCCD/SEVerA structured proposal preflight v2 gated | GATE_BLOCK | skip |",
            ]
        )
        + "\n",
    )


def test_req_report_3258_spec_anchor_exists() -> None:
    """REQ-REPORT-3258: OpenSpec declares the capstone before implementation."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3258" in spec
    assert "SCENARIO-REPORT-3258" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert Path(mod.__file__).exists()


def test_scenario_report_3258_builds_v301_capstone_from_matrix_v34(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3258: capstone reports no blocker reduction and next runtime gap."""

    _write_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=10.0, now_s=13.5)
    sources = {row["experiment_id"]: row for row in artifact["source_artifacts"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["experiment_id"] == "exp3258"
    assert artifact["task_id"] == "exp3258-capstone-v301"
    assert artifact["milestone"] == "2026.05.301"
    assert artifact["capstone_v301_ready"] is True
    assert artifact["paper_ready"] is False
    assert artifact["publication_blocker_count"] == 106
    assert artifact["blocker_delta_from_v300"] == 0
    assert artifact["duration_s"] == pytest.approx(3.5)

    local_sota = artifact["local_sota_receipt_status"]
    assert local_sota["status"] == "blocked"
    assert local_sota["advanced"] is False
    assert local_sota["selected_python_cuda"]["next_smoke_allowed"] is False
    assert local_sota["selected_python_cuda"]["root_cause_class"] == "cuda_bindings_runtime_failure"
    assert local_sota["llama_cpp_cuda"]["artifact_status"] == "gate_blocked"
    assert local_sota["sota_gguf_receipt"]["sota_gguf_receipt_ready"] is False
    assert local_sota["receipt_chain_ready"] is False

    prompt = artifact["prompt_injection_v4_status"]
    assert prompt["status"] == "gate_blocked"
    assert prompt["constraint_tax_plan_advanced"] is True
    assert prompt["teacher_labels"]["artifact_status"] == "gate_blocked"
    assert prompt["teacher_labels"]["teacher_label_shard_ready"] is False
    assert prompt["kan_shard"]["train_eval_completed"] is False

    dccd = artifact["dccd_severa_preflight_status"]
    assert dccd["status"] == "gate_blocked"
    assert dccd["structured_proposal_preflight_ready"] is False
    assert dccd["repair_acceptance_claimed"] is False

    fr11 = artifact["fr11_lifelong_retention_status"]
    assert fr11["status"] == "complete"
    assert fr11["advanced"] is True
    assert fr11["retention_score"] == 1.0
    assert fr11["model_weight_update_claimed"] is False

    pdit = artifact["pdit_potts_status"]
    assert pdit["status"] == "complete"
    assert pdit["advanced"] is True
    assert pdit["exact_fallback_preserved"] is True
    assert pdit["hardware_speedup_claim_allowed"] is False

    assert artifact["next_top_gap"] == "keep_exp3248_blocked_repair_cuda_runtime"
    assert "selected-Python CUDA" in artifact["recommended_next_milestone_theme"]
    assert artifact["protected_files_untouched"] == {
        "research-roadmap.yaml": True,
        "scripts/research_conductor.py": True,
    }
    assert "Do NOT push." in artifact["operator_safe_notes"]
    assert "Do NOT modify scripts/research_conductor.py." in artifact["operator_safe_notes"]
    assert sources["exp3257"]["present"] is True
    assert sources["exp3249"]["present"] is False
    assert sources["exp3252"]["status"] == "gate_blocked"
    assert artifact["honest_verdict"].startswith("complete:")
    assert "paper_ready=false" in artifact["honest_verdict"]
    mod.validate_artifact(artifact)


def test_req_report_3258_writer_fail_closed_and_paper_ready_gate(tmp_path: Path) -> None:
    """REQ-REPORT-3258: writer persists output and paper readiness needs zero blockers."""

    _write_sources(tmp_path)
    output = mod.write_artifact(tmp_path, started_s=1.0, now_s=2.0)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["capstone_v301_ready"] is True

    empty = mod.build_artifact(tmp_path / "empty", started_s=1.0, now_s=1.0)
    assert empty["capstone_v301_ready"] is False
    assert empty["paper_ready"] is False
    assert empty["publication_blocker_count"] == 106
    assert empty["blocker_delta_from_v300"] == 0
    assert "matrix v34 is missing or not ready" in empty["invariant_violations"]
    assert "capstone v300 is missing or not ready" in empty["invariant_violations"]

    ready_root = tmp_path / "ready"
    _write_sources(ready_root, paper_ready=True, blockers=0)
    ready = mod.build_artifact(ready_root, started_s=1.0, now_s=1.0)
    assert ready["paper_ready"] is True
    assert ready["publication_blocker_count"] == 0
    assert ready["blocker_delta_from_v300"] == -106
    assert ready["honest_verdict"].startswith("complete:")
    assert "paper_ready=true" in ready["honest_verdict"]


def test_req_report_3258_defensive_helpers_and_validation(tmp_path: Path) -> None:
    """REQ-REPORT-3258: helper branches remain explicit for malformed evidence."""

    assert mod.read_json_object(tmp_path / "missing.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{bad", encoding="utf-8")
    assert mod.read_json_object(bad) == {}
    non_object = tmp_path / "list.json"
    non_object.write_text("[]", encoding="utf-8")
    assert mod.read_json_object(non_object) == {}
    assert mod.sha256_file(tmp_path / "missing.json") is None
    assert mod._read_text(tmp_path / "missing.log") == ""
    assert mod._as_mapping({"x": 1}) == {"x": 1}
    assert mod._as_mapping([]) == {}
    assert mod._int_value(7) == 7
    assert mod._int_value(True) == 0
    assert mod._duration(5.0, 4.0) == 0.0
    assert mod._normal_status("gate-blocked") == "gate_blocked"
    assert mod._normal_status("odd") == "missing"

    assert mod._status_from_ready(True, "missing") == "complete"
    assert mod._status_from_ready(False, "gate_blocked") == "gate_blocked"
    assert mod._status_from_ready(False, "complete") == "blocked"
    assert mod._source_status(_gate_payload(1, "blocked"), {}, "ready", True) == "gate_blocked"
    assert mod._source_status({"ready": True}, {}, "ready", True) == "complete"
    assert mod._source_status({"ready": False}, {}, "ready", True) == "missing"
    assert mod._is_gate_blocked({"schema": "blocked_gate_check_v1"}) is True
    assert mod._is_gate_blocked({"blocked_at_layer": "conductor_pre_gate"}) is True
    assert mod._is_gate_blocked({"honest_verdict": "blocked_gate_check_failed"}) is True
    assert mod._is_gate_blocked({"honest_verdict": "complete"}) is False

    assert mod._recommended_next_milestone_theme("prompt_injection_teacher_labels_after_sota_receipt").startswith(
        "Recover prompt-injection"
    )
    assert mod._recommended_next_milestone_theme("dccd_severa_preflight_after_clean_sota_receipt").startswith(
        "Run DCCD/SEVerA"
    )
    assert mod._recommended_next_milestone_theme("fr11_lifelong_retention_audit").startswith(
        "Continue FR-11"
    )
    assert mod._recommended_next_milestone_theme("pdit_potts_diagnostic_mapping").startswith(
        "Extend p-dit"
    )
    assert mod._recommended_next_milestone_theme("publication_blocker_retirement_review").startswith(
        "Review publication"
    )

    assert mod._required_evidence_exists(
        {"receipt_chain_ready": True},
        {"publication_evidence_ready": True},
        {"structured_proposal_preflight_ready": True},
        {"status": "complete"},
        {"status": "complete"},
    ) is True
    assert mod._required_evidence_exists({}, {}, {}, {}, {}) is False
    assert mod._next_top_gap({}, {"status": "complete"}, {}, {}, {}, {}, "") == (
        "prompt_injection_teacher_labels_and_kan_shard_after_sota_receipt"
    )
    assert mod._next_top_gap(
        {},
        {"status": "complete"},
        {"publication_evidence_ready": True},
        {"status": "blocked"},
        {},
        {},
        "",
    ) == "dccd_severa_preflight_after_clean_sota_receipt"
    assert mod._next_top_gap(
        {},
        {"status": "complete"},
        {"publication_evidence_ready": True},
        {"status": "complete"},
        {"status": "blocked"},
        {},
        "",
    ) == "fr11_lifelong_retention_audit"
    assert mod._next_top_gap(
        {},
        {"status": "complete"},
        {"publication_evidence_ready": True},
        {"status": "complete"},
        {"status": "complete"},
        {"status": "blocked"},
        "",
    ) == "pdit_potts_diagnostic_mapping"

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.0)
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({})
    with pytest.raises(ValueError, match="experiment_id"):
        mod.validate_artifact(artifact | {"experiment_id": "bad"})
    with pytest.raises(ValueError, match="task_id"):
        mod.validate_artifact(artifact | {"task_id": "bad"})
    with pytest.raises(ValueError, match="milestone"):
        mod.validate_artifact(artifact | {"milestone": "bad"})
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(artifact | {"inference_substrate": "bad"})
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(artifact | {"honest_verdict": "blocked"})
    with pytest.raises(ValueError, match="publication_blocker_count"):
        mod.validate_artifact(artifact | {"publication_blocker_count": -1})
    with pytest.raises(ValueError, match="paper_ready cannot be true"):
        mod.validate_artifact(artifact | {"paper_ready": True, "publication_blocker_count": 1})
    with pytest.raises(ValueError, match="protected_files_untouched"):
        mod.validate_artifact(
            artifact | {"protected_files_untouched": {"scripts/research_conductor.py": True}}
        )
