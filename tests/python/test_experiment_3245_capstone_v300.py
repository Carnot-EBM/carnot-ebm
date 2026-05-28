"""Tests for Exp 3245 milestone .300 capstone.

Spec refs: REQ-REPORT-3245, SCENARIO-REPORT-3245.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from carnot.reporting import capstone_v300_3245 as mod


REQUIRED_FIELDS = {
    "experiment_id",
    "task_id",
    "milestone",
    "inference_substrate",
    "principle_annotations",
    "capstone_v300_ready",
    "paper_ready",
    "publication_blocker_count",
    "blocker_delta_from_v299",
    "local_sota_receipt_state",
    "prompt_injection_v4_state",
    "fr11_failure_memory_state",
    "next_top_gap",
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _matrix_v33() -> dict[str, Any]:
    return {
        "experiment_id": "exp3244",
        "task_id": "exp3244-cross-corpus-matrix-v33",
        "cross_corpus_matrix_v33_ready": True,
        "paper_ready": False,
        "publication_blocker_count": 106,
        "publication_blocker_delta_from_v299": 6,
        "next_top_gap": "repair_selected_python_torch_cuda_before_exp3237",
        "runtime_receipt_state": {
            "state": "blocked_selected_python_cuda_smoke_failed",
            "cuda_driver_visible": True,
            "selected_python_torch_cuda_available": False,
            "cuda_python_smoke_passed": False,
            "llama_cpp_cuda_receipt_ready": False,
            "sota_gguf_receipt_ready": False,
            "receipt_chain_ready": False,
            "blocking_artifacts": ["exp3236", "exp3237", "exp3238"],
            "next_action": "repair_selected_python_torch_cuda_before_exp3237",
        },
        "prompt_injection_v4_state": {
            "state": "blocked_after_manifest_teacher_label_shard_gate_blocked",
            "manifest_ready": True,
            "teacher_label_plan_ready": True,
            "teacher_label_shard_status": "gate_blocked",
            "train_eval_shard_status": "gate_blocked",
            "publication_evidence_ready": False,
            "blocking_artifacts": ["exp3240", "exp3241"],
        },
        "structured_proposal_state": {
            "state": "missing_gate_blocked_on_exp3238_clean_rerun_allowed",
            "artifact_status": "missing",
            "structured_proposal_preflight_ready": False,
            "repair_acceptance_claimed": False,
            "blocking_artifacts": ["exp3242"],
        },
        "fr11_failure_memory_state": {
            "state": "ready_controller_memory_update_no_model_weight_training",
            "fr11_controller_update_ready": True,
            "failure_trace_count": 28,
            "heldout_replay_count": 28,
            "heldout_replay_delta": 28,
            "doomed_rerun_avoidance_count": 27,
            "model_weight_update_claimed": False,
            "controller_memory_updates_are_not_training": True,
        },
        "protected_files_untouched": {"scripts/research_conductor.py": True},
        "honest_verdict": (
            "complete: cross_corpus_matrix_v33_ready=true; paper_ready=false; "
            "publication_blocker_count=106; next_top_gap=repair_selected_python_torch_cuda_before_exp3237"
        ),
    }


def _capstone_v299() -> dict[str, Any]:
    return {
        "experiment_id": "exp3223",
        "task_id": "exp3223-capstone-v299-single-focus",
        "capstone_v299_ready": True,
        "paper_ready": False,
        "publication_blocker_count": 100,
        "v4_outcome": "blocked_missing_exp3222_result",
        "next_top_gap": "cuda_chain_for_full_local_sota_receipts",
        "honest_verdict": (
            "complete: capstone_v299_ready=true; paper_ready=false; "
            "publication_blocker_count=100; next_top_gap=cuda_chain_for_full_local_sota_receipts"
        ),
    }


def _blocked_gate_payload(experiment: int, summary: str) -> dict[str, Any]:
    return {
        "experiment": experiment,
        "schema": "blocked_gate_check_v1",
        "status": "blocked",
        "blocked_at_layer": "conductor_pre_gate",
        "gate_check_summary": summary,
        "honest_verdict": "blocked_gate_check_failed",
    }


def _write_sources(root: Path) -> None:
    _write_json(root, mod.MATRIX_V33_REL_PATH, _matrix_v33())
    _write_json(root, mod.CAPSTONE_V299_REL_PATH, _capstone_v299())
    _write_json(
        root,
        mod.EXP3236_REL_PATH,
        {
            "experiment_id": "exp3236",
            "cuda_python_smoke_passed": False,
            "selected_python_torch_cuda_available": False,
            "honest_verdict": "complete: cuda_python_smoke_passed=false",
        },
    )
    _write_json(
        root,
        mod.EXP3237_REL_PATH,
        _blocked_gate_payload(
            3237,
            "1 of 1 gate(s) failed; first failure: exp3236.cuda_python_smoke_passed",
        ),
    )
    _write_json(root, mod.EXP3239_REL_PATH, {"experiment_id": "exp3239", "v4_manifest_ready": True})
    _write_json(
        root,
        mod.EXP3240_REL_PATH,
        _blocked_gate_payload(
            3240,
            "1 of 2 gate(s) failed; first failure: exp3238.sota_gguf_receipt_ready",
        ),
    )
    _write_json(
        root,
        mod.EXP3241_REL_PATH,
        _blocked_gate_payload(
            3241,
            "1 of 1 gate(s) failed; first failure: exp3240.teacher_label_shard_ready",
        ),
    )
    _write_json(
        root,
        mod.EXP3243_REL_PATH,
        {
            "experiment_id": "exp3243",
            "fr11_controller_update_ready": True,
            "model_weight_update_claimed": False,
            "controller_memory_updates_are_not_training": True,
            "honest_verdict": "complete: fr11 failure-memory controller update ready=true",
        },
    )
    _write_text(
        root,
        mod.CONDUCTOR_LOG_REL_PATH,
        "| 2026-05-28 05:13 UTC | DCCD exact-row structured proposal preflight gated | "
        "GATE_BLOCK | Pre-emptive skip: upstream retired (exp3238-sota-gguf-receipt-v7) |\n",
    )


def test_req_report_3245_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3245: OpenSpec declares the capstone contract first."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3245" in spec
    assert "SCENARIO-REPORT-3245" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_report_3245_builds_capstone_from_v33_evidence(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3245: capstone closes .300 without publication overclaiming."""

    _write_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=5.0, now_s=8.25)
    sources = {row["role"]: row for row in artifact["source_artifacts"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["capstone_v300_ready"] is True
    assert artifact["paper_ready"] is False
    assert artifact["publication_blocker_count"] == 106
    assert artifact["blocker_delta_from_v299"] == 6
    assert artifact["duration_s"] == pytest.approx(3.25)
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["honest_verdict"].startswith("complete:")
    assert "public submission" not in artifact["honest_verdict"]
    assert "paper publication" not in artifact["honest_verdict"]

    local_sota = artifact["local_sota_receipt_state"]
    assert local_sota["status"] == "blocked"
    assert local_sota["completed"] is False
    assert local_sota["sota_gguf_receipt_ready"] is False
    assert local_sota["blocking_artifacts"] == ["exp3236", "exp3237", "exp3238"]
    assert local_sota["operator_safe_note"].startswith("Do not rerun full GGUF")

    prompt = artifact["prompt_injection_v4_state"]
    assert prompt["status"] == "gate_blocked"
    assert prompt["manifest_ready"] is True
    assert prompt["teacher_label_shard_status"] == "gate_blocked"
    assert prompt["train_eval_shard_status"] == "gate_blocked"
    assert prompt["completed"] is False

    proposal = artifact["structured_proposal_preflight_state"]
    assert proposal["status"] == "gate_blocked"
    assert proposal["completed"] is False
    assert proposal["repair_acceptance_claimed"] is False

    fr11 = artifact["fr11_failure_memory_state"]
    assert fr11["status"] == "complete"
    assert fr11["completed"] is True
    assert fr11["model_weight_update_claimed"] is False
    assert fr11["controller_memory_updates_are_not_training"] is True

    assert artifact["next_top_gap"] == "repair_selected_python_torch_cuda_before_exp3237"
    assert "selected Python CUDA smoke blocks exp3237" in artifact["next_top_gap_rationale"]
    assert artifact["protected_files_untouched"] == {
        "research-roadmap.yaml": True,
        "scripts/research_conductor.py": True,
    }
    assert "Do NOT push." in artifact["operator_safe_notes"]
    assert "Do NOT modify scripts/research_conductor.py." in artifact["operator_safe_notes"]
    assert sources["matrix_v33"]["sha256"] == _sha256(tmp_path / mod.MATRIX_V33_REL_PATH)
    assert sources["prompt_injection_teacher_label_shard"]["present"] is True
    assert sources["dccd_structured_proposal_preflight"]["present"] is False
    mod.validate_artifact(artifact)


def test_req_report_3245_writer_and_fail_closed_priors(tmp_path: Path) -> None:
    """REQ-REPORT-3245: writer persists output and missing authorities fail closed."""

    _write_sources(tmp_path)
    output = mod.write_artifact(tmp_path, started_s=1.0, now_s=2.0)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["capstone_v300_ready"] is True

    no_prior = mod.build_artifact(tmp_path / "empty", started_s=1.0, now_s=1.5)
    assert no_prior["capstone_v300_ready"] is False
    assert no_prior["paper_ready"] is False
    assert no_prior["publication_blocker_count"] == 0
    assert "matrix v33 is missing or not ready" in no_prior["invariant_violations"]
    assert "capstone v299 is missing or not ready" in no_prior["invariant_violations"]
    assert no_prior["honest_verdict"].startswith("complete:")
    mod.validate_artifact(no_prior)


def test_req_report_3245_defensive_helpers_and_validation(tmp_path: Path) -> None:
    """REQ-REPORT-3245: helper branches classify evidence without overclaiming."""

    assert mod.read_json_object(tmp_path / "missing.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{bad", encoding="utf-8")
    assert mod.read_json_object(bad) == {}
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    assert mod.read_json_object(list_json) == {}
    assert mod.sha256_file(tmp_path / "missing.json") is None
    assert mod._as_mapping({"x": 1}) == {"x": 1}
    assert mod._as_mapping([]) == {}
    assert mod._int_value(5) == 5
    assert mod._int_value(True) == 0
    assert mod._duration(9.0, 7.0) == 0.0

    ready_runtime = {
        "receipt_chain_ready": True,
        "cuda_python_smoke_passed": True,
        "llama_cpp_cuda_receipt_ready": True,
        "sota_gguf_receipt_ready": True,
        "state": "complete_runtime_receipt_chain_ready",
    }
    assert mod._local_sota_receipt_state(ready_runtime)["status"] == "complete"
    assert (
        mod._local_sota_receipt_state(
            {
                "receipt_chain_ready": False,
                "cuda_python_smoke_passed": True,
                "llama_cpp_cuda_receipt_ready": False,
                "state": "gate_blocked_llama_cpp_cuda_receipt_missing",
            }
        )["status"]
        == "gate_blocked"
    )

    assert mod._prompt_injection_v4_state({"publication_evidence_ready": True})["status"] == (
        "complete"
    )
    assert mod._prompt_injection_v4_state({"manifest_ready": False})["status"] == "blocked"
    assert (
        mod._structured_proposal_preflight_state({"structured_proposal_preflight_ready": True})[
            "status"
        ]
        == "complete"
    )
    assert mod._structured_proposal_preflight_state({"artifact_status": "missing"})["status"] == (
        "gate_blocked"
    )
    assert (
        mod._fr11_failure_memory_state(
            {"fr11_controller_update_ready": True, "model_weight_update_claimed": True}
        )["status"]
        == "blocked"
    )
    assert (
        mod._fr11_failure_memory_state(
            {
                "fr11_controller_update_ready": True,
                "model_weight_update_claimed": False,
                "controller_memory_updates_are_not_training": True,
            }
        )["status"]
        == "complete"
    )
    assert (
        mod._next_gap_rationale(
            "repair_selected_python_torch_cuda_before_exp3237",
            {"status": "blocked"},
            {"status": "gate_blocked"},
            {"status": "gate_blocked"},
            {"status": "complete"},
        )
        == "selected Python CUDA smoke blocks exp3237, the SOTA GGUF receipt, prompt-injection shards, and structured proposal preflight"
    )
    assert mod._paper_ready(True, {"paper_ready": True}, 0) is True
    assert mod._paper_ready(True, {"paper_ready": False}, 0) is False
    assert mod._paper_ready(False, {"paper_ready": True}, 0) is False
    assert mod._paper_ready(True, {"paper_ready": True}, 1) is False

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
    with pytest.raises(ValueError, match="protected_files_untouched"):
        mod.validate_artifact(
            artifact | {"protected_files_untouched": {"scripts/research_conductor.py": True}}
        )
