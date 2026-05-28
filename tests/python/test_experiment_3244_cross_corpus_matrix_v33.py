"""Tests for Exp 3244 cross-corpus matrix v33.

Spec refs: REQ-REPORT-3244, SCENARIO-REPORT-3244.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from carnot.reporting import cross_corpus_matrix_v33_3244 as mod


REQUIRED_FIELDS = {
    "experiment_id",
    "task_id",
    "milestone",
    "inference_substrate",
    "principle_annotations",
    "cross_corpus_matrix_v33_ready",
    "artifact_inventory",
    "runtime_receipt_state",
    "prompt_injection_v4_state",
    "structured_proposal_state",
    "fr11_failure_memory_state",
    "paper_ready",
    "publication_blocker_count",
    "next_top_gap",
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


def _blocked_gate_payload(
    experiment: int,
    title: str,
    upstream: str,
    field: str,
    actual: Any,
) -> dict[str, Any]:
    return {
        "experiment": experiment,
        "schema": "blocked_gate_check_v1",
        "status": "blocked",
        "title": title,
        "gate_check_summary": (
            f"1 of 1 gate(s) failed; first failure: {upstream}.{field} "
            f"(actual={actual!r} == expected=True)"
        ),
        "gates_evaluated": [
            {
                "upstream": upstream,
                "artifact_field": field,
                "expected": True,
                "actual": actual,
                "passed": False,
            }
        ],
        "blocked_at_layer": "conductor_pre_gate",
        "honest_verdict": "blocked_gate_check_failed",
    }


def _write_prior_authorities(root: Path) -> None:
    _write_json(
        root,
        mod.PREVIOUS_MATRIX_REL_PATH,
        {
            "experiment_id": "exp3231",
            "cross_corpus_matrix_v32_ready": True,
            "paper_ready": False,
            "publication_blocker_count": 100,
            "blocker_delta_from_v31": 8,
            "next_top_gap": "repair_system_driver_cuda_runtime_boundary_to_unblock_cuda_offload_receipt",
            "honest_verdict": "complete: cross_corpus_matrix_v32_ready=true",
        },
    )
    _write_json(
        root,
        mod.CAPSTONE_V299_REL_PATH,
        {
            "experiment_id": "exp3223",
            "capstone_v299_ready": True,
            "paper_ready": False,
            "publication_blocker_count": 100,
            "v4_outcome": "blocked_missing_exp3222_result",
            "next_top_gap": "cuda_chain_for_full_local_sota_receipts",
            "honest_verdict": "complete: capstone_v299_ready=true",
        },
    )


def _write_dot300_sources(root: Path) -> None:
    _write_json(
        root,
        mod.EXP3233_REL_PATH,
        {
            "experiment_id": "exp3233",
            "task_id": "exp3233-archive-v299-activate-v300",
            "archive_v299_activate_v300_ready": True,
            "prior_paper_ready": False,
            "prior_publication_blocker_count": 100,
            "prior_v4_outcome": "blocked_missing_exp3222_result",
            "next_top_gap": "cuda_chain_for_full_local_sota_receipts",
            "honest_verdict": "complete: archive_v299_activate_v300_ready=true",
        },
    )
    _write_json(
        root,
        mod.EXP3234_REL_PATH,
        {
            "experiment_id": "exp3234",
            "task_id": "exp3234-cli-backend-failure-root-cause-ledger-v1",
            "split_run_plan_ready": True,
            "exp3222_artifact_exists": False,
            "exp3222_failure_count": 3,
            "monolith_rerun_allowed": False,
            "model_spec_gap_found": True,
            "honest_verdict": "complete: split_run_plan_ready=true",
        },
    )
    _write_json(
        root,
        mod.EXP3235_REL_PATH,
        {
            "experiment_id": "exp3235",
            "task_id": "exp3235-cuda-driver-boundary-operator-package-v1",
            "cuda_boundary_package_ready": True,
            "full_gguf_rerun_allowed_now": False,
            "recommended_next_task": "exp3236-isolated-cuda-python-smoke-v1",
            "honest_verdict": "complete: cuda_boundary_package_ready=true",
        },
    )
    _write_json(
        root,
        mod.EXP3236_REL_PATH,
        {
            "experiment_id": "exp3236",
            "task_id": "exp3236-isolated-cuda-python-smoke-v1",
            "cuda_driver_visible": True,
            "selected_python_torch_import_ok": True,
            "selected_python_torch_cuda_available": False,
            "selected_python_device_count": 0,
            "cuda_bindings_import_ok": True,
            "cuda_bindings_device_count": 0,
            "cuda_python_smoke_passed": False,
            "smoke_block_reasons": [
                "selected_python_torch_cuda_unavailable",
                "cuda_bindings_runtime_no_devices",
            ],
            "recommended_next_task": "repair_selected_python_torch_cuda_before_exp3237",
            "honest_verdict": "complete: cuda_python_smoke_passed=false",
        },
    )
    _write_json(
        root,
        mod.EXP3237_REL_PATH,
        _blocked_gate_payload(
            3237,
            "llama.cpp CUDA receipt smoke v2 gated on selected-Python CUDA",
            "exp3236-isolated-cuda-python-smoke-v1",
            "cuda_python_smoke_passed",
            False,
        ),
    )
    _write_json(
        root,
        mod.EXP3239_REL_PATH,
        {
            "experiment_id": "exp3239",
            "task_id": "exp3239-prompt-injection-kan-v4-resource-manifest-v1",
            "v4_manifest_ready": True,
            "teacher_label_plan_ready": True,
            "delong_plan_ready": True,
            "garak_config_ready": True,
            "no_llm_invoked": True,
            "no_new_teacher_labeling": True,
            "no_kan_training": True,
            "downstream_deliverables": [
                {
                    "role": "teacher_label_shard",
                    "path": mod.EXP3240_REL_PATH.as_posix(),
                    "task_id": "exp3240-prompt-injection-kan-teacher-label-shard-v1",
                }
            ],
            "honest_verdict": "complete: v4_manifest_ready=true",
        },
    )
    _write_json(
        root,
        mod.EXP3240_REL_PATH,
        _blocked_gate_payload(
            3240,
            "Prompt-injection KAN v4 teacher-label shard gated on manifest and SOTA receipt",
            "exp3238-sota-gguf-receipt-v7",
            "sota_gguf_receipt_ready",
            None,
        ),
    )
    _write_json(
        root,
        mod.EXP3241_REL_PATH,
        _blocked_gate_payload(
            3241,
            "Prompt-injection KAN v4 shard train/eval with non-headline guardrail",
            "exp3240-prompt-injection-kan-teacher-label-shard-v1",
            "teacher_label_shard_ready",
            None,
        ),
    )
    _write_json(
        root,
        mod.EXP3243_REL_PATH,
        {
            "experiment_id": "exp3243",
            "task_id": "exp3243-fr11-failure-memory-controller-v1",
            "fr11_controller_update_ready": True,
            "failure_trace_count": 28,
            "heldout_replay_count": 28,
            "heldout_replay_delta": 28,
            "nonforgetting_delta": 0,
            "stale_premise_rejection_count": 2,
            "doomed_rerun_avoidance_count": 27,
            "model_weight_update_claimed": False,
            "controller_memory_updates_are_not_training": True,
            "honest_verdict": "complete: fr11 failure-memory controller update ready=true",
        },
    )
    _write_text(
        root,
        mod.CONDUCTOR_LOG_REL_PATH,
        "\n".join(
            [
                "| 2026-05-28 04:46 UTC | llama.cpp CUDA receipt smoke v2 gated on selected- | GATE_BLOCK | 1 of 1 gate(s) failed; first failure: exp3236-isolated-cuda-python-smoke-v1.cuda_python_smoke_passed |",
                "| 2026-05-28 04:52 UTC | Mandated local SOTA GGUF receipt v7 gated on llama | GATE_BLOCK | Pre-emptive skip: upstream retired (exp3237-llama-cpp-cuda-receipt-smoke-v2) |",
                "| 2026-05-28 05:07 UTC | Prompt-injection KAN v4 teacher-label shard gated  | GATE_BLOCK | 1 of 2 gate(s) failed; first failure: exp3238-sota-gguf-receipt-v7.sota_gguf_receipt_ready |",
                "| 2026-05-28 05:13 UTC | DCCD exact-row structured proposal preflight gated | GATE_BLOCK | Pre-emptive skip: upstream retired (exp3238-sota-gguf-receipt-v7) |",
            ]
        )
        + "\n",
    )


def test_req_report_3244_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3244: OpenSpec declares matrix v33 before implementation."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3244" in spec
    assert "SCENARIO-REPORT-3244" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_report_3244_builds_v33_from_dot300_artifacts(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3244: matrix v33 records complete, gated, and missing inputs."""

    _write_prior_authorities(tmp_path)
    _write_dot300_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=10.0, now_s=12.5)
    inventory = artifact["artifact_inventory"]
    rows = {row["experiment_id"]: row for row in inventory["planned_artifacts"]}
    complete = {row["experiment_id"] for row in inventory["complete_artifacts"]}
    gated = {row["experiment_id"] for row in inventory["gate_blocked_artifacts"]}
    missing = {row["experiment_id"] for row in inventory["missing_artifacts"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["cross_corpus_matrix_v33_ready"] is True
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert inventory["status_counts"] == {
        "complete": 5,
        "blocked": 1,
        "gate_blocked": 3,
        "missing": 2,
        "partial": 0,
    }
    assert complete == {"exp3233", "exp3234", "exp3235", "exp3239", "exp3243"}
    assert gated == {"exp3237", "exp3238", "exp3240", "exp3241", "exp3242"}
    assert missing == {"exp3238", "exp3242"}
    assert rows["exp3236"]["status"] == "blocked"
    assert rows["exp3238"]["gated_skip_evidence"]["status"] == "gate_blocked"
    assert rows["exp3242"]["gated_skip_evidence"]["status"] == "gate_blocked"

    assert artifact["prior_authorities"]["matrix_v32"]["paper_ready"] is False
    assert artifact["prior_authorities"]["capstone_v299"]["publication_blocker_count"] == 100
    assert artifact["paper_ready"] is False
    assert artifact["publication_blocker_count"] == 106
    assert artifact["publication_blocker_delta_from_v299"] == 6
    assert artifact["next_top_gap"] == "repair_selected_python_torch_cuda_before_exp3237"
    assert artifact["honest_verdict"].startswith("complete:")
    assert "missing_artifacts=2" in artifact["honest_verdict"]

    runtime = artifact["runtime_receipt_state"]
    assert runtime["state"] == "blocked_selected_python_cuda_smoke_failed"
    assert runtime["cuda_python_smoke_passed"] is False
    assert runtime["llama_cpp_cuda_receipt_ready"] is False
    assert runtime["sota_gguf_receipt_ready"] is False
    assert runtime["blocking_artifacts"] == ["exp3236", "exp3237", "exp3238"]

    prompt = artifact["prompt_injection_v4_state"]
    assert prompt["state"] == "blocked_after_manifest_teacher_label_shard_gate_blocked"
    assert prompt["manifest_ready"] is True
    assert prompt["teacher_label_shard_status"] == "gate_blocked"
    assert prompt["train_eval_shard_status"] == "gate_blocked"
    assert prompt["publication_evidence_ready"] is False

    proposal = artifact["structured_proposal_state"]
    assert proposal["state"] == "missing_gate_blocked_on_exp3238_clean_rerun_allowed"
    assert proposal["structured_proposal_preflight_ready"] is False
    assert proposal["repair_acceptance_claimed"] is False

    fr11 = artifact["fr11_failure_memory_state"]
    assert fr11["state"] == "ready_controller_memory_update_no_model_weight_training"
    assert fr11["fr11_controller_update_ready"] is True
    assert fr11["model_weight_update_claimed"] is False
    assert artifact["source_checksums"][mod.EXP3236_REL_PATH.as_posix()] == _sha256(
        tmp_path / mod.EXP3236_REL_PATH
    )
    mod.validate_artifact(artifact)


def test_req_report_3244_write_artifact_and_fail_closed_priors(tmp_path: Path) -> None:
    """REQ-REPORT-3244: writer persists schema-complete output and priors fail closed."""

    _write_prior_authorities(tmp_path)
    _write_dot300_sources(tmp_path)
    output = mod.write_artifact(tmp_path, started_s=4.0, now_s=5.0)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["cross_corpus_matrix_v33_ready"] is True

    no_prior = mod.build_artifact(tmp_path / "empty", started_s=1.0, now_s=1.25)
    assert no_prior["cross_corpus_matrix_v33_ready"] is False
    assert no_prior["paper_ready"] is False
    assert no_prior["publication_blocker_count"] == 11
    assert "prior matrix v32 is missing or not ready" in no_prior["invariant_violations"]
    assert no_prior["honest_verdict"].startswith("complete:")
    mod.validate_artifact(no_prior)


def test_req_report_3244_defensive_helpers_and_validation(tmp_path: Path) -> None:
    """REQ-REPORT-3244: helper functions classify edge cases without overclaiming."""

    spec = mod.SourceSpec("exp9999", "task", Path("missing.json"), "unknown", "ready")
    assert mod._normal_status("gate-blocked") == "gate_blocked"
    assert mod._normal_status("bad") == "missing"
    assert mod._as_mapping({"a": 1}) == {"a": 1}
    assert mod._as_mapping([]) == {}
    assert mod._bool_value(True) is True
    assert mod._bool_value("true") is False
    assert mod._int_value(7) == 7
    assert mod._int_value(False) == 0
    assert mod._int_value("7") == 0
    assert mod._duration(3.0, 2.0) == 0.0
    assert mod._read_text(tmp_path / "missing.log") == ""
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{bad", encoding="utf-8")
    assert mod.read_json_object(bad) == {}
    assert mod.sha256_file(tmp_path / "missing.json") is None

    assert mod._classify(mod.SOURCE_SPECS[0], {"archive_v299_activate_v300_ready": True})[0] == (
        "complete"
    )
    assert mod._classify(mod.SOURCE_SPECS[3], {"cuda_python_smoke_passed": False})[0] == (
        "blocked"
    )
    assert mod._classify(spec, {"status": "partial"})[0] == "partial"
    assert mod._classify(spec, {"schema": "blocked_gate_check_v1"})[0] == "gate_blocked"
    assert mod._classify(spec, {})[0] == "missing"
    assert mod._reported_experiment_id({"experiment": 12}, "fallback") == "exp12"
    assert mod._reported_experiment_id({"experiment_id": "exp1"}, "fallback") == "exp1"
    assert mod._reported_experiment_id({}, "fallback") == "fallback"
    assert mod._gate_evidence(spec, "no gate") == {"status": "absent"}
    titled = mod.SourceSpec("exp9998", "task", Path("x.json"), "role", "ready", "known title")
    assert mod._gate_evidence(titled, "different title GATE_BLOCK") == {"status": "absent"}
    assert mod._gate_status([], "exp9998") == "absent"
    assert mod._row_status([], "exp9998") == "missing"

    ready_runtime_rows = [
        {"experiment_id": "exp3237", "status": "complete"},
        {"experiment_id": "exp3238", "status": "complete"},
    ]
    ready_runtime_payloads = {
        "exp3235": {"cuda_boundary_package_ready": True},
        "exp3236": {"cuda_python_smoke_passed": True},
    }
    assert mod._runtime_receipt_state(ready_runtime_rows, ready_runtime_payloads)["state"] == (
        "complete_runtime_receipt_chain_ready"
    )
    assert mod._runtime_receipt_state(
        [{"experiment_id": "exp3237", "status": "blocked"}],
        ready_runtime_payloads,
    )["state"] == "gate_blocked_llama_cpp_cuda_receipt_missing"
    assert mod._runtime_receipt_state(
        [{"experiment_id": "exp3237", "status": "complete"}],
        ready_runtime_payloads,
    )["state"] == "gate_blocked_sota_gguf_receipt_missing"

    ready_prompt = mod._prompt_injection_v4_state(
        [
            {"experiment_id": "exp3240", "status": "complete"},
            {"experiment_id": "exp3241", "status": "complete"},
        ],
        {
            "exp3239": {
                "v4_manifest_ready": True,
                "teacher_label_plan_ready": True,
                "delong_plan_ready": True,
                "garak_config_ready": True,
            }
        },
    )
    assert ready_prompt["state"] == "complete_prompt_injection_v4_split_run_ready"

    ready_proposal = mod._structured_proposal_state(
        [{"experiment_id": "exp3242", "status": "complete"}],
        {"exp3242": {"structured_proposal_preflight_ready": True}},
    )
    assert ready_proposal["state"] == "complete_dccd_structured_proposal_preflight_ready"
    gated_proposal = mod._structured_proposal_state(
        [{"experiment_id": "exp3242", "status": "gate_blocked"}],
        {"exp3242": {}},
    )
    assert gated_proposal["state"] == "gate_blocked_on_exp3238_clean_rerun_allowed"

    runtime_ready = {"receipt_chain_ready": True, "next_action": ""}
    prompt_blocked = {"publication_evidence_ready": False}
    prompt_ready = {"publication_evidence_ready": True}
    proposal_blocked = {"structured_proposal_preflight_ready": False}
    proposal_ready = {"structured_proposal_preflight_ready": True}
    fr11_blocked = {"fr11_controller_update_ready": False}
    fr11_ready = {"fr11_controller_update_ready": True}
    assert mod._next_top_gap(runtime_ready, prompt_blocked, proposal_ready, fr11_ready) == (
        "prompt_injection_v4_teacher_label_and_train_eval_shards"
    )
    assert mod._next_top_gap(runtime_ready, prompt_ready, proposal_blocked, fr11_ready) == (
        "dccd_structured_proposal_preflight_after_clean_sota_receipt"
    )
    assert mod._next_top_gap(runtime_ready, prompt_ready, proposal_ready, fr11_blocked) == (
        "fr11_failure_memory_controller_ready_replay"
    )
    assert mod._next_top_gap(runtime_ready, prompt_ready, proposal_ready, fr11_ready) == (
        "publication_blocker_retirement_review"
    )
    assert "artifact inventory" in mod._invariant_violations(
        {"cross_corpus_matrix_v32_ready": True},
        {"capstone_v299_ready": True},
        [],
    )[0]

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
