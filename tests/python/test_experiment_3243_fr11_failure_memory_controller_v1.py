"""Tests for Exp 3243 FR-11 failure-memory controller update.

Spec refs: REQ-LEARN-3243, SCENARIO-LEARN-3243,
SCENARIO-LEARN-3243-BLOCKED.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from carnot.eval import fr11_failure_memory_controller_v1 as mod


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec/capabilities/self-learning/spec.md"


def _write_json(root: Path, rel_path: Path | str, payload: Mapping[str, Any]) -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _exp3229_payload() -> dict[str, Any]:
    return {
        "experiment_id": "experiment_3229_fr11_nonforgetting_promotion_controller_v3",
        "accepted_trace_count": 28,
        "rejected_trace_count": 2,
        "stale_premise_rejection_count": 2,
        "negative_control_regression_count": 0,
        "nonforgetting_budget_exceeded": False,
        "model_weight_update_claimed": False,
        "controller_memory_training_boundary": {
            "controller_memory_updates_are_not_training": True,
            "model_weight_training": False,
            "model_weight_mutation": False,
            "base_model_weights_updated": False,
        },
        "stale_premise_invalidations": {
            "affected_route_count": 2,
            "affected_routes": [
                {
                    "route_node_id": "route:trace-drift-a",
                    "row_id": "drift-a",
                    "replay_role": "drift",
                    "routing_outcome": "skip_redundant_recheck",
                }
            ],
        },
        "honest_verdict": (
            "complete: promotion_allowed=true; accepted_trace_count=28; "
            "rejected_trace_count=2; stale_premise_rejection_count=2; "
            "model_weight_update_claimed=false; controller_memory_updates_are_not_training"
        ),
    }


def _exp3230_payload() -> dict[str, Any]:
    return {
        "experiment_id": "experiment_3230_kan_cl_certificate_boundary_audit_v2",
        "missing_certificate_count": 4,
        "certificate_boundary_ready": False,
        "kan_sidecar_promotion_allowed": False,
        "model_weight_update_claimed": False,
        "requirement_evidence_matrix": [
            {
                "requirement_id": "per_knot_budget",
                "evidence_status": "missing",
                "missing_evidence": "per-knot or per-template nonforgetting budget",
            }
        ],
        "honest_verdict": (
            "complete: missing_certificate_count=4; "
            "kan_sidecar_promotion_allowed=false; model_weight_update_claimed=false"
        ),
    }


def _exp3232_payload() -> dict[str, Any]:
    return {
        "experiment_id": "exp3232",
        "local_sota_receipt_status": "missing_full_local_sota_receipt_v6_after_exp3221_gate_blocked",
        "repair_gate_status": "blocked_v7_blocker_count_9",
        "continuous_self_learning_status": (
            "controller_memory_promotion_allowed_28_accepted_no_model_weight_update_"
            "kan_sidecar_blocked_missing_certificates_4"
        ),
        "publication_blockers": [
            {
                "experiment_id": "exp3222",
                "path": "results/experiment_3222_full_local_sota_receipt_v6.json",
                "role": "full_local_sota_receipt_v6",
                "source_field": "clean_rerun_allowed",
                "status": "missing",
                "status_rationale": "expected `.298` artifact is absent or malformed",
            },
            {
                "experiment_id": "exp3228",
                "path": "results/experiment_3228_multi_turn_repair_ladder_v8.json",
                "role": "multi_turn_repair_ladder_v8",
                "source_field": "repair_ladder_complete",
                "status": "gate_blocked",
                "status_rationale": "artifact was blocked by a conductor pre-gate",
            },
        ],
        "honest_verdict": (
            "complete: capstone_ready=true; paper_ready=false; "
            "next_top_gap=repair_system_driver_cuda_runtime_boundary_to_unblock_cuda_offload_receipt"
        ),
    }


def _exp3223_payload() -> dict[str, Any]:
    return {
        "experiment_id": "exp3223",
        "task_id": "exp3223-capstone-v299-single-focus",
        "v4_outcome": "blocked_missing_exp3222_result",
        "next_top_gap": "cuda_chain_for_full_local_sota_receipts",
        "source_artifacts": [
            {
                "path": "results/experiment_3222_prompt_injection_kan_distill_v4_15k.json",
                "role": "prompt_injection_kan_v4",
                "present": False,
                "readable_json_object": False,
            }
        ],
        "honest_verdict": (
            "complete: capstone_v299_ready=true; paper_ready=false; "
            "v4_outcome=blocked_missing_exp3222_result; "
            "next_top_gap=cuda_chain_for_full_local_sota_receipts"
        ),
    }


def _exp3236_payload() -> dict[str, Any]:
    return {
        "experiment_id": "exp3236",
        "task_id": "exp3236-isolated-cuda-python-smoke-v1",
        "milestone": "2026.05.300",
        "cuda_python_smoke_passed": False,
        "selected_python_torch_cuda_available": False,
        "cuda_bindings_runtime_ok": False,
        "smoke_block_reasons": [
            "selected_python_torch_cuda_unavailable",
            "cuda_bindings_runtime_no_devices",
        ],
        "recommended_next_task": "repair_selected_python_torch_cuda_before_exp3237",
        "model_weight_update_claimed": False,
        "honest_verdict": (
            "complete: cuda_python_smoke_passed=false; "
            "blocked_by=selected_python_torch_cuda_unavailable,cuda_bindings_runtime_no_devices; "
            "recommended_next_task=repair_selected_python_torch_cuda_before_exp3237"
        ),
    }


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
        "honest_verdict": "blocked_gate_check_failed",
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
                "reason": f"actual={actual!r} == expected=True",
            }
        ],
        "blocked_at_layer": "conductor_pre_gate",
    }


def _write_sources(root: Path) -> None:
    (root / mod.CONDUCTOR_LOG_REL_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / mod.CONDUCTOR_LOG_REL_PATH).write_text(
        "\n".join(
            [
                "| 2026-05-28 04:46 UTC | llama.cpp CUDA receipt smoke v2 gated on selected- | GATE_BLOCK | 1 of 1 gate(s) failed; first failure: exp3236-isolated-cuda-python-smoke-v1.cuda_python_smoke_passed |",
                "| 2026-05-28 04:48 UTC | llama.cpp CUDA receipt smoke v2 gated on selected- | GATE_BLOCK | 1 of 1 gate(s) failed; first failure: exp3236-isolated-cuda-python-smoke-v1.cuda_python_smoke_passed |",
                "| 2026-05-28 04:50 UTC | llama.cpp CUDA receipt smoke v2 gated on selected- | GATE_BLOCK | 1 of 1 gate(s) failed; first failure: exp3236-isolated-cuda-python-smoke-v1.cuda_python_smoke_passed |",
                "| 2026-05-28 05:07 UTC | Prompt-injection KAN v4 teacher-label shard gated  | GATE_BLOCK | 1 of 2 gate(s) failed; first failure: exp3238-sota-gguf-receipt-v7.sota_gguf_receipt_ready |",
                "| 2026-05-28 05:09 UTC | Prompt-injection KAN v4 teacher-label shard gated  | GATE_BLOCK | 1 of 2 gate(s) failed; first failure: exp3238-sota-gguf-receipt-v7.sota_gguf_receipt_ready |",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(root, mod.EXP3229_REL_PATH, _exp3229_payload())
    _write_json(root, mod.EXP3230_REL_PATH, _exp3230_payload())
    _write_json(root, mod.EXP3232_REL_PATH, _exp3232_payload())
    _write_json(root, mod.EXP3223_REL_PATH, _exp3223_payload())
    _write_json(root, mod.EXP3236_REL_PATH, _exp3236_payload())
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


def test_req_learn_3243_spec_anchor_exists() -> None:
    """REQ-LEARN-3243: OpenSpec declares the failure-memory contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-3243" in spec
    assert "SCENARIO-LEARN-3243" in spec
    assert "SCENARIO-LEARN-3243-BLOCKED" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "failure_memory_schema_ready" in spec
    assert "controller_memory_updates_are_not_training=true" in spec
    assert "honest_verdict` SHALL begin with `complete:`" in spec


def test_req_learn_3243_schema_and_trace_extraction(tmp_path: Path) -> None:
    """REQ-LEARN-3243-1/2: traces cover schema keys and failure classes."""

    _write_sources(tmp_path)
    sources = mod.load_sources(tmp_path)
    traces = mod.collect_failure_traces(sources)
    categories = {trace["category"] for trace in traces}

    assert set(mod.SCHEMA_KEYS) == {
        "prerequisite",
        "failure_signature",
        "stale_premise",
        "accepted_next_action",
        "retirement_risk",
    }
    assert mod.failure_memory_schema()["schema_ready"] is True
    assert "missing_artifact" in categories
    assert "repeated_gate_block" in categories
    assert "stale_premise" in categories
    assert "backend_failure" in categories
    assert all(set(mod.SCHEMA_KEYS) <= trace.keys() for trace in traces)
    assert mod.count_stale_rejections(traces) == 2


def test_scenario_learn_3243_writes_ready_controller_update(tmp_path: Path) -> None:
    """SCENARIO-LEARN-3243: failure memory avoids doomed reruns."""

    _write_sources(tmp_path)
    output = mod.write_artifact(
        tmp_path,
        output_path=tmp_path / mod.OUTPUT_REL_PATH,
        started_s=10.0,
        now_s=12.0,
        tests_run=["SCENARIO-LEARN-3243 focused"],
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert mod.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert artifact["experiment_id"] == "exp3243"
    assert artifact["task_id"] == "exp3243-fr11-failure-memory-controller-v1"
    assert artifact["milestone"] == "2026.05.300"
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["failure_memory_schema_ready"] is True
    assert artifact["failure_trace_count"] >= 4
    assert artifact["heldout_replay_count"] >= 1
    assert artifact["heldout_replay_delta"] > 0
    assert artifact["nonforgetting_delta"] >= 0
    assert artifact["stale_premise_rejection_count"] == 2
    assert artifact["doomed_rerun_avoidance_count"] >= 1
    assert artifact["model_weight_update_claimed"] is False
    assert artifact["controller_memory_updates_are_not_training"] is True
    assert artifact["fr11_controller_update_ready"] is True
    assert artifact["tests_run"] == ["SCENARIO-LEARN-3243 focused"]
    assert artifact["duration_s"] == pytest.approx(2.0)
    assert artifact["honest_verdict"].startswith("complete:")
    assert "no model weights were updated" in artifact["honest_verdict"]
    assert any(row["controller_decision"] == "force_prerequisite_gate" for row in artifact["heldout_replays"])
    assert any(row["check_id"] == "stale_rejection_retention" for row in artifact["nonforgetting_checks"])
    mod.validate_artifact(artifact)


def test_req_learn_3243_nonforgetting_and_validation_guards(tmp_path: Path) -> None:
    """REQ-LEARN-3243-3/4/5/6: readiness requires replay and no-training gates."""

    _write_sources(tmp_path)
    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=2.0)

    assert mod.score_heldout_replays([]) == ([], 0, 0)
    assert mod.nonforgetting_delta(artifact["nonforgetting_checks"]) == 0
    assert mod.fr11_controller_update_ready(artifact) is True

    with pytest.raises(ValueError, match="model_weight_update_claimed"):
        mod.validate_artifact(artifact | {"model_weight_update_claimed": True})
    with pytest.raises(ValueError, match="controller_memory_updates_are_not_training"):
        mod.validate_artifact(artifact | {"controller_memory_updates_are_not_training": False})
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(artifact | {"honest_verdict": "complete: weights changed"})
    with pytest.raises(ValueError, match="readiness"):
        mod.validate_artifact(artifact | {"fr11_controller_update_ready": False})

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
    with pytest.raises(ValueError, match="failure trace count"):
        mod.validate_artifact(artifact | {"failure_trace_count": 999})
    with pytest.raises(ValueError, match="heldout replay count"):
        mod.validate_artifact(artifact | {"heldout_replay_count": 999})
    with pytest.raises(ValueError, match="nonforgetting_checks"):
        mod.validate_artifact(artifact | {"nonforgetting_checks": "bad"})
    with pytest.raises(ValueError, match="nonforgetting_delta"):
        mod.validate_artifact(artifact | {"nonforgetting_delta": 99})


def test_req_learn_3243_defensive_extractors_and_helpers() -> None:
    """REQ-LEARN-3243-2/3: extractor guards classify only useful evidence."""

    log_text = "\n".join(
        [
            "not a table",
            "| too | short |",
            "| 2026-05-26 00:00 UTC | Old gate | GATE_BLOCK | first failure: old.upstream |",
            "| 2026-05-28 00:00 UTC | Non gate | OK | all good |",
            "| 2026-05-28 00:01 UTC | Retired gate | GATE_BLOCK | Pre-emptive skip: upstream retired (exp-old) |",
            "| 2026-05-28 00:02 UTC | Retired gate | GATE_BLOCK | Pre-emptive skip: upstream retired (exp-old) |",
        ]
    )
    traces = mod.log_gate_block_traces(log_text)

    assert len(traces) == 1
    assert traces[0]["prerequisite"] == "exp-old"
    assert mod.parse_log_rows("not a table\n| too | short |\n") == []
    assert mod.first_failure_signature("") == "unknown_gate_block"
    assert mod.prerequisite_from_signature("plain") == "unknown_prerequisite"
    assert mod.capstone_blocker_traces([]) == []
    assert mod.capstone_blocker_traces({"publication_blockers": "bad"}) == []
    assert mod.capstone_blocker_traces(
        {"publication_blockers": [None, {"status": "complete"}]}
    ) == []
    assert mod.missing_source_artifact_traces([]) == []
    assert mod.missing_source_artifact_traces({"source_artifacts": "bad"}) == []
    assert mod.stale_premise_traces([]) == []
    assert mod.stale_premise_traces({"stale_premise_rejection_count": 0}) == []
    assert mod.certificate_failure_traces([]) == []
    assert mod.certificate_failure_traces({"missing_certificate_count": 0}) == []
    assert mod.backend_failure_traces([]) == []
    assert mod.backend_failure_traces({"cuda_python_smoke_passed": True}) == []
    assert mod.blocked_gate_artifact_traces("exp3237", []) == []
    assert mod.blocked_gate_artifact_traces(
        "exp3237", {"status": "blocked", "gates_evaluated": []}
    ) == []
    assert mod.controller_decision({"category": "other", "accepted_next_action": ""}) == (
        "no_memory_action"
    )
    assert mod.controller_decision({"category": "other", "accepted_next_action": "hold"}) == "hold"
    assert mod.nonforgetting_delta([]) == 0
    assert mod.safe_int("not-int") == 0


def test_scenario_learn_3243_blocked_sources_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-LEARN-3243-BLOCKED: missing evidence stays schema-complete."""

    artifact = mod.build_artifact(tmp_path, started_s=3.0, now_s=3.5)

    assert mod.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert artifact["failure_memory_schema_ready"] is True
    assert artifact["failure_trace_count"] == 0
    assert artifact["heldout_replay_count"] == 0
    assert artifact["heldout_replay_delta"] == 0
    assert artifact["nonforgetting_delta"] == 0
    assert artifact["stale_premise_rejection_count"] == 0
    assert artifact["doomed_rerun_avoidance_count"] == 0
    assert artifact["model_weight_update_claimed"] is False
    assert artifact["controller_memory_updates_are_not_training"] is True
    assert artifact["fr11_controller_update_ready"] is False
    assert artifact["honest_verdict"].startswith("complete:")
    assert "no model weights were updated" in artifact["honest_verdict"]
    mod.validate_artifact(artifact)
