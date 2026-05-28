"""Tests for Exp 3255 FR-11 lifelong failure-memory retention audit.

Spec refs: REQ-LEARN-3255, SCENARIO-LEARN-3255,
SCENARIO-LEARN-3255-BLOCKED.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from carnot.eval import fr11_lifelong_failure_memory_retention_audit_v1 as mod


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec/capabilities/self-learning/spec.md"


def _write_json(root: Path, rel_path: Path | str, payload: Mapping[str, Any]) -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _exp3243_payload() -> dict[str, Any]:
    return {
        "experiment_id": "exp3243",
        "task_id": "exp3243-fr11-failure-memory-controller-v1",
        "milestone": "2026.05.300",
        "fr11_controller_update_ready": True,
        "failure_trace_count": 4,
        "heldout_replay_count": 4,
        "doomed_rerun_avoidance_count": 3,
        "model_weight_update_claimed": False,
        "controller_memory_updates_are_not_training": True,
        "failure_traces": [
            {
                "trace_id": "fm-exp3236-backend",
                "category": "backend_failure",
                "source": "results/experiment_3236_isolated_cuda_python_smoke_v1.json",
                "failure_signature": (
                    "selected_python_torch_cuda_unavailable,"
                    "cuda_bindings_runtime_no_devices"
                ),
                "accepted_next_action": "repair_selected_python_torch_cuda_before_exp3237",
                "stale_premise": False,
            },
            {
                "trace_id": "fm-exp3237-gate",
                "category": "gate_block",
                "source": "results/experiment_3237_llama_cpp_cuda_receipt_smoke_v2.json",
                "failure_signature": (
                    "exp3236-isolated-cuda-python-smoke-v1.cuda_python_smoke_passed"
                ),
                "accepted_next_action": "force_prerequisite_gate",
                "stale_premise": False,
            },
            {
                "trace_id": "fm-exp3240-gate",
                "category": "missing_artifact",
                "source": "results/experiment_3240_prompt_injection_kan_teacher_label_shard_v1.json",
                "failure_signature": "exp3238-sota-gguf-receipt-v7.sota_gguf_receipt_ready",
                "accepted_next_action": "force_prerequisite_gate",
                "stale_premise": False,
            },
            {
                "trace_id": "fm-stale",
                "category": "stale_premise",
                "source": "results/experiment_3229_fr11_nonforgetting_promotion_controller_v3.json",
                "failure_signature": "stale_premise_rejection_count=2",
                "accepted_next_action": "reject_stale_controller_memory_trace",
                "stale_premise": True,
            },
        ],
        "heldout_replays": [
            {
                "replay_id": "heldout-001",
                "source_trace_id": "fm-exp3236-backend",
                "baseline_action": "rerun_without_failure_memory",
                "controller_decision": "repair_backend_before_rerun",
                "avoided_doomed_rerun": True,
                "force_gate": False,
                "replay_delta": 1,
            },
            {
                "replay_id": "heldout-002",
                "source_trace_id": "fm-exp3237-gate",
                "baseline_action": "rerun_without_failure_memory",
                "controller_decision": "force_prerequisite_gate",
                "avoided_doomed_rerun": True,
                "force_gate": True,
                "replay_delta": 1,
            },
            {
                "replay_id": "heldout-003",
                "source_trace_id": "fm-exp3240-gate",
                "baseline_action": "rerun_without_failure_memory",
                "controller_decision": "force_prerequisite_gate",
                "avoided_doomed_rerun": True,
                "force_gate": True,
                "replay_delta": 1,
            },
            {
                "replay_id": "heldout-004",
                "source_trace_id": "fm-stale",
                "baseline_action": "rerun_without_failure_memory",
                "controller_decision": "reject_stale_premise",
                "avoided_doomed_rerun": False,
                "force_gate": False,
                "replay_delta": 1,
            },
        ],
        "honest_verdict": (
            "complete: fr11 failure-memory controller update ready=true; "
            "model_weight_update_claimed=false; no model weights were updated"
        ),
    }


def _accepted_trace(row_id: str) -> dict[str, Any]:
    return {
        "trace_id": f"trace-accepted-{row_id}",
        "row_id": row_id,
        "replay_role": "heldout",
        "decision": "accepted",
        "reward_weight": 0.75,
    }


def _exp3229_payload(
    *,
    negative_control_regression_count: int = 0,
    accepted_traces: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    traces = accepted_traces or [_accepted_trace("a"), _accepted_trace("b"), _accepted_trace("c")]
    return {
        "experiment_id": "experiment_3229_fr11_nonforgetting_promotion_controller_v3",
        "milestone": "2026.05.298",
        "accepted_trace_count": len(traces),
        "accepted_traces": traces,
        "rejected_trace_count": 1,
        "negative_control_regression_count": negative_control_regression_count,
        "stale_premise_rejection_count": 1,
        "model_weight_update_claimed": False,
        "promotion_allowed": negative_control_regression_count == 0,
        "honest_verdict": "complete: unit exp3229; model_weight_update_claimed=false",
    }


def _label(
    row_id: str,
    *,
    role: str = "negative_control",
    reward_weight: float = 1.0,
    rollback: str = "none",
) -> dict[str, Any]:
    return {
        "trace_id": f"trace-{role}-{row_id}",
        "row_id": row_id,
        "replay_role": role,
        "exact_verifier_outcome": "exact_accept_answered",
        "prior_route_utility": "suppress_redundant_recheck",
        "routing_outcome": "skip_redundant_recheck",
        "reward_weight": reward_weight,
        "rollback_or_retraction_status": rollback,
        "model_weight_update_claimed": False,
    }


def _exp3215_payload(
    *,
    labels: list[dict[str, Any]] | None = None,
    negative_control_regression_count: int = 0,
) -> dict[str, Any]:
    replay_labels = labels or [_label("neg-1"), _label("neg-2"), _label("held-1", role="heldout")]
    return {
        "experiment_id": "experiment_3215_fr11_evidence_gated_trace_replay_controller_v2",
        "milestone": "2026.05.297",
        "continuous_self_learning_task": True,
        "trace_count": len(replay_labels),
        "replay_utility_label_count": len(replay_labels),
        "negative_control_regression_count": negative_control_regression_count,
        "rollback_event_count": 0,
        "model_weight_update_claimed": False,
        "promotion_allowed": negative_control_regression_count == 0,
        "replay_utility_labels": replay_labels,
        "honest_verdict": "complete: unit exp3215; model_weight_update_claimed=false",
    }


def _blocked_gate_payload(upstream: str, field: str, actual: Any) -> dict[str, Any]:
    return {
        "schema": "blocked_gate_check_v1",
        "status": "blocked",
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
    }


def _write_sources(
    root: Path,
    *,
    exp3215: Mapping[str, Any] | None = None,
    exp3229: Mapping[str, Any] | None = None,
) -> None:
    (root / mod.CONDUCTOR_LOG_REL_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / mod.CONDUCTOR_LOG_REL_PATH).write_text(
        "\n".join(
            [
                "| 2026-05-28 04:46 UTC | llama.cpp CUDA receipt smoke v2 gated on selected- | GATE_BLOCK | 1 of 1 gate(s) failed; first failure: exp3236-isolated-cuda-python-smoke-v1.cuda_python_smoke_passed |",
                "| 2026-05-28 04:48 UTC | llama.cpp CUDA receipt smoke v2 gated on selected- | GATE_BLOCK | 1 of 1 gate(s) failed; first failure: exp3236-isolated-cuda-python-smoke-v1.cuda_python_smoke_passed |",
                "| 2026-05-28 07:11 UTC | Isolated selected-Python CUDA smoke v2 gated on ro | GATE_BLOCK | 1 of 1 gate(s) failed; first failure: exp3247-selected-python-cuda-root-cause-surgery-v1.next_smoke_allowed |",
                "| 2026-05-28 07:14 UTC | Isolated selected-Python CUDA smoke v2 gated on ro | GATE_BLOCK | 1 of 1 gate(s) failed; first failure: exp3247-selected-python-cuda-root-cause-surgery-v1.next_smoke_allowed |",
                "| 2026-05-28 07:18 UTC | llama.cpp CUDA receipt smoke v3 gated on selected- | GATE_BLOCK | Pre-emptive skip: upstream retired (exp3248-isolated-cuda-selected-python-smoke-v2) |",
                "| 2026-05-28 07:20 UTC | Mandated SOTA GGUF receipt v8 gated on llama.cpp C | GATE_BLOCK | 1 of 1 gate(s) failed; first failure: exp3249-llama-cpp-cuda-receipt-smoke-v3.llama_cpp_cuda_receipt_ready |",
                "| 2026-05-28 07:51 UTC | Prompt-injection teacher-label shard v2 gated on S | GATE_BLOCK | Pre-emptive skip: upstream retired (exp3250-sota-gguf-receipt-v8) |",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (root / mod.RESEARCH_REFERENCES_REL_PATH).write_text(
        "LifelongAgentBench evaluates retention, adaptation, and forgetting.\n",
        encoding="utf-8",
    )
    _write_json(root, mod.EXP3243_REL_PATH, _exp3243_payload())
    _write_json(root, mod.EXP3229_REL_PATH, exp3229 or _exp3229_payload())
    _write_json(root, mod.EXP3215_REL_PATH, exp3215 or _exp3215_payload())
    _write_json(
        root,
        mod.EXP3247_REL_PATH,
        {
            "experiment_id": "exp3247",
            "task_id": "exp3247-selected-python-cuda-root-cause-surgery-v1",
            "milestone": "2026.05.301",
            "next_smoke_allowed": False,
            "selected_python_cuda_repaired_candidate": False,
            "recommended_next_task": "keep_exp3248_blocked_repair_cuda_runtime",
            "honest_verdict": "complete: next_smoke_allowed=false",
        },
    )
    _write_json(
        root,
        mod.EXP3248_REL_PATH,
        _blocked_gate_payload(
            "exp3247-selected-python-cuda-root-cause-surgery-v1",
            "next_smoke_allowed",
            False,
        ),
    )
    _write_json(
        root,
        mod.EXP3250_REL_PATH,
        _blocked_gate_payload(
            "exp3249-llama-cpp-cuda-receipt-smoke-v3",
            "llama_cpp_cuda_receipt_ready",
            None,
        ),
    )


def test_req_learn_3255_spec_anchor_exists() -> None:
    """REQ-LEARN-3255: OpenSpec declares the lifelong audit contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-3255" in spec
    assert "SCENARIO-LEARN-3255" in spec
    assert "SCENARIO-LEARN-3255-BLOCKED" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "retention_score" in spec
    assert "adaptation_score" in spec
    assert "forgetting_score" in spec
    assert "model_weight_update_claimed=false" in spec
    assert "no_new_llm_invoked=true" in spec


def test_req_learn_3255_slice_mapping_and_scores(tmp_path: Path) -> None:
    """REQ-LEARN-3255-1/2/3/4/5: slices map lifelong metrics to trace evidence."""

    _write_sources(tmp_path)
    sources = mod.load_sources(tmp_path)
    remembered = mod.remembered_slice(sources["exp3243"])
    adapted = mod.adapted_slice(sources)
    heldout = mod.heldout_negative_control_slice(sources["exp3215"])
    mapping = mod.lifelong_metric_mapping(sources["research_references"])

    assert set(mapping) == {"retention", "adaptation", "forgetting", "source_note"}
    assert "LifelongAgentBench" in mapping["source_note"]
    assert len(remembered) == 4
    assert len(adapted) >= 4
    assert {row["slice"] for row in remembered + adapted + heldout} == {
        "remembered",
        "adapted",
        "held_out_negative_control",
    }
    assert {row["milestone_bucket"] for row in adapted} >= {"2026.05.300", "2026.05.301"}
    assert mod.score_retention(remembered) == pytest.approx(1.0)
    assert mod.score_adaptation(adapted) == pytest.approx(1.0)
    assert mod.score_forgetting(sources["exp3229"], 0) == pytest.approx(1.0)
    assert mod.score_ratio(0, 0) == 0.0
    assert mod.safe_int("bad") == 0


def test_scenario_learn_3255_writes_ready_lifelong_audit(tmp_path: Path) -> None:
    """SCENARIO-LEARN-3255: lifelong audit retains and adapts without weight updates."""

    _write_sources(tmp_path)
    output = mod.write_artifact(
        tmp_path,
        output_path=tmp_path / mod.OUTPUT_REL_PATH,
        started_s=10.0,
        now_s=12.0,
        tests_run=["SCENARIO-LEARN-3255 focused"],
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert mod.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert artifact["experiment_id"] == "exp3255"
    assert artifact["task_id"] == "exp3255-fr11-lifelong-failure-memory-retention-audit-v1"
    assert artifact["milestone"] == "2026.05.301"
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["fr11_controller_update_ready"] is True
    assert artifact["lifelong_eval_ready"] is True
    assert artifact["failure_trace_count"] == len(artifact["failure_traces"])
    assert artifact["heldout_replay_count"] == len(
        artifact["evaluation_slices"]["held_out_negative_control"]
    )
    assert artifact["retention_score"] == pytest.approx(1.0)
    assert artifact["adaptation_score"] == pytest.approx(1.0)
    assert artifact["forgetting_score"] == pytest.approx(1.0)
    assert artifact["negative_control_regression_count"] == 0
    assert artifact["doomed_rerun_avoidance_count"] >= 3
    assert artifact["model_weight_update_claimed"] is False
    assert artifact["no_new_llm_invoked"] is True
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["tests_run"] == ["SCENARIO-LEARN-3255 focused"]
    assert artifact["duration_s"] == pytest.approx(2.0)
    assert artifact["honest_verdict"].startswith("complete:")
    assert "controller memory only" in artifact["honest_verdict"]
    assert "foundation-model weights were not updated" in artifact["honest_verdict"]
    assert any(row["milestone_bucket"] == "2026.05.300" for row in artifact["failure_traces"])
    assert any(row["milestone_bucket"] == "2026.05.301" for row in artifact["failure_traces"])
    mod.validate_artifact(artifact)


def test_req_learn_3255_validation_and_regression_guards(tmp_path: Path) -> None:
    """REQ-LEARN-3255-4/6/7: regressions and bad claims block readiness."""

    exp3215 = _exp3215_payload(
        labels=[_label("neg-bad", reward_weight=-1.0, rollback="rollback")],
        negative_control_regression_count=1,
    )
    _write_sources(tmp_path, exp3215=exp3215)
    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.5)

    assert artifact["negative_control_regression_count"] == 2
    assert artifact["forgetting_score"] < 1.0
    assert artifact["lifelong_eval_ready"] is False
    mod.validate_artifact(artifact)

    ready = mod.build_artifact(tmp_path, started_s=2.0, now_s=3.0)
    with pytest.raises(ValueError, match="model_weight_update_claimed"):
        mod.validate_artifact(ready | {"model_weight_update_claimed": True})
    with pytest.raises(ValueError, match="no_new_llm_invoked"):
        mod.validate_artifact(ready | {"no_new_llm_invoked": False})
    with pytest.raises(ValueError, match="experiment_id"):
        mod.validate_artifact(ready | {"experiment_id": "bad"})
    with pytest.raises(ValueError, match="task_id"):
        mod.validate_artifact(ready | {"task_id": "bad"})
    with pytest.raises(ValueError, match="milestone"):
        mod.validate_artifact(ready | {"milestone": "bad"})
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(ready | {"inference_substrate": "bad"})
    with pytest.raises(ValueError, match="heldout_replay_count"):
        mod.validate_artifact(ready | {"heldout_replay_count": 999})
    with pytest.raises(ValueError, match="failure_trace_count"):
        mod.validate_artifact(ready | {"failure_trace_count": 999})
    with pytest.raises(ValueError, match="retention_score"):
        mod.validate_artifact(ready | {"retention_score": 1.5})
    with pytest.raises(ValueError, match="lifelong_eval_ready"):
        mod.validate_artifact(ready | {"lifelong_eval_ready": not ready["lifelong_eval_ready"]})
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(ready | {"reproducibility_checksum": "bad"})
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(ready | {"honest_verdict": "done"})
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({})


def test_req_learn_3255_defensive_helpers_cover_malformed_inputs() -> None:
    """REQ-LEARN-3255-2/3/4: malformed rows do not create positive evidence."""

    assert mod.remembered_slice({"heldout_replays": "bad"}) == []
    assert mod.remembered_slice({"heldout_replays": [None]}) == []
    assert mod.trace_signature_lookup({"failure_traces": "bad"}) == {}
    assert mod.adapted_from_exp3243({"failure_traces": "bad"}) == []
    assert mod.adapted_from_exp3243({"failure_traces": [None, {"trace_id": "old"}]}) == []
    assert mod.heldout_negative_control_slice({"replay_utility_labels": "bad"}) == []
    assert (
        mod.adapted_from_log(
            "\n".join(
                [
                    "not a table",
                    "| too | short |",
                    "| 2026-05-28 07:00 UTC | Non gate | OK | first failure: x.y |",
                ]
            )
        )
        == []
    )
    assert mod.first_failure_signature("") == "unknown_gate_block"
    assert mod.controller_action_for_trace({"category": "stale_premise"}) == "reject_stale_premise"

    row = mod.make_adapted_row(
        source="ops/conductor-log.md",
        signature="same",
        milestone_bucket="2026.05.301",
        controller_action="force_prerequisite_gate",
        basis="duplicate",
    )
    assert mod.dedupe_rows([row, row]) == [row]


def test_scenario_learn_3255_blocked_sources_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-LEARN-3255-BLOCKED: missing evidence stays schema-complete."""

    artifact = mod.build_artifact(tmp_path, started_s=3.0, now_s=3.25)

    assert mod.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert artifact["fr11_controller_update_ready"] is False
    assert artifact["lifelong_eval_ready"] is False
    assert artifact["failure_trace_count"] == 0
    assert artifact["heldout_replay_count"] == 0
    assert artifact["retention_score"] == 0.0
    assert artifact["adaptation_score"] == 0.0
    assert artifact["forgetting_score"] == 0.0
    assert artifact["negative_control_regression_count"] == 0
    assert artifact["doomed_rerun_avoidance_count"] == 0
    assert artifact["model_weight_update_claimed"] is False
    assert artifact["no_new_llm_invoked"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    assert "controller memory only" in artifact["honest_verdict"]
    mod.validate_artifact(artifact)
