"""Tests for REQ-CL-7348 and SCENARIO-CL-7348-*.

The tests use sealed public fixture bytes and synthetic model replies. They do
not load a model or open the repository's private evaluator manifest.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7348_v645_plan_capture as mod


ROOT = Path(__file__).resolve().parents[2]


def _public_manifest() -> dict[str, object]:
    """Load the exact public panel without opening evaluator-only fields."""

    return json.loads(mod.PUBLIC_MANIFEST_PATH.read_text(encoding="utf-8"))


def _runtime_identity() -> dict[str, object]:
    """Provide one task-owned CUDA identity for synthetic terminal rows."""

    return {
        "pid": 7348,
        "start_time_ticks": 88,
        "owned_by_task": True,
        "command": ["llama-server", "--model", "/cache/qwen.gguf"],
        "model_path": "/cache/qwen.gguf",
        "model_sha256": "sha256:" + "a" * 64,
        "served_model": "/cache/qwen.gguf",
        "gpu_uuid": "GPU-test",
        "lease_id": "lease-test",
        "cuda_provenance_ok": True,
    }


def _valid_reply(schedule_row: dict[str, object]) -> str:
    """Build one exact public plan from the first allowed start per activity."""

    request = schedule_row["public_request"]
    assert isinstance(request, dict)
    return json.dumps(
        {
            "request_id": request["request_id"],
            "assignments": {
                activity: request["allowed_starts"][activity][0]
                for activity in request["activities"]
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _response(raw_reply: str, *, error: str | None = None) -> dict[str, object]:
    """Return one retained native-response shape for the production row builder."""

    return {
        "raw_reply": raw_reply,
        "raw_response": {"model": "/cache/qwen.gguf", "choices": [{}]},
        "prompt_tokens": 90,
        "completion_tokens": 24,
        "latency_s": 1.5,
        "finish_reason": "stop",
        "error": error,
    }


def _terminal_rows(schedule: list[dict[str, object]]) -> list[dict[str, object]]:
    """Build one valid terminal response for every frozen call."""

    return [
        mod.build_call_row(
            schedule_row=row,
            response=_response(_valid_reply(row)),
            runtime_identity=_runtime_identity(),
        )
        for row in schedule
    ]


def test_scenario_cl_7348_schedule_freezes_128_blinded_calls() -> None:
    """SCENARIO-CL-7348-SCHEDULE fixes pairs, candidates, prompts, and budgets."""

    manifest = _public_manifest()
    schedule = mod.build_schedule(manifest)
    assert len(schedule) == 128
    assert mod.schedule_errors(schedule, manifest) == []
    assert {row["candidate_index"] for row in schedule} == {0, 1}
    assert {row["pair_side"] for row in schedule} == {"original", "twin"}
    assert len({row["call_id"] for row in schedule}) == 128
    assert len({row["request_id"] for row in schedule}) == 64
    assert all(row["max_generated_tokens"] == 256 for row in schedule)
    assert all(
        sum(item["request_id"] == row["request_id"] for item in schedule) == 2 for row in schedule
    )
    assert all("private_rules" not in row["prompt"] for row in schedule)
    assert all("acceptance_witness" not in row["prompt"] for row in schedule)
    assert all("evaluator" not in row["prompt"].lower() for row in schedule)
    assert mod.MODEL_SPECS == [{"hf_id": "unsloth/Qwen3.8-27B-GGUF", "quantization": "Q4_K_M"}]

    changed = deepcopy(schedule)
    changed[0]["candidate_index"] = 9
    assert "schedule_rebuild_mismatch" in mod.schedule_errors(changed, manifest)
    with pytest.raises(ValueError, match="live_proposal_panel_count"):
        mod.build_schedule({"live_proposal_panel": []})


def test_scenario_cl_7348_fidelity_separates_source_and_hidden_failures() -> None:
    """SCENARIO-CL-7348-FIDELITY does not relabel parse errors as oracle failures."""

    schedule = mod.build_schedule(_public_manifest())[:4]
    rows = [
        mod.build_call_row(
            schedule_row=schedule[0],
            response=_response(_valid_reply(schedule[0])),
            runtime_identity=_runtime_identity(),
        ),
        mod.build_call_row(
            schedule_row=schedule[1],
            response=_response(_valid_reply(schedule[1])),
            runtime_identity=_runtime_identity(),
        ),
        mod.build_call_row(
            schedule_row=schedule[2],
            response=_response("not-json"),
            runtime_identity=_runtime_identity(),
        ),
        mod.build_call_row(
            schedule_row=schedule[3],
            response=_response(_valid_reply(schedule[3])),
            runtime_identity=_runtime_identity(),
        ),
    ]
    evaluator_rows = [
        {"call_id": schedule[0]["call_id"], "evaluated": True, "hidden_rule_accepted": True},
        {"call_id": schedule[1]["call_id"], "evaluated": True, "hidden_rule_accepted": False},
        {
            "call_id": schedule[2]["call_id"],
            "evaluated": False,
            "hidden_rule_accepted": None,
            "reason": "source_parse_failure",
        },
        {"call_id": schedule[3]["call_id"], "evaluated": True, "hidden_rule_accepted": True},
    ]
    reduced = mod.reduce_calls(
        schedule,
        rows,
        evaluator_rows,
        load_receipt={"attempted": True, "completed": True},
        model_load_duration_s=5.0,
    )
    assert reduced["usable_candidate_count"] == 3
    assert reduced["source_parse_failure_count"] == 1
    assert reduced["hidden_rule_failure_count"] == 1
    assert reduced["source_fidelity_rows"][2]["parse_status"] == "invalid"
    assert reduced["source_fidelity_rows"][2]["hidden_rule_accepted"] is None
    assert reduced["source_fidelity_rows"][1]["parse_status"] == "valid"
    assert reduced["source_fidelity_rows"][1]["hidden_rule_accepted"] is False
    assert len(reduced["renamed_pair_rows"]) == 2
    first_pair = reduced["renamed_pair_rows"][0]
    assert first_pair["paired_observation_count"] == 1
    assert first_pair["both_source_valid"] is True
    assert first_pair["renamed_assignments_match"] is True


def test_scenario_cl_7348_terminal_keeps_censoring_in_fixed_denominator() -> None:
    """SCENARIO-CL-7348-TERMINAL counts explicit censoring as complete accounting."""

    schedule = mod.build_schedule(_public_manifest())
    rows = _terminal_rows(schedule[:-1])
    rows.append(mod.censored_call_row(schedule[-1], _runtime_identity(), "generation_deadline"))
    evaluator_rows = [
        {
            "call_id": row["call_id"],
            "evaluated": row["parse_status"] == "valid",
            "hidden_rule_accepted": True if row["parse_status"] == "valid" else None,
        }
        for row in rows
    ]
    reduced = mod.reduce_calls(
        schedule,
        rows,
        evaluator_rows,
        load_receipt={"attempted": True, "completed": True},
        model_load_duration_s=12.8,
    )
    assert reduced["plan_capture_complete_score"] == 1
    assert reduced["sample_size_budget"] == {
        "planned_units": 128,
        "attempted_units": 127,
        "completed_units": 127,
        "failed_units": 0,
        "cancelled_units": 1,
        "censored_units": 1,
    }
    assert reduced["invocation_counts"]["generation_calls_attempted"] == 127
    assert reduced["invocation_counts"]["generation_calls_cancelled"] == 1
    assert len(reduced["generation_cost_rows"]) == 128
    assert reduced["generation_cost_rows"][0]["allocated_model_load_s"] == pytest.approx(0.1)
    assert reduced["generation_cost_rows"][-1]["censored"] is True

    missing = mod.reduce_calls(
        schedule,
        rows[:-1],
        evaluator_rows[:-1],
        load_receipt={"attempted": True, "completed": True},
        model_load_duration_s=12.8,
    )
    assert missing["plan_capture_complete_score"] == 0


def test_scenario_cl_7348_replay_detects_row_cost_and_pair_drift() -> None:
    """SCENARIO-CL-7348-REPLAY rebuilds the artifact from sealed call evidence."""

    schedule = mod.build_schedule(_public_manifest())
    rows = _terminal_rows(schedule)
    evaluator_rows = [
        {"call_id": row["call_id"], "evaluated": True, "hidden_rule_accepted": True} for row in rows
    ]
    reduced = mod.reduce_calls(
        schedule,
        rows,
        evaluator_rows,
        load_receipt={"attempted": True, "completed": True},
        model_load_duration_s=12.8,
    )
    artifact = {
        "schedule": schedule,
        "rows": rows,
        "raw_call_manifest": {"schedule": deepcopy(schedule), "calls": deepcopy(rows)},
        "evaluator_rows": evaluator_rows,
        "load_receipt": {"attempted": True, "completed": True},
        "model_load_duration_s": 12.8,
        **reduced,
    }
    assert mod.independent_reduce(artifact) == []

    artifact["sample_size_budget"] = {
        **artifact["sample_size_budget"],
        "max_generated_tokens_per_unit": mod.MAX_GENERATED_TOKENS,
        "model_load_timeout_s": mod.MODEL_LOAD_TIMEOUT_S,
        "generation_timeout_s": mod.GENERATION_TIMEOUT_S,
        "stopping_rule": "fixed calls or deadline",
    }
    assert mod.independent_reduce(artifact) == []

    changed = deepcopy(artifact)
    changed["sample_size_budget"]["completed_units"] -= 1
    assert "sample_size_budget_mismatch" in mod.independent_reduce(changed)

    changed = deepcopy(artifact)
    changed["generation_cost_rows"][0]["completion_tokens"] += 1
    assert "generation_cost_rows_mismatch" in mod.independent_reduce(changed)
    changed = deepcopy(artifact)
    changed["source_fidelity_rows"][0]["window_fidelity"] = False
    assert "source_fidelity_rows_mismatch" in mod.independent_reduce(changed)
    changed = deepcopy(artifact)
    changed["raw_call_manifest"]["calls"][0]["raw_reply"] = "changed"
    assert "rows_manifest_mismatch" in mod.independent_reduce(changed)


def test_req_cl_7348_checkpoint_rejects_identity_or_row_tampering(tmp_path: Path) -> None:
    """REQ-CL-7348 checkpoints each terminal call under one frozen identity."""

    schedule = mod.build_schedule(_public_manifest())
    identity = mod.checkpoint_identity(schedule, "sha256:" + "1" * 64, "sha256:" + "2" * 64)
    path = tmp_path / "checkpoint.json"
    first = mod.build_call_row(
        schedule_row=schedule[0],
        response=_response(_valid_reply(schedule[0])),
        runtime_identity=_runtime_identity(),
    )
    mod.write_checkpoint(path, identity, [first])
    assert mod.resume_checkpoint(path, identity) == [first]

    changed_identity = deepcopy(identity)
    changed_identity["schedule_sha256"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="checkpoint_identity"):
        mod.resume_checkpoint(path, changed_identity)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["rows"][0]["raw_reply"] = "changed"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint_row_hash"):
        mod.resume_checkpoint(path, identity)


def test_req_cl_7348_dependency_gate_rejects_partial_or_quarantined_inputs() -> None:
    """REQ-CL-7348 rejects unready producers before any current model attempt."""

    fixture = json.loads((ROOT / mod.FIXTURE_PATH).read_text(encoding="utf-8"))
    canary = json.loads((ROOT / mod.CANARY_PATH).read_text(encoding="utf-8"))
    assert all(row["passed"] for row in mod.dependency_gate_rows(fixture, canary))

    changed = deepcopy(canary)
    changed["status"] = "partial"
    changed["verdict_class"] = "partial"
    checks = mod.dependency_gate_rows(fixture, changed)
    assert any(row["check"] == "plan_canary_ready" and not row["passed"] for row in checks)
    quarantined = deepcopy(fixture)
    quarantined["honest_verdict"] = "complete_quarantined_fixture"
    checks = mod.dependency_gate_rows(quarantined, canary)
    assert any(row["check"] == "executor_fixture_ready" and not row["passed"] for row in checks)


def test_req_cl_7348_complete_zero_usable_is_null_and_failed_checks_disqualify() -> None:
    """REQ-CL-7348 keeps accounting completion separate from proposal value."""

    outcome = mod.classify_terminal(
        complete_score=1,
        usable_candidate_count=0,
        required_checks_passed=True,
        flagged_adversarial=False,
    )
    assert outcome == {
        "status": "complete",
        "verdict_class": "null",
        "honest_verdict": "complete_null_plan_capture_zero_usable_candidates",
        "plan_capture_complete_score": 1,
        "value_ready_score": 0,
        "promotion_ready_score": 0,
    }
    failed = mod.classify_terminal(
        complete_score=1,
        usable_candidate_count=8,
        required_checks_passed=False,
        flagged_adversarial=False,
    )
    assert failed["verdict_class"] == "disqualified"
    assert failed["plan_capture_complete_score"] == 0
    assert failed["value_ready_score"] == 0
    assert failed["promotion_ready_score"] == 0


def test_req_cl_7348_fail_closed_branches_remain_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CL-7348 names malformed schedules, checkpoints, and replay evidence."""

    manifest = _public_manifest()
    malformed = deepcopy(manifest)
    malformed["live_proposal_panel"][0] = "bad"
    with pytest.raises(ValueError, match="live_proposal_pair"):
        mod.build_schedule(malformed)
    duplicate = deepcopy(manifest)
    duplicate["live_proposal_panel"][1]["panel_id"] = duplicate["live_proposal_panel"][0][
        "panel_id"
    ]
    with pytest.raises(ValueError, match="panel_identity"):
        mod.build_schedule(duplicate)
    missing_request = deepcopy(manifest)
    missing_request["live_proposal_panel"][0]["original"] = None
    with pytest.raises(ValueError, match="public_request"):
        mod.build_schedule(missing_request)
    monkeypatch.setattr(mod, "PLANNED_CALLS", 127)
    with pytest.raises(ValueError, match="scheduled_call_count"):
        mod.build_schedule(manifest)
    monkeypatch.setattr(mod, "PLANNED_CALLS", 128)

    schedule = mod.build_schedule(manifest)
    assert mod.schedule_errors(schedule, {"live_proposal_panel": []})[0].startswith(
        "schedule_source_invalid:"
    )
    drift = deepcopy(schedule[:-1])
    drift[0]["call_id"] = drift[1]["call_id"]
    drift[0]["max_generated_tokens"] = 1
    drift[0]["request_id"] = "changed"
    errors = mod.schedule_errors(drift, manifest)
    assert {
        "schedule_rebuild_mismatch",
        "scheduled_call_count",
        "call_identity",
        "token_budget",
        "request_candidate_allocation",
    } <= set(errors)

    identity = mod.checkpoint_identity(schedule, "source", "model")
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(
        json.dumps({"identity": identity, "rows": {}, "row_hashes": []}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="checkpoint_rows"):
        mod.resume_checkpoint(checkpoint, identity)

    incomplete = mod.classify_terminal(
        complete_score=0,
        usable_candidate_count=1,
        required_checks_passed=True,
        flagged_adversarial=False,
    )
    assert incomplete["honest_verdict"] == "complete_null_plan_capture_incomplete_accounting"
    positive = mod.classify_terminal(
        complete_score=1,
        usable_candidate_count=3,
        required_checks_passed=True,
        flagged_adversarial=False,
    )
    assert positive["verdict_class"] == "circular_positive"

    assert mod.independent_reduce({}) == ["raw_call_manifest_unavailable"]
    assert mod.independent_reduce({"raw_call_manifest": {}}) == ["raw_call_rows_unavailable"]
    replay = {
        "schedule": [],
        "rows": [],
        "raw_call_manifest": {"schedule": [], "calls": []},
        "evaluator_rows": [],
        "load_receipt": {},
        "model_load_duration_s": 0,
        **mod.reduce_calls([], [], [], load_receipt={}, model_load_duration_s=0),
    }
    replay["schedule"] = ["changed"]
    assert "schedule_manifest_mismatch" in mod.independent_reduce(replay)

    destination = tmp_path / "atomic.json"

    def failed_replace(_source: object, _destination: object) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(mod.os, "replace", failed_replace)
    with pytest.raises(OSError, match="replace failed"):
        mod._atomic_json(destination, {"value": 1})
    assert list(tmp_path.glob(".atomic.json.*.tmp")) == []

    passing = [{"name": "one", "passed": True, "exit_code": 0}]
    assert mod._receipts_pass(passing, ["one"]) is True
    assert mod._receipts_pass(passing, ["one", "two"]) is False
