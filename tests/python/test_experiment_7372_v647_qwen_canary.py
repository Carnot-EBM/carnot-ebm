"""Tests for the bounded Qwen assignment transport canary.

Spec refs: REQ-CL-7372 and SCENARIO-CL-7372-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7372_v647_qwen_canary as mod
from carnot.reporting.experiment_7303_validation_scope import REQUIRED_CHECK_NAMES


def _formula_row() -> dict[str, object]:
    """Build one small formula whose partial assignments have known outcomes."""

    return {
        "cohort": "development",
        "family": "unit_test",
        "formula_id": "dev-unit-test-0",
        "n_vars": 4,
        "seed": 7372500,
        "formula": {
            "version": "dev-unit-test-0-v1",
            "n_vars": 4,
            "source_hash": "sha256:" + "f" * 64,
            "raw_clause_order": [[1, 1], [-1, 2], [3, 4]],
            "clauses": [
                {"clause_id": 0, "literals": [1, 1]},
                {"clause_id": 1, "literals": [-1, 2]},
                {"clause_id": 2, "literals": [3, 4]},
            ],
        },
    }


def _schedule_row(index: int = 0) -> dict[str, object]:
    """Create one deterministic canary request from a sealed formula row."""

    formula = deepcopy(_formula_row())
    formula["formula_id"] = f"dev-unit-test-{index}"
    formula["formula"]["version"] = f"dev-unit-test-{index}-v1"  # type: ignore[index]
    return mod.schedule_row(formula, index)


def _identity(*, owned: bool = True) -> dict[str, object]:
    """Represent one task-owned native CUDA server."""

    return {
        "pid": 4242,
        "start_time_ticks": 101,
        "owned_by_task": owned,
        "command": ["/opt/llama-server", "--model", "/cache/qwen.gguf"],
        "model_path": "/cache/qwen.gguf",
        "model_sha256": "sha256:" + "a" * 64,
        "server_executable_sha256": "sha256:" + "b" * 64,
        "served_model": "/cache/qwen.gguf",
        "gpu_uuid": "GPU-test",
        "gpu_index": 1,
        "lease_id": "lease-test",
        "cuda_provenance_ok": True,
        "task_owned_vram_mb": 16384,
        "quantization": "Q4_K_M",
    }


def _call(index: int, raw_reply: str, *, owned: bool = True) -> dict[str, object]:
    """Build one response row through the production parser and evaluator."""

    schedule = _schedule_row(index)
    return mod.build_call_row(
        schedule,
        {
            "raw_request": {"messages": [{"role": "user", "content": schedule["prompt"]}]},
            "raw_reply": raw_reply,
            "raw_response": {"model": "/cache/qwen.gguf", "choices": [{}]},
            "prompt_tokens": 90,
            "completion_tokens": 12,
            "latency_s": 1.5,
            "finish_reason": "stop",
            "error": None,
        },
        _identity(owned=owned),
    )


def _load_receipt() -> dict[str, object]:
    """Return one complete load receipt for reduction tests."""

    return {
        "attempted": True,
        "completed": True,
        "failed": False,
        "cancelled": False,
        "in_flight": False,
    }


def test_req_cl_7372_selects_only_four_sealed_development_formulas() -> None:
    """REQ-CL-7372 uses development formulas and does not read future prompts."""

    fixture = json.loads(mod.DEVELOPMENT_FORMULAS_PATH.read_text(encoding="utf-8"))
    schedule = mod.build_canary_schedule(fixture)
    assert len(schedule) == 4
    assert [row["formula_id"] for row in schedule] == [
        row["formula_id"] for row in fixture["development_formulas"][:4]
    ]
    assert all(row["cohort"] == "development_canary" for row in schedule)
    assert all("FORMULA=" in row["prompt"] for row in schedule)
    assert all("assignments" in row["prompt"] for row in schedule)
    assert all("answer_id" not in row["prompt"] for row in schedule)
    assert mod.MODEL_SPECS == ["unsloth/Qwen3.8-27B-GGUF"]

    with pytest.raises(ValueError, match="development_formulas"):
        mod.build_canary_schedule({})
    with pytest.raises(ValueError, match="development_formula_count"):
        mod.build_canary_schedule({"development_formulas": []})
    duplicate = deepcopy(fixture)
    duplicate["development_formulas"][1]["formula_id"] = duplicate["development_formulas"][0][
        "formula_id"
    ]
    with pytest.raises(ValueError, match="development_formula_identity"):
        mod.build_canary_schedule(duplicate)


def test_scenario_cl_7372_parser_accepts_only_exact_assignment_json() -> None:
    """SCENARIO-CL-7372-PARSER rejects schema and literal drift without repair."""

    assert mod.decode_assignment('{"assignments":[1,-2,4]}', 4) == {
        "parse_status": "valid",
        "parse_errors": [],
        "assignments": [1, -2, 4],
        "schema_valid": True,
        "variable_references_valid": True,
        "response_fidelity_valid": True,
    }
    invalid = {
        "markdown": '```json\n{"assignments":[1,2]}\n```',
        "array": "[1,2]",
        "extra": '{"assignments":[1,2],"answer_id":7}',
        "not_list": '{"assignments":{"1":true}}',
        "too_short": '{"assignments":[1]}',
        "boolean": '{"assignments":[true,2]}',
        "zero": '{"assignments":[0,2]}',
        "range": '{"assignments":[1,5]}',
        "duplicate": '{"assignments":[1,-1]}',
    }
    parsed = {name: mod.decode_assignment(raw, 4) for name, raw in invalid.items()}
    assert all(row["parse_status"] == "invalid" for row in parsed.values())
    assert parsed["markdown"]["parse_errors"] == ["json_object"]
    assert parsed["array"]["parse_errors"] == ["json_object"]
    assert parsed["extra"]["parse_errors"] == ["top_level_fields"]
    assert parsed["not_list"]["parse_errors"] == ["assignments_list"]
    assert parsed["too_short"]["parse_errors"] == ["assignment_count"]
    assert parsed["boolean"]["parse_errors"] == ["literal_type"]
    assert parsed["zero"]["parse_errors"] == ["variable_reference"]
    assert parsed["range"]["parse_errors"] == ["variable_reference"]
    assert parsed["duplicate"]["parse_errors"] == ["duplicate_variable"]


def test_scenario_cl_7372_transport_is_separate_from_formula_validity() -> None:
    """SCENARIO-CL-7372-TRANSPORT permits one unusable or false proposal."""

    rows = [
        _call(0, '{"assignments":[1,2]}'),
        _call(1, '{"assignments":[-1,2]}'),
        _call(2, '{"assignments":[1,3]}'),
        _call(3, "not-json"),
    ]
    assert rows[0]["formula_valid"] is True
    assert rows[1]["formula_valid"] is False
    assert rows[1]["usable_proposal"] is True
    assert rows[3]["formula_valid"] is None
    reduced = mod.reduce_raw_calls(rows, _load_receipt())
    assert reduced["qwen_assignment_transport_ready_score"] == 1
    assert reduced["usable_proposal_count"] == 3
    assert reduced["formula_valid_proposal_count"] == 2
    assert reduced["invocation_counts"]["generation_calls_completed"] == 4

    two_usable = deepcopy(rows)
    two_usable[2] = _call(2, '{"assignments":[1]}')
    assert (
        mod.reduce_raw_calls(two_usable, _load_receipt())["qwen_assignment_transport_ready_score"]
        == 0
    )


def test_scenario_cl_7372_runtime_requires_one_owned_native_instance() -> None:
    """SCENARIO-CL-7372-RUNTIME binds all rows to one server and lease identity."""

    rows = [_call(index, '{"assignments":[1,2]}') for index in range(4)]
    assert mod.reduce_raw_calls(rows, _load_receipt())["runtime_identity_consistent"] is True

    unowned = deepcopy(rows)
    unowned[0]["runtime_identity_receipt"]["owned_by_task"] = False
    assert (
        mod.reduce_raw_calls(unowned, _load_receipt())["qwen_assignment_transport_ready_score"] == 0
    )

    second_server = deepcopy(rows)
    second_server[3]["runtime_identity_receipt"]["pid"] = 9999
    reduced = mod.reduce_raw_calls(second_server, _load_receipt())
    assert reduced["runtime_identity_consistent"] is False
    assert reduced["qwen_assignment_transport_ready_score"] == 0

    failed = deepcopy(rows)
    failed[3]["terminal_state"] = "request_error"
    failed[3]["error"] = "timeout"
    reduced = mod.reduce_raw_calls(failed, _load_receipt())
    assert reduced["invocation_counts"]["generation_calls_failed"] == 1
    assert reduced["qwen_assignment_transport_ready_score"] == 0


def test_scenario_cl_7372_gate_authenticates_exact_exp7371_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-CL-7372-GATE requires the eligible, hash-bound producer."""

    checks, context = mod.collect_dependency_preconditions(mod.REPO_ROOT)
    assert checks
    assert all(row["passed"] for row in checks)
    assert context["producer"]["proof_boundary_ready_score"] == 1
    assert len(context["schedule"]) == 4

    producer = deepcopy(context["producer"])
    producer["verdict_class"] = "blocked"
    rejected = mod.producer_gate_rows(
        producer,
        producer_path=mod.PRODUCER_PATH,
        producer_hash=mod.sha256_file(mod.PRODUCER_PATH),
        formulas_hash=mod.sha256_file(mod.DEVELOPMENT_FORMULAS_PATH),
        excluded=False,
    )
    failed = [row for row in rejected if not row["passed"]]
    assert [row["artifact_field"] for row in failed] == ["verdict_class"]
    summary = mod.gate_check_summary(rejected)
    assert summary["failed_check"] == "producer_verdict_class"
    assert summary["observed_value"] == "blocked"

    def reject_schedule(_fixture: object) -> list[dict[str, object]]:
        raise ValueError("sealed_schedule_invalid")

    monkeypatch.setattr(mod, "build_canary_schedule", reject_schedule)
    failed_checks, failed_context = mod.collect_dependency_preconditions(mod.REPO_ROOT)
    selection = next(
        row for row in failed_checks if row["check"] == "four_sealed_development_formulas"
    )
    assert selection["passed"] is False
    assert selection["observed_value"]["selection_error"] == "ValueError:sealed_schedule_invalid"
    assert failed_context["schedule"] == []


def test_req_cl_7372_validation_plan_is_exp7358_scoped(tmp_path: Path) -> None:
    """REQ-CL-7372 derives the fixed affected checks and creates private parents."""

    plan = mod.build_validation_plan(mod.REPO_ROOT, tmp_path / "private")
    assert [row.name for row in plan] == list(REQUIRED_CHECK_NAMES)
    assert "full_python_suite" not in {row.name for row in plan}
    assert mod.validate_validation_plan(mod.REPO_ROOT, plan) == []
    assert (tmp_path / "private/basetemp").is_dir()
    assert (tmp_path / "private/coverage").is_dir()


def test_scenario_cl_7372_artifact_reducer_detects_raw_drift(tmp_path: Path) -> None:
    """SCENARIO-CL-7372-ARTIFACT reloads exact call files before scoring."""

    rows = [_call(index, '{"assignments":[1,2]}') for index in range(4)]
    raw_rows: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        path = tmp_path / f"call_{index}.json"
        path.write_text(json.dumps(row, sort_keys=True), encoding="utf-8")
        raw_rows.append(
            {
                **deepcopy(row),
                "raw_path": str(path),
                "raw_sha256": mod.sha256_file(path),
            }
        )
    reduced = mod.reduce_raw_calls(rows, _load_receipt())
    artifact = {
        "status": "complete_assignment_transport_ready",
        "verdict_class": "positive",
        "rows": rows,
        "raw_call_rows": raw_rows,
        "load_receipt": _load_receipt(),
        **reduced,
    }
    assert mod.independent_reduce_artifact(artifact) == []

    changed = deepcopy(artifact)
    changed["usable_proposal_count"] = 0
    assert "usable_proposal_count_mismatch" in mod.independent_reduce_artifact(changed)

    disqualified = deepcopy(artifact)
    disqualified["verdict_class"] = "disqualified"
    disqualified["observed_qwen_assignment_transport_ready_score"] = 1
    disqualified["qwen_assignment_transport_ready_score"] = 0
    assert mod.independent_reduce_artifact(disqualified) == []

    changed = deepcopy(artifact)
    changed["rows"][0]["decoded_assignments"] = [-1, 2]
    assert "raw_call_row_mismatch:0" in mod.independent_reduce_artifact(changed)

    Path(raw_rows[0]["raw_path"]).write_text("{}", encoding="utf-8")
    assert "raw_call_hash_mismatch:0" in mod.independent_reduce_artifact(artifact)
    assert mod.independent_reduce_artifact({}) == ["raw_evidence_unavailable"]

    invalid_manifest = deepcopy(artifact)
    invalid_manifest["raw_call_rows"][0] = "bad"
    errors = mod.independent_reduce_artifact(invalid_manifest)
    assert "raw_call_manifest_invalid:0" in errors

    relative_missing = deepcopy(artifact)
    relative_missing["raw_call_rows"][0]["raw_path"] = "missing-call.json"
    relative_missing["raw_call_rows"][0]["raw_sha256"] = "sha256:" + "0" * 64
    errors = mod.independent_reduce_artifact(relative_missing, root=tmp_path)
    assert "raw_call_path_missing:0" in errors


def test_req_cl_7372_checksum_and_date_boundary() -> None:
    """REQ-CL-7372 binds terminal bytes and rejects another execution date."""

    artifact = {"schema": mod.SCHEMA, "reproducibility_checksum": "ignored", "rows": []}
    checksum = mod.artifact_checksum(artifact)
    artifact["reproducibility_checksum"] = checksum
    assert mod.artifact_checksum(artifact) == checksum
    assert checksum.startswith("sha256:")
    assert mod._date_argument("20260917") == "20260917"
    with pytest.raises(ValueError, match="date must be 20260917"):
        mod._date_argument("20260918")


def test_req_cl_7372_small_failure_helpers_preserve_exact_state(tmp_path: Path) -> None:
    """REQ-CL-7372 keeps malformed local inputs and receipt failures explicit."""

    missing = tmp_path / "missing.json"
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    array = tmp_path / "array.json"
    array.write_text("[]", encoding="utf-8")
    assert mod._load_object(missing) == {}
    assert mod._load_object(malformed) == {}
    assert mod._load_object(array) == {}

    with pytest.raises(ValueError, match="development_formula_shape"):
        mod.schedule_row({}, 0)

    row = mod.build_call_row(
        _schedule_row(),
        {
            "raw_reply": '{"assignments":[1,2]}',
            "raw_response": {"model": "/cache/qwen.gguf"},
            "finish_reason": "length",
        },
        _identity(),
    )
    assert row["raw_request"] == mod._native_request_payload(str(row["prompt"]))
    assert row["truncated"] is True
    assert mod._identity_sound({}) is False

    receipt = {"name": "one", "passed": True, "exit_code": 0, "timed_out": False}
    assert mod._receipts_pass([receipt], ["one"]) is True
    assert mod._receipts_pass([receipt], ["one", "missing"]) is False
    assert all(value == 0 for value in mod._zero_counts().values())
