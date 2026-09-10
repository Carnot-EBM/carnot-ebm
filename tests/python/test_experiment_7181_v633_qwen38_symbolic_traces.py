"""Tests for REQ-VERIFY-7181 and SCENARIO-VERIFY-7181-*.

The live model boundary is exercised by the executable experiment. These tests
cover the deterministic contracts that decide whether its evidence is usable.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7181_v633_qwen38_symbolic_traces as exp
from carnot.inference.llama_server_supervisor import supervisor_contract


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/constraint-verification/spec.md"
GENERATION_VIEW = REPO / "results/experiment_7180_v633_symbolic_edit_fixture_generation_view.jsonl"


@pytest.fixture(scope="module")
def generation_rows() -> list[dict[str, object]]:
    """Load only the public worker view used by the frozen schedule."""

    return [json.loads(line) for line in GENERATION_VIEW.read_text(encoding="utf-8").splitlines()]


@pytest.fixture(scope="module")
def schedule(generation_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Build the immutable 192-row schedule once for focused tests."""

    return exp.build_schedule(generation_rows)


def _valid_output(prompt: str) -> str:
    evidence = prompt.split("Evidence:\n", 1)[1].split("\n\nClaim:", 1)[0]
    return json.dumps(
        {
            "direct_decision": "supported",
            "claim_tuple": {
                "subject": "a",
                "relation": "precedes",
                "object": "b",
                "polarity": "positive",
                "quantity": 1,
                "unit": "code_points",
            },
            "evidence_tuple": None,
            "source_start": 0,
            "source_end": len(evidence),
            "missing_fields": [],
        },
        separators=(",", ":"),
    )


def _response(schedule_row: dict[str, object], *, raw_output: str | None = None) -> dict[str, object]:
    output = raw_output if raw_output is not None else _valid_output(str(schedule_row["prompt"]))
    return {
        "raw_output": output,
        "raw_response": {"choices": [{"message": {"content": output}}]},
        "prompt_tokens": 42,
        "completion_tokens": 80,
        "latency_s": 0.25,
        "finish_reason": "stop",
        "error": None,
    }


def _completion_rows(schedule: list[dict[str, object]], count: int) -> list[dict[str, object]]:
    resource = {
        "server_pid": 123,
        "server_pid_start_ticks": 456,
        "gpu_uuid": "GPU-test",
        "lease_id": "lease-test",
    }
    return [exp.build_completion_row(row, _response(row), resource) for row in schedule[:count]]


def test_req_verify_7181_spec_precedes_implementation() -> None:
    """REQ-VERIFY-7181 names every focused risk and required artifact field."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-VERIFY-7181") :]
    for scenario in (
        "PREFLIGHT",
        "BLINDING",
        "CHECKPOINT",
        "CANARY",
        "OWNERSHIP",
        "ROWS",
        "TERMINAL",
        "ARTIFACT",
    ):
        assert f"SCENARIO-VERIFY-7181-{scenario}" in section
    assert all(f"`{field}`" in section for field in exp.REQUIRED_ARTIFACT_FIELDS)


def test_scenario_verify_7181_blinding_uses_only_public_rows(
    generation_rows: list[dict[str, object]], schedule: list[dict[str, object]]
) -> None:
    """SCENARIO-VERIFY-7181-BLINDING rejects authority data in worker input."""

    assert len(schedule) == 192
    assert exp.schedule_errors(schedule) == []
    assert all(set(row["model_input"]) == {"unit_id", "text", "response_schema"} for row in schedule)
    assert all("expected_response" not in json.dumps(row) for row in schedule)

    exposed = deepcopy(generation_rows)
    exposed[0]["variant"] = "original"
    with pytest.raises(ValueError, match="worker_view_shape"):
        exp.build_schedule(exposed)


def test_scenario_verify_7181_rows_retain_parse_failures_and_request_errors(
    schedule: list[dict[str, object]],
) -> None:
    """SCENARIO-VERIFY-7181-ROWS keeps invalid and negative transport outcomes."""

    resource = {
        "server_pid": 123,
        "server_pid_start_ticks": 456,
        "gpu_uuid": "GPU-test",
        "lease_id": "lease-test",
    }
    valid = exp.build_completion_row(schedule[0], _response(schedule[0]), resource)
    invalid = exp.build_completion_row(schedule[1], _response(schedule[1], raw_output="{"), resource)
    failed = exp.build_completion_row(
        schedule[2],
        {
            "raw_output": "",
            "raw_response": {},
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "latency_s": 120.0,
            "finish_reason": None,
            "error": "TimeoutError:request deadline",
        },
        resource,
    )
    assert valid["parse_status"] == "valid"
    assert valid["direct_decision"] == "supported"
    assert invalid["terminal_state"] == "response"
    assert invalid["parse_status"] == "failed"
    assert failed["terminal_state"] == "request_error"
    assert failed["request_error"] == "TimeoutError:request deadline"


def test_scenario_verify_7181_checkpoint_binds_exact_bytes_and_hashes(
    tmp_path: Path, schedule: list[dict[str, object]]
) -> None:
    """SCENARIO-VERIFY-7181-CHECKPOINT rejects byte or identity drift."""

    rows = _completion_rows(schedule, 8)
    identity = exp.checkpoint_identity(schedule, "sha256:generation", "sha256:model")
    path = tmp_path / "checkpoint.json"
    payload = exp.write_checkpoint(path, identity, rows)
    assert payload["row_count"] == 8
    assert payload["row_hashes"] == [exp.sha256_json(row) for row in rows]
    assert exp.resume_checkpoint(path, identity) == rows

    changed = deepcopy(identity)
    changed["model_sha256"] = "sha256:changed"
    with pytest.raises(ValueError, match="checkpoint_identity_mismatch"):
        exp.resume_checkpoint(path, changed)
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["rows"][0]["raw_output"] += " "
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint_row_hash_mismatch"):
        exp.resume_checkpoint(path, identity)


def test_scenario_verify_7181_ownership_refuses_changed_process() -> None:
    """SCENARIO-VERIFY-7181-OWNERSHIP delegates cleanup to the identity guard."""

    recorded = {
        "pid": 10,
        "start_time_ticks": 20,
        "uid": 30,
        "command_hash": "sha256:command",
        "process_group_id": 10,
        "parent_identity": {"pid": 1, "start_time_ticks": 2},
        "owned_by_task": True,
    }

    class ProcessOps:
        def __init__(self) -> None:
            self.signals: list[object] = []

        def send_signal(self, *args: object, **kwargs: object) -> None:
            self.signals.append((args, kwargs))

        def wait_for_exit(self, pid: int, timeout_s: float) -> str:
            return "timeout"

    operations = ProcessOps()
    changed = deepcopy(recorded)
    changed["start_time_ticks"] = 21
    changed["exists"] = True
    receipt = exp.cleanup_owned_process(
        recorded,
        lambda _pid: changed,
        operations,
        supervisor_contract(),
    )
    assert receipt["action"] == "refused"
    assert receipt["reason"] == "identity_mismatch"
    assert operations.signals == []


@pytest.mark.parametrize(
    ("checks", "canary", "row_count", "expected"),
    [
        ([{"passed": False}], None, 0, ("blocked", "blocked_no_run", 0)),
        ([{"passed": True}], {"terminal_state": "response"}, 0, ("partial", "model_bounded_generation", 0)),
        ([{"passed": True}], {"terminal_state": "response"}, 8, ("partial", "model_full_generation", 0)),
        ([{"passed": True}], {"terminal_state": "response"}, 192, ("complete", "model_full_generation", 1)),
    ],
)
def test_scenario_verify_7181_terminal_classifies_actual_work(
    checks: list[dict[str, object]],
    canary: dict[str, object] | None,
    row_count: int,
    expected: tuple[str, str, int],
) -> None:
    """SCENARIO-VERIFY-7181-TERMINAL separates block, canary, partial, and full work."""

    rows = [{"terminal_state": "request_error"}] * row_count
    result = exp.classify_terminal(checks, canary, rows, provenance_ok=True)
    assert (result["status"], result["inference_substrate_class"], result["score"]) == expected


def test_scenario_verify_7181_artifact_accepts_complete_parse_failures(
    schedule: list[dict[str, object]],
) -> None:
    """SCENARIO-VERIFY-7181-ARTIFACT scores transport completeness, not correctness."""

    rows = _completion_rows(schedule, 192)
    rows[0] = exp.build_completion_row(
        schedule[0], _response(schedule[0], raw_output="not json"), {
            "server_pid": 123,
            "server_pid_start_ticks": 456,
            "gpu_uuid": "GPU-test",
            "lease_id": "lease-test",
        }
    )
    artifact = exp.base_artifact(exp.RUN_DATE, root=REPO)
    artifact.update(
        {
            "status": "complete",
            "inference_substrate_class": "model_full_generation",
            "duration_s": 75.0,
            "rows": rows,
            "completion_rows": rows,
            "trace_capture_complete_score": 1,
            "preconditions_checked": [{"passed": True}],
            "gate_check_summary": exp.gate_row(
                "trace_capture_complete", True, True, True,
                upstream="experiment_7181", field="trace_capture_complete_score",
            ),
            "verdict_class": "positive",
            "honest_verdict": "complete_positive_transport_capture_no_correctness_claim",
            "raw_manifest": {"schedule": schedule, "checkpoint_receipts": []},
            "model_specs": {"sha256": "sha256:model"},
            "gpu_receipts": {"provenance_ok": True},
            "runner_receipt": {"model_count": 1, "runner": "llama.cpp"},
        }
    )
    artifact["reproducibility_checksum"] = exp.artifact_checksum(artifact)
    assert exp.validate_artifact(artifact, check_source_hashes=False) == []

    forged = deepcopy(artifact)
    forged["completion_rows"][1]["raw_output"] += " "
    forged["reproducibility_checksum"] = exp.artifact_checksum(forged)
    assert "row_1:raw_output_hash_mismatch" in exp.validate_artifact(
        forged, check_source_hashes=False
    )

