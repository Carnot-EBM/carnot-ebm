"""Tests for the repaired V653 paired claim-span capture.

Spec refs: REQ-VERIFY-7451 and SCENARIO-VERIFY-7451-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7442_v652_span_capture as shared
from carnot import experiment_7451_v653_span_capture as capture


REPO = Path(__file__).resolve().parents[2]


def _lifecycle_artifact() -> dict[str, object]:
    return json.loads((REPO / capture.LIFECYCLE_PATH).read_text(encoding="utf-8"))


def _development_schedule() -> list[dict[str, object]]:
    with capture.capture_contract():
        return shared._fixture_schedules()[0]


def _response(
    row: dict[str, object],
    reply: str,
    *,
    attempted: bool = True,
    terminal_state: str = "response",
    finish_reason: str | None = "stop",
) -> dict[str, object]:
    return {
        "raw_request": {"messages": [{"role": "user", "content": row["prompt"]}]},
        "raw_response": {"reply": reply},
        "raw_reply": reply,
        "attempted": attempted,
        "terminal_state": terminal_state,
        "finish_reason": finish_reason,
        "prompt_tokens": 10,
        "completion_tokens": 4,
        "latency_s": 0.1,
        "error": "transport" if terminal_state != "response" else None,
    }


def _valid_reply(row: dict[str, object]) -> str:
    paragraph = str(row["paragraph"])
    end = paragraph.find(".") + 1
    end = end if end > 0 else len(paragraph)
    if row["arm"] == "span":
        return json.dumps({"claims": [[0, end]]})
    return json.dumps({"claims": [paragraph[:end]]})


def test_req_verify_7451_contract_is_exact_and_scoped() -> None:
    """REQ-VERIFY-7451 names the fixed model, venue, scope, and spec first."""

    section = (REPO / capture.SPEC_PATH).read_text(encoding="utf-8")
    section = section[section.index("REQ-VERIFY-7451") :]
    for anchor in (
        "SCENARIO-VERIFY-7451-PRECONDITIONS",
        "SCENARIO-VERIFY-7451-DEVELOPMENT",
        "SCENARIO-VERIFY-7451-RAW",
        "SCENARIO-VERIFY-7451-REDUCTION",
        "SCENARIO-VERIFY-7451-TERMINAL",
    ):
        assert anchor in section
    assert capture.MODEL_SPECS == ["unsloth/Qwen3.8-27B-GGUF"]
    assert capture.INFERENCE_SUBSTRATE_CLASS == "model_bounded_generation"
    assert capture.EXECUTION_VENUE == "host"
    assert capture.PHASE == 2
    assert capture.VALIDATION_MANIFEST.test_paths == (
        capture.TEST_PATH.as_posix(),
        "tests/python/test_experiment_7442_v652_span_capture.py",
        "tests/python/test_experiment_7448_v653_capture_lifecycle.py",
    )
    assert capture.VALIDATION_MANIFEST.changed_modules == (capture.MODULE_PATH.as_posix(),)


def test_scenario_verify_7451_preconditions_authenticate_exact_lifecycle_values() -> None:
    """SCENARIO-VERIFY-7451-PRECONDITIONS keeps each upstream operand explicit."""

    value = _lifecycle_artifact()
    rows = capture.lifecycle_gate_rows(value)
    assert [row["field"] for row in rows] == [
        "experiment_id",
        "capture_lifecycle_ready_score",
        "verdict_class",
        "flagged_adversarial",
    ]
    assert all(row["passed"] is True for row in rows)
    changed = deepcopy(value)
    changed["capture_lifecycle_ready_score"] = 0
    assert capture.lifecycle_gate_rows(changed)[1]["observed"] == 0
    assert capture.lifecycle_gate_rows(changed)[1]["passed"] is False
    missing = capture.lifecycle_gate_rows({})
    assert missing[1]["observed"] is None
    assert missing[1]["passed"] is False

    checks, context = capture.collect_preconditions(REPO)
    by_name = {row["check"]: row for row in checks}
    assert by_name["driving_requirement_v653"]["passed"] is True
    assert by_name["capture_lifecycle_artifact_hash"]["passed"] is True
    assert by_name["capture_lifecycle_ready"]["observed"] == 1
    assert (
        context["source_hashes"][capture.LIFECYCLE_PATH.as_posix()]["original_flagged_adversarial"]
        is False
    )


@pytest.mark.parametrize(
    ("reply", "attempted", "terminal_state", "finish_reason", "expected"),
    [
        ("valid", True, "response", "stop", "usable_nonempty"),
        ('{"claims":[]}', True, "response", "stop", "correct_empty"),
        ("{", True, "response", "stop", "malformed"),
        ("valid", True, "response", "length", "truncated"),
        ("", True, "response", "stop", "missing_output"),
        ("", True, "request_error", None, "transport_failure"),
        ("", False, "unstarted", None, "unstarted"),
    ],
)
def test_scenario_verify_7451_development_dispositions_are_distinct(
    reply: str,
    attempted: bool,
    terminal_state: str,
    finish_reason: str | None,
    expected: str,
) -> None:
    """SCENARIO-VERIFY-7451-DEVELOPMENT separates transport from extraction."""

    row = _development_schedule()[0]
    actual_reply = _valid_reply(row) if reply == "valid" else reply
    result = capture.build_capture_row(
        row,
        _response(
            row,
            actual_reply,
            attempted=attempted,
            terminal_state=terminal_state,
            finish_reason=finish_reason,
        ),
    )
    assert result["development_disposition"] == expected
    assert result["development_usable"] is (expected == "usable_nonempty")


def test_scenario_verify_7451_correct_empty_does_not_open_evaluation() -> None:
    """SCENARIO-VERIFY-7451-DEVELOPMENT excludes correct-empty from coverage."""

    schedule = _development_schedule()
    rows = [capture.build_capture_row(row, _response(row, _valid_reply(row))) for row in schedule]
    empty_index = next(index for index, row in enumerate(rows) if row["arm"] == "span")
    rows[empty_index] = capture.build_capture_row(
        schedule[empty_index], _response(schedule[empty_index], '{"claims":[]}')
    )
    gate = capture.reduce_development_gate(rows)
    assert gate["capture_open"] is True
    assert gate["disposition_counts"]["correct_empty"] == 1
    assert gate["usable_by_arm"] == {"span": 3, "verbatim": 4}
    second = next(
        index for index, row in enumerate(rows) if row["arm"] == "span" and index != empty_index
    )
    rows[second] = capture.build_capture_row(
        schedule[second], _response(schedule[second], '{"claims":[]}')
    )
    assert capture.reduce_development_gate(rows)["capture_open"] is False
    assert (
        capture._development_disposition(
            {
                "attempted": True,
                "terminal_state": "response",
                "finish_reason": "stop",
                "raw_reply": '{"claims":["text"]}',
                "parse_valid": True,
                "claims": ["text"],
                "completed_valid_output": False,
            }
        )
        == "malformed"
    )


def test_scenario_verify_7451_terminal_allows_clean_closed_canary_completion() -> None:
    """SCENARIO-VERIFY-7451-TERMINAL counts explicit unstarted rows as accounted."""

    development = [
        capture.build_capture_row(row, _response(row, '{"claims":[]}'))
        for row in _development_schedule()
    ]
    evaluation = shared.unstarted_evaluation_rows(shared._fixture_schedules()[1])
    assert (
        capture.capture_readiness(
            development_rows=development,
            evaluation_rows=evaluation,
            require_terminal=True,
            affected_ok=True,
            receipt_errors=[],
            runner={"all_layers_offloaded": True, "lease_released": True},
            flagged_adversarial=False,
            producer_runtime_error=None,
        )
        == 1
    )
    assert (
        capture.capture_readiness(
            development_rows=development,
            evaluation_rows=evaluation,
            require_terminal=True,
            affected_ok=True,
            receipt_errors=["failed"],
            runner={"all_layers_offloaded": True, "lease_released": True},
            flagged_adversarial=False,
            producer_runtime_error=None,
        )
        == 0
    )
    assert capture.reportable_constructed_delta(capture_open=False, measured=0.0) is None
    assert capture.reportable_constructed_delta(capture_open=True, measured=0.0) == 0.0


def test_scenario_verify_7451_fixture_replays_with_new_identity_and_fields() -> None:
    """SCENARIO-VERIFY-7451-REDUCTION independently replays the composed artifact."""

    artifact = capture.build_fixture_artifact()
    assert capture.validate_artifact(artifact, require_terminal=True) == []
    assert capture.independent_reduce_artifact(artifact, require_terminal=True) == []
    assert artifact["schema"] == capture.SCHEMA
    assert artifact["experiment_id"] == capture.EXPERIMENT_ID
    assert artifact["milestone"] == capture.MILESTONE
    assert artifact["phase"] == capture.PHASE
    assert artifact["MODEL_SPECS"] == capture.MODEL_SPECS
    assert len(artifact["development_rows"]) == 8
    assert all("development_disposition" in row for row in artifact["development_rows"])
    assert "development_gate" in artifact["field_principles"]
    assert "semantic_scope" in artifact["field_principles"]
    assert artifact["semantic_scope"]["real_paragraphs"] == "unannotated_extraction_metrics_only"
    receipt = artifact["source_artifact_hashes"][capture.LIFECYCLE_PATH.as_posix()]
    assert receipt["original_verdict_class"] == "null"
    assert receipt["original_flagged_adversarial"] is False

    changed = deepcopy(artifact)
    changed["preconditions_checked"] = [
        row for row in changed["preconditions_checked"] if row["check"] != "capture_lifecycle_ready"
    ]
    changed["reproducibility_checksum"] = capture.artifact_checksum(changed)
    assert "lifecycle_gate_missing:capture_lifecycle_ready" in capture.validate_artifact(
        changed, require_terminal=True
    )

    mutations = (
        ("phase", 3, "declaration_mismatch:phase"),
        ("semantic_scope", {}, "semantic_scope_mismatch"),
        ("development_rows", [], "development_dispositions_mismatch"),
        ("span_capture_complete_score", 0, "span_capture_complete_score_mismatch"),
        ("reproducibility_checksum", "sha256:changed", "reproducibility_checksum_mismatch"),
    )
    for field, replacement, expected in mutations:
        changed = deepcopy(artifact)
        changed[field] = replacement
        if field != "reproducibility_checksum":
            changed["reproducibility_checksum"] = capture.artifact_checksum(changed)
        assert expected in capture.validate_artifact(changed, require_terminal=True)


def test_scenario_verify_7451_blocked_artifact_has_zero_current_work() -> None:
    """SCENARIO-VERIFY-7451-PRECONDITIONS blocks without invented execution."""

    artifact = capture.build_blocked_artifact(capture.lifecycle_gate_rows({}))
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked_")
    assert artifact["model_invoked"] is False
    assert sum(artifact["invocation_counts"].values()) == 0
    assert artifact["MODEL_SPECS"] == capture.MODEL_SPECS
    assert artifact["inference_substrate_class"] == "no_model_load"
    assert artifact["gate_check_summary"]["observed"] is None
    assert capture.validate_artifact(artifact, require_terminal=True) == []


def test_req_verify_7451_contract_restores_shared_module_and_builds_cold_commands() -> None:
    """REQ-VERIFY-7451 reuses shared logic without leaving process-global drift."""

    before = {
        "experiment_id": shared.EXPERIMENT_ID,
        "result_path": shared.RESULT_PATH,
        "build_capture_row": shared.build_capture_row,
    }
    with capture.capture_contract():
        assert shared.EXPERIMENT_ID == capture.EXPERIMENT_ID
        assert shared.RESULT_PATH == capture.RESULT_PATH
        assert shared.build_capture_row is capture.build_capture_row
    assert shared.EXPERIMENT_ID == before["experiment_id"]
    assert shared.RESULT_PATH == before["result_path"]
    assert shared.build_capture_row is before["build_capture_row"]

    commands = capture.terminal_commands(REPO, REPO / "results/private-candidate.json")
    assert tuple(row.name for row in commands) == capture.TERMINAL_CHECK_NAMES
    assert capture.WRAPPER_PATH.as_posix() in commands[0].argv
    assert "experiment_7451_v653_span_capture" in " ".join(commands[1].argv)


def test_req_verify_7451_date_and_defensive_boundaries() -> None:
    """REQ-VERIFY-7451 rejects date drift and malformed artifact input."""

    assert capture.date_argument(capture.RUN_DATE) == capture.RUN_DATE
    with pytest.raises(ValueError, match="date must be 20260920"):
        capture.date_argument("20260919")
    assert capture.load_object(REPO / "missing-exp7451-fixture.json") == {}
    assert capture.validate_artifact([], require_terminal=True) == ["artifact_not_object"]
