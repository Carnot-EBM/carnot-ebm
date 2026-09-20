"""Tests for the paired V652 claim-span capture.

Spec refs: REQ-VERIFY-7442 and SCENARIO-VERIFY-7442-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7442_v652_span_capture as experiment


REPO_ROOT = Path(__file__).resolve().parents[2]


def _protocol_fixture() -> tuple[dict[str, object], dict[str, object]]:
    panel = experiment.protocol.seal_panel(experiment.protocol._fixture_predictors())
    artifact = {
        "experiment_id": "exp7437-v652-span-protocol",
        "milestone": "2026.09.652",
        "span_protocol_ready_score": 1,
        "verdict_class": "null",
        "flagged_adversarial": False,
    }
    return artifact, panel


def _transport(schedule: dict[str, object], reply: str, *, tokens: int = 8) -> dict[str, object]:
    return {
        "raw_request": {
            "messages": [{"role": "user", "content": schedule["prompt"]}],
            "max_tokens": 256,
        },
        "raw_response": {
            "choices": [{"message": {"content": reply}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 30, "completion_tokens": tokens},
        },
        "raw_reply": reply,
        "attempted": True,
        "terminal_state": "response",
        "finish_reason": "stop",
        "prompt_tokens": 30,
        "completion_tokens": tokens,
        "latency_s": 0.2,
        "runtime_identity_receipt": {"pid": 123, "start_time_ticks": 456},
    }


def _reply(schedule: dict[str, object], *, full: bool = True) -> str:
    paragraph = str(schedule["paragraph"])
    claim = paragraph if full else paragraph.split(".", 1)[0]
    claims: object = [[0, len(claim)]] if schedule["arm"] == "span" else [claim]
    return json.dumps({"claims": claims}, ensure_ascii=False)


def _terminal_receipts() -> list[dict[str, object]]:
    return [
        {"name": name, "passed": True, "exit_code": 0, "timed_out": False}
        for name in (*experiment.AFFECTED_CHECK_NAMES, *experiment.TERMINAL_CHECK_NAMES)
    ]


def test_protocol_gate_requires_exact_ready_eligible_unflagged_fields() -> None:
    """SCENARIO-VERIFY-7442-PRECONDITIONS: all three upstream fields gate work."""

    artifact, _ = _protocol_fixture()
    rows = experiment.protocol_gate_rows(artifact)
    assert [row["passed"] for row in rows] == [True, True, True, True]

    artifact["span_protocol_ready_score"] = 0
    artifact["verdict_class"] = "blocked"
    artifact["flagged_adversarial"] = True
    assert [row["passed"] for row in experiment.protocol_gate_rows(artifact)] == [True, False, False, False]


def test_schedules_have_eight_development_and_96_paired_evaluation_calls() -> None:
    """SCENARIO-VERIFY-7442-DEVELOPMENT: both arms use each sealed unit once."""

    _, panel = _protocol_fixture()
    development = experiment.build_development_schedule(panel["development"])
    evaluation = experiment.build_evaluation_schedule(
        panel["evaluation"], experiment.protocol.constructed_qualifier_pairs()
    )

    assert len(development) == 8
    assert len(evaluation) == 96
    assert len({row["paired_unit_id"] for row in evaluation}) == 48
    assert {row["source_kind"] for row in evaluation} == {"ragtruth", "constructed"}
    assert sum(row["source_kind"] == "ragtruth" for row in evaluation) == 48
    assert sum(row["source_kind"] == "constructed" for row in evaluation) == 48
    for schedule in (development, evaluation):
        for index in range(0, len(schedule), 2):
            pair = schedule[index : index + 2]
            assert {row["arm"] for row in pair} == {"span", "verbatim"}
            assert pair[0]["paragraph"] == pair[1]["paragraph"]
            assert all(row["max_new_tokens"] == 256 for row in pair)
            assert all(row["temperature"] == 0.0 for row in pair)
            assert all(row["request_timeout_s"] == 45.0 for row in pair)


def test_raw_row_hashes_transport_before_parsing_and_reconstructs_unicode() -> None:
    """SCENARIO-VERIFY-7442-RAW: raw hashes bind exact transport before parsing."""

    _, panel = _protocol_fixture()
    schedule = experiment.build_development_schedule(panel["development"])[0]
    reply = _reply(schedule)
    row = experiment.build_capture_row(schedule, _transport(schedule, reply))

    assert row["persisted_before_parse"] is True
    assert row["raw_request_sha256"] == experiment.canonical_hash(row["raw_request"])
    assert row["raw_response_sha256"] == experiment.canonical_hash(row["raw_response"])
    assert row["raw_reply_sha256"] == experiment.sha256_text(reply)
    assert row["parse_valid"] is True
    assert row["literal_span_reconstruction"] is True
    assert row["claims"] == [schedule["paragraph"]]


def test_truncated_and_malformed_rows_are_measured_failures() -> None:
    """SCENARIO-VERIFY-7442-RAW: bad replies are failures rather than missing resources."""

    _, panel = _protocol_fixture()
    schedule = experiment.build_development_schedule(panel["development"])[0]
    truncated_transport = _transport(schedule, '{"claims":[[0,5]]')
    truncated_transport["finish_reason"] = "length"
    malformed = experiment.build_capture_row(schedule, truncated_transport)

    assert malformed["attempted"] is True
    assert malformed["truncated"] is True
    assert malformed["parse_valid"] is False
    assert malformed["usable_output"] is False
    assert malformed["disposition"] == "response_truncated"


def test_development_gate_requires_three_usable_outputs_in_each_arm() -> None:
    """SCENARIO-VERIFY-7442-DEVELOPMENT: one weak arm keeps evaluation closed."""

    _, panel = _protocol_fixture()
    schedule = experiment.build_development_schedule(panel["development"])
    rows = [
        experiment.build_capture_row(row, _transport(row, _reply(row))) for row in schedule
    ]
    assert experiment.reduce_development_gate(rows)["evaluation_open"] is True

    changed = deepcopy(rows)
    for row in changed:
        if row["arm"] == "verbatim" and row["case_index"] >= 2:
            row["usable_output"] = False
    gate = experiment.reduce_development_gate(changed)
    assert gate["evaluation_open"] is False
    assert gate["by_arm"]["span"]["usable"] == 4
    assert gate["by_arm"]["verbatim"]["usable"] == 2


def test_closed_canary_emits_all_evaluation_units_as_unstarted() -> None:
    """SCENARIO-VERIFY-7442-DEVELOPMENT: a null keeps 96 explicit dispositions."""

    _, panel = _protocol_fixture()
    schedule = experiment.build_evaluation_schedule(
        panel["evaluation"], experiment.protocol.constructed_qualifier_pairs()
    )
    rows = experiment.unstarted_rows(schedule, "development_gate_closed")

    assert len(rows) == 96
    assert {row["disposition"] for row in rows} == {"unstarted"}
    assert all(row["attempted"] is False for row in rows)
    assert all(row["unstarted_reason"] == "development_gate_closed" for row in rows)


def test_reducer_separates_real_diagnostics_from_constructed_authority() -> None:
    """SCENARIO-VERIFY-7442-REDUCTION: only constructed pairs have exact semantics."""

    _, panel = _protocol_fixture()
    schedule = experiment.build_evaluation_schedule(
        panel["evaluation"], experiment.protocol.constructed_qualifier_pairs()
    )
    rows = [experiment.build_capture_row(row, _transport(row, _reply(row))) for row in schedule]
    reduced = experiment.reduce_capture(rows, schedule)

    assert reduced["planned_call_count"] == 96
    assert reduced["attempted_call_count"] == 96
    assert reduced["completed_call_count"] == 96
    assert reduced["paired_unit_count"] == 48
    assert reduced["span_capture_complete_observed"] == 1
    assert all(
        row["semantic_fidelity"] == "unknown"
        for row in reduced["extraction_rows"]
        if row["source_kind"] == "ragtruth"
    )
    assert len(reduced["semantic_pair_rows"]) == 12
    assert all(row["authority"] == "constructed_exact_string" for row in reduced["semantic_pair_rows"])


def test_value_gate_requires_positive_ci_no_qualifier_loss_and_lower_tokens() -> None:
    """SCENARIO-VERIFY-7442-REDUCTION: all paired value endpoints are necessary."""

    passing = experiment.value_gate(
        completion_ci95=[0.02, 0.20],
        span_qualifier_rate=1.0,
        verbatim_qualifier_rate=1.0,
        paired_output_token_delta=-3.0,
    )
    assert passing == 1
    assert experiment.value_gate(
        completion_ci95=[0.0, 0.20],
        span_qualifier_rate=1.0,
        verbatim_qualifier_rate=1.0,
        paired_output_token_delta=-3.0,
    ) == 0
    assert experiment.value_gate(
        completion_ci95=[0.02, 0.20],
        span_qualifier_rate=0.9,
        verbatim_qualifier_rate=1.0,
        paired_output_token_delta=-3.0,
    ) == 0
    assert experiment.value_gate(
        completion_ci95=[0.02, 0.20],
        span_qualifier_rate=1.0,
        verbatim_qualifier_rate=1.0,
        paired_output_token_delta=0.0,
    ) == 0


def test_paired_bootstrap_is_seeded_and_retains_zero_advantage() -> None:
    """SCENARIO-VERIFY-7442-REDUCTION: the paired interval is reproducible."""

    first = experiment.paired_bootstrap_ci([0, 1, 0, 1], seed=7442, samples=500)
    second = experiment.paired_bootstrap_ci([0, 1, 0, 1], seed=7442, samples=500)
    assert first == second
    assert first[0] <= 0.0 <= first[1]
    assert experiment.paired_bootstrap_ci([], seed=7442, samples=10) == [None, None]


def test_fixture_artifact_is_complete_and_cold_replayable(tmp_path: Path) -> None:
    """REQ-VERIFY-7442: terminal completeness and scientific value stay separate."""

    artifact = experiment.build_artifact_for_test(tmp_path)
    assert experiment.validate_artifact(artifact, root=tmp_path, require_terminal=True) == []
    assert experiment.independent_reduce_artifact(artifact, root=tmp_path, require_terminal=True) == []
    assert artifact["MODEL_SPECS"] == ["unsloth/Qwen3.8-27B-GGUF"]
    assert artifact["model_invoked"] is True
    assert artifact["invocation_counts"]["model_loads_completed"] == 1
    assert artifact["invocation_counts"]["generation_calls_completed"] == 104
    assert artifact["inference_substrate_class"] == "model_bounded_generation"
    assert artifact["execution_venue"] == "host"
    assert artifact["span_capture_complete_score"] == 1
    assert artifact["promotion_score"] == 0
    assert len(artifact["development_rows"]) == 8
    assert len(artifact["extraction_rows"]) == 96
    assert len(artifact["rows"]) == 96
    assert len(artifact["semantic_pair_rows"]) == 12


def test_validator_rejects_row_counter_receipt_and_checksum_drift(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7442-TERMINAL: raw and terminal evidence fails closed."""

    artifact = experiment.build_artifact_for_test(tmp_path)

    changed = deepcopy(artifact)
    changed["extraction_rows"][0]["raw_reply"] = "changed"
    changed["rows"] = deepcopy(changed["extraction_rows"])
    changed["reproducibility_checksum"] = experiment.artifact_checksum(changed)
    assert "raw_reply_hash_mismatch:0" in experiment.validate_artifact(
        changed, root=tmp_path, require_terminal=True
    )

    changed = deepcopy(artifact)
    changed["invocation_counts"]["generation_calls_completed"] = 103
    changed["reproducibility_checksum"] = experiment.artifact_checksum(changed)
    assert "current_invocation_receipt_invalid" in experiment.validate_artifact(
        changed, root=tmp_path, require_terminal=True
    )

    changed = deepcopy(artifact)
    changed["validation_receipts"] = [
        row for row in changed["validation_receipts"] if row["name"] != "adversarial_verify"
    ]
    changed["reproducibility_checksum"] = experiment.artifact_checksum(changed)
    assert "required_validation_receipts_missing:adversarial_verify" in experiment.validate_artifact(
        changed, root=tmp_path, require_terminal=True
    )

    changed = deepcopy(artifact)
    changed["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum_mismatch" in experiment.validate_artifact(
        changed, root=tmp_path, require_terminal=True
    )


def test_blocked_artifact_has_zero_work_and_exact_gate_summary() -> None:
    """SCENARIO-VERIFY-7442-PRECONDITIONS: external absence is not fabricated."""

    failed = experiment.gate_row(
        "span_protocol_ready",
        experiment.PROTOCOL_PATH.as_posix(),
        "span_protocol_ready_score",
        "==",
        1,
        None,
        principle="Only ready protocol evidence can authorize model work.",
    )
    artifact = experiment.build_blocked_artifact([failed])

    assert artifact["honest_verdict"] == "blocked_span_protocol_ready"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["model_invoked"] is False
    assert set(artifact["invocation_counts"].values()) == {0}
    assert artifact["sample_size_budget"]["unstarted"] == 96
    assert artifact["gate_check_summary"]["observed_value"] is None


def test_terminal_receipt_names_and_date_are_closed() -> None:
    """SCENARIO-VERIFY-7442-TERMINAL: exact command names and date cannot drift."""

    assert experiment.receipts_pass(_terminal_receipts(), experiment.TERMINAL_CHECK_NAMES)
    missing = [row for row in _terminal_receipts() if row["name"] != "adversarial_verify"]
    assert experiment.receipts_pass(missing, experiment.TERMINAL_CHECK_NAMES) is False
    assert experiment.date_argument("20260920") == "20260920"
    with pytest.raises(ValueError, match="date must be 20260920"):
        experiment.date_argument("20260919")


def test_validation_rejects_invalid_declarations_and_blocked_prefix(tmp_path: Path) -> None:
    """REQ-VERIFY-7442: ordinary schema fields keep closed meanings."""

    artifact = experiment.build_artifact_for_test(tmp_path)
    for field, replacement, expected in [
        ("MODEL_SPECS", [], "declaration_mismatch:MODEL_SPECS"),
        ("inference_substrate_class", "no_model_load", "substrate_class_mismatch"),
        ("execution_venue", "container", "declaration_mismatch:execution_venue"),
        ("field_principles", {}, "field_principles_mismatch"),
        ("verdict_class", "unknown", "verdict_class_invalid"),
    ]:
        changed = deepcopy(artifact)
        changed[field] = replacement
        changed["reproducibility_checksum"] = experiment.artifact_checksum(changed)
        assert expected in experiment.validate_artifact(changed, root=tmp_path, require_terminal=True)

    blocked = experiment.build_blocked_artifact(
        [
            experiment.gate_row(
                "missing", "upstream", "field", "==", True, None, principle="fixture"
            )
        ]
    )
    blocked["honest_verdict"] = "wrong"
    blocked["reproducibility_checksum"] = experiment.artifact_checksum(blocked)
    assert "blocked_verdict_prefix_invalid" in experiment.validate_artifact(
        blocked, root=tmp_path, require_terminal=False
    )


def test_fixture_canary_failure_remains_complete_null(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7442-DEVELOPMENT: completed low-quality canary is a null."""

    artifact = experiment.build_artifact_for_test(tmp_path, canary_open=False)
    assert experiment.validate_artifact(artifact, root=tmp_path, require_terminal=True) == []
    assert artifact["honest_verdict"] == "complete_null_development_gate_closed"
    assert artifact["verdict_class"] == "null"
    assert artifact["span_capture_complete_score"] == 0
    assert artifact["sample_size_budget"]["attempted"] == 0
    assert artifact["sample_size_budget"]["unstarted"] == 96
    assert all(row["disposition"] == "unstarted" for row in artifact["extraction_rows"])
