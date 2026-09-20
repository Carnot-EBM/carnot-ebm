"""Tests for the V654 factual claim-span development canary.

Spec refs: REQ-VERIFY-7467 and SCENARIO-VERIFY-7467-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7442_v652_span_capture as engine
from carnot import experiment_7451_v653_span_capture as predecessor
from carnot import experiment_7467_v654_factual_span_canary as capture


REPO = Path(__file__).resolve().parents[2]


def _response(
    row: dict[str, object],
    reply: str,
    *,
    terminal_state: str = "response",
    finish_reason: str | None = "stop",
) -> dict[str, object]:
    return {
        "raw_request": {"messages": [{"role": "user", "content": row["prompt"]}]},
        "raw_response": {"reply": reply},
        "raw_reply": reply,
        "attempted": True,
        "terminal_state": terminal_state,
        "finish_reason": finish_reason,
        "prompt_tokens": 10,
        "completion_tokens": 4,
        "latency_s": 0.1,
        "error": "transport" if terminal_state != "response" else None,
    }


def _valid_reply(row: dict[str, object]) -> str:
    paragraph = str(row["paragraph"])
    if row["arm"] == "span":
        return json.dumps({"claims": [[0, len(paragraph)]]})
    return json.dumps({"claims": [paragraph]})


def _development_schedule() -> list[dict[str, object]]:
    panel = capture.select_development_panel(REPO)
    with capture.factual_span_contract():
        return capture.build_development_schedule(panel["paragraphs"])


def _development_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for schedule in _development_schedule():
        reply = (
            '{"claims":[]}'
            if schedule["canary_kind"] == "nonfactual_control"
            else _valid_reply(schedule)
        )
        rows.append(capture.build_capture_row(schedule, _response(schedule, reply)))
    return rows


def test_req_verify_7467_contract_is_scoped_and_specified_first() -> None:
    """REQ-VERIFY-7467 fixes identity, compute class, budget, and validation scope."""

    spec = (REPO / capture.SPEC_PATH).read_text(encoding="utf-8")
    section = spec[spec.index("REQ-VERIFY-7467") :]
    for anchor in (
        "SCENARIO-VERIFY-7467-SELECTION",
        "SCENARIO-VERIFY-7467-GATE",
        "SCENARIO-VERIFY-7467-ROSTER",
        "SCENARIO-VERIFY-7467-VALUE",
        "SCENARIO-VERIFY-7467-TERMINAL",
    ):
        assert anchor in section
    assert capture.MODEL_SPECS == ["unsloth/Qwen3.8-27B-GGUF"]
    assert capture.INFERENCE_SUBSTRATE_CLASS == "model_bounded_generation"
    assert capture.MAX_NEW_TOKENS == 256
    assert capture.MAX_GENERATION_CALLS == 108
    assert capture.VALIDATION_MANIFEST.test_paths == (capture.TEST_PATH.as_posix(),)
    assert capture.VALIDATION_MANIFEST.changed_modules == (capture.MODULE_PATH.as_posix(),)


def test_scenario_verify_7467_selection_is_annotated_exact_and_disjoint() -> None:
    """SCENARIO-VERIFY-7467-SELECTION freezes annotated text before inference."""

    selected = capture.select_development_panel(REPO)
    assert selected["selection_model_outputs_consulted"] is False
    assert selected["factual_count"] == 4
    assert selected["nonfactual_count"] == 2
    assert [row["response_id"] for row in selected["paragraphs"][:4]] == [
        "0",
        "7",
        "12",
        "18",
    ]
    assert all(row["human_annotations"] == [] for row in selected["paragraphs"][:4])
    assert all(
        row["human_annotation_status"] == "no_unsupported_span_annotated"
        for row in selected["paragraphs"][:4]
    )
    assert all(row["canary_kind"] == "factual" for row in selected["paragraphs"][:4])
    assert all(row["canary_kind"] == "nonfactual_control" for row in selected["paragraphs"][4:])
    assert selected["evaluation_overlap"] == {
        "group_ids": [],
        "paragraph_sha256": [],
        "response_ids": [],
    }
    assert selected["selection_rules"]["source_order"] == "response.jsonl byte order"
    assert selected["selection_rules"]["model_success_filter"] is False

    changed = REPO / "results/raw/experiment_7437_v652_span_protocol/sealed_schedule.json"
    data = json.loads(changed.read_text(encoding="utf-8"))
    assert capture.select_development_panel(REPO, sealed_schedule=data) == selected
    with pytest.raises(ValueError, match="annotated_factual_selection_mismatch"):
        capture.select_development_panel(REPO, expected_response_ids=("7", "0", "12", "18"))
    overlapping = deepcopy(data)
    overlapping["rows"].append(
        {
            "response_id": "not-a-selected-id",
            "group_id": "not-a-selected-group",
            "paragraph_sha256": selected["paragraphs"][0]["paragraph_sha256"],
        }
    )
    with pytest.raises(ValueError, match="development_evaluation_overlap"):
        capture.select_development_panel(REPO, sealed_schedule=overlapping)


def test_scenario_verify_7467_gate_separates_factual_and_controls() -> None:
    """SCENARIO-VERIFY-7467-GATE keeps factual and correct-empty credit separate."""

    rows = _development_rows()
    gate = capture.reduce_development_gate(rows)
    assert gate["capture_open"] is True
    assert gate["factual"]["planned_per_arm"] == 4
    assert gate["factual"]["usable_by_arm"] == {"span": 4, "verbatim": 4}
    assert gate["nonfactual"]["planned_per_arm"] == 2
    assert gate["nonfactual"]["correct_empty_by_arm"] == {"span": 2, "verbatim": 2}
    assert gate["correct_empty_counts_as_factual_success"] is False

    factual_span = next(
        index
        for index, row in enumerate(rows)
        if row["canary_kind"] == "factual" and row["arm"] == "span"
    )
    schedule = _development_schedule()[factual_span]
    rows[factual_span] = capture.build_capture_row(schedule, _response(schedule, '{"claims":[]}'))
    still_open = capture.reduce_development_gate(rows)
    assert still_open["capture_open"] is True
    assert still_open["factual"]["usable_by_arm"]["span"] == 3
    assert still_open["factual"]["failures"]["span"][0]["disposition"] == "correct_empty"
    assert rows[factual_span]["factual_recall_success"] is False

    second = next(
        index
        for index, row in enumerate(rows)
        if row["canary_kind"] == "factual" and row["arm"] == "span" and index != factual_span
    )
    schedule = _development_schedule()[second]
    rows[second] = capture.build_capture_row(schedule, _response(schedule, '{"claims":[]}'))
    closed = capture.reduce_development_gate(rows)
    assert closed["capture_open"] is False
    assert closed["repeated_factual_canary_failure_retires_construction"] is True

    control = next(
        index
        for index, row in enumerate(_development_rows())
        if row["canary_kind"] == "nonfactual_control"
    )
    control_rows = _development_rows()
    schedule = _development_schedule()[control]
    control_rows[control] = capture.build_capture_row(
        schedule, _response(schedule, _valid_reply(schedule))
    )
    assert capture.reduce_development_gate(control_rows)["capture_open"] is False
    assert control_rows[control]["correct_empty_control"] is False
    assert control_rows[control]["factual_recall_success"] is None


def test_scenario_verify_7467_roster_and_reply_shards_remain_accountable() -> None:
    """SCENARIO-VERIFY-7467-ROSTER retains unstarted calls and empty replies."""

    checks, context = capture.collect_preconditions(REPO)
    assert all(row["passed"] is True for row in checks)
    assert len(context["development_schedule"]) == 12
    assert len(context["evaluation_schedule"]) == 96
    predecessor = json.loads((REPO / capture.PREDECESSOR_PATH).read_text(encoding="utf-8"))
    assert (
        context["evaluation_schedule"][0]["evaluation_schedule_sha256"]
        == predecessor["protocol_receipt"]["schedule_sha256"]
    )
    assert all(
        path not in {capture.MODULE_PATH, capture.WRAPPER_PATH, capture.TEST_PATH}
        for path in [Path(str(row.get("path"))) for row in checks]
    )

    rows = _development_rows()
    references = [
        {
            "path": f"responses/{index}.json",
            "sha256": f"sha256:{index:064x}",
            "bytes": index + 1,
            "phase": "responses",
        }
        for index in range(len(rows))
    ]
    exposed = capture.build_raw_reply_shards(rows, references)
    assert len(exposed) == 4
    assert all(row["correct_empty_control"] is True for row in exposed)
    assert all(row["raw_reply_empty"] is False for row in exposed)
    assert all(str(row["path"]).startswith(capture.RAW_DIR.as_posix()) for row in exposed)
    with pytest.raises(ValueError, match="reply_shard_count_mismatch"):
        capture.build_raw_reply_shards(rows, references[:-1])


def test_scenario_verify_7467_value_uses_paired_token_ci_upper() -> None:
    """SCENARIO-VERIFY-7467-VALUE requires the registered token confidence bound."""

    differences = [-3.0] * 48
    token_ci = capture.paired_interval(differences)
    completion_ci = capture.paired_interval([1.0] * 48)
    assert token_ci["ci95_high"] == -3.0
    assert completion_ci["ci95_low"] == 1.0
    gates = capture.span_value_gates(completion_ci, token_ci, qualifier_delta=0.0)
    assert all(gate["passed"] is True for gate in gates)
    assert capture.reduce_span_value(completion_ci, token_ci, 0.0) == 1

    zero_cost = capture.paired_interval([0.0] * 48)
    assert capture.reduce_span_value(completion_ci, zero_cost, 0.0) == 0
    qualifier_loss = capture.span_value_gates(completion_ci, token_ci, qualifier_delta=-0.1)
    assert (
        next(row for row in qualifier_loss if row["check"] == "qualifier_nonloss")["passed"]
        is False
    )
    assert capture.paired_interval([]) == {
        "ci95_high": None,
        "ci95_low": None,
        "draws": capture.BOOTSTRAP_DRAWS,
        "estimate": None,
        "pairs": 0,
        "seed": capture.RANDOM_SEED,
    }


def test_scenario_verify_7467_terminal_fixture_replays_and_detects_mutation() -> None:
    """SCENARIO-VERIFY-7467-TERMINAL cold-reduces the complete field contract."""

    artifact = capture.build_fixture_artifact()
    assert capture.validate_artifact(artifact, require_terminal=True) == []
    assert capture.independent_reduce_artifact(artifact, require_terminal=True) == []
    assert artifact["schema"] == capture.SCHEMA
    assert artifact["experiment_id"] == capture.EXPERIMENT_ID
    assert artifact["milestone"] == capture.MILESTONE
    assert artifact["phase"] == capture.PHASE
    assert len(artifact["development_rows"]) == 12
    assert artifact["factual_development_gate"] == artifact["development_gate"]
    assert artifact["sample_size_budget"]["planned"] == 108
    assert artifact["sample_size_budget"]["development_planned"] == 12
    assert "factual_development_gate" in artifact["field_principles"]
    assert "raw_reply_shards" in artifact["field_principles"]
    assert (
        artifact["qualifier_retention_report"]["human_natural_language_annotations"][
            "factual_truth_established"
        ]
        is False
    )

    changed = deepcopy(artifact)
    changed["factual_development_gate"]["capture_open"] = False
    changed["reproducibility_checksum"] = capture.artifact_checksum(changed)
    assert "factual_development_gate_mismatch" in capture.validate_artifact(
        changed, require_terminal=True
    )

    mutations = (
        ("development_gate", {}, "development_gate_mismatch"),
        ("paragraph_selection", {}, "paragraph_selection_mismatch"),
        ("paired_output_token_ci95", {}, "paired_output_token_ci95_mismatch"),
        (
            "sample_size_budget",
            {**artifact["sample_size_budget"], "planned": 96, "development_planned": 8},
            "sample_size_budget_planned_mismatch",
        ),
        ("semantic_scope", {}, "semantic_scope_mismatch"),
        ("field_principles", {}, "field_principles_mismatch"),
        ("reproducibility_checksum", "sha256:changed", "reproducibility_checksum_mismatch"),
    )
    for field, replacement, expected in mutations:
        changed = deepcopy(artifact)
        changed[field] = replacement
        if field != "reproducibility_checksum":
            changed["reproducibility_checksum"] = capture.artifact_checksum(changed)
        errors = capture.validate_artifact(changed, require_terminal=True)
        assert expected in errors
    changed = deepcopy(artifact)
    changed["sample_size_budget"]["planned"] = 108
    changed["sample_size_budget"]["development_planned"] = 8
    changed["reproducibility_checksum"] = capture.artifact_checksum(changed)
    assert "sample_size_budget_development_mismatch" in capture.validate_artifact(
        changed, require_terminal=True
    )


def test_scenario_verify_7467_defensive_replay_and_blocked_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-VERIFY-7467-TERMINAL keeps malformed evidence fail-closed."""

    artifact = capture.build_fixture_artifact()
    attempted = [*artifact["development_rows"], *artifact["extraction_rows"]]
    references = [
        {
            "path": f"responses/{index}.json",
            "sha256": f"sha256:{index:064x}",
            "bytes": index + 1,
            "phase": "responses",
        }
        for index in range(len(attempted))
    ]
    with_replies = deepcopy(artifact)
    with_replies["source_artifact_hashes"]["current_response_shards"] = references
    with_replies["raw_reply_shards"] = capture.build_raw_reply_shards(attempted, references)
    with_replies["reproducibility_checksum"] = capture.artifact_checksum(with_replies)
    assert capture.validate_artifact(with_replies, require_terminal=True) == []

    wrong_replies = deepcopy(with_replies)
    wrong_replies["raw_reply_shards"] = []
    wrong_replies["reproducibility_checksum"] = capture.artifact_checksum(wrong_replies)
    assert "raw_reply_shards_mismatch" in capture.validate_artifact(
        wrong_replies, require_terminal=True
    )
    short_references = deepcopy(with_replies)
    short_references["source_artifact_hashes"]["current_response_shards"] = references[:-1]
    short_references["reproducibility_checksum"] = capture.artifact_checksum(short_references)
    assert "reply_shard_count_mismatch" in capture.validate_artifact(
        short_references, require_terminal=True
    )

    blocked = capture.build_blocked_artifact(predecessor_checks := predecessor_gate_rows())
    assert predecessor_checks
    assert blocked["verdict_class"] == "blocked"
    assert capture.validate_artifact(blocked, require_terminal=True) == []

    monkeypatch.setattr(
        capture,
        "select_development_panel",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("selection")),
    )
    errors = capture.validate_artifact(artifact, require_terminal=True)
    assert "paragraph_selection_replay_failed:ValueError:selection" in errors
    checks, context = capture.collect_preconditions(REPO)
    selection_gate = next(
        row for row in checks if row["check"] == "frozen_factual_development_panel"
    )
    assert selection_gate["passed"] is False
    assert context["development_schedule"] == []

    monkeypatch.setattr(
        capture,
        "reduce_evaluation",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("reduction")),
    )
    assert "v654_reduction_failed:ValueError:reduction" in capture.validate_artifact(
        artifact, require_terminal=True
    )


def predecessor_gate_rows() -> list[dict[str, object]]:
    """Return exact failed historical-shaped gates for blocked fixture coverage."""

    return predecessor.lifecycle_gate_rows({})


def test_scenario_verify_7467_closed_gate_verdicts_are_explicit() -> None:
    """SCENARIO-VERIFY-7467-GATE distinguishes factual retirement from controls."""

    assert capture.development_terminal_outcome({"capture_open": True}) == {}
    assert (
        capture.development_terminal_outcome(
            {
                "capture_open": False,
                "repeated_factual_canary_failure_retires_construction": True,
            }
        )["honest_verdict"]
        == "complete_null_factual_span_canary_development_gate_closed"
    )
    assert (
        capture.development_terminal_outcome(
            {
                "capture_open": False,
                "repeated_factual_canary_failure_retires_construction": False,
            }
        )["honest_verdict"]
        == "complete_null_factual_span_empty_control_gate_closed"
    )


def test_req_verify_7467_contract_restores_engine_and_rejects_date_drift() -> None:
    """REQ-VERIFY-7467 isolates reused globals and fixes fresh-process commands."""

    before = (engine.EXPERIMENT_ID, engine.DEVELOPMENT_CALLS, engine._ShardRecorder)
    with capture.factual_span_contract():
        assert engine.EXPERIMENT_ID == capture.EXPERIMENT_ID
        assert engine.DEVELOPMENT_CALLS == 12
        assert engine._ShardRecorder is capture.FactualShardRecorder
    assert (engine.EXPERIMENT_ID, engine.DEVELOPMENT_CALLS, engine._ShardRecorder) == before

    commands = capture.terminal_commands(REPO, REPO / "results/private-exp7467.json")
    assert tuple(row.name for row in commands) == capture.TERMINAL_CHECK_NAMES
    assert capture.WRAPPER_PATH.as_posix() in commands[0].argv
    assert "experiment_7467_v654_factual_span_canary" in " ".join(commands[1].argv)
    assert capture.date_argument(capture.RUN_DATE) == capture.RUN_DATE
    with pytest.raises(ValueError, match="date must be 20260920"):
        capture.date_argument("20260919")
    assert capture.validate_artifact([], require_terminal=True) == ["artifact_not_object"]
