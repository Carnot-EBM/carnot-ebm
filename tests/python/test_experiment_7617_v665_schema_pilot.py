"""Changed-behavior tests for REQ-REPORT-7617 and its scenarios."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7617_v665_schema_pilot as exp
from carnot import experiment_7616_v665_evidence_schema as schema


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def records() -> list[dict[str, object]]:
    """Use the frozen pilot bytes so tests cover the real lossless contract."""

    path = (
        ROOT / "results/raw/experiment_7602_v664_evidence_requalification/pilot_model_inputs.jsonl"
    )
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _valid_output(record: dict[str, object]) -> str:
    authority = schema.build_schema_authority(record)
    response_id = authority["allowed_response_sentence_ids"][0]
    source_id = authority["allowed_source_sentence_ids"][0]
    return json.dumps(
        [
            {
                "response_sentence_id": response_id,
                "source_sentence_ids": [source_id],
                "relation": "supports",
                "entity_type": "other",
                "abstention_reason": "",
            }
        ]
    )


def _row(
    record: dict[str, object],
    arm: str,
    *,
    valid: bool = True,
    duration_s: float = 2.0,
) -> dict[str, object]:
    text = _valid_output(record) if valid else "[{"
    parsed = exp.parse_pilot_response(record, text, finish_reason="stop")
    return {
        "component_hash": record["component_hash"],
        "arm": arm,
        "status": "completed",
        "transport_completed": True,
        "finish_reason": "stop",
        "generation_s": duration_s,
        "parser_result": parsed,
        "censored": not valid,
        "seed": exp.RANDOM_SEED,
        "direction": "syntax_transport_only",
        "numerator": int(valid),
        "denominator": 1,
        "raw_provenance": "test_fixture",
    }


def test_paired_requests_differ_only_by_supported_grammar(records: list[dict[str, object]]) -> None:
    """SCENARIO-REPORT-7617-PAIRED: both arms keep one lossless input."""

    control = exp.build_arm_request(records[0], exp.CONTROL_ARM)
    grammar = exp.build_arm_request(records[0], exp.GRAMMAR_ARM)
    grammar_without_decoder = deepcopy(grammar)
    compiled = grammar_without_decoder.pop("grammar")

    assert grammar_without_decoder == control
    assert isinstance(compiled, str) and "root" in compiled
    assert control["temperature"] == 0.0
    assert control["max_tokens"] == 512
    assert control["chat_template_kwargs"] == {"enable_thinking": False}
    assert "/no_think" in control["messages"][0]["content"]
    user_input = json.loads(control["messages"][1]["content"])
    assert set(user_input) == {"source_sentences", "question_sentences", "response_sentences"}


def test_seeded_order_pairs_every_group_once(records: list[dict[str, object]]) -> None:
    """SCENARIO-REPORT-7617-PAIRED: order is frozen without changing sample size."""

    first = exp.seeded_paired_schedule(records, seed=exp.ARM_ORDER_SEED)
    second = exp.seeded_paired_schedule(records, seed=exp.ARM_ORDER_SEED)

    assert first == second
    assert len(first) == 16
    assert {row["arm"] for row in first} == {exp.CONTROL_ARM, exp.GRAMMAR_ARM}
    assert all(
        [row["arm"] for row in first if row["component_hash"] == record["component_hash"]]
        in ([exp.CONTROL_ARM, exp.GRAMMAR_ARM], [exp.GRAMMAR_ARM, exp.CONTROL_ARM])
        for record in records
    )


def test_independent_parser_keeps_invalid_ids_and_truncation_failed(
    records: list[dict[str, object]],
) -> None:
    """SCENARIO-REPORT-7617-SELECT: grammar acceptance never replaces validation."""

    valid = exp.parse_pilot_response(records[0], _valid_output(records[0]), finish_reason="stop")
    bad_id = json.loads(_valid_output(records[0]))
    bad_id[0]["source_sentence_ids"] = ["S999"]
    invalid = exp.parse_pilot_response(records[0], json.dumps(bad_id), finish_reason="stop")
    truncated = exp.parse_pilot_response(
        records[0], _valid_output(records[0]), finish_reason="length"
    )

    assert valid["valid_completed"] is True
    assert valid["unknown_fraction"] >= 0.0
    assert invalid["valid_completed"] is False
    assert invalid["invalid_id_reference"] is True
    assert truncated["valid_completed"] is False
    assert truncated["parser_error"] == "truncated_output"


def test_selection_prefers_grammar_then_control_and_can_stop(
    records: list[dict[str, object]],
) -> None:
    """SCENARIO-REPORT-7617-SELECT: the frozen six-of-eight rule fails closed."""

    grammar_wins = [
        *[_row(record, exp.GRAMMAR_ARM, valid=index < 6) for index, record in enumerate(records)],
        *[_row(record, exp.CONTROL_ARM, valid=True) for record in records],
    ]
    chosen = exp.select_configuration(grammar_wins)
    assert chosen["selected_arm"] == exp.GRAMMAR_ARM
    assert chosen["evidence_transport_ready_score"] == 1

    control_wins = deepcopy(grammar_wins)
    for row in control_wins:
        if row["arm"] == exp.GRAMMAR_ARM:
            row["parser_result"]["valid_completed"] = False
    assert exp.select_configuration(control_wins)["selected_arm"] == exp.CONTROL_ARM

    no_winner = [
        _row(record, arm, valid=index < 5)
        for arm in exp.ARMS
        for index, record in enumerate(records)
    ]
    stopped = exp.select_configuration(no_winner)
    assert stopped["selected_arm"] is None
    assert stopped["scale_decision"] == "stop_scaling"


def test_projection_uses_p90_and_keeps_three_fixed_rosters(
    records: list[dict[str, object]],
) -> None:
    """SCENARIO-REPORT-7617-PROJECT: rosters are independent and never resized."""

    rows = [
        _row(record, exp.GRAMMAR_ARM, duration_s=float(index + 1))
        for index, record in enumerate(records)
    ]
    projections = exp.project_fixed_rosters(rows, model_load_s=100.0, validation_reserve_s=300.0)

    assert set(projections) == {"fit_tune_policy", "online", "evaluation"}
    assert [projections[name]["roster_size"] for name in projections] == [120, 80, 40]
    assert all(row["per_row_p90_s"] == 8.0 for row in projections.values())
    assert projections["fit_tune_policy"]["projected_generation_s"] == 1200.0
    assert projections["fit_tune_policy"]["projected_total_task_s"] == 1600.0
    assert all(row["feasible_score"] == 1 for row in projections.values())


def test_required_artifact_fields_validate(records: list[dict[str, object]]) -> None:
    """REQ-REPORT-7617: required terminal semantics remain explicit."""

    rows = [_row(record, arm) for record in records for arm in exp.ARMS]
    artifact = exp.build_test_artifact(rows)

    assert exp.validate_artifact(artifact, root=ROOT) == []
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["verdict_class"] == "null"
    assert artifact["MODEL_SPECS"] == ["unsloth/Qwen3.8-27B-GGUF"]
    assert artifact["semantic_benefit_claim"] is False
    assert artifact["sample_size_budget"]["observed"] == 8
    assert len(artifact["paired_pilot_rows"]) == 16
    assert {gate["category"] for gate in artifact["acceptance_gate_results"]} == {
        "validity",
        "readiness",
        "benefit",
        "retention",
        "freshness",
    }


def test_changed_behavior_guards_fail_closed(records: list[dict[str, object]]) -> None:
    """REQ-REPORT-7617: malformed arm, pair, and timing inputs cannot scale."""

    with pytest.raises(ValueError, match="pilot_arm_invalid"):
        exp.build_arm_request(records[0], "unknown_arm")
    with pytest.raises(ValueError, match="exactly_eight_disjoint"):
        exp.seeded_paired_schedule(records[:7])
    with pytest.raises(ValueError, match="exactly_sixteen"):
        exp.select_configuration([])
    with pytest.raises(ValueError, match="paired_arm_groups_invalid"):
        exp.select_configuration(
            [_row(record, exp.CONTROL_ARM) for record in records for _ in range(2)]
        )
    with pytest.raises(ValueError, match="completed_generation_timing_absent"):
        exp.project_fixed_rosters([], model_load_s=0.0)
