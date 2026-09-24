"""Focused tests for REQ-REPORT-7631 and its paired-pilot scenarios."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7616_v665_evidence_schema as schema
from carnot import experiment_7631_v666_schema_pilot as exp


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def records() -> list[dict[str, object]]:
    """Load the eight immutable pilot groups named by Exp7616."""

    path = ROOT / exp.PILOT_INPUT
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _valid_output(record: dict[str, object]) -> str:
    authority = schema.build_schema_authority(record)
    return json.dumps(
        [
            {
                "response_sentence_id": authority["allowed_response_sentence_ids"][0],
                "source_sentence_ids": [authority["allowed_source_sentence_ids"][0]],
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
    truncated: bool = False,
    request_s: float = 2.0,
) -> dict[str, object]:
    response = _valid_output(record) if valid else "[{"
    parsed = exp.parse_pilot_response(
        record, response, finish_reason="length" if truncated else "stop"
    )
    return {
        "component_hash": record["component_hash"],
        "arm": arm,
        "status": "completed",
        "transport_completed": True,
        "finish_reason": "length" if truncated else "stop",
        "request_s": request_s,
        "first_token_latency_s": request_s / 2,
        "tokenization_s": 0.25,
        "grammar_compile_s": 0.1 if arm == exp.CONSTRAINED_ARM else 0.0,
        "prompt_tokens": 100,
        "output_tokens": 20,
        "raw_request_bytes": 10,
        "raw_response_bytes": len(response.encode()),
        "raw_request_sha256": exp.text_hash("request"),
        "raw_response_sha256": exp.text_hash(response),
        "parser_result": parsed,
        "schema_valid": parsed["accepted"],
        "pointer_valid": parsed["pointer_valid"],
        "independent_rejection_reasons": parsed["rejection_reasons"],
        "censored": truncated or not valid,
        "seed": exp.RANDOM_SEED,
        "direction": "transport_readiness_only",
        "numerator": int(valid and not truncated),
        "denominator": 1,
        "raw_provenance": "test_fixture",
    }


def test_requests_are_lossless_and_differ_only_by_decoder(
    records: list[dict[str, object]],
) -> None:
    """SCENARIO-REPORT-7631-PAIRED: paired inputs and budgets are identical."""

    explicit = exp.build_arm_request(records[0], exp.EXPLICIT_ARM)
    constrained = exp.build_arm_request(records[0], exp.CONSTRAINED_ARM)
    grammar = constrained.pop("grammar")

    assert constrained == explicit
    assert isinstance(grammar, str) and "root" in grammar
    assert explicit["temperature"] == 0.0
    assert explicit["seed"] == 7631
    assert explicit["max_tokens"] == 512
    visible = json.loads(explicit["messages"][1]["content"])
    assert set(visible) == {"source_sentences", "question_sentences", "response_sentences"}


def test_counterbalanced_schedule_keeps_eight_independent_groups(
    records: list[dict[str, object]],
) -> None:
    """SCENARIO-REPORT-7631-PAIRED: order does not multiply sample size."""

    schedule = exp.counterbalanced_schedule(records)
    assert schedule == exp.counterbalanced_schedule(records)
    assert len(schedule) == 16
    assert {row["arm"] for row in schedule} == set(exp.ARMS)
    assert all(
        {row["arm"] for row in schedule if row["component_hash"] == record["component_hash"]}
        == set(exp.ARMS)
        for record in records
    )


def test_parser_reports_pointer_and_truncation_failures(
    records: list[dict[str, object]],
) -> None:
    """REQ-REPORT-7631: JSON shape alone is not semantic pointer success."""

    valid = exp.parse_pilot_response(records[0], _valid_output(records[0]), finish_reason="stop")
    wrong = json.loads(_valid_output(records[0]))
    wrong[0]["source_sentence_ids"] = ["S999"]
    invalid = exp.parse_pilot_response(records[0], json.dumps(wrong), finish_reason="stop")
    truncated = exp.parse_pilot_response(
        records[0], _valid_output(records[0]), finish_reason="length"
    )
    assert valid["pointer_valid"] is True
    assert invalid["pointer_valid"] is False
    assert "source_sentence_id_invalid" in invalid["rejection_reasons"]
    assert truncated["valid_completed"] is False
    assert "truncated_output" in truncated["rejection_reasons"]


def test_selection_requires_eight_of_eight_and_authenticated_decoder(
    records: list[dict[str, object]],
) -> None:
    """SCENARIO-REPORT-7631-SELECT: both arms fail closed at less than 8/8."""

    rows = [_row(record, arm) for record in records for arm in exp.ARMS]
    chosen = exp.select_configuration(rows, decoder_supported=True)
    assert chosen["selected_arm"] == exp.CONSTRAINED_ARM
    assert chosen["evidence_transport_ready_score"] == 1

    unsupported = exp.select_configuration(rows, decoder_supported=False)
    assert unsupported["selected_arm"] == exp.EXPLICIT_ARM
    constrained_bad = deepcopy(rows)
    next(row for row in constrained_bad if row["arm"] == exp.CONSTRAINED_ARM)["pointer_valid"] = (
        False
    )
    assert exp.select_configuration(constrained_bad, decoder_supported=True)["selected_arm"] == (
        exp.EXPLICIT_ARM
    )

    both_bad = deepcopy(constrained_bad)
    next(row for row in both_bad if row["arm"] == exp.EXPLICIT_ARM)["censored"] = True
    stopped = exp.select_configuration(both_bad, decoder_supported=True)
    assert stopped["selected_arm"] is None
    assert stopped["scale_decision"] == "stop_scaling"


def test_projection_uses_selected_arm_load_tokenization_and_p90(
    records: list[dict[str, object]],
) -> None:
    """SCENARIO-REPORT-7631-PROJECT: three fixed rosters retain their sizes."""

    rows = [
        _row(record, exp.CONSTRAINED_ARM, request_s=float(index + 1))
        for index, record in enumerate(records)
    ]
    projections = exp.project_fixed_rosters(rows, model_load_s=100.0, validation_reserve_s=300.0)
    assert list(projections) == ["fit_tune_policy", "online", "evaluation"]
    assert [row["roster_size"] for row in projections.values()] == [120, 80, 40]
    assert all(row["request_p90_s"] == 8.0 for row in projections.values())
    assert all(row["tokenization_p90_s"] == 0.25 for row in projections.values())
    assert projections["fit_tune_policy"]["projected_capture_s"] == 1090.0
    assert projections["fit_tune_policy"]["projected_total_s"] == 1390.0
    assert all(row["feasible_score"] == 1 for row in projections.values())


def test_schema_role_custody_is_exact() -> None:
    """SCENARIO-REPORT-7631-CUSTODY: V663/V665 role identity is unchanged."""

    custody = exp.authenticate_schema_custody(ROOT)
    assert custody["passed"] is True
    assert custody["selection_salt"] == "v663-evidence-20260924"
    assert custody["restored_group_count"] == 480
    assert custody["selected_scored_group_count"] == 240
    assert custody["role_counts"] == {
        "evaluation": 40,
        "fit": 80,
        "online": 80,
        "pilot": 8,
        "policy": 20,
        "tune": 20,
    }
    assert custody["fit_partition_counts"] == {"anchor": 16, "optimization": 64}


def test_blocked_artifact_names_planned_model_and_zero_current_work() -> None:
    """SCENARIO-REPORT-7631-RESOURCE: external absence is terminal blocked."""

    failed = exp.gate_row(
        "owned_cuda_capacity",
        category="readiness",
        upstream="nvidia-smi",
        path="/tmp/inventory",
        field="owned_idle_device",
        operator="eq",
        expected=True,
        observed=False,
        passed=False,
    )
    artifact = exp.build_blocked_artifact([failed], duration_s=1.0)
    assert exp.validate_artifact(artifact) == []
    assert artifact["honest_verdict"] == "complete_blocked_owned_cuda_capacity"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["inference_substrate"] == "no_model_load"
    assert artifact["MODEL_SPECS"] == []
    assert artifact["planned_MODEL_SPECS"] == ["unsloth/Qwen3.8-27B-GGUF"]
    assert artifact["model_invoked"] is False
    assert set(artifact["invocation_counts"].values()) == {0}
    assert artifact["gate_check_summary"]["first_failure"] == failed


def test_complete_fixture_validates_and_cold_reduces(
    records: list[dict[str, object]], tmp_path: Path
) -> None:
    """SCENARIO-REPORT-7631-TERMINAL: stored rows control the comparison."""

    rows = [_row(record, arm) for record in records for arm in exp.ARMS]
    artifact = exp.build_test_artifact(rows)
    assert exp.validate_artifact(artifact) == []
    assert artifact["selected_config_path"] == exp.SELECTED_CONFIG_PATH.as_posix()
    assert artifact["semantic_benefit_claim"] is False
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(artifact), encoding="utf-8")
    reduced = exp.independent_reduce_artifact(candidate)
    assert reduced["passed"] is True
    assert reduced["selection"]["selected_arm"] == exp.CONSTRAINED_ARM


def test_invalid_inputs_fail_closed(records: list[dict[str, object]]) -> None:
    """REQ-REPORT-7631: malformed arms, groups, rows, and timings cannot scale."""

    with pytest.raises(ValueError, match="pilot_arm_invalid"):
        exp.build_arm_request(records[0], "wrong")
    with pytest.raises(ValueError, match="exactly_eight_disjoint"):
        exp.counterbalanced_schedule(records[:7])
    with pytest.raises(ValueError, match="exactly_sixteen"):
        exp.select_configuration([], decoder_supported=True)
    with pytest.raises(ValueError, match="paired_arm_groups_invalid"):
        exp.select_configuration(
            [_row(record, exp.EXPLICIT_ARM) for record in records for _ in range(2)],
            decoder_supported=True,
        )
    with pytest.raises(ValueError, match="completed_timing_absent"):
        exp.project_fixed_rosters([], model_load_s=0.0)
    with pytest.raises(ValueError, match="blocked_artifact_requires_failed_check"):
        exp.build_blocked_artifact([{"check": "passing", "passed": True}], duration_s=0.1)


def test_independent_reducer_handles_blocked_and_malformed_rows(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7631-TERMINAL: cold reduction fails closed without invention."""

    failure = exp.gate_row(
        "capacity",
        category="readiness",
        upstream="fixture",
        path="fixture",
        field="available",
        operator="eq",
        expected=True,
        observed=False,
        passed=False,
    )
    blocked_path = tmp_path / "blocked.json"
    blocked_path.write_text(json.dumps(exp.build_blocked_artifact([failure], duration_s=1.0)))
    assert exp.independent_reduce_artifact(blocked_path)["blocked_without_fabricated_rows"] is True

    malformed = exp.build_blocked_artifact([failure], duration_s=1.0)
    malformed["verdict_class"] = "null"
    malformed["reproducibility_checksum"] = exp.reproducibility_checksum(malformed)
    malformed_path = tmp_path / "malformed.json"
    malformed_path.write_text(json.dumps(malformed))
    reduced = exp.independent_reduce_artifact(malformed_path)
    assert reduced["passed"] is False
    assert reduced["errors"] == ["exactly_sixteen_paired_rows_required"]
