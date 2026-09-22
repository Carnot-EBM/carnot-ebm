"""Tests for the V659 4096-context native tool-source pilot.

Spec refs: REQ-VERIFY-7535 and SCENARIO-VERIFY-7535-*.
"""

from __future__ import annotations

from copy import deepcopy
import math

import pytest

from carnot import experiment_7535_v659_native_pilot as pilot


def _candidates() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(48):
        tool = ("grep", "curl", "search")[index % 3]
        context = f"{tool} source {index} " + ("x " * (index + 2))
        answer = f"answer {index}"
        rows.append(
            {
                "component_hash": pilot.sha256_text(f"component-{index}"),
                "official_split": "train",
                "tool_type": tool,
                "context": context,
                "question": f"question {index}",
                "answer": answer,
                "context_hash": pilot.sha256_text(context.casefold()),
                "answer_hash": pilot.sha256_text(answer.casefold()),
            }
        )
    return rows


def _token_count(text: str) -> int:
    return len(text.split()) + 1


def _complete_row(request: dict[str, object], index: int) -> dict[str, object]:
    order = list(request["option_order"])
    logits = {order[0]: 2.0 + index / 100.0, order[1]: -1.0}
    peak = max(logits.values())
    weights = {key: math.exp(value - peak) for key, value in logits.items()}
    total = sum(weights.values())
    prompt = str(request["prompt"])
    token_ids = list(range(1, int(request["prompt_token_count"]) + 1))
    return {
        **deepcopy(request),
        "call_id": f"call-{index:03d}",
        "disposition": "complete",
        "prompt_sha256": pilot.sha256_text(prompt),
        "prompt_utf8_bytes": len(prompt.encode("utf-8")),
        "prompt_token_ids": token_ids,
        "prompt_token_count": len(token_ids),
        "requested_score_position": len(token_ids) - 1,
        "actual_last_evaluated_position": len(token_ids) - 1,
        "display_labels": [" A", " B"],
        "label_token_ids": [10, 11],
        "label_to_option_id": {" A": order[0], " B": order[1]},
        "full_logits_by_option_id": logits,
        "probabilities_by_option_id": {key: weight / total for key, weight in weights.items()},
        "generated_tokens": 0,
        "forward_seconds": 0.25 + index / 1000.0,
        "state_reset": True,
        "server_receipt": {
            "transport": "native_in_process",
            "pid": 123,
            "process_start_ticks": 456,
            "gpu_uuid": "GPU-test",
            "n_ctx": 4096,
        },
        "error": None,
    }


def test_panel_freezes_twelve_label_blind_length_strata() -> None:
    """REQ-VERIFY-7535; SCENARIO-VERIFY-7535-PANEL."""

    excluded = {str(row["component_hash"]) for row in _candidates()[:3]}
    first = pilot.select_development_panel(
        _candidates(), excluded_component_hashes=excluded, token_count=_token_count
    )
    second = pilot.select_development_panel(
        list(reversed(_candidates())),
        excluded_component_hashes=excluded,
        token_count=_token_count,
    )
    assert first == second
    assert len(first["groups"]) == 12
    assert len({row["component_hash"] for row in first["groups"]}) == 12
    assert len({row["tool_type"] for row in first["groups"]}) >= 2
    assert {row["length_stratum"] for row in first["groups"]} == set(range(12))
    assert all(row["role"] == "development_outside_exp7533_roles" for row in first["groups"])
    assert all(row["component_hash"] != row["donor_component_hash"] for row in first["groups"])
    assert all(row["tool_type"] == row["donor_tool_type"] for row in first["groups"])
    assert not (excluded & {str(row["component_hash"]) for row in first["groups"]})


def test_panel_rejects_capacity_and_single_tool() -> None:
    """REQ-VERIFY-7535 prevents silent sample shrinking."""

    with pytest.raises(pilot.NativePilotError, match="development_capacity"):
        pilot.select_development_panel(
            _candidates()[:10], excluded_component_hashes=set(), token_count=_token_count
        )
    one_tool = _candidates()
    for row in one_tool:
        row["tool_type"] = "grep"
    with pytest.raises(pilot.NativePilotError, match="tool_type_span"):
        pilot.select_development_panel(
            one_tool, excluded_component_hashes=set(), token_count=_token_count
        )


def test_schedule_has_exact_intervention_shape_and_text() -> None:
    """REQ-VERIFY-7535; SCENARIO-VERIFY-7535-CUSTODY."""

    panel = pilot.select_development_panel(
        _candidates(), excluded_component_hashes=set(), token_count=_token_count
    )
    schedule = pilot.build_forward_schedule(panel["groups"], token_count=_token_count)
    assert len(schedule) == 72
    assert {row["condition"] for row in schedule} == {"original", "absent", "donor"}
    assert {tuple(row["option_order"]) for row in schedule} == set(pilot.OPTION_ORDERS)
    assert all(row["generated_token_budget"] == 0 for row in schedule)
    assert all(row["readout_kind"] == "option_logits" for row in schedule)
    assert all(row["prompt"] and row["prompt_sha256"] for row in schedule)
    donor_rows = [row for row in schedule if row["condition"] == "donor"]
    assert all(str(row["donor_context"]) in str(row["prompt"]) for row in donor_rows)


def test_native_reducer_accepts_exact_complete_rows() -> None:
    """REQ-VERIFY-7535 keeps transport independent of favorable predictions."""

    panel = pilot.select_development_panel(
        _candidates(), excluded_component_hashes=set(), token_count=_token_count
    )
    schedule = pilot.build_forward_schedule(panel["groups"], token_count=_token_count)
    rows = [_complete_row(request, index) for index, request in enumerate(schedule)]
    reduction = pilot.reduce_native_rows(rows, schedule)
    assert reduction["passed"] is True
    assert reduction["complete_forward_count"] == 72
    assert reduction["error_forward_count"] == 0
    assert reduction["generated_token_count"] == 0
    assert reduction["semantic_order_mapping_valid"] is True
    assert len(reduction["comparative_rows"]) == 12
    assert all(
        set(row["absolute_condition_probabilities"]) == {"original", "absent", "donor"}
        for row in reduction["comparative_rows"]
    )


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        ("position", "last_token_position"),
        ("truncation", "prompt_token_custody"),
        ("nonfinite", "finite_probabilities"),
        ("mapping", "semantic_order_mapping"),
        ("generation", "zero_generation"),
        ("missing", "schedule_identity"),
    ],
)
def test_native_reducer_fails_closed(mutation: str, expected_error: str) -> None:
    """REQ-VERIFY-7535 rejects custody and numerical defects before science."""

    panel = pilot.select_development_panel(
        _candidates(), excluded_component_hashes=set(), token_count=_token_count
    )
    schedule = pilot.build_forward_schedule(panel["groups"], token_count=_token_count)
    rows = [_complete_row(request, index) for index, request in enumerate(schedule)]
    if mutation == "position":
        rows[0]["actual_last_evaluated_position"] = 0
    elif mutation == "truncation":
        rows[0]["prompt_token_ids"] = rows[0]["prompt_token_ids"][:-1]
    elif mutation == "nonfinite":
        order = rows[0]["option_order"]
        rows[0]["probabilities_by_option_id"][order[0]] = math.nan
    elif mutation == "mapping":
        rows[0]["label_to_option_id"] = {" A": "supported", " B": "supported"}
    elif mutation == "generation":
        rows[0]["generated_tokens"] = 1
    else:
        rows.pop()
    reduction = pilot.reduce_native_rows(rows, schedule)
    assert reduction["passed"] is False
    assert expected_error in reduction["failed_checks"]


def test_forecasts_are_independent_and_bounded() -> None:
    """REQ-VERIFY-7535; SCENARIO-VERIFY-7535-FORECAST."""

    forecast = pilot.forecast_captures(
        load_seconds=100.0,
        forward_seconds=[0.5, 1.0, 1.5, 2.0],
        fit_checkpoint_seconds=20.0,
        eval_checkpoint_seconds=900.0,
    )
    assert forecast["formula"] == "load + 1440*p95_forward + checkpoint + 600"
    assert forecast["fit"]["planned_forwards"] == 1440
    assert forecast["evaluation"]["planned_forwards"] == 1440
    assert forecast["fit"]["forecast_seconds"] < forecast["evaluation"]["forecast_seconds"]
    assert forecast["fit_capture_feasible_score"] == 1
    assert forecast["eval_capture_feasible_score"] == 0

    with pytest.raises(pilot.NativePilotError, match="forward_durations_missing"):
        pilot.forecast_captures(
            load_seconds=1.0,
            forward_seconds=[],
            fit_checkpoint_seconds=0.0,
            eval_checkpoint_seconds=0.0,
        )


def test_invocation_reduction_balances_failures_without_generation() -> None:
    """REQ-VERIFY-7535 requires current call accounting."""

    events = [
        {"operation": "model_load", "state": "attempted"},
        {"operation": "model_load", "state": "completed"},
        {"operation": "forward", "state": "attempted"},
        {"operation": "forward", "state": "failed"},
    ]
    counts = pilot.reduce_invocation_events(events)
    assert counts["model_loads_attempted"] == 1
    assert counts["model_loads_completed"] == 1
    assert counts["forward_calls_attempted"] == 1
    assert counts["failures"] == 1
    assert counts["generation_calls_attempted"] == 0
    assert counts["in_flight"] == 0


def test_gate_summary_preserves_exact_failed_operand() -> None:
    """REQ-VERIFY-7535; SCENARIO-VERIFY-7535-BLOCKED."""

    gates = [
        pilot.gate(
            "upstream_ready",
            "validity",
            1,
            None,
            "==",
            False,
            "Missing upstream evidence must block measurement.",
            upstream="results/upstream.json",
            field="ready",
        )
    ]
    summary = pilot.gate_check_summary(gates)
    assert summary["failed_count"] == 1
    assert summary["first_failure"]["upstream"] == "results/upstream.json"
    assert summary["first_failure"]["field"] == "ready"
    assert summary["first_failure"]["expected"] == 1
    assert summary["first_failure"]["observed"] is None


def test_fixture_artifact_is_cold_valid_and_checksum_bound() -> None:
    """REQ-VERIFY-7535; SCENARIO-VERIFY-7535-E2E."""

    artifact = pilot.build_artifact_for_test()
    assert pilot.validate_artifact(artifact, require_validation=False) == []
    assert artifact["native_tool_ready_score"] == 1
    assert artifact["fit_capture_feasible_score"] in (0, 1)
    assert artifact["eval_capture_feasible_score"] in (0, 1)
    assert artifact["readout_kind"] == "option_logits"
    assert artifact["inference_substrate"] == "live_llm_embedding_extraction"
    assert artifact["inference_substrate_class"] == "model_load_no_generation"
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["positive_claim"] is False

    changed = deepcopy(artifact)
    changed["pilot_cost_rows"][0]["measured_seconds"] += 1.0
    assert "reproducibility_checksum_mismatch" in pilot.validate_artifact(
        changed, require_validation=False
    )


def test_blocked_artifact_is_complete_and_never_fabricates_rows() -> None:
    """REQ-VERIFY-7535; SCENARIO-VERIFY-7535-BLOCKED."""

    failed = pilot.gate(
        "owned_gpu_available",
        "external_precondition",
        True,
        False,
        "is",
        False,
        "Only an owned CUDA device can support native evidence.",
        upstream="nvidia-smi",
        field="admissible_gpu",
    )
    artifact = pilot.build_blocked_artifact(
        failed_gate=failed,
        preconditions=[failed],
        source_hashes=[],
        duration_s=2.1,
        phase_spans=[],
    )
    assert pilot.validate_artifact(artifact, require_validation=False) == []
    assert artifact["honest_verdict"] == "complete_blocked_owned_gpu_available"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["inference_substrate_class"] == "blocked_no_run"
    assert artifact["planned_inference_substrate_class"] == "model_load_no_generation"
    assert artifact["rows"] == []
    assert artifact["sample_size_budget"]["unstarted"] == 72
    assert artifact["native_tool_ready_score"] == 0
    assert artifact["gate_check_summary"]["first_failure"]["field"] == "admissible_gpu"


def test_panel_filters_nontrain_incomplete_and_replaces_single_type_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-7535 keeps public selection deterministic and diverse."""

    candidates = _candidates()
    candidates.extend(
        [
            {**deepcopy(candidates[0]), "component_hash": "test-row", "official_split": "test"},
            {"component_hash": "incomplete", "official_split": "train"},
        ]
    )

    def rank(component_hash: object, stratum: int) -> str:
        row = next(
            (item for item in candidates if item.get("component_hash") == component_hash),
            {},
        )
        prefix = "0" if row.get("tool_type") == "grep" else "1"
        return f"{prefix}:{stratum:02d}:{component_hash}"

    monkeypatch.setattr(pilot, "_selection_rank", rank)
    panel = pilot.select_development_panel(
        candidates, excluded_component_hashes=set(), token_count=_token_count
    )
    assert len({row["tool_type"] for row in panel["groups"]}) >= 2


def test_panel_and_schedule_reject_missing_donor_count_and_overlength() -> None:
    """REQ-VERIFY-7535 fails closed instead of changing the fixed sample."""

    rows = _candidates()
    for index, row in enumerate(rows):
        row["tool_type"] = f"tool-{index}"
    with pytest.raises(pilot.NativePilotError, match="development_donor_missing"):
        pilot.select_development_panel(
            rows, excluded_component_hashes=set(), token_count=_token_count
        )

    panel = pilot.select_development_panel(
        _candidates(), excluded_component_hashes=set(), token_count=_token_count
    )
    with pytest.raises(pilot.NativePilotError, match="pilot_group_count"):
        pilot.build_forward_schedule(panel["groups"][:-1], token_count=_token_count)
    with pytest.raises(pilot.NativePilotError, match="prompt_overlength"):
        pilot.build_forward_schedule(panel["groups"], token_count=lambda _text: pilot.N_CTX + 1)


@pytest.mark.parametrize(
    "mutation",
    [
        "identity",
        "requested_position",
        "logits_shape",
        "probabilities_shape",
        "probability_sum",
        "probability_softmax",
        "state_reset",
        "failed_row",
        "bad_probability_value",
        "duplicate_index",
    ],
)
def test_additional_native_custody_mutations_fail(mutation: str) -> None:
    """REQ-VERIFY-7535 exercises every fail-closed custody operand."""

    panel = pilot.select_development_panel(
        _candidates(), excluded_component_hashes=set(), token_count=_token_count
    )
    schedule = pilot.build_forward_schedule(panel["groups"], token_count=_token_count)
    rows = [_complete_row(request, index) for index, request in enumerate(schedule)]
    if mutation == "identity":
        rows[0]["condition"] = "changed"
    elif mutation == "requested_position":
        rows[0]["requested_score_position"] = 0
    elif mutation == "logits_shape":
        rows[0]["full_logits_by_option_id"] = {}
    elif mutation == "probabilities_shape":
        rows[0]["probabilities_by_option_id"] = {}
    elif mutation == "probability_sum":
        order = rows[0]["option_order"]
        rows[0]["probabilities_by_option_id"] = {order[0]: 0.2, order[1]: 0.2}
    elif mutation == "probability_softmax":
        order = rows[0]["option_order"]
        rows[0]["probabilities_by_option_id"] = {order[0]: 0.5, order[1]: 0.5}
    elif mutation == "state_reset":
        rows[0]["state_reset"] = False
    elif mutation == "failed_row":
        rows[0]["disposition"] = "failed"
        rows[0]["error"] = "boom"
    elif mutation == "bad_probability_value":
        rows[0]["probabilities_by_option_id"]["contains_unsupported"] = "bad"
    else:
        rows[1]["schedule_index"] = rows[0]["schedule_index"]
    assert pilot.reduce_native_rows(rows, schedule)["passed"] is False


def test_single_duration_percentile_and_gate_path() -> None:
    """REQ-VERIFY-7535 retains exact forecast and failed-path operands."""

    assert (
        pilot.forecast_captures(
            load_seconds=1.0,
            forward_seconds=[1.0],
            fit_checkpoint_seconds=1.0,
            eval_checkpoint_seconds=1.0,
        )["fit"]["p95_forward_seconds"]
        == 1.0
    )
    row = pilot.gate(
        "x",
        "validity",
        True,
        False,
        "is",
        False,
        "principle",
        upstream="upstream",
        field="field",
        path="path/to/field",
    )
    assert row["path"] == "path/to/field"


def test_cold_validator_rejects_nonobjects_and_terminal_drift() -> None:
    """REQ-VERIFY-7535 cold replay rejects changed terminal declarations."""

    assert pilot.validate_artifact(None, require_validation=False) == ["artifact_not_object"]
    baseline = pilot.build_artifact_for_test()
    mutations = {
        "schema_mismatch": ("schema", "changed"),
        "verdict_class_invalid": ("verdict_class", "unknown"),
        "honest_verdict_not_terminal": ("honest_verdict", "null"),
        "acceptance_gates_invalid": ("acceptance_gate_results", None),
        "native_tool_ready_score_not_bare_binary": ("native_tool_ready_score", True),
        "invocation_evidence_invalid": ("current_invocation_events", None),
        "field_principles_missing": ("field_principles", {}),
    }
    for expected, (field, value) in mutations.items():
        artifact = deepcopy(baseline)
        artifact[field] = value
        artifact["reproducibility_checksum"] = pilot.artifact_checksum(artifact)
        assert expected in pilot.validate_artifact(artifact, require_validation=False)


def test_cold_validator_rejects_gate_counter_and_blocked_drift() -> None:
    """REQ-VERIFY-7535 preserves reductions and blocks fabricated work."""

    baseline = pilot.build_artifact_for_test()
    gate_drift = deepcopy(baseline)
    gate_drift["gate_check_summary"] = {}
    gate_drift["reproducibility_checksum"] = pilot.artifact_checksum(gate_drift)
    assert "gate_check_summary_mismatch" in pilot.validate_artifact(
        gate_drift, require_validation=False
    )

    count_drift = deepcopy(baseline)
    count_drift["invocation_counts"]["forward_calls_completed"] -= 1
    count_drift["reproducibility_checksum"] = pilot.artifact_checksum(count_drift)
    assert "invocation_counts_mismatch" in pilot.validate_artifact(
        count_drift, require_validation=False
    )
    assert "required_validation_failed" in pilot.validate_artifact(baseline)
    assert pilot._validation_passed(None) is False

    failed = pilot.gate(
        "gpu",
        "external_precondition",
        True,
        False,
        "is",
        False,
        "principle",
        upstream="nvidia-smi",
        field="gpu",
    )
    blocked = pilot.build_blocked_artifact(
        failed_gate=failed,
        preconditions=[failed],
        source_hashes=[],
        duration_s=2.0,
        phase_spans=[],
    )
    blocked["rows"] = [{"fabricated": True}]
    blocked["native_tool_ready_score"] = 1
    blocked["reproducibility_checksum"] = pilot.artifact_checksum(blocked)
    errors = pilot.validate_artifact(blocked, require_validation=False)
    assert "blocked_artifact_fabricated_work" in errors
    assert "blocked_artifact_ready" in errors
