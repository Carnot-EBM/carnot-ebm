"""Tests for REQ-VERIFY-7293 and SCENARIO-VERIFY-7293-*.

The model-shaped rows use only the public fixture during generation setup.
Private labels enter only the scorer call after all response rows exist.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from itertools import groupby

import pytest

from carnot import experiment_7291_v641_reuse_fixture as fixture
from carnot import experiment_7293_v641_reuse_measurement as mod


def _evaluation_bundle() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Return the frozen public groups and separate evaluation labels."""

    public, authority = fixture.build_fixture()
    groups = deepcopy(public["evaluation_groups"])
    labels = [row for row in authority["labels"] if row["split"] == "evaluation"]
    return groups, labels


def _completion_rows(schedule: list[dict[str, object]]) -> list[dict[str, object]]:
    """Create complete response rows with measured model and CPU components."""

    rows: list[dict[str, object]] = []
    for sealed in schedule:
        call_type = str(sealed["call_type"])
        if call_type == "direct":
            parsed: dict[str, object] = {"decision": "a"}
            compiled = {"decision": "supported"}
        else:
            document = sealed["document"]
            assert isinstance(document, dict)
            parsed = fixture.extract_completion(document, call_type)
            compiled = fixture.pointer.compile_pointer_completion(document, parsed, call_type)
        rows.append(
            {
                "call_order": sealed["call_order"],
                "call_id": sealed["call_id"],
                "unit_id": sealed["unit_id"],
                "group_id": sealed["group_id"],
                "source_id": sealed["source_id"],
                "source_version": sealed["source_version"],
                "arm": sealed["comparison_arm"],
                "call_type": call_type,
                "draw": sealed.get("draw"),
                "parsed_completion": parsed,
                "compiled_completion": compiled,
                "transport_complete": True,
                "parse_valid": True,
                "usable": True,
                "terminal_state": "complete",
                "errors": [],
                "prompt_tokens": 20,
                "completion_tokens": 8,
                "latency_s": 1.0,
                "native_cache_receipt": {
                    "cache_policy": "cache_prompt_true",
                    "evidence_present": True,
                    "cached_tokens": 19,
                    "cache_n": 19,
                },
                "measured_processing_s": {
                    "query_extraction": 0.01 if call_type == "claim" else 0.0,
                    "source_compilation": 0.02 if call_type == "source" else 0.0,
                    "prefix_prefill": 0.03,
                    "lookup_or_invalidation": 0.004,
                    "verification": 0.005,
                    "synchronization": 0.006,
                },
            }
        )
    return rows


def test_scenario_verify_7293_schedule_freezes_complete_group_blocks() -> None:
    """SCENARIO-VERIFY-7293-SCHEDULE freezes 672 calls in fair group blocks."""

    groups, _labels = _evaluation_bundle()
    orders = mod.freeze_arm_orders(groups)
    schedule = mod.build_schedule(groups, orders)

    assert len(schedule) == 672
    assert Counter(row["comparison_arm"] for row in schedule) == {
        "warm_prefix_direct": 256,
        "fresh_verifier": 256,
        "versioned_reuse_verifier": 160,
    }
    assert mod.schedule_errors(schedule, groups, orders) == []
    assert len(set(tuple(value) for value in orders.values())) > 1
    assert "expected_decision" not in mod.canonical_json(schedule)
    assert all(row["output_token_budget"] == 128 for row in schedule)
    assert all(row["held_out_eligible"] is True for row in schedule)

    for group_index, group in enumerate(groups):
        block = schedule[group_index * 42 : (group_index + 1) * 42]
        assert {row["group_id"] for row in block} == {group["group_id"]}
        observed_order = [key for key, _rows in groupby(row["comparison_arm"] for row in block)]
        assert observed_order == orders[group["group_id"]]

    changed = deepcopy(schedule)
    changed[42], changed[43] = changed[43], changed[42]
    assert mod.schedule_errors(changed, groups, orders)


def test_scenario_verify_7293_source_failure_reaches_each_dependent_claim() -> None:
    """SCENARIO-VERIFY-7293-FAILURE keeps one source error on four reuse rows."""

    groups, labels = _evaluation_bundle()
    one_group = groups[:1]
    orders = mod.freeze_arm_orders(one_group)
    schedule = mod.build_schedule(one_group, orders, require_full_denominator=False)
    rows = _completion_rows(schedule)
    failed = next(
        row
        for row in rows
        if row["arm"] == "versioned_reuse_verifier"
        and row["call_type"] == "source"
        and row["source_version"] == 1
    )
    failed["usable"] = False
    failed["compiled_completion"] = {
        "outcome": "unknown",
        "relations": [],
        "errors": ["source_compile_failed"],
    }

    reduced = mod.reduce_measurement(
        one_group,
        schedule,
        rows,
        [row for row in labels if row["group_id"] == one_group[0]["group_id"]],
        model_initialization_s=9.0,
    )
    affected = [
        row
        for row in reduced["rows"]
        if row["arm"] == "versioned_reuse_verifier" and row["source_version"] == 1
    ]

    assert len(affected) == 4
    assert all(row["prediction"] == "unknown" for row in affected)
    assert all(row["abstention"] is True for row in affected)
    assert all("source_compile_failed" in row["error"] for row in affected)
    assert all(row["source_compilation_call_id"] == failed["call_id"] for row in affected)
    assert reduced["source_version_receipts"][0]["served_dependent_claims"] == 4
    assert reduced["source_version_receipts"][0]["compilation_usable"] is False


def test_scenario_verify_7293_cost_uses_elapsed_work_not_cached_tokens() -> None:
    """SCENARIO-VERIFY-7293-COST charges unique calls and symmetric initialization."""

    groups, labels = _evaluation_bundle()
    one_group = groups[:1]
    orders = mod.freeze_arm_orders(one_group)
    schedule = mod.build_schedule(one_group, orders, require_full_denominator=False)
    rows = _completion_rows(schedule)
    reduced = mod.reduce_measurement(
        one_group,
        schedule,
        rows,
        [row for row in labels if row["group_id"] == one_group[0]["group_id"]],
        model_initialization_s=9.0,
    )

    at_eight = [row for row in reduced["amortization_rows"] if row["claim_count"] == 8]
    by_arm = {row["arm"]: row for row in at_eight}
    assert by_arm["warm_prefix_direct"]["unique_generation_calls"] == 16
    assert by_arm["fresh_verifier"]["unique_generation_calls"] == 16
    assert by_arm["versioned_reuse_verifier"]["unique_generation_calls"] == 10
    assert {row["model_initialization_allocation_s"] for row in at_eight} == {3.0}
    assert (
        by_arm["versioned_reuse_verifier"]["steady_total_s"]
        < by_arm["fresh_verifier"]["steady_total_s"]
    )
    assert all(row["cold_total_s"] == row["steady_total_s"] + 3.0 for row in at_eight)
    assert all("cached_tokens" not in row["measured_cost_components_s"] for row in at_eight)
    assert reduced["cost_accounting"]["cached_tokens_used_as_wall_time"] is False
    assert {row["claim_count"] for row in reduced["amortization_rows"]} == {1, 2, 4, 8}


def test_scenario_verify_7293_capture_and_terminal_classes_fail_closed() -> None:
    """SCENARIO-VERIFY-7293-CAPTURE separates blocked, full, and null outcomes."""

    no_calls = mod.classify_inference(mod.ZERO_INVOCATION_COUNTS)
    assert no_calls["inference_substrate_class"] == "blocked_no_run"

    load_only = deepcopy(mod.ZERO_INVOCATION_COUNTS)
    load_only["model_loads_attempted"] = 1
    assert mod.classify_inference(load_only)["inference_substrate_class"] == (
        "model_load_no_generation"
    )

    generated = deepcopy(load_only)
    generated["generation_calls_attempted"] = 1
    assert mod.classify_inference(generated) == {
        "model_invoked": True,
        "inference_substrate": "live_llm_inference",
        "inference_substrate_class": "model_full_generation",
        "inference_mode": "live_gpu",
    }

    gates = [
        mod.acceptance_row("capture_complete", 16, 16, True),
        mod.acceptance_row("semantic_parity", 0, 1, False),
    ]
    assert mod.capture_complete_score(gates) == 1
    assert mod.value_score(gates) == 0
    assert mod.classify_verdict(gates, verifier_is_oracle=True) == (
        "null",
        "complete_null_reuse_value_gate_failed",
    )

    checks = [
        mod.gate_row(
            "reuse_canary_ready",
            1,
            0,
            False,
            upstream="exp7292-reuse-canary",
            field="reuse_canary_ready_score",
        )
    ]
    assert mod.gate_summary(checks) == {
        "failed_check": "reuse_canary_ready",
        "upstream": "exp7292-reuse-canary",
        "field": "reuse_canary_ready_score",
        "expected_value": 1,
        "observed_value": 0,
    }


def test_req_verify_7293_defensive_boundaries_remain_explicit() -> None:
    """REQ-VERIFY-7293 rejects schedule drift and keeps empty-cost boundaries visible."""

    groups, _labels = _evaluation_bundle()
    one_group = groups[:1]
    orders = mod.freeze_arm_orders(one_group)
    schedule = mod.build_schedule(one_group, orders, require_full_denominator=False)

    assert mod.gate_summary([])["failed_check"] is None
    assert mod.artifact_checksum({"answer": 1, "duration_s": 2.0}) == mod.artifact_checksum(
        {"answer": 1, "duration_s": 9.0}
    )
    with pytest.raises(ValueError, match="evaluation_group_denominator"):
        mod.build_schedule(one_group, orders)
    with pytest.raises(ValueError, match="source_version_identity"):
        mod._source_for(one_group[0], 3)
    with pytest.raises(ValueError, match="comparison_arm"):
        mod._append_arm_block([], one_group[0], "not_an_arm")

    bad_claims = deepcopy(one_group)
    bad_claims[0]["claims"] = bad_claims[0]["claims"][:-1]
    with pytest.raises(ValueError, match="evaluation_claim_denominator"):
        mod.build_schedule(bad_claims, orders, require_full_denominator=False)
    with pytest.raises(ValueError, match="group_arm_order"):
        mod.build_schedule(one_group, {}, require_full_denominator=False)
    bad_revision = deepcopy(one_group[0])
    bad_revision["claims"][0]["source_version"] = 2
    with pytest.raises(ValueError, match="revision_claim_denominator"):
        mod._append_arm_block([], bad_revision, "versioned_reuse_verifier")
    assert mod.schedule_errors(schedule, one_group, {})[0].startswith("schedule_rebuild:")

    short = schedule[:-1]
    short_errors = mod.schedule_errors(short, one_group, orders)
    assert {"call_denominator", "rebuilt_denominator", "arm_call_denominators"}.issubset(
        short_errors
    )
    leaked = deepcopy(schedule)
    leaked[0]["expected_decision"] = "supported"
    assert "authority_leakage" in mod.schedule_errors(leaked, one_group, orders)

    assert mod._compiled(None)["errors"] == ["missing_call"]
    assert mod._row_errors(None, "missing") == ["missing"]
    error_row = {
        "compiled_completion": {"outcome": "unknown", "relations": []},
        "errors": [],
        "model_response_error": "transport_failed",
    }
    assert mod._row_errors(error_row, "fallback") == ["transport_failed"]
    assert mod._decision({}, error_row | {"usable": True}, None)["errors"] == [
        "claim_call_unusable"
    ]
    assert mod._call_cost(None)["failed_calls"] == 1
    missing_cost, missing_total = mod._aggregate_unique_cost(
        ["missing"],
        {},
        extra_lookup_s=0.0,
        extra_verification_s=0.0,
        extra_synchronization_s=0.0,
    )
    assert missing_cost["failed_or_censored_generation_calls"] == 1
    assert missing_total == 0.0
    assert mod._latency_distribution([])["count"] == 0

    passing = [mod.acceptance_row(name, True, True, True) for name in mod.GATE_PRINCIPLES]
    assert mod.classify_verdict(passing, verifier_is_oracle=True)[0] == "circular_positive"
    assert mod.classify_verdict(passing, verifier_is_oracle=False)[0] == "positive"


def test_scenario_verify_7293_raw_replay_projects_measurement_metadata() -> None:
    """SCENARIO-VERIFY-7293-CAPTURE replays native bytes before cost annotations."""

    retained = {
        "call_id": "call-1",
        "transport_complete": True,
        "attempted": True,
        "censored": False,
        "censoring_reason": None,
        "measured_processing_s": {"source_compilation": 0.125},
        "row_sha256": "measurement-row-hash",
    }

    projected = mod._canary_replay_projection(retained)

    assert projected == {
        "call_id": "call-1",
        "transport_complete": True,
        "attempted": True,
        "row_sha256": mod.heldout.capture._row_hash(
            {"call_id": "call-1", "transport_complete": True, "attempted": True}
        ),
    }
    assert retained["attempted"] is True
    assert retained["measured_processing_s"] == {"source_compilation": 0.125}
