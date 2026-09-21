"""Tests for REQ-VERIFY-7492 and SCENARIO-VERIFY-7492-*.

The tests use analytic rows and frozen Exp7491 protocol bytes. They do not load
the Qwen model or claim semantic benefit.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from carnot import experiment_7492_v656_window_pilot as exp


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_req_verify_7492_spec_precedes_implementation() -> None:
    """REQ-VERIFY-7492 and every required scenario exist before implementation."""

    text = exp.SPEC_PATH.read_text(encoding="utf-8")
    assert "REQ-VERIFY-7492" in text
    for suffix in ("PREREQUISITE", "SCHEDULE", "TRANSPORT", "ORDER", "FORECAST", "E2E"):
        assert f"SCENARIO-VERIFY-7492-{suffix}" in text


def test_scenario_verify_7492_prerequisite_requires_exact_structured_gate() -> None:
    """SCENARIO-VERIFY-7492-PREREQUISITE rejects changed producer flags."""

    value = json.loads(exp.UPSTREAM_PATH.read_text(encoding="utf-8"))
    checks = exp.reduce_upstream_gate(value, validator_errors=[])
    assert checks["passed"] is True
    assert all(row["passed"] and row["principle"] for row in checks["checks"])

    for field, changed in (
        ("schema", "wrong"),
        ("terminal_status", "partial"),
        ("honest_verdict", "blocked_changed"),
        ("verdict_class", "blocked"),
        ("flagged_adversarial", True),
        ("window_protocol_ready_score", 0),
    ):
        altered = deepcopy(value)
        altered[field] = changed
        assert exp.reduce_upstream_gate(altered, validator_errors=[])["passed"] is False
    assert exp.reduce_upstream_gate(value, validator_errors=["raw_hash_changed"])["passed"] is False


def test_scenario_verify_7492_schedule_uses_eight_fixed_training_groups() -> None:
    """SCENARIO-VERIFY-7492-SCHEDULE preserves whole prompts and both orders."""

    raw = exp.REPO_ROOT / exp.UPSTREAM_RAW_DIR
    schedule = exp.build_pilot_schedule(
        _load_jsonl(raw / "predictors.jsonl"),
        _load_jsonl(raw / "groups.jsonl"),
        _load_jsonl(raw / "windows.jsonl"),
        _load_jsonl(raw / "requests.jsonl"),
    )
    assert {row["group_id"] for row in schedule} == set(exp.PILOT_GROUP_IDS)
    assert len({row["group_id"] for row in schedule}) == 8
    assert len(schedule) <= exp.FORWARD_BUDGET == 72
    assert all(row["role"] == "training" for row in schedule)
    assert all(row["prompt_sha256"] == row["sealed_prompt_sha256"] for row in schedule)
    for group_id in exp.PILOT_GROUP_IDS:
        rows = [row for row in schedule if row["group_id"] == group_id]
        assert {tuple(row["option_order"]) for row in rows} == {
            exp.OPTION_IDS,
            tuple(reversed(exp.OPTION_IDS)),
        }
        assert sum(row["arm"] == "whole_response" for row in rows) == 2
    spans = exp.schedule_span_summary(schedule)
    assert spans["group_count"] == 8
    assert spans["response_byte_max"] > spans["response_byte_min"]
    assert spans["sentence_count_max"] > spans["sentence_count_min"]


def test_schedule_rejects_missing_or_changed_sealed_requests() -> None:
    """REQ-VERIFY-7492 fails closed before model work when prompt identity drifts."""

    raw = exp.REPO_ROOT / exp.UPSTREAM_RAW_DIR
    predictors = _load_jsonl(raw / "predictors.jsonl")
    groups = _load_jsonl(raw / "groups.jsonl")
    windows = _load_jsonl(raw / "windows.jsonl")
    requests = _load_jsonl(raw / "requests.jsonl")
    missing = deepcopy(requests)
    missing.pop(
        next(
            index for index, row in enumerate(missing) if row["group_id"] == exp.PILOT_GROUP_IDS[0]
        )
    )
    with pytest.raises(exp.WindowPilotError, match="sealed_request_missing"):
        exp.build_pilot_schedule(predictors, groups, windows, missing)
    changed = deepcopy(requests)
    target = next(row for row in changed if row["group_id"] == exp.PILOT_GROUP_IDS[0])
    target["prompt_sha256"] = "sha256:changed"
    with pytest.raises(exp.WindowPilotError, match="sealed_prompt_hash_mismatch"):
        exp.build_pilot_schedule(predictors, groups, windows, changed)


def test_schedule_rejects_group_window_and_budget_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-VERIFY-7492 rejects changed frozen groups, windows, and call ceilings."""

    raw = exp.REPO_ROOT / exp.UPSTREAM_RAW_DIR
    predictors = _load_jsonl(raw / "predictors.jsonl")
    groups = _load_jsonl(raw / "groups.jsonl")
    windows = _load_jsonl(raw / "windows.jsonl")
    requests = _load_jsonl(raw / "requests.jsonl")
    missing = [row for row in predictors if row["group_id"] != exp.PILOT_GROUP_IDS[0]]
    with pytest.raises(exp.WindowPilotError, match="frozen_group_missing"):
        exp.build_pilot_schedule(missing, groups, windows, requests)
    ineligible = deepcopy(groups)
    next(row for row in ineligible if row["group_id"] == exp.PILOT_GROUP_IDS[0])["eligible"] = False
    with pytest.raises(exp.WindowPilotError, match="frozen_group_not_eligible_training"):
        exp.build_pilot_schedule(predictors, ineligible, windows, requests)
    changed_windows = deepcopy(windows)
    next(row for row in changed_windows if row["group_id"] == exp.PILOT_GROUP_IDS[0])[
        "byte_end"
    ] += 1
    with pytest.raises(exp.WindowPilotError, match="sealed_windows_mismatch"):
        exp.build_pilot_schedule(predictors, groups, changed_windows, requests)
    monkeypatch.setattr(exp, "FORWARD_BUDGET", 1)
    with pytest.raises(exp.WindowPilotError, match="forward_budget_exceeded"):
        exp.build_pilot_schedule(predictors, groups, windows, requests)


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    (
        ("label_swap", "label_mapping_valid"),
        ("token_position", "final_position_valid"),
        ("state_reuse", "fresh_kv_valid"),
        ("prompt_hash", "sealed_prompt_identity_valid"),
        ("generation", "zero_generation_valid"),
        ("nonfinite", "finite_logits_valid"),
    ),
)
def test_scenario_verify_7492_transport_rejects_native_receipt_mutations(
    mutation: str, failed_check: str
) -> None:
    """SCENARIO-VERIFY-7492-TRANSPORT rejects each non-semantic defect."""

    schedule = exp.controlled_schedule_fixture()
    rows = exp.controlled_call_fixture(schedule)
    assert exp.reduce_pilot_calls(rows, schedule)["passed"] is True
    reduction = exp.reduce_pilot_calls(exp.mutate_call_fixture(rows, mutation), schedule)
    assert reduction[failed_check] is False
    assert reduction["passed"] is False


def test_scenario_verify_7492_order_reports_sensitivity_without_gating() -> None:
    """SCENARIO-VERIFY-7492-ORDER keeps actual order disagreement observational."""

    schedule = exp.controlled_schedule_fixture()
    rows = exp.controlled_call_fixture(schedule)
    rows[1]["raw_logits_by_option_id"] = {
        "supported": 0.0,
        "contains_unsupported": 3.8918202981106256,
    }
    rows[1]["raw_logits_by_display_label"] = {
        label: rows[1]["raw_logits_by_option_id"][option_id]
        for label, option_id in rows[1]["label_to_option_id"].items()
    }
    rows[1]["probabilities_by_option_id"] = {
        "supported": 0.02,
        "contains_unsupported": 0.98,
    }
    rows[1]["log_odds_contains_unsupported"] = 3.8918202981106256
    reduction = exp.reduce_pilot_calls(rows, schedule)
    assert reduction["passed"] is True
    assert reduction["order_equality_is_gate"] is False
    group_rows = exp.reduce_group_rows(rows)
    assert group_rows[0]["maximum_order_probability_delta"] > 0.7


def test_analytic_mapping_fixture_is_separate_from_semantic_rows() -> None:
    """REQ-VERIFY-7492 uses known mapping controls without calling them benefit."""

    fixture = exp.analytic_token_mapping_fixture()
    reduced = exp.reduce_analytic_fixture(fixture)
    assert reduced == {
        "passed": True,
        "stable_mapping_valid": True,
        "expected_probabilities_valid": True,
        "semantic_observation": False,
    }
    altered = deepcopy(fixture)
    altered[1]["label_to_option_id"] = altered[0]["label_to_option_id"]
    assert exp.reduce_analytic_fixture(altered)["passed"] is False


def test_scenario_verify_7492_forecast_uses_p90_tokens_load_and_fixed_cap() -> None:
    """SCENARIO-VERIFY-7492-FORECAST applies the conservative measured estimate."""

    samples = [
        {"prompt_token_count": 100, "prefill_s": 1.0},
        {"prompt_token_count": 200, "prefill_s": 4.0},
        {"prompt_token_count": 400, "prefill_s": 12.0},
    ]
    manifests = {
        "fit_capture": [100, 200],
        "evaluation_capture": [400],
    }
    rows = exp.build_capture_budget_forecasts(samples, load_s=5.0, manifests=manifests)
    assert [row["capture"] for row in rows] == ["fit_capture", "evaluation_capture"]
    assert all(row["p90_seconds_per_token"] == 0.03 for row in rows)
    assert rows[0]["forecast_s"] == pytest.approx(5.0 + 300 * 0.03 * 1.25)
    assert all(row["uncertainty"]["sample_size"] == 3 for row in rows)
    assert all(row["hard_cap_s"] == 3300.0 and row["feasible"] for row in rows)

    blocked = exp.build_capture_budget_forecasts(
        [{"prompt_token_count": 1, "prefill_s": 10.0}],
        load_s=100.0,
        manifests={"fit_capture": [300], "evaluation_capture": [400]},
    )
    assert all(row["feasible"] is False for row in blocked)
    assert [row["unstarted_calls"] for row in blocked] == [1, 1]
    with pytest.raises(ValueError, match="positive_prefill_samples_required"):
        exp.build_capture_budget_forecasts([], load_s=1.0, manifests=manifests)


def test_acceptance_gates_keep_validity_readiness_and_benefit_independent() -> None:
    """REQ-VERIFY-7492 annotates every gate with the prevented failure."""

    gates = exp.build_acceptance_gates(
        validity_passed=True,
        transport_passed=True,
        forecasts_passed=True,
        validation_passed=True,
    )
    assert all(row["passed"] and row["principle"] for row in gates)
    assert {row["category"] for row in gates} == {"validity", "readiness", "benefit"}
    failed = exp.build_acceptance_gates(
        validity_passed=True,
        transport_passed=True,
        forecasts_passed=False,
        validation_passed=True,
    )
    assert exp.gate_check_summary(failed)["first_failure"]["check"] == "capture_forecasts_fit"


def test_artifact_fixture_cold_reduces_scores_and_every_field_principle(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7492-E2E recomputes raw rows, forecasts, and scores."""

    artifact = exp.build_artifact_for_test(tmp_path)
    assert exp.validate_artifact(artifact, root=tmp_path, require_validation=False) == []
    assert artifact["MODEL_SPECS"] == [exp.MODEL_HF_ID]
    assert artifact["model_specs"] == artifact["MODEL_SPECS"]
    assert artifact["inference_substrate_class"] == "model_load_no_generation"
    assert artifact["window_native_ready_score"] == 1
    assert artifact["pilot_complete_score"] == 1
    assert artifact["scored_runtime_parity_score"] == 0
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"].startswith("complete_null")
    assert set(artifact["field_principles"]) == set(artifact)
    assert all("prevent" in text for text in artifact["field_principles"].values())

    altered = deepcopy(artifact)
    altered["pilot_call_rows"][0]["actual_last_evaluated_position"] += 1
    assert "pilot_call_reduction_mismatch" in exp.validate_artifact(
        altered, root=tmp_path, require_validation=False
    )


def test_pilot_complete_survives_measured_infeasible_forecast(tmp_path: Path) -> None:
    """REQ-VERIFY-7492 keeps completion separate from later capture readiness."""

    artifact = exp.build_artifact_for_test(tmp_path, feasible=False)
    assert artifact["window_native_ready_score"] == 0
    assert artifact["pilot_complete_score"] == 1
    assert artifact["verdict_class"] == "null"
    assert exp.validate_artifact(artifact, root=tmp_path, require_validation=False) == []


def test_invalid_fixture_builds_an_explicit_disqualified_terminal_record(tmp_path: Path) -> None:
    """REQ-VERIFY-7492 does not turn invalid identity evidence into a null."""

    artifact = exp.build_artifact_for_test(tmp_path, validity=False)
    assert artifact["verdict_class"] == "disqualified"
    assert artifact["honest_verdict"].startswith("complete_disqualified")
    assert artifact["window_native_ready_score"] == 0
    assert artifact["pilot_complete_score"] == 0
    assert exp.validate_artifact(artifact, root=tmp_path, require_validation=False) == []


def test_validation_manifest_is_scoped_and_full_suite_is_absent() -> None:
    """SCENARIO-VERIFY-7492-E2E freezes only the three affected files."""

    manifest = exp.VALIDATION_MANIFEST
    assert manifest.test_paths == (exp.TEST_PATH.as_posix(),)
    assert manifest.changed_modules == (exp.MODULE_PATH.as_posix(),)
    assert manifest.static_paths == (exp.WRAPPER_PATH.as_posix(),)
    assert all(path not in {"tests", "tests/python"} for path in manifest.test_paths)


def test_validator_fails_closed_on_identity_accounting_shard_and_receipts(tmp_path: Path) -> None:
    """REQ-VERIFY-7492 names each invalid evidence boundary."""

    cases: list[tuple[str, dict[str, object]]] = []
    for expected, mutate in (
        ("identity_mismatch:schema", lambda value: value.__setitem__("schema", "wrong")),
        (
            "invocation_counts_mismatch",
            lambda value: value["invocation_counts"]["forward_calls"].__setitem__("completed", 0),
        ),
        (
            "field_principles_incomplete",
            lambda value: value.__setitem__("field_principles", {}),
        ),
        (
            "gate_summary_mismatch",
            lambda value: value.__setitem__("gate_check_summary", {}),
        ),
        (
            "window_native_ready_score_mismatch",
            lambda value: value.__setitem__("window_native_ready_score", 0),
        ),
        (
            "pilot_complete_score_mismatch",
            lambda value: value.__setitem__("pilot_complete_score", 0),
        ),
    ):
        value = exp.build_artifact_for_test(tmp_path)
        mutate(value)
        cases.append((expected, value))
    for expected, value in cases:
        assert expected in exp.validate_artifact(value, root=tmp_path, require_validation=False)

    shard = exp.build_artifact_for_test(tmp_path)
    (tmp_path / shard["raw_shards"]["pilot_call_rows"]["path"]).write_text("changed\n")
    assert "raw_call_shard_invalid" in exp.validate_artifact(
        shard, root=tmp_path, require_validation=False
    )
    no_receipts = exp.build_artifact_for_test(tmp_path)
    assert "required_validation_failed" in exp.validate_artifact(
        no_receipts, root=tmp_path, require_validation=True
    )


def test_defensive_reducers_reject_unknown_or_malformed_inputs() -> None:
    """REQ-VERIFY-7492 closes malformed fixture, event, and artifact paths."""

    rows = exp.controlled_call_fixture(exp.controlled_schedule_fixture())
    with pytest.raises(ValueError, match="unknown_mutation"):
        exp.mutate_call_fixture(rows, "unknown")
    assert exp.validate_artifact([], root=exp.REPO_ROOT, require_validation=False) == [
        "artifact_not_object"
    ]
    counts = exp.reduce_invocation_events(exp.fixture_invocation_events(forwards=4))
    assert exp.invocation_counts_balanced(counts) is True
    counts["generation_calls"]["attempted"] = 1
    assert exp.invocation_counts_balanced(counts) is False
    assert exp._validation_names_passed({}, ("one",)) is False

    malformed_fixture = exp.analytic_token_mapping_fixture()
    malformed_fixture[0]["raw_logits_by_display_label"] = None
    assert exp.reduce_analytic_fixture(malformed_fixture)["passed"] is False
    malformed_rows = exp.controlled_call_fixture(exp.controlled_schedule_fixture())
    malformed_rows[0]["raw_logits_by_option_id"] = None
    malformed_rows[0]["probabilities_by_option_id"] = None
    reduction = exp.reduce_pilot_calls(malformed_rows, exp.controlled_schedule_fixture())
    assert reduction["finite_logits_valid"] is False
    assert reduction["normalization_valid"] is False
    malformed_rows = exp.controlled_call_fixture(exp.controlled_schedule_fixture())
    malformed_rows[0]["disposition"] = "failed"
    assert exp.reduce_group_rows(malformed_rows)[0]["paired_arms"] == 1


def test_raw_shard_size_guard_rejects_oversized_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-VERIFY-7492 keeps each committed sidecar below 20 MiB."""

    monkeypatch.setattr(exp, "_jsonl_bytes", lambda rows: b"x" * (20 * 1024 * 1024))
    with pytest.raises(exp.WindowPilotError, match="raw_call_shard_exceeds_20_mib"):
        exp._seal_call_rows(tmp_path, Path("too-large.jsonl"), [])


def test_validator_defensive_shapes_cover_raw_and_reduction_failures(tmp_path: Path) -> None:
    """REQ-VERIFY-7492 fails closed for malformed terminal evidence shapes."""

    no_receipt = exp.build_artifact_for_test(tmp_path)
    no_receipt["raw_shards"] = {}
    assert "raw_call_shard_invalid" in exp.validate_artifact(
        no_receipt, root=tmp_path, require_validation=False
    )

    bad_json = exp.build_artifact_for_test(tmp_path)
    shard_path = tmp_path / bad_json["raw_shards"]["pilot_call_rows"]["path"]
    shard_path.write_text("{\n", encoding="utf-8")
    bad_json["raw_shards"]["pilot_call_rows"]["sha256"] = (
        "sha256:" + hashlib.sha256(b"{\n").hexdigest()
    )
    assert "raw_call_shard_invalid" in exp.validate_artifact(
        bad_json, root=tmp_path, require_validation=False
    )

    cases: list[tuple[str, dict[str, object]]] = []
    missing_rows = exp.build_artifact_for_test(tmp_path)
    missing_rows["pilot_schedule"] = None
    cases.append(("pilot_rows_invalid", missing_rows))
    bad_group_reduction = exp.build_artifact_for_test(tmp_path)
    bad_group_reduction["rows"] = []
    cases.append(("group_reduction_mismatch", bad_group_reduction))
    missing_events = exp.build_artifact_for_test(tmp_path)
    missing_events["current_invocation_events"] = None
    cases.append(("invocation_evidence_invalid", missing_events))
    bad_forecast_value = exp.build_artifact_for_test(tmp_path)
    bad_forecast_value["duration_breakdown_s"]["model_load"] = "invalid"
    cases.append(("capture_forecast_mismatch", bad_forecast_value))
    bad_forecast_shape = exp.build_artifact_for_test(tmp_path)
    bad_forecast_shape["capture_request_lengths"] = None
    cases.append(("capture_forecast_inputs_invalid", bad_forecast_shape))
    bad_gates = exp.build_artifact_for_test(tmp_path)
    bad_gates["acceptance_gate_results"] = None
    cases.append(("acceptance_gates_invalid", bad_gates))
    bad_principle = exp.build_artifact_for_test(tmp_path)
    first_key = next(iter(bad_principle["field_principles"]))
    bad_principle["field_principles"][first_key] = "missing failure language"
    cases.append(("field_principles_missing_failure_mode", bad_principle))
    for expected, value in cases:
        assert expected in exp.validate_artifact(value, root=tmp_path, require_validation=False)
