"""Tests for the V643 held-out batch measurement.

Spec refs: REQ-VERIFY-7321 and SCENARIO-VERIFY-7321-*.
"""

from __future__ import annotations

import base64
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7306_v642_batch_fixture as fixture
from carnot import experiment_7317_v643_batch_harness as harness
from carnot import experiment_7320_v643_batch_canary as canary
from carnot import experiment_7321_v643_batch_measurement as measurement
from carnot.reporting import experiment_7303_validation_scope as scoped


ROOT = Path(__file__).resolve().parents[2]


def _fixture() -> tuple[dict[str, object], dict[str, object]]:
    """Use the frozen builder while keeping public and private values separate."""

    return fixture.build_fixture()


def _qualified(path: str) -> dict[str, object]:
    """Load one current qualified producer for exact dependency tests."""

    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _response_for(sealed: dict[str, object]) -> dict[str, object]:
    """Create valid native-shaped bytes from the deterministic transport."""

    request = sealed["fixture_request"]
    response = fixture.CpuFakeTransport().call(request)
    content = canary.model_content_from_fixture_response(sealed, response)
    body = {
        "choices": [{"finish_reason": "stop", "message": {"content": content}}],
        "usage": {"prompt_tokens": 31, "completion_tokens": 17},
        "timings": {"prompt_ms": 2.0, "predicted_ms": 8.0},
    }
    request_payload = canary.request_payload(sealed)
    return {
        "raw_request": request_payload,
        "raw_request_bytes_b64": base64.b64encode(
            canary.canonical_json(request_payload).encode("utf-8")
        ).decode("ascii"),
        "raw_response": body,
        "raw_response_bytes_b64": base64.b64encode(
            canary.canonical_json(body).encode("utf-8")
        ).decode("ascii"),
        "raw_completion": content,
        "prompt_tokens": 31,
        "completion_tokens": 17,
        "finish_reason": "stop",
        "latency_s": 0.01,
        "error": None,
    }


def _valid_call_rows(schedule: list[dict[str, object]]) -> list[dict[str, object]]:
    """Build complete call receipts without invoking a model in unit tests."""

    resource = {
        "server_pid": 123,
        "server_pid_start_ticks": 456,
        "gpu_uuid": "GPU-test",
        "lease_id": "lease-test",
        "cuda_offload_confirmed": True,
        "gpu_sample_sha256": "sha256:" + "1" * 64,
    }
    return [canary.build_per_call_row(row, _response_for(row), resource) for row in schedule]


def _passing_validation() -> dict[str, object]:
    """Return complete named receipts for artifact assembly unit tests."""

    names = (
        *scoped.REQUIRED_CHECK_NAMES,
        *measurement.TERMINAL_CHECK_NAMES,
        "private_evaluator",
    )
    receipts = [
        {
            "name": name,
            "command": f"fixture:{name}",
            "command_argv": [name],
            "scope": "explicit_test_fixture",
            "exit_code": 0,
            "duration_s": 0.001,
            "log_path": f"/tmp/{name}.log",
            "log_sha256": "sha256:" + "a" * 64,
            "passed": True,
            "timed_out": False,
        }
        for name in names
    ]
    return {
        "validation_receipts": receipts,
        "required_checks_passed": True,
        "missing_required_commands": [],
        "failed_required_commands": [],
        "duplicate_required_commands": [],
        "repository_health": scoped.build_repository_health([]),
        "validation_entrypoint_receipt": {
            "runner": harness.SCOPED_RUNNER,
            "called": True,
            "test_paths": [measurement.TEST_PATH.as_posix()],
            "changed_modules": [measurement.MODULE_PATH.as_posix()],
            "static_paths": [measurement.WRAPPER_PATH.as_posix()],
            "legacy_launcher_called": False,
            "repository_wide_target_present": False,
        },
    }


def test_scenario_verify_7321_dependency_preserves_exact_failed_value() -> None:
    """SCENARIO-VERIFY-7321-DEPENDENCY rejects failed classes and quarantine."""

    upstream_harness = _qualified("results/experiment_7317_v643_batch_harness.json")
    upstream_canary = _qualified("results/experiment_7320_v643_batch_canary.json")
    assert all(
        row["passed"] for row in measurement.dependency_gate_rows(upstream_harness, upstream_canary)
    )

    for field, value, expected_check in (
        ("verdict_class", "disqualified", "exp7317_terminal_class"),
        ("verdict_class", "blocked", "exp7317_terminal_class"),
        ("verdict_class", "partial", "exp7317_terminal_class"),
        ("flagged_adversarial", True, "exp7317_quarantine"),
    ):
        changed = deepcopy(upstream_harness)
        changed[field] = value
        checks = measurement.dependency_gate_rows(changed, upstream_canary)
        summary = measurement.gate_check_summary(checks)
        assert summary["failed_check"] == expected_check
        assert summary["observed_value"] == value
        assert summary["upstream"] == harness.EXPERIMENT_ID

    changed_canary = deepcopy(upstream_canary)
    changed_canary["batch_canary_ready_score"] = 0
    summary = measurement.gate_check_summary(
        measurement.dependency_gate_rows(upstream_harness, changed_canary)
    )
    assert summary == {
        "failed_check": "exp7320_ready_score",
        "upstream": canary.EXPERIMENT_ID,
        "field": "batch_canary_ready_score",
        "expected_value": 1,
        "observed_value": 0,
    }
    assert (
        measurement.gate_check_summary(measurement.dependency_gate_rows(None, None))[
            "observed_value"
        ]
        == "missing_artifact"
    )
    assert measurement.gate_check_summary([])["failed_check"] is None


def test_scenario_verify_7321_schedule_has_frozen_denominators_and_orders() -> None:
    """SCENARIO-VERIFY-7321-SCHEDULE preserves all held-out calls and rotations."""

    public, _labels = _fixture()
    schedule = measurement.build_schedule(public)

    assert len(schedule) == 256
    assert sum(row["output_token_budget"] for row in schedule) == 122_880
    assert [row["call_order"] for row in schedule] == list(range(256))
    assert measurement.schedule_errors(schedule, public) == []
    assert Counter(row["arm"] for row in schedule) == {
        "serial_versioned_verifier": 160,
        "batched_versioned_verifier": 64,
        "batched_warm_prefix_direct": 32,
    }
    budgets: defaultdict[tuple[str, int, str], int] = defaultdict(int)
    claim_orders: dict[tuple[str, int], tuple[str, ...]] = {}
    for row in schedule:
        budgets[(row["group_id"], row["source_version"], row["arm"])] += row["output_token_budget"]
        claim_orders.setdefault((row["group_id"], row["source_version"]), tuple(row["claim_ids"]))
        assert row["split"] == "held_out"
        assert "expected_decision" not in canary.canonical_json(row)
    assert set(budgets.values()) == {1280}
    assert len(claim_orders) == 32
    assert len(set(claim_orders.values())) > 2
    assert measurement.call_budget_contract()["planned"]["claim_arm_rows"] == 384

    changed = deepcopy(schedule)
    changed[0]["output_token_budget"] += 1
    changed[1]["claim_ids"] = []
    assert {"schedule_mismatch", "allocated_output_tokens"}.issubset(
        measurement.schedule_errors(changed, public)
    )
    short = deepcopy(schedule[:-1])
    short[0]["arm"] = "wrong"
    errors = measurement.schedule_errors(short, public)
    assert {"call_count", "arm_call_counts"}.issubset(errors)
    assert any(error.startswith("arm_version_budget") for error in errors)
    assert measurement.schedule_errors([], {})[0].startswith("schedule_rebuild")
    with pytest.raises(ValueError, match="held_out_group_denominator"):
        measurement.build_schedule({})


def test_scenario_verify_7321_authority_scores_only_after_public_replay() -> None:
    """SCENARIO-VERIFY-7321-AUTHORITY keeps labels out of prediction rows."""

    public, evaluator = _fixture()
    schedule = measurement.build_schedule(public)
    call_rows = _valid_call_rows(schedule)
    replay = measurement.replay_public_predictions(public, schedule, call_rows)

    assert replay["receipt"]["call_count"] == 256
    assert replay["receipt"]["prediction_row_count"] == 384
    assert replay["receipt"]["transport_complete"] is True
    assert len(replay["predictions"]) == 384
    assert all("expected_decision" not in row for row in replay["predictions"])
    assert all("metric" not in row for row in replay["predictions"])
    assert all("source_spans" in row for row in replay["predictions"])

    scored = measurement.score_predictions(replay["predictions"], evaluator)
    assert len(scored) == 384
    assert set(row["expected_decision"] for row in scored) == {
        "supported",
        "contradicted",
        "unknown",
    }
    assert all(row["correct"] == 1 for row in scored)
    assert all(row["support_status"] in {"supported", "unsupported"} for row in scored)
    assert (
        measurement._source_spans(
            {"arm": "serial_versioned_verifier", "group_id": "missing", "source_version": 1},
            public,
            {},
        )
        == []
    )

    broken = deepcopy(replay["predictions"])
    joint = next(row for row in broken if row["arm"] == "batched_versioned_verifier")
    joint["censored"] = True
    joint["prediction"] = "unknown"
    joint["errors"] = ["required_call_unusable"]
    failed = measurement.score_predictions(broken, evaluator)
    retained = next(
        row for row in failed if row["unit_id"] == joint["unit_id"] and row["arm"] == joint["arm"]
    )
    assert retained["failed"] is True
    assert retained["correct"] == 0
    with pytest.raises(ValueError, match="label_identity"):
        measurement.score_predictions(replay["predictions"], {"labels": []})
    with pytest.raises(ValueError, match="label_identity"):
        measurement.score_predictions(replay["predictions"], {"labels": None})


def test_scenario_verify_7321_cost_charges_every_arm_without_shared_double_count() -> None:
    """SCENARIO-VERIFY-7321-COST separates warm work and shared initialization."""

    public, evaluator = _fixture()
    schedule = measurement.build_schedule(public)
    call_rows = _valid_call_rows(schedule)
    replay = measurement.replay_public_predictions(public, schedule, call_rows)
    scored = measurement.score_predictions(replay["predictions"], evaluator)
    cost = {
        "shared_initialization_s": 12.0,
        "native_parsing_s": 0.256,
        "prediction_replay_s": 0.384,
        "evaluation_s": 0.192,
        "serialization_s": 0.096,
    }
    groups, summary = measurement.per_source_group_results(scored, call_rows, cost)

    assert len(groups) == 16
    assert summary["shared_initialization_s"] == 12.0
    assert summary["cold_total_wall_s"] == pytest.approx(12.0 + summary["warm_complete_cost_s"])
    assert summary["summed_call_time_s"] == pytest.approx(2.56)
    assert summary["shared_initialization_counted_in_warm_cost"] is False
    assert all(set(row["arms"]) == set(fixture.ARMS) for row in groups)
    assert all(row["arms"]["batched_versioned_verifier"]["row_count"] == 8 for row in groups)
    with pytest.raises(ValueError, match="source_group_denominator"):
        measurement.per_source_group_results(
            [row for row in scored if row["group_id"] != "eval-batch-g15"], call_rows, cost
        )


def test_scenario_verify_7321_intervals_resample_sixteen_groups() -> None:
    """SCENARIO-VERIFY-7321-INTERVALS uses whole groups for every seeded draw."""

    groups = []
    for index in range(16):
        groups.append(
            {
                "group_id": f"g{index:02d}",
                "semantic_mismatches": 0,
                "stale_constraints_served": 0,
                "arms": {
                    "serial_versioned_verifier": {
                        "accuracy": 1.0,
                        "coverage": 1.0,
                        "false_accepts": 0,
                        "warm_complete_cost_s": 4.0,
                    },
                    "batched_versioned_verifier": {
                        "accuracy": 1.0,
                        "coverage": 1.0,
                        "false_accepts": 0,
                        "warm_complete_cost_s": 1.0,
                    },
                    "batched_warm_prefix_direct": {
                        "accuracy": 1.0,
                        "coverage": 1.0,
                        "false_accepts": 0,
                        "warm_complete_cost_s": 2.0,
                    },
                },
            }
        )
    intervals = measurement.paired_intervals(groups, draws=10_000, seed=7321003)

    assert intervals["bootstrap_unit"] == "source_group"
    assert intervals["group_count"] == 16
    assert intervals["draws"] == 10_000
    assert intervals["seed"] == 7321003
    assert len(intervals["group_vectors"]["accuracy_difference_vs_direct"]) == 16
    assert intervals["metrics"]["accuracy_difference_vs_direct"]["one_sided_95_lower"] == 0
    assert intervals["metrics"]["full_cost_speedup_vs_serial"]["one_sided_95_lower"] == 4
    assert intervals["metrics"]["full_cost_speedup_vs_direct"]["one_sided_95_lower"] == 2
    assert measurement.paired_intervals(groups, draws=10_000, seed=7321003) == intervals
    with pytest.raises(ValueError, match="source_group_denominator"):
        measurement.paired_intervals(groups[:-1], draws=10, seed=1)


def test_scenario_verify_7321_terminal_separates_capture_from_value() -> None:
    """SCENARIO-VERIFY-7321-TERMINAL keeps a complete efficacy null auditable."""

    public, evaluator = _fixture()
    schedule = measurement.build_schedule(public)
    call_rows = _valid_call_rows(schedule)
    replay = measurement.replay_public_predictions(public, schedule, call_rows)
    rows = measurement.score_predictions(replay["predictions"], evaluator)
    cost = {
        "shared_initialization_s": 12.0,
        "native_parsing_s": 0.0,
        "prediction_replay_s": 0.0,
        "evaluation_s": 0.0,
        "serialization_s": 0.0,
    }
    groups, cost_summary = measurement.per_source_group_results(rows, call_rows, cost)
    intervals = measurement.paired_intervals(groups, draws=100, seed=measurement.BOOTSTRAP_SEED)
    validation = _passing_validation()
    gates = measurement.acceptance_gates(
        rows,
        groups,
        intervals,
        capture_complete=True,
        validation_passed=True,
    )
    assert any(row["passed"] is False for row in gates if row["value_gate"])

    capture = {
        "model_loaded": True,
        "runtime_error": None,
        "gpu_receipts": {"provenance_ok": True},
        "runner_receipt": {"runner": "native_llama.cpp_server", "model_count": 1},
    }
    artifact = measurement.assemble_artifact(
        measurement.RUN_DATE,
        preconditions=[],
        schedule=schedule,
        per_call_rows=call_rows,
        replay=replay,
        rows=rows,
        group_results=groups,
        intervals=intervals,
        gates=gates,
        capture=capture,
        validation=validation,
        source_hashes={"fixture": {"sha256": "sha256:" + "b" * 64}},
        duration_s=60.0,
        phase_spans=[],
        timestamps={"started_at_utc": "a", "completed_at_utc": "b"},
        model_identity={
            "hf_id": measurement.MODEL_ID,
            "quantization": measurement.QUANTIZATION,
            "gguf_path": "/cache/model.gguf",
            "revision": "revision",
            "gguf_sha256": "sha256:" + "c" * 64,
            "binary_sha256": "sha256:" + "d" * 64,
        },
        cost_summary=cost_summary,
        authority_separation={
            "predictions_sealed_before_labels": True,
            "evaluation_labels_read_during_prediction": False,
            "evaluator_process_separate": True,
        },
    )
    assert artifact["batch_capture_complete_score"] == 1
    assert artifact["batch_value_score"] == 0
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["production_path_enabled"] is False
    assert measurement.validate_artifact(artifact) == []

    capture_failed_gates = deepcopy(gates)
    next(row for row in capture_failed_gates if row["capture_gate"])["passed"] = False
    capture_null = measurement.assemble_artifact(
        measurement.RUN_DATE,
        preconditions=[],
        schedule=schedule,
        per_call_rows=call_rows,
        replay=replay,
        rows=rows,
        group_results=groups,
        intervals=intervals,
        gates=capture_failed_gates,
        capture=capture,
        validation=validation,
        source_hashes={},
        duration_s=60.0,
        phase_spans=[],
        timestamps=artifact["timestamps"],
        model_identity=artifact["runtime_model_identity"],
        cost_summary=cost_summary,
        authority_separation=artifact["authority_separation"],
    )
    assert capture_null["honest_verdict"] == "complete_null_batch_capture_incomplete"

    passing_gates = deepcopy(gates)
    for row in passing_gates:
        row["passed"] = True
    circular = measurement.assemble_artifact(
        measurement.RUN_DATE,
        preconditions=[],
        schedule=schedule,
        per_call_rows=call_rows,
        replay=replay,
        rows=rows,
        group_results=groups,
        intervals=intervals,
        gates=passing_gates,
        capture=capture,
        validation=validation,
        source_hashes={},
        duration_s=60.0,
        phase_spans=[],
        timestamps=artifact["timestamps"],
        model_identity=artifact["runtime_model_identity"],
        cost_summary=cost_summary,
        authority_separation=artifact["authority_separation"],
    )
    assert circular["verdict_class"] == "circular_positive"
    assert circular["batch_value_score"] == 1

    corruptions = (
        ("schema", "bad"),
        ("run_date", "bad"),
        ("field_principles", {}),
        ("MODEL_SPECS", []),
        ("execution_host", ""),
        ("verifier_is_oracle", False),
        ("production_path_enabled", True),
        ("verdict_class", "bad"),
        ("invocation_counts", None),
        ("rows", []),
        ("per_call_rows", []),
        ("per_source_group_results", []),
        ("paired_intervals", {}),
        ("batch_capture_complete_score", 0),
        ("batch_value_score", 1),
    )
    observed_errors: set[str] = set()
    for field, value in corruptions:
        changed = deepcopy(artifact)
        changed[field] = value
        changed["reproducibility_checksum"] = measurement.artifact_checksum(changed)
        observed_errors.update(measurement.validate_artifact(changed))
    assert {
        "schema",
        "run_identity",
        "field_principles",
        "MODEL_SPECS",
        "execution_identity",
        "verifier_is_oracle",
        "production_path_enabled",
        "verdict_class",
        "invocation_counts",
        "row_denominator",
        "call_denominator",
        "group_denominator",
        "paired_intervals",
        "batch_capture_complete_score",
        "batch_value_score",
    }.issubset(observed_errors)
    wrong_checksum = deepcopy(artifact)
    wrong_checksum["reproducibility_checksum"] = "wrong"
    assert "reproducibility_checksum" in measurement.validate_artifact(wrong_checksum)
    wrong_inference = deepcopy(artifact)
    wrong_inference["inference_mode"] = "cpu"
    wrong_inference["reproducibility_checksum"] = measurement.artifact_checksum(wrong_inference)
    assert "inference_classification" in measurement.validate_artifact(wrong_inference)
    missing_receipt = deepcopy(artifact)
    missing_receipt["validation_receipts"] = missing_receipt["validation_receipts"][:-1]
    missing_receipt["reproducibility_checksum"] = measurement.artifact_checksum(missing_receipt)
    assert "validation_receipts" in measurement.validate_artifact(missing_receipt)
    positive = deepcopy(artifact)
    positive["verdict_class"] = "positive"
    positive["reproducibility_checksum"] = measurement.artifact_checksum(positive)
    assert "oracle_positive_forbidden" in measurement.validate_artifact(positive)
    disqualified_with_pass = deepcopy(artifact)
    disqualified_with_pass["verdict_class"] = "disqualified"
    disqualified_with_pass["reproducibility_checksum"] = measurement.artifact_checksum(
        disqualified_with_pass
    )
    assert "disqualified_without_validation_failure" in measurement.validate_artifact(
        disqualified_with_pass
    )

    runtime_blocked = measurement.assemble_artifact(
        measurement.RUN_DATE,
        preconditions=[],
        schedule=schedule,
        per_call_rows=call_rows[:1],
        replay=replay,
        rows=rows,
        group_results=groups,
        intervals=intervals,
        gates=capture_failed_gates,
        capture={**capture, "runtime_error": "external"},
        validation=validation,
        source_hashes={},
        duration_s=60.0,
        phase_spans=[],
        timestamps=artifact["timestamps"],
        model_identity=artifact["runtime_model_identity"],
        cost_summary=cost_summary,
        authority_separation=artifact["authority_separation"],
    )
    assert runtime_blocked["verdict_class"] == "blocked"

    failed_validation = deepcopy(validation)
    failed_validation["required_checks_passed"] = False
    failed_validation["failed_required_commands"] = ["focused_pytest"]
    failed_gates = measurement.acceptance_gates(
        rows,
        groups,
        intervals,
        capture_complete=True,
        validation_passed=False,
    )
    disqualified = measurement.assemble_artifact(
        measurement.RUN_DATE,
        preconditions=[],
        schedule=schedule,
        per_call_rows=call_rows,
        replay=replay,
        rows=rows,
        group_results=groups,
        intervals=intervals,
        gates=failed_gates,
        capture=capture,
        validation=failed_validation,
        source_hashes={},
        duration_s=60.0,
        phase_spans=[],
        timestamps=artifact["timestamps"],
        model_identity=artifact["runtime_model_identity"],
        cost_summary=cost_summary,
        authority_separation=artifact["authority_separation"],
    )
    assert disqualified["batch_capture_complete_score"] == 0
    assert disqualified["batch_value_score"] == 0
    assert disqualified["verdict_class"] == "disqualified"


def test_req_verify_7321_blocked_and_inference_shapes_fail_closed() -> None:
    """REQ-VERIFY-7321 validates blocked, load-only, and full generation states."""

    assert measurement.classify_inference(measurement.ZERO_INVOCATION_COUNTS) == {
        "model_invoked": False,
        "inference_substrate": "blocked_before_qualifying_computation",
        "inference_substrate_class": "blocked_no_run",
        "inference_mode": "not_run",
    }
    loads = {**measurement.ZERO_INVOCATION_COUNTS, "model_loads_attempted": 1}
    assert measurement.classify_inference(loads)["inference_substrate_class"] == (
        "model_load_no_generation"
    )
    live = {
        **loads,
        "model_loads_completed": 1,
        "generation_calls_attempted": 256,
        "generation_calls_completed": 256,
    }
    assert measurement.classify_inference(live)["inference_substrate"] == ("model_full_generation")

    check = measurement.gate_row(
        "exp7320_terminal_class",
        canary.EXPERIMENT_ID,
        "verdict_class",
        "not blocked, partial, or disqualified",
        "blocked",
        False,
        "A failed canary blocks the held-out run.",
    )
    blocked = measurement.blocked_artifact(
        measurement.RUN_DATE,
        [check],
        duration_s=0.2,
        timestamps={"started_at_utc": "a", "completed_at_utc": "b"},
        model_identity=_qualified("results/experiment_7320_v643_batch_canary.json")[
            "runtime_model_identity"
        ],
    )
    assert blocked["status"] == "blocked"
    assert blocked["honest_verdict"] == "blocked_exp7320_terminal_class"
    assert blocked["batch_capture_complete_score"] == 0
    assert blocked["batch_value_score"] == 0
    assert blocked["invocation_counts"] == measurement.ZERO_INVOCATION_COUNTS
    assert measurement.validate_artifact(blocked, require_validation=False) == []
    broken = deepcopy(blocked)
    broken["batch_capture_complete_score"] = 1
    broken["reproducibility_checksum"] = measurement.artifact_checksum(broken)
    assert "blocked_terminal_state" in measurement.validate_artifact(
        broken, require_validation=False
    )


def test_req_verify_7321_terminal_validator_and_date_are_strict(tmp_path: Path) -> None:
    """REQ-VERIFY-7321 rejects malformed terminal identity and dates."""

    assert measurement.validate_artifact(None) == ["artifact_mapping"]
    assert measurement.validate_artifact({})[0].startswith("missing_required_field")
    marker = tmp_path / "marker.txt"
    marker.write_text("bytes", encoding="utf-8")
    assert measurement.sha256_file(marker) == measurement.sha256_bytes(b"bytes")
    assert measurement._utc_now().endswith("+00:00")
    assert measurement._date_argument(measurement.RUN_DATE) == measurement.RUN_DATE
    with pytest.raises(Exception, match="--date"):
        measurement._date_argument("bad")
