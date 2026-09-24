"""Tests for REQ-ARC-WMTE-7581's bounded ARC transport canary."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7581_v662_arc_bounded_canary as exp


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _protocol() -> dict[str, object]:
    rows = []
    panels = {"A": ["su15", "sp80", "ft09"], "B": ["sb26", "g50t", "dc22"]}
    for panel, games in panels.items():
        for game in games:
            for seed in (7582001, 7582002):
                for arm in ("current_verifier", "integrity_guard"):
                    rows.append(
                        {
                            "panel": panel,
                            "game": game,
                            "seed": seed,
                            "arm": arm,
                            "censored": False,
                            "provenance": "frozen_protocol_unstarted",
                        }
                    )
    return {
        "schema": exp.PROTOCOL_SCHEMA,
        "panels": panels,
        "seeds": [7582001, 7582002],
        "arms": ["current_verifier", "integrity_guard"],
        "max_actions_per_episode": 600,
        "max_inductions_per_episode": 1,
        "rows": rows,
    }


def _fixture_root(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    root = tmp_path.resolve()
    runner = {
        "experiment_id": "exp7574-v662-measurement-requalification",
        "arc_runner_ready_score": 1,
        "flagged_adversarial": False,
        "verdict_class": "circular_positive",
    }
    protocol = _protocol()
    _write_json(root / exp.EXP7574_REL, runner)
    _write_json(root / exp.PROTOCOL_REL, protocol)
    _write_json(
        root / exp.EXP7580_REL,
        {
            "experiment_id": 7580,
            "verifier_support_ready_score": 1,
            "flagged_adversarial": False,
            "live_panel_protocol_path": exp.PROTOCOL_REL.as_posix(),
        },
    )
    for relative in exp.REQUIRED_REPOSITORY_INPUTS:
        path = root / relative
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("REQ-ARC-WMTE-7581\n", encoding="utf-8")
    return root, protocol


def _resolved_model(tmp_path: Path) -> dict[str, object]:
    model = tmp_path / "Qwen3.8-27B-Q4_K_M.gguf"
    model.write_bytes(b"gguf-test-bytes")
    return {
        "name": "Qwen3.8-27B",
        "hf_id": exp.MODEL_ID,
        "gpu": 0,
        "model_path": str(model),
        "selection_role": "current_headline",
    }


def test_scenario_arc_wmte_7581_upstream_block_names_every_operand(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7581-UPSTREAM-BLOCK: absence is complete blocked."""

    root, _ = _fixture_root(tmp_path)
    (root / exp.EXP7574_REL).unlink()
    checks, context = exp.collect_preconditions(
        root,
        model_resolver=lambda: _resolved_model(tmp_path),
        expected_hashes={exp.EXP7574_REL.as_posix(): "sha256:" + "0" * 64},
    )
    artifact = exp.build_blocked_artifact(
        checks=checks,
        context=context,
        duration_s=0.25,
        reason="exp7574_available",
    )
    failure = artifact["gate_check_summary"]["first_failure"]
    assert artifact["honest_verdict"].startswith("complete_blocked_")
    assert artifact["verdict_class"] == "blocked"
    assert artifact["inference_substrate_class"] == "blocked_no_run"
    assert artifact["invocation_counts"]["generation_calls_attempted"] == 0
    assert {"check", "upstream", "path", "field", "op", "expected", "observed"} <= set(failure)


def test_scenario_arc_wmte_7581_upstream_hashes_are_exact(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7581 binds Exp7574 and Exp7580 protocol exact bytes."""

    root, _ = _fixture_root(tmp_path)
    expected = {
        exp.EXP7574_REL.as_posix(): exp.sha256_file(root / exp.EXP7574_REL),
        exp.PROTOCOL_REL.as_posix(): exp.sha256_file(root / exp.PROTOCOL_REL),
    }
    checks, context = exp.collect_preconditions(
        root, model_resolver=lambda: _resolved_model(tmp_path), expected_hashes=expected
    )
    assert all(row["passed"] for row in checks)
    assert context["runner"]["arc_runner_ready_score"] == 1
    assert context["protocol"]["schema"] == exp.PROTOCOL_SCHEMA
    (root / exp.PROTOCOL_REL).write_text("{}\n", encoding="utf-8")
    changed, _ = exp.collect_preconditions(
        root, model_resolver=lambda: _resolved_model(tmp_path), expected_hashes=expected
    )
    assert "exp7580_protocol_hash" in exp.failed_checks(changed)


def test_scenario_arc_wmte_7581_future_requests_are_full_and_unsent() -> None:
    """SCENARIO-ARC-WMTE-7581-FUTURE-REQUESTS: retain all 24 units at 4096."""

    requests = exp.construct_future_requests(_protocol())
    assert len(requests) == 24
    assert {row["panel"] for row in requests} == {"A", "B"}
    assert all(row["max_tokens"] == 4096 for row in requests)
    assert all(row["capture_max_tokens"] == 4096 for row in requests)
    assert all(row["request_timeout_s"] == 240.0 for row in requests)
    assert all(row["request_sent"] is False for row in requests)
    assert all(row["requests_per_episode"] == 1 for row in requests)


@pytest.mark.parametrize("field", ["max_tokens", "capture_max_tokens"])
def test_scenario_arc_wmte_7581_rejects_inherited_256_ceiling(field: str) -> None:
    """SCENARIO-ARC-WMTE-7581-FUTURE-REQUESTS rejects the old 256 ceiling."""

    row = exp.construct_future_requests(_protocol())[0]
    row[field] = 256
    with pytest.raises(ValueError, match="future_request_contract"):
        exp.validate_future_requests([row], expected_count=1)


def _historical_transport() -> dict[str, object]:
    durations = [
        7.217714256999898,
        7.24059899400163,
        7.251299809999182,
        7.278156687003502,
        7.2892155299996375,
        7.299257055005,
        7.30383809000341,
        7.342770928000391,
        7.349703095998848,
        7.3554384730014135,
        7.481708917002834,
        7.517523009999422,
        9.815444686995761,
        10.02905843899498,
        10.251130500000727,
        12.392603503998544,
    ]
    return {
        "honest_verdict": "complete_null_live_arc_seam_observation",
        "verdict_class": "null",
        "flagged_adversarial": False,
        "MODEL_SPECS": [{"hf_id": exp.MODEL_ID}],
        "invocation_counts": {"generation_calls_completed": 16},
        "rows": [
            {
                "server_request_rows": [
                    {
                        "elapsed_s": duration,
                        "requested_max_tokens": 256,
                        "response_observed": True,
                    }
                ]
            }
            for duration in durations
        ],
    }


def test_scenario_arc_wmte_7581_panel_forecasts_are_independent_and_complete() -> None:
    """SCENARIO-ARC-WMTE-7581-PANEL-FORECASTS keeps twelve rows per panel."""

    history = _historical_transport()
    profile = exp.historical_request_profile(history, source_hash="sha256:" + "1" * 64)
    forecasts = exp.forecast_panels(
        exp.construct_future_requests(_protocol()),
        profile,
        measured_load_s=30.0,
        measured_capture_s=2.0,
    )
    assert profile["sample_count"] == 16
    assert profile["p95_request_s"] == pytest.approx(10.786498751000181)
    assert set(forecasts) == {"A", "B"}
    assert all(row["request_count"] == 12 for row in forecasts.values())
    assert all(row["episode_count"] == 12 for row in forecasts.values())
    assert all(row["max_tokens"] == 4096 for row in forecasts.values())
    assert all(row["request_timeout_s"] == 240.0 for row in forecasts.values())
    assert all(row["panel_feasible_score"] == 1 for row in forecasts.values())
    assert forecasts["A"] is not forecasts["B"]
    assert exp.can_start_future_request(240.0)
    assert not exp.can_start_future_request(239.999)


def test_scenario_arc_wmte_7581_panel_budget_failure_does_not_change_roster() -> None:
    """REQ-ARC-WMTE-7581 closes feasibility without outcome-driven shrinking."""

    history = _historical_transport()
    for row in history["rows"]:  # type: ignore[index]
        row["server_request_rows"][0]["elapsed_s"] = 240.0
    profile = exp.historical_request_profile(history, source_hash="sha256:" + "2" * 64)
    forecasts = exp.forecast_panels(
        exp.construct_future_requests(_protocol()),
        profile,
        measured_load_s=200.0,
        measured_capture_s=20.0,
    )
    assert all(row["request_count"] == 12 for row in forecasts.values())
    assert all(row["panel_feasible_score"] == 0 for row in forecasts.values())
    assert all(row["forecast_measurement_s"] > 3000.0 for row in forecasts.values())


def _request_receipt(index: int, *, owned: bool = True) -> dict[str, object]:
    return {
        "request_id": f"canary-{index}",
        "prompt": exp.NEUTRAL_PROMPTS[index],
        "reply": "Acknowledged.",
        "prompt_sha256": "sha256:" + str(index) * 64,
        "reply_sha256": "sha256:" + str(index + 2) * 64,
        "requested_max_tokens": 64,
        "actual_completion_tokens": 3,
        "finish_reason": "stop",
        "elapsed_s": 5.25,
        "request_started": True,
        "response_observed": True,
        "owned_runtime": owned,
        "process_identity": {"pid": 99, "pid_start_ticks": 123},
        "model_sha256": "sha256:" + "a" * 64,
        "censored": False,
        "provenance": "owned_bounded_transport",
    }


def test_scenario_arc_wmte_7581_owned_transport_requires_exactly_two_calls() -> None:
    """SCENARIO-ARC-WMTE-7581-OWNED-TRANSPORT authenticates exactly two calls."""

    receipts = [_request_receipt(0), _request_receipt(1)]
    reduction = exp.reduce_transport_receipts(receipts, generation_duration_s=10.5)
    assert reduction["arc_transport_ready_score"] == 1
    assert reduction["generation_calls_attempted"] == 2
    assert reduction["generation_calls_completed"] == 2
    assert reduction["completion_tokens"] == 6
    assert reduction["duration_floor_passed"] is True
    assert (
        exp.reduce_transport_receipts(receipts[:1], generation_duration_s=10.5)[
            "arc_transport_ready_score"
        ]
        == 0
    )
    not_owned = deepcopy(receipts)
    not_owned[1]["owned_runtime"] = False
    assert (
        exp.reduce_transport_receipts(not_owned, generation_duration_s=10.5)[
            "arc_transport_ready_score"
        ]
        == 0
    )


def test_scenario_arc_wmte_7581_no_call_is_never_live_inference() -> None:
    """SCENARIO-ARC-WMTE-7581-NO-CALL forbids a load-only generation claim."""

    reduction = exp.reduce_transport_receipts([], generation_duration_s=31.0)
    assert reduction["arc_transport_ready_score"] == 0
    assert reduction["generation_calls_attempted"] == 0
    assert exp.actual_substrate_class(reduction) == "blocked_no_run"


def test_foreign_gpu_owners_are_observed_but_never_stopped() -> None:
    """REQ-ARC-WMTE-7581 waits for capacity without signaling foreign PIDs."""

    inventory = [
        {
            "index": 0,
            "uuid": "GPU-foreign",
            "memory_total_mb": 24576,
            "memory_used_mb": 18000,
            "memory_free_mb": 6500,
            "processes": [{"pid": 772157, "name": "llama-server"}],
        }
    ]
    progress_rows: list[dict[str, object]] = []
    selected, receipt = exp.wait_for_cuda_capacity(
        inventory_fn=lambda: inventory,
        max_wait_s=0.0,
        poll_s=0.0,
        progress_fn=lambda row: progress_rows.append(row),
    )
    assert selected is None
    assert receipt["foreign_processes"][0]["pid"] == 772157
    assert receipt["signals_sent"] == []
    assert progress_rows


def test_scenario_arc_wmte_7581_learning_lifecycle_persists_and_reloads(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7581-LIFECYCLE-AND-TERMINAL exercises every stage."""

    receipt = exp.exercise_learning_lifecycle(tmp_path / "learner-state.json")
    assert receipt["operations"] == ["predict", "release", "update", "persist", "reload"]
    assert receipt["passed"] is True
    assert receipt["state_sha256_before_reload"] == receipt["state_sha256_after_reload"]


def test_request_receipt_reads_exact_prompt_reply_tokens_and_finish_reason() -> None:
    """REQ-ARC-WMTE-7581 raw receipts bind transport bytes and actual usage."""

    prompt = exp.NEUTRAL_PROMPTS[0]
    request_bytes = json.dumps(
        {"messages": [{"role": "user", "content": prompt}], "max_tokens": 64}
    ).encode()
    response_bytes = json.dumps(
        {
            "choices": [{"message": {"content": "Ready."}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 9, "completion_tokens": 2, "total_tokens": 11},
        }
    ).encode()
    receipt = exp.request_receipt_from_bytes(
        request_id="canary-0",
        request_bytes=request_bytes,
        response_bytes=response_bytes,
        elapsed_s=4.0,
        process_identity={"pid": 7, "pid_start_ticks": 11},
        runtime_identity={"server_pid": 8, "server_pid_start_ticks": 12},
        model_sha256="sha256:" + "a" * 64,
    )
    assert receipt["prompt"] == prompt
    assert receipt["reply"] == "Ready."
    assert receipt["actual_prompt_tokens"] == 9
    assert receipt["actual_completion_tokens"] == 2
    assert receipt["finish_reason"] == "stop"
    assert receipt["requested_max_tokens"] == 64
    assert receipt["owned_runtime"] is True
    assert receipt["prompt_sha256"].startswith("sha256:")
    assert receipt["reply_sha256"].startswith("sha256:")


def test_comparative_rows_preserve_all_frozen_units_and_raw_operands() -> None:
    """REQ-ARC-WMTE-7581 emits one reproducible row per canary and panel unit."""

    future = exp.construct_future_requests(_protocol())
    profile = exp.historical_request_profile(
        _historical_transport(), source_hash="sha256:" + "3" * 64
    )
    forecasts = exp.forecast_panels(future, profile, measured_load_s=30.0, measured_capture_s=2.0)
    rows = exp.build_comparative_rows(
        future_requests=future,
        raw_request_receipts=[_request_receipt(0), _request_receipt(1)],
        forecasts=forecasts,
    )
    assert len(rows) == 26
    assert len([row for row in rows if row["row_kind"] == "current_canary"]) == 2
    assert len([row for row in rows if row["row_kind"] == "future_panel"]) == 24
    required = {
        "comparison_unit",
        "arm",
        "raw_numerator",
        "raw_denominator",
        "metric",
        "metric_direction",
        "seed",
        "censored",
        "provenance",
    }
    assert all(required <= set(row) for row in rows)
    reduced = exp.independent_reduce_rows(rows)
    assert reduced["canary"]["success_numerator"] == 2
    assert reduced["canary"]["denominator"] == 2
    assert reduced["panels"]["A"]["row_count"] == 12
    assert reduced["panels"]["B"]["row_count"] == 12


def test_complete_artifact_is_mutation_sensitive_and_has_required_principles(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-7581-LIFECYCLE-AND-TERMINAL cold-checks raw rows."""

    artifact = exp.build_test_artifact(tmp_path)
    assert exp.validate_artifact(artifact, require_validation=False) == []
    required = {
        "honest_verdict",
        "verdict_class",
        "flagged_adversarial",
        "gate_check_summary",
        "acceptance_gate_results",
        "rows",
        "inference_substrate_class",
        "MODEL_SPECS",
        "invocation_counts",
        "duration_s",
        "source_artifact_hashes",
        "validation_receipts",
        "field_principles",
        "verifier_is_oracle",
        "arc_transport_ready_score",
        "panel_a_feasible_score",
        "panel_b_feasible_score",
        "raw_request_receipts",
        "solve_provenance",
    }
    assert required <= set(artifact)
    assert required <= set(artifact["field_principles"])
    assert artifact["verdict_class"] == "null"
    assert artifact["positive_claim"] is False
    assert artifact["solve_provenance"] == "canary_no_solve_claim"
    changed = deepcopy(artifact)
    changed["rows"][0]["raw_numerator"] = 0
    assert "row_reduction_mismatch" in exp.validate_artifact(changed, require_validation=False)


def test_validation_manifest_and_commands_are_explicit_and_private(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7581 freezes serial scoped validation with local coverage."""

    manifest = exp.affected_validation_manifest()
    assert manifest["test_paths"] == [exp.TEST_REL.as_posix()]
    assert manifest["changed_modules"] == [exp.MODULE_REL.as_posix()]
    commands = exp.build_validation_commands(exp.REPO_ROOT, tmp_path)
    by_name = {command.name: command for command in commands}
    assert set(by_name) == set(exp.validation_scope.REQUIRED_CHECK_NAMES)
    focused = by_name["focused_pytest"].argv
    assert ("-n", "0") == focused[1:3]
    assert "--no-cov" in focused
    coverage = by_name["changed_module_coverage"].argv
    assert coverage[:2] == ("/usr/bin/env", f"COVERAGE_FILE={tmp_path / 'coverage/.coverage'}")
    for command in commands:
        exp.prepare_command_parent(command)
    assert (tmp_path / "pytest").is_dir()
    assert (tmp_path / "coverage").is_dir()


def test_e2e_and_terminal_commands_name_every_required_check(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7581 declares ARC E2E and exact terminal readers."""

    e2e = exp.build_e2e_commands(exp.REPO_ROOT, tmp_path)
    assert {row.name for row in e2e} == set(exp.E2E_NAMES)
    smoke = next(row for row in e2e if row.name == "foreign_cwd_llm_off_e3_smoke")
    assert "/usr/bin/env" == smoke.argv[0]
    assert "-C" in smoke.argv
    assert "CARNOT_ARC_DISABLE_INDUCTION=1" in smoke.argv
    candidate = tmp_path / "candidate.json"
    terminal = exp.build_terminal_commands(exp.REPO_ROOT, candidate)
    assert {row.name for row in terminal} == set(exp.TERMINAL_NAMES)
    strict = next(row for row in terminal if row.name == "verdict_row_consistency_strict")
    assert "--strict" in strict.argv


def test_cold_replay_reduces_rows_and_lifecycle(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7581-LIFECYCLE-AND-TERMINAL reloads exact bytes."""

    artifact = exp.build_test_artifact(tmp_path)
    path = tmp_path / "candidate.json"
    _write_json(path, artifact)
    replay = exp.cold_replay(path, require_validation=False)
    assert replay["passed"] is True
    assert replay["row_reduction"] == artifact["independent_reduction"]
    assert (
        replay["learning_state_hash"] == artifact["learning_lifecycle"]["state_sha256_after_reload"]
    )


def test_parse_args_requires_exact_root_and_date(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7581 keeps the public wrapper thin and explicit."""

    args = exp.parse_args(["--root", str(tmp_path), "--date", exp.RUN_DATE])
    assert args.root == tmp_path
    assert args.date == exp.RUN_DATE
    with pytest.raises(SystemExit):
        exp.parse_args(["--root", str(tmp_path), "--date", "20260923"])


def test_boundary_helpers_preserve_exact_bytes_and_progress(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-ARC-WMTE-7581 makes byte persistence and progress observable."""

    assert exp.utc_now().endswith("Z")
    exp.progress(0.0, "unit", "before", item=1)
    assert "phase=unit event=before" in capsys.readouterr().out
    path = tmp_path / "value.json"
    exp.atomic_json(path, {"ok": True})
    assert exp.load_json(path) == {"ok": True}
    path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="json_object_required"):
        exp.load_json(path)
    assert exp._read_optional_object(tmp_path / "absent.json") == {}
    assert exp._model_declaration(None)["resolution_status"] == "cache_miss"
    root, _ = _fixture_root(tmp_path / "default-preconditions")
    checks, _ = exp.collect_preconditions(root, model_resolver=lambda: _resolved_model(tmp_path))
    assert "historical_transport_hash" in exp.failed_checks(checks)


def test_future_request_contract_rejects_all_roster_corruption() -> None:
    """SCENARIO-ARC-WMTE-7581-FUTURE-REQUESTS rejects malformed frozen units."""

    valid = exp.construct_future_requests(_protocol())
    with pytest.raises(ValueError, match="count"):
        exp.validate_future_requests(valid[:-1])
    missing = deepcopy(valid)
    missing[0].pop("arm")
    with pytest.raises(ValueError, match="missing_fields"):
        exp.validate_future_requests(missing)
    duplicate = deepcopy(valid)
    duplicate[-1].update({key: duplicate[0][key] for key in ("panel", "game", "seed", "arm")})
    with pytest.raises(ValueError, match="duplicate_identity"):
        exp.validate_future_requests(duplicate)
    with pytest.raises(ValueError, match="protocol_schema"):
        exp.construct_future_requests({})
    with pytest.raises(ValueError, match="protocol_rows"):
        exp.construct_future_requests({"schema": exp.PROTOCOL_SCHEMA, "rows": None})
    malformed = _protocol()
    malformed["rows"][0] = "bad"  # type: ignore[index]
    with pytest.raises(ValueError, match="protocol_row"):
        exp.construct_future_requests(malformed)
    one_panel = _protocol()
    for row in one_panel["rows"]:  # type: ignore[union-attr]
        row["panel"] = "A"
    with pytest.raises(ValueError, match="panel_counts"):
        exp.construct_future_requests(one_panel)


def test_historical_profile_and_forecast_reject_unauthenticated_inputs() -> None:
    """REQ-ARC-WMTE-7581 forecasts only from authenticated timing operands."""

    with pytest.raises(ValueError, match="timings_empty"):
        exp._linear_p95([])
    assert exp._linear_p95([2.5]) == 2.5
    history = _historical_transport()
    history["rows"].extend([None, {"server_request_rows": [None]}])  # type: ignore[union-attr]
    with pytest.raises(ValueError, match="not_authenticated"):
        exp.historical_request_profile(history, source_hash="unbound")
    profile = exp.historical_request_profile(
        _historical_transport(), source_hash="sha256:" + "4" * 64
    )
    profile["authenticated"] = False
    with pytest.raises(ValueError, match="history_invalid"):
        exp.forecast_panels(
            exp.construct_future_requests(_protocol()),
            profile,
            measured_load_s=0,
            measured_capture_s=0,
        )


def test_row_reducer_rejects_duplicate_denominator_panel_and_kind() -> None:
    """REQ-ARC-WMTE-7581 independent reduction rejects ambiguous raw rows."""

    base = {
        "comparison_unit": "u",
        "arm": "a",
        "raw_numerator": 1,
        "raw_denominator": 1,
        "row_kind": "current_canary",
    }
    with pytest.raises(ValueError, match="duplicate_comparison_row"):
        exp.independent_reduce_rows([base, base])
    with pytest.raises(ValueError, match="invalid_row_denominator"):
        exp.independent_reduce_rows([{**base, "raw_denominator": 0}])
    with pytest.raises(ValueError, match="invalid_panel"):
        exp.independent_reduce_rows([{**base, "row_kind": "future_panel", "panel": "C"}])
    with pytest.raises(ValueError, match="invalid_row_kind"):
        exp.independent_reduce_rows([{**base, "row_kind": "unknown"}])


def _rehash(value: dict[str, object]) -> dict[str, object]:
    value["reproducibility_checksum"] = exp.reproducibility_checksum(value)
    return value


def test_artifact_validator_names_each_independent_contract_failure(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7581-LIFECYCLE-AND-TERMINAL rejects each drift class."""

    original = exp.build_test_artifact(tmp_path)
    mutations = [
        ("schema_mismatch", {"schema": "wrong"}),
        ("honest_verdict_not_terminal", {"honest_verdict": "running"}),
        ("verdict_class_invalid", {"verdict_class": "maybe"}),
        ("rows_invalid", {"rows": None}),
        ("raw_request_receipts_invalid", {"raw_request_receipts": None}),
        ("model_specs_missing", {"MODEL_SPECS": []}),
        ("model_specs_identity_mismatch", {"MODEL_SPECS": [{"hf_id": "wrong"}]}),
        ("field_principles_incomplete", {"field_principles": {}}),
        ("claim_scope_invalid", {"positive_claim": True}),
        ("transport_score_mismatch", {"arc_transport_ready_score": 0}),
        ("substrate_class_mismatch", {"inference_substrate_class": "wrong"}),
    ]
    for expected, update in mutations:
        changed = deepcopy(original)
        changed.update(update)
        _rehash(changed)
        assert expected in exp.validate_artifact(changed, require_validation=False)

    malformed_rows = deepcopy(original)
    malformed_rows["rows"][0]["raw_denominator"] = 0
    _rehash(malformed_rows)
    assert "row_reduction_invalid" in exp.validate_artifact(
        malformed_rows, require_validation=False
    )
    assert "required_validation_failed" in exp.validate_artifact(original, require_validation=True)

    blocked = exp.build_blocked_artifact(
        checks=[
            exp.gate_row(
                "missing",
                upstream="fixture",
                path="/missing",
                field="available",
                expected=True,
                observed=False,
            )
        ],
        context={"protocol": _protocol()},
        duration_s=0,
    )
    blocked["gate_check_summary"] = {}
    blocked["invocation_counts"]["generation_calls_attempted"] = 1
    _rehash(blocked)
    errors = exp.validate_artifact(blocked, require_validation=False)
    assert "blocked_gate_summary_invalid" in errors
    assert "blocked_generation_count_nonzero" in errors
    with pytest.raises(ValueError, match="requires_failed_check"):
        exp.build_blocked_artifact(
            checks=[exp.gate_row("ok", upstream="x", path="x", field="x", expected=1, observed=1)],
            context={},
            duration_s=0,
        )


def test_command_and_capacity_helpers_cover_owned_and_foreign_boundaries(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7581 prepares private paths and never signals foreign owners."""

    e2e = exp.build_e2e_commands(exp.REPO_ROOT, tmp_path)
    smoke = next(row for row in e2e if row.name == "foreign_cwd_llm_off_e3_smoke")
    exp.prepare_command_parent(smoke)
    assert (tmp_path / "foreign-cwd").is_dir()

    idle = {
        "index": 0,
        "uuid": "GPU-idle",
        "memory_total_mb": 24576,
        "memory_used_mb": 0,
        "memory_free_mb": 24576,
        "processes": [],
    }
    selected, receipt = exp.wait_for_cuda_capacity(
        inventory_fn=lambda: [idle], max_wait_s=0, poll_s=0
    )
    assert selected == idle
    assert receipt["signals_sent"] == []
    inventories = iter(
        [
            [{**idle, "memory_used_mb": 1000, "processes": [{"pid": 7}]}],
            [idle],
        ]
    )
    selected_after_wait, _ = exp.wait_for_cuda_capacity(
        inventory_fn=lambda: next(inventories), max_wait_s=1, poll_s=0
    )
    assert selected_after_wait == idle
    assert exp.process_start_tick(None) is None
    assert exp.process_start_tick(999_999_999) is None
    assert isinstance(exp.process_start_tick(__import__("os").getpid()), int)


def test_validation_receipt_reducers_and_blocked_enrichment(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7581 reduces command exits and blocked forecast evidence."""

    names = (*exp.validation_scope.REQUIRED_CHECK_NAMES, *exp.E2E_NAMES, *exp.TERMINAL_NAMES)
    passing = [{"name": name, "passed": True, "exit_code": 0, "timed_out": False} for name in names]
    assert exp._validation_passed(passing)
    assert exp._all_commands_passed(passing)
    checks: list[dict[str, object]] = []
    exp._append_failed_command_gate(checks, passing, check="commands")
    assert checks[0]["passed"] is True
    failed = [{"name": "x", "passed": False, "exit_code": 1, "timed_out": False}]
    assert not exp._all_commands_passed(failed)
    exp._append_failed_command_gate(checks, failed, check="failed_commands")
    assert checks[-1]["observed"][0]["name"] == "x"

    blocked = exp.build_blocked_artifact(
        checks=[
            exp.gate_row(
                "missing",
                upstream="fixture",
                path="/missing",
                field="available",
                expected=True,
                observed=False,
            )
        ],
        context={"protocol": {}},
        duration_s=0,
    )
    forecasts = exp._blocked_forecasts("no_measurement")
    enriched = exp._enrich_blocked_artifact(
        blocked,
        future_requests=[],
        historical_profile=None,
        panel_forecasts=forecasts,
        learning_lifecycle=None,
        runtime_receipt=None,
    )
    assert enriched["panel_forecasts"]["A"]["panel_feasible_score"] == 0
    path = tmp_path / "artifact.json"
    _write_json(path, enriched)
    assert exp.independent_reduce_artifact(path) == enriched["independent_reduction"]
    assert exp._parse_csv_line(" a, b ") == ["a", "b"]
    assert exp.source_hashes(exp.REPO_ROOT)[exp.MODULE_REL.as_posix()].startswith("sha256:")
