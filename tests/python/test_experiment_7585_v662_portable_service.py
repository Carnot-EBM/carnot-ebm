"""Tests for REQ-CL-7585, REQ-REPORT-7585, and their scenarios."""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any

import pytest
import numpy as np

from carnot import experiment_7585_v662_portable_service as exp
from carnot.experiment_7561_v661_recalibration_prototype import (
    constraint_errors,
    quadratic_objective,
    solve_constrained_map,
    statistics_from_examples,
)
from carnot.reporting import experiment_7303_validation_scope as validation_scope


ROOT = Path(__file__).resolve().parents[2]


def _receipt(name: str, passed: bool = True) -> dict[str, Any]:
    """Build a private receipt without implying that a command ran."""

    return {
        "name": name,
        "command": f"private {name}",
        "command_argv": ["private", name],
        "scope": "private_fixture",
        "exit_code": 0 if passed else 1,
        "duration_s": 0.01,
        "log_path": f"/tmp/{name}.log",
        "log_sha256": "sha256:" + "a" * 64,
        "output_tail": "private fixture",
        "passed": passed,
        "timed_out": False,
    }


def _receipts() -> list[dict[str, Any]]:
    names = [*validation_scope.REQUIRED_CHECK_NAMES, *exp.TERMINAL_CHECK_NAMES]
    return [_receipt(name) for name in names]


def _service_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode_index, mode in enumerate(("cold", "warm")):
        for repeat in range(30):
            pair = f"{mode}:{repeat}"
            for arm, service_ns, kernel_ns in (
                ("python", 2_000_000 + mode_index * 100_000 + repeat, 900_000),
                ("rust", 1_000_000 + mode_index * 50_000 + repeat, 100_000),
            ):
                rows.append(
                    {
                        "row_type": "service_trace",
                        "unit_id": pair,
                        "pair_id": pair,
                        "arm": arm,
                        "mode": mode,
                        "seed": 7_585_000 + repeat,
                        "numerator": service_ns,
                        "denominator": 1,
                        "metric": service_ns,
                        "metric_name": "whole_service_ns",
                        "metric_direction": "lower_is_better",
                        "kernel_ns": kernel_ns,
                        "state_bytes": 2048,
                        "censored": False,
                        "provenance": "private_equal_durability_fixture",
                        "durability_policy": exp.DURABILITY_POLICY,
                    }
                )
    return rows


def _parity_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stream in range(4):
        for arm in ("python", "rust"):
            rows.append(
                {
                    "row_type": "parity_stream",
                    "unit_id": f"stream:{stream}",
                    "pair_id": f"stream:{stream}",
                    "arm": arm,
                    "seed": 7_585_100 + stream,
                    "numerator": 0.0,
                    "denominator": 16,
                    "metric": 0.0,
                    "metric_name": "probability_absolute_error",
                    "metric_direction": "lower_is_better",
                    "censored": False,
                    "provenance": "private_process_fixture",
                    "typed_decision_mismatches": 0,
                    "acknowledgment_match": True,
                }
            )
    return rows


def test_preconditions_authenticate_exp7574_independently() -> None:
    """REQ-CL-7585; SCENARIO-CL-7585-BLOCKED."""

    context = exp.collect_preconditions(ROOT)

    assert context["kernel_branch_ready"] is True
    assert context["blocker"] is None
    assert context["requalification"]["recalibration_ready_score"] == 1
    assert context["requalification"]["flagged_adversarial"] is False
    assert all(row["passed"] for row in context["rows"])


def test_board_rows_preserve_exp7571_bytes_and_scope() -> None:
    """REQ-REPORT-7585; SCENARIO-REPORT-7585-BOARDS."""

    context = exp.collect_preconditions(ROOT)
    rows = exp.build_board_rows(context["portable_artifact"])

    assert [row["board"] for row in rows] == ["KV260", "PolarFire", "GateMate"]
    assert rows[0]["future_access"] == "ssh kria only"
    assert rows[0]["architecture_limit"] == "k_max<=5"
    assert rows[1]["fpga_sampling_claimed"] is False
    assert rows[2]["current_disposition"] == "blocked_unchanged_physical_prerequisite"
    assert rows[2]["gate_check_summary"]["observed"] == 0
    assert all(row["hardware_operations_issued"] == [] for row in rows)


def test_seeded_streams_include_boundaries_and_are_repeatable() -> None:
    """SCENARIO-CL-7585-PARITY covers 1,000 seeded fixture streams."""

    first = exp.seeded_streams(1_000)
    second = exp.seeded_streams(1_000)

    assert first == second
    assert len(first) == 1_000
    assert first[0]["probabilities"][:6] == [0.0, 1.0, 0.04, 0.16, 0.2, 0.8]
    assert all(len(row["probabilities"]) == 16 for row in first)


def test_portable_solver_preserves_qualified_objective_and_constraints() -> None:
    """REQ-CL-7585 ports the qualified objective, bounds, and tolerance."""

    stream = exp.seeded_streams(1)[0]
    gram, target = statistics_from_examples(stream["probabilities"], stream["labels"])
    qualified, qualified_receipt = solve_constrained_map(gram, target)
    portable, portable_receipt = exp.solve_portable_map(gram, target)

    assert qualified_receipt["converged"] is True
    assert portable_receipt["converged"] is True
    assert constraint_errors(portable) == []
    assert quadratic_objective(portable, gram, target) <= (
        quadratic_objective(qualified, gram, target) + 1e-7
    )
    assert np.max(np.abs(portable - exp.solve_portable_map(gram, target)[0])) == 0.0


def test_service_reducer_uses_paired_whole_service_rows() -> None:
    """REQ-CL-7585; SCENARIO-CL-7585-SERVICE and -HEADROOM."""

    summary = exp.summarize_service_rows(_service_rows())

    assert summary["paired_repeat_count"] == {"cold": 30, "warm": 30}
    assert summary["cold"]["python"]["p50_ns"] > summary["cold"]["rust"]["p50_ns"]
    assert summary["warm"]["whole_service_speedup"]["lower95"] > 1.0
    assert summary["warm"]["kernel_speedup"]["estimate"] == pytest.approx(9.0)
    assert summary["warm"]["whole_service_speedup"]["estimate"] < 3.0
    assert summary["amdahl_ceiling"]["denominator"] == "measured_whole_service_ns"
    assert summary["amdahl_ceiling"]["ideal_update_only_speedup"] < 2.0


def test_reducer_rejects_unpaired_or_unequal_durability_rows() -> None:
    """SCENARIO-REPORT-7585-ROWS requires exact pairs and one durability policy."""

    rows = _service_rows()
    with pytest.raises(ValueError, match="service_pair_incomplete"):
        exp.summarize_service_rows(rows[:-1])

    changed = deepcopy(rows)
    changed[0]["durability_policy"] = "memory_only"
    with pytest.raises(ValueError, match="durability_policy_mismatch"):
        exp.summarize_service_rows(changed)


def test_fixture_artifact_separates_parity_service_and_boards(tmp_path: Path) -> None:
    """REQ-REPORT-7585; SCENARIO-REPORT-7585-GATES."""

    artifact = exp.build_test_artifact(
        tmp_path,
        service_rows=_service_rows(),
        parity_rows=_parity_rows(),
        validation_receipts=_receipts(),
    )
    reduction = exp.independent_reduce(artifact)

    assert reduction["portable_parity_score"] == 1
    assert reduction["service_measurement_complete_score"] == 1
    assert reduction["board_continuity_complete_score"] == 1
    assert artifact["whole_service_speedup"]["warm"]["lower95"] > 1.0
    assert artifact["verdict_class"] == "positive"
    assert artifact["honest_verdict"].startswith("complete_positive_")
    assert artifact["MODEL_SPECS"] == artifact["model_specs"] == []
    assert artifact["no_model_load"] is True
    assert artifact["verifier_is_oracle"] is False
    assert exp.validate_artifact(artifact, root=tmp_path, verify_sources=False) == []


def test_artifact_mutations_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7585-TERMINAL rejects promoted or changed evidence."""

    artifact = exp.build_test_artifact(
        tmp_path,
        service_rows=_service_rows(),
        parity_rows=_parity_rows(),
        validation_receipts=_receipts(),
    )
    mutations = [
        (
            "portable_parity_score_mismatch",
            lambda value: value.update(portable_parity_score=0),
        ),
        (
            "board_continuity_complete_score_mismatch",
            lambda value: value["board_rows"].pop(),
        ),
        (
            "current_invocation_claim_invalid",
            lambda value: value["invocation_counts"].update(generation_calls=1),
        ),
        (
            "required_validation_failed",
            lambda value: value["validation_receipts"][0].update(passed=False),
        ),
        (
            "field_principles_incomplete",
            lambda value: value["field_principles"].pop("whole_service_speedup"),
        ),
    ]
    for expected, mutate in mutations:
        changed = deepcopy(artifact)
        mutate(changed)
        changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
        assert expected in exp.validate_artifact(changed, root=tmp_path, verify_sources=False)

    changed = deepcopy(artifact)
    changed["reproducibility_checksum"] = "sha256:" + "0" * 64
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(
        changed, root=tmp_path, verify_sources=False
    )


def test_blocked_branch_names_exact_upstream_and_keeps_boards(tmp_path: Path) -> None:
    """SCENARIO-CL-7585-BLOCKED never gates board accounting."""

    blocker = {
        "check": "exp7574_recalibration_ready_score",
        "upstream": "Exp7574",
        "path": exp.REQUALIFICATION_PATH.as_posix(),
        "field": "recalibration_ready_score",
        "op": "eq",
        "expected": 1,
        "observed": 0,
    }
    artifact = exp.build_blocked_artifact(blocker, tmp_path, _receipts())

    assert artifact["honest_verdict"] == "complete_blocked_exp7574_recalibration_ready_score"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["kernel_branch_disposition"] == "blocked_numerical_prerequisite"
    assert artifact["gate_check_summary"] == blocker
    assert artifact["portable_parity_score"] == 0
    assert artifact["service_measurement_complete_score"] == 0
    assert artifact["board_continuity_complete_score"] == 1
    assert len(artifact["board_rows"]) == 3
    assert exp.validate_artifact(artifact, root=tmp_path, verify_sources=False) == []


def test_python_service_request_persists_reloads_and_rejects_duplicate(tmp_path: Path) -> None:
    """SCENARIO-CL-7585-PARITY exercises the durable Python arm."""

    events = exp.seeded_streams(1)[0]
    state = tmp_path / "python-state.json"
    response = exp.run_python_service_request(
        {
            "operation": "trace",
            "state_path": str(state),
            "events": exp.stream_events(events),
        }
    )

    assert response["ok"] is True
    assert response["acknowledged_release_count"] == 2
    assert response["processed_event_count"] == 16
    assert response["reloaded_state_matches"] is True
    assert response["state_bytes"] > 0
    assert response["stage_ns"]["predict"] > 0
    assert response["stage_ns"]["solve"] > 0
    assert response["stage_ns"]["fsync"] > 0

    duplicate = exp.run_python_service_request(
        {
            "operation": "trace",
            "state_path": str(state),
            "events": exp.stream_events(events),
        }
    )
    assert duplicate["ok"] is False
    assert duplicate["error"].startswith("duplicate_feedback:")


def test_source_custody_detects_changed_bytes(tmp_path: Path) -> None:
    """REQ-REPORT-7585 binds each conclusion to exact source bytes."""

    source = tmp_path / "source.json"
    source.write_text('{"recalibration_ready_score":1}\n', encoding="utf-8")
    artifact = exp.build_test_artifact(
        tmp_path,
        service_rows=_service_rows(),
        parity_rows=_parity_rows(),
        validation_receipts=_receipts(),
    )
    artifact["source_artifact_hashes"] = {"source.json": exp.source_row(source, tmp_path)}
    artifact["reproducibility_checksum"] = exp.reproducibility_checksum(artifact)
    assert exp.validate_artifact(artifact, root=tmp_path) == []

    source.write_text('{"recalibration_ready_score":0}\n', encoding="utf-8")
    assert "source_hash_invalid:source.json" in exp.validate_artifact(artifact, root=tmp_path)


def test_cold_replay_and_independent_cli_modes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-REPORT-7585-TERMINAL runs fresh read-only modes."""

    artifact = exp.build_test_artifact(
        tmp_path,
        service_rows=_service_rows(),
        parity_rows=_parity_rows(),
        validation_receipts=_receipts(),
    )
    candidate = tmp_path / "candidate.json"
    exp.atomic_json(candidate, artifact)

    common = ["--date", exp.RUN_DATE, "--root", str(tmp_path), "--no-source-check"]
    assert exp.main([*common, "--cold-replay", str(candidate)]) == 0
    assert exp.main([*common, "--independent-reduce", str(candidate)]) == 0
    assert '"valid": true' in capsys.readouterr().out.lower()

    candidate.write_text("[]\n", encoding="utf-8")
    assert exp.main([*common, "--cold-replay", str(candidate)]) == 1


def test_validation_manifest_and_terminal_commands_are_scoped(tmp_path: Path) -> None:
    """REQ-REPORT-7585 freezes only affected files and exact readers."""

    commands = exp.build_validation_commands(ROOT, tmp_path / "private")
    assert [row.name for row in commands] == list(validation_scope.REQUIRED_CHECK_NAMES)
    by_name = {row.name: row for row in commands}
    assert exp.TEST_PATH.as_posix() in by_name["focused_pytest"].argv
    assert "--fail-under=100" in by_name["changed_module_coverage_report"].argv
    coverage_env = dict(by_name["changed_module_coverage_report"].command_environment)
    assert coverage_env["COVERAGE_FILE"].startswith(str(tmp_path))
    assert all(Path(path).parent.exists() for path in exp.private_basetemps(commands))

    terminal = exp.terminal_commands(tmp_path / "candidate.json", ROOT)
    assert [row.name for row in terminal] == list(exp.TERMINAL_CHECK_NAMES)
    assert "--strict" in terminal[-1].argv


def test_parser_binds_fixed_date_paths_and_worker_mode() -> None:
    """REQ-REPORT-7585 keeps the public entrypoint thin and dated."""

    args = exp.parse_args(["--date", exp.RUN_DATE, "--root", str(ROOT)])
    assert args.root == ROOT
    assert args.output == exp.RESULT_PATH
    with pytest.raises(ValueError, match="run_date_must_equal_20260924"):
        exp.parse_args(["--date", "20260923"])


def test_checksum_ignores_runtime_noise_only() -> None:
    """REQ-REPORT-7585 binds evidence while allowing replay timing to differ."""

    value = {"alpha": 1, "duration_s": 1.0, "reproducibility_checksum": "old"}
    first = exp.reproducibility_checksum(value)
    value["duration_s"] = 2.0
    assert exp.reproducibility_checksum(value) == first
    value["alpha"] = 2
    assert exp.reproducibility_checksum(value) != first


def test_rust_source_declares_cli_not_pyo3() -> None:
    """REQ-CL-7585 ports a process boundary without claiming a binding."""

    text = (ROOT / exp.RUST_SOURCE_PATH).read_text(encoding="utf-8")
    assert "carnot.recalibration.sufficient_statistics.v1" in text
    assert "stdin" in text.lower()
    assert "fsync" in text.lower() or "sync_all" in text
    assert "pyo3" not in text.lower()


def test_numerical_and_state_guards_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CL-7585 rejects malformed numerical and lifecycle inputs."""

    with pytest.raises(FileNotFoundError):
        exp.source_row(tmp_path / "missing.json", tmp_path)
    outside = tmp_path.parent / "outside-exp7585.json"
    outside.write_text("{}\n", encoding="utf-8")
    assert exp.source_row(outside, tmp_path)["path"] == str(outside.resolve())
    with pytest.raises(ValueError, match="exp7571_board_rows_incomplete"):
        exp.build_board_rows({})
    with pytest.raises(ValueError, match="stream_count_must_be_positive"):
        exp.seeded_streams(0)
    assert len(exp.service_stream(count=3)["probabilities"]) == 3
    with pytest.raises(ValueError, match="probability_label_length_mismatch"):
        exp.stream_events({"stream_id": 1, "probabilities": [0.1], "labels": []})
    assert exp._project_monotone(np.asarray([1.0, 0.0])).tolist() == [0.5, 0.5]

    with pytest.raises(ValueError, match="sufficient_statistic_shape_invalid"):
        exp.solve_portable_map(np.zeros((2, 2)), np.zeros(2))
    bad = np.zeros((9, 9))
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="sufficient_statistic_not_finite"):
        exp.solve_portable_map(bad, np.zeros(9))

    machine = exp.SufficientStatisticMap.create()
    with pytest.raises(ValueError, match="binary_label_required"):
        exp._portable_update_batch(machine, [("bad", 0.5, 2)])
    monkeypatch.setattr(
        exp,
        "solve_portable_map",
        lambda *_args, **_kwargs: (np.asarray(exp.KNOTS), {"converged": False}),
    )
    with pytest.raises(RuntimeError, match="recalibration_solver_did_not_converge"):
        exp._portable_update_batch(machine, [("event", 0.5, 1)])

    invalid_state = tmp_path / "invalid.json"
    invalid_state.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="state_payload_not_object"):
        exp._load_state(invalid_state)

    fake = exp.SufficientStatisticMap.create()
    fake.sample_count = 1
    monkeypatch.setattr(exp, "_load_state", lambda _path: fake)
    with pytest.raises(ValueError, match="identity_state_reload_failed"):
        exp.initialize_state(tmp_path / "identity.json")


def test_worker_response_and_service_reducers_cover_failure_modes(tmp_path: Path) -> None:
    """SCENARIO-CL-7585-PARITY and -SERVICE reduce raw process outcomes."""

    state = exp.SufficientStatisticMap.create().to_payload()
    response = {
        "ok": True,
        "predictions": [{"probability": 0.5, "action": "escalate"}],
        "acknowledgments": [0],
        "state": state,
    }
    assert exp.compare_worker_responses(response, deepcopy(response))["passed"] is True
    failed = exp.compare_worker_responses({"ok": False, "error": "x"}, response)
    assert failed["passed"] is False
    assert (
        exp._failure_match(
            {"ok": False, "error": "recalibration_solver_did_not_converge"},
            {
                "ok": False,
                "error": "recalibration_solver_did_not_converge",
                "stage_ns": {},
            },
        )
        is True
    )
    with pytest.raises(ValueError, match="prediction_count_mismatch"):
        exp.compare_worker_responses(response, {**response, "predictions": []})
    assert exp._numeric_nested_error([1.0], []) == math.inf

    state_path = tmp_path / "state.json"
    exp.initialize_state(state_path)
    assert exp.run_python_service_request({"operation": "solver_failure"})["ok"] is False
    assert exp.run_python_service_request({"operation": "unknown"})["error"] == "unknown_operation"
    assert exp._percentile([3.0], 0.5) == 3.0
    with pytest.raises(ValueError, match="percentile_requires_values"):
        exp._percentile([], 0.5)

    rows = _service_rows()
    only_cold = [row for row in rows if row["mode"] == "cold"]
    with pytest.raises(ValueError, match="service_arm_or_mode_missing"):
        exp.summarize_service_rows(only_cold)
    reduction = exp.independent_reduce(
        {"rows": [rows[0]], "board_rows": [], "validation_receipts": []}
    )
    assert reduction["service_measurement_complete_score"] == 0


def test_reload_mismatch_is_a_typed_worker_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-CL-7585-PARITY does not acknowledge a changed reload."""

    state_path = tmp_path / "state.json"
    exp.initialize_state(state_path)
    first = exp.SufficientStatisticMap.create()
    changed = exp.SufficientStatisticMap.create()
    changed.theta[0] = 0.01
    loads = iter((first, changed))
    monkeypatch.setattr(exp, "_load_state", lambda _path: next(loads))
    result = exp.run_python_service_request(
        {
            "operation": "trace",
            "state_path": str(state_path),
            "events": [{"event_id": "e", "probability": 0.5, "label": 1}],
        }
    )
    assert result["ok"] is False
    assert result["error"] == "reloaded_state_mismatch"


def test_rust_full_trace_survives_registered_reload_tolerance(tmp_path: Path) -> None:
    """SCENARIO-CL-7585-SERVICE reloads all 20 durable release blocks."""

    state = tmp_path / "rust-state.json"
    exp.initialize_state(state)
    worker = exp._start_worker(ROOT, "rust")
    try:
        response, _elapsed = exp._exchange(
            worker,
            {
                "operation": "trace",
                "state_path": str(state),
                "events": exp.stream_events(exp.service_stream()),
            },
        )
    finally:
        exp._stop_worker(worker)
    assert response["ok"] is True
    assert response["processed_event_count"] == 160
    assert response["acknowledged_release_count"] == 20


def test_null_fixture_and_reader_diagnostics_are_explicit(tmp_path: Path) -> None:
    """REQ-REPORT-7585 retains nulls and names each malformed terminal field."""

    failed_receipts = _receipts()
    failed_receipts[0]["passed"] = False
    failed_receipts[0]["exit_code"] = 1
    null_artifact = exp.build_test_artifact(
        tmp_path,
        service_rows=_service_rows(),
        parity_rows=_parity_rows(),
        validation_receipts=failed_receipts,
    )
    assert null_artifact["verdict_class"] == "null"

    good = exp.build_test_artifact(
        tmp_path,
        service_rows=_service_rows(),
        parity_rows=_parity_rows(),
        validation_receipts=_receipts(),
    )
    mutations = [
        ("artifact_identity_mismatch", lambda value: value.update(schema="bad")),
        ("task_binding_mismatch", lambda value: value.update(run_date="bad")),
        ("model_specs_not_empty", lambda value: value.update(MODEL_SPECS=["bad"])),
        ("inference_declaration_invalid", lambda value: value.update(execution_venue="fpga")),
        (
            "score_not_bare_numeric:portable_parity_score",
            lambda value: value.update(portable_parity_score=True),
        ),
        (
            "whole_service_speedup_mismatch",
            lambda value: value["whole_service_speedup"].update(warm={}),
        ),
        ("unblocked_gate_summary_must_be_null", lambda value: value.update(gate_check_summary={})),
        ("unauthorized_hardware_claim", lambda value: value.update(hardware_smoke=True)),
    ]
    for expected, mutate in mutations:
        changed = deepcopy(good)
        mutate(changed)
        assert expected in exp.validate_artifact(changed, root=tmp_path, verify_sources=False)

    malformed = deepcopy(good)
    malformed["rows"] = 42
    assert "independent_reduction_failed" in exp.validate_artifact(
        malformed, root=tmp_path, verify_sources=False
    )

    blocker = {
        "check": "exp7574_recalibration_ready_score",
        "upstream": "Exp7574",
        "path": exp.REQUALIFICATION_PATH.as_posix(),
        "field": "recalibration_ready_score",
        "op": "eq",
        "expected": 1,
        "observed": 0,
    }
    blocked = exp.build_blocked_artifact(blocker, tmp_path, _receipts())
    blocked_mutations = [
        (
            "external_blocker_incomplete",
            lambda value: value.update(external_blocker={"check": "x"}),
        ),
        ("gate_check_summary_mismatch", lambda value: value.update(gate_check_summary={})),
        (
            "terminal_verdict_mismatch",
            lambda value: value.update(honest_verdict="complete_null_bad"),
        ),
        (
            "kernel_branch_disposition_mismatch",
            lambda value: value.update(kernel_branch_disposition="measured"),
        ),
        (
            "blocked_measurement_must_be_unstarted",
            lambda value: value.update(portable_parity_score=1),
        ),
    ]
    for expected, mutate in blocked_mutations:
        changed = deepcopy(blocked)
        mutate(changed)
        assert expected in exp.validate_artifact(changed, root=tmp_path, verify_sources=False)

    source = tmp_path / "sized.json"
    source.write_text("{}\n", encoding="utf-8")
    sized = deepcopy(good)
    sized["source_artifact_hashes"] = {"sized.json": exp.source_row(source, tmp_path)}
    sized["source_artifact_hashes"]["sized.json"]["bytes"] += 1
    assert "source_size_invalid:sized.json" in exp.validate_artifact(sized, root=tmp_path)

    missing = tmp_path / "missing-candidate.json"
    assert exp.independent_replay(missing, root=tmp_path) == ["artifact_not_object"]
