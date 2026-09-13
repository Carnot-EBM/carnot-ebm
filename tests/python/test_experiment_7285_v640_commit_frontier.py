"""Tests for the qualified acknowledgment commit frontier.

Spec refs: REQ-CL-7285 and SCENARIO-CL-7285-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys

import pytest

from carnot import experiment_7230_v636_native_belief as exp7230
from carnot import experiment_7285_v640_commit_frontier as exp7285


@pytest.fixture(scope="session")
def native():
    """REQ-CL-7285: load the exact native controller named by Exp7284."""

    source = exp7285._read_json(exp7285.REPO_ROOT / exp7285.EXP7256_RELATIVE)
    return exp7230.load_native_extension(Path(source["native_binary_receipt"]["module_file"]))


def test_scenario_cl_7285_preconditions_are_exact_or_terminal(tmp_path: Path) -> None:
    """SCENARIO-CL-7285-PRECONDITIONS: changed prototype evidence blocks all rows."""

    paths = exp7285.ExperimentPaths.under(tmp_path)
    checks, evidence = exp7285.collect_preconditions(exp7285.REPO_ROOT, paths)
    assert exp7285.gate_summary(checks)["passed"] is True
    assert evidence["exp7284"]["commit_protocol_ready_score"] == 1
    assert evidence["exp7284"]["status"] == "complete"
    assert all(paths.writable_targets())

    changed = tmp_path / "changed.json"
    changed.write_text('{"status":"complete"}\n', encoding="utf-8")
    failed, _ = exp7285.collect_preconditions(exp7285.REPO_ROOT, paths, exp7284_path=changed)
    blocked = exp7285.blocked_artifact_for_test(next(row for row in failed if not row["passed"]))
    assert exp7285.validate_artifact(blocked) == []
    assert blocked["status"] == "blocked"
    assert blocked["verdict_class"] == "blocked"
    assert blocked["rows"] == blocked["latency_throughput_rows"] == []
    assert blocked["component_rows"] == []
    assert blocked["gate_check_summary"]["upstream"]
    assert blocked["gate_check_summary"]["artifact_field"]


def test_req_cl_7285_trial_plan_freezes_pairing_and_random_order() -> None:
    """REQ-CL-7285: all fixed arms share events but use seeded randomized order."""

    plan = exp7285.freeze_trial_plan()
    assert len(plan) == 20 * 3 * 3
    assert {row["arrival_condition"] for row in plan} == set(exp7285.ARRIVAL_CONDITIONS)
    assert {row["max_group_size"] for row in plan} == {1, 4, 16}
    assert all(row["event_count"] == 128 for row in plan)
    assert all(row["max_wait_ms"] == (0 if row["max_group_size"] == 1 else 10) for row in plan)
    for seed in exp7285.PAIRED_SEEDS:
        for condition in exp7285.ARRIVAL_CONDITIONS:
            cells = [
                row for row in plan if row["seed"] == seed and row["arrival_condition"] == condition
            ]
            assert sorted(row["arm_order"] for row in cells) == [0, 1, 2]
            assert len({tuple(row["event_ids"]) for row in cells}) == 1
    assert any(
        [row["max_group_size"] for row in plan if row["seed"] == seed][:3] != [1, 4, 16]
        for seed in exp7285.PAIRED_SEEDS
    )


def test_scenario_cl_7285_e2e_real_arrival_commit_ack_and_restore(native, tmp_path: Path) -> None:
    """SCENARIO-CL-7285-E2E: a real queue reaches durable ack and cold restore."""

    event_rows, latency_rows, component_rows, parity_rows = exp7285.run_frontier_benchmark(
        native,
        tmp_path / "benchmark",
        seeds=(exp7285.PAIRED_SEEDS[0],),
        events_per_trial=16,
        max_duration_s=60.0,
    )
    assert len(event_rows) == 3 * 3 * 16
    assert len(latency_rows) == len(component_rows) == len(parity_rows) == 9
    assert all(row["censored"] is False for row in event_rows)
    assert all(row["acknowledged"] is True for row in event_rows)
    assert all(row["lost"] is False for row in event_rows)
    assert all(row["exact_final_state_parity"] is True for row in parity_rows)
    assert all(row["component_sum_matches"] is True for row in component_rows)
    interactive = [
        row for row in latency_rows if row["arrival_condition"] == "interactive_dependent"
    ]
    assert all(row["durable_group_count"] == 16 for row in interactive)
    assert all(row["no_amortization_control"] is True for row in interactive)
    assert all(row["improvement_claimed"] is False for row in interactive)
    assert all(row["visibility_p95_ns"] >= row["acknowledgment_p95_ns"] for row in interactive)

    controls = exp7285.run_protocol_failure_controls(
        native, Path(native.__file__), tmp_path / "controls"
    )
    assert controls["passed"] is True
    assert controls["queue_controls"]["all_controls_passed"] is True
    assert controls["crash_failure_count"] == 0
    assert controls["e2e"]["cold_restore"] is True
    assert controls["e2e"]["lost_acknowledged_event_count"] == 0
    assert controls["e2e"]["exact_final_state_parity"] is True


def test_scenario_cl_7285_timeout_preserves_declared_population(native, tmp_path: Path) -> None:
    """SCENARIO-CL-7285-CENSORING: a timeout keeps every planned event and trial."""

    event_rows, latency_rows, component_rows, parity_rows = exp7285.run_frontier_benchmark(
        native,
        tmp_path / "timeout",
        seeds=(exp7285.PAIRED_SEEDS[0],),
        events_per_trial=16,
        max_duration_s=0.0,
    )
    assert len(event_rows) == 3 * 3 * 16
    assert len(latency_rows) == len(component_rows) == len(parity_rows) == 9
    assert all(row["censored"] is True for row in event_rows)
    assert all(row["interval_complete"] is False for row in latency_rows)
    assert all(row["censoring_reason"] == "measurement_budget_exhausted" for row in event_rows)


def test_req_cl_7285_cold_reducer_and_fixed_value_gate() -> None:
    """REQ-CL-7285: paired burst throughput and steady p95 define bounded value."""

    rows, latency, components, parity = exp7285.synthetic_frontier_rows(
        grouped_speedup=2.0, steady_p95_ns=40_000_000
    )
    reduced = exp7285.reduce_saved_rows(rows, latency, components, parity)
    assert reduced["population_complete"] is True
    assert reduced["component_cost_complete"] is True
    assert reduced["zero_lost_acknowledged_events"] is True
    assert reduced["exact_final_state_parity"] is True
    assert reduced["selected_group_size"] in {4, 16}
    assert reduced["burst_throughput_lower_ci95"] > 1.5
    assert reduced["steady_acknowledgment_p95_ns"] <= 50_000_000
    assert reduced["value_gate_passed"] is True
    assert reduced["interactive_no_amortization_control"]["improvement_claimed"] is False

    slow_rows, slow_latency, slow_components, slow_parity = exp7285.synthetic_frontier_rows(
        grouped_speedup=1.0, steady_p95_ns=60_000_000
    )
    null = exp7285.reduce_saved_rows(slow_rows, slow_latency, slow_components, slow_parity)
    assert null["population_complete"] is True
    assert null["value_gate_passed"] is False


def test_req_cl_7285_artifact_schema_validation_and_null_is_finished() -> None:
    """REQ-CL-7285: complete nulls remain terminal and positive is oracle-forbidden."""

    artifact = exp7285.complete_artifact_fixture_for_test(value_passed=False)
    assert exp7285.validate_artifact(artifact) == []
    assert artifact["status"] == "complete"
    assert artifact["honest_verdict"].startswith("complete_null")
    assert artifact["verdict_class"] == "null"
    assert artifact["commit_cost_complete_score"] == 1
    assert artifact["commit_cost_value_score"] == 0
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert not any(artifact["invocation_counts"].values())
    assert artifact["inference_substrate"] == "cpu_exact_solver_or_simulator"
    assert artifact["inference_substrate_class"] == "cpu_exact_solver_or_simulator"
    assert artifact["execution_venue"] == "host"
    assert artifact["claim_boundary"]["board_or_tsu_speed_claim"] is False
    assert artifact["claim_boundary"]["interactive_improvement_claim"] is False
    assert artifact["acceleration_envelope"]["nfr_01_10x_target"] == 10.0
    assert artifact["acceleration_envelope"]["target_100x"] == 100.0

    positive = exp7285.complete_artifact_fixture_for_test(value_passed=True)
    assert exp7285.validate_artifact(positive) == []
    assert positive["verdict_class"] == "circular_positive"
    assert positive["verdict_class"] != "positive"

    changed = deepcopy(artifact)
    changed["latency_throughput_rows"][0]["throughput_events_per_s"] += 1.0
    changed["reproducibility_checksum"] = exp7285.artifact_checksum(changed)
    assert "cold_reduction" in exp7285.validate_artifact(changed)


def test_scenario_cl_7285_atomic_publish_and_thin_entrypoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-CL-7285-TERMINAL: publish follows checks and the script only delegates."""

    script = exp7285.REPO_ROOT / "scripts/experiments/experiment_7285_v640_commit_frontier.py"
    calls: list[object] = []
    original = exp7285.main
    monkeypatch.setattr(exp7285, "main", lambda argv=None: calls.append(argv) or 0)
    monkeypatch.setattr(sys, "argv", [str(script), "--date", exp7285.RUN_DATE])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(script), run_name="__main__")
    assert exit_info.value.code == 0
    assert calls == [None]
    assert len(script.read_text(encoding="utf-8").splitlines()) <= 12
    monkeypatch.setattr(exp7285, "main", original)

    artifact = exp7285.complete_artifact_fixture_for_test(value_passed=False)
    passed = exp7285.check("fixture", "fixture", "field", True, True, True)
    monkeypatch.setattr(
        exp7285,
        "collect_preconditions",
        lambda *_args, **_kwargs: ([passed], {"hashes": {"fixture": "sha256:fixture"}}),
    )
    monkeypatch.setattr(exp7285, "build_artifact", lambda *_args, **_kwargs: deepcopy(artifact))
    commands: list[list[str]] = []

    def run_command(command: list[str], **_kwargs):
        commands.append(command)
        return {"command": command, "exit_code": 0, "output": "ok\n"}

    monkeypatch.setattr(exp7285, "_stream_subprocess", run_command)
    output = tmp_path / "terminal.json"
    result = exp7285.run_experiment(tmp_path, output, exp7285.RUN_DATE)
    assert json.loads(output.read_text(encoding="utf-8")) == result
    assert len(commands) == len(exp7285._scoped_validation_commands(tmp_path)) + 3

    output.unlink()
    monkeypatch.setattr(
        exp7285,
        "_stream_subprocess",
        lambda command, **_kwargs: {"command": command, "exit_code": 1, "output": "failed"},
    )
    with pytest.raises(RuntimeError, match="scoped validation failed"):
        exp7285.run_experiment(tmp_path, output, exp7285.RUN_DATE)
    assert not output.exists()


def test_req_cl_7285_cli_validation_is_read_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-CL-7285: read-only validation never changes candidate bytes."""

    artifact = exp7285.complete_artifact_fixture_for_test(value_passed=False)
    path = tmp_path / "candidate.json"
    exp7285.atomic_write(path, artifact)
    before = path.read_bytes()
    assert exp7285.main(["--validate", str(path)]) == 0
    assert path.read_bytes() == before
    path.write_text("{}\n", encoding="utf-8")
    assert exp7285.main(["--validate", str(path)]) == 2
    assert exp7285.main(["--date", "bad", "--output", str(path)]) == 2

    monkeypatch.setattr(
        exp7285,
        "run_experiment",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("failed")),
    )
    assert exp7285.main(["--output", str(path)]) == 2


def test_req_cl_7285_cold_reducer_builder_and_defensive_paths(
    native, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-CL-7285: raw bytes reduce independently and failures stay explicit."""

    sequence = tmp_path / "sequence.json"
    sequence.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not an object"):
        exp7285._read_json(sequence)
    fixture = exp7285.complete_artifact_fixture_for_test(value_passed=False)
    original_checksum = exp7285.artifact_checksum
    monkeypatch.setattr(
        exp7285,
        "artifact_checksum",
        lambda _artifact: (_ for _ in ()).throw(TypeError("bad checksum")),
    )
    assert exp7285._checksum_valid({}) is False
    assert "reproducibility_checksum" in exp7285.validate_artifact(fixture)
    monkeypatch.setattr(exp7285, "artifact_checksum", original_checksum)

    monkeypatch.setattr(
        exp7285.exp7270,
        "_stream_subprocess",
        lambda command, **_kwargs: {"command": command, "exit_code": 0, "output": "ok"},
    )
    assert exp7285._stream_subprocess(["true"], root=tmp_path, operation="test")["exit_code"] == 0

    measured = exp7285.synthetic_frontier_rows(grouped_speedup=1.0, steady_p95_ns=60_000_000)
    raw = {
        "rows": measured[0],
        "latency_throughput_rows": measured[1],
        "component_rows": measured[2],
        "parity_rows": measured[3],
    }
    paths = exp7285.ExperimentPaths.under(tmp_path / "cold")
    exp7285.atomic_write(paths.raw_rows, raw)
    reduced = exp7285.reduce_saved_rows(*measured)

    def successful_cold(command: list[str], **_kwargs):
        exp7285.atomic_write(paths.reduced, reduced)
        return {"command": command, "exit_code": 0, "output": "cold ok"}

    monkeypatch.setattr(exp7285, "_stream_subprocess", successful_cold)
    observed, receipt = exp7285._reduce_raw_in_fresh_process(tmp_path, paths)
    assert observed == reduced
    assert receipt["exit_code"] == 0
    monkeypatch.setattr(
        exp7285,
        "_stream_subprocess",
        lambda command, **_kwargs: {"command": command, "exit_code": 1, "output": "bad"},
    )
    with pytest.raises(RuntimeError, match="cold row reduction failed"):
        exp7285._reduce_raw_in_fresh_process(tmp_path, paths)

    passed = exp7285.check("fixture", "fixture", "field", True, True, True)
    failed = exp7285.check("missing", "external", "path", "file", None, False)
    blocked = exp7285.build_artifact(
        tmp_path,
        exp7285.ExperimentPaths.under(tmp_path / "blocked"),
        validation_receipts=[],
        precondition_bundle=([failed], {"hashes": {}}),
    )
    assert blocked["status"] == "blocked"

    monkeypatch.setattr(exp7285.exp7230, "load_native_extension", lambda _path: native)
    monkeypatch.setattr(exp7285, "run_frontier_benchmark", lambda *_args, **_kwargs: measured)
    monkeypatch.setattr(
        exp7285,
        "_reduce_raw_in_fresh_process",
        lambda *_args: (reduced, {"path": "fixture", "sha256": "sha256:fixture"}),
    )
    monkeypatch.setattr(
        exp7285, "run_protocol_failure_controls", lambda *_args: exp7285._fixture_controls()
    )
    built = exp7285.build_artifact(
        tmp_path,
        exp7285.ExperimentPaths.under(tmp_path / "built"),
        validation_receipts=[exp7285.exp7284.validation_receipt("fixture", ["true"], 0, "ok", 0.1)],
        precondition_bundle=(
            [passed],
            {
                "hashes": {"fixture": "sha256:fixture"},
                "native_module_path": str(Path(native.__file__)),
            },
        ),
        seeds=(exp7285.PAIRED_SEEDS[0],),
        events_per_trial=1,
    )
    assert built["commit_cost_complete_score"] == 1
    assert exp7285.validate_artifact(built) == []
    monkeypatch.setattr(
        exp7285,
        "_reduce_raw_in_fresh_process",
        lambda *_args: ({}, {"path": "fixture", "sha256": "sha256:fixture"}),
    )
    with pytest.raises(ValueError, match="cold reducer mismatch"):
        exp7285.build_artifact(
            tmp_path,
            exp7285.ExperimentPaths.under(tmp_path / "mismatch"),
            validation_receipts=[],
            precondition_bundle=(
                [passed],
                {"hashes": {}, "native_module_path": str(Path(native.__file__))},
            ),
            seeds=(exp7285.PAIRED_SEEDS[0],),
            events_per_trial=1,
        )


def test_scenario_cl_7285_runner_and_cli_failure_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-CL-7285-TERMINAL: invalid candidates and CLI errors cannot publish."""

    with pytest.raises(ValueError, match="run date"):
        exp7285.run_experiment(tmp_path, tmp_path / "bad-date.json", "bad")
    failed = exp7285.check("missing", "external", "path", "file", None, False)
    monkeypatch.setattr(
        exp7285,
        "collect_preconditions",
        lambda *_args, **_kwargs: ([failed], {"hashes": {}}),
    )
    blocked_output = tmp_path / "blocked.json"
    assert exp7285.run_experiment(tmp_path, blocked_output, exp7285.RUN_DATE)["status"] == "blocked"
    monkeypatch.setattr(exp7285, "build_artifact", lambda *_args, **_kwargs: {})
    with pytest.raises(ValueError, match="invalid blocked"):
        exp7285.run_experiment(tmp_path, tmp_path / "invalid-blocked.json", exp7285.RUN_DATE)

    passed = exp7285.check("fixture", "fixture", "field", True, True, True)
    monkeypatch.setattr(
        exp7285,
        "collect_preconditions",
        lambda *_args, **_kwargs: ([passed], {"hashes": {}}),
    )
    monkeypatch.setattr(
        exp7285,
        "_stream_subprocess",
        lambda command, **_kwargs: {"command": command, "exit_code": 0, "output": "ok"},
    )
    monkeypatch.setattr(exp7285, "build_artifact", lambda *_args, **_kwargs: {})
    with pytest.raises(ValueError, match="invalid Exp7285 candidate"):
        exp7285.run_experiment(tmp_path, tmp_path / "invalid.json", exp7285.RUN_DATE)

    complete = exp7285.complete_artifact_fixture_for_test(value_passed=False)
    monkeypatch.setattr(exp7285, "build_artifact", lambda *_args, **_kwargs: deepcopy(complete))
    call_count = 0

    def fail_candidate(command: list[str], **_kwargs):
        nonlocal call_count
        call_count += 1
        return {
            "command": command,
            "exit_code": int(call_count == len(exp7285._scoped_validation_commands(tmp_path)) + 1),
            "output": "candidate failed",
        }

    monkeypatch.setattr(exp7285, "_stream_subprocess", fail_candidate)
    with pytest.raises(RuntimeError, match="candidate validation failed"):
        exp7285.run_experiment(tmp_path, tmp_path / "candidate-fail.json", exp7285.RUN_DATE)

    invalid = tmp_path / "invalid-json.json"
    invalid.write_text("{", encoding="utf-8")
    assert exp7285.main(["--validate", str(invalid)]) == 2
    assert exp7285.main(["--reduce-raw", str(invalid)]) == 2
    raw_path = tmp_path / "raw.json"
    reduced_path = tmp_path / "reduced.json"
    exp7285.atomic_write(
        raw_path,
        {
            "rows": complete["rows"],
            "latency_throughput_rows": complete["latency_throughput_rows"],
            "component_rows": complete["component_rows"],
            "parity_rows": complete["parity_rows"],
        },
    )
    assert exp7285.main(["--reduce-raw", str(raw_path), "--reduced-output", str(reduced_path)]) == 0
    assert exp7285._read_json(reduced_path) == complete["independent_raw_row_reducer"]
    outputs: list[Path] = []
    monkeypatch.setattr(
        exp7285,
        "run_experiment",
        lambda _root, output, _date: outputs.append(output) or complete,
    )
    absolute = tmp_path / "absolute.json"
    assert exp7285.main(["--output", str(absolute)]) == 0
    assert exp7285.main(["--output", "relative.json"]) == 0
    assert outputs == [absolute, exp7285.REPO_ROOT / "relative.json"]
