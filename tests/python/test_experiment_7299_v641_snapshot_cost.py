"""Tests for persistent full-snapshot storage cost.

Spec refs: REQ-CL-7299 and SCENARIO-CL-7299-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import runpy
import sqlite3
import sys

import pytest

from carnot import experiment_7230_v636_native_belief as exp7230
from carnot import experiment_7299_v641_snapshot_cost as exp7299


@pytest.fixture(scope="session")
def native():
    """REQ-CL-7299: load the installed native controller authenticated upstream."""

    upstream = exp7299._read_json(exp7299.REPO_ROOT / exp7299.EXP7298_RELATIVE)
    module_path = Path(upstream["native_identity"]["module_file"])
    return exp7230.load_native_extension(module_path)


def test_scenario_cl_7299_preconditions_authenticate_upstream_and_outputs(
    tmp_path: Path,
) -> None:
    """SCENARIO-CL-7299-PRECONDITIONS: changed readiness blocks row-free."""

    paths = exp7299.ExperimentPaths.under(tmp_path)
    checks, evidence = exp7299.collect_preconditions(exp7299.REPO_ROOT, paths)
    assert exp7299.gate_summary(checks)["passed"] is True
    assert evidence["exp7298"]["snapshot_journal_ready_score"] == 1
    assert (
        evidence["native_module_sha256"] == evidence["exp7298"]["native_identity"]["module_sha256"]
    )
    assert all(paths.writable_targets())

    changed = deepcopy(evidence["exp7298"])
    changed["snapshot_journal_ready_score"] = 0
    changed["reproducibility_checksum"] = exp7299.artifact_checksum(changed)
    changed_path = tmp_path / "changed-upstream.json"
    changed_path.write_text(json.dumps(changed), encoding="utf-8")
    failed, _ = exp7299.collect_preconditions(exp7299.REPO_ROOT, paths, exp7298_path=changed_path)
    blocked = exp7299.blocked_artifact_for_test(next(row for row in failed if not row["passed"]))
    assert exp7299.validate_artifact(blocked) == []
    assert blocked["status"] == "blocked"
    assert blocked["verdict_class"] == "blocked"
    assert blocked["rows"] == blocked["per_run_results"] == []
    assert blocked["phase_cost_rows"] == blocked["independent_recovery_rows"] == []
    assert blocked["gate_check_summary"]["artifact_field"]


def test_scenario_cl_7299_pairing_freezes_fixed_candidate_before_measurement() -> None:
    """SCENARIO-CL-7299-PAIRING: matched writers receive identical frozen work."""

    plan = exp7299.freeze_trial_plan()
    assert len(plan) == 8 * 3 * 3 * 2
    assert {row["storage_arm"] for row in plan} == {"atomic_replace", "sqlite_persist"}
    assert {row["max_group_size"] for row in plan} == {1, 4, 16}
    assert {row["arrival_process"] for row in plan} == set(exp7299.ARRIVAL_PROCESSES)
    assert all(row["event_count"] == 256 for row in plan)
    assert all(row["max_wait_ms"] == (0 if row["max_group_size"] == 1 else 10) for row in plan)
    for seed in exp7299.EVALUATION_SEEDS:
        for arrival in exp7299.ARRIVAL_PROCESSES:
            for group in exp7299.GROUP_SIZES:
                pair = [
                    row
                    for row in plan
                    if row["seed"] == seed
                    and row["arrival_process"] == arrival
                    and row["max_group_size"] == group
                ]
                assert sorted(row["arm_order"] for row in pair) == [0, 1]
                assert len({tuple(row["event_ids"]) for row in pair}) == 1
                assert len({row["initial_state_sha256"] for row in pair}) == 1
                assert len({row["arrival_schedule_sha256"] for row in pair}) == 1
    assert exp7299.DEPLOYMENT_CANDIDATE == {
        "storage_arm": "sqlite_persist",
        "max_group_size": 16,
    }


def test_scenario_cl_7299_real_writers_preserve_cost_parity_and_restart(
    native, tmp_path: Path
) -> None:
    """SCENARIO-CL-7299-COSTS: real writers reconcile phases and restore exactly."""

    rows, runs, costs, recovery = exp7299.run_snapshot_benchmark(
        native,
        tmp_path / "storage",
        seeds=(exp7299.EVALUATION_SEEDS[0],),
        events_per_trial=16,
        max_duration_s=60.0,
    )
    assert len(rows) == 3 * 3 * 2 * 16
    assert len(runs) == len(costs) == len(recovery) == 18
    assert all(row["censored"] is False for row in rows)
    assert all(row["acknowledged"] is True for row in rows)
    assert all(row["missing_after_restart"] is False for row in rows)
    assert all(row["exact_native_state_parity"] is True for row in runs)
    assert all(row["original_queue_limits"] is True for row in runs)
    assert all(row["phase_sum_matches"] is True for row in costs)
    assert all(row["acknowledged_sequence_match"] is True for row in recovery)
    assert all(row["exact_state_bytes_match"] is True for row in recovery)
    assert all(row["next_native_decision_match"] is True for row in recovery)
    assert all(row["recovery_ns"] > 0 for row in recovery)
    assert {
        "initialization_ns",
        "queue_wait_ns",
        "native_transition_ns",
        "serialization_ns",
        "write_ns",
        "sync_ns",
        "acknowledgment_ns",
        "retry_ns",
        "recovery_ns",
        "cold_total_ns",
        "steady_total_ns",
    } <= costs[0].keys()


def test_scenario_cl_7299_budget_censors_complete_pairs(native, tmp_path: Path) -> None:
    """SCENARIO-CL-7299-PAIRING: exhausted work keeps every paired unit."""

    rows, runs, costs, recovery = exp7299.run_snapshot_benchmark(
        native,
        tmp_path / "censored",
        seeds=(exp7299.EVALUATION_SEEDS[0],),
        events_per_trial=16,
        max_duration_s=0.0,
    )
    assert len(rows) == 18 * 16
    assert len(runs) == len(costs) == len(recovery) == 18
    assert all(row["censored"] is True for row in rows + runs + costs + recovery)
    assert all(row["censoring_reason"] == "benchmark_budget_exhausted" for row in rows)


def test_scenario_cl_7299_cluster_reducer_uses_fixed_group16_joint_gate() -> None:
    """SCENARIO-CL-7299-REDUCTION: fixed group 16 uses all frozen deployment gates."""

    measured = exp7299.synthetic_snapshot_rows(
        burst_ratio=2.0,
        cold_burst_ratio=1.8,
        steady_p95_ns=40_000_000,
        interactive_ratio=1.0,
    )
    reduced = exp7299.reduce_saved_rows(*measured)
    assert reduced["bootstrap_draws"] == 10_000
    assert reduced["deployment_candidate"] == exp7299.DEPLOYMENT_CANDIDATE
    assert reduced["burst_throughput_ratio_ci95"][0] >= 1.5
    assert reduced["cold_burst_throughput_ratio_ci95"][0] >= 1.5
    assert reduced["steady_acknowledgment_p95_ns"] <= 50_000_000
    assert reduced["interactive_latency_ratio_ci95"][1] <= 1.05
    assert reduced["snapshot_value_gate_passed"] is True
    assert {row["max_group_size"] for row in reduced["sensitivity_results"]} == {1, 4}
    assert all(row["selected"] is False for row in reduced["sensitivity_results"])

    null_rows = exp7299.synthetic_snapshot_rows(
        burst_ratio=1.4,
        cold_burst_ratio=1.2,
        steady_p95_ns=55_000_000,
        interactive_ratio=1.1,
    )
    null = exp7299.reduce_saved_rows(*null_rows)
    assert null["capture_complete"] is True
    assert null["snapshot_value_gate_passed"] is False
    assert null["nfr01_full_boundary_passed"] is False


def test_req_cl_7299_artifact_schema_separates_capture_value_and_nfr01() -> None:
    """REQ-CL-7299: complete null evidence stays terminal and makes no hardware claim."""

    artifact = exp7299.complete_artifact_fixture_for_test(value_passed=False)
    assert exp7299.validate_artifact(artifact) == []
    assert artifact["status"] == "complete"
    assert artifact["snapshot_capture_complete_score"] == 1
    assert artifact["snapshot_value_score"] == 0
    assert artifact["honest_verdict"].startswith("complete_null")
    assert artifact["verdict_class"] == "null"
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert not any(artifact["invocation_counts"].values())
    assert artifact["nfr01_assessment"]["target_speedup"] == 10.0
    assert artifact["nfr01_assessment"]["local_gate_does_not_satisfy_nfr01"] is True
    assert artifact["limitations"]["physical_power_loss_proven"] is False
    assert artifact["claim_boundary"]["fpga_performance_claimed"] is False
    assert artifact["claim_boundary"]["tenfold_carnot_acceleration_claimed"] is False

    positive = exp7299.complete_artifact_fixture_for_test(value_passed=True)
    assert exp7299.validate_artifact(positive) == []
    assert positive["verdict_class"] == "circular_positive"
    assert positive["verdict_class"] != "positive"

    changed = deepcopy(artifact)
    changed["per_run_results"][0]["throughput_events_per_s"] += 1.0
    changed["reproducibility_checksum"] = exp7299.artifact_checksum(changed)
    assert "independent_reduction" in exp7299.validate_artifact(changed)


def test_req_cl_7299_thin_entrypoint_atomic_publish_and_read_only_cli(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-CL-7299: validation precedes atomic publication and CLI validation is read-only."""

    script = exp7299.REPO_ROOT / "scripts/experiments/experiment_7299_v641_snapshot_cost.py"
    calls: list[object] = []
    original_main = exp7299.main
    monkeypatch.setattr(exp7299, "main", lambda argv=None: calls.append(argv) or 0)
    monkeypatch.setattr(sys, "argv", [str(script), "--date", exp7299.RUN_DATE])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(script), run_name="__main__")
    assert exit_info.value.code == 0
    assert calls == [None]
    assert len(script.read_text(encoding="utf-8").splitlines()) <= 12
    monkeypatch.setattr(exp7299, "main", original_main)

    artifact = exp7299.complete_artifact_fixture_for_test(value_passed=False)
    path = tmp_path / "candidate.json"
    exp7299.atomic_write(path, artifact)
    before = path.read_bytes()
    assert exp7299.main(["--validate", str(path)]) == 0
    assert path.read_bytes() == before
    path.write_text("{}\n", encoding="utf-8")
    assert exp7299.main(["--validate", str(path)]) == 2
    assert exp7299.main(["--date", "bad", "--output", str(path)]) == 2


def test_req_cl_7299_builder_cold_reducer_and_publish_fail_closed(
    native, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-CL-7299: fresh reduction and validation failures cannot publish terminal bytes."""

    measured = exp7299.synthetic_snapshot_rows(
        burst_ratio=1.4,
        cold_burst_ratio=1.2,
        steady_p95_ns=55_000_000,
        interactive_ratio=1.1,
    )
    reduced = exp7299.reduce_saved_rows(*measured)
    paths = exp7299.ExperimentPaths.under(tmp_path / "build")
    exp7299.atomic_write(
        paths.raw_rows,
        {
            "rows": measured[0],
            "per_run_results": measured[1],
            "phase_cost_rows": measured[2],
            "independent_recovery_rows": measured[3],
        },
    )

    def cold_success(command: list[str], **_kwargs):
        exp7299.atomic_write(paths.reduced, reduced)
        return exp7299.validation_receipt("cold", command, 0, "ok", 0.1)

    monkeypatch.setattr(exp7299, "_stream_subprocess", cold_success)
    observed, receipt = exp7299._reduce_raw_in_fresh_process(tmp_path, paths)
    assert observed == reduced
    assert receipt["exit_code"] == 0

    passed = exp7299.check("fixture", "fixture", "field", True, True, True)
    monkeypatch.setattr(exp7299.exp7230, "load_native_extension", lambda _path: native)
    monkeypatch.setattr(exp7299, "run_snapshot_benchmark", lambda *_args, **_kwargs: measured)
    monkeypatch.setattr(
        exp7299,
        "_reduce_raw_in_fresh_process",
        lambda *_args: (reduced, exp7299.validation_receipt("cold", ["true"], 0, "ok", 0.1)),
    )
    built = exp7299.build_artifact(
        tmp_path,
        paths,
        validation_receipts=[exp7299.validation_receipt("fixture", ["true"], 0, "ok", 0.1)],
        precondition_bundle=(
            [passed],
            {
                "hashes": {"fixture": "sha256:fixture"},
                "source_artifact_hashes": {"files": {}, "artifacts": {}},
                "native_module_path": str(Path(native.__file__)),
                "native_module_sha256": exp7299.sha256_file(Path(native.__file__)),
                "exp7298": exp7299._read_json(exp7299.REPO_ROOT / exp7299.EXP7298_RELATIVE),
            },
        ),
        seeds=exp7299.EVALUATION_SEEDS,
        events_per_trial=1,
    )
    assert built["snapshot_capture_complete_score"] == 1
    assert exp7299.validate_artifact(built) == []

    output = tmp_path / "terminal.json"
    monkeypatch.setattr(
        exp7299,
        "collect_preconditions",
        lambda *_args, **_kwargs: ([passed], {"hashes": {"fixture": "sha256:fixture"}}),
    )
    monkeypatch.setattr(exp7299, "build_artifact", lambda *_args, **_kwargs: deepcopy(built))
    monkeypatch.setattr(
        exp7299,
        "_stream_subprocess",
        lambda command, **_kwargs: exp7299.validation_receipt("test", command, 1, "failed", 0.1),
    )
    with pytest.raises(RuntimeError, match="validation failed"):
        exp7299.run_experiment(tmp_path, output, exp7299.RUN_DATE)
    assert not output.exists()


def test_scenario_cl_7299_sqlite_queue_and_failure_boundaries(
    native, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-CL-7299-COSTS: measured SQLite failures cannot gain acknowledgment."""

    state = exp7299.exp7257.seed_cost_state(4)
    releases = exp7299._trial_releases(exp7299.EVALUATION_SEEDS[0], "burst", 3)

    def adapter(name: str, **kwargs):
        return exp7299._MeasuredSQLiteJournal.from_state(
            native,
            state,
            tmp_path / name / "state.sqlite3",
            max_group_size=kwargs.pop("max_group_size", 4),
            max_wait_ms=10,
            **kwargs,
        )

    queue = adapter("queue", max_pending_events=1)
    assert queue.enqueue({})["disposition"] == "invalid_release"
    invalid = dict(releases[0])
    invalid["release_index"] = -1
    assert queue.enqueue(invalid)["disposition"] == "invalid_release"
    assert queue.enqueue(releases[0])["accepted"] is True
    assert queue.enqueue(releases[0])["disposition"] == "duplicate_id"
    assert queue.enqueue(releases[1])["disposition"] == "backpressure_events"
    assert queue.flush(reason="test")["acknowledged_event_ids"] == [releases[0]["event_id"]]
    assert queue.enqueue(releases[0])["disposition"] == "duplicate_id"
    assert queue.flush(reason="empty")["disposition"] == "flush_empty"
    queue._failed_restore_required = True
    with pytest.raises(RuntimeError, match="fresh restore"):
        queue.enqueue(releases[2])
    with pytest.raises(RuntimeError, match="fresh restore"):
        queue.flush(reason="failed")
    queue.close()

    byte_bound = adapter("bytes", max_pending_bytes=1)
    assert byte_bound.enqueue(releases[0])["disposition"] == "backpressure_bytes"
    byte_bound.close()

    native_failure = adapter("native-failure")
    native_failure.enqueue(releases[0])
    monkeypatch.setattr(
        native_failure._controller,
        "commit_batch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("native")),
    )
    assert native_failure.flush(reason="native")["disposition"] == "failed_rolled_back"
    native_failure.close()

    commit_failure = adapter("commit-failure")
    commit_failure.enqueue(releases[0])
    monkeypatch.setattr(
        commit_failure,
        "_commit_snapshot",
        lambda *_args: (_ for _ in ()).throw(sqlite3.OperationalError("commit")),
    )
    failed = commit_failure.flush(reason="commit")
    assert failed["disposition"] == "failed_restore_required"
    assert failed["acknowledged_event_ids"] == []
    commit_failure.close()

    closed = adapter("closed")
    closed.enqueue(releases[0])
    closed._connection.close()
    assert closed.flush(reason="closed")["disposition"] == "failed_restore_required"


def test_scenario_cl_7299_sqlite_snapshot_integrity_boundaries(
    native, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-CL-7299-RECOVERY: bad size, sequence, and row reads fail closed."""

    state = exp7299.exp7257.seed_cost_state(4)
    journal = exp7299._MeasuredSQLiteJournal.from_state(
        native,
        state,
        tmp_path / "integrity/state.sqlite3",
        max_group_size=4,
        max_wait_ms=10,
    )
    state_bytes = journal.state_bytes
    with pytest.raises(ValueError, match="state-byte bound"):
        journal._commit_snapshot(b"x" * (exp7299.exp7298.MAX_STATE_BYTES + 1), 1)
    with pytest.raises(exp7299.exp7298.SnapshotSequenceError, match="exactly one"):
        journal._commit_snapshot(state_bytes, 2)
    journal._connection.execute("UPDATE snapshot SET sequence=7 WHERE slot=1")
    with pytest.raises(exp7299.exp7298.SnapshotSequenceError, match="parent sequence"):
        journal._commit_snapshot(state_bytes, 1)
    journal._connection.execute("UPDATE snapshot SET sequence=0 WHERE slot=1")

    original_read = exp7299.exp7298._read_snapshot
    calls = 0

    def bad_pending(connection):
        nonlocal calls
        calls += 1
        observed = original_read(connection)
        return (observed[0], 9, observed[2], observed[3]) if calls == 1 else observed

    monkeypatch.setattr(exp7299.exp7298, "_read_snapshot", bad_pending)
    with pytest.raises(exp7299.exp7298.SnapshotCorruptionError, match="transactional"):
        journal._commit_snapshot(state_bytes, 1)
    monkeypatch.setattr(exp7299.exp7298, "_read_snapshot", original_read)

    calls = 0

    def bad_committed(connection):
        nonlocal calls
        calls += 1
        observed = original_read(connection)
        return (observed[0], 9, observed[2], observed[3]) if calls == 2 else observed

    monkeypatch.setattr(exp7299.exp7298, "_read_snapshot", bad_committed)
    with pytest.raises(exp7299.exp7298.SnapshotCorruptionError, match="committed"):
        journal._commit_snapshot(state_bytes, 1)
    journal.close()


def test_req_cl_7299_helpers_blocked_build_and_cold_failures(
    native, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-CL-7299: helper failures stay explicit and do not invent measurements."""

    assert (
        exp7299.ExperimentPaths.defaults().artifact == exp7299.REPO_ROOT / exp7299.RESULT_RELATIVE
    )
    sequence = tmp_path / "sequence.json"
    sequence.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not an object"):
        exp7299._read_json(sequence)
    original_checksum = exp7299.artifact_checksum
    monkeypatch.setattr(
        exp7299,
        "artifact_checksum",
        lambda _artifact: (_ for _ in ()).throw(TypeError("bad")),
    )
    assert exp7299._checksum_valid({}) is False
    monkeypatch.setattr(exp7299, "artifact_checksum", original_checksum)
    assert exp7299._bootstrap([], 1) == [None, None]

    monkeypatch.setattr(
        exp7299.exp7270,
        "_stream_subprocess",
        lambda command, **_kwargs: {"command": command, "exit_code": 0, "output": "ok"},
    )
    assert exp7299._stream_subprocess(["true"], root=tmp_path, operation="test")["exit_code"] == 0
    plain = exp7299._result_receipt("plain", ["true"], {"exit_code": 0, "output": "ok"}, 0.1)
    assert plain["log_sha256"].startswith("sha256:")

    failed = exp7299.check("missing", "external", "field", 1, 0, False)
    blocked = exp7299.build_artifact(
        tmp_path,
        exp7299.ExperimentPaths.under(tmp_path / "blocked"),
        validation_receipts=[],
        precondition_bundle=([failed], {"hashes": {}}),
    )
    assert blocked["status"] == "blocked"
    assert blocked["rows"] == []

    paths = exp7299.ExperimentPaths.under(tmp_path / "cold-fail")
    monkeypatch.setattr(
        exp7299,
        "_stream_subprocess",
        lambda command, **_kwargs: {"command": command, "exit_code": 1, "output": "bad"},
    )
    with pytest.raises(RuntimeError, match="cold row reduction failed"):
        exp7299._reduce_raw_in_fresh_process(tmp_path, paths)

    measured = exp7299.synthetic_snapshot_rows(
        burst_ratio=1.4,
        cold_burst_ratio=1.2,
        steady_p95_ns=55_000_000,
        interactive_ratio=1.1,
    )
    passed = exp7299.check("fixture", "fixture", "field", True, True, True)
    upstream = exp7299._read_json(exp7299.REPO_ROOT / exp7299.EXP7298_RELATIVE)
    monkeypatch.setattr(exp7299.exp7230, "load_native_extension", lambda _path: native)
    monkeypatch.setattr(exp7299, "run_snapshot_benchmark", lambda *_args, **_kwargs: measured)
    monkeypatch.setattr(
        exp7299,
        "_reduce_raw_in_fresh_process",
        lambda *_args: (
            {"wrong": True},
            exp7299.validation_receipt("cold", ["true"], 0, "ok", 0.1),
        ),
    )
    with pytest.raises(ValueError, match="cold reducer mismatch"):
        exp7299.build_artifact(
            tmp_path,
            exp7299.ExperimentPaths.under(tmp_path / "mismatch"),
            validation_receipts=[],
            precondition_bundle=(
                [passed],
                {
                    "hashes": {},
                    "native_module_path": str(Path(native.__file__)),
                    "native_module_sha256": exp7299.sha256_file(Path(native.__file__)),
                    "exp7298": upstream,
                },
            ),
        )


def test_req_cl_7299_run_and_cli_terminal_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-CL-7299: blocked, complete, candidate-failure, and CLI paths terminate honestly."""

    failed = exp7299.check("missing", "external", "field", 1, 0, False)
    blocked = exp7299.blocked_artifact_for_test(failed)
    monkeypatch.setattr(exp7299, "collect_preconditions", lambda *_args, **_kwargs: ([failed], {}))
    monkeypatch.setattr(exp7299, "build_artifact", lambda *_args, **_kwargs: deepcopy(blocked))
    blocked_output = tmp_path / "blocked.json"
    assert exp7299.run_experiment(tmp_path, blocked_output, exp7299.RUN_DATE)["status"] == "blocked"
    assert blocked_output.is_file()
    with pytest.raises(ValueError, match="run date"):
        exp7299.run_experiment(tmp_path, blocked_output, "bad")

    complete = exp7299.complete_artifact_fixture_for_test(value_passed=False)
    passed = exp7299.check("fixture", "fixture", "field", True, True, True)
    monkeypatch.setattr(exp7299, "collect_preconditions", lambda *_args, **_kwargs: ([passed], {}))
    monkeypatch.setattr(exp7299, "build_artifact", lambda *_args, **_kwargs: deepcopy(complete))
    monkeypatch.setattr(exp7299, "_scoped_validation_commands", lambda _root: [])
    monkeypatch.setattr(
        exp7299,
        "_stream_subprocess",
        lambda command, **_kwargs: exp7299.validation_receipt("candidate", command, 0, "ok", 0.1),
    )
    complete_output = tmp_path / "complete.json"
    result = exp7299.run_experiment(tmp_path, complete_output, exp7299.RUN_DATE)
    assert result["status"] == "complete"
    assert complete_output.is_file()
    assert len(result["validation_receipts"]) == 3

    calls = 0

    def fail_candidate(command, **_kwargs):
        nonlocal calls
        calls += 1
        return exp7299.validation_receipt("candidate", command, 1, "bad", 0.1)

    monkeypatch.setattr(exp7299, "_stream_subprocess", fail_candidate)
    with pytest.raises(RuntimeError, match="candidate validation failed"):
        exp7299.run_experiment(tmp_path, tmp_path / "candidate-fail.json", exp7299.RUN_DATE)
    assert calls == 1

    measured = exp7299.synthetic_snapshot_rows(
        burst_ratio=1.4,
        cold_burst_ratio=1.2,
        steady_p95_ns=55_000_000,
        interactive_ratio=1.1,
    )
    raw = tmp_path / "raw.json"
    raw.write_text(
        json.dumps(
            {
                "rows": measured[0],
                "per_run_results": measured[1],
                "phase_cost_rows": measured[2],
                "independent_recovery_rows": measured[3],
            }
        ),
        encoding="utf-8",
    )
    reduced = tmp_path / "reduced.json"
    assert exp7299.main(["--reduce-raw", str(raw), "--reduced-output", str(reduced)]) == 0
    assert reduced.is_file()
    assert exp7299.main(["--reduce-raw", str(raw)]) == 2
    raw.write_text("{}\n", encoding="utf-8")
    assert exp7299.main(["--reduce-raw", str(raw), "--reduced-output", str(reduced)]) == 2
    invalid = tmp_path / "invalid.json"
    invalid.write_text("not-json", encoding="utf-8")
    assert exp7299.main(["--validate", str(invalid)]) == 2

    monkeypatch.setattr(exp7299, "run_experiment", lambda *_args: complete)
    assert exp7299.main(["--output", str(tmp_path / "main.json")]) == 0
    monkeypatch.setattr(
        exp7299,
        "run_experiment",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("failed")),
    )
    assert exp7299.main(["--output", str(tmp_path / "main-fail.json")]) == 2


def test_req_cl_7299_terminal_validation_rejections(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-CL-7299: no invalid blocked, measured, or receipt-updated artifact publishes."""

    failed = exp7299.check("missing", "external", "field", 1, 0, False)
    blocked = exp7299.blocked_artifact_for_test(failed)
    monkeypatch.setattr(exp7299, "collect_preconditions", lambda *_args, **_kwargs: ([failed], {}))
    monkeypatch.setattr(exp7299, "build_artifact", lambda *_args, **_kwargs: deepcopy(blocked))
    monkeypatch.setattr(exp7299, "validate_artifact", lambda _artifact: ["bad"])
    with pytest.raises(ValueError, match="invalid blocked"):
        exp7299.run_experiment(tmp_path, tmp_path / "bad-blocked.json", exp7299.RUN_DATE)

    complete = exp7299.complete_artifact_fixture_for_test(value_passed=False)
    passed = exp7299.check("fixture", "fixture", "field", True, True, True)
    monkeypatch.setattr(exp7299, "collect_preconditions", lambda *_args, **_kwargs: ([passed], {}))
    monkeypatch.setattr(exp7299, "build_artifact", lambda *_args, **_kwargs: deepcopy(complete))
    monkeypatch.setattr(exp7299, "_scoped_validation_commands", lambda _root: [])
    with pytest.raises(ValueError, match="invalid Exp7299 candidate"):
        exp7299.run_experiment(tmp_path, tmp_path / "bad-candidate.json", exp7299.RUN_DATE)

    validations = iter(([], ["bad-final"]))
    monkeypatch.setattr(exp7299, "validate_artifact", lambda _artifact: next(validations))
    monkeypatch.setattr(
        exp7299,
        "_stream_subprocess",
        lambda command, **_kwargs: exp7299.validation_receipt("candidate", command, 0, "ok", 0.1),
    )
    with pytest.raises(ValueError, match="invalid validated"):
        exp7299.run_experiment(tmp_path, tmp_path / "bad-final.json", exp7299.RUN_DATE)


def test_req_cl_7299_preserves_authenticated_full_suite_baseline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-CL-7299: an upstream-known full-suite failure stays explicit and non-gating."""

    complete = exp7299.complete_artifact_fixture_for_test(value_passed=False)
    passed = exp7299.check("fixture", "fixture", "field", True, True, True)
    evidence = {"exp7298": {"baseline_validation_failures": [{"exit_code": 1}]}}
    checkpoint = tmp_path / "prior.json"
    assert exp7299._prior_full_suite_failures(checkpoint) == []
    prior = exp7299.validation_receipt(
        "full_python_suite", ["pytest", "tests/python"], 1, "baseline", 0.1
    )
    exp7299.atomic_write(checkpoint, {"validation_receipts": [prior]})
    assert exp7299._prior_full_suite_failures(checkpoint) == [prior]
    monkeypatch.setattr(
        exp7299, "collect_preconditions", lambda *_args, **_kwargs: ([passed], evidence)
    )
    monkeypatch.setattr(exp7299, "build_artifact", lambda *_args, **_kwargs: deepcopy(complete))
    monkeypatch.setattr(
        exp7299,
        "_scoped_validation_commands",
        lambda _root: [("full_python_suite", ["pytest", "tests/python"], 10)],
    )
    calls = 0

    def results(command, **_kwargs):
        nonlocal calls
        calls += 1
        return exp7299.validation_receipt(
            "child", command, 1 if calls == 1 else 0, "baseline" if calls == 1 else "ok", 0.1
        )

    monkeypatch.setattr(exp7299, "_stream_subprocess", results)
    output = tmp_path / "baseline.json"
    artifact = exp7299.run_experiment(tmp_path, output, exp7299.RUN_DATE)
    assert output.is_file()
    assert len(artifact["baseline_validation_failures"]) == 1
    assert artifact["baseline_validation_failures"][0]["exit_code"] == 1
    assert all(row["exit_code"] == 0 for row in artifact["validation_receipts"])

    retry_checkpoint = exp7299.ExperimentPaths.under(tmp_path).checkpoint
    exp7299.atomic_write(retry_checkpoint, {"validation_receipts": [prior]})
    monkeypatch.setattr(
        exp7299,
        "_scoped_validation_commands",
        lambda _root: [
            ("focused_pytest", ["focused"], 10),
            ("full_python_suite", ["pytest", "tests/python"], 10),
        ],
    )
    commands: list[list[str]] = []

    def retry_results(command, **_kwargs):
        commands.append(command)
        return exp7299.validation_receipt("child", command, 0, "ok", 0.1)

    monkeypatch.setattr(exp7299, "_stream_subprocess", retry_results)
    retry = exp7299.run_experiment(tmp_path, tmp_path / "retry.json", exp7299.RUN_DATE)
    assert retry["baseline_validation_failures"] == [prior]
    assert ["focused"] in commands
    assert ["pytest", "tests/python"] not in commands
