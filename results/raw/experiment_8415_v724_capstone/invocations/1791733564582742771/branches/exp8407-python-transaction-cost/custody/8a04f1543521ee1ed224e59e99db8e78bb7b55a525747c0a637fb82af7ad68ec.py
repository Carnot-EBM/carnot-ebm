"""REQ-REPORT-8407 / REQ-VERIFY-8407: complete local service cost stays auditable."""

from copy import deepcopy
import json
import os
from pathlib import Path
import sys

import pytest

from carnot.verify import python_transaction_cost_8407 as k
from carnot.reporting import python_transaction_cost_8407 as e
from carnot.reporting import python_transaction_runner_8407 as runner
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.v709_execution import child


@pytest.fixture
def bundle(tmp_path):
    """SCENARIO-VERIFY-8407-DELAY: synthetic labels qualify scheduling mechanics only."""
    rows = [
        dict(slot=i, unit_id=str(i), source_cluster_id=str(i), x=[0, 0.2, 0.3, 0.4, 0.5])
        for i in range(1, 129)
    ]
    rows[7]["x"] = None
    rows[63]["x"] = None
    labels = tmp_path / "labels.json"
    atomic_json(
        labels,
        dict(
            rows=[
                dict(slot=i, unit_id=str(i), source_cluster_id=str(i), y=i % 2)
                for i in range(1, 129)
            ]
        ),
    )
    return dict(
        head=dict(
            arm="spline34",
            coefficients=[0.0] * 34,
            temperature=2.0,
            knots=k.s.evaluator.kernel.KNOTS,
            degree=3,
        ),
        slots=rows,
        labels=e.reference(labels),
    )


@pytest.mark.parametrize("ratio,batch", [(1, 32), (8, 8), (64, 32)])
def test_fixed_tape_and_tail(bundle, tmp_path, ratio, batch):
    """SCENARIO-VERIFY-8407-DELAY: all remainders and designated releases survive."""
    cell = dict(workload="natural", batch=batch, predictions_per_update=ratio)
    result = k.run_repeat(bundle, cell, 1, tmp_path / "repeat")
    summary = k.reduce_repeat(bundle, result)
    assert summary["prediction_attempts"] == 8 * ratio
    assert summary["feedback_attempts"] == summary["released"] == 8
    assert summary["pending"] == summary["lost"] == 0
    assert summary["wall_seconds"] >= summary["tail_drain_seconds"] > 0
    assert summary["completed_decisions"] == 8 * ratio
    assert any(row["status"] == "nonapplicable" for row in result["rows"])
    assert result["initial_state"]["pending"] == []
    assert summary["fsync_count"] > 0
    bad = deepcopy(result)
    bad["rows"][-1]["tick"] -= 1
    with pytest.raises(ValueError):
        k.reduce_repeat(bundle, bad)


@pytest.mark.parametrize(
    "mutation", ["zero_clock", "lost_tail", "manufactured", "duplicate", "count", "stage"]
)
def test_primitive_tamper_rejected(bundle, tmp_path, mutation):
    """SCENARIO-VERIFY-8407-DELAY: repaired hashes cannot make omitted work real."""
    result = k.run_repeat(
        bundle, dict(workload="natural", batch=8, predictions_per_update=1), 2, tmp_path
    )
    if mutation == "zero_clock":
        result["transactions"][0]["start_ns"] = 0
    if mutation == "lost_tail":
        result["rows"].pop()
    if mutation == "manufactured":
        result["rows"][-1]["changed"] = 999
    if mutation == "duplicate":
        result["rows"][1]["request_id"] = result["rows"][0]["request_id"]
    if mutation == "count":
        result["cell"]["predictions_per_update"] = 8
    if mutation == "stage":
        result["transactions"][0]["stages"][0]["end_ns"] = result["end_ns"] + 1
    with pytest.raises(ValueError):
        k.reduce_repeat(bundle, result)


def test_workloads_intervals_and_early_gate(bundle, tmp_path):
    """REQ-VERIFY-8407: ordinary/boundary are constructed numeric controls, never science."""
    results = []
    for repeat in range(1, 6):
        cell = dict(
            workload="boundary" if repeat == 1 else "ordinary", batch=32, predictions_per_update=1
        )
        result = k.run_repeat(bundle, cell, repeat, tmp_path / str(repeat))
        results.append(k.reduce_repeat(bundle, result))
    intervals = k.block_interval(results)
    assert intervals["resamples"] == 10000
    assert intervals["low"] <= intervals["high"]
    with pytest.raises(ValueError):
        k.block_interval(results[:4])
    with pytest.raises(ValueError):
        k.release_target(bundle, bundle["slots"][0], 1, 8, True)
    with pytest.raises(ValueError):
        k.release_target(bundle, bundle["slots"][0], 1, 9, False)


def test_authentication_and_absence(tmp_path):
    """SCENARIO-REPORT-8407-CUSTODY: current task and exact upstream are authenticated."""
    work = e.measure(tmp_path / "absent", tmp_path / "raw", tmp_path)
    value = e.build(work, [dict(passed=True)])
    assert value["verdict_class"] == "blocked" and value["python_cost_ready_score"] == 0
    assert len(value["repeat_rows"]) == 135
    assert value["completed_count"] == 0 and value["censored_count"] == value["intended_count"]
    assert any(g["observed"] is None for g in value["gate_check_summary"])
    assert e.build(work, [dict(passed=False)])["verdict_class"] == "disqualified"
    path = tmp_path / (e.NAME + ".json")
    atomic_json(path, value)
    assert e.replay(path)
    value["completed_count"] += 1
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(path, value)
    assert not e.replay(path) and not e.replay(tmp_path / "missing")


def test_real_validation_controls_and_runner(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8407-CUSTODY: real CLI controls and consumers check a blocked receipt."""
    monkeypatch.setenv(
        "COVERAGE_FILE", os.environ.get("COVERAGE_FILE", str(tmp_path / ".coverage"))
    )
    monkeypatch.setenv(
        "COVERAGE_RCFILE", os.environ.get("COVERAGE_RCFILE", str(tmp_path / "coverage.ini"))
    )
    work = e.measure(tmp_path / "missing-root", tmp_path / "raw", tmp_path)
    path = tmp_path / (e.NAME + ".json")
    atomic_json(path, e.build(work, [dict(passed=True)]))
    for name, args, expected in [
        ("cold", ["--cold-replay", str(path)], 0),
        ("missing", ["--cold-replay", str(tmp_path / "missing")], 1),
        ("error", ["--deliberate-error"], 1),
        ("date", ["--date", "20261010"], 2),
    ]:
        receipt = child(
            name,
            [sys.executable, "-u", str(e.ROOT / e.CLI), *args],
            tmp_path / "cli",
            expected=expected,
        )
        assert receipt["passed"]
    assert runner.validate(path, tmp_path / "validators")["passed"]
    assert runner.main(["--deliberate-error"]) == 1
    assert runner.main(["--cold-replay", str(path)]) == 0
    assert runner.main(["--cold-replay", str(tmp_path / "missing")]) == 1
    private = tmp_path / "plan"
    private.mkdir()
    frozen = runner.plan(private)
    assert any(r["name"] == "private_E2E020" for r in frozen)
    assert not any(r["name"] == "private_E2E021" for r in frozen)

    def tiny_plan(folder):
        """SCENARIO-VERIFY-8407-VALIDATION: produce genuine child coverage for the private main path."""
        config = folder / "coverage.ini"
        config.write_text(
            "[run]\nparallel=true\npatch=subprocess\ndata_file="
            + str(folder / ".coverage")
            + "\ninclude=\n"
            + "".join("    " + str(e.ROOT / name) + "\n" for name in e.OWNED)
        )
        monkeypatch.setenv("COVERAGE_FILE", str(folder / ".coverage"))
        monkeypatch.setenv("COVERAGE_RCFILE", str(config))
        coverage = str(e.ROOT / ".venv/bin/coverage")
        return [
            dict(
                name="real_child_coverage",
                argv=[
                    coverage,
                    "run",
                    "--rcfile=" + str(config),
                    str(e.ROOT / e.CLI),
                    "--deliberate-error",
                ],
                expected=1,
                deadline=60,
                scope="owned",
            ),
            dict(
                name="coverage_combine",
                argv=[coverage, "combine", "--rcfile=" + str(config), "--keep"],
                expected=0,
                deadline=60,
                scope="owned",
            ),
            dict(
                name="coverage_json",
                argv=[
                    coverage,
                    "json",
                    "--rcfile=" + str(config),
                    "-o",
                    str(folder / "coverage.json"),
                ],
                expected=0,
                deadline=60,
                scope="owned",
            ),
        ]

    monkeypatch.setattr(runner, "plan", tiny_plan)
    output = tmp_path / "published" / (e.NAME + ".json")
    assert runner.main(["--root", str(tmp_path / "absent"), "--output", str(output)]) == 0
    terminal = json.loads(output.read_bytes())
    assert terminal["verdict_class"] == "blocked"
    assert Path(terminal["terminal_validation_sidecar_path"]).is_file()
    monkeypatch.setattr(
        runner, "plan", lambda p: (_ for _ in ()).throw(ValueError("deliberate_owned_failure"))
    )
    assert runner.main(["--output", str(tmp_path / "error.json")]) == 1


def test_actual_measure_loop_and_source_tamper(tmp_path, monkeypatch):
    """REQ-REPORT-8407: exercise discarded warmup plus five real owned timing blocks."""
    monkeypatch.setattr(k, "CELLS", [dict(workload="natural", batch=1, predictions_per_update=8)])
    work = e.measure(e.ROOT, tmp_path / "raw", tmp_path)
    assert not work["gate_check_summary"] and len(work["results"]) == 6
    work["owned_coverage"] = dict(percent_covered=100)
    atomic_json(Path(work["work_path"]), work)
    value = e.build(work, [dict(passed=True)])
    path = tmp_path / (e.NAME + ".json")
    atomic_json(path, value)
    assert e.replay(path)
    value["fsync_count"] += 1
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(path, value)
    assert not e.replay(path)
    work["bundle"]["head"]["coefficients"][2] += 0.1
    atomic_json(Path(work["work_path"]), work)
    value["work_reference"] = e.reference(Path(work["work_path"]))
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(path, value)
    assert not e.replay(path)


def test_invalid_prediction_and_durable_corruption(bundle, tmp_path):
    """SCENARIO-VERIFY-8407-DELAY: invalid features cannot manufacture a coefficient update."""
    bundle["slots"][0]["x"] = [0.0, 9.0, 0.0, 0.0, 0.0]
    result = k.run_repeat(
        bundle, dict(workload="natural", batch=8, predictions_per_update=1), 1, tmp_path
    )
    assert result["rows"][0]["status"] == "failed"
    assert k.reduce_repeat(bundle, result)["failed_count"] == 1
    event = dict(kind="feedback", tick=8, issued_tick=1, cycle=0)
    state = k.s.initial(bundle["head"])
    with pytest.raises(ValueError):
        k.step(bundle, result["cell"], state, event, "bad")
    with pytest.raises(ValueError):
        k.run_repeat(bundle, dict(workload="invalid"), 9, tmp_path)
    with pytest.raises(ValueError):
        k.reduce_repeat(bundle, dict(result, repeat=9))
    with pytest.raises(ValueError):
        k.reduce_repeat(bundle, dict(result, initial_state={}))
    pointer = tmp_path / "store/current.json"
    atomic_json(pointer, dict(file="../outside"))
    with pytest.raises(ValueError):
        k.read_snapshot(tmp_path / "store")
    atomic_json(
        pointer, dict(file=Path(result["transactions"][-1]["snapshot_path"]).name, sha256="changed")
    )
    with pytest.raises(ValueError):
        k.read_snapshot(tmp_path / "store")


@pytest.mark.parametrize(
    "mutation",
    ["transaction", "stages", "snapshot", "final", "row_clock", "bytes", "repeat_id", "groups"],
)
def test_durable_clock_and_batch_tamper(bundle, tmp_path, mutation):
    """SCENARIO-VERIFY-8407-DELAY: self-consistent counters cannot replace fixed durable groups."""
    result = k.run_repeat(
        bundle, dict(workload="natural", batch=32, predictions_per_update=1), 1, tmp_path
    )
    tx = result["transactions"][0]
    if mutation == "transaction":
        tx["start_ns"] = 0
    if mutation == "stages":
        tx["stages"] = []
    if mutation == "snapshot":
        tx["snapshot_sha256"] = "changed"
    if mutation == "final":
        result["final_state"]["pending"] = [999]
    if mutation == "row_clock":
        result["rows"][0]["start_ns"] = 0
    if mutation == "bytes":
        tx["bytes"] += 1
    if mutation == "repeat_id":
        result["repeat_id"] = "forged"
    if mutation == "groups":
        result["transactions"][0]["request_ids"].reverse()
    with pytest.raises(ValueError):
        k.reduce_repeat(bundle, result)


def test_resource_and_repeat_deadline_failures(bundle, tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8407-VALIDATION: unavailable tools and exhausted time cannot qualify."""
    monkeypatch.setattr(e.os, "access", lambda *a: False)
    monkeypatch.setattr(e.shutil, "disk_usage", lambda *a: type("Disk", (), {"free": 0})())
    work = e.measure(tmp_path / "absent", tmp_path / "raw", tmp_path)
    assert any(g["check"] == "executable_available" for g in work["gate_check_summary"])
    with monkeypatch.context() as m:
        ticks = iter([1, 2, 700_000_000_001])
        m.setattr(k.time, "monotonic_ns", lambda: next(ticks))
        with pytest.raises(TimeoutError):
            k.run_repeat(
                bundle,
                dict(workload="natural", batch=32, predictions_per_update=1),
                1,
                tmp_path / "timeout",
            )


def test_cold_reduction_file_and_long_child(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8407-CUSTODY: independent reductions retain their own bounded child output."""
    work = e.measure(tmp_path / "absent", tmp_path / "raw", tmp_path)
    path = tmp_path / (e.NAME + ".json")
    atomic_json(path, e.build(work, [dict(passed=True)]))
    reduced = tmp_path / "cold.json"
    assert e.replay(path, reduced)
    assert json.loads(reduced.read_bytes())["intended_count"] == 27360
    assert not e.replay(tmp_path / "absent", reduced)
    receipts = []

    def observed(name, argv, logs, **kwargs):
        receipts.append(kwargs)
        return dict(passed=True)

    monkeypatch.setattr(runner, "base_child", observed)
    runner.bounded_child("cold", ["python", "--cold-replay", str(path)], tmp_path, deadline=60)
    runner.bounded_child("rows", ["python", "rows.py"], tmp_path, deadline=60)
    assert [r["deadline"] for r in receipts] == [600, 60]


def test_remaining_clock_and_shard_controls(bundle, tmp_path):
    """SCENARIO-VERIFY-8407-VALIDATION: default clocks, lost groups and reclassified warmups fail."""
    assert len(k.batches([dict(kind="prediction", tick=1)], 32)) == 1
    result = k.run_repeat(
        bundle, dict(workload="natural", batch=8, predictions_per_update=1), 0, tmp_path / "repeat"
    )
    for field, change in [("start_ns", 1), ("transactions", [])]:
        bad = dict(result, **{field: change})
        with pytest.raises(ValueError):
            k.reduce_repeat(bundle, bad)
    path = tmp_path / "primitive.json"
    atomic_json(path, result)
    work = dict(results=[dict(e.reference(path), warmup=False)], bundle=bundle)
    with pytest.raises(ValueError):
        e.reduction(work)
    work = e.measure(tmp_path / "absent", tmp_path / "raw", tmp_path)
    work["coverage_shards"] = [e.reference(path)]
    atomic_json(Path(work["work_path"]), work)
    value = e.build(work, [dict(passed=True)])
    candidate = tmp_path / (e.NAME + ".json")
    value["reproducibility_checksum"] = "invalid"
    atomic_json(candidate, value)
    assert not e.replay(candidate)


def test_replay_uses_frozen_authority_and_rejects_label_identity(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8407-CUSTODY: future roadmap bytes cannot replace the invocation's authority."""
    work = e.measure(tmp_path / "absent", tmp_path / "raw", tmp_path)
    work["bundle"] = dict(head={}, slots=[], labels=dict(sha256="original"))
    atomic_json(Path(work["work_path"]), work)
    value = e.build(work, [dict(passed=True)])
    candidate = tmp_path / (e.NAME + ".json")
    atomic_json(candidate, value)
    monkeypatch.setattr(
        e, "authenticate", lambda *a: dict(head={}, slots=[], labels=dict(sha256="changed"))
    )
    assert not e.replay(candidate)


def test_claimed_coverage_requires_exact_report(tmp_path):
    """SCENARIO-VERIFY-8407-VALIDATION: a claimed percentage alone cannot authorize coverage."""
    work = e.measure(tmp_path / "absent", tmp_path / "raw", tmp_path)
    work["owned_coverage"] = dict(percent_covered=100)
    value = e.build(work, [dict(passed=True)])
    assert value["acceptance_gates"]["coverage"] is False
    assert work["preconditions_checked"]["private_permissions"]
    assert work["preconditions_checked"]["disk_backed"]
    assert work["preconditions_checked"]["tool_versions"]["coverage"]
