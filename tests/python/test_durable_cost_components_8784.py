"""REQ-REPORT-8784, REQ-VERIFY-8784: clocks must not invent complete costs."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from carnot.reporting import durable_cost_components_8784 as e
from carnot.reporting import durable_component_clocks_8784 as clocks
from carnot.reporting import durable_component_runner_8784 as runner


def test_roster_and_identity():
    """REQ-REPORT-8784: six matched windows retain all three arms and conditions."""
    rows = e.roster()
    assert len(rows) == 36 and rows == e.roster()
    assert sum(r["updates"] for r in rows) == 1152
    assert {r["offset"] for r in rows} == {0, 2032, 4064}
    assert {r["condition"] for r in rows} == {"plain", "instrumented"}
    assert e.descriptor("now")["task_id"] == "exp8784-durable-cost-components"
    assert not e.projection([10**11] * 4, 5 * 10**11)["admitted"]
    assert e.projection([10**6] * 4, 10**8)["admitted"]


def test_exclusive_clocks_and_rejection():
    """SCENARIO-REPORT-8784-COMPONENTS: nesting is subtracted, gaps stay unassigned."""
    result = clocks.partition([("serialization", 10, 80), ("file_sync", 20, 50)], 0, 100)
    assert result["costs_ns"] == {"serialization": 40, "file_sync": 30}
    assert result["unassigned_ns"] == 30
    assert result["reconciliation_error_fraction"] == 0
    with pytest.raises(ValueError, match="clock_span"):
        clocks.partition([("bad", 20, 10)], 0, 100)
    with pytest.raises(ValueError, match="clock_overlap"):
        clocks.partition([("a", 0, 50), ("b", 20, 70)], 0, 100)


def test_overhead_gates_keep_complete_costs():
    """SCENARIO-REPORT-8784-COMPONENTS: one slow pair rejects all attribution."""
    rows = [
        dict(r, status="completed", complete_ns=100, update_rows=[], segments=[])
        for r in e.roster()
    ]
    next(r for r in rows if r["condition"] == "instrumented")["complete_ns"] = 200
    reduced = e.reduce(rows)
    assert len(reduced["observer_overhead_rows"]) == 18
    assert not reduced["component_ready"]
    assert len(reduced["process_pair_rows"]) == 6
    assert len(reduced["per_window_rows"]) == 36
    assert not e.reduce([])["component_ready"]


def test_cli_bad_inputs(tmp_path):
    """REQ-VERIFY-8784: invalid commands cannot reach production publication."""
    assert runner.main(["--date", "bad"]) == 1
    assert runner.main(["--worker", str(tmp_path / "missing")]) == 1
    with pytest.raises(ValueError, match="private_output"):
        runner.run(e.ROOT / "results" / (e.NAME + ".json"), "20261104", private=True)


@pytest.fixture(scope="module")
def packet(tmp_path_factory):
    """REQ-VERIFY-8784: build independent private arithmetic operands, never read primaries."""
    root = tmp_path_factory.mktemp("components-operands")
    numeric = clocks.numeric.numeric
    job = dict(
        head=numeric.original.legacy.head(),
        boundary=False,
        boundary_offset=0,
        vectors=[dict(x=[0.4] * 9)],
        labels=[0, 1],
    )
    f = numeric.Fixed(32, 24)
    state = numeric.encode(job["head"], f)
    trace, starts = [], []
    ack = e.canonical_hash(dict(job_hash=e.canonical_hash(job), next_index=0, state=state))
    for index in range(4097):
        if index in [0, 2032, 4064]:
            path = root / f"start-{index}.json"
            e.atomic_json(
                path,
                dict(
                    state=state,
                    counts=f.counts,
                    operations=f.operations,
                    maximum=f.maximum,
                    ack=ack,
                    next_index=index,
                ),
            )
            starts.append(dict(offset=index, reference=e.reference(path)))
        if index == 4096:
            break
        state, receipt = numeric.step(state, job["vectors"][0]["x"], job["labels"][index % 2], f)
        p, _ = numeric.probability(state, job["vectors"][0]["x"], f)
        ack = e.canonical_hash(dict(previous_ack=ack, next_index=index + 1, state=state))
        trace.append(
            dict(
                update_index=index + 1,
                state_hash=e.canonical_hash(state),
                probability=p,
                dense_writes=receipt["renormalization_writes"],
                passed=True,
                ack=ack,
                counters_identical=True,
            )
        )
    extension = Path(clocks.numeric.resident.DEFAULT_EXTENSION)
    value = dict(
        private_roster=[
            r
            for r in e.roster()
            if r["repeat"] == 0 and r["offset"] == 2032 and r["condition"] == "instrumented"
        ],
        failures=[],
        refs=[],
        job=job,
        trace=trace,
        historical_probe={},
        trace_reference=e.reference(root / "start-0.json"),
        extension=e.reference(extension),
        qualified=dict(ordinary_starting_state_rows=starts),
    )
    path = root / "packet.json"
    e.atomic_json(path, value)
    return path


def test_worker_real_restart_all_arms(packet, tmp_path):
    """REQ-REPORT-8784: direct clocks preserve exactly the same acknowledged sequence."""
    admitted = e.read(e.reference(packet))
    for arm in e.ARMS:
        for condition in ["plain", "instrumented"]:
            spec = dict(
                unit_id=arm + condition,
                repeat=0,
                offset=0,
                arm=arm,
                condition=condition,
                updates=32,
            )
            result = runner.cell(spec, admitted, tmp_path)
            assert result["passed"] and result["restart_verified"]
            assert len(result["update_rows"]) == 32
            assert all(u["passed"] for u in result["update_rows"])
            spans = [s for segment in result["segments"] for s in segment["spans"]]
            assert bool(spans) == (condition == "instrumented")
            assert all(s["components"]["unassigned_ns"] >= 0 for s in result["segments"])


@pytest.fixture(scope="module")
def private(packet, tmp_path_factory):
    """SCENARIO-VERIFY-8784-REPLAY: the real CLI receives private input-only fixtures."""
    root = tmp_path_factory.mktemp("components-cli")
    output = root / (e.NAME + ".json")
    receipt = runner.child(
        "private_cli",
        [sys.executable, "-u", e.CLI, "--private-input", str(packet), "--output", str(output)],
        root / "logs",
        deadline=450,
    )
    assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
    return output


def test_private_cli_and_replay(private, tmp_path):
    """SCENARIO-VERIFY-8784-REPLAY: summary and identity tampering cannot self-authenticate."""
    value = json.loads(private.read_bytes())
    assert value["completed_count"] == 3 and len(value["per_update_rows"]) == 96
    assert value["durable_cost_rows_ready_score"] == 0
    assert runner.main(["--cold-replay", str(private)]) == 0
    changed = tmp_path / "changed.json"
    e.atomic_json(changed, dict(value, task_id="exp8784-alias"))
    with pytest.raises(ValueError, match="producer_identity"):
        runner.replay(changed)
    e.atomic_json(changed, dict(value, durable_cost_rows_ready_score=1))
    with pytest.raises(ValueError, match="report_drift"):
        runner.replay(changed)


def test_build_readiness_and_failures(private):
    """REQ-REPORT-8784: admission concerns complete execution, never an observed gain."""
    value = json.loads(private.read_bytes())
    work = e.read(value["primitive_reference"])
    work["private_control"] = False
    source = {r["arm"]: r for r in work["window_rows"]}
    work["roster"] = e.roster()
    work["window_rows"] = [dict(deepcopy(source[r["arm"]]), **r) for r in e.roster()]
    receipts = [dict(name="coverage_report", passed=True)]
    result = e.build(work, receipts, private)
    assert result["durable_cost_rows_ready_score"] == 1
    assert result["verdict_class"] == "null"
    assert e.build(work, [], private)["verdict_class"] == "disqualified"
    work["failures"] = [e.failure(e.reference(private), "upstream", 1, 0)]
    assert e.build(work, receipts, private)["verdict_class"] == "blocked"


@pytest.mark.parametrize("mode", ["valid", "missing", "field", "zero", "alias", "terminal"])
def test_private_upstream_gates(mode, tmp_path, monkeypatch):
    """REQ-VERIFY-8784: private producer copies preserve exact missingness diagnostics."""
    producer = dict(
        experiment_id=8783,
        task_id=e.identity.TASK,
        durable_producer_ready_score=1,
        required_checks_passed=True,
        flagged_adversarial=False,
        terminal_validation_sidecar_path=str(tmp_path / "terminal.json"),
    )
    if mode == "field":
        producer.pop("durable_producer_ready_score")
    if mode == "zero":
        producer["durable_producer_ready_score"] = 0
    if mode == "alias":
        producer["task_id"] = "exp8783-alias"
    path = tmp_path / "upstream.json"
    e.atomic_json(path, producer)
    pin = e.reference(path)["sha256"]
    e.atomic_json(
        tmp_path / "terminal.json",
        dict(passed=mode != "terminal", publication=dict(primary_sha256=pin)),
    )
    monkeypatch.setattr(e, "UPSTREAM", path)
    monkeypatch.setattr(e, "PIN", pin)
    monkeypatch.setattr(e.identity, "inputs", lambda: dict(failures=[], refs=[]))
    if mode == "missing":
        path.unlink()
    result = e.inputs()
    assert bool(result["failures"]) == (mode != "valid")
    if result["failures"]:
        assert all(
            k in result["failures"][0]
            for k in [
                "upstream",
                "path",
                "sha256",
                "check",
                "field",
                "operator",
                "expected",
                "observed",
            ]
        )


def test_failures_keep_intended_updates(packet, tmp_path, monkeypatch):
    """REQ-REPORT-8784: a failed real child remains a failed window with missing costs."""
    admitted = e.read(e.reference(packet))
    spec = dict(unit_id="failed", repeat=0, offset=0, arm="python", condition="plain", updates=32)
    monkeypatch.setattr(runner, "child", lambda *a, **kw: dict(passed=False))
    row = runner.cell(spec, admitted, tmp_path)
    assert row["status"] == "failed" and len(row["update_rows"]) == 32
    assert row["complete_ns"] is None and not row["restart_verified"]


def test_worker_rejects_changed_binary_and_checkpoint(packet, tmp_path, monkeypatch):
    """REQ-VERIFY-8784: hash drift and a different durable read cannot qualify."""
    admitted = e.read(e.reference(packet))
    request = e.historical.request(admitted, 0, "json_native")
    request["extension"]["sha256"] = "changed"
    with pytest.raises(ValueError, match="loaded_binary_identity"):
        runner.worker(request, tmp_path / "binary")
    request = e.historical.request(admitted, 0, "python", 1)
    real_read = clocks.numeric.store_module.Store.read
    monkeypatch.setattr(
        clocks.numeric.store_module.Store, "read", lambda self: dict(real_read(self), changed=True)
    )
    with pytest.raises(ValueError, match="post_restart_verification"):
        runner.worker(request, tmp_path / "checkpoint")


def test_public_orchestration_health_and_missing_fixture(private, tmp_path, monkeypatch):
    """REQ-VERIFY-8784: full repository health cannot masquerade as scoped acceptance."""
    with pytest.raises(ValueError, match="private_input_required"):
        runner.run(tmp_path / "bad.json", "20261104", private=True)
    value = json.loads(private.read_bytes())
    work = e.read(value["primitive_reference"])
    work["private_control"] = False
    source = {r["arm"]: r for r in work["window_rows"]}
    work["roster"] = e.roster()
    work["window_rows"] = [dict(deepcopy(source[r["arm"]]), **r) for r in e.roster()]
    monkeypatch.setattr(runner, "measure", lambda *a: deepcopy(work))
    monkeypatch.setattr(runner, "seal", lambda *a: None)
    monkeypatch.setattr(
        runner.runtime, "validate", lambda *a: [dict(name="coverage_report", passed=True)]
    )
    monkeypatch.setattr(
        runner, "child", lambda *a, **kw: dict(passed=False, scope="repository_health")
    )
    monkeypatch.setattr(runner.runtime, "terminal", lambda *a: dict(passed=True))
    path = tmp_path / (e.NAME + ".json")
    assert runner.run(path, "20261104") == 0
    published = json.loads(path.read_bytes())
    assert not published["repository_health_reference"]["full_suite_pass_claimed"]
    assert published["durable_cost_rows_ready_score"] == 1
    monkeypatch.setattr(runner, "run", lambda *a, **kw: 1)
    assert runner.main(["--output", str(tmp_path / "other.json")]) == 1


def test_corrupt_segment_and_instrumentation(packet, tmp_path, monkeypatch):
    """REQ-VERIFY-8784: counters and stored state must be checked before a window passes."""
    admitted = e.read(e.reference(packet))
    spec = dict(unit_id="corrupt", repeat=0, offset=0, arm="python", condition="plain", updates=32)
    monkeypatch.setattr(runner, "child", lambda *a, **kw: dict(passed=True))
    monkeypatch.setattr(
        e, "read", lambda ref: dict(updates=15, durable_commits=30, fsync_calls=120)
    )
    with pytest.raises(ValueError, match="durable_window"):
        runner.cell(spec, admitted, tmp_path)


def test_blocked_private_measurement(tmp_path, monkeypatch):
    """REQ-VERIFY-8784: absent external authority produces terminal blocked rows, not partial."""
    packet = tmp_path / "blocked-input.json"
    e.atomic_json(
        packet,
        dict(
            failures=[
                e.failure(
                    dict(path="/private/absent", sha256=None),
                    "durable_producer_ready_score",
                    1,
                    None,
                    "missing_file",
                )
            ],
            refs=[],
        ),
    )
    # Collection itself is covered by the real private CLI; this control isolates external absence.
    monkeypatch.setattr(
        runner.runtime,
        "child",
        lambda *a, **kw: dict(passed=True, stdout_path=str(tmp_path / "nodes")),
    )
    (tmp_path / "nodes").write_text(
        "tests/python/test_durable_cost_components_8784.py::test_roster_and_identity\n"
    )
    work = runner.measure(tmp_path / "blocked-work", "20261104", packet)
    assert len(work["window_rows"]) == 36
    for row in work["window_rows"]:
        row.setdefault("restart_verified", False)
    e.atomic_json(Path(work["raw"]) / "validation.json", dict(receipts=[]))
    value = e.build(
        work, [dict(name="private_control_only", passed=True)], tmp_path / (e.NAME + ".json")
    )
    assert value["verdict_class"] == "blocked" and value["censored_count"] == 36
