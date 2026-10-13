"""REQ-VERIFY-8494, REQ-REPORT-8494: matched durable work and honest publication."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting import append_service_cost_8494 as r
from carnot.verify import append_service_cost_8494 as k


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    """SCENARIO-VERIFY-8494-PAIRS: real fresh children retain original sealed work."""
    private = tmp_path_factory.mktemp("append8494")
    value = r.read(r.reference(r.ROOT / r.upstream.UPSTREAM))
    tape = r.read(value["transition_fixture"]["tape_references"][1])
    atomic_json(private / "frozen.json", tape)
    rows = []
    for arm in ("snapshot", "append"):
        proc = subprocess.run(
            [
                sys.executable,
                "-u",
                str(r.ROOT / r.CLI),
                "--worker",
                str(private / "frozen.json"),
                "--store",
                str(private / arm),
                "--arm",
                arm,
            ],
            capture_output=True,
            timeout=120,
        )
        assert proc.returncode == 0, proc.stderr.decode()
        assert proc.stdout.decode().count("phase=request_after_") == len(tape["transactions"])
        rows.append(json.loads((private / arm / "primitive.json").read_bytes()))
    return tape, rows


def test_equal_complete_work(measured):
    """SCENARIO-VERIFY-8494-PAIRS: predictions and feedback retain exact durable accounting."""
    tape, rows = measured
    a, b = [k.reduce(tape, row) for row in rows]
    assert a["decision_attempts"] == b["decision_attempts"] == 64
    assert a["applied_updates"] == b["applied_updates"] == 7
    assert a["scheduled_feedback"] == b["scheduled_feedback"] == 8
    assert a["nonapplicable_feedback"] == b["nonapplicable_feedback"] == 1
    assert a["pending"] == b["missing"] == b["lost"] == 0
    assert b["file_fsyncs"] > 0 and a["directory_fsyncs"] > 0
    assert a["wall_s"] > a["cold_reconstruction_s"] > 0
    assert k.pair(tape, *rows)["accounting_equality"]


@pytest.mark.parametrize("field", ["final", "ack", "request", "clock", "sync", "arm", "cell"])
def test_primitive_rejects(measured, field):
    """SCENARIO-VERIFY-8494-REJECTION: repaired summary hashes cannot replace operands."""
    tape, rows = measured
    row = deepcopy(rows[1])
    if field == "final":
        row["final"]["state"]["release_cursor"] += 1
    elif field == "ack":
        row["acknowledgments"].pop()
    elif field == "request":
        row["requests"][0]["end_ns"] = row["end_ns"] + 1
    elif field == "clock":
        row["cold_end_ns"] = row["end_ns"] + 1
    elif field == "sync":
        row["file_fsyncs"] = 0
    elif field == "arm":
        row["arm"] = "other"
    else:
        row["cell"]["batch"] = 32
    with pytest.raises(ValueError):
        k.reduce(tape, row)


def test_bootstrap():
    """SCENARIO-VERIFY-8494-PAIRS: use the sealed seed and whole repeat indices."""
    pairs = [dict(snapshot_wall_s=2 + i, journal_wall_s=1 + i) for i in range(5)]
    result = k.interval(pairs)
    ids = np.random.default_rng(7298480).integers(0, 5, (10000, 5))
    samples = np.array([2 + i for i in range(5)])[ids].sum(1) / np.array([1 + i for i in range(5)])[
        ids
    ].sum(1)
    assert result["speedup_ci95"] == np.quantile(samples, [0.025, 0.975]).tolist()
    with pytest.raises(ValueError):
        k.interval(pairs[:4])


def test_invalid_work(measured, tmp_path):
    """SCENARIO-VERIFY-8494-REJECTION: unregistered arms and cells never enter measurement."""
    tape, _ = measured
    with pytest.raises(ValueError):
        k.run_repeat(tape, tmp_path / "bad", "other")
    bad = deepcopy(tape)
    bad["cell"]["batch"] = 32
    with pytest.raises(ValueError):
        k.run_repeat(bad, tmp_path / "bad", "append")


def test_missing_cli(tmp_path):
    """SCENARIO-REPORT-8494-TERMINAL: private absence is blocked, never an inferred null."""
    output = tmp_path / (r.NAME + ".json")
    proc = subprocess.run(
        [
            sys.executable,
            "-u",
            str(r.ROOT / r.CLI),
            "--private-control",
            "--root",
            str(tmp_path / "missing"),
            "--output",
            str(output),
        ],
        capture_output=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr.decode()
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked" and value["censored_count"] == 24
    assert value["primary_speedup_ci95"] is None and value["MODEL_SPECS"] == []
    assert r.replay(output)
    value["service_cost_ready_score"] = 999
    value["reproducibility_checksum"] = r.checksum(value)
    atomic_json(output, value)
    assert not r.replay(output)


def test_operations_and_pair_rejects(measured):
    """SCENARIO-VERIFY-8494-REJECTION: sync summaries require original operation operands."""
    tape, rows = measured
    changed = deepcopy(rows[1])
    changed["operation_rows"].pop()
    with pytest.raises(ValueError, match="operation_accounting"):
        k.reduce(tape, changed)
    with pytest.raises(ValueError, match="paired_semantics"):
        k.pair(tape, rows[1], rows[0])


@pytest.mark.parametrize("failure", ["none", "score", "consumer", "code", "replay", "missing"])
def test_admission_rejects(tmp_path, monkeypatch, failure):
    """SCENARIO-REPORT-8494-TERMINAL: qualified scores cannot replace producer bytes."""
    original = r.read(r.reference(r.ROOT / r.UPSTREAM))
    monkeypatch.setattr(r, "ORIGINAL_ADMISSION", lambda *args: dict(refs=[]))
    reader = r.read

    def read(ref):
        if ref["path"] == str(r.ROOT / r.UPSTREAM):
            return deepcopy(original)
        return reader(ref)

    monkeypatch.setattr(r, "read", read)
    monkeypatch.setattr(r, "child", lambda *args, **kwargs: dict(passed=failure != "replay"))
    if failure == "score":
        original["append_store_ready_score"] = 0
    elif failure == "consumer":
        monkeypatch.setattr(r, "reader_receipt", lambda *args, **kwargs: dict(passed=False))
    elif failure == "code":
        original["code_config_hashes"][0]["sha256"] = "forged"
    if failure == "none":
        inputs = r.admission(r.ROOT, tmp_path)
        assert inputs["store_receipt"]["passed"]
        assert inputs["receipt"]["store_qualification_receipt"]["passed"]
    else:
        with pytest.raises(r.upstream.GateFailure) as caught:
            r.admission(tmp_path if failure == "missing" else r.ROOT, tmp_path)
        assert caught.value.row["upstream"] == "exp8493-append-store-qualification"
        assert caught.value.row["missing"] is (failure == "missing")


def test_roster_failure_and_deadline(measured, tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8494-REJECTION: every intended arm survives a failure or timeout."""
    tape, rows = measured
    monkeypatch.setattr(
        r,
        "admission",
        lambda *args: dict(
            authority={},
            refs=[],
            tapes=[tape, tape],
            upstream=dict(historical_model_provenance=[], preserved_v728={}),
            receipt=dict(passed=True),
        ),
    )
    reduce = k.reduce
    monkeypatch.setattr(k, "reduce", lambda frozen, row: reduce(tape, row))

    def child(name, argv, logs, **kwargs):
        arm = argv[-1]
        target = Path(argv[argv.index("--store") + 1])
        atomic_json(target / "primitive.json", rows[arm == "append"])
        return dict(passed=True)

    monkeypatch.setattr(r, "child", child)
    happy = r.measure(r.ROOT, tmp_path / "happy")
    assert len(happy["rows"]) == len(happy["primitives"]) == 24
    assert [row["arm"] for row in happy["rows"][:4]] == ["snapshot", "append", "append", "snapshot"]
    monkeypatch.setattr(r, "child", lambda *args, **kwargs: dict(passed=False))
    failed = r.measure(r.ROOT, tmp_path / "failed")
    assert len(failed["owned_failures"]) == 24
    clock = [r.time.monotonic_ns()]

    def tick():
        clock[0] += 1801 * 10**9
        return clock[0]

    monkeypatch.setattr(r.time, "monotonic_ns", tick)
    expired = r.measure(r.ROOT, tmp_path / "expired")
    assert all(row["missing_reason"] == "measurement_deadline" for row in expired["rows"])


def test_primary_gates(measured, tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8494-PAIRS: secondary gains cannot rescue primary or incomplete work."""
    tape, primitives = measured
    work = r.measure(tmp_path / "missing", tmp_path / "work")
    work["gate_check_summary"] = []
    for row in work["rows"]:
        row.update(
            status="completed",
            missing_reason=None,
            primitive_reference=r.reference(
                Path(primitives[row["arm"] == "append"]["directory"]) / "primitive.json"
            ),
        )
    atomic_json(Path(work["raw"]) / "measurement.json", work)
    receipts = [dict(name=name, passed=True) for name in r.REQUIRED]
    monkeypatch.setattr(r.legacy, "coverage_ready", lambda report: True)
    original_pair = k.pair

    def pair(*args):
        result = original_pair(*args)
        result["journal"]["p95_s"] = result["snapshot"]["p95_s"]
        return result

    monkeypatch.setattr(k, "pair", pair)
    monkeypatch.setattr(
        k, "interval", lambda blocks: dict(pooled_speedup=12.0, speedup_ci95=[10.0, 14.0])
    )
    positive = r.build(work, receipts, Path(work["raw"]), tmp_path / "candidate.json")
    assert positive["verdict_class"] == "circular_positive"
    assert positive["append_speed_benefit_score"] == positive["tenfold_service_score"] == 1
    monkeypatch.setattr(
        k, "interval", lambda blocks: dict(pooled_speedup=1.0, speedup_ci95=[0.9, 1.1])
    )
    null = r.build(work, receipts, Path(work["raw"]), tmp_path / "candidate.json")
    assert null["service_cost_ready_score"] == 1 and null["append_speed_benefit_score"] == 0
    work["rows"][0]["status"] = "censored"
    incomplete = r.build(work, receipts, Path(work["raw"]), tmp_path / "candidate.json")
    assert incomplete["primary_speedup_ci95"] is incomplete["secondary_speedup_ci95"] is None
    assert incomplete["service_cost_ready_score"] == 0


@pytest.fixture
def replay_work(measured, tmp_path):
    """SCENARIO-REPORT-8494-TERMINAL: real secondary arms occupy their exact registered slots."""
    tape, primitives = measured
    work = r.measure(tmp_path / "missing", tmp_path / "raw")
    work.update(tapes=[tape, tape], gate_check_summary=[])
    for index, primitive in zip((15, 14), primitives, strict=True):
        row = work["rows"][index]
        row.update(
            status="completed",
            missing_reason=None,
            primitive_reference=r.reference(Path(primitive["directory"]) / "primitive.json"),
            metrics=k.reduce(tape, primitive),
        )
    atomic_json(Path(work["raw"]) / "measurement.json", work)
    return work


@pytest.mark.parametrize(
    "mutation",
    [
        "none",
        "identity",
        "length",
        "authority",
        "tapes",
        "hash",
        "roster",
        "frozen",
        "metrics",
        "store_hash",
        "disk_state",
    ],
)
def test_cold_replay_controls(replay_work, tmp_path, monkeypatch, mutation):
    """SCENARIO-REPORT-8494-TERMINAL: repaired hashes still need independent original work."""
    work = deepcopy(replay_work)
    sealed = deepcopy(work["tapes"])
    monkeypatch.setattr(r, "admission", lambda *args: dict(tapes=sealed))
    value = r.build(work, [], Path(work["raw"]), tmp_path / "candidate.json")
    if mutation == "identity":
        value["experiment_id"] = 1
    elif mutation == "length":
        work["rows"].pop()
    elif mutation == "authority":
        work["authority"].update(activated=True, tasks=[])
    elif mutation == "tapes":
        work["tapes"][0]["cell"]["batch"] = 32
    elif mutation == "hash":
        value["code_config_hashes"][0]["sha256"] = "forged"
    elif mutation == "roster":
        work["rows"][0]["arm"] = "append"
    elif mutation == "metrics":
        work["rows"][14]["metrics"]["wall_s"] += 1
    elif mutation in ("frozen", "store_hash", "disk_state"):
        row = work["rows"][15]
        primitive = deepcopy(r.read(row["primitive_reference"]))
        if mutation == "frozen":
            frozen = r.read(primitive["frozen_reference"])
            frozen["cell"]["batch"] = 1
            atomic_json(tmp_path / "changed_frozen.json", frozen)
            primitive["frozen_reference"] = r.reference(tmp_path / "changed_frozen.json")
        elif mutation == "store_hash":
            primitive["store_references"][0]["sha256"] = "forged"
        else:
            import shutil

            target = tmp_path / "older" / "store"
            shutil.copytree(Path(primitive["directory"]) / "store", target)
            primitive["directory"] = str(target.parent)
            initial = k.base.oracle(sealed[1], 0)
            k.base.Snapshot(target, sealed[1]).store.commit(initial["state"])
            primitive["store_references"] = [
                r.reference(p) for p in sorted(target.iterdir()) if p.is_file()
            ]
        atomic_json(tmp_path / "changed_primitive.json", primitive)
        row["primitive_reference"] = r.reference(tmp_path / "changed_primitive.json")
    atomic_json(Path(work["raw"]) / "measurement.json", work)
    value["work_reference"] = r.reference(Path(work["raw"]) / "measurement.json")
    value["reproducibility_checksum"] = r.checksum(value)
    atomic_json(tmp_path / "candidate.json", value)
    assert r.replay(tmp_path / "candidate.json") is (mutation == "none")


def test_cli_branches(tmp_path):
    """SCENARIO-REPORT-8494-TERMINAL: missing worker arguments and wrong dates reject."""
    from carnot.reporting import append_service_cost_runner_8494 as runner

    assert runner.main(["--deliberate-error"]) == 7
    assert runner.main(["--private-control"]) == 2
    assert runner.main(["--cold-replay", str(tmp_path / "absent")]) == 1
    with pytest.raises(SystemExit):
        runner.main(["--date", "20261012"])
    with pytest.raises(SystemExit):
        runner.main(["--worker", str(tmp_path / "absent")])
