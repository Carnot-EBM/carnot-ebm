"""REQ-VERIFY-8466, REQ-REPORT-8466: complete durable clocks and honest gates."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting.current_work_receipt import atomic_json
from carnot.verify import journal_equivalence_8465 as stores
from carnot.verify import journal_service_cost_8466 as k
from carnot.reporting import journal_service_cost_8466 as r
from test_persistence_localization_8464 import bundle  # noqa: F401


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    """SCENARIO-VERIFY-8466-PAIRS: actual private child clocks use sealed natural work."""
    private = tmp_path_factory.mktemp("service8466")
    deps = r.dependencies(r.ROOT, private, private)
    frozen = stores.freeze(deps["bundle"], k.CELLS[1], 1)
    atomic_json(private / "frozen.json", frozen)
    results = []
    for arm in ("snapshot", "journal"):
        target = private / arm
        proc = subprocess.run(
            [
                sys.executable,
                "-u",
                str(r.ROOT / r.CLI),
                "--worker",
                str(private / "frozen.json"),
                "--store",
                str(target),
                "--arm",
                arm,
            ],
            capture_output=True,
            timeout=120,
        )
        assert proc.returncode == 0, proc.stderr.decode()
        results.append(json.loads((target / "primitive.json").read_bytes()))
    return deps, frozen, results


def test_real_equal_work(measured):
    """SCENARIO-VERIFY-8466-PAIRS: all service clocks charge cold reconstruction and tail."""
    _, frozen, results = measured
    a, b = [k.reduce(frozen, result) for result in results]
    assert a["decision_attempts"] == b["decision_attempts"] == 64
    assert a["feedback_attempts"] == b["feedback_attempts"] == 8
    assert a["pending"] == b["pending"] == 0
    assert a["wall_s"] >= a["cold_reconstruction_s"] > 0
    assert a["bytes"] > 0 and b["fsyncs"] > 0
    assert k.pair(frozen, *results)["accounting_equality"]


@pytest.mark.parametrize(
    "field", ["final", "acknowledgments", "latency", "clock", "fsyncs", "binding"]
)
def test_primitive_tamper(measured, field):
    """SCENARIO-VERIFY-8466-FAILURES: hashes cannot replace original logical operands."""
    _, frozen, results = measured
    changed = deepcopy(results[0])
    if field == "final":
        changed["final"]["state"]["release_cursor"] += 1
    elif field == "acknowledgments":
        changed["acknowledgments"].pop()
    elif field == "latency":
        changed["requests"][0]["end_ns"] = changed["end_ns"] + 1
    elif field == "clock":
        changed["cold_end_ns"] = changed["end_ns"] + 1
    elif field == "fsyncs":
        changed["fsyncs"] = 0
    else:
        changed["cell"]["batch"] = 1
    with pytest.raises(ValueError):
        k.reduce(frozen, changed)


def test_paired_bootstrap():
    """SCENARIO-VERIFY-8466-PAIRS: resample five repeat blocks, never request siblings."""
    pairs = [dict(snapshot_wall_s=2.0, journal_wall_s=1.0) for _ in range(5)]
    result = k.interval(pairs)
    assert result["pooled_speedup"] == 2
    assert result["speedup_ci95"] == [2, 2]
    with pytest.raises(ValueError, match="five"):
        k.interval(pairs[:4])


def test_invalid_arm_and_cell(bundle, tmp_path):  # noqa: F811
    """SCENARIO-VERIFY-8466-FAILURES: unregistered schedules never enter timing."""
    frozen = stores.freeze(bundle, deepcopy(k.CELLS[0]), 0)
    with pytest.raises(ValueError):
        k.run_repeat(frozen, tmp_path / "bad", "other")
    frozen["cell"]["batch"] = 32
    with pytest.raises(ValueError):
        k.run_repeat(frozen, tmp_path / "bad", "journal")


def test_missing_and_cli(tmp_path):
    """SCENARIO-REPORT-8466-TERMINAL: real private publication and fresh replay retain absence."""
    output = tmp_path / (r.NAME + ".json")
    proc = subprocess.run(
        [
            sys.executable,
            "-u",
            str(r.ROOT / r.CLI),
            "--date",
            "20261013",
            "--private-control",
            "--root",
            str(tmp_path / "missing"),
            "--output",
            str(output),
        ],
        capture_output=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr.decode()
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked"
    assert value["censored_count"] == 24
    assert value["speedup_ci95"] is None
    assert value["MODEL_SPECS"] == [] and value["run_date"] == "20261013"
    assert r.replay(output)
    value["service_cost_ready_score"] = 999
    value["reproducibility_checksum"] = r.checksum(value)
    atomic_json(output, value)
    assert not r.replay(output)


def test_measurement_roster_failures_and_deadline(measured, tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8466-FAILURES: admission retains all arms through failures and timeout."""
    deps, frozen, primitives = measured
    monkeypatch.setattr(r, "dependencies", lambda *args: deepcopy(deps))
    original_reduce = k.reduce
    monkeypatch.setattr(k, "reduce", lambda f, row: original_reduce(frozen, row))
    calls = []

    def child(name, argv, logs, **kwargs):
        calls.append(name)
        target = Path(argv[argv.index("--store") + 1])
        arm = argv[-1]
        primitive = deepcopy(primitives[arm == "journal"])
        atomic_json(target / "primitive.json", primitive)
        return dict(passed=True)

    monkeypatch.setattr(r, "child", child)
    happy = r.measure(r.ROOT, tmp_path / "happy", tmp_path)
    assert len(happy["rows"]) == len(happy["primitives"]) == len(calls) == 24
    assert calls[:4] == [
        "cell0-repeat0-snapshot",
        "cell0-repeat0-journal",
        "cell0-repeat1-journal",
        "cell0-repeat1-snapshot",
    ]
    monkeypatch.setattr(r, "child", lambda *args, **kwargs: dict(passed=False))
    failed = r.measure(r.ROOT, tmp_path / "failed", tmp_path)
    assert len(failed["owned_failures"]) == 24
    counter = [r.time.monotonic_ns()]

    def clock():
        counter[0] += 1801 * 10**9
        return counter[0]

    monkeypatch.setattr(r.time, "monotonic_ns", clock)
    deadline = r.measure(r.ROOT, tmp_path / "deadline", tmp_path)
    assert all(row["missing_reason"] == "measurement_deadline" for row in deadline["rows"])


def test_ready_null_and_speed_gates(measured, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8466-TERMINAL: qualification, benefit and tenfold gates remain separate."""
    _, frozen, primitives = measured
    work = r.measure(tmp_path / "missing", tmp_path / "work", tmp_path)
    work["gate_check_summary"] = []
    for arm, primitive in zip(("snapshot", "journal"), primitives, strict=True):
        atomic_json(tmp_path / (arm + ".json"), primitive)
    for row in work["rows"]:
        row.update(
            status="completed",
            primitive_reference=r.reference(tmp_path / (row["arm"] + ".json")),
            missing_reason=None,
        )
    atomic_json(Path(work["raw"]) / "measurement.json", work)
    receipts = [dict(name=name, passed=True) for name in r.REQUIRED]
    monkeypatch.setattr(r, "coverage_ready", lambda report: True)
    value = r.build(work, receipts, tmp_path / "candidate.json")
    assert value["service_cost_ready_score"] == 1
    assert len(value["paired_repeat_rows"]) == 10
    monkeypatch.setattr(
        k, "interval", lambda blocks: dict(pooled_speedup=12.0, speedup_ci95=[10.0, 14.0])
    )
    original_pair = k.pair

    def fast(*args):
        paired = original_pair(*args)
        paired["journal"]["p95_s"] = paired["snapshot"]["p95_s"]
        return paired

    monkeypatch.setattr(k, "pair", fast)
    value = r.build(work, receipts, tmp_path / "candidate.json")
    assert value["verdict_class"] == "circular_positive"
    assert value["journal_speed_benefit_score"] == value["tenfold_service_score"] == 1
    work["adversarial_findings"] = [dict(severity="unknown")]
    assert r.build(work, receipts, tmp_path / "candidate.json")["verdict_class"] == "disqualified"
    with pytest.raises(ValueError, match="paired"):
        original_pair(frozen, primitives[1], primitives[0])
    changed = deepcopy(primitives[0])
    changed["final"]["state"]["release_cursor"] += 1
    atomic_json(tmp_path / "snapshot.json", changed)
    work["adversarial_findings"] = []
    invalid = r.build(work, receipts, tmp_path / "candidate.json")
    assert invalid["service_cost_ready_score"] == 0
    assert invalid["acceptance_gates"]["exact_accounting"] is False


@pytest.mark.parametrize("restart", [False, True])
def test_actual_state_failure(bundle, tmp_path, monkeypatch, restart):  # noqa: F811
    """SCENARIO-VERIFY-8466-FAILURES: final and restart corruption reject real durable writes."""
    frozen = stores.freeze(bundle, deepcopy(k.CELLS[1]), 0)
    frozen["transactions"] = frozen["transactions"][:1]
    original = stores.Snapshot.read
    calls = []

    def changed(self):
        view = original(self)
        calls.append(1)
        if len(calls) == (3 if restart else 2):
            view["state"]["release_cursor"] += 1
        return view

    monkeypatch.setattr(stores.Snapshot, "read", changed)
    with pytest.raises(ValueError, match="restart_state" if restart else "exact_final_state"):
        k.run_repeat(frozen, tmp_path, "snapshot")


def test_direct_cli_controls(tmp_path):
    """SCENARIO-REPORT-8466-TERMINAL: actual CLI branches preserve expected exits and date."""
    from carnot.reporting import journal_service_cost_runner_8466 as runner

    assert runner.main(["--deliberate-error"]) == 7
    assert runner.main(["--cold-replay", str(tmp_path / "absent")]) == 1
    assert runner.main(["--private-control"]) == 2
    with pytest.raises(SystemExit):
        runner.main(["--date", "20261012"])


def test_sealed_measurement_resume(tmp_path):
    """SCENARIO-REPORT-8466-TERMINAL: retry validation without discarding measured repeats."""
    from carnot.reporting import journal_service_cost_runner_8466 as runner

    root = tmp_path / "missing"
    work = r.measure(root, tmp_path / "original", tmp_path)
    atomic_json(tmp_path / "original" / "original_owned_sources.json", dict(rows=[]))
    output = tmp_path / "retry" / (r.NAME + ".json")
    assert (
        runner.main(
            [
                "--resume-work",
                str(tmp_path / "original" / "measurement.json"),
                "--root",
                str(root),
                "--output",
                str(output),
                "--private-control",
            ]
        )
        == 0
    )
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked"
    assert value["prior_attempts"] and value["completed_count"] == 0
    assert value["source_artifact_hashes"]
    assert value["duration_s"] > (work["ended_monotonic_ns"] - work["started_monotonic_ns"]) / 1e9


@pytest.fixture
def replay_work(measured, tmp_path):
    """SCENARIO-VERIFY-8466-FAILURES: two real arms fit their exact registered replay slots."""
    original_deps, _, primitives = measured
    deps = deepcopy(original_deps)
    work = r.measure(tmp_path / "missing", tmp_path / "raw", tmp_path)
    work.update(
        root=str(r.ROOT), bundle=deps["bundle"], workload=deps["workload"], refs=deps["refs"]
    )
    for index, primitive in zip((15, 14), primitives, strict=True):
        target = tmp_path / f"primitive{index}.json"
        atomic_json(target, primitive)
        row = work["rows"][index]
        row.update(
            status="completed",
            missing_reason=None,
            primitive_reference=r.reference(target),
            metrics=k.reduce(r.read(primitive["frozen_reference"]), primitive),
        )
        work["primitives"].append(r.reference(target))
    atomic_json(Path(work["raw"]) / "measurement.json", work)
    receipts = [dict(name=name, passed=True) for name in r.REQUIRED]
    return work, r.build(work, receipts, tmp_path / "candidate.json")


@pytest.mark.parametrize(
    "mutation",
    [
        "none",
        "identity",
        "length",
        "authority",
        "bundle",
        "hash",
        "roster",
        "frozen",
        "metrics",
        "store_hash",
        "disk_state",
    ],
)
def test_independent_cold_controls(replay_work, tmp_path, mutation):
    """SCENARIO-REPORT-8466-TERMINAL: self-consistently rehashed operands retain independent negative checks."""
    work, value = replay_work
    if mutation == "identity":
        value["experiment_id"] = 1
    elif mutation == "length":
        work["rows"].pop()
    elif mutation == "authority":
        work["authority"].update(activated=True, tasks=[])
    elif mutation == "bundle":
        work["bundle"]["head"]["coefficients"][0] += 1
    elif mutation == "hash":
        value["code_config_hashes"][0]["sha256"] = "forged"
    elif mutation == "roster":
        work["rows"][0]["arm"] = "journal"
    elif mutation == "metrics":
        work["rows"][14]["metrics"]["wall_s"] += 1
    elif mutation in ("frozen", "store_hash", "disk_state"):
        row = work["rows"][15]
        primitive = r.read(row["primitive_reference"])
        if mutation == "frozen":
            frozen = r.read(primitive["frozen_reference"])
            frozen["cell"]["batch"] = 1
            atomic_json(tmp_path / "changed_frozen.json", frozen)
            primitive["frozen_reference"] = r.reference(tmp_path / "changed_frozen.json")
        elif mutation == "store_hash":
            primitive["store_references"][0]["sha256"] = "forged"
        else:
            import shutil

            shutil.copytree(Path(primitive["directory"]) / "store", tmp_path / "older" / "store")
            primitive["directory"] = str(tmp_path / "older")
            store = tmp_path / "older" / "store"
            target = next(
                p
                for p in store.glob("version-*.json")
                if json.loads(p.read_bytes())["state"]["version"] == 0
            )
            atomic_json(
                store / "current.json",
                dict(file=target.name, sha256=json.loads(target.read_bytes())["sha256"]),
            )
            primitive["store_references"] = [
                r.reference(p) for p in sorted(store.iterdir()) if p.is_file()
            ]
        atomic_json(tmp_path / "changed_primitive.json", primitive)
        row["primitive_reference"] = r.reference(tmp_path / "changed_primitive.json")
    atomic_json(Path(work["raw"]) / "measurement.json", work)
    value["work_reference"] = r.reference(Path(work["raw"]) / "measurement.json")
    value["reproducibility_checksum"] = r.checksum(value)
    atomic_json(tmp_path / "candidate.json", value)
    assert r.replay(tmp_path / "candidate.json") is (mutation == "none")


def test_dependency_failures(measured, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8466-TERMINAL: producer readiness and original store bytes are required."""
    deps, _, _ = measured
    original = json.loads((r.ROOT / r.UPSTREAM).read_bytes())
    monkeypatch.setattr(r, "read", lambda ref: deepcopy(original))
    original["journal_ready_score"] = 0
    with pytest.raises(ValueError, match="journal_ready_score"):
        r.dependencies(r.ROOT, tmp_path, tmp_path)
    original["journal_ready_score"] = 1
    original["code_config_hashes"][0]["sha256"] = "forged"
    monkeypatch.setattr(r, "ORIGINAL_DEPENDENCIES", lambda *args: deepcopy(deps))
    with pytest.raises(ValueError, match="qualified_store_bytes"):
        r.dependencies(r.ROOT, tmp_path, tmp_path)
