"""REQ-VERIFY-8464, REQ-REPORT-8464: complete clocks and sealed private publication."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.verify import persistence_localization_8464 as k
from carnot.reporting import persistence_localization_8464 as r
from carnot.reporting import persistence_localization_runner_8464 as runner
from carnot.reporting.current_work_receipt import atomic_json


@pytest.fixture
def bundle(tmp_path):
    """SCENARIO-VERIFY-8464-PAIRS: fixtures qualify mechanics without scientific claims."""
    slots = [
        dict(slot=i, unit_id=str(i), source_cluster_id=str(i), x=[0, 0.2, 0.3, 0.4, 0.5])
        for i in range(1, 129)
    ]
    slots[7]["x"] = None
    labels = tmp_path / "labels.json"
    atomic_json(labels, dict(rows=[dict(s, y=s["slot"] % 2) for s in slots]))
    return dict(
        head=dict(
            arm="spline34",
            coefficients=[0.0] * 34,
            temperature=2.0,
            knots=k.legacy.s.evaluator.kernel.KNOTS,
            degree=3,
        ),
        slots=slots,
        labels=r.reference(labels),
    )


@pytest.mark.parametrize("cell", k.CELLS)
def test_pair_and_partition(bundle, tmp_path, cell):
    """SCENARIO-VERIFY-8464-PAIRS: same requests produce exact states and responses."""
    baseline = k.run_repeat(bundle, cell, 1, tmp_path / "baseline", False)
    instrumented = k.run_repeat(bundle, cell, 1, tmp_path / "instrumented", True)
    row = k.pair(bundle, baseline, instrumented)
    assert row["status"] == "completed"
    assert row["instrumentation_overhead_s"] == pytest.approx(
        row["instrumented_wall_s"] - row["baseline_wall_s"]
    )
    assert baseline["final_state"] == instrumented["final_state"]
    assert row["pending"] == 0 and row["feedback_attempts"] == 8
    assert sum(row["stage_ns"].values()) == instrumented["end_ns"] - instrumented["start_ns"]
    assert set(k.STAGES) <= set(row["stage_ns"])
    assert row["tail_drain_s"] > 0 and row["fsync_count"] > 0


@pytest.mark.parametrize("mutation", ["nesting", "name", "state", "response", "fsync", "tail"])
def test_primitive_negative_controls(bundle, tmp_path, mutation):
    """SCENARIO-VERIFY-8464-PAIRS: repaired local hashes cannot restore causality."""
    value = k.run_repeat(bundle, k.CELLS[1], 1, tmp_path / "repeat", True)
    if mutation == "nesting":
        value["persistence_spans"][0]["stages"][0]["end_ns"] = value["end_ns"] + 1
    elif mutation == "name":
        value["persistence_spans"][0]["stages"][0]["name"] = "invented"
    elif mutation == "state":
        value["final_state"]["release_cursor"] += 1
    elif mutation == "response":
        value["response_hash"] = "sha256:wrong"
    elif mutation == "fsync":
        value["transactions"][0]["fsync_count"] = 3
    else:
        value["tail_start_ns"] = value["end_ns"]
    with pytest.raises(ValueError):
        k.reduce(bundle, value)


def test_original_byte_budget(bundle, tmp_path, monkeypatch):
    """REQ-VERIFY-8464: the adapter preserves the original state byte rejection."""
    monkeypatch.setattr(k.legacy.s, "MAX_BYTES", 1)
    with pytest.raises(ValueError, match="state_byte_budget"):
        k.run_repeat(bundle, k.CELLS[0], 1, tmp_path / "small", True)


def test_missing_owned_failures_and_replay(tmp_path):
    """SCENARIO-REPORT-8464-TERMINAL: absence blocks; owned failures disqualify."""
    work = r.measure(tmp_path / "absent", tmp_path / "raw", tmp_path)
    receipts = [dict(name=n, passed=True) for n in r.REQUIRED]
    value = r.build(work, receipts, tmp_path / (r.NAME + ".json"))
    assert value["verdict_class"] == "blocked"
    assert value["censored_count"] == value["intended_count"] == 10
    assert any(g["observed"] is None for g in value["gate_check_summary"])
    assert (
        r.build(work, [dict(passed=False)], tmp_path / "output")["verdict_class"] == "disqualified"
    )
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, value)
    assert r.replay(candidate)
    value["persistence_profile_ready_score"] = 1
    value["reproducibility_checksum"] = r.checksum(value)
    atomic_json(candidate, value)
    assert not r.replay(candidate) and not r.replay(tmp_path / "missing")


def test_private_cli(tmp_path):
    """SCENARIO-REPORT-8464-TERMINAL: real children publish only into private paths."""
    cli = str(r.ROOT / r.CLI)
    output = tmp_path / "results" / (r.NAME + ".json")
    result = subprocess.run(
        [
            sys.executable,
            "-u",
            cli,
            "--root",
            str(tmp_path / "absent"),
            "--output",
            str(output),
            "--private-control",
        ],
        capture_output=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr.decode()
    assert json.loads(output.read_bytes())["verdict_class"] == "blocked"
    assert runner.main(["--deliberate-error"]) == 7
    assert runner.main(["--cold-replay", str(output)]) == 0
    assert runner.main(["--cold-replay", str(tmp_path / "missing")]) == 1
    assert runner.main(["--private-control"]) == 2
    with pytest.raises(SystemExit):
        runner.main(["--date", "20261011"])


@pytest.fixture(scope="module")
def frozen_pair(tmp_path_factory):
    """REQ-VERIFY-8464: cache real disk evidence once for independent negative controls."""
    private = tmp_path_factory.mktemp("frozen8464")
    deps = r.dependencies(r.ROOT, private, private)
    baseline = k.run_repeat(deps["bundle"], k.CELLS[1], 1, private / "baseline", False)
    instrumented = k.run_repeat(deps["bundle"], k.CELLS[1], 1, private / "instrumented", True)
    return deps, baseline, instrumented


@pytest.mark.parametrize(
    "mutation", ["roster", "binding", "topology", "bytes", "cell", "pair", "parent"]
)
def test_additional_clock_controls(frozen_pair, mutation):
    """SCENARIO-VERIFY-8464-PAIRS: independent file and topology checks reject forged spans."""
    deps, baseline, original = frozen_pair
    value = deepcopy(original)
    if mutation == "roster":
        value["persistence_spans"].pop()
    elif mutation == "binding":
        value["instrumented"] = False
    elif mutation == "topology":
        value["persistence_spans"][0]["fsync_count"] = 3
    elif mutation == "bytes":
        value["persistence_spans"][1]["bytes"] += 1
    elif mutation == "parent":
        value["persistence_spans"][0]["start_ns"] = 0
    elif mutation == "cell":
        with pytest.raises(ValueError, match="registered_cells_only"):
            k.run_repeat(deps["bundle"], dict(workload="ordinary"), 1, Path("/unused"), True)
        return
    else:
        with pytest.raises(ValueError, match="paired_semantics"):
            k.pair(deps["bundle"], original, original)
        return
    with pytest.raises(ValueError):
        k.reduce(deps["bundle"], value)


def test_owned_measurement_roster_and_failures(frozen_pair, tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8464-FAILURES: all24 registered arms remain accountable."""
    deps, baseline_primitive, instrumented_primitive = frozen_pair
    monkeypatch.setattr(r, "dependencies", lambda *args: deepcopy(deps))
    calls = []

    def observed(bundle, cell, repeat, path, instrumented):
        calls.append((cell, repeat, instrumented))
        return deepcopy(instrumented_primitive if instrumented else baseline_primitive)

    monkeypatch.setattr(k, "run_repeat", observed)
    monkeypatch.setattr(k, "reduce", lambda *args: {})
    real_reduce = r.reduce
    monkeypatch.setattr(r, "reduce", lambda *args: ([], []))
    work = r.measure(r.ROOT, tmp_path / "happy", tmp_path)
    assert len(calls) == len(work["primitives"]) == 24
    assert all(c[0] in k.CELLS for c in calls)
    assert [c[2] for c in calls[:4]] == [False, True, True, False]

    def failed(*args):
        raise OSError("deliberate_durable_write_failure")

    monkeypatch.setattr(k, "run_repeat", failed)
    work = r.measure(r.ROOT, tmp_path / "failures", tmp_path)
    assert len(work["owned_failures"]) == 12 and not work["primitives"]
    monkeypatch.setattr(r, "reduce", real_reduce)
    rows, warmups = real_reduce(work)
    assert all(row["status"] == "failed" for row in rows + warmups)
    real_clock = r.time.monotonic_ns
    counter = [real_clock()]

    def deadline_clock():
        counter[0] += 1801 * 10**9
        return counter[0]

    monkeypatch.setattr(r.time, "monotonic_ns", deadline_clock)
    work = r.measure(r.ROOT, tmp_path / "deadline", tmp_path)
    assert len(work["owned_failures"]) == 12
    assert set(work["owned_failures"].values()) == {"measurement_deadline"}


def test_missing_resources_and_exact_task(tmp_path, monkeypatch):
    """REQ-REPORT-8464: missing tools and wrong task registration close admission."""
    monkeypatch.setattr(r.os, "access", lambda *args: False)
    monkeypatch.setattr(r.shutil, "disk_usage", lambda *args: type("Disk", (), {"free": 0})())
    monkeypatch.setattr(
        r.authority,
        "authority",
        lambda *args: dict(activated=True, tasks=[dict(id=r.TASK, MODEL_SPECS=["wrong"])]),
    )
    work = r.measure(tmp_path, tmp_path / "raw", tmp_path)
    assert {"exact_task_and_model_specs", "disk_free", "tool_available"} <= {
        g["field"] for g in work["gate_check_summary"]
    }


def test_reduction_and_rehashed_operands(frozen_pair, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8464-TERMINAL: cold reconstruction binds rows, hashes and coverage."""
    deps, baseline, instrumented = frozen_pair
    work = r.measure(tmp_path / "absent", tmp_path / "raw", tmp_path)
    work.update(bundle=deps["bundle"], root=str(r.ROOT), refs=deps["refs"])
    for arm, primitive in [("baseline", baseline), ("instrumented", instrumented)]:
        path = tmp_path / (arm + ".json")
        atomic_json(path, primitive)
        work["primitives"].append(dict(r.reference(path), cell_index=1, repeat=1, arm=arm))
    work["rows"], work["warmups"] = r.reduce(work)
    assert len(work["rows"]) == 10 and work["rows"][5]["status"] == "completed"
    receipts = [dict(name=n, passed=True) for n in r.REQUIRED]
    candidate = tmp_path / "candidate.json"
    atomic_json(tmp_path / "raw/measurement.json", work)
    value = r.build(work, receipts, candidate)
    atomic_json(candidate, value)
    assert r.replay(candidate)
    for mutation in ("identity", "checksum", "source", "rows"):
        changed = deepcopy(value)
        if mutation == "identity":
            changed["experiment_id"] = 8465
        elif mutation == "checksum":
            changed["reproducibility_checksum"] = "wrong"
        elif mutation == "source":
            changed["raw_shard_hashes"][0]["sha256"] = "wrong"
        else:
            broken = deepcopy(work)
            broken["rows"][5]["complete_wall_s"] += 1
            atomic_json(tmp_path / "raw/measurement.json", broken)
            changed["work_reference"] = r.reference(tmp_path / "raw/measurement.json")
        if mutation != "checksum":
            changed["reproducibility_checksum"] = r.checksum(changed)
        atomic_json(candidate, changed)
        assert not r.replay(candidate)
    row = work["rows"][5]
    assert not r.coverage_ready(dict(missing_lines=0, percent_covered=100))
    monkeypatch.setattr(r, "coverage_ready", lambda report: True)
    work.update(
        rows=[row] * 10,
        primitives=work["primitives"] * 12,
        gate_check_summary=[],
        owned_coverage=dict(missing_lines=0, percent_covered=100),
    )
    assert r.build(work, receipts, candidate)["verdict_class"] == "null"
    work["adversarial_findings"] = [dict(severity="warning")]
    assert r.build(work, receipts, candidate)["persistence_profile_ready_score"] == 0


@pytest.mark.parametrize("failure", ["ready", "pin", "qualification", "operand"])
def test_original_admission_controls(frozen_pair, tmp_path, monkeypatch, failure):
    """REQ-REPORT-8464: historical admission rejects failures independently of current YAML."""
    real_read = r.read
    primary = real_read(dict(path=str(r.ROOT / r.UPSTREAM), sha256=r.UPSTREAM_PIN))
    if failure in ("ready", "pin"):

        def changed_read(ref):
            value = deepcopy(real_read(ref))
            if failure == "ready" and ref["path"] == str(r.ROOT / r.UPSTREAM):
                value["service_inputs_ready_score"] = 0
            if failure == "pin" and ref == primary["primitive_reference"]:
                next(r for r in value["history"]["references"] if "experiment_8407_" in r["path"])[
                    "sha256"
                ] = "wrong"
            return value

        monkeypatch.setattr(r, "read", changed_read)
    elif failure == "qualification":
        monkeypatch.setattr(r.historical, "qualify", lambda *args: dict(qualified=False))
    else:
        deps, _, _ = frozen_pair
        broken_path = deps["refs"][-1]["path"]
        real_hash = r.shared.sha256_file
        monkeypatch.setattr(
            r.shared, "sha256_file", lambda p: "wrong" if str(p) == broken_path else real_hash(p)
        )
    with pytest.raises(ValueError):
        r.dependencies(r.ROOT, tmp_path, tmp_path)


def test_manifest_genuine_child_coverage(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8464-FAILURES: archive genuine Coverage.py failure child shards."""
    private = tmp_path / "private"
    plan = runner.manifest(private)
    assert {row["name"] for row in plan} == r.REQUIRED
    config = "--rcfile=" + str(private / "coverage.ini")
    for command, expected in [
        (["run", config, str(r.ROOT / r.CLI), "--deliberate-error"], 7),
        (["combine", config, "--keep"], 0),
        (["json", config, "-o", str(private / "coverage.json")], 0),
    ]:
        result = subprocess.run(
            [sys.executable, "-m", "coverage", *command], capture_output=True, timeout=60
        )
        assert result.returncode == expected, result.stderr.decode()
    monkeypatch.setattr(runner, "manifest", lambda *args: [])
    output = tmp_path / "results" / (r.NAME + ".json")
    assert runner.run(tmp_path / "absent", output, private) == 0
    value = json.loads(output.read_bytes())
    assert value["owned_coverage"]["covered_lines"] > 0
    assert any(".coverage" in ref["path"] for ref in value["raw_shard_hashes"])


def test_private_measured_primitive_child(frozen_pair, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8464-TERMINAL: fresh children reject self-consistently changed primitives."""
    deps, baseline, instrumented = frozen_pair

    def private_measure(root, raw, private):
        work = r.measure(tmp_path / "absent", raw, private)
        work.update(root=str(r.ROOT), bundle=deps["bundle"], refs=deps["refs"])
        for arm, primitive in [("baseline", baseline), ("instrumented", instrumented)]:
            path = raw / (arm + ".json")
            atomic_json(path, primitive)
            work["primitives"].append(dict(r.reference(path), cell_index=1, repeat=1, arm=arm))
        work["rows"], work["warmups"] = r.reduce(work)
        return work

    measured = r.measure

    def saved(root, raw, private):
        with monkeypatch.context() as local:
            local.setattr(r, "measure", measured)
            return private_measure(root, raw, private)

    monkeypatch.setattr(r, "measure", saved)
    output = tmp_path / "results" / (r.NAME + ".json")
    assert runner.run(r.ROOT, output, tmp_path, True) == 0
    value = json.loads(output.read_bytes())
    receipt = next(c for c in value["validation_receipts"] if c["name"] == "rehashed_primitive")
    assert receipt["passed"] and receipt["exit_code"] == 1


def test_registered_slots_and_replay_authority(frozen_pair, tmp_path, monkeypatch):
    """REQ-REPORT-8464: pair registration and independent contract digests stay binding."""
    deps, baseline, instrumented = frozen_pair
    work = r.measure(tmp_path / "absent", tmp_path / "raw", tmp_path)
    work.update(root=str(r.ROOT), bundle=deps["bundle"], refs=deps["refs"])
    for arm, primitive in [("baseline", baseline), ("instrumented", instrumented)]:
        path = tmp_path / (arm + ".json")
        atomic_json(path, primitive)
        work["primitives"].append(dict(r.reference(path), cell_index=0, repeat=1, arm=arm))
    with pytest.raises(ValueError, match="registered_pair_slot_binding"):
        r.reduce(work)
    work["primitives"] = []
    receipts = [dict(name=n, passed=True) for n in r.REQUIRED]
    candidate = tmp_path / "candidate.json"
    work["authority"] = dict(activated=True, tasks=[])
    atomic_json(tmp_path / "raw/measurement.json", work)
    atomic_json(candidate, r.build(work, receipts, candidate))
    assert not r.replay(candidate)
    work["authority"]["activated"] = False
    atomic_json(tmp_path / "raw/measurement.json", work)
    atomic_json(candidate, r.build(work, receipts, candidate))
    monkeypatch.setattr(r, "dependencies", lambda *args: dict(bundle={}))
    assert not r.replay(candidate)


def test_coverage_report_gate_negative_controls(monkeypatch):
    """REQ-REPORT-8464: constructed reports test the gate, never supply invocation coverage."""
    reference = dict(path="/unused/constructed-report.json", sha256="constructed_control")
    parsed = dict(
        totals=dict(missing_lines=0, percent_covered=100),
        files={path: dict(summary=dict(num_statements=1, missing_lines=0)) for path in r.OWNED},
    )
    monkeypatch.setattr(r, "read", lambda ref: deepcopy(parsed))
    report = dict(parsed["totals"], reference=reference)
    assert r.coverage_ready(report)
    parsed["files"][r.OWNED[-1]]["summary"]["missing_lines"] = 1
    assert not r.coverage_ready(report)
    parsed["files"][r.OWNED[-1]]["summary"]["missing_lines"] = 0
    report["percent_covered"] = 99
    assert not r.coverage_ready(report)
