"""REQ-REPORT-8449, REQ-VERIFY-8449: preserve causal work and durable bytes."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import persistence_localization_8449 as r
from carnot.reporting import persistence_localization_runner_8449 as runner
from carnot.verify import persistence_localization_8449 as k
from carnot.reporting.current_work_receipt import atomic_json


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    """SCENARIO-VERIFY-8449-SPANS: use original authenticated service operands."""
    directory = tmp_path_factory.mktemp("exp8449-real-pair")
    original = r.read(r.reference(r.ROOT / r.PRODUCER))
    bundle = r.read(original["work_reference"])["bundle"]
    values = [k.run_repeat(bundle, k.CELLS[0], 1, directory / arm, arm) for arm in k.ARMS]
    return bundle, values


def test_real_pair(pair):
    """SCENARIO-VERIFY-8449-SPANS: instrumentation preserves original bytes."""
    bundle, values = pair
    summaries = [k.reduce_repeat(bundle, value) for value in values]
    assert values[0]["semantic_hash"] == values[1]["semantic_hash"]
    assert [t["snapshot_sha256"] for t in values[0]["transactions"]] == [
        t["snapshot_sha256"] for t in values[1]["transactions"]
    ]
    for value, summary in zip(values, summaries, strict=True):
        assert summary["fsync_count"] == 4 * (1 + len(value["transactions"]))
        assert summary["complete_wall_s"] >= summary["service_wall_s"]
        assert sum(s["exclusive_ns"] for s in summary["stage_rows"]) == (
            value["outer_end_ns"] - value["outer_start_ns"]
        )
    names = {s["name"] for s in summaries[1]["stage_rows"]}
    assert set(k.REQUIRED_STAGES) <= names


@pytest.mark.parametrize("mutation", ["overlap", "parent", "name", "semantics", "outer"])
def test_reject_primitive_drift(pair, mutation):
    """SCENARIO-VERIFY-8449-SPANS: a new checksum cannot repair damaged evidence."""
    bundle, values = pair
    value = deepcopy(values[1])
    if mutation == "overlap":
        value["spans"][1]["end_ns"] = value["outer_end_ns"] + 1
    elif mutation == "parent":
        value["spans"][1]["parent"] = 99
    elif mutation == "name":
        value["spans"][1]["name"] = "forged_stage"
    elif mutation == "semantics":
        value["rows"][0]["action"] = "forged_action"
    else:
        value["outer_start_ns"] = value["initialized_ns"]
    with pytest.raises(ValueError):
        k.reduce_repeat(bundle, value)


def test_blocked_roster_and_replay(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8449-GATES: absence retains all intended arms and warmups."""
    monkeypatch.setattr(r, "historical", lambda *args: ({}, [], [], []))
    root = tmp_path / "missing-root"
    root.mkdir()
    work = r.measure(root, tmp_path / "raw", tmp_path)
    assert work["gate_check_summary"]
    assert len(work["rows"]) == 20 and len(work["warmups"]) == 4
    assert all(x["status"] == "censored" for x in work["rows"])
    output = tmp_path / (r.NAME + ".json")
    receipts = [dict(name="private_child", passed=True)]
    value = r.build(work, receipts, output)
    assert value["verdict_class"] == "blocked"
    assert value["complete_wall_s"] is None
    atomic_json(output, value)
    assert r.replay(output)
    value["persistence_profile_ready_score"] = 1
    value["reproducibility_checksum"] = r.checksum(value)
    atomic_json(output, value)
    assert not r.replay(output)
    assert not r.replay(tmp_path / "absent")


def test_bad_input_hash(tmp_path):
    """SCENARIO-REPORT-8449-REPLAY: bytes precede imported scalar readiness."""
    path = tmp_path / "source"
    atomic_json(path, dict(x=1))
    ref = r.reference(path)
    path.write_text(json.dumps(dict(x=2)))
    with pytest.raises(ValueError):
        r.read(ref)


def test_fixed_work_units(tmp_path):
    """REQ-REPORT-8449: only preregistered cells, arms and repeat IDs execute."""
    for cell, repeat, arm in [
        (dict(batch=32), 1, "baseline"),
        (k.CELLS[0], 6, "baseline"),
        (k.CELLS[0], 1, "other"),
    ]:
        with pytest.raises(ValueError):
            k.run_repeat({}, cell, repeat, tmp_path, arm)


def test_observer_failure_closes_span(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8449-SPANS: failed publication retains its actual interval."""

    def fail(*args):
        raise OSError("private deliberate write failure")

    monkeypatch.setattr(k.source.s, "atomic_json", fail)
    recorder = k.Recorder()
    with recorder.observe(), pytest.raises(OSError):
        k.source.s.atomic_json(tmp_path / "pointer", {})
    assert not recorder.pointer and not recorder.stack
    assert recorder.spans[0]["end_ns"] >= recorder.spans[0]["start_ns"]


@pytest.mark.parametrize("mutation", ["semantic_hash", "fsync", "siblings"])
def test_more_span_failures(pair, mutation):
    """SCENARIO-VERIFY-8449-SPANS: conservation detects corrupt counters and nesting."""
    bundle, values = pair
    value = deepcopy(values[1])
    if mutation == "semantic_hash":
        value["semantic_hash"] = "forged"
    elif mutation == "fsync":
        next(s for s in value["spans"] if s["name"] == "file_fsync")["name"] = "write"
    else:
        siblings = [s for s in value["spans"] if s["parent"] == 0]
        siblings[1]["start_ns"] = siblings[0]["end_ns"] - 1
    with pytest.raises(ValueError):
        k.reduce_repeat(bundle, value)


def test_historical_custody(tmp_path):
    """REQ-REPORT-8449: the real producer qualifies under sealed original authority."""
    bundle, refs, gates, provenance = r.historical(r.ROOT, tmp_path / "raw", tmp_path)
    assert bundle and refs and not gates
    assert (
        provenance
        and provenance[0]["original_producer"]["sha256"] == "sha256:" + r.history.PINS[8407]
    )
    assert r.historical(tmp_path / "missing", tmp_path / "missing-raw", tmp_path)[2]


def admitted(monkeypatch, bundle):
    """Use explicit unit fixtures to test orchestration independently of numerical costs."""
    monkeypatch.setattr(
        r.authority,
        "authority",
        lambda *args: dict(
            activated=True,
            gate_check_summary=[],
            authority_snapshots={},
            tasks=[dict(id=r.TASK, MODEL_SPECS=[], deliverable="results/" + r.NAME + ".json")],
        ),
    )
    monkeypatch.setattr(r, "historical", lambda *args: (bundle, [], [], []))


def test_measurement_success_and_failed_children(tmp_path, monkeypatch, pair):
    """SCENARIO-REPORT-8449-GATES: one owned failed child cannot disappear from the roster."""
    bundle, values = pair
    admitted(monkeypatch, bundle)
    original = k.reduce_repeat(bundle, values[0])

    def child(name, argv, *args, **kwargs):
        request = json.loads(Path(argv[-1]).read_bytes())
        index = int(name.split("_")[-1])
        primitive = deepcopy(values[0])
        primitive.update(cell=request["cell"], repeat=request["repeat"], arm=request["arm"])
        atomic_json(Path(request["output"]), primitive)
        return dict(passed=index != 3, timed_out=index == 3)

    monkeypatch.setattr(r, "child", child)
    monkeypatch.setattr(
        k,
        "reduce_repeat",
        lambda bundle, value: dict(
            original, cell=value["cell"], repeat=value["repeat"], arm=value["arm"]
        ),
    )
    work = r.measure(r.ROOT, tmp_path / "raw", tmp_path)
    assert len(work["operands"]) == 23
    assert any(row["missing_reason"] == "repeat_deadline" for row in work["rows"])
    receipts = [dict(name=name, passed=True) for name in r.REQUIRED]
    work["owned_coverage"] = dict(percent_covered=100, missing_lines=0, num_statements=1)
    value = r.build(work, receipts, tmp_path / (r.NAME + ".json"))
    assert value["verdict_class"] == "blocked"
    lost = next(row for row in work["rows"] if row["status"] == "censored")
    lost.update(
        deepcopy(
            next(
                row
                for row in work["rows"]
                if row["cell"] == lost["cell"]
                and row["repeat"] == lost["repeat"]
                and row["status"] == "completed"
            )
        ),
        arm="baseline",
    )
    assert r.build(work, receipts, tmp_path / (r.NAME + ".json"))["verdict_class"] == "null"
    assert r.build(work, [], tmp_path / (r.NAME + ".json"))["verdict_class"] == "disqualified"
    lost["semantic_hash"] = "forged"
    assert r.build(work, receipts, tmp_path / (r.NAME + ".json"))["verdict_class"] == "disqualified"


def test_deadline_roster(tmp_path, monkeypatch, pair):
    """REQ-REPORT-8449: expiration retains pending measurements with a missing reason."""
    admitted(monkeypatch, pair[0])
    calls = iter([0, *[1801] * 24])
    monkeypatch.setattr(r.time, "monotonic", lambda: next(calls))
    work = r.measure(r.ROOT, tmp_path / "raw", tmp_path)
    assert all(row["missing_reason"] == "measurement_deadline" for row in work["rows"])


def test_private_cli_lifecycle(tmp_path):
    """SCENARIO-REPORT-8449-REPLAY: actual children publish and replay outside the checkout."""
    output = tmp_path / (r.NAME + ".json")
    argv = [sys.executable, "-u", str(r.ROOT / r.CLI)]
    commands = [
        (
            ["--private-control", "--root", str(tmp_path / "absent-root"), "--output", str(output)],
            0,
        ),
        (["--cold-replay", str(output)], 0),
        (["--deliberate-error"], 7),
        (["--date", "20261011"], 2),
        (["--private-control"], 2),
        (["--history-request", str(tmp_path / "absent")], 2),
    ]
    for args, expected in commands:
        completed = subprocess.run(
            [*argv, *args], cwd=tmp_path, capture_output=True, text=True, timeout=180
        )
        assert completed.returncode == expected, (completed.stdout, completed.stderr)
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked"
    assert value["required_checks_passed"] and not value["flagged_adversarial"]
    assert all(row["passed"] for row in value["validation_receipts"])


def test_rehashed_snapshot_corruption(tmp_path, pair):
    """SCENARIO-VERIFY-8449-SPANS: checksum-consistent state drift fails causal replay."""
    bundle, values = pair
    value = deepcopy(values[1])
    tx = value["transactions"][0]
    original = Path(tx["snapshot_path"])
    envelope = json.loads(original.read_bytes())
    envelope["state"]["version"] += 1
    envelope["sha256"] = r.canonical_hash(envelope["state"])
    changed = tmp_path / original.name
    atomic_json(changed, envelope)
    tx.update(
        snapshot_path=str(changed),
        snapshot_sha256=r.sha256_file(changed),
        state_hash=envelope["sha256"],
    )
    with pytest.raises(ValueError):
        k.reduce_repeat(bundle, value)


def test_historical_failed_children_and_operands(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8449-GATES: original custody failure cannot import a ready score."""
    monkeypatch.setattr(r, "child", lambda *a, **kw: dict(passed=False))
    assert r.historical(r.ROOT, tmp_path / "failed-child", tmp_path)[2]

    def qualified(*args, **kwargs):
        atomic_json(Path(args[1][-1]), dict(qualified=True))
        return dict(passed=True)

    monkeypatch.setattr(r, "child", qualified)
    original = r.read

    def altered(ref):
        value = original(ref)
        if "bundle" in value:
            value["bundle"]["labels"]["sha256"] = "forged"
        return value

    monkeypatch.setattr(r, "read", altered)
    assert r.historical(r.ROOT, tmp_path / "damaged", tmp_path)[2]


def test_resource_failures(tmp_path, monkeypatch, pair):
    """SCENARIO-REPORT-8449-GATES: missing required tools and storage stay explicit."""
    admitted(monkeypatch, pair[0])
    monkeypatch.setattr(r.os, "access", lambda *args: False)
    monkeypatch.setattr(r.shutil, "disk_usage", lambda *args: type("Disk", (), {"free": 0})())
    work = r.measure(r.ROOT, tmp_path / "raw", tmp_path)
    assert any(g["artifact_field"] == "disk_free" for g in work["gate_check_summary"])
    assert (
        sum(g["artifact_field"] == "executable_available" for g in work["gate_check_summary"]) == 5
    )


def test_manifest_and_coverage_publication(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8449-VALIDATION: private publication retains genuine coverage bytes."""
    from coverage import Coverage

    private = tmp_path / "private"
    private.mkdir()
    plan = runner.manifest(private)
    assert {x["name"] for x in plan} == r.REQUIRED
    assert any(r.TEST in x["argv"] for x in plan)
    coverage = Coverage(
        data_file=str(private / ".coverage.fixture"),
        include=[str(r.ROOT / r.OWNED[1])],
        config_file=False,
    )
    coverage.start()
    r.checksum(dict(private_fixture=True))
    coverage.stop()
    coverage.save()
    coverage.json_report(outfile=str(private / "coverage.json"))
    monkeypatch.setattr(r, "historical", lambda *args: ({}, [], [], []))
    monkeypatch.setattr(
        runner,
        "manifest",
        lambda *args: [
            dict(
                name="private_child",
                argv=[sys.executable, "-u", "-c", "print('private bounded child')"],
                deadline=30,
                expected=0,
                scope="owned",
            )
        ],
    )
    output = tmp_path / (r.NAME + ".json")
    assert runner.run(tmp_path / "absent", output, private, False) == 0
    value = json.loads(output.read_bytes())
    assert value["owned_coverage"]["num_statements"] > 0
    assert any("coverage_shards" in x["path"] for x in value["raw_shard_hashes"])


def test_repeat_child_and_owned_failure(tmp_path, monkeypatch, pair):
    """SCENARIO-REPORT-8449-REPLAY: an actual worker emits original primitive evidence."""
    request = tmp_path / "request.json"
    output = tmp_path / "primitive.json"
    atomic_json(
        request,
        dict(
            bundle=pair[0],
            cell=k.CELLS[1],
            repeat=1,
            arm="baseline",
            directory=str(tmp_path / "repeat"),
            output=str(output),
        ),
    )
    result = subprocess.run(
        [sys.executable, "-u", str(r.ROOT / r.CLI), "--repeat-request", str(request)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert k.reduce_repeat(pair[0], json.loads(output.read_bytes()))["prediction_count"] == 64

    def fail(*args):
        raise ValueError("owned private failure")

    monkeypatch.setattr(runner, "run", fail)
    assert runner.main(["--output", str(tmp_path / (r.NAME + ".json"))]) == 1


@pytest.mark.parametrize(
    "mutation", ["checksum", "source", "workload_ref", "tape", "roster", "summary", "journal"]
)
def test_cold_replay_rehashed_controls(tmp_path, monkeypatch, pair, mutation):
    """SCENARIO-REPORT-8449-REPLAY: cold reduction binds clocks and frozen journal units."""
    monkeypatch.setattr(r, "historical", lambda *args: (pair[0], [], [], []))
    work = r.measure(tmp_path / "absent", tmp_path / "raw", tmp_path)
    summary = k.reduce_repeat(pair[0], pair[1][1])
    primitive = tmp_path / "primitive.json"
    atomic_json(primitive, pair[1][1])
    work["rows"][0].update(summary, primitive_reference=r.reference(primitive), missing_reason=None)
    work["operands"] = [r.reference(primitive)]
    frozen = r.freeze_workload([slot for slot in r.roster() if slot["warmup"]] + work["rows"])
    atomic_json(Path(work["workload_reference"]["path"]), frozen)
    work["workload_reference"] = r.reference(Path(work["workload_reference"]["path"]))
    atomic_json(Path(work["raw"]) / "measurement.json", work)
    output = tmp_path / (r.NAME + ".json")
    receipts = [dict(name="private_child", passed=True)]
    value = r.build(work, receipts, output)
    atomic_json(output, value)
    assert r.replay(output)
    if mutation == "checksum":
        value["reproducibility_checksum"] = "invalid"
    elif mutation == "source":
        value["code_config_hashes"][0]["sha256"] = "forged"
        value["reproducibility_checksum"] = r.checksum(value)
    elif mutation == "workload_ref":
        value["workload_reference"] = r.reference(primitive)
        value["reproducibility_checksum"] = r.checksum(value)
    else:
        if mutation == "tape":
            frozen["event_tape"][0]["tick"] = 999
        elif mutation == "journal":
            frozen["expected_units"][0]["candidate_journal_work_units"][0][
                "expected_response_hash"
            ] = "forged"
        elif mutation == "roster":
            work["rows"][0]["repeat"] = 5
        else:
            work["rows"][0]["complete_wall_s"] *= 2
        atomic_json(Path(work["workload_reference"]["path"]), frozen)
        work["workload_reference"] = r.reference(Path(work["workload_reference"]["path"]))
        atomic_json(Path(work["raw"]) / "measurement.json", work)
        value = r.build(work, receipts, output)
    atomic_json(output, value)
    assert not r.replay(output)
