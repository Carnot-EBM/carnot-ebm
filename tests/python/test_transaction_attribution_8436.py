"""REQ-REPORT-8436, REQ-VERIFY-8436: authentic primitive clocks and private readers."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import transaction_attribution_8436 as r
from carnot.reporting import transaction_attribution_runner_8436 as runner
from carnot.verify import transaction_attribution_8436 as k
from carnot.reporting.current_work_receipt import atomic_json


@pytest.fixture(scope="module")
def actual_work(tmp_path_factory):
    """Keep one private producer snapshot for the whole audit test invocation."""
    private = tmp_path_factory.mktemp("actual_8436")
    return r.measure(r.ROOT, private / "raw", private)


def primitive():
    """Known intervals leave real gaps so a stage-only denominator cannot pass."""
    cell = dict(workload="ordinary", batch=1, predictions_per_update=1)
    prefix = "ordinary-b1-r1-rep1"
    names = list(k.PERSISTENCE) + [
        "direct_scoring",
        "coefficient_update",
        "python_boundary_conversion",
        "response",
    ]
    stages = [dict(name=n, start_ns=110 + i, end_ns=111 + i) for i, n in enumerate(names)]
    return dict(
        cell=cell,
        repeat=1,
        repeat_id=prefix,
        start_ns=100,
        initialized_ns=105,
        tail_start_ns=120,
        end_ns=140,
        rows=[],
        transactions=[
            dict(
                transaction_id=prefix + "-tx0",
                start_ns=110,
                end_ns=130,
                stages=stages,
                request_ids=[],
            )
        ],
    )


def test_complete_partition():
    """SCENARIO-VERIFY-8436-CLOCKS: initialization, gaps and tail are retained."""
    out = k.attribute(primitive())
    assert out["complete_wall_s"] == 40 / 1e9
    assert out["kernel_s"] == 2 / 1e9
    assert out["conversion_s"] == 1 / 1e9
    assert out["tail_drain_s"] == 20 / 1e9
    assert out["unattributed_s"] > 0
    assert sum(
        out[n]
        for n in (
            "kernel_s",
            "conversion_s",
            "persistence_stage_s",
            "other_stage_s",
            "unattributed_s",
        )
    ) == pytest.approx(out["complete_wall_s"])


@pytest.mark.parametrize(
    "mutation", ["double", "tail", "identity", "negative", "transaction", "stage", "missing_stage"]
)
def test_clock_controls(mutation):
    """SCENARIO-VERIFY-8436-CLOCKS: malformed intervals cannot become attribution."""
    value = primitive()
    if mutation == "double":
        value["transactions"][0]["stages"][0]["end_ns"] += 2
    elif mutation == "tail":
        del value["tail_start_ns"]
    elif mutation == "identity":
        value["repeat_id"] = "swapped"
    elif mutation == "negative":
        value["end_ns"] = 99
    elif mutation == "transaction":
        value["transactions"][0]["start_ns"] = 104
    elif mutation == "stage":
        value["transactions"][0]["stages"][0]["end_ns"] = 200
    else:
        value["transactions"][0]["stages"].pop(0)
    with pytest.raises((ValueError, KeyError)):
        k.attribute(value)


def test_actual_primitive_and_native_rejection(tmp_path, actual_work):
    """REQ-REPORT-8436: historical bytes qualify clocks without another benchmark."""
    work = actual_work
    assert not work["gate_check_summary"]
    assert len(work["rows"]) == 135

    assert all(row["status"] == "completed" for row in work["rows"])
    assert len(work["warmups"]) == 27
    assert work["native_consumer_failure"]["reproduced"]
    assert not work["native_consumer_failure"]["consumer"]["passed"]
    value = r.build(
        work, [dict(name=name, passed=True) for name in r.REQUIRED], tmp_path / "out.json"
    )
    assert value["clock_attribution_ready_score"] == 1
    assert value["native_parity_ready_score"] == 0
    path = tmp_path / "candidate.json"
    atomic_json(path, value)
    assert r.replay(path)
    changed = deepcopy(value)
    changed["rows"][0]["kernel_s"] *= 2
    changed["reproducibility_checksum"] = r.checksum(changed)
    atomic_json(path, changed)
    assert not r.replay(path)


def test_private_cli_missing_and_failures(tmp_path):
    """SCENARIO-REPORT-8436-CUSTODY: actual CLI children cover honest external absence."""
    output = tmp_path / "results" / (r.NAME + ".json")
    argv = [sys.executable, "-u", str(r.ROOT / r.CLI)]
    result = subprocess.run(
        [
            *argv,
            "--date",
            "20261012",
            "--root",
            str(tmp_path),
            "--output",
            str(output),
            "--private-control",
        ],
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr.decode()
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked"
    assert value["censored_count"] == 135
    assert value["clock_attribution_ready_score"] == 0
    assert r.replay(output)
    assert (
        subprocess.run([*argv, "--deliberate-error"], capture_output=True, timeout=30).returncode
        == 7
    )
    assert (
        subprocess.run([*argv, "--date", "wrong"], capture_output=True, timeout=30).returncode == 2
    )
    assert not r.replay(tmp_path / "absent")


def test_additional_interval_failures(actual_work):
    """REQ-VERIFY-8436: response and request clocks share the same containing transaction."""
    value = primitive()
    value["transactions"][0]["stages"][0]["name"] = "unrecognized"
    with pytest.raises(ValueError, match="missing_stage"):
        k.attribute(value)
    value = primitive()
    value["transactions"][0]["request_ids"] = ["absent"]
    with pytest.raises(ValueError, match="request_transaction_binding"):
        k.attribute(value)
    original = r.read(actual_work["operands"][1])
    value = deepcopy(original)
    old = value["rows"][0]["request_id"]
    value["rows"][0]["request_id"] = "swapped"
    value["transactions"][0]["request_ids"] = [
        "swapped" if identity == old else identity
        for identity in value["transactions"][0]["request_ids"]
    ]
    with pytest.raises(ValueError, match="request_identity"):
        k.attribute(value)
    value = deepcopy(original)
    value["tail_start_ns"] = value["initialized_ns"]
    with pytest.raises(ValueError, match="tail_boundary"):
        k.attribute(value)


def test_reduction_invalid_rows_retained(actual_work, monkeypatch):
    """SCENARIO-REPORT-8436-CUSTODY: invalid rows cannot disappear from the 135 slots."""
    with pytest.raises(ValueError, match="repeat_roster"):
        r.reduce(r.ROOT, [])
    operands = deepcopy(actual_work["operands"])
    operands[1]["sha256"] = "changed"
    rows, _, _ = r.reduce(r.ROOT, operands)
    assert rows[0]["status"] == "failed"
    assert len(rows) == 135
    real_read = r.read
    for field in ("repeat", "rows"):

        def changed_read(ref):
            value = real_read(ref)
            if ref["path"] == actual_work["operands"][1]["path"]:
                value[field] = 99 if field == "repeat" else []
            return value

        monkeypatch.setattr(r, "read", changed_read)
        rows, _, _ = r.reduce(r.ROOT, actual_work["operands"])
        assert rows[0]["status"] == "failed"
    monkeypatch.setattr(r, "read", real_read)
    with pytest.raises(ValueError, match="input_hash"):
        r.read(dict(path=actual_work["operands"][0]["path"], sha256="wrong"))


@pytest.mark.parametrize(
    "mutation",
    ["checksum", "source", "rows", "timing", "native_bytes", "native_proof", "authority"],
)
def test_rehashed_replay_controls(actual_work, tmp_path, mutation):
    """SCENARIO-REPORT-8436-CUSTODY: independent operands defeat coordinated rehashing."""
    work = deepcopy(actual_work)
    work["raw"] = str(tmp_path)
    atomic_json(
        tmp_path / "attribution_rows.json", dict(rows=work["rows"], warmups=work["warmups"])
    )
    if mutation == "rows":
        work["rows"][0]["kernel_s"] *= 2
    if mutation == "native_bytes":
        changed = tmp_path / "native.json"
        changed.write_text("{}")
        work["native_consumer_failure"]["candidate"] = r.reference(changed)
    if mutation == "native_proof":
        work["native_consumer_failure"]["proof"]["max_error"] = 999
    if mutation == "authority":
        changed = tmp_path / "active.yaml"
        changed.write_text("tasks: []")
        work["authority"]["authority_snapshots"]["active"]["snapshot_path"] = str(changed)
    atomic_json(tmp_path / "measurement.json", work)
    value = r.build(
        work, [dict(name=name, passed=True) for name in r.REQUIRED], tmp_path / "candidate.json"
    )
    if mutation == "checksum":
        value["reproducibility_checksum"] = "wrong"
    if mutation == "source":
        value["source_artifact_hashes"][0]["sha256"] = "wrong"
    if mutation == "timing":
        atomic_json(tmp_path / "attribution_rows.json", dict(rows=[], warmups=[]))
        value["attribution_rows_sha256"] = r.reference(tmp_path / "attribution_rows.json")["sha256"]
    if mutation != "checksum":
        value["reproducibility_checksum"] = r.checksum(value)
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, value)
    assert not r.replay(candidate)


@pytest.mark.parametrize(
    "failure", ["terminal", "sidecar", "qualification", "source", "empty", "primitive"]
)
def test_input_authentication_failures(actual_work, tmp_path, monkeypatch, failure):
    """REQ-REPORT-8436: an unavailable required operand names its failing field."""
    root = tmp_path / "root"
    name = next(iter(r.PINS))
    target = root / name
    target.parent.mkdir(parents=True)
    target.write_bytes((r.ROOT / name).read_bytes())
    primary = r.read(r.reference(target))
    upstream = r.read(primary["work_reference"])
    real_read, real_hash = r.read, r.sha256_file
    bound_reader = r.read_bound_sidecar
    monkeypatch.setattr(
        r, "read_bound_sidecar", lambda copied, sidecar: bound_reader(r.ROOT / name, sidecar)
    )
    if failure in ("terminal", "sidecar"):
        monkeypatch.setattr(r, "TERMINAL_PIN" if failure == "terminal" else "SIDECAR_PIN", "wrong")
    if failure == "qualification":
        monkeypatch.setattr(r, "read_bound_sidecar", lambda *args: dict(report=dict(passed=False)))
    if failure == "source":
        broken = upstream["refs"][0]["path"]
        monkeypatch.setattr(
            r, "sha256_file", lambda path: "wrong" if str(path) == broken else real_hash(path)
        )
    if failure == "primitive":

        def changed_primitive_ref(ref):
            value = real_read(ref)
            if ref == primary["work_reference"]:
                value["results"][0]["sha256"] = "wrong"
            return value

        monkeypatch.setattr(r, "read", changed_primitive_ref)
    if failure == "empty":
        empty = tmp_path / "empty"
        empty.write_bytes(b"")

        def changed_read(ref):
            value = real_read(ref)
            if ref == primary["work_reference"]:
                value["refs"] = [r.reference(empty)]
            return value

        monkeypatch.setattr(r, "read", changed_read)
    work = r.measure(root, tmp_path / "raw", tmp_path)
    assert work["gate_check_summary"]
    assert any(
        g["field"] == "authenticated_terminal_and_operands" for g in work["gate_check_summary"]
    )
    assert len(work["rows"]) == 135
    expected = dict(
        terminal="terminal_receipt_hash",
        sidecar="terminal_sidecar_hash",
        qualification="unqualified_producer",
        source="historical_source_or_code_hash",
        empty="empty_historical_operand",
        primitive="primitive_hash",
    )[failure]
    assert any(g["observed"] == expected for g in work["gate_check_summary"]), work[
        "gate_check_summary"
    ]


def test_missing_tools_and_disk(tmp_path, monkeypatch):
    """REQ-REPORT-8436: disk and tool failures remain explicit before independent CPU work."""
    from types import SimpleNamespace

    original = r.ROOT
    for name in [*r.OWNED, r.TEST, *r.GUARDS]:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((original / name).read_bytes())
    monkeypatch.setattr(r, "ROOT", tmp_path)
    monkeypatch.setattr(r.shutil, "disk_usage", lambda path: SimpleNamespace(free=0))
    work = r.measure(tmp_path, tmp_path / "raw", tmp_path)
    assert sum(g["field"] == "tool_available" for g in work["gate_check_summary"]) == 5
    assert any(g["field"] == "disk_free" for g in work["gate_check_summary"])
    value = r.build(work, [dict(passed=False)], tmp_path / "out")
    assert value["verdict_class"] == "disqualified"


def test_manifest_real_coverage_and_runner_failure(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8436-CUSTODY: actual failure child shards and production copy paths."""
    private = tmp_path / "private"
    plan = runner.manifest(private)
    assert {row["name"] for row in plan} == r.REQUIRED
    config = private / "coverage.ini"
    commands = [
        [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--rcfile=" + str(config),
            str(r.ROOT / r.CLI),
            "--deliberate-error",
        ],
        [sys.executable, "-m", "coverage", "combine", "--rcfile=" + str(config), "--keep"],
        [
            sys.executable,
            "-m",
            "coverage",
            "json",
            "--rcfile=" + str(config),
            "-o",
            str(private / "coverage.json"),
        ],
    ]
    for command, expected in zip(commands, (7, 0, 0), strict=True):
        result = subprocess.run(command, capture_output=True, timeout=60)
        assert result.returncode == expected, result.stderr.decode()
    monkeypatch.setattr(runner, "manifest", lambda folder: [])
    output = tmp_path / "results" / (r.NAME + ".json")
    assert runner.run(tmp_path, output, private, False) == 0
    value = json.loads(output.read_bytes())
    assert value["owned_coverage"]["covered_lines"] > 0
    assert value["raw_shard_hashes"]

    def failed_run(*args):
        raise ValueError("deliberate_owned_failure")

    monkeypatch.setattr(runner, "run", failed_run)
    assert runner.main(["--private-control", "--root", str(tmp_path)]) == 1


def test_real_primitive_cli_controls(actual_work, tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8436-CLOCKS: each damaged primitive reaches a fresh actual child."""

    def saved_measure(root, raw, private):
        work = deepcopy(actual_work)
        work["raw"] = str(raw)
        atomic_json(raw / "attribution_rows.json", dict(rows=work["rows"], warmups=work["warmups"]))
        atomic_json(raw / "measurement.json", work)
        return work

    monkeypatch.setattr(r, "measure", saved_measure)
    output = tmp_path / "results" / (r.NAME + ".json")
    assert runner.run(r.ROOT, output, tmp_path, True) == 1
    value = json.loads(output.read_bytes())
    receipts = [
        row
        for row in value["validation_receipts"]
        if row["name"] in {"doubled_stage", "deleted_tail", "swapped_repeat", "rehashed_timings"}
    ]
    assert len(receipts) == 4
    assert all(row["passed"] and row["exit_code"] == 1 for row in receipts)
    assert value["clock_attribution_ready_score"] == 0
