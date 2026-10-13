"""REQ-VERIFY-8493, REQ-REPORT-8493: durable append qualification."""

from copy import deepcopy
import json
from pathlib import Path
import struct
import subprocess
import sys

import pytest

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.verify import append_store_8493 as k


@pytest.fixture(scope="module")
def tape():
    """SCENARIO-VERIFY-8493-RECOVERY: independently sealed original service inputs."""
    upstream = json.loads(
        (k.ROOT / "results/experiment_8492_v730_publication_cost_model.json").read_bytes()
    )
    return json.loads(
        Path(upstream["transition_fixture"]["tape_references"][0]["path"]).read_bytes()
    )


def test_stable_append_parity_and_readers(tape, tmp_path):
    """SCENARIO-VERIFY-8493-RECOVERY: stable inode, snapshots, feedback and stale copies."""
    store = k.Store(tmp_path / "append", tape)
    store.initialize()
    inode = store.journal.stat().st_ino
    pinned = store.read()
    snapshot = k.base.Snapshot(tmp_path / "snapshot", tape)
    snapshot.initialize()
    for tx in tape["transactions"]:
        assert store.apply(tx) == snapshot.apply(tx)
    store.drain()
    expected = k.base.oracle(tape, len(tape["transactions"]))
    assert store.read(expected["ledger"]) == snapshot.read() == expected
    assert pinned == k.base.oracle(tape, 0)
    assert store.journal.stat().st_ino == inode
    assert not list(store.path.glob("commit-*"))
    assert len(list(store.path.glob("checkpoint-[0-9]*.json"))) == 1
    assert any(r["operation"] == "unlink" for r in store.operations)
    assert store.apply(tape["transactions"][-1]) == expected["replies"][-1]
    store.initialize()


def test_unacknowledged_tail_corruption_and_ids(tape, tmp_path):
    """SCENARIO-VERIFY-8493-RECOVERY: complete frames replay once; conflicts fail closed."""
    store = k.Store(tmp_path, tape)
    store.initialize()
    first = k.framing.frame(tape, 1, k.framing.GENESIS)
    store.journal.write_bytes(first + b"\x00\x00")
    assert store.read()["sequence"] == 1
    assert store.journal.read_bytes() == first
    assert store.apply(tape["transactions"][0]) == k.base.oracle(tape, 1)["replies"][0]
    changed = deepcopy(tape["transactions"][0])
    changed["events"][0]["tick"] += 1
    with pytest.raises(ValueError, match="duplicate_conflict"):
        store.apply(changed)
    with pytest.raises(ValueError, match="sealed_transaction_order"):
        store.apply(tape["transactions"][2])
    with pytest.raises(ValueError, match="acknowledged_ledger_loss"):
        store.read(["absent"])
    for bad in (first[:-1] + b"!", struct.pack(">I", 999999) + first[4:]):
        store.journal.write_bytes(bad)
        with pytest.raises(ValueError):
            store.read()
    changed_tape = deepcopy(tape)
    changed_tape["transactions"][0]["events"][0]["tick"] += 1
    store.journal.write_bytes(k.framing.frame(changed_tape, 1, k.framing.GENESIS))
    with pytest.raises(ValueError, match="independent_transition_frame"):
        store.read()


def test_short_write_and_sync_failures(tape, tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8493-RECOVERY: partial writes complete; sync failures cannot acknowledge."""
    store = k.Store(tmp_path, tape)
    store.initialize()
    original = k.os.write
    monkeypatch.setattr(k.os, "write", lambda fd, data: original(fd, data[:7]))
    assert store.apply(tape["transactions"][0])
    monkeypatch.setattr(k.os, "write", lambda fd, data: 0)
    with pytest.raises(OSError, match="short_write"):
        store.apply(tape["transactions"][1])
    monkeypatch.setattr(k.os, "write", original)
    phases = []

    def failed(fd):
        raise OSError("failed_sync")

    monkeypatch.setattr(k.os, "fsync", failed)
    with pytest.raises(OSError, match="failed_sync"):
        store.apply(tape["transactions"][1], phases.append)
    assert "before_acknowledgment" not in phases


@pytest.mark.parametrize("barrier", k.BARRIERS)
def test_actual_process_exits(tape, tmp_path, barrier):
    """SCENARIO-VERIFY-8493-RECOVERY: every actual exit retains returned acknowledgments."""
    from carnot.reporting import append_store_qualification_8493 as r

    primitive = r.crash(tape, tmp_path / barrier, barrier)
    assert primitive["receipt"]["actual_exit"] == 73
    assert primitive["passed"]
    assert primitive["final"] == k.base.oracle(tape, len(tape["transactions"]))
    assert r.reduce(primitive)


def test_reporting_blocked_and_cli(tmp_path):
    """SCENARIO-REPORT-8493-TERMINAL: missing authority blocks and rehashed readiness rejects."""
    from carnot.reporting import append_store_qualification_8493 as r

    raw = tmp_path / "raw"
    work = r.measure(tmp_path / "missing", raw)
    atomic_json(raw / "measurement.json", work)
    value = r.build(work, [], raw, tmp_path / (r.NAME + ".json"))
    assert value["verdict_class"] == "blocked"
    path = tmp_path / (r.NAME + ".json")
    atomic_json(path, value)
    assert r.replay(path)
    value["append_store_ready_score"] = 1
    value["reproducibility_checksum"] = r.checksum(value)
    atomic_json(path, value)
    assert not r.replay(path)
    command = [sys.executable, "-u", str(r.ROOT / r.CLI)]
    for args, code in (
        (["--deliberate-error"], 7),
        (["--date", "20261012"], 2),
        (["--private-control"], 2),
        (["--cold-replay", str(tmp_path / "absent")], 1),
    ):
        child = subprocess.run(command + args, capture_output=True, timeout=30)
        assert child.returncode == code, child.stderr.decode()


def test_store_rejections_and_limits(tape, tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8493-RECOVERY: original limits and independent checkpoint checks."""
    store = k.Store(tmp_path, tape)
    store.initialize()
    seal = tmp_path / "initial-seal.json"
    seal_bytes = seal.read_bytes()
    seal.unlink()
    with pytest.raises(ValueError, match="missing_initial_seal"):
        store.initialize()
    seal.write_text("{}")
    with pytest.raises(ValueError, match="independent_initial_seal"):
        store.read()
    seal.write_bytes(seal_bytes)
    pointer = tmp_path / "checkpoint-current.json"
    saved = pointer.read_bytes()
    value = json.loads(saved)
    atomic_json(pointer, dict(value, sequence=99))
    with pytest.raises(ValueError, match="checkpoint_prefix"):
        store.read()
    pointer.write_bytes(saved)
    checkpoint = tmp_path / value["file"]
    original = checkpoint.read_bytes()
    altered = json.loads(original)
    altered["view"]["state"]["release_cursor"] = 9
    atomic_json(checkpoint, altered)
    with pytest.raises(ValueError, match="sealed_checkpoint"):
        store.read()
    checkpoint.write_bytes(original)
    with monkeypatch.context() as m:
        m.setattr(k.base, "JOURNAL_BYTES", 1)
        with pytest.raises(ValueError, match="journal_ceiling"):
            store.apply(tape["transactions"][0])
    with monkeypatch.context() as m:
        m.setattr(k.base.source.s, "MAX_EVENTS", 0)
        with pytest.raises(ValueError, match="event_budget"):
            store.apply(tape["transactions"][0])
    with monkeypatch.context() as m:
        m.setattr(k.base.source.s, "MAX_BYTES", 1)
        with pytest.raises(ValueError, match="state_byte_budget"):
            store.apply(tape["transactions"][0])
    original_step = k.base.source.step
    k.base.oracle(tape, 1)

    def wrong_step(*args):
        row = original_step(*args)
        args[2]["release_cursor"] += 1
        return row

    with monkeypatch.context() as m:
        m.setattr(k.base.source, "step", wrong_step)
        with pytest.raises(ValueError, match="independent_transition"):
            store.apply(tape["transactions"][0])
    assert store.read()["sequence"] == 0


def test_failed_namespace_sync_retried(tape, tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8493-RECOVERY: namespace failures retain sync obligations before retry."""
    store = k.Store(tmp_path, tape)
    store.initialize()
    sync = store._directory
    calls = []

    def failed(path):
        calls.append(path)
        raise OSError("namespace_sync_failed")

    with monkeypatch.context() as m:
        m.setattr(store, "_directory", failed)
        with pytest.raises(OSError):
            store.drain()
    assert store.namespace_pending

    def recorded(path):
        calls.append(path)
        sync(path)

    monkeypatch.setattr(store, "_directory", recorded)
    assert store.apply(tape["transactions"][0])
    assert len(calls) >= 2
    assert not store.namespace_pending


def test_small_worker_and_concurrent_clients(tape, tmp_path):
    """SCENARIO-VERIFY-8493-RECOVERY: real writers serialize duplicate requests."""
    from carnot.reporting import append_store_qualification_8493 as r

    small = deepcopy(tape)
    small["transactions"] = small["transactions"][:2]
    frozen = tmp_path / "frozen.json"
    atomic_json(frozen, small)
    command = [
        sys.executable,
        "-u",
        str(r.ROOT / r.CLI),
        "--worker",
        str(frozen),
        "--store",
        str(tmp_path / "worker"),
        "--barrier",
        "none",
    ]
    child = subprocess.run(command, capture_output=True, timeout=60)
    assert child.returncode == 0, child.stderr.decode()
    script = (
        "import json,sys; from pathlib import Path; "
        "from carnot.verify.append_store_8493 import Store; "
        "t=json.loads(Path(sys.argv[1]).read_bytes()); "
        "s=Store(Path(sys.argv[2]),t); s.initialize(); "
        'print(json.dumps(s.apply(t["transactions"][0])))'
    )
    children = [
        subprocess.Popen(
            [sys.executable, "-u", "-c", script, str(frozen), str(tmp_path / "concurrent")],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for _ in range(2)
    ]
    outputs = [c.communicate(timeout=60) for c in children]
    assert all(c.returncode == 0 for c in children), outputs
    assert outputs[0][0] == outputs[1][0]
    assert k.Store(tmp_path / "concurrent", small).read()["sequence"] == 1


def test_private_blocked_runner_and_manifest(tmp_path):
    """SCENARIO-VERIFY-8493-VALIDATION: exact collections and actual private publication."""
    from carnot.reporting import append_store_qualification_8493 as r
    from carnot.reporting import append_store_runner_8493 as runner

    private = tmp_path / "plan"
    private.mkdir()
    plan = runner.manifest(private)
    for spec in plan:
        if (
            spec["name"].startswith("collect_")
            and spec["collected_node_ids"][0].split("::")[0] != r.TEST
        ):
            nodes = [
                n
                for shard in plan
                if shard["name"].startswith("historical_")
                for n in shard["collected_node_ids"]
                if n.split("::")[0] == spec["collected_node_ids"][0].split("::")[0]
            ]
            assert nodes == spec["collected_node_ids"]
    assert "patch=subprocess,_exit" in (private / "coverage.ini").read_text()
    output = tmp_path / (r.NAME + ".json")
    child = subprocess.run(
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
        timeout=120,
    )
    assert child.returncode == 0, child.stdout.decode() + child.stderr.decode()
    assert json.loads(output.read_bytes())["verdict_class"] == "blocked"
    assert r.replay(output)


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    """SCENARIO-REPORT-8493-TERMINAL: authenticate and execute the complete original tape panel."""
    from carnot.reporting import append_store_qualification_8493 as r

    raw = tmp_path_factory.mktemp("qualified8493")
    result = r.measure(r.ROOT, raw)
    assert not result["gate_check_summary"], result["gate_check_summary"]
    assert all(row["passed"] for row in result["rows"]), result["owned_failures"]
    atomic_json(raw / "measurement.json", result)
    return result


def test_full_primitive_replay_and_negative_controls(work, tmp_path):
    """SCENARIO-REPORT-8493-TERMINAL: independently reconstructed states reject repaired evidence."""
    from carnot.reporting import append_store_qualification_8493 as r

    raw = Path(work["raw"])
    path = tmp_path / (r.NAME + ".json")
    value = r.build(work, [], raw, path)
    atomic_json(path, value)
    assert r.replay(path)
    assert not r.replay(tmp_path / "absent")
    for change in ("final", "responses", "acknowledgments", "recovered", "exit"):
        primitive = deepcopy(r.read(work["primitives"][2]))
        if change == "final":
            primitive["final"]["state"]["release_cursor"] += 1
        elif change == "responses":
            primitive["response_hash"] = "wrong"
        elif change == "acknowledgments":
            primitive["acknowledgments"] = []
        elif change == "recovered":
            primitive["recovered"]["state"]["release_cursor"] += 1
        else:
            primitive["actual_exit"] = 0
        assert not r.reduce(primitive), change
    for change in (
        "checksum",
        "rows",
        "code",
        "authority",
        "tapes",
        "history",
        "primitive",
        "store",
    ):
        altered = deepcopy(work)
        target = tmp_path / change
        altered["raw"] = str(target)
        if change == "rows":
            altered["rows"].pop()
        elif change == "code":
            altered["code_config_hashes"][0]["sha256"] = "wrong"
        elif change == "authority":
            altered["authority"]["tasks"][8]["prompt"] += " changed"
        elif change == "tapes":
            altered["tapes"][0]["transactions"][0]["events"][0]["tick"] += 1
        elif change == "history":
            altered["preserved_v728"]["pooled_speedup"] = 99
        elif change in {"primitive", "store"}:
            primitive = deepcopy(r.read(altered["primitives"][0]))
            if change == "primitive":
                primitive["final"]["sequence"] += 1
            else:
                primitive["store_references"][0]["sha256"] = "wrong"
            atomic_json(target / "primitive.json", primitive)
            ref = r.reference(target / "primitive.json")
            altered["primitives"][0] = ref
            altered["rows"][0]["primitive_reference"] = ref
        atomic_json(target / "measurement.json", altered)
        value = r.build(altered, [], target, path)
        if change == "checksum":
            value["reproducibility_checksum"] = "wrong"
        atomic_json(path, value)
        assert not r.replay(path), change


@pytest.mark.parametrize(
    "failure", ["authority", "deliverable", "readiness", "consumer", "sidecar", "protocol", "child"]
)
def test_admission_fail_closed(work, tmp_path, monkeypatch, failure):
    """SCENARIO-REPORT-8493-TERMINAL: external qualification never substitutes for required seals."""
    from carnot.reporting import append_store_qualification_8493 as r

    authority = deepcopy(work["authority"])
    monkeypatch.setattr(r.upstream.methods, "authority", lambda *args: authority)
    if failure == "authority":
        authority["activated"] = False
    elif failure == "deliverable":
        authority["tasks"][8]["deliverable"] = "results/foreign.json"
        monkeypatch.setattr(
            r.upstream.methods, "tasks_digest", lambda *a: r.upstream.methods.TASKS_PIN
        )
    elif failure == "readiness":
        original = r.read

        def changed(ref):
            value = original(ref)
            if value.get("experiment_id") == 8492:
                value["append_protocol_ready_score"] = 0
            return value

        monkeypatch.setattr(r, "read", changed)
    elif failure == "consumer":
        monkeypatch.setattr(r, "reader_receipt", lambda *a, **kw: dict(passed=False))
    elif failure == "sidecar":
        monkeypatch.setattr(r, "read_bound_sidecar", lambda *a: dict(report=dict(passed=False)))
    elif failure == "protocol":
        monkeypatch.setattr(k.framing, "protocol", lambda: {})
    else:
        monkeypatch.setattr(r, "child", lambda *a, **kw: dict(passed=False))
    result = r.measure(r.ROOT, tmp_path / "raw")
    assert result["gate_check_summary"]
    if failure == "readiness":
        assert result["gate_check_summary"][0]["observed"]["append_protocol_ready_score"] == 0
    assert all(row["status"] == "censored" for row in result["rows"])


def test_owned_failures_and_wrong_replies(work, tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8493-VALIDATION: resource, deadline and measurement failures disqualify."""
    from carnot.reporting import append_store_qualification_8493 as r

    with monkeypatch.context() as m:
        m.setattr(r.shutil, "which", lambda *a: None)
        result = r.measure(tmp_path / "missing", tmp_path / "resources")
        assert result["owned_failures"]
    inputs = dict(
        authority=work["authority"],
        refs=work["refs"],
        tapes=work["tapes"],
        upstream=r.read(r.reference(r.ROOT / r.UPSTREAM)),
        receipt=dict(passed=True),
    )
    monkeypatch.setattr(r, "admission", lambda *a: inputs)
    original_equivalence = r.equivalence

    def failed(*args):
        raise ValueError("deliberate_owned_failure")

    monkeypatch.setattr(r, "equivalence", failed)
    monkeypatch.setattr(r, "crash", failed)
    result = r.measure(r.ROOT, tmp_path / "failed")
    assert result["owned_failures"]
    assert result["rows"][0]["status"] == "failed"
    with monkeypatch.context() as m:
        counter = iter([0, 3601 * 10**9, 3602 * 10**9])
        m.setattr(r.time, "monotonic_ns", lambda: next(counter))
        result = r.measure(r.ROOT, tmp_path / "deadline")
        assert "measurement_deadline" in result["owned_failures"].values()
    with monkeypatch.context() as m:
        m.setattr(r, "equivalence", original_equivalence)
        m.setattr(k.base.Snapshot, "apply", lambda *a: [])
        with pytest.raises(ValueError, match="snapshot_response_parity"):
            r.equivalence(work["tapes"][0], tmp_path / "parity")


def test_genuine_coverage_copy_and_parser(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8493-VALIDATION: actual Coverage.py shards survive private runner copy."""
    import coverage
    from carnot.reporting import append_store_qualification_8493 as r
    from carnot.reporting import append_store_runner_8493 as runner

    private = tmp_path / "validation"
    private.mkdir()
    cov = coverage.Coverage(
        data_file=str(private / ".coverage.private"),
        include=[str(r.ROOT / r.OWNED[1])],
        config_file=False,
    )
    cov.start()
    r.progress("genuine_private_coverage")
    cov.stop()
    cov.save()
    cov.json_report(outfile=str(private / "coverage.json"))
    monkeypatch.setattr(runner, "manifest", lambda *a: [])
    output = tmp_path / (r.NAME + ".json")
    assert runner.run(tmp_path / "missing", output, private) == 0
    value = json.loads(output.read_bytes())
    assert value["owned_coverage"]["covered_lines"] > 0
    assert any(".coverage" in ref["path"] for ref in value["raw_shard_hashes"])
    assert value["append_store_ready_score"] == 0
    with pytest.raises(SystemExit) as error:
        runner.main(["--worker", str(tmp_path / "absent")])
    assert error.value.code == 2


def test_additional_disk_and_row_replay_rejections(work, tmp_path):
    """SCENARIO-REPORT-8493-TERMINAL: primitive metadata cannot overrule independently read disk."""
    from carnot.reporting import append_store_qualification_8493 as r

    primitive = deepcopy(r.read(work["primitives"][2]))
    changed = deepcopy(primitive)
    changed["recovered_state_hash"] = "wrong"
    assert not r.reduce(changed)
    changed = deepcopy(primitive)
    changed["namespace_operation_rows"] = []
    assert not r.reduce(changed)
    changed = deepcopy(primitive)
    directory = tmp_path / "crashed"
    shutil = __import__("shutil")
    shutil.copytree(Path(primitive["directory"]) / "crashed", directory / "crashed")
    atomic_json(
        directory / "operations-final.json", dict(rows=primitive["namespace_operation_rows"])
    )
    store = k.Store(directory / "crashed", r.read(primitive["frozen_reference"]))
    store.apply(store.frozen["transactions"][31])
    changed["directory"] = str(directory)
    assert not r.reduce(changed)
    for field in ("slot", "primitive_slot"):
        altered = deepcopy(work)
        raw = tmp_path / field
        altered["raw"] = str(raw)
        if field == "slot":
            altered["rows"][0]["arm"] = "foreign"
        else:
            p = deepcopy(r.read(altered["primitives"][0]))
            p["barrier"] = "foreign"
            atomic_json(raw / "primitive.json", p)
            ref = r.reference(raw / "primitive.json")
            altered["primitives"][0] = ref
            altered["rows"][0]["primitive_reference"] = ref
        atomic_json(raw / "measurement.json", altered)
        output = tmp_path / (r.NAME + ".json")
        atomic_json(output, r.build(altered, [], raw, output))
        assert not r.replay(output)


def test_missing_sealed_input_hash(work, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8493-TERMINAL: hash read failure is distinct from an observed readiness zero."""
    from carnot.reporting import append_store_qualification_8493 as r

    monkeypatch.setattr(r.upstream.methods, "authority", lambda *a: deepcopy(work["authority"]))

    def fail(ref):
        raise ValueError("required_hash_mismatch")

    monkeypatch.setattr(r, "read", fail)
    result = r.measure(r.ROOT, tmp_path / "raw")
    gate = result["gate_check_summary"][0]
    assert not gate["missing"] and gate["expected_sha256"] == r.UPSTREAM_PIN
    assert gate["observed"] is False
    assert gate["field"] == "authenticated_input_readable"
