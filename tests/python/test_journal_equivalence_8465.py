"""REQ-VERIFY-8465, REQ-REPORT-8465: independently replay durable journal work."""

from copy import deepcopy
import json
import os
from pathlib import Path

import pytest

from carnot.verify import journal_equivalence_8465 as k
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from test_persistence_localization_8464 import bundle  # noqa: F401


@pytest.fixture
def frozen(bundle):  # noqa: F811
    """SCENARIO-VERIFY-8465-EQUIVALENCE: fixed operands precede journal writes."""
    return k.freeze(bundle, dict(workload="natural", batch=1, predictions_per_update=8), 0)


@pytest.mark.parametrize("batch", [1, 8])
def test_exact_service_and_duplicates(bundle, tmp_path, batch):  # noqa: F811
    """SCENARIO-VERIFY-8465-EQUIVALENCE: both stores preserve replies and causal state."""
    f = k.freeze(bundle, dict(workload="natural", batch=batch, predictions_per_update=8), 0)
    journal, snapshot = k.Journal(tmp_path / "journal", f), k.Snapshot(tmp_path / "snapshot", f)
    journal.initialize()
    snapshot.initialize()
    pinned = journal.read()
    for tx in f["transactions"]:
        assert journal.apply(tx) == snapshot.apply(tx)
    journal.drain()
    assert journal.read() == snapshot.read() == k.oracle(f, len(f["transactions"]))
    assert pinned["sequence"] == 0
    last = f["transactions"][-1]
    before = journal.read()
    assert journal.apply(last) == before["replies"][-1]
    assert snapshot.apply(last) == before["replies"][-1]
    assert journal.read() == before
    bad = deepcopy(last)
    bad["events"][0]["tick"] += 1
    for store in (journal, snapshot):
        with pytest.raises(ValueError, match="conflict"):
            store.apply(bad)
    assert before["state"]["release_cursor"] == 8 and before["state"]["pending"] == []
    assert (journal.path / "checkpoint-current.json").is_file()


def test_journal_tail_corruption_and_independent_tamper(frozen, tmp_path):
    """SCENARIO-VERIFY-8465-RECOVERY: only an uncommitted final tail may disappear."""
    store = k.Journal(tmp_path, frozen)
    store.initialize()
    for tx in frozen["transactions"][:2]:
        store.apply(tx)
    before = store.read()
    log = tmp_path / "events.jsonl"
    with log.open("ab") as stream:
        stream.write(b'{"unfinished":')
    assert store.read() == before
    original = log.read_bytes()
    rows = [json.loads(line) for line in original.splitlines()]
    log.write_bytes(original[:-4])
    with pytest.raises(ValueError, match="acknowledged"):
        store.read()
    log.write_bytes(original)
    rows[0]["payload"]["events"][0]["tick"] += 1
    rows[0] = k.record(rows[0]["payload"], 1, k.GENESIS, frozen)
    rows[1] = k.record(rows[1]["payload"], 2, rows[0]["digest"], frozen)
    log.write_bytes(b"".join(k.encode(row) + b"\n" for row in rows))
    for row in rows:
        atomic_json(tmp_path / f"commit-{row['sequence']:04d}.json", k.certificate(row))
    with pytest.raises(ValueError, match="sealed"):
        store.read()


@pytest.mark.parametrize("mutation", ["reorder", "interior", "checkpoint", "seal", "receipt"])
def test_corruption_controls(frozen, tmp_path, mutation):
    """SCENARIO-VERIFY-8465-RECOVERY: checks use independent operands and sequence."""
    store = k.Journal(tmp_path, frozen)
    store.initialize()
    for tx in frozen["transactions"][:2]:
        store.apply(tx)
    store.drain()
    log = tmp_path / "events.jsonl"
    if mutation == "reorder":
        lines = log.read_bytes().splitlines(keepends=True)
        log.write_bytes(b"".join(reversed(lines)))
    elif mutation == "interior":
        log.write_bytes(b"garbage\n" + log.read_bytes())
    elif mutation == "checkpoint":
        atomic_json(tmp_path / "checkpoint-current.json", dict(sequence=0, digest=k.GENESIS))
    elif mutation == "seal":
        atomic_json(tmp_path / "initial-seal.json", dict(sha256="forged"))
    else:
        (tmp_path / "commit-0001.json").unlink()
    with pytest.raises((ValueError, KeyError)):
        store.read()


def test_limits_and_bad_order(frozen, tmp_path, monkeypatch):
    """REQ-VERIFY-8465: overflow blocks before acknowledged history is changed."""
    store = k.Journal(tmp_path, frozen)
    store.initialize()
    with pytest.raises(ValueError, match="sealed"):
        store.apply(frozen["transactions"][1])
    monkeypatch.setattr(k, "JOURNAL_BYTES", 1)
    with pytest.raises(ValueError, match="journal_ceiling"):
        store.apply(frozen["transactions"][0])
    monkeypatch.setattr(k, "JOURNAL_BYTES", 16 * 1024 * 1024)
    monkeypatch.setattr(k.source.s, "MAX_EVENTS", 0)
    with pytest.raises(ValueError, match="event_budget"):
        store.apply(frozen["transactions"][0])
    monkeypatch.setattr(k.source.s, "MAX_EVENTS", 256)
    monkeypatch.setattr(k.source.s, "MAX_BYTES", 1)
    with pytest.raises(ValueError, match="state_byte_budget"):
        store.apply(frozen["transactions"][0])
    assert store.read()["sequence"] == 0


def test_process_crashes_and_cold_replay(work, tmp_path):
    """SCENARIO-VERIFY-8465-RECOVERY: real exits retain both stores and exact ledgers."""
    from carnot.reporting import journal_equivalence_8465 as r

    assert work["gate_check_summary"] == []
    assert len(work["rows"]) == 18
    assert all(row["passed"] for row in work["rows"])
    assert all(row["actual_exit"] == 73 for row in work["rows"] if row["kind"] == "crash")
    receipts = [dict(name=name, passed=True) for name in r.REQUIRED]
    value = r.build(work, receipts, tmp_path / (r.NAME + ".json"))
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, value)
    assert r.replay(candidate)
    value["journal_ready_score"] = 999
    value["reproducibility_checksum"] = r.checksum(value)
    atomic_json(candidate, value)
    assert not r.replay(candidate)


def test_missing_and_owned_failure(tmp_path):
    """SCENARIO-REPORT-8465-TERMINAL: missing operands block while owned failure disqualifies."""
    from carnot.reporting import journal_equivalence_8465 as r

    work = r.measure(tmp_path / "absent", tmp_path / "raw", tmp_path)
    receipts = [dict(name=name, passed=True) for name in r.REQUIRED]
    value = r.build(work, receipts, tmp_path / (r.NAME + ".json"))
    assert value["verdict_class"] == "blocked"
    assert value["censored_count"] == value["intended_count"]
    assert any(g["observed"] is None for g in value["gate_check_summary"])
    assert (
        r.build(work, [dict(name="fail", passed=False)], tmp_path / "out")["verdict_class"]
        == "disqualified"
    )
    atomic_json(tmp_path / "candidate.json", value)
    assert r.replay(tmp_path / "candidate.json") and not r.replay(tmp_path / "absent")


def test_real_private_cli(tmp_path):
    """SCENARIO-REPORT-8465-TERMINAL: actual child publication and negative replay stay private."""
    import subprocess
    import sys
    from carnot.reporting import journal_equivalence_8465 as r
    from carnot.reporting import journal_equivalence_runner_8465 as runner

    output = tmp_path / "results" / (r.NAME + ".json")
    process = subprocess.run(
        [
            sys.executable,
            "-u",
            str(r.ROOT / r.CLI),
            "--root",
            str(tmp_path / "absent"),
            "--output",
            str(output),
            "--private-control",
        ],
        capture_output=True,
        timeout=120,
    )
    assert process.returncode == 0, process.stderr.decode()
    assert json.loads(output.read_bytes())["verdict_class"] == "blocked"
    assert runner.main(["--deliberate-error"]) == 7
    assert runner.main(["--cold-replay", str(output)]) == 0
    assert runner.main(["--cold-replay", str(tmp_path / "absent")]) == 1
    assert runner.main(["--private-control"]) == 2
    with pytest.raises(SystemExit):
        runner.main(["--date", "20261011"])


@pytest.mark.parametrize("arm", ["journal", "snapshot"])
def test_concurrent_reader_pins_old_state(frozen, tmp_path, arm):
    """SCENARIO-VERIFY-8465-EQUIVALENCE: real readers retain complete states while a writer advances."""
    import subprocess
    import sys
    import time
    from carnot.reporting import journal_equivalence_8465 as r

    directory = tmp_path / arm
    frozen_path = tmp_path / "frozen.json"
    atomic_json(frozen_path, frozen)
    store = (k.Journal if arm == "journal" else k.Snapshot)(directory, frozen)
    store.initialize()
    store.apply(frozen["transactions"][0])
    pin = tmp_path / "pin.json"
    release = tmp_path / "release"
    process = subprocess.Popen(
        [
            sys.executable,
            "-u",
            str(r.ROOT / r.CLI),
            "--reader",
            str(frozen_path),
            "--store",
            str(directory),
            "--arm",
            arm,
            "--pin",
            str(pin),
            "--release",
            str(release),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 30
        while not pin.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert pin.exists()
        pinned = json.loads(pin.read_bytes())
        store.apply(frozen["transactions"][1])
        release.write_text("release")
        out, err = process.communicate(timeout=30)
        assert process.returncode == 0, (out, err)
        assert pinned == k.oracle(frozen, 1)
        assert store.read() == k.oracle(frozen, 2)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_extra_store_failure_paths(frozen, tmp_path):
    """SCENARIO-VERIFY-8465-RECOVERY: neither missing seals nor forged checkpoints create state."""
    store = k.Journal(tmp_path / "journal", frozen)
    store.initialize()
    store.initialize()
    store.apply(frozen["transactions"][0])
    expected = store.read()["ledger"]
    with pytest.raises(ValueError, match="acknowledged_ledger_loss"):
        store.read(acknowledged=expected + ["absent"])
    assert store.read(acknowledged=expected)["sequence"] == 1
    pointer = json.loads((store.path / "checkpoint-current.json").read_bytes())
    checkpoint = store.path / pointer["file"]
    envelope = json.loads(checkpoint.read_bytes())
    envelope["view"]["state"]["release_cursor"] = 99
    atomic_json(checkpoint, envelope)
    pointer["sha256"] = canonical_hash(envelope)
    atomic_json(store.path / "checkpoint-current.json", pointer)
    with pytest.raises(ValueError, match="sealed_checkpoint"):
        store.read()
    pointer["sequence"] = 999
    atomic_json(store.path / "checkpoint-current.json", pointer)
    with pytest.raises(ValueError, match="checkpoint_prefix"):
        store.read()
    (store.path / "initial-seal.json").unlink()
    with pytest.raises(ValueError, match="missing_initial_seal"):
        store.initialize()
    snapshot = k.Snapshot(tmp_path / "snapshot", frozen)
    snapshot.initialize()
    state = k.oracle(frozen, 0)["state"]
    state["version"] = 999
    snapshot.store.commit(state)
    with pytest.raises(ValueError, match="snapshot_sequence"):
        snapshot.read()
    state["version"] = 0
    state["release_cursor"] = 99
    snapshot.store.commit(state)
    with pytest.raises(ValueError, match="sealed_snapshot"):
        snapshot.read()


def test_cached_oracle_is_independent_and_returns_copies(frozen):
    """SCENARIO-VERIFY-8465-EQUIVALENCE: memoized prefixes never share mutable caller state."""
    original = k.oracle(frozen, 2)
    changed = k.oracle(frozen, 2)
    changed["state"]["release_cursor"] = 999
    assert k.oracle(frozen, 2) == original
    with pytest.raises(ValueError, match="oracle_prefix"):
        k.oracle(frozen, len(frozen["transactions"]) + 1)


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    """REQ-REPORT-8465: reuse one real retained process panel for independent mutations."""
    from carnot.reporting import journal_equivalence_8465 as r

    private = tmp_path_factory.mktemp("journal-process-panel")
    return r.measure(r.ROOT, private / "raw", private)


def test_resources_deadline_and_owned_failure(work, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8465-TERMINAL: operands and owned errors retain every intended slot."""
    from carnot.reporting import journal_equivalence_8465 as r

    deps = {key: work[key] for key in ("bundle", "workload", "refs", "historical_model_provenance")}
    monkeypatch.setattr(r, "dependencies", lambda *args: deepcopy(deps))
    monkeypatch.setattr(
        r.upstream.authority, "authority", lambda *args: deepcopy(work["authority"])
    )
    with monkeypatch.context() as control:
        control.setattr(r.os, "access", lambda *args: False)
        control.setattr(r.shutil, "disk_usage", lambda *args: type("Disk", (), {"free": 0})())
        control.setattr(r.upstream, "filesystem", lambda *args: dict(disk_backed=False))
        observed = r.measure(r.ROOT, tmp_path / "missing-resources", tmp_path)
        assert {"disk_backed", "disk_free", "tool_available"} <= {
            g["field"] for g in observed["gate_check_summary"]
        }
    with monkeypatch.context() as control:
        counter = [time_ns := r.time.monotonic_ns()]

        def clock():
            counter[0] += 3601 * 10**9
            return counter[0]

        control.setattr(r.time, "monotonic_ns", clock)
        observed = r.measure(r.ROOT, tmp_path / "deadline", tmp_path)
        assert len(observed["owned_failures"]) == 18 and time_ns > 0
    with monkeypatch.context() as control:

        def fail(*args):
            raise ValueError("owned_deliberate_failure")

        control.setattr(r, "equivalence", fail)
        control.setattr(r, "evidence", fail)
        observed = r.measure(r.ROOT, tmp_path / "owned-failure", tmp_path)
        assert len(observed["owned_failures"]) == 18
    with monkeypatch.context() as control:
        authority = deepcopy(work["authority"])
        next(t for t in authority["tasks"] if t["id"] == r.TASK)["MODEL_SPECS"] = ["wrong"]
        control.setattr(r.upstream.authority, "authority", lambda *args: authority)
        observed = r.measure(r.ROOT, tmp_path / "wrong-task", tmp_path)
        assert any(g["field"] == "exact_task_contract" for g in observed["gate_check_summary"])


def test_worker_success_reader_timeout_and_manifest(frozen, tmp_path):
    """SCENARIO-VERIFY-8465-VALIDATION: real worker completion, reader timeout and child coverage."""
    from carnot.reporting import journal_equivalence_8465 as r
    from carnot.reporting import journal_equivalence_runner_8465 as runner

    tiny = deepcopy(frozen)
    tiny["transactions"] = tiny["transactions"][:2]
    path = tmp_path / "frozen.json"
    atomic_json(path, tiny)
    assert r.worker(path, tmp_path / "worker", "journal", "none") == 0
    assert (
        r.reader(
            path,
            tmp_path / "worker/store",
            "journal",
            tmp_path / "pin",
            tmp_path / "absent",
            timeout_s=0,
        )
        == 1
    )
    plan = runner.manifest(tmp_path / "coverage")
    assert {"owned_tests", "private_E2E018_consumers", "private_E2E021", "coverage_report"} <= {
        p["name"] for p in plan
    }
    assert "patch=subprocess,_exit" in (tmp_path / "coverage/coverage.ini").read_text()


@pytest.mark.parametrize(
    "mutation",
    [
        "identity",
        "authority",
        "bundle",
        "source",
        "roster",
        "frozen",
        "store",
        "final",
        "response",
        "acknowledgments",
        "recovered",
        "receipt",
        "row",
        "lost_row",
    ],
)
def test_independent_rehashed_replay_controls(work, tmp_path, mutation):
    """SCENARIO-REPORT-8465-TERMINAL: repaired candidate and primitive hashes do not qualify altered meaning."""
    from carnot.reporting import journal_equivalence_8465 as r

    changed = deepcopy(work)
    candidate = tmp_path / "candidate.json"
    index = 2
    primitive = r.read(changed["rows"][index]["primitive_reference"])
    if mutation == "authority":
        changed["authority"]["tasks"][0]["phase"] = 999
    elif mutation == "bundle":
        changed["bundle"]["head"]["coefficients"][0] += 1
    elif mutation == "lost_row":
        changed["rows"].pop()
    elif mutation == "roster":
        changed["rows"][0]["barrier"] = "wrong"
    elif mutation == "frozen":
        operands = r.read(primitive["frozen_reference"])
        operands["cell"]["batch"] = 32
        atomic_json(tmp_path / "frozen.json", operands)
        primitive["frozen_reference"] = r.reference(tmp_path / "frozen.json")
    elif mutation == "store":
        primitive["store_references"][0]["sha256"] = "wrong"
    elif mutation == "final":
        primitive["final"]["state"]["release_cursor"] += 1
    elif mutation == "response":
        primitive["response_hash"] = "wrong"
    elif mutation == "acknowledgments":
        primitive["acknowledgments"].pop()
    elif mutation == "recovered":
        primitive["recovered"]["state"]["release_cursor"] += 1
    elif mutation == "receipt":
        primitive["receipt"]["actual_exit"] = 0
    elif mutation == "row":
        changed["rows"][index]["recovered_state_hash"] = "wrong"
    atomic_json(tmp_path / "primitive.json", primitive)
    changed["rows"][index]["primitive_reference"] = r.reference(tmp_path / "primitive.json")
    changed["raw"] = str(tmp_path)
    atomic_json(tmp_path / "measurement.json", changed)
    value = r.build(changed, [dict(name=n, passed=True) for n in r.REQUIRED], candidate)
    if mutation == "identity":
        value["experiment_id"] = 999
    elif mutation == "source":
        value["source_artifact_hashes"][0]["sha256"] = "wrong"
    value["reproducibility_checksum"] = r.checksum(value)
    atomic_json(candidate, value)
    assert not r.replay(candidate)


def test_additional_sealed_sequence_and_truncated_record(frozen, tmp_path):
    """SCENARIO-VERIFY-8465-RECOVERY: valid JSON without its committed delimiter is still truncated."""
    store = k.Journal(tmp_path / "truncated", frozen)
    store.initialize()
    store.apply(frozen["transactions"][0])
    path = store.path / "events.jsonl"
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="acknowledged_record_truncated"):
        store.read()
    empty = deepcopy(frozen)
    empty["transactions"] = []
    other = k.Journal(tmp_path / "extra", empty)
    other.initialize()
    row = k.record(frozen["transactions"][0], 1, k.GENESIS, empty)
    (other.path / "events.jsonl").write_bytes(k.encode(row) + b"\n")
    atomic_json(other.path / "commit-0001.json", k.certificate(row))
    with pytest.raises(ValueError, match="sealed_sequence"):
        other.read()


@pytest.mark.parametrize("mutation", ["ready", "tape"])
def test_upstream_qualification_controls(work, tmp_path, monkeypatch, mutation):
    """REQ-REPORT-8465: original producer seals and exact tape are independent required operands."""
    from carnot.reporting import journal_equivalence_8465 as r

    original = r.read

    def changed(ref):
        value = deepcopy(original(ref))
        if str(ref["path"]).endswith(r.UPSTREAM) and mutation == "ready":
            value["persistence_profile_ready_score"] = 0
        if str(ref["path"]).endswith("frozen_workload.json") and mutation == "tape":
            value["event_tape"][0]["tick"] += 1
        return value

    monkeypatch.setattr(r, "read", changed)
    with pytest.raises(ValueError, match="qualified_8464_tape|independently_sealed_tape"):
        r.dependencies(r.ROOT, tmp_path, tmp_path)


def test_private_measured_runner_with_genuine_coverage(work, tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8465-VALIDATION: retain actual coverage shards and primitive negative child checks."""
    import subprocess
    import sys
    from carnot.reporting import journal_equivalence_8465 as r
    from carnot.reporting import journal_equivalence_runner_8465 as runner

    private = tmp_path / "validation"
    runner.manifest(private)
    rc = "--rcfile=" + str(private / "coverage.ini")
    for args, expected in [
        (["run", rc, str(r.ROOT / r.CLI), "--deliberate-error"], 7),
        (["combine", rc, "--keep"], 0),
        (["json", rc, "-o", str(private / "coverage.json")], 0),
    ]:
        process = subprocess.run(
            [sys.executable, "-m", "coverage", *args], capture_output=True, timeout=60
        )
        assert process.returncode == expected, process.stderr.decode()

    def saved(root, raw, scratch):
        changed = deepcopy(work)
        changed["raw"] = str(raw)
        atomic_json(raw / "measurement.json", changed)
        return changed

    monkeypatch.setattr(r, "measure", saved)
    monkeypatch.setattr(runner, "manifest", lambda *args: [])
    output = tmp_path / "results" / (r.NAME + ".json")
    assert runner.run(r.ROOT, output, private) == 0
    value = json.loads(output.read_bytes())
    assert value["owned_coverage"]["covered_lines"] > 0
    assert any(".coverage" in ref["path"] for ref in value["raw_shard_hashes"])
    assert value["journal_ready_score"] == 0
    assert next(c for c in value["validation_receipts"] if c["name"] == "rehashed_primitive")[
        "passed"
    ]


def test_oracle_rejects_changed_sealed_label_bytes(frozen):
    """SCENARIO-VERIFY-8465-RECOVERY: cached arithmetic cannot absorb a changed label file."""
    path = Path(frozen["bundle"]["labels"]["path"])
    original = path.read_bytes()
    k.oracle(frozen, 2)
    changed = json.loads(original)
    changed["rows"][15]["y"] = 1 - changed["rows"][15]["y"]
    try:
        atomic_json(path, changed)
        with pytest.raises(ValueError, match="independent_label_seal"):
            k.oracle(frozen, 2)
    finally:
        path.write_bytes(original)


def test_equivalence_detects_changed_replies_and_independent_expectation(
    frozen, tmp_path, monkeypatch
):
    """SCENARIO-VERIFY-8465-EQUIVALENCE: equal arithmetic alone cannot replace the sealed producer expectation."""
    from carnot.reporting import journal_equivalence_8465 as r

    tiny = deepcopy(frozen)
    tiny["transactions"] = tiny["transactions"][:1]
    expected = dict(state_hash="wrong", response_hash="wrong")
    with monkeypatch.context() as control:
        original = k.Snapshot.apply

        def changed(self, tx):
            response = original(self, tx)
            response[0]["probability"] = 999
            return response

        control.setattr(k.Snapshot, "apply", changed)
        with pytest.raises(ValueError, match="response_equivalence"):
            r.equivalence(tiny, tmp_path / "replies", expected)
    with pytest.raises(ValueError, match="independent_8464_expectation"):
        r.equivalence(tiny, tmp_path / "independent", expected)


def test_pinned_reader_detects_reclamation(frozen, tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8465-EQUIVALENCE: losing an old version while pinned is a detected failure."""
    from carnot.reporting import journal_equivalence_8465 as r

    path = tmp_path / "frozen.json"
    atomic_json(path, frozen)
    store = k.Journal(tmp_path / "store", frozen)
    store.initialize()
    release = tmp_path / "release"

    def reclaim(seconds):
        (store.path / "checkpoint-0000.json").unlink()
        release.write_text("released")

    monkeypatch.setattr(r.time, "sleep", reclaim)
    assert r.reader(path, store.path, "journal", tmp_path / "pin.json", release) == 1


@pytest.mark.parametrize("mutation", ["incomplete_final", "bad_prefix", "marker", "recovered_hash"])
def test_retained_store_replay_negative_controls(work, tmp_path, mutation):
    """SCENARIO-REPORT-8465-TERMINAL: semantically valid older states cannot hide lost acknowledgments."""
    import shutil
    from carnot.reporting import journal_equivalence_8465 as r

    changed = deepcopy(work)
    index = 0 if mutation == "incomplete_final" else 10
    primitive = r.read(changed["rows"][index]["primitive_reference"])
    frozen = r.read(primitive["frozen_reference"])
    if mutation == "recovered_hash":
        primitive["recovered_state_hash"] = "wrong"
    else:
        copy = tmp_path / "copied"
        shutil.copytree(primitive["directory"], copy)
        primitive["directory"] = str(copy)
        if mutation == "marker":
            atomic_json(copy / "crash_marker.json", dict(barrier="wrong", sequence=32))
        else:
            store = copy / ("snapshot" if mutation == "incomplete_final" else "crashed")
            version = 1 if mutation == "incomplete_final" else 30
            target = next(
                p
                for p in store.glob("version-*.json")
                if json.loads(p.read_bytes())["state"]["version"] == version
            )
            envelope = json.loads(target.read_bytes())
            atomic_json(store / "current.json", dict(file=target.name, sha256=envelope["sha256"]))
            if mutation == "bad_prefix":
                primitive["recovered"] = k.oracle(frozen, 30)
    atomic_json(tmp_path / "primitive.json", primitive)
    changed["rows"][index]["primitive_reference"] = r.reference(tmp_path / "primitive.json")
    changed["raw"] = str(tmp_path)
    atomic_json(tmp_path / "measurement.json", changed)
    value = r.build(
        changed, [dict(name=n, passed=True) for n in r.REQUIRED], tmp_path / "candidate.json"
    )
    atomic_json(tmp_path / "candidate.json", value)
    assert not r.replay(tmp_path / "candidate.json")


def test_snapshot_rejects_signed_zero_rehash(frozen, tmp_path):
    """SCENARIO-VERIFY-8465-RECOVERY: bit-for-bit identity includes zero signs after attacker rehashing."""
    store = k.Snapshot(tmp_path, frozen)
    store.initialize()
    state = k.oracle(frozen, 0)["state"]
    state["head"]["coefficients"][3] = -0.0
    store.store.commit(state)
    with pytest.raises(ValueError, match="sealed_snapshot"):
        store.read()


def test_duplicate_numeric_representation_is_conflicting(frozen, tmp_path):
    """SCENARIO-VERIFY-8465-EQUIVALENCE: the same ID binds exact original operand types."""
    for arm, cls in [("journal", k.Journal), ("snapshot", k.Snapshot)]:
        store = cls(tmp_path / arm, frozen)
        store.initialize()
        store.apply(frozen["transactions"][0])
        changed = deepcopy(frozen["transactions"][0])
        changed["events"][0]["tick"] = float(changed["events"][0]["tick"])
        with pytest.raises(ValueError, match="duplicate_conflict"):
            store.apply(changed)
