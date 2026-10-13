"""REQ-REPORT-8492, REQ-VERIFY-8492: sealed clocks and recovery fixtures."""

from copy import deepcopy
from pathlib import Path
import struct

import pytest

from carnot.verify import publication_cost_model_8492 as k


def frozen():
    """SCENARIO-VERIFY-8492-FIXTURES: independent tape rejects repaired hashes."""
    return dict(
        transactions=[dict(id="tx1", events=[dict(kind="prediction", tick=1)], request_ids=["q1"])],
        code_hash="code",
        oracle_hash="oracle",
        config_hash="config",
    )


def test_frames_and_recovery():
    """SCENARIO-VERIFY-8492-FIXTURES: only a final incomplete frame is discarded."""
    tape = frozen()
    data = k.frame(tape, 1, k.GENESIS)
    assert k.recover(data, tape)["committed_ids"] == ["tx1"]
    assert k.recover(data + b"\x00", tape)["discarded_tail_bytes"] == 1
    assert k.recover(data + struct.pack(">I", 100) + b"tail", tape)["discarded_tail_bytes"] == 8
    assert k.recover(b"", tape)["sequence"] == 0
    with pytest.raises(ValueError):
        k.recover(struct.pack(">I", 0), tape)
    with pytest.raises(ValueError):
        k.recover(data[:-1] + b"!", tape)
    with pytest.raises(ValueError):
        k.recover(data + data, tape)
    altered = deepcopy(tape)
    altered["transactions"][0]["events"][0]["tick"] = 2
    with pytest.raises(ValueError):
        k.recover(k.frame(altered, 1, k.GENESIS), tape)


def test_sync_order_and_limits():
    """SCENARIO-VERIFY-8492-FIXTURES: sync and checkpoint precede observable release."""
    good = [
        "create",
        "file_sync",
        "directory_sync",
        "append",
        "file_sync",
        "ack",
        "checkpoint",
        "file_sync",
        "directory_sync",
        "reclaim",
        "directory_sync",
    ]
    assert k.ordering(good)
    for bad in (
        ["append", "ack"],
        ["create", "append"],
        ["reclaim"],
        ["checkpoint", "file_sync", "reclaim"],
        ["unknown"],
        ["create"],
    ):
        with pytest.raises(ValueError):
            k.ordering(bad)
    assert k.protocol()["MAX_EVENTS"] == 256
    assert k.protocol()["MAX_BYTES"] == 2 * 1024 * 1024
    assert k.protocol()["journal_ceiling_bytes"] == 16 * 1024 * 1024


def test_namespace_counts():
    """SCENARIO-REPORT-8492-SEALED: all original synchronization calls stay charged."""
    for arm in ("snapshot", "journal"):
        row = k.namespace(arm, 72)
        assert row["total_fsyncs"] == (292 if arm == "snapshot" else 243)
        assert row["directory_syncs_without_namespace_change"] == 0
    with pytest.raises(ValueError):
        k.namespace("append", 72)


def test_disjoint_costs():
    """SCENARIO-REPORT-8492-SEALED: residuals and initial sync costs remain in bounds."""
    stages = [dict(name=n, start_ns=i * 10, end_ns=(i + 1) * 10) for i, n in enumerate(k.STAGES)]
    primitive = dict(
        start_ns=0,
        end_ns=2000,
        initialized_ns=1000,
        persistence_spans=[
            dict(start_ns=0, end_ns=120, stages=stages),
            dict(
                start_ns=1000,
                end_ns=1120,
                stages=[
                    dict(s, start_ns=s["start_ns"] + 1000, end_ns=s["end_ns"] + 1000)
                    for s in stages
                ],
            ),
        ],
    )
    row = k.partition(primitive)
    assert sum(row["category_ns"].values()) == 2000
    assert sum(row["fractions"].values()) == pytest.approx(1)
    assert row["removable_transaction_ns"] == 50
    with pytest.raises(ValueError):
        k.partition(dict(primitive, end_ns=1))
    changed = deepcopy(primitive)
    changed["persistence_spans"][0]["stages"][0]["start_ns"] = 1
    with pytest.raises(ValueError):
        k.partition(changed)


def test_conditional_bound():
    """SCENARIO-REPORT-8492-SEALED: counterfactuals never claim observed speedup."""
    assert k.ceiling(100, 50)["conditional_speed_ceiling"] == 2
    assert k.ceiling(100, 101)["conditional_speed_ceiling"] is None
    assert k.ceiling(100, 50)["measured_speedup"] is None


def test_interior_length_corruption_and_retry():
    """SCENARIO-VERIFY-8492-FIXTURES: an interior bad length cannot hide a later record."""
    import json

    tape = frozen()
    tape["transactions"].append(dict(tape["transactions"][0], id="tx2"))
    first = k.frame(tape, 1, k.GENESIS)
    second = k.frame(tape, 2, json.loads(first[4:])["digest"])
    with pytest.raises(ValueError):
        k.recover(struct.pack(">I", 100000) + first[4:] + second, tape)
    assert k.recover(first + second[:-5], tape)["sequence"] == 1
    with pytest.raises(ValueError):
        k.ordering(["create", "directory_sync"])
    conflict = deepcopy(tape)
    conflict["transactions"][1]["id"] = "tx1"
    with pytest.raises(ValueError, match="conflicting_id"):
        k.recover(
            k.frame(conflict, 1, k.GENESIS) + k.frame(conflict, 2, json.loads(first[4:])["digest"]),
            conflict,
        )
    with pytest.raises(ValueError):
        k.partition(dict(start_ns=1, end_ns=1, initialized_ns=1, persistence_spans=[]))
    with pytest.raises(ValueError):
        k.partition(
            dict(
                start_ns=0,
                end_ns=20,
                initialized_ns=0,
                persistence_spans=[dict(start_ns=0, end_ns=20, stages=[])],
            )
        )


def test_reporting_blocked_and_tamper(tmp_path):
    """SCENARIO-VERIFY-8492-CLI: absent external authority is blocked, not partial."""
    from carnot.reporting import publication_cost_model_8492 as r
    from carnot.reporting.current_work_receipt import atomic_json

    raw = tmp_path / "raw"
    work = r.measure(tmp_path / "missing", raw)
    atomic_json(raw / "measurement.json", work)
    output = tmp_path / (r.NAME + ".json")
    value = r.build(work, [], raw, output)
    assert value["verdict_class"] == "blocked"
    assert value["append_protocol_ready_score"] == 0
    assert value["censored_count"] == value["intended_count"]
    atomic_json(output, value)
    assert r.replay(output)
    value["append_protocol_ready_score"] = 1
    value["reproducibility_checksum"] = r.checksum(value)
    atomic_json(output, value)
    assert not r.replay(output)
    assert not r.replay(tmp_path / "absent.json")
    work["owned_failures"] = ["deliberate_validation_failure"]
    assert r.build(work, [], raw, output)["verdict_class"] == "disqualified"


def test_cli_private_controls(tmp_path):
    """SCENARIO-VERIFY-8492-CLI: actual child exits are independently asserted."""
    import json
    import subprocess
    import sys
    from carnot.reporting import publication_cost_model_8492 as r

    command = [sys.executable, "-u", str(r.ROOT / r.CLI)]
    for args, expected in (
        (["--deliberate-error"], 7),
        (["--date", "20261012"], 2),
        (["--private-control"], 2),
        (["--cold-replay", str(tmp_path / "absent")], 1),
        (["--history-request", str(tmp_path / "absent")], 2),
    ):
        p = subprocess.run(command + args, capture_output=True, timeout=30)
        assert p.returncode == expected, p.stderr.decode()
    output = tmp_path / (r.NAME + ".json")
    p = subprocess.run(
        command
        + ["--private-control", "--root", str(tmp_path / "missing"), "--output", str(output)],
        capture_output=True,
        timeout=90,
    )
    assert p.returncode == 0, p.stdout.decode() + p.stderr.decode()
    assert json.loads(output.read_bytes())["verdict_class"] == "blocked"


def test_real_historical_reduction(tmp_path):
    """SCENARIO-REPORT-8492-SEALED: read all original clocks and test independent bounds."""
    from carnot.reporting import publication_cost_model_8492 as r

    refs = [
        r.reference(next((r.ROOT / "results").glob(f"experiment_{n}_*.json")))
        for n in (8464, 8465, 8466)
    ]
    result = r.reconstruct(refs)
    assert len(result["rows"]) == 36
    assert len(result["stage_rows"]) == 12
    assert len(result["namespace_operation_rows"]) == 24
    assert result["transition_fixture"]["passed"]
    assert result["transition_fixture"]["feedback_attempts"] == 16
    changed = deepcopy(refs)
    changed[0]["sha256"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError):
        r.reconstruct(changed)


@pytest.fixture(scope="module")
def authenticated_work(tmp_path_factory):
    """SCENARIO-REPORT-8492-SEALED: original bounded children authenticate all three producers."""
    from carnot.reporting import publication_cost_model_8492 as r

    raw = tmp_path_factory.mktemp("qualified8492")
    work = r.measure(r.ROOT, raw)
    assert not work["gate_check_summary"], work["gate_check_summary"]
    assert all(q["qualified"] for q in work["qualifications"])
    return work


def test_authenticated_replay_and_tampering(authenticated_work, tmp_path):
    """SCENARIO-VERIFY-8492-CLI: independent operands reject self-consistently rehashed edits."""
    import json
    from carnot.reporting import publication_cost_model_8492 as r
    from carnot.reporting.current_work_receipt import atomic_json

    for change in ("none", "checksum", "source", "summary", "protocol", "ordering", "contract"):
        work = deepcopy(authenticated_work)
        raw = tmp_path / change
        work["raw"] = str(raw)
        if change == "source":
            work["code_config_hashes"][0]["sha256"] = "sha256:" + "0" * 64
        elif change == "summary":
            work["evidence"]["stage_rows"][0]["category_ns"]["directory_sync"] += 1
        elif change == "protocol":
            work["append_protocol"]["fsync_before_ack"] = False
        elif change == "ordering":
            work["ordering_fixture"]["events"] = ["append", "ack"]
        elif change == "contract":
            work["contract"]["tasks"][7]["prompt"] += " changed authority"
        atomic_json(raw / "measurement.json", work)
        output = tmp_path / (r.NAME + ".json")
        value = r.build(work, [], raw, output)
        if change == "checksum":
            value["reproducibility_checksum"] = "invalid"
        atomic_json(output, value)
        assert r.replay(output) == (change == "none"), change
    assert json.loads(output.read_bytes())["measured_append_speedup"] is None


@pytest.mark.parametrize(
    "failure", ["tools", "deliverable", "methods", "sidecar", "authority", "child", "qualification"]
)
def test_admission_rejections(authenticated_work, tmp_path, monkeypatch, failure):
    """SCENARIO-REPORT-8492-SEALED: admission reports exact unavailable authority without measurement."""
    from carnot.reporting import publication_cost_model_8492 as r
    from carnot.reporting.current_work_receipt import atomic_json

    contract = deepcopy(authenticated_work["contract"])
    monkeypatch.setattr(r.methods, "authority", lambda *args: contract)
    if failure == "tools":
        monkeypatch.setattr(r.shutil, "which", lambda name: None)
        contract["activated"] = False
    elif failure == "deliverable":
        contract["tasks"][7]["deliverable"] = "results/foreign.json"
    elif failure == "methods":
        monkeypatch.setattr(r, "reader_receipt", lambda *a, **kw: dict(passed=False))
    elif failure == "sidecar":
        monkeypatch.setattr(r, "read_bound_sidecar", lambda *a: dict(report=dict(passed=False)))
    elif failure == "authority":
        old = r.sealed.freeze
        monkeypatch.setattr(r.sealed, "freeze", lambda *a: dict(old(*a), sha256="wrong"))
    elif failure == "child":
        monkeypatch.setattr(r, "child", lambda *a, **kw: dict(passed=False))
    else:

        def rejected(name, argv, logs, **kw):
            atomic_json(
                Path(argv[-1]),
                dict(qualified=False, gate_check_summary=[dict(check="original_missing")]),
            )
            return dict(passed=True)

        monkeypatch.setattr(r, "child", rejected)
    work = r.measure(r.ROOT, tmp_path / "raw")
    assert work["gate_check_summary"]
    if failure == "tools":
        assert work["owned_failures"] == ["preconditions"]


@pytest.mark.parametrize("failure", ["denominator", "counts", "transition", "roster"])
def test_reconstruction_fail_closed(authenticated_work, monkeypatch, failure):
    """SCENARIO-REPORT-8492-SEALED: independently counted calls and logical fixtures must agree."""
    from carnot.reporting import publication_cost_model_8492 as r

    if failure == "denominator":
        old_partition = k.partition
        monkeypatch.setattr(k, "partition", lambda *a: dict(old_partition(*a), complete_wall_ns=0))
    elif failure == "counts":
        old = k.namespace
        monkeypatch.setattr(k, "namespace", lambda *a: dict(old(*a), total_fsyncs=0))
    elif failure == "transition":
        old_recover = k.recover
        monkeypatch.setattr(
            k, "recover", lambda *a: dict(old_recover(*a), committed_ids=["conflict"])
        )
    else:
        old_read = r.read

        def omit(ref):
            value = old_read(ref)
            if "primitives" in value and "pairs" in value:
                value["primitives"] = value["primitives"][:-1]
            return value

        monkeypatch.setattr(r, "read", omit)
    with pytest.raises(ValueError):
        r.reconstruct(authenticated_work["upstreams"])


def test_exact_validation_manifest(tmp_path):
    """SCENARIO-VERIFY-8492-CLI: the unchanged historical shard union is exact and disjoint."""
    from carnot.reporting import publication_cost_runner_8492 as runner

    plan = runner.manifest(tmp_path)
    collections = [p for p in plan if p["name"].startswith("collect_")]
    historical = [p for p in plan if p["name"].startswith("historical_")]
    assert len(collections) == 5
    nodes = [n for p in collections[1:] for n in p["collected_node_ids"]]
    executed = [n for p in historical for n in p["collected_node_ids"]]
    assert sorted(executed) == sorted(nodes)
    assert len(executed) == len(set(executed))
    assert all(p["deadline"] <= 300 for p in historical)
    assert all("-n" in p["argv"] and "--no-cov" in p["argv"] for p in historical)


def test_genuine_coverage_copy_and_owned_failure(tmp_path):
    """SCENARIO-VERIFY-8492-CLI: real coverage shards survive private parent cleanup."""
    import coverage
    from carnot.reporting import publication_cost_runner_8492 as runner

    private = tmp_path / "private"
    private.mkdir()
    cov = coverage.Coverage(
        data_file=str(private / ".coverage.actual"),
        config_file=False,
        include=[str(Path(k.__file__))],
    )
    cov.start()
    assert k.ceiling(100, 50)["conditional_speed_ceiling"] == 2
    cov.stop()
    cov.save()
    cov.json_report(outfile=str(private / "coverage.json"))
    output = tmp_path / (runner.r.NAME + ".json")
    assert runner.run(tmp_path / "missing", output, private, True) == 0
    from carnot.reporting.current_work_receipt import atomic_json

    work = runner.r.read(
        runner.r.reference(
            next((output.parent / "raw" / runner.r.NAME / "invocations").glob("*/measurement.json"))
        )
    )
    assert len(work["coverage_shards"]) == 1
    raw = Path(work["raw"])
    work["owned_failures"] = ["deliberate_owned_failure"]
    atomic_json(raw / "measurement.json", work)
    value = runner.r.build(work, [], raw, output)
    assert value["verdict_class"] == "disqualified"


def test_parser_entry_boundaries(authenticated_work, tmp_path):
    """SCENARIO-VERIFY-8492-CLI: direct entry retains real historical and private-control exits."""
    from carnot.reporting import publication_cost_runner_8492 as runner

    with pytest.raises(SystemExit) as error:
        runner.main(["--history-request", str(tmp_path / "absent")])
    assert error.value.code == 2
    request = Path(authenticated_work["raw"]) / "request-8464.json"
    assert (
        runner.main(
            ["--history-request", str(request), "--history-output", str(tmp_path / "history.json")]
        )
        == 0
    )
    assert runner.main(["--private-control"]) == 2
    assert (
        runner.main(
            [
                "--private-control",
                "--root",
                str(tmp_path / "missing"),
                "--output",
                str(tmp_path / (runner.r.NAME + ".json")),
            ]
        )
        == 0
    )


def test_owned_reconstruction_failure(authenticated_work, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8492-SEALED: authenticated inputs cannot excuse an owned reducer failure."""
    from carnot.reporting import publication_cost_model_8492 as r
    from carnot.reporting.current_work_receipt import atomic_json

    def fail(refs):
        raise ValueError("deliberate_owned_reducer_failure")

    monkeypatch.setattr(r, "reconstruct", fail)
    work = r.measure(r.ROOT, tmp_path / "raw")
    assert len(work["upstreams"]) == 3
    assert work["owned_failures"]
    raw = Path(work["raw"])
    atomic_json(raw / "measurement.json", work)
    value = r.build(work, [], raw, tmp_path / (r.NAME + ".json"))
    assert value["verdict_class"] == "disqualified"
