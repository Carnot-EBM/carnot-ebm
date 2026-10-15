"""REQ-REPORT-8555, REQ-VERIFY-8555: actual private CLI and failure evidence."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from carnot.reporting import canonical_update_8555 as e
from carnot.reporting import canonical_update_runner_8555 as r
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import child


def test_actual_private_cli(tmp_path):
    """SCENARIO-VERIFY-8555-VALIDATION: real processes publish and reject rehashed drift."""
    output = tmp_path / (e.NAME + ".json")
    argv = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    receipt = child(
        "private",
        [*argv, "--date", "20261015", "--private-control", "--output", str(output)],
        tmp_path / "logs",
        deadline=180,
    )
    assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
    value = json.loads(output.read_bytes())
    assert value["online_adapter_ready_score"] == 0
    assert value["verdict_class"] == "disqualified"
    assert e.replay(output)
    tamper = deepcopy(value)
    tamper["metrics"]["physical_coordinate_writes"] += 1
    tamper["reproducibility_checksum"] = e.checksum(tamper)
    atomic_json(tmp_path / "tamper.json", tamper)
    for name, args, expected in [
        ("cold", ["--cold-replay", str(output)], 0),
        ("missing", ["--cold-replay", str(tmp_path / "absent")], 1),
        ("tamper", ["--cold-replay", str(tmp_path / "tamper.json")], 1),
        ("error", ["--deliberate-error"], 7),
        ("date", ["--date", "20261014"], 2),
        ("unsafe", ["--private-control"], 2),
    ]:
        assert child(name, [*argv, *args], tmp_path / "logs", expected=expected, deadline=60)[
            "passed"
        ]
    assert not e.replay(tmp_path / "absent")
    assert not e.replay(tmp_path / "tamper.json")
    assert r.main(["--date", "20261014"]) == 2
    assert r.main(["--deliberate-error"]) == 7
    assert r.main(["--private-control"]) == 2
    assert r.main(["--cold-replay", str(output)]) == 0


def test_actual_restart_children(tmp_path):
    """SCENARIO-VERIFY-8555-CONTROLS: exit73 is a real child, not a mocked failure."""
    receipts, passed = r.restart(tmp_path)
    assert passed and [row["actual_exit"] for row in receipts] == [0, 73, 0]
    assert all(row["passed"] for row in receipts)


def test_negative_validator_control(tmp_path):
    """SCENARIO-REPORT-8555-PUBLICATION: final numeric metrics reach unchanged TAUTOLOGY."""
    _, metrics, _, _ = e.canonical(e.controls())
    receipt = r.negative_control(dict(metrics=metrics), tmp_path)
    assert receipt["passed"]
    assert any(row["kind"] == "TAUTOLOGY" for row in receipt["findings"])


def test_replay_rejects_source_drift(tmp_path):
    """SCENARIO-REPORT-8555-PUBLICATION: repairing an envelope cannot reseal originals."""
    source = tmp_path / "source.json"
    atomic_json(source, dict(original=True))
    value = dict(
        source_artifact_hashes=[e.reference(source)],
        code_config_hashes=[],
        raw_shard_hashes=[],
        reproducibility_checksum="wrong",
    )
    output = tmp_path / "candidate.json"
    atomic_json(output, value)
    assert not e.replay(output)
    source.write_text("{}")
    value["reproducibility_checksum"] = e.checksum(value)
    atomic_json(output, value)
    assert not e.replay(output)


def test_empty_and_alias_integrity():
    """SCENARIO-REPORT-8555-COUNTERS: empty evidence cannot fabricate an observation."""
    records, metrics, receipt, _ = e.canonical([])
    assert not receipt["passed"]
    assert all(row["value"] == 0 and row["source_event_ids"] == [] for row in records)
    assert not set(e.ALIAS_MAP) & set(metrics)


def test_terminal_gate_classes(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8555-PUBLICATION: blockers and owned failures cannot grant readiness."""
    from carnot.reporting import canonical_update_report_8555 as report

    atomic_json(tmp_path / "events.json", e.controls())
    work = dict(
        raw=str(tmp_path),
        private_control=False,
        events=e.reference(tmp_path / "events.json"),
        sources=[],
        code=[],
        receipts=[],
        failures=[],
        numerical=[dict(passed=True)],
        restart_passed=True,
        upstream_consumer=dict(passed=True),
        historical=dict(historical_learning_validation_ready_score=1),
        union=dict(passed=True),
        preconditions={},
        start_ns=1,
        end_ns=2,
        negative_control=dict(passed=True),
        findings=[],
        phases=[],
        historical_validation_reference=None,
    )
    atomic_json(tmp_path / "measurement.json", work)
    monkeypatch.setattr(report, "REQUIRED", set())
    output = tmp_path / (e.NAME + ".json")
    assert report.build(work, output)["verdict_class"] == "circular_positive"
    work["failures"] = [e.gate(tmp_path / "absent", "resource", 1, None)]
    assert report.build(work, output)["verdict_class"] == "blocked"
    work["receipts"] = [
        dict(
            name="failed",
            passed=False,
            stdout_path=str(tmp_path / "log"),
            stdout_sha256=None,
            stderr_path=str(tmp_path / "err"),
            stderr_sha256=None,
        )
    ]
    assert report.build(work, output)["verdict_class"] == "disqualified"
    assert report.build(work, output)["online_adapter_ready_score"] == 0


def test_shard_collection_and_deadline_rejections(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8555-VALIDATION: missing nodes and exhausted clocks fail closed."""
    monkeypatch.setattr(
        e.historical,
        "original",
        lambda: ({}, dict(nodes=["suite::test"], authorities={"suite": {}})),
    )
    monkeypatch.setattr(r, "collect", lambda raw, suite: (["suite::foreign"], dict(passed=True)))
    monkeypatch.setattr(e.historical, "partition", lambda nodes: [nodes])
    monkeypatch.setattr(e.historical, "exact_union", lambda nodes, receipts: dict(passed=False))
    work = dict(receipts=[], failures=[], start_ns=0)
    r.historical_shards(work, tmp_path, tmp_path)
    assert {row["field"] for row in work["failures"]} == {
        "exact_original_collection",
        "measurement_deadline_s",
    }
    assert not work["union"]["passed"]


def test_production_orchestration_failure_receipt(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8555-VALIDATION: a failed tool/resource cannot turn into readiness."""
    from types import SimpleNamespace

    current = r.coverage.Coverage.current

    class ControlTracer:
        def __init__(self, **kw):
            pass

        def start(self):
            pass

        def stop(self):
            pass

        def save(self):
            pass

        @staticmethod
        def current():
            return current()

    monkeypatch.setattr(r.coverage, "Coverage", ControlTracer)
    monkeypatch.setattr(e, "authenticate", lambda root: ({}, [], []))
    monkeypatch.setattr(e, "history", lambda: (e.controls(), []))
    monkeypatch.setattr(e, "numerical", lambda scratch: [dict(passed=True)])
    monkeypatch.setattr(r, "restart", lambda raw: ([], True))
    monkeypatch.setattr(
        r, "historical_shards", lambda work, raw, private: work.update(union=dict(passed=True))
    )
    monkeypatch.setattr(
        r.resources,
        "resources",
        lambda private: dict(
            filesystem="tmpfs", tools={}, available_memory_bytes=0, free_disk_bytes=0
        ),
    )
    monkeypatch.setattr(r, "scan", lambda *args: ([], []))

    def fake_child(name, argv, raw, **kw):
        raw.mkdir(parents=True, exist_ok=True)
        stdout, stderr = raw / (name + ".stdout"), raw / (name + ".stderr")
        stdout.write_text("synthetic owned failure control")
        stderr.write_text("")
        return dict(
            name=name,
            passed=False,
            stdout_path=str(stdout),
            stderr_path=str(stderr),
            stdout_sha256=e.reference(stdout)["sha256"],
            stderr_sha256=e.reference(stderr)["sha256"],
        )

    monkeypatch.setattr(r, "child", fake_child)
    private = tmp_path / "private"
    private.mkdir()
    atomic_json(private / "coverage.json", dict(control="not a qualification receipt"))
    output = tmp_path / (e.NAME + ".json")
    assert r.run(output, private) == 0
    value = json.loads(output.read_bytes())
    assert value["online_adapter_ready_score"] == 0
    assert any(row["field"] == "disk_memory_tools" for row in value["gate_check_summary"])
    assert not value["negative_validator_control"]["passed"]
    recovery = tmp_path / "recovery"
    recovery.mkdir()
    (recovery / "initial_controller_failure.log").write_text("actual failure fixture")
    monkeypatch.setattr(
        r, "restore_historical", lambda work, raw: work.update(union=dict(passed=True))
    )
    assert r.run(output, private, resume=recovery) == 0
    resumed = json.loads(output.read_bytes())
    assert resumed["verdict_class"] == "disqualified"
    assert any(row.get("owned_required_check") for row in resumed["gate_check_summary"])
    monkeypatch.setattr(r, "run", lambda *args, **kw: 0)
    assert r.main(["--output", str(output)]) == 0
    assert r.main(["--fixture-worker", str(tmp_path / "worker")]) == 1


def test_rehashed_child_receipt_rejection(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8555-PUBLICATION: a repaired envelope cannot rewrite a child exit."""
    from carnot.reporting import canonical_update_report_8555 as report

    atomic_json(tmp_path / "events.json", e.controls())
    receipt = dict(
        name="owned",
        passed=True,
        stdout_path=str(tmp_path / "owned.stdout"),
        stdout_sha256=None,
        stderr_path=str(tmp_path / "owned.stderr"),
        stderr_sha256=None,
    )
    work = dict(
        raw=str(tmp_path),
        private_control=True,
        events=e.reference(tmp_path / "events.json"),
        sources=[],
        code=[],
        receipts=[receipt],
        failures=[],
        numerical=[dict(passed=True)],
        restart_passed=True,
        upstream_consumer=dict(passed=True),
        historical={},
        union=dict(passed=False),
        preconditions={},
        start_ns=1,
        end_ns=2,
        negative_control=dict(passed=True),
        findings=[],
        phases=[],
        historical_validation_reference=None,
    )
    r.snapshot(work, tmp_path)
    output = tmp_path / (e.NAME + ".json")
    value = report.build(work, output)
    atomic_json(output, value)
    assert e.replay(output)
    work["receipts"][0]["passed"] = False
    atomic_json(tmp_path / "measurement.json", work)
    atomic_json(output, report.build(work, output))
    assert not e.replay(output)
    r.snapshot(work, tmp_path)
    work["private_control"] = False
    work["historical"] = dict(historical_learning_validation_ready_score=1)
    work["collected_node_ids"] = ["foreign"]
    monkeypatch.setattr(e.historical, "original", lambda: ({}, dict(nodes=["original"])))
    r.snapshot(work, tmp_path)
    atomic_json(output, report.build(work, output))
    assert not e.replay(output)


def test_live_parent_shards_survive_combine(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8555-VALIDATION: combining must preserve a live parent's SQLite shard."""
    from types import SimpleNamespace

    current = r.coverage.Coverage.current

    def forbidden_live_parent_access():
        raise AssertionError("validation must not save a live parent tracer")

    monkeypatch.setattr(
        r.coverage, "Coverage", SimpleNamespace(current=forbidden_live_parent_access)
    )
    seen = []

    def command(name, argv, raw, **kw):
        seen.append((name, argv))
        return dict(name=name, passed=True)

    monkeypatch.setattr(r, "child", command)
    work = dict(receipts=[])
    r.verification(work, tmp_path, tmp_path)
    assert "--keep" in next(argv for name, argv in seen if name == "coverage_combine")
    assert current is not None


def test_restore_completed_original_children(tmp_path):
    """SCENARIO-VERIFY-8555-VALIDATION: recovery authenticates real completed child evidence."""
    raw = e.ROOT / "results/raw" / e.NAME / "invocations/1792102958750669530"
    work = dict(receipts=[], sources=[])
    r.restore_historical(work, raw)
    assert work["union"]["passed"] and work["union"]["terminal_count"] == 158
    assert len(work["collected_node_ids"]) == 158
    first = json.loads((raw / "children/shard_0.receipt.json").read_bytes())
    first["stdout_sha256"] = "sha256:wrong"
    atomic_json(tmp_path / "children/shard_0.receipt.json", first)
    with pytest.raises(ValueError, match="recovery_child_log_hash"):
        r.restore_historical(dict(receipts=[], sources=[]), tmp_path)
    with pytest.raises(OSError):
        r.restore_historical(dict(receipts=[], sources=[]), tmp_path / "absent")


def test_recovery_rejects_union_and_collection_drift(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8555-VALIDATION: repaired summaries cannot authorize an incomplete union."""
    raw = e.ROOT / "results/raw" / e.NAME / "invocations/1792102958750669530"
    monkeypatch.setattr(e.historical, "exact_union", lambda *args: dict(passed=False))
    with pytest.raises(ValueError, match="recovery_original_union"):
        r.restore_historical(dict(receipts=[], sources=[]), raw)
    monkeypatch.setattr(e.historical, "original", lambda: ({}, dict(nodes=["expected"])))
    monkeypatch.setattr(e.historical, "partition", lambda nodes: [])
    monkeypatch.setattr(e.historical, "exact_union", lambda *args: dict(passed=True))
    atomic_json(tmp_path / "collection.json", dict(collected_node_ids=["foreign"]))
    with pytest.raises(ValueError, match="recovery_original_collection"):
        r.restore_historical(dict(receipts=[], sources=[]), tmp_path)


def test_real_scoped_historical_launch(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8555-VALIDATION: the launch path executes a real exact scoped child."""
    import time

    producer, original = e.historical.original()
    suite = "tests/python/test_current_work_receipt.py"
    nodes = [n for n in original["nodes"] if n.split("::")[0] == suite]
    assert len(nodes) == 4
    selected = dict(original, nodes=nodes, authorities={suite: original["authorities"][suite]})
    monkeypatch.setattr(e.historical, "original", lambda: (producer, selected))
    work = dict(receipts=[], failures=[], start_ns=time.monotonic_ns())
    r.historical_shards(work, tmp_path / "raw", tmp_path / "scratch")
    assert work["union"]["passed"] and work["union"]["terminal_count"] == 4
    assert all(row["passed"] for row in work["receipts"])
