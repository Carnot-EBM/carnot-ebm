"""REQ-REPORT-8848: full durable costs retain preparation and failed cells."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot.reporting import memory_size_cost_8848 as m


def test_roster_and_workloads():
    """SCENARIO-REPORT-8848-CONTRACT: paired cells share a fixed64-query workload."""
    from carnot.memory.causal_memory_8802 import normalize

    rows = m.roster()
    assert len(rows) == 60 and rows == m.roster()
    assert len({r["unit_id"] for r in rows}) == 60
    for block in range(10):
        cells = [r for r in rows if r["block"] == block]
        assert {(r["occupancy"], r["arm"]) for r in cells} == {
            (n, a) for n in (0, 8, 32) for a in ("read_only", "combined")
        }
        assert len({m.canonical_hash(m.workload(r["block"])) for r in cells}) == 1
    shapes = [normalize(m.payload(0, i)) for i in range(64)]
    assert len(set(shapes)) == 64
    workload = m.workload(0)
    assert len(workload) == 64
    assert sum(p["template_index"] < 8 for p in workload) == 32
    assert all(p["template_index"] < 8 or p["template_index"] >= 32 for p in workload)


def test_preflight(tmp_path, monkeypatch):
    """REQ-REPORT-8848: every historical byte and code pin must authenticate."""
    pre = m.preflight(tmp_path / "good")
    assert not pre["failures"] and pre["checks"]["repository_virtualenv"]
    monkeypatch.setattr(m, "COST_PIN", "wrong")
    assert m.preflight(tmp_path / "bad")["failures"]


@pytest.fixture(scope="module")
def observed(tmp_path_factory):
    """REQ-REPORT-8848: use real admitted checkpoints and fresh child processes."""
    raw = tmp_path_factory.mktemp("memory-size")
    pre = m.preflight(raw / "pre")
    spec = dict(m.roster()[0], block=0, occupancy=32, arm="combined")
    prep = m.prepare(raw / "prep", spec, pre["head"])
    assert prep["status"] == "completed" and prep["final_occupancy"] == 32
    row = m.window(raw / "window", spec, pre["head"], prep)
    return raw, pre, prep, row


def test_actual_lifecycle(observed):
    """REQ-REPORT-8848: full memory rejects additions and preserves exact restart."""
    _, _, prep, row = observed
    assert row["status"] == "completed" and m.parity(row)["passed"]
    assert row["query_count"] == 64 and row["initial_occupancy"] == 32
    assert row["final_occupancy"] == 32 and row["template_additions"] == 0
    assert row["template_hits"] == 32 and row["capacity_rejections"] == 32
    assert row["attempted_updates"] == row["applied_updates"] == 64
    assert row["fsync_counts"] == {"file": 130, "directory": 130}
    assert len(prep["segments"][0]["releases"]) == 32
    for key, val in [("query_count", 63), ("complete_ns", -1), ("status", "failed")]:
        bad = deepcopy(row)
        bad[key] = val
        assert not m.parity(bad)["passed"]
    bad = deepcopy(row)
    bad["segments"][1]["restart_input_hash"] = "wrong"
    assert not m.parity(bad)["passed"]


def test_readonly_and_empty(observed, tmp_path):
    """REQ-REPORT-8848: empty and frozen cells retain the same64 requests."""
    _, pre, _, _ = observed
    spec = dict(m.roster()[0], block=1, occupancy=0, arm="read_only")
    prep = m.prepare(tmp_path / "prep", spec, pre["head"])
    row = m.window(tmp_path / "window", spec, pre["head"], prep)
    assert m.parity(row)["passed"] and row["final_occupancy"] == 0
    assert row["applied_updates"] == row["template_hits"] == 0
    assert row["attempted_updates"] == 64


def test_child_failure(tmp_path, observed):
    """REQ-REPORT-8848: timeouts and bad child requests remain failed evidence."""
    _, pre, prep, spec = observed
    row = m.window(tmp_path / "killed", spec, pre["head"], prep, deadline=0.001)
    assert row["status"] == "failed" and not m.parity(row)["passed"]
    assert row["commands"][0]["timed_out"]
    request = tmp_path / "absent"
    result = m.process(request, 0, 60)
    assert result["status"] == "failed"


def test_reductions():
    """REQ-REPORT-8848: simultaneous uncertainty uses ten independent paired blocks."""
    rows = [
        dict(r, status="completed", complete_ns=64000 + 100 * r["occupancy"] + r["block"])
        for r in m.roster()
    ]
    reduced = m.reduce_costs(rows, [])
    assert len(reduced["paired_cost_intervals"]) == 6
    assert all(
        r["paired_blocks"] == 10 and r["familywise_confidence_percent"] == 99
        for r in reduced["paired_cost_intervals"]
    )
    assert len(reduced["occupancy_rows"]) == 6
    rows[0]["status"] = "failed"
    assert any(r["missing_pairs"] == 1 for r in m.reduce_costs(rows, [])["paired_cost_intervals"])
    assert m.reduce_costs([], [])["paired_cost_intervals"][0]["lower_ns"] is None


def test_replay_absent(tmp_path):
    """SCENARIO-REPORT-8848-CONTRACT: absent or malformed evidence cannot pass."""
    assert not m.replay(tmp_path / "absent")["passed"]
    m.durable(tmp_path / "bad", {})
    assert not m.replay(tmp_path / "bad")["passed"]


@pytest.fixture
def sealed(tmp_path, monkeypatch):
    """REQ-REPORT-8848: retain all scheduled cells while isolating orchestration."""
    from carnot.reporting import memory_size_runner_8848 as r

    def prepare(raw, spec, head):
        return dict(spec, status="failed", complete_ns=1, segments=[])

    monkeypatch.setattr(m, "prepare", prepare)
    work = r.run(tmp_path / "run", "20261105")
    validation = dict(
        passed=True, receipts=[], coverage=dict(complete=True, percent=100, zero_exclusions=True)
    )
    m.durable(Path(work["raw"]) / "validation.json", validation)
    return work, validation


def test_build_replay_and_tamper(sealed, tmp_path):
    """SCENARIO-REPORT-8848-CONTRACT: failed pilots disqualify; false summaries fail replay."""
    from carnot.reporting import memory_size_runner_8848 as r

    work, validation = sealed
    value = r.build(work, validation)
    assert value["verdict_class"] == "disqualified" and value["whole_cost_ready_score"] == 0
    assert value["intended_count"] == value["censored_count"] == 60
    path = tmp_path / "candidate.json"
    m.durable(path, value)
    assert m.replay(path)["passed"]
    for field, supplied in [("whole_cost_ready_score", 1), ("completed_count", 60)]:
        m.durable(path, dict(value, **{field: supplied}))
        assert not m.replay(path)["passed"]
    m.durable(path, value)
    Path(value["raw_shard_hashes"][0]["path"]).write_text("{}")
    assert not m.replay(path)["passed"]
    work["preconditions"]["failures"] = [dict(observed="missing_file")]
    assert r.build(work, validation)["verdict_class"] == "blocked"
    validation["passed"] = False
    assert r.build(work, validation)["verdict_class"] == "disqualified"


def test_foreign_cli(sealed, tmp_path):
    """SCENARIO-REPORT-8848-CONTRACT: real cold replay has no checkout environment."""
    import os
    import subprocess
    import sys
    from carnot.reporting import memory_size_runner_8848 as r

    work, validation = sealed
    path = tmp_path / "candidate.json"
    m.durable(path, r.build(work, validation))
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    argv = [sys.executable, "-u", str(m.ROOT / m.CLI), "--replay", str(path)]
    child = subprocess.run(argv, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)
    assert child.returncode == 0 and '"replay_passed": true' in child.stdout
    m.durable(path, {})
    assert (
        subprocess.run(argv, cwd=tmp_path, env=env, capture_output=True, timeout=60).returncode == 1
    )


def test_preparation_integrity(observed):
    """REQ-REPORT-8848: a changed checkpoint cannot claim actual verified admissions."""
    _, _, prep, _ = observed
    assert m.preparation_parity(prep)
    bad = deepcopy(prep)
    bad["segments"][0]["state"]["templates"].pop()
    assert not m.preparation_parity(bad)
    assert not m.preparation_parity(dict(prep, status="failed"))


def test_scheduled_execution(observed, tmp_path, monkeypatch):
    """REQ-REPORT-8848: one good pilot admits all remaining scheduled cells."""
    from carnot.reporting import memory_size_runner_8848 as r

    _, pre, prep, row = observed
    monkeypatch.setattr(m, "preflight", lambda raw: pre)
    monkeypatch.setattr(m, "prepare", lambda raw, spec, head: dict(prep, **spec))
    monkeypatch.setattr(m, "preparation_parity", lambda p: True)
    monkeypatch.setattr(m, "window", lambda raw, spec, head, prepared: dict(row, **spec))
    monkeypatch.setattr(m, "parity", lambda row: dict(unit_id=row["unit_id"], passed=True))
    work = r.run(tmp_path / "run", "20261105")
    assert len(work["window_rows"]) == 60 and len(work["preparation_cost_rows"]) == 30
    assert work["projection"]["admitted"]
    validation = dict(passed=True, coverage=dict(complete=True), receipts=[])
    m.durable(Path(work["raw"]) / "validation.json", validation)
    value = r.build(work, validation)
    assert value["whole_cost_ready_score"] == 1 and value["verdict_class"] == "null"
    assert value["numerical_training_calls"] > 3840


def test_validation_publication_dispatch(sealed, tmp_path, monkeypatch):
    """REQ-REPORT-8848: actual validation failures cannot qualify a primary."""
    from coverage import Coverage
    from carnot.reporting import memory_size_runner_8848 as r

    work, validation = sealed
    raw = Path(work["raw"])
    recorded = []
    log = raw / "owned.log"
    log.write_text("unit orchestration control\n")

    def commands(root, plan, **kwargs):
        recorded.extend(c.name for c in plan)
        return [
            dict(name=c.name, passed=True, log_path=str(log), log_sha256=m.sha256_file(log))
            for c in plan
        ]

    monkeypatch.setattr(r, "run_commands", commands)
    monkeypatch.setattr(Coverage, "combine", lambda *args, **kwargs: None)
    monkeypatch.setattr(Coverage, "save", lambda *args, **kwargs: None)
    monkeypatch.setattr(Coverage, "analysis2", lambda *args: ("file", [1], [], [], ""))
    m.durable(raw / ".coverage", {})
    assert r.validate(raw)["passed"]
    monkeypatch.setattr(Coverage, "analysis2", lambda *args: ("file", [1], [], [1], "1"))
    assert not r.validate(raw)["passed"]
    assert r.terminal(tmp_path / "candidate", raw)["passed"]
    assert {"cold_replay", "private_E2E_018_021"} <= set(recorded)
    monkeypatch.setattr(r, "validate", lambda raw: validation)
    monkeypatch.setattr(r, "terminal", lambda *args: dict(passed=True))
    m.durable(raw / "validation.json", validation)
    result = r.qualify(work, tmp_path / (m.NAME + ".json"))
    assert result["whole_cost_ready_score"] == 0
    monkeypatch.setattr(r, "run", lambda *args: work)
    monkeypatch.setattr(r, "qualify", lambda *args: result)
    assert r.main(["--raw", str(raw), "--output", str(tmp_path / "out")]) == 0
    assert r.main(["--output", str(tmp_path / "out")]) == 0
    assert r.main(["--replay", str(tmp_path / "absent")]) == 1
    monkeypatch.setattr(m, "worker", lambda *args: None)
    assert r.main(["--worker", str(tmp_path / "request")]) == 0
    with pytest.raises(SystemExit):
        r.main(["--date", "20261104"])


def test_gate_operand_drift(tmp_path, monkeypatch):
    """REQ-REPORT-8848: missing and altered original operands fail closed."""
    monkeypatch.setattr(
        m.old,
        "authenticate",
        lambda *args: (
            dict(
                source_artifact_hashes=[dict(path=str(tmp_path / "absent"), sha256="wrong")],
                code_config_hashes=[],
                raw_shard_hashes=[],
            ),
            None,
        ),
    )
    monkeypatch.setattr(
        m.old,
        "preflight",
        lambda raw: dict(failures=[], sources=[], checks={}, internal_failure=False),
    )
    assert m.preflight(tmp_path)["failures"][0]["observed"] == "missing_file"
    p = tmp_path / "absent"
    p.write_text("changed")
    assert m.preflight(tmp_path)["failures"][0]["observed"] == "rejected_producer"


def test_replay_raw_controls(sealed, observed, tmp_path):
    """REQ-REPORT-8848: rehashed work still needs valid lifecycle and warm admissions."""
    from carnot.reporting import memory_size_runner_8848 as r

    work, validation = sealed
    _, _, prep, row = observed
    raw = Path(work["raw"])
    path = tmp_path / "candidate.json"
    work["window_rows"][0] = deepcopy(row)
    work["preparation_cost_rows"] = [deepcopy(prep)]
    work["pilot_window_rows"] = []

    def seal():
        m.durable(raw / "measurement.json", work)
        m.durable(path, r.build(work, validation))

    seal()
    assert m.replay(path)["passed"]
    work["window_rows"][0]["query_count"] = 63
    seal()
    assert not m.replay(path)["passed"]
    work["window_rows"][0] = deepcopy(row)
    work["preparation_cost_rows"][0]["segments"][0]["state"]["templates"].pop()
    seal()
    assert not m.replay(path)["passed"]
    work["preparation_cost_rows"] = [deepcopy(prep)]
    work["window_rows"][0]["segments"][0]["state_file"]["sha256"] = "wrong"
    seal()
    assert not m.replay(path)["passed"]


def test_bound_validation_logs(sealed):
    """REQ-REPORT-8848: retained receipts bind actual coverage and log bytes."""
    from carnot.reporting import memory_size_runner_8848 as r

    work, validation = sealed
    log = Path(work["raw"]) / "owned.log"
    log.write_text("private validation control\n")
    validation["receipts"] = [dict(passed=True, log_path=str(log), log_sha256=m.sha256_file(log))]
    validation["coverage"]["combined_reference"] = m.reference(log)
    m.durable(Path(work["raw"]) / "validation.json", validation)
    assert m.reference(log) in r.build(work, validation)["raw_shard_hashes"]
