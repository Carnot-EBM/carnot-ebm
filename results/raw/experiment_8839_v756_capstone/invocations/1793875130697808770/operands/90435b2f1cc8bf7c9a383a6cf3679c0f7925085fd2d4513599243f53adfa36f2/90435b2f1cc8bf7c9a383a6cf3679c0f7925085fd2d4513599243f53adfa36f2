"""REQ-REPORT-8835 and REQ-VERIFY-8835: complete costs retain exact evidence."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot.reporting import complete_adaptation_cost_8835 as m


def test_authority_and_schedule(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8835-CONTRACT: historical readiness has no new-head gate."""
    pre = m.preflight(tmp_path)
    assert not pre["failures"]
    assert pre["arm_admission_rows"][0]["admitted"]
    assert not pre["arm_admission_rows"][1]["admitted"]
    roster = m.roster()
    assert len(roster) == 90
    assert {r["block"] for r in roster} == set(range(30))
    assert all(sum(r["arm"] == a for r in roster) == 30 for a in m.ARMS)


def test_gate_failure_types(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8835-PARITY: missing bytes and absent gates differ."""
    path = tmp_path / "source.json"
    assert m.authenticate(path, "sha256:absent", "ready")[1]["observed"] == "missing_file"
    for value, expected in [
        ({}, "missing_field"),
        ({"ready": 0}, "observed_zero"),
        ({"ready": 1, "required_checks_passed": False}, "rejected_producer"),
    ]:
        m.atomic_json(path, value)
        assert m.authenticate(path, m.sha256_file(path), "ready")[1]["observed"] == expected
    assert m.authenticate(path, "wrong", "ready")[1]["observed"] == "rejected_producer"


@pytest.fixture(scope="module")
def real_window(tmp_path_factory):
    """REQ-VERIFY-8835: retain a genuine process operand for replay attacks."""
    tmp_path = tmp_path_factory.mktemp("whole-cost")
    pre = m.preflight(tmp_path / "pre")
    return m.window(tmp_path / "window", m.roster()[0], pre["head"], deadline=60)


def test_real_window_and_controls(real_window) -> None:
    """SCENARIO-VERIFY-8835-PARITY: real fresh processes recover and drain."""
    row = real_window
    assert row["status"] == "completed"
    assert m.parity(row)["passed"]
    assert len(row["segments"]) == 2
    assert row["complete_ns"] >= row["service_ns"] >= 0
    for field, value in [("complete_ns", -1), ("status", "failed")]:
        bad = deepcopy(row)
        bad[field] = value
        assert not m.parity(bad)["passed"]
    bad = deepcopy(row)
    bad["segments"][1]["state"]["acks"].append("e0")
    assert not m.parity(bad)["passed"]


def test_intervals_keep_failed_pairs() -> None:
    """REQ-REPORT-8835: interventions are paired blocks, never component fractions."""
    rows = [
        dict(r, status="completed", complete_ns=(i + 1) * 1000, service_ns=100)
        for i, r in enumerate(m.roster())
    ]
    reduced = m.reduce_costs(rows)
    assert len(reduced["paired_cost_intervals"]) == 2
    assert all(r["paired_blocks"] == 30 for r in reduced["paired_cost_intervals"])
    rows[0]["status"] = "failed"
    assert any(r["paired_blocks"] == 29 for r in m.reduce_costs(rows)["paired_cost_intervals"])
    assert m.reduce_costs([])["paired_cost_intervals"][0]["lower99_ns"] is None


def test_replay_missing(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8835-CONTRACT: absent operands cannot become evidence."""
    assert not m.replay(tmp_path / "missing")["passed"]
    m.atomic_json(tmp_path / "bad", {})
    assert not m.replay(tmp_path / "bad")["passed"]


def test_preflight_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-8835-PARITY: altered or absent original operands fail closed."""
    path = tmp_path / "bad.json"
    path.write_text("broken json")
    assert m.authenticate(path, m.sha256_file(path), "ready")[1]["observed"] == "rejected_producer"
    monkeypatch.setattr(m, "PIN", "wrong")
    assert m.preflight(tmp_path / "failed")["failures"]
    monkeypatch.setattr(m, "PIN", m.sha256_file(m.ROOT / m.PRIMARY))
    monkeypatch.setattr(m, "DESIGN", "absent-authority")
    assert m.preflight(tmp_path / "missing")["failures"]
    monkeypatch.setattr(m, "DESIGN", "AGENTS.md")
    assert m.preflight(tmp_path / "changed")["failures"]


def test_process_failure_and_deadline(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8835-PARITY: killed windows retain output and timing."""
    request = tmp_path / "missing.json"
    failed = m.process(request, 0, 60)
    assert failed["status"] == "failed"
    assert Path(failed["log"]["path"]).is_file()
    timed = m.process(request, 0, 0.001)
    assert timed["timed_out"] and timed["status"] == "failed"
    pre = m.preflight(tmp_path / "pre")
    row = m.window(tmp_path / "window", m.roster()[0], pre["head"], deadline=0.001)
    assert row["status"] == "failed" and not m.parity(row)["passed"]


def test_worker_requires_parent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-VERIFY-8835: a child cannot run an unstarted service window."""
    from io import StringIO
    import sys

    pre = m.preflight(tmp_path / "pre")
    request = tmp_path / "request.json"
    m.atomic_json(request, dict(m.roster()[0], head=pre["head"]))
    monkeypatch.setattr(sys, "stdin", StringIO("stop\n"))
    with pytest.raises(ValueError, match="parent_start_required"):
        m.worker(request, 0)


@pytest.fixture
def sealed_work(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """REQ-REPORT-8835: keep the full schedule while isolating orchestration tests."""
    from carnot.reporting import complete_adaptation_runner_8835 as r

    def window(raw, spec, head, **kwargs):
        return dict(
            spec,
            status="censored",
            missing_reason="private_test_control",
            metric=None,
            complete_ns=100,
            service_ns=10,
            segments=[],
            commands=[],
        )

    monkeypatch.setattr(m, "window", window)
    original_parity = m.parity
    monkeypatch.setattr(m, "parity", lambda row: dict(unit_id=row["unit_id"], passed=True))
    work = r.run(tmp_path / "run", "20261105")
    monkeypatch.setattr(m, "parity", original_parity)
    validation = dict(
        passed=True, coverage=dict(complete=True, percent=100, zero_exclusions=True), receipts=[]
    )
    m.atomic_json(Path(work["raw"]) / "validation.json", validation)
    return work, validation


def test_build_replay_and_tamper(sealed_work, tmp_path: Path) -> None:
    """SCENARIO-REPORT-8835-CONTRACT: rehashed aggregate drift still fails replay."""
    from carnot.reporting import complete_adaptation_runner_8835 as r

    work, validation = sealed_work
    value = r.build(work, validation)
    assert value["verdict_class"] == "disqualified"
    path = tmp_path / "candidate.json"
    m.atomic_json(path, value)
    assert m.replay(path)["passed"]
    for key, supplied in [
        ("completed_count", 90),
        ("whole_cost_ready_score", 1),
        ("numerical_training_calls", 4),
        ("required_checks_passed", False),
    ]:
        changed = deepcopy(value)
        changed[key] = supplied
        m.atomic_json(path, changed)
        assert not m.replay(path)["passed"]
    m.atomic_json(path, value)
    Path(value["raw_shard_hashes"][1]["path"]).write_text("{}")
    assert not m.replay(path)["passed"]
    work["preconditions"]["failures"] = [dict(observed="missing_file")]
    assert r.build(work, validation)["verdict_class"] == "blocked"
    validation["passed"] = False
    assert r.build(work, validation)["verdict_class"] == "disqualified"


def test_run_blocked_and_pilot_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-8835: failures preserve every scheduled denominator."""
    from carnot.reporting import complete_adaptation_runner_8835 as r

    monkeypatch.setattr(m, "PIN", "wrong")
    work = r.run(tmp_path / "blocked", "20261105")
    assert not work["pilot_window_rows"]
    assert all(row["status"] == "censored" for row in work["window_rows"])
    monkeypatch.setattr(m, "PIN", m.sha256_file(m.ROOT / m.PRIMARY))
    monkeypatch.setattr(
        m, "window", lambda raw, spec, head: dict(spec, status="failed", segments=[], complete_ns=1)
    )
    work = r.run(tmp_path / "pilot_fail", "20261105")
    assert not work["projection"]["admitted"]
    assert len(work["pilot_window_rows"]) == 2


def test_validation_publication_dispatch(
    sealed_work, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-VERIFY-8835: failing owned commands cannot grant readiness."""
    from coverage import Coverage
    from carnot.reporting import complete_adaptation_runner_8835 as r

    work, validation = sealed_work
    raw = Path(work["raw"])
    recorded = []

    def commands(root, plan, **kwargs):
        recorded.extend(c.name for c in plan)
        return [dict(name=c.name, passed=True) for c in plan]

    monkeypatch.setattr(r, "run_commands", commands)
    monkeypatch.setattr(Coverage, "combine", lambda *args, **kwargs: None)
    monkeypatch.setattr(Coverage, "save", lambda *args, **kwargs: None)
    monkeypatch.setattr(Coverage, "analysis2", lambda *args: ("file", [1], [], [], ""))
    m.atomic_json(raw / ".coverage", {})
    assert r.validate(raw)["passed"]
    monkeypatch.setattr(Coverage, "analysis2", lambda *args: ("file", [1], [], [1], "1"))
    assert not r.validate(raw)["passed"]
    assert r.terminal(tmp_path / "candidate", raw)["passed"]
    assert "cold_replay" in recorded and "private_E2E_018_021" in recorded
    monkeypatch.setattr(r, "validate", lambda raw: validation)
    monkeypatch.setattr(r, "terminal", lambda *args: dict(passed=True))
    m.atomic_json(raw / "validation.json", validation)
    result = r.qualify(work, tmp_path / (m.NAME + ".json"))
    assert result["whole_cost_ready_score"] == 0
    monkeypatch.setattr(r, "run", lambda *args: work)
    monkeypatch.setattr(r, "qualify", lambda *args: result)
    assert r.main(["--raw", str(raw), "--output", str(tmp_path / "out")]) == 0
    assert r.main(["--output", str(tmp_path / "out")]) == 0
    assert r.main(["--replay", str(tmp_path / "missing")]) == 1
    monkeypatch.setattr(m, "worker", lambda *args: None)
    assert r.main(["--worker", str(tmp_path / "request")]) == 0
    with pytest.raises(SystemExit):
        r.main(["--date", "20261104"])


def test_original_control_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-VERIFY-8835: unchanged primary hashes cannot excuse altered original assertions."""
    original = json.loads((m.ROOT / m.PRIMARY).read_text())
    original["rows"] = []
    monkeypatch.setattr(m, "authenticate", lambda *args: (original, None))
    pre = m.preflight(tmp_path)
    assert pre["internal_failure"] and pre["failures"][0]["check"] == "original_controls"


def test_private_foreign_cli(sealed_work, tmp_path: Path) -> None:
    """SCENARIO-REPORT-8835-CONTRACT: valid cold replay works without PYTHONPATH."""
    import os
    import subprocess
    import sys
    from carnot.reporting import complete_adaptation_runner_8835 as r

    work, validation = sealed_work
    candidate = tmp_path / "candidate.json"
    m.atomic_json(candidate, r.build(work, validation))
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    command = [sys.executable, "-u", str(m.ROOT / m.CLI), "--replay", str(candidate)]
    child = subprocess.run(
        command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60
    )
    assert child.returncode == 0 and '"replay_passed": true' in child.stdout
    m.atomic_json(candidate, {})
    child = subprocess.run(
        command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60
    )
    assert child.returncode == 1


def test_replay_lifecycle_failures(
    sealed_work, real_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-8835-PARITY: raw states need valid lifecycle and byte bindings."""
    from carnot.reporting import complete_adaptation_runner_8835 as r

    work, validation = sealed_work
    work["pilot_window_rows"] = [deepcopy(real_window)]
    raw = Path(work["raw"])
    path = tmp_path / "candidate.json"

    def seal():
        m.atomic_json(raw / "measurement.json", work)
        m.atomic_json(path, r.build(work, validation))

    seal()
    assert m.replay(path)["passed"]
    work["pilot_window_rows"][0]["segments"][0]["state_file"]["sha256"] = "wrong"
    seal()
    assert not m.replay(path)["passed"]
    work["pilot_window_rows"][0]["complete_ns"] = -1
    seal()
    assert not m.replay(path)["passed"]


def test_build_binds_owned_logs(sealed_work) -> None:
    """REQ-REPORT-8835: owned receipts bind actual log and coverage bytes."""
    from carnot.reporting import complete_adaptation_runner_8835 as r

    work, validation = sealed_work
    log = Path(work["raw"]) / "owned.log"
    log.write_text("private validation control\n")
    validation["receipts"] = [dict(passed=True, log_path=str(log), log_sha256=m.sha256_file(log))]
    validation["coverage"]["combined_reference"] = m.reference(log)
    m.atomic_json(Path(work["raw"]) / "validation.json", validation)
    result = r.build(work, validation)
    assert m.reference(log) in result["raw_shard_hashes"]
