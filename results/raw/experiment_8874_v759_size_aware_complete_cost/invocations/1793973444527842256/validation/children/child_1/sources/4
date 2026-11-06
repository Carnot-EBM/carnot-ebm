"""REQ-REPORT-8874: complete costs, fixed denominators and small-size guards."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from carnot.reporting import size_aware_complete_cost_8874 as m
from carnot.reporting import size_aware_complete_runner_8874 as r
from scripts.experiments import size_aware_cost_worker_8874 as w
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def fixture_rows():
    """Make deterministic paid clocks so a timing guard has an exact answer."""
    rows = []
    for world in m.freeze():
        for boundary in m.BOUNDARIES:
            for repeat in range(3):
                for arm in m.ARMS:
                    for q in world["requests"]:
                        rows.append(
                            dict(
                                unit_id=q["unit_id"],
                                cluster_id=world["cluster_id"],
                                size=world["size"],
                                population=world["population"],
                                boundary=boundary,
                                repeat=repeat,
                                arm=arm,
                                status="completed",
                                missing_reason=None,
                                truth=q["truth"],
                                decision=q["truth"],
                                witness=None,
                                primitive_operations=100 if arm == "scan" else 50,
                                complete_wall_time_s=2.0 if arm == "scan" else 1.0,
                                operation_counts={},
                                denominator=1,
                            )
                        )
    return rows


def test_freeze():
    """SCENARIO-REPORT-8874-CONTRACT: freeze independent worlds before clocks."""
    worlds = m.freeze()
    assert worlds == m.freeze() and len(worlds) == 220
    assert len({v["cluster_id"] for v in worlds}) == 220
    assert len({tuple(v["arm_order"]) for v in worlds}) == 6
    assert sum(v["population"] == "stress" for v in worlds) == 40
    assert not {v["cluster_id"] for v in m.freeze(pilot=True)} & {v["cluster_id"] for v in worlds}
    assert all(len(v["requests"]) == 4 and set(v["arm_order"]) == set(m.ARMS) for v in worlds)
    assert all(len(q["input"]["facts"]) == v["size"] for v in worlds for q in v["requests"])


def test_reduce(monkeypatch):
    """REQ-REPORT-8874: a faster kernel cannot override either small-size guard."""
    monkeypatch.setattr(m, "DRAWS", 100)
    rows = fixture_rows()
    result = m.reduce(rows, 0)
    assert result["size_cost_benefit_score"] == 1
    assert len(result["per_world_results"]) == 440
    interval = result["paired_intervals"]["resident"]["size_aware"]["mixture"]["speed_ratio"]
    assert interval["lower"] == interval["upper"] == 2
    assert interval["simultaneous_endpoint_count"] == 160
    for size in (8, 16):
        slow = deepcopy(rows)
        for row in slow:
            if row["arm"] == "size_aware" and row["size"] == size:
                row["complete_wall_time_s"] = 2.2
        reduced = m.reduce(slow, 0)
        assert reduced["size_cost_benefit_score"] == 0
        assert (
            reduced["paired_intervals"]["resident"]["size_aware"]["mixture"]["speed_ratio"]["lower"]
            > 1
        )
    for mutate, gate in [
        (dict(witness={"?x": "fake"}), "exact_completed_parity"),
        (dict(decision="support", truth="refute"), "no_false_accept_increase"),
        (dict(status="censored", decision="insufficient"), "no_coverage_loss"),
    ]:
        changed = deepcopy(rows)
        target = next(x for x in changed if x["arm"] == "size_aware" and x["truth"] == "support")
        target.update(mutate)
        assert not m.reduce(changed, 0)["quality_gates"][gate]
    for bad in (rows[:-1], rows + [rows[0]]):
        with pytest.raises(ValueError, match="roster"):
            m.reduce(bad, 0)
    assert m.reduce([], 0)["size_cost_benefit_score"] == 0


def test_worker():
    """REQ-REPORT-8874: all entries pay the same complete boundary and cap."""
    baseline = w.compile_baseline(m.baseline_source())
    for world in [m.freeze()[0], m.freeze()[30], m.freeze()[60]]:
        for req in world["requests"]:
            encoded = json.dumps(req["input"], sort_keys=True)
            pair = [w.attempt(encoded, arm, baseline, m.CAP) for arm in m.ARMS]
            assert all(x["decision"] == req["truth"] for x in pair)
            assert pair[0]["witness"] == pair[1]["witness"] == pair[2]["witness"]
            for x in pair:
                assert x["primitive_operations"] == sum(x["operation_counts"].values())
                assert x["operation_counts"]["request_disposal"] > len(encoded)
                assert x["wall_time_s"] > 0
            assert w.attempt(encoded, "size_aware", baseline, 0)["status"] == "censored"
    with pytest.raises(ValueError, match="arm"):
        w.attempt("{}", "foreign", baseline, 100)


def test_preconditions(tmp_path, monkeypatch):
    """REQ-REPORT-8874: missing external authority blocks before any benchmark."""
    pre = r.preconditions(tmp_path / "good")
    assert pre["external_ready"] and pre["resources"]["passed"], pre["diagnostics"]
    assert pre["package_hash"]
    monkeypatch.setattr(
        r, "HISTORY", [(tmp_path / "absent", "sha256:no", "size_dispatch_ready_score")]
    )
    assert r.preconditions(tmp_path / "missing")["diagnostics"][0]["check"] == "missing_file"


def test_private_cli_and_replay(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8874-CONTRACT: real installed children and cold replay retain all repeats."""
    output = tmp_path / (m.NAME + ".json")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    command = [sys.executable, "-u", str(m.ROOT / m.CLI)]
    child = subprocess.run(
        [
            *command,
            "--date",
            "20261106",
            "--private-output",
            "--private-world-limit",
            "1",
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    value = r.load(r.reference(output))
    assert value["size_cost_ready_score"] == 0 and len(value["rows"]) == 72
    assert len(value["cold_process_rows"]) == 36 and len(value["resident_rows"]) == 1
    assert value["completed_count"] == 72 and value["independent_count"] == 1
    assert r.replay(output)
    original = r.load(value["work_reference"])
    with monkeypatch.context() as control:
        control.setattr(r, "stream", lambda *args, **kwargs: dict(passed=False))
        assert not r.replay(output)
    with monkeypatch.context() as control:
        control.setattr(r, "stream", lambda *args, **kwargs: dict(passed=True))
        control.setattr(r, "process", lambda *args, **kwargs: dict(rows=[]))
        assert not r.replay(output)
    for mode in ("parent", "request"):
        work = deepcopy(original)
        old_ref = next(
            ref
            for ref in work["raw_shard_hashes"]
            if Path(ref["path"]).parent.name == "processes"
            and Path(ref["path"]).name.startswith("cold")
        )
        operand = r.load(old_ref)
        if mode == "parent":
            operand["receipt"]["wall_time_s"] += 1
            for clock in work["clocks"]:
                if clock["name"] == operand["receipt"]["name"]:
                    clock["wall_time_s"] += 1
        else:
            operand["rows"][0]["complete_wall_time_s"] += 1
            key = lambda x: (x["boundary"], x["unit_id"], x["arm"], x["repeat"])
            for row in work["rows"]:
                if key(row) == key(operand["rows"][0]):
                    row["complete_wall_time_s"] += 1
        raw_path = tmp_path / mode / "processes" / Path(old_ref["path"]).name
        atomic_json(raw_path, operand)
        work["raw_shard_hashes"] = [
            r.reference(raw_path) if ref == old_ref else ref for ref in work["raw_shard_hashes"]
        ]
        forged_work = tmp_path / (mode + "-work.json")
        atomic_json(forged_work, work)
        candidate = r.artifact(
            work,
            r.load(value["validation_reference"]),
            "20261106",
            r.reference(forged_work),
            value["validation_reference"],
        )
        candidate_path = tmp_path / (mode + ".json")
        atomic_json(candidate_path, candidate)
        assert not r.replay(candidate_path, execute=False)
    assert not r.replay(tmp_path / "absent")
    for mutate in [
        lambda x: x["current_work_receipt"].update(duration_s=-1),
        lambda x: x.update(worlds=[]),
        lambda x: x["rows"][0].update(decision="fake"),
        lambda x: x["clocks"][0].update(wall_time_s=99),
    ]:
        work = deepcopy(original)
        mutate(work)
        path = tmp_path / "forged-work.json"
        atomic_json(path, work)
        candidate = dict(value, work_reference=r.reference(path))
        atomic_json(tmp_path / "forged.json", candidate)
        assert not r.replay(tmp_path / "forged.json", execute=False)
    candidate = dict(value, required_checks_passed=True)
    atomic_json(tmp_path / "coverage.json", candidate)
    assert not r.replay(tmp_path / "coverage.json", execute=False)
    candidate = dict(value, code_config_hashes=[dict(path=str(output), sha256="sha256:wrong")])
    atomic_json(tmp_path / "hash.json", candidate)
    assert not r.replay(tmp_path / "hash.json", execute=False)
    replayed = subprocess.run(
        [*command, "--cold-replay", str(output)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert replayed.returncode == 0 and "replay_passed" in replayed.stdout
    value["rows"][0]["decision"] = "invented"
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(output, value)
    assert not r.replay(output)


def test_process_failures(tmp_path, monkeypatch):
    """REQ-REPORT-8874: supervision failures retain every scheduled observation."""
    world = m.freeze()[0]
    for boundary in m.BOUNDARIES:
        result = r.process(
            tmp_path / "missing-python",
            tmp_path / "baseline",
            [world],
            [0],
            boundary,
            tmp_path,
            tmp_path / boundary,
            float("inf"),
        )
        assert len(result["rows"]) == 12 and not result["receipt"]["passed"]
    expired = r.process(
        Path(sys.executable),
        tmp_path / "baseline",
        [world],
        [0],
        "cold",
        tmp_path,
        tmp_path / "expired",
        0,
    )
    assert not expired["receipt"]["passed"]
    failed = r.process(
        Path(sys.executable),
        tmp_path / "missing-baseline",
        [world],
        [0],
        "cold",
        tmp_path,
        tmp_path / "child",
        float("inf"),
    )
    assert not failed["receipt"]["passed"]

    class WaitingChild:
        returncode = 0

        def __init__(self, *args, **kwargs):
            self.calls = 0

        def communicate(self, value=None, timeout=None):
            self.calls += 1
            if self.calls == 1:
                time.sleep(0.01)
                raise subprocess.TimeoutExpired("controlled", timeout, stderr=b"progress\n")
            return json.dumps(dict(rows=[], installed_imports=[])), "progress\n"

        def kill(self):
            self.killed = True

    monkeypatch.setattr(r.subprocess, "Popen", WaitingChild)
    for budget in (float("inf"), 0.005):
        deadline = time.monotonic() + budget
        result = r.process(
            Path(sys.executable),
            tmp_path / "baseline",
            [world],
            [0],
            "cold",
            tmp_path,
            tmp_path / str(deadline),
            deadline,
        )
        assert not result["receipt"]["passed"] and len(result["rows"]) == 12


def test_cli_arguments():
    """REQ-REPORT-8874: private subsets cannot become public measurements."""
    for args in [
        ["--date", "20261107"],
        ["--private-output"],
        ["--private-world-limit", "1"],
        ["--private-output", "--output", "/tmp/a.json", "--private-world-limit", "0"],
    ]:
        with pytest.raises(SystemExit):
            r.main(args)


def test_worker_entry(tmp_path, monkeypatch, capsys):
    """REQ-REPORT-8874: the actual worker entry retains resident clocks and installed modules."""
    import io
    import runpy

    baseline = tmp_path / "scan.py"
    baseline.write_text(m.baseline_source())
    monkeypatch.setattr(sys, "argv", ["worker", str(baseline)])
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(json.dumps(dict(worlds=m.freeze()[:1], repeats=[0], cap=m.CAP, seed=m.SEED))),
    )
    with pytest.raises(SystemExit) as exited:
        runpy.run_path(str(m.ROOT / m.WORKER), run_name="__main__")
    assert exited.value.code == 0
    result = json.loads(capsys.readouterr().out)
    assert len(result["rows"]) == 12 and len(result["installed_imports"]) == 4


def test_owned_orchestration(tmp_path, monkeypatch):
    """REQ-REPORT-8874: owned failures disqualify while missing authority blocks."""
    worlds = m.freeze()
    pre = dict(
        external_ready=True,
        diagnostics=[],
        references=[],
        resources=dict(passed=True),
        package_hash={},
    )
    monkeypatch.setattr(r, "preconditions", lambda raw: deepcopy(pre))
    monkeypatch.setattr(m, "freeze", lambda: worlds[:1])
    monkeypatch.setattr(
        r, "measure", lambda *args: (_ for _ in ()).throw(ValueError("owned failure"))
    )
    failed = r.prepare(tmp_path / "failed")
    assert failed["internal_errors"] and len(failed["rows"]) == 72
    pre["external_ready"] = False
    blocked = r.prepare(tmp_path / "blocked")
    validation = dict(
        receipts=[dict(passed=True)],
        union=dict(passed=True),
        coverage=dict(complete=True, percent=100, files=[]),
        raw_references=[],
    )
    ref = dict(path="control-only", sha256="sha256:control")
    assert r.artifact(blocked, validation, "20261106", ref, ref)["verdict_class"] == "blocked"
    assert r.artifact(failed, validation, "20261106", ref, ref)["verdict_class"] == "disqualified"
    monkeypatch.setattr(m, "freeze", lambda: worlds)
    monkeypatch.setattr(m, "DRAWS", 100)
    rows = fixture_rows()
    work = dict(
        failed,
        worlds=worlds,
        rows=rows,
        clocks=[{}] * 7921,
        private_world_limit=None,
        internal_errors=[],
        installed_imports_valid=True,
        preconditions_checked=dict(pre, external_ready=True),
    )
    positive = r.artifact(work, validation, "20261106", ref, ref)
    assert positive["size_cost_ready_score"] == positive["size_cost_benefit_score"] == 1
    assert positive["verdict_class"] == "circular_positive"
    for row in rows:
        row["complete_wall_time_s"] = 1.0
    assert r.artifact(work, validation, "20261106", ref, ref)["verdict_class"] == "null"
    proxy = type(
        "Validator",
        (),
        dict(validate=lambda *args: validation, terminal=lambda *args: dict(passed=True)),
    )
    monkeypatch.setattr(r, "namespace", lambda *args: proxy)
    assert r.validate(tmp_path / "checks", tmp_path) == validation
    assert r.terminal(tmp_path / "candidate", tmp_path)["passed"]
    monkeypatch.setattr(r, "prepare", lambda *args: blocked)
    monkeypatch.setattr(
        r,
        "publish_primary",
        lambda output, value, validator: dict(passed=validator(output)["passed"]),
    )
    assert r.main(["--output", str(tmp_path / (m.NAME + ".json"))]) == 0
    monkeypatch.setattr(r, "replay", lambda path: False)
    assert r.main(["--cold-replay", str(tmp_path / "missing")]) == 1


def test_installation_and_pilot_failure(tmp_path, monkeypatch):
    """REQ-REPORT-8874: failed owned setup and pilot projections cannot disappear."""
    wheel = tmp_path / "wheel.whl"
    wheel.write_bytes(b"control fixture")
    package = dict(path=str(wheel))
    monkeypatch.setattr(r, "stream", lambda *args, **kwargs: dict(passed=False))
    with pytest.raises(ValueError, match="installation_failed"):
        r.measure(tmp_path / "install", m.freeze()[:1], package)
    monkeypatch.setattr(r, "stream", lambda *args, **kwargs: dict(passed=True))
    monkeypatch.setattr(r, "process", lambda *args, **kwargs: dict(rows=[dict(status="failed")]))
    with pytest.raises(ValueError, match="pilot_deadline_or_execution"):
        r.measure(tmp_path / "pilot", m.freeze()[:1], package)


def test_no_paid_operations(monkeypatch):
    """REQ-REPORT-8874: zero paid operations do not manufacture an operation speedup."""
    monkeypatch.setattr(m, "DRAWS", 10)
    rows = fixture_rows()
    for row in rows:
        row["primitive_operations"] = 0
    result = m.reduce(rows, 0)
    assert (
        result["paired_intervals"]["resident"]["indexed"]["8"]["operation_ratio"]["lower"] is None
    )
    incomplete = [x for x in rows if x["unit_id"] != rows[0]["unit_id"]]
    with pytest.raises(ValueError, match="roster_world"):
        m.reduce(incomplete, 0)
    for row in rows:
        row["complete_wall_time_s"] = 0.0
    assert m.reduce(rows, 0)["size_cost_benefit_score"] == 0


def test_package_custody(tmp_path, monkeypatch):
    """REQ-REPORT-8874: a ready producer cannot excuse a changed installed wheel."""
    monkeypatch.setattr(r, "sha256_file", lambda path: "sha256:changed")
    pre = r.preconditions(tmp_path / "changed")
    assert not pre["external_ready"] and pre["diagnostics"][-1]["artifact_field"] == "package_hash"
