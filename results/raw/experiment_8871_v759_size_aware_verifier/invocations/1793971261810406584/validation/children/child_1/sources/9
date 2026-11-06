"""REQ-REPORT-8861: installed costs retain complete requests and honest gates."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import installed_cost_8861 as m
from carnot.reporting import installed_cost_runner_8861 as r
from scripts.experiments import installed_cost_worker_8861 as w
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def test_frozen_worlds():
    """SCENARIO-REPORT-8861-CONTRACT: world identities and labels precede timing."""
    worlds = m.freeze()
    assert len(worlds) == len({u["cluster_id"] for u in worlds}) == 160
    assert worlds == m.freeze()
    for size in m.SIZES:
        regular = [u for u in worlds if u["size"] == size and u["population"] == "deployment"]
        assert len(regular) == 30
    assert len([u for u in worlds if u["population"] == "adversarial"]) == 40
    conditions = set()
    for world in worlds:
        assert len(world["requests"]) == 4
        for q in world["requests"]:
            assert len(q["input"]["facts"]) == world["size"]
            assert len(q["input"]["query"]) == 8
            assert q["input_sha256"] == canonical_hash(q["input"])
            assert set(q["input"]) == {"facts", "query"}
            conditions.add(q["condition"])
    assert {"support", "refute", "missing", "binding", "conflict"} <= conditions
    assert len({tuple(u["arm_order"]) for u in worlds}) == 2


def test_worker_costs():
    """REQ-REPORT-8861: both installed arms pay bytes, validation and construction."""
    baseline = w.compile_baseline(m.baseline_source())
    for world in [m.freeze()[0], m.freeze()[1]]:
        for req in world["requests"]:
            encoded = json.dumps(req["input"], sort_keys=True)
            pair = [w.attempt(encoded, arm, baseline, m.CAP) for arm in m.ARMS]
            assert pair[0]["decision"] == pair[1]["decision"] == req["truth"]
            assert pair[0]["witness"] == pair[1]["witness"]
            for row in pair:
                assert row["primitive_operations"] == sum(row["operation_counts"].values())
                assert row["operation_counts"]["request_disposal"] > 0
                assert row["wall_time_s"] > 0
            for arm in m.ARMS:
                for cap in (
                    0,
                    len(encoded.encode()) * 2 + 1,
                    pair[m.ARMS.index(arm)]["primitive_operations"] - 1,
                ):
                    row = w.attempt(encoded, arm, baseline, cap)
                    assert row["status"] == "censored" and row["witness"] is None
                    assert row["primitive_operations"] == cap
                assert w.attempt("{}", arm, baseline, m.CAP)["status"] == "failed"
            validation = w.meter.Counter(None)
            token = w.meter.CURRENT.set(validation)
            w.meter.parse(req["input"])
            w.meter.CURRENT.reset(token)
            paid_validation_cap = len(encoded.encode()) * 2 + 1 + 3 * validation.total
            assert (
                w.attempt(encoded, m.ARMS[1], baseline, paid_validation_cap)["status"] == "censored"
            )
    with pytest.raises(ValueError, match="arm"):
        w.attempt("{}", "invented", baseline, m.CAP)


def fake_rows():
    rows, clocks = [], []
    for world in m.freeze():
        for boundary in ("cold", "resident_inner"):
            for arm in m.ARMS:
                clocks.append(
                    dict(
                        cluster_id=world["cluster_id"],
                        size=world["size"],
                        population=world["population"],
                        boundary=boundary,
                        arm=arm,
                        wall_time_s=2.0 if arm == m.ARMS[0] else 1.0,
                    )
                )
                for q in world["requests"]:
                    rows.append(
                        dict(
                            unit_id=q["unit_id"],
                            cluster_id=world["cluster_id"],
                            size=world["size"],
                            population=world["population"],
                            boundary=boundary,
                            arm=arm,
                            seed=m.SEED,
                            decision=q["truth"],
                            witness={"?x": "same"} if q["truth"] == "support" else None,
                            truth=q["truth"],
                            status="completed",
                            missing_reason=None,
                            denominator=1,
                            metric=int(q["truth"] != "insufficient"),
                            primitive_operations=2,
                            world_wall_time_s=2.0 if arm == m.ARMS[0] else 1.0,
                            operation_counts={"test": 2},
                        )
                    )
    return rows, clocks


def test_whole_world_reduction():
    """REQ-REPORT-8861: censors remain in the mixture and quality can veto speed."""
    rows, clocks = fake_rows()
    result = m.reduce(rows, clocks, 0.0)
    assert result["installed_cost_benefit_score"] == 1
    assert len(result["per_world_results"]) == 320
    assert result["paired_intervals"]["cold"]["mixture"]["speed_ratio"]["lower"] == 2.0
    assert (
        result["paired_intervals"]["cold"]["mixture"]["speed_ratio"]["familywise_confidence"]
        == 0.99
    )
    bad = deepcopy(rows)
    bad[1]["witness"] = {"?x": "wrong"}
    assert not m.reduce(bad, clocks, 0.0)["quality_gates"]["exact_completed_parity"]
    bad = deepcopy(rows)
    index = next(
        i for i, x in enumerate(bad) if x["arm"] == m.ARMS[1] and x["decision"] == "support"
    )
    bad[index].update(status="censored", decision="insufficient", witness=None)
    assert not m.reduce(bad, clocks, 0.0)["quality_gates"]["no_coverage_loss"]
    bad[index].update(status="completed", decision="support", truth="refute")
    assert not m.reduce(bad, clocks, 0.0)["quality_gates"]["no_false_accept_increase"]
    slow = deepcopy(clocks)
    for clock in slow:
        if clock["arm"] == m.ARMS[1] and clock["size"] == 16:
            clock["wall_time_s"] = 3
    slower_rows = [
        dict(row, world_wall_time_s=3) if row["arm"] == m.ARMS[1] and row["size"] == 16 else row
        for row in rows
    ]
    assert m.reduce(slower_rows, slow, 0)["installed_cost_benefit_score"] == 0
    with pytest.raises(ValueError, match="parent_clock_drift"):
        m.reduce(rows, slow, 0)
    with pytest.raises(ValueError, match="roster"):
        m.reduce(rows[:-1], clocks, 0)
    with pytest.raises(ValueError, match="roster"):
        m.reduce(rows + [rows[0]], clocks, 0)
    assert m.reduce([], [], 0)["installed_cost_benefit_score"] == 0
    for row in rows:
        row["primitive_operations"] = 0
    assert (
        m.reduce(rows, clocks, 0)["paired_intervals"]["cold"]["mixture"]["operation_ratio"]["lower"]
        is None
    )


def test_authority_failures(tmp_path, monkeypatch):
    """REQ-REPORT-8861: absent and changed external authorities cannot become measurements."""
    pre = r.preconditions(tmp_path / "ok")
    assert pre["external_ready"] and pre["resources"]["passed"], pre["diagnostics"]
    upstream = tmp_path / "upstream.json"
    monkeypatch.setattr(r, "UPSTREAM", upstream)
    assert r.preconditions(tmp_path / "absent")["diagnostics"][0]["check"] == "missing_file"
    original = json.loads(
        (m.ROOT / "results/experiment_8858_v758_installed_indexed_verifier.json").read_bytes()
    )
    for change, expected in [
        (lambda v: v.pop("installed_index_ready_score"), "missing_field"),
        (lambda v: v.update(installed_index_ready_score=0), "observed_zero"),
        (lambda v: v.update(task_id="fake"), "rejected_producer"),
    ]:
        value = deepcopy(original)
        change(value)
        atomic_json(upstream, value)
        assert r.preconditions(tmp_path / expected)["diagnostics"][0]["check"] == expected
    atomic_json(upstream, original)
    monkeypatch.setattr(r, "PIN", r.sha256_file(upstream))
    original["raw_shard_hashes"].append(
        dict(path=str(tmp_path / "missing-operand"), sha256="sha256:missing")
    )
    atomic_json(upstream, original)
    monkeypatch.setattr(r, "PIN", r.sha256_file(upstream))
    assert r.preconditions(tmp_path / "operand")["diagnostics"][0]["check"] == "missing_file"
    operand = tmp_path / "changed-operand"
    operand.write_text("changed bytes")
    original["raw_shard_hashes"] = [dict(path=str(operand), sha256="sha256:wrong")]
    atomic_json(upstream, original)
    monkeypatch.setattr(r, "PIN", r.sha256_file(upstream))
    assert r.preconditions(tmp_path / "drift")["diagnostics"][0]["check"] == "rejected_producer"


def test_private_cli_and_replay(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8861-CONTRACT: actual installed children replay outside the checkout."""
    output = tmp_path / (m.NAME + ".json")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    cmd = [sys.executable, "-u", str(m.ROOT / m.CLI)]
    result = subprocess.run(
        [*cmd, "--private-output", "--private-world-limit", "1", "--output", str(output)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_bytes())
    assert len(value["rows"]) == 16 and value["installed_cost_ready_score"] == 0
    assert len(value["cold_process_rows"]) == 2 and len(value["resident_rows"]) == 1
    assert value["package_hashes"] and value["completed_count"] == 16
    assert r.replay(output)
    saved = r.load(value["work_reference"])

    def reject_work(mutator):
        forged = deepcopy(saved)
        mutator(forged)
        work_path = tmp_path / "forged-work.json"
        atomic_json(work_path, forged)
        candidate = dict(value, work_reference=r.reference(work_path))
        path = tmp_path / "forged-primary.json"
        atomic_json(path, candidate)
        assert not r.replay(path, execute=False)

    reject_work(lambda work: work["current_work_receipt"].update(duration_s=-1))
    reject_work(lambda work: work.update(worlds=[]))
    reject_work(
        lambda work: work.update(
            raw_shard_hashes=[
                ref
                for ref in work["raw_shard_hashes"]
                if not (
                    Path(ref["path"]).parent.name == "processes"
                    and "always_joint" in Path(ref["path"]).name
                )
            ]
        )
    )
    reject_work(lambda work: work["rows"][0].update(decision="forged"))
    reject_work(lambda work: work["clocks"][0].update(wall_time_s=99))
    reject_work(lambda work: work["cold_process_rows"][0].update(duration_s=99))
    bad_hash = dict(value, source_artifact_hashes=[dict(path=str(output), sha256="sha256:wrong")])
    atomic_json(tmp_path / "bad-hash.json", bad_hash)
    assert not r.replay(tmp_path / "bad-hash.json")
    atomic_json(tmp_path / "forged-coverage.json", dict(value, required_checks_passed=True))
    assert not r.replay(tmp_path / "forged-coverage.json", execute=False)
    changed = deepcopy(saved["rows"])
    changed[0]["decision"] = "changed-owned-decision"
    with monkeypatch.context() as control:
        control.setattr(r, "measure", lambda *args: dict(rows=changed))
        assert not r.replay(output)
    replayed = subprocess.run(
        [*cmd, "--cold-replay", str(output)],
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
    assert not r.replay(tmp_path / "missing.json")


def test_cli_invalid_inputs():
    """REQ-REPORT-8861: subsets stay private and calendar identity stays fixed."""
    for args in (
        ["--date", "20261107"],
        ["--private-output"],
        ["--private-world-limit", "1"],
        ["--private-output", "--output", "/tmp/private8861.json", "--private-world-limit", "0"],
    ):
        with pytest.raises(SystemExit):
            r.main(args)


def test_worker_entry(tmp_path, monkeypatch):
    """REQ-REPORT-8861: the actual wrapper emits every result and authenticated import path."""
    import io
    import runpy

    world = m.freeze()[0]
    baseline = tmp_path / "baseline.py"
    baseline.write_text(m.baseline_source())
    for arm in (*m.ARMS, "paired"):
        output = io.StringIO()
        monkeypatch.setattr(
            sys, "stdin", io.StringIO(json.dumps(dict(worlds=[world], cap=m.CAP, seed=m.SEED)))
        )
        monkeypatch.setattr(sys, "stdout", output)
        monkeypatch.setattr(sys, "argv", ["worker", str(baseline), arm])
        with pytest.raises(SystemExit) as exited:
            runpy.run_path(str(m.ROOT / m.WORKER), run_name="__main__")
        assert exited.value.code == 0
        result = json.loads(output.getvalue())
        assert (
            len(result["rows"]) == (8 if arm == "paired" else 4)
            and len(result["installed_imports"]) == 3
        )
        if arm == "paired":
            assert [row["arm"] for row in result["rows"]] == [
                selected for selected in world["arm_order"] for _ in range(4)
            ]


def test_owned_failures_keep_roster(tmp_path, monkeypatch):
    """REQ-REPORT-8861: missing children, deadlines and invalid outputs retain all four slots."""
    import time

    world = m.freeze()[0]
    for boundary in ("cold", "resident"):
        result = r.process(
            tmp_path / "missing-python",
            tmp_path / "baseline",
            [world],
            m.ARMS[0],
            boundary,
            tmp_path,
            tmp_path / boundary,
        )
        assert len(result["rows"]) == 4 and not result["receipt"]["passed"]
        assert all(x["status"] == "failed" for x in result["rows"])
    failed_child = r.process(
        Path(sys.executable),
        tmp_path / "missing-baseline",
        [world],
        m.ARMS[0],
        "cold",
        tmp_path,
        tmp_path / "failed-child",
    )
    assert not failed_child["receipt"]["passed"]
    expired = r.process(
        Path(sys.executable),
        tmp_path / "baseline",
        [world],
        m.ARMS[0],
        "cold",
        tmp_path,
        tmp_path / "expired",
        0,
    )
    assert not expired["receipt"]["passed"]

    class WaitingChild:
        returncode = 0
        args = ["controlled-child"]

        def __init__(self, *args, **kwargs):
            self.calls = 0
            self.killed = False

        def communicate(self, value=None, timeout=None):
            self.calls += 1
            if self.calls == 1:
                time.sleep(0.01)
                raise subprocess.TimeoutExpired(self.args, timeout, stderr=b"child progress\n")
            return "{}", "child progress\n"

        def kill(self):
            self.killed = True

    monkeypatch.setattr(r.subprocess, "Popen", WaitingChild)
    for budget in (None, 0.005):
        deadline = float("inf") if budget is None else time.monotonic() + budget
        result = r.process(
            Path(sys.executable),
            tmp_path / "baseline",
            [world],
            m.ARMS[0],
            "cold",
            tmp_path,
            tmp_path / str(deadline),
            deadline,
        )
        assert len(result["rows"]) == 4 and not result["receipt"]["passed"]
    rows, clocks = fake_rows()
    missing = [row for row in rows if row["unit_id"] != rows[0]["unit_id"]]
    with pytest.raises(ValueError, match="roster_world"):
        m.reduce(missing, clocks, 0)


def test_artifact_and_orchestration_controls(tmp_path, monkeypatch):
    """REQ-REPORT-8861: owned failures disqualify; a complete null remains distinct from blocking."""
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
        r, "measure", lambda *args: (_ for _ in ()).throw(ValueError("owned control failed"))
    )
    failed = r.prepare(tmp_path / "owned")
    assert len(failed["rows"]) == 16 and failed["internal_errors"]
    pre["external_ready"] = False
    pre["diagnostics"] = [
        dict(
            check="missing_file",
            upstream="authority",
            artifact_field="ready",
            expected=1,
            observed=None,
            path="missing",
            sha256=None,
        )
    ]
    blocked = r.prepare(tmp_path / "blocked")
    validation = dict(
        receipts=[dict(passed=True)],
        union=dict(passed=True),
        coverage=dict(complete=True, percent=100, files=[]),
        raw_references=[],
    )
    ref = dict(path="in-memory-control", sha256="sha256:control")
    blocked.update(code_config_hashes=[], raw_shard_hashes=[])
    assert r.artifact(blocked, validation, "20261106", ref, ref)["verdict_class"] == "blocked"
    failed.update(code_config_hashes=[], raw_shard_hashes=[])
    assert r.artifact(failed, validation, "20261106", ref, ref)["verdict_class"] == "disqualified"
    rows, clocks = fake_rows_from_worlds(worlds)
    work = dict(
        failed,
        worlds=worlds,
        rows=rows,
        clocks=clocks,
        private_world_limit=None,
        internal_errors=[],
        installed_imports_valid=True,
        installation_s=0,
        preconditions_checked=dict(
            external_ready=True, resources=dict(passed=True), references=[], diagnostics=[]
        ),
    )
    positive = r.artifact(work, validation, "20261106", ref, ref)
    assert positive["installed_cost_ready_score"] == positive["installed_cost_benefit_score"] == 1
    assert positive["verdict_class"] == "circular_positive"
    for clock in clocks:
        clock["wall_time_s"] = 1
    for row in rows:
        row["world_wall_time_s"] = 1
    assert r.artifact(work, validation, "20261106", ref, ref)["verdict_class"] == "null"
    monkeypatch.setattr(
        r,
        "namespace",
        lambda *args: type(
            "Validator",
            (),
            dict(validate=lambda *args: validation, terminal=lambda *args: dict(passed=True)),
        ),
    )
    assert r.validate(tmp_path / "validation", tmp_path) == validation
    assert r.terminal(tmp_path / "candidate", tmp_path)["passed"]
    monkeypatch.setattr(r, "prepare", lambda *args: blocked)
    monkeypatch.setattr(
        r,
        "publish_primary",
        lambda output, value, validator: dict(passed=validator(output)["passed"]),
    )
    assert r.main(["--output", str(tmp_path / (m.NAME + ".json"))]) == 0
    wheel = tmp_path / "wheel.whl"
    wheel.write_bytes(b"owned install failure control")
    target = tmp_path / "install-failure"
    target.mkdir()
    monkeypatch.undo()
    monkeypatch.setattr(r, "stream", lambda *args, **kwargs: dict(passed=False))
    with pytest.raises(ValueError, match="installation_failed"):
        r.measure(target, worlds[:1], dict(path=str(wheel)))


def fake_rows_from_worlds(worlds):
    """Build explicit control measurements without presenting them as executed research."""
    original = m.freeze
    try:
        m.freeze = lambda: worlds
        return fake_rows()
    finally:
        m.freeze = original
