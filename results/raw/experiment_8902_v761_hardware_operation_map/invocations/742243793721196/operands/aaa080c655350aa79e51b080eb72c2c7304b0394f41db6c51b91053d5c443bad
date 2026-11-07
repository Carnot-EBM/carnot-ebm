"""REQ-REPORT-8882: exclusive clocks and observer limits precede cost claims."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import request_cost_attribution_8882 as m
from carnot.reporting import request_cost_runner_8882 as r
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from scripts.experiments import request_cost_worker_8882 as w


def test_frozen_subset():
    """SCENARIO-REPORT-8882-CONTRACT: select exposed worlds without consulting clocks."""
    worlds = m.freeze()
    assert worlds == m.freeze() and len(worlds) == 60
    assert {s: sum(x["size"] == s for x in worlds) for s in m.SIZES} == dict.fromkeys(m.SIZES, 10)
    assert len({x["cluster_id"] for x in worlds}) == 60
    assert all(len(x["requests"]) == 4 for x in worlds)
    assert all(sorted(x["observer_order"]) == [False, True] for x in worlds)
    assert {tuple(x["observer_order"]) for x in worlds} == {(False, True), (True, False)}


def test_exclusive_clock():
    """REQ-REPORT-8882: nesting and exceptions cannot double count a clock interval."""
    ticks = iter(range(0, 1000, 10))
    timer = w.ExclusiveClock(lambda: next(ticks))
    with timer.span("parse_validate"):
        with timer.span("preparation"):
            pass
        with pytest.raises(ValueError), timer.span("exact_joins"):
            raise ValueError("owned")
    assert timer.stack == []
    assert timer.totals == dict(parse_validate=30, preparation=10, exact_joins=10)


def test_worker_parity():
    """REQ-REPORT-8882: wrappers retain exact caps, rejection and output witnesses."""
    baseline = w.compile_baseline(m.baseline_source())
    for world in [m.freeze()[0], m.freeze()[20]]:
        for unit in world["requests"]:
            encoded = json.dumps(unit["input"], sort_keys=True)
            for arm in m.ARMS:
                for cap in (0, m.CAP):
                    plain = w.evaluate(encoded, arm, baseline, cap, False)
                    timed = w.evaluate(encoded, arm, baseline, cap, True)
                    assert {k: plain[k] for k in m.SEMANTICS} == {k: timed[k] for k in m.SEMANTICS}
                    assert sum(timed["components_ns"].values()) <= timed["request_elapsed_ns"]
                    assert all(x >= 0 for x in timed["components_ns"].values())
    assert w.evaluate("{}", "scan", baseline, m.CAP, True)["status"] == "failed"
    with pytest.raises(ValueError, match="arm"):
        w.evaluate("{}", "foreign", baseline, m.CAP, True)


def synthetic_rows(overhead=0.01):
    """Use exact clocks to distinguish readiness from a faster algorithm."""
    rows = []
    for world in m.freeze():
        for boundary in m.BOUNDARIES:
            for arm in m.ARMS:
                for repeat in range(3):
                    for q in world["requests"]:
                        for instrumented in (False, True):
                            elapsed = 1 + overhead if instrumented else 1
                            rows.append(
                                dict(
                                    unit_id=q["unit_id"],
                                    cluster_id=world["cluster_id"],
                                    size=world["size"],
                                    boundary=boundary,
                                    arm=arm,
                                    repeat=repeat,
                                    instrumented=instrumented,
                                    complete_wall_time_s=elapsed,
                                    components_ns=dict(
                                        parse_validate=elapsed * 1e9 / 2,
                                        preparation=0,
                                        exact_joins=elapsed * 1e9 / 2,
                                        response_encode=0,
                                    ),
                                    startup_import_ns=0,
                                    status="completed",
                                    missing_reason=None,
                                    decision=q["truth"],
                                    witness=None,
                                    operation_counts={},
                                    primitive_operations=1,
                                    denominator=1,
                                    metric=1,
                                    seed=m.SEED,
                                )
                            )
    return rows


def test_attribution_gates(monkeypatch):
    """REQ-REPORT-8882: tiny-request observer cost yields null without a correction."""
    monkeypatch.setattr(m, "DRAWS", 100)
    rows = synthetic_rows()
    reduced = m.reduce(rows)
    assert reduced["cost_attribution_ready_score"] == 1
    assert len(reduced["component_rows"]) == 4320
    assert len(reduced["uninstrumented_rows"]) == 4320
    assert len(reduced["whole_world_intervals"]) == 36
    assert m.reduce(synthetic_rows(0.2))["cost_attribution_ready_score"] == 0
    bad = deepcopy(rows)
    bad[1]["components_ns"]["exact_joins"] = 2e9
    assert m.reduce(bad)["cost_attribution_ready_score"] == 0
    bad = deepcopy(rows)
    bad[1]["witness"] = {"fake": "binding"}
    assert not m.reduce(bad)["quality_gates"]["paired_semantics"]
    for invalid in (rows[:-1], rows + [rows[0]]):
        with pytest.raises(ValueError, match="roster"):
            m.reduce(invalid)
    assert m.reduce([])["cost_attribution_ready_score"] == 0


def test_preconditions(tmp_path, monkeypatch):
    """REQ-REPORT-8882: original authority and every operand must authenticate."""
    assert r.preconditions(tmp_path / "real")["external_ready"]
    for kind, value in [
        ("missing_file", None),
        ("missing_field", {}),
        ("observed_zero", {"ready": 0}),
        ("rejected_producer", {"ready": 1, "required_checks_passed": True}),
    ]:
        path = tmp_path / (kind + ".json")
        if value is not None:
            atomic_json(path, value)
        monkeypatch.setattr(r, "HISTORY", [(path, "sha256:original", "ready")])
        pre = r.preconditions(tmp_path / kind)
        assert not pre["external_ready"] and pre["diagnostics"][0]["check"] == kind


def test_private_cli(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8882-CONTRACT: actual CLI and cold replay work outside checkout."""
    output = tmp_path / (m.NAME + ".json")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    cmd = [sys.executable, "-u", str(m.ROOT / m.CLI)]
    args = [
        "--date",
        "20261106",
        "--private-output",
        "--private-world-limit",
        "1",
        "--output",
        str(output),
    ]
    child = subprocess.run(
        cmd + args, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=180
    )
    assert child.returncode == 0, child.stdout + child.stderr
    value = json.loads(output.read_text())
    assert value["intended_count"] == 144 and value["completed_count"] == 144
    assert value["cost_attribution_ready_score"] == 0
    replay = subprocess.run(
        cmd + ["--cold-replay", str(output)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert replay.returncode == 0, replay.stdout + replay.stderr
    assert r.replay(output, execute=False)
    bad_source = deepcopy(value)
    bad_source["code_config_hashes"] = [dict(path=str(output), sha256="sha256:wrong")]
    bad_path = tmp_path / "bad-source.json"
    atomic_json(bad_path, bad_source)
    assert not r.replay(bad_path, execute=False)
    bad_validation = deepcopy(value)
    bad_validation["required_checks_passed"] = True
    bad_path = tmp_path / "bad-validation.json"
    atomic_json(bad_path, bad_validation)
    from types import SimpleNamespace

    with monkeypatch.context() as patcher:
        patcher.setattr(
            r, "namespace", lambda *args: SimpleNamespace(validation_evidence=lambda value: False)
        )
        assert not r.replay(bad_path, execute=False)
    original_work = r.load(value["work_reference"])
    validation = r.load(value["validation_reference"])
    with monkeypatch.context() as patcher:
        patcher.setattr(r, "stream", lambda *args, **kwargs: dict(passed=False))
        assert not r.replay(output)
    with monkeypatch.context() as patcher:
        patcher.setattr(r, "stream", lambda *args, **kwargs: dict(passed=True))
        patcher.setattr(r, "process", lambda *args, **kwargs: dict(rows=[]))
        assert not r.replay(output)
    with monkeypatch.context() as patcher:
        patcher.setattr(r, "validate_current_work_receipt", lambda *args: ["bad"])
        assert not r.replay(output, execute=False)
    for mutation in (
        "world",
        "frozen",
        "clocks",
        "rows",
        "parent",
        "request",
        "interval",
        "totals",
    ):
        work = deepcopy(original_work)
        if mutation == "world":
            work["worlds"] = []
        elif mutation == "frozen":
            old = next(
                ref
                for ref in work["raw_shard_hashes"]
                if Path(ref["path"]).name == "frozen_worlds.json"
            )
            altered = tmp_path / mutation / "frozen_worlds.json"
            atomic_json(altered, dict(worlds=[]))
            work["raw_shard_hashes"] = [
                r.reference(altered) if ref == old else ref for ref in work["raw_shard_hashes"]
            ]
        elif mutation == "clocks":
            work["clocks"][0]["wall_time_s"] += 1
        elif mutation == "rows":
            work["rows"][0]["complete_wall_time_s"] += 1
        else:
            old = next(
                ref
                for ref in work["raw_shard_hashes"]
                if Path(ref["path"]).parent.name == "True"
                and Path(ref["path"]).parent.parent.name == "processes"
                and Path(ref["path"]).name.startswith("cold")
            )
            operand = r.load(old)
            row = operand["rows"][0]
            key = lambda r: (r["unit_id"], r["arm"], r["repeat"], r["boundary"], r["instrumented"])
            if mutation == "parent":
                operand["receipt"]["wall_time_s"] += 1
                for clock in work["clocks"]:
                    if clock["name"] == operand["receipt"]["name"] and clock["instrumented"]:
                        clock["wall_time_s"] += 1
            elif mutation == "request":
                row["complete_wall_time_s"] += 1
            elif mutation == "interval":
                row["component_intervals"][0]["start_ns"] = row["started_monotonic_ns"] - 1
            elif mutation == "totals":
                row["components_ns"]["parse_validate"] += 1
            work["rows"] = [row if key(r) == key(row) else r for r in work["rows"]]
            altered = tmp_path / mutation / "processes" / "True" / Path(old["path"]).name
            atomic_json(altered, operand)
            work["raw_shard_hashes"] = [
                r.reference(altered) if ref == old else ref for ref in work["raw_shard_hashes"]
            ]
        work_path = tmp_path / mutation / "work.json"
        atomic_json(work_path, work)
        candidate = r.artifact(
            work, validation, "20261106", r.reference(work_path), value["validation_reference"]
        )
        candidate_path = tmp_path / mutation / "forged.json"
        atomic_json(candidate_path, candidate)
        assert not r.replay(candidate_path, execute=False), mutation
    assert not r.replay(tmp_path / "missing.json")
    value["rows"][0]["decision"] = "fake"
    value["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in value.items() if k != "reproducibility_checksum"}
    )
    atomic_json(output, value)
    assert not r.replay(output, execute=False)


def test_missing_clocks_and_world_roster(monkeypatch):
    """REQ-REPORT-8882: absence, clock loss and incomplete worlds cannot qualify."""
    monkeypatch.setattr(m, "DRAWS", 10)
    rows = synthetic_rows()
    for row in rows:
        row["complete_wall_time_s"] = 0
    assert not m.reduce(rows)["cost_attribution_ready_score"]
    rows = synthetic_rows()
    del rows[1]["startup_import_ns"]
    assert not m.reduce(rows)["cost_attribution_ready_score"]
    with pytest.raises(ValueError, match="roster_world"):
        m.reduce(synthetic_rows()[2:])


def test_worker_entry(tmp_path, monkeypatch):
    """REQ-REPORT-8882: real entry loop retains noop controls and exclusive segments."""
    import io
    import runpy

    baseline = tmp_path / "baseline.py"
    baseline.write_text(m.baseline_source())
    for observer in (False, True):
        world = dict(m.freeze()[0], instrumented=observer)
        monkeypatch.setattr(sys, "argv", [str(m.ROOT / m.WORKER), str(baseline)])
        monkeypatch.setattr(
            sys,
            "stdin",
            io.StringIO(json.dumps(dict(worlds=[world], repeats=[0], cap=m.CAP, seed=m.SEED))),
        )
        output = io.StringIO()
        monkeypatch.setattr(sys, "stdout", output)
        with pytest.raises(SystemExit) as ended:
            runpy.run_path(str(m.ROOT / m.WORKER), run_name="__main__")
        assert ended.value.code == 0
        result = json.loads(output.getvalue())
        assert len(result["rows"]) == 12 and len(result["rows"][0]["noop_clock_ns"]) == 100
        assert result["installed_imports"]


def test_prepare_failure_and_blocking(tmp_path, monkeypatch):
    """REQ-REPORT-8882: owned execution failure disqualifies and external absence blocks."""
    worlds = m.freeze()[:1]
    real_pre = r.preconditions(tmp_path / "pre")
    monkeypatch.setattr(m, "freeze", lambda: worlds)
    monkeypatch.setattr(r, "preconditions", lambda raw: deepcopy(real_pre))

    def fail(*args):
        raise ValueError("owned_failed")

    monkeypatch.setattr(r, "measure", fail)
    work = r.prepare(tmp_path / "failed")
    assert len(work["rows"]) == 144 and all(x["status"] == "failed" for x in work["rows"])
    value = r.artifact(
        work,
        r.unqualified(),
        "20261106",
        dict(path="work", sha256="x"),
        dict(path="validation", sha256="x"),
    )
    assert value["verdict_class"] == "disqualified"
    real_pre["external_ready"] = False
    real_pre["diagnostics"] = [dict(check="missing_file")]
    blocked = r.prepare(tmp_path / "blocked")
    assert not blocked["clocks"]
    value = r.artifact(
        blocked,
        r.unqualified(),
        "20261106",
        dict(path="work", sha256="x"),
        dict(path="validation", sha256="x"),
    )
    assert value["verdict_class"] == "blocked" and value["gate_check_summary"]


def test_measure_failed_controls(tmp_path, monkeypatch):
    """REQ-REPORT-8882: installation, pilot execution and runtime controls are owned."""
    import shutil

    package = r.preconditions(tmp_path / "pre")["package_hash"]
    worlds = m.freeze()[:1]
    for mode in ("installation", "pilot", "budget"):
        raw = tmp_path / mode
        raw.mkdir()
        monkeypatch.setattr(
            r, "stream", lambda *args, **kwargs: dict(passed=mode != "installation")
        )
        monkeypatch.setattr(r, "process", lambda *args: dict(receipt=dict(passed=mode != "pilot")))
        if mode == "budget":
            ticks = iter([0, 0, 0, 0, 100, 100])
            monkeypatch.setattr(r.time, "monotonic", lambda: next(ticks))
        with pytest.raises(ValueError, match="installation_failed|pilot_failed|pilot_budget"):
            r.measure(raw, worlds, package)
    assert shutil.which("uv")


def test_validation_adapter(tmp_path, monkeypatch):
    """REQ-REPORT-8882: invoke the existing unit/child/E2E validator unchanged."""
    from types import SimpleNamespace

    monkeypatch.setattr(
        r, "namespace", lambda *args: SimpleNamespace(validate=lambda *args: dict(owned=True))
    )
    assert r.validate(tmp_path, tmp_path)["owned"]


def test_missing_authority_keeps_roster(tmp_path, monkeypatch):
    """REQ-REPORT-8882: unavailable external bytes cannot erase scheduled units."""
    expected = m.freeze()
    monkeypatch.setattr(m, "ROOT", tmp_path)
    assert m.freeze() == expected


def test_main_control_paths(tmp_path, monkeypatch):
    """REQ-REPORT-8882: private arguments cannot publish subsets as a qualified run."""
    for args in (["--private-output"], ["--private-world-limit", "0"], ["--date", "wrong"]):
        with pytest.raises(SystemExit) as rejected:
            r.main(args)
        assert rejected.value.code == 2
    assert r.main(["--cold-replay", str(tmp_path / "absent.json")]) == 1
    pre = r.preconditions(tmp_path / "pre")
    pre["external_ready"] = False
    monkeypatch.setattr(r, "preconditions", lambda *args: pre)
    monkeypatch.setattr(m, "freeze", lambda: [])
    work = r.prepare(tmp_path / "work")
    monkeypatch.setattr(r, "prepare", lambda *args: deepcopy(work))
    monkeypatch.setattr(r, "validate", lambda *args: r.unqualified())

    def publish(output, value, validator):
        atomic_json(output, value)
        return dict(private_unit_control=True)

    monkeypatch.setattr(r, "publish_primary", publish)
    assert r.main(["--output", str(tmp_path / (m.NAME + ".json"))]) == 0
