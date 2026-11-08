"""REQ-REPORT-8921: private clocks preserve the original complete request."""

from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import inline_clock_methods_8921 as m
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from scripts.experiments import inline_clock_worker_8921 as w


def test_inline_semantics_and_boundaries():
    """SCENARIO-REPORT-8921-CONTRACT: every arm keeps decisions and exact closure."""
    source = m.prior.baseline_source()
    payloads = [u["input"] for u in m.freeze()[0]["requests"]]
    payloads += [{}, {"facts": [], "query": []}, None]
    for arm in m.prior.ARMS:
        for cap in (0, m.prior.CAP):
            for payload in payloads:
                results = []
                for mode in m.MODES:
                    timer = w.old["BufferedClock"](128)
                    invoke = w.evaluator(source, arm, cap, mode, timer)
                    result = invoke(payload)
                    results.append({k: result[k] for k in m.prior.SEMANTICS})
                    if mode != "off":
                        events = timer.events(0, timer.cursor)
                        p = m.prior.partition(
                            events, events[0][0], events[-1][0], endpoint_count=len(events)
                        )
                        assert p["valid"], p
                        assert p["tolerance_ns"] == 0
                    if mode == "inline":
                        assert "mark" not in invoke.__code__.co_names
                assert results[0] == results[1] == results[2]
    with pytest.raises(ValueError, match="mode"):
        w.evaluator(source, "scan", 10, "unknown", timer)


def sample_rows(extra=0):
    rows = []
    for size in m.prior.SIZES:
        for boundary in m.prior.BOUNDARIES:
            for arm in m.prior.ARMS:
                for world in range(4):
                    for block, mode in enumerate(m.MODES + m.MODES[::-1]):
                        rows.append(
                            dict(
                                unit_id=f"{size}/{world}/q/{boundary}/{arm}/{block}",
                                cluster_id=f"{size}/{world}",
                                size=size,
                                boundary=boundary,
                                arm=arm,
                                block=block,
                                mode=mode,
                                elapsed_ns=10000 + (extra if mode == "inline" else 0),
                                closure_valid=True,
                                status="completed",
                                semantics=[{"decision": "support"}],
                                denominator=1,
                                metric=1,
                                seed=m.SEED,
                                missing_reason=None,
                            )
                        )
    return rows


def test_reduce_and_tampering():
    """REQ-REPORT-8921: numerical null differs from broken owned evidence."""
    rows = sample_rows()
    assert m.reduce(rows)["inline_clock_ready_score"] == 1
    assert m.reduce(sample_rows(1000))["inline_clock_ready_score"] == 0
    assert not m.reduce([])["complete"]
    for key, value in [
        ("semantics", []),
        ("closure_valid", False),
        ("elapsed_ns", -1),
        ("status", "failed"),
    ]:
        bad = deepcopy(rows)
        bad[0][key] = value
        assert m.reduce(bad)["inline_clock_ready_score"] == 0
    assert not m.reduce(rows[1:])["complete"]
    with pytest.raises(ValueError, match="duplicate"):
        m.reduce(rows + rows[:1])


@pytest.mark.parametrize(
    "kind", ["missing_file", "missing_field", "observed_zero", "rejected_producer", "admitted"]
)
def test_external_gate(tmp_path, kind):
    """REQ-REPORT-8921: external diagnostics retain the exact rejection reason."""
    path = tmp_path / "upstream.json"
    value = dict(
        experiment_id=8912,
        task_id="exp8912-complete-request-cost",
        required_checks_passed=True,
        complete_cost_rows_ready_score=int(kind != "observed_zero"),
    )
    if kind == "missing_field":
        value.pop("complete_cost_rows_ready_score")
    if kind != "missing_file":
        atomic_json(path, value)
    pin = m.reference(path)["sha256"] if path.exists() else "missing"
    result = m.gate(path, pin if kind != "rejected_producer" else "wrong")
    assert result["check"] == kind
    assert (
        set(["upstream", "artifact_field", "expected", "observed", "path", "hash"]) <= result.keys()
    )


def test_freeze_and_controls():
    """REQ-REPORT-8921: fresh development worlds cannot open the original future set."""
    worlds = m.freeze()
    assert len(worlds) == 24 and len({x["cluster_id"] for x in worlds}) == 24
    assert {x["size"] for x in worlds} == set(m.prior.SIZES)
    assert all(x["cluster_id"].startswith("development8921/") for x in worlds)
    assert worlds == m.freeze()
    assert m.prior.controls()["passed"]
    assert w.allocation_control()["event_container_growth"] == 0


def test_cli_and_replay(tmp_path):
    """SCENARIO-REPORT-8921-CONTRACT: an actual foreign-directory CLI rejects tampering."""
    output = tmp_path / (m.NAME + ".json")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    command = [sys.executable, "-u", str(m.ROOT / m.CLI)]
    result = subprocess.run(
        command
        + [
            "--private-output",
            "--output",
            str(output),
            "--gate-path",
            str(tmp_path / "absent.json"),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    value = m.load(m.reference(output))
    assert value["verdict_class"] == "blocked" and not value["inline_clock_ready_score"]
    replay = subprocess.run(
        command + ["--cold-replay", str(output)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert replay.returncode == 0, replay.stdout + replay.stderr
    work_path = Path(value["work_reference"]["path"])
    original_work = m.load(value["work_reference"])
    changed_work = deepcopy(original_work)
    changed_work["worlds"] = []
    atomic_json(work_path, changed_work)
    altered = dict(value, work_reference=m.reference(work_path))
    atomic_json(output, altered)
    assert not m.replay(output)
    atomic_json(work_path, original_work)
    binary = tmp_path / "source.bin"
    binary.write_bytes(b"original")
    altered = deepcopy(value)
    altered["code_config_hashes"].append(m.reference(binary))
    binary.write_bytes(b"changed")
    atomic_json(output, altered)
    assert not m.replay(output)
    value["inline_clock_ready_score"] = 1
    atomic_json(output, value)
    assert not m.replay(output)
    assert not m.replay(tmp_path / "missing")


def test_measure_single_request(tmp_path, monkeypatch):
    """REQ-REPORT-8921: installation is separate and every parent has one request."""
    from types import SimpleNamespace
    from zipfile import ZipFile
    from carnot.reporting import inline_clock_runner_8921 as r

    raw = tmp_path / "raw"
    primary = tmp_path / "original.json"
    atomic_json(primary, dict(work_reference={}))
    wheel = tmp_path / "original.whl"
    names = (
        "_indexed_witness_core",
        "_indexed_witness_meter",
        "indexed_witness",
        "size_aware_witness",
    )
    with ZipFile(wheel, "w") as archive:
        for name in names:
            archive.write(m.ROOT / f"python/carnot/verify/{name}.py", f"carnot/verify/{name}.py")
    evidence = dict(
        original_primary=m.reference(primary),
        retained_operand_manifest=[
            dict(original_path=str(m.ROOT / "scripts/experiments/old.py"), **m.reference(wheel))
        ],
    )
    original = SimpleNamespace(load=lambda _: dict(package_hashes=[m.reference(wheel)]))
    calls = []

    def process(python, baseline, worlds, repeats, boundary, private, path, deadline):
        assert len(worlds) == len(worlds[0]["requests"]) == len(repeats) == 1
        calls.append(worlds[0]["instrumented"])
        events = [
            [x[0] + 30, *x[1:]]
            for x in [
                [0, 4, 1],
                [10, 0, -1],
                [20, 1, 0],
                [30, 2, 0],
                [40, 3, 0],
                [50, 4, 4],
                [60, 0, -4],
                [70, 5, 0],
            ]
        ]
        row = dict(
            unit_id=worlds[0]["requests"][0]["unit_id"],
            cluster_id=worlds[0]["cluster_id"],
            arm="scan",
            size=8,
            instrumented=worlds[0]["instrumented"],
            start_ns=30,
            end_ns=100,
            block_start_ns=20,
            block_end_ns=110,
            endpoints=events,
            endpoint_count=8,
            decision="support",
            witness={},
            status="completed",
            missing_reason=None,
            primitive_operations=1,
            operation_counts={},
        )
        result = dict(
            rows=[row],
            receipt=dict(
                name="control",
                passed=True,
                started_monotonic_ns=10,
                ended_monotonic_ns=120,
                installed_imports=[
                    dict(
                        path=str(private / f"venv/lib/site-packages/carnot/verify/{name}.py"),
                        sha256=m.sha256_file(m.ROOT / f"python/carnot/verify/{name}.py"),
                    )
                    for name in names
                ],
            ),
        )
        atomic_json(path / "control.json", result)
        return result

    monkeypatch.setattr(
        r,
        "namespace",
        lambda module, name: (
            SimpleNamespace(m=None, process=process) if module is r.parent else SimpleNamespace()
        ),
    )
    monkeypatch.setattr(r, "stream", lambda *a, **kw: dict(passed=True))
    world = dict(
        cluster_id="development/control",
        size=8,
        arm_order=["scan"],
        mode_order=list(m.MODES) + list(m.MODES[::-1]),
        requests=[dict(unit_id="q", input={})],
    )
    result = r.measure(raw, [world], evidence, original)
    assert len(result["rows"]) == 12 and all(x["closure_valid"] for x in result["rows"])
    assert all(m.authenticate_row(x) for x in result["rows"])
    changed = deepcopy(result["rows"][0])
    changed["elapsed_ns"] += 1
    assert not m.authenticate_row(changed)
    changed = deepcopy(result["rows"][0])
    changed["request_id"] = "another-request"
    assert not m.authenticate_row(changed)
    assert len(calls) == 12
    assert result["installed_import_pins"]
    monkeypatch.setattr(r, "stream", lambda *a, **kw: dict(passed=False))
    with pytest.raises(ValueError, match="installation"):
        r.measure(tmp_path / "failed", [world], evidence, original)


def test_prepare_owned_and_external(tmp_path, monkeypatch):
    """REQ-REPORT-8921: custody loss blocks; owned execution errors disqualify."""
    from carnot.reporting import inline_clock_runner_8921 as r

    upstream = tmp_path / "upstream.json"
    atomic_json(upstream, dict(original_roster_hash="sealed"))
    monkeypatch.setattr(m, "gate", lambda _: dict(check="admitted"))
    monkeypatch.setattr(m, "freeze", lambda: [])
    monkeypatch.setattr(
        m.custody,
        "open_observer",
        lambda: (None, None, dict(original_primary={}, execution_snapshot={})),
    )
    monkeypatch.setattr(
        r, "measure", lambda *a: dict(rows=[], process_references=[], installation_receipts=[])
    )
    work = r.prepare(tmp_path / "ok", upstream)
    assert work["preconditions_checked"]["external_ready"] and not work["internal_errors"]
    for error in [
        ValueError("owned_control"),
        m.custody.CustodyError(dict(check="missing_file", path="external")),
    ]:

        def fail(*a):
            raise error

        monkeypatch.setattr(m.custody, "open_observer", fail)
        result = r.prepare(tmp_path / type(error).__name__, upstream)
        assert bool(result["internal_errors"]) == (type(error) is ValueError)
        assert result["preconditions_checked"]["external_ready"] == (type(error) is ValueError)


def test_artifact_and_direct_main(tmp_path, monkeypatch):
    """REQ-REPORT-8921: readiness cannot survive failed owned checks or private output."""
    from types import SimpleNamespace
    from carnot.reporting import inline_clock_runner_8921 as r

    work = r.prepare(tmp_path / "blocked", tmp_path / "absent")
    work["worlds"] = [
        dict(
            size=s,
            cluster_id=f"{s}/{i}",
            arm_order=list(m.prior.ARMS),
            mode_order=list(m.MODES) + list(m.MODES[::-1]),
            requests=[dict(unit_id=f"{s}/{i}/q")],
        )
        for s in m.prior.SIZES
        for i in range(4)
    ]
    validation = dict(
        union=dict(passed=True),
        coverage=dict(complete=True, percent=100),
        receipts=[],
        raw_references=[],
    )
    for extra, verdict in [(0, "circular_positive"), (1000, "null")]:
        work["rows"] = sample_rows(extra)
        for row in work["rows"]:
            row.update(
                parent_elapsed_ns=row["elapsed_ns"],
                startup_ns=0,
                exit_ns=0,
                child_remainder_ns=0,
                raw_reference={},
            )
        work["preconditions_checked"]["external_ready"] = True
        value = r.artifact(work, validation, {}, {})
        assert value["verdict_class"] == verdict
    work["internal_errors"] = ["owned_control"]
    assert r.artifact(work, validation, {}, {})["verdict_class"] == "disqualified"
    for args in [
        ["--private-output"],
        ["--gate-path", str(tmp_path / "gate")],
        ["--date", "20260101"],
    ]:
        with pytest.raises(SystemExit):
            r.main(args)
    monkeypatch.setattr(m, "replay", lambda _: True)
    assert r.main(["--cold-replay", str(tmp_path)]) == 0
    monkeypatch.setattr(r, "prepare", lambda *a: work)
    monkeypatch.setattr(
        r,
        "namespace",
        lambda *a: SimpleNamespace(
            m=None, validate=lambda *a: validation, terminal=lambda *a: dict(passed=True)
        ),
    )
    monkeypatch.setattr(r, "publish_primary", lambda path, value, validate: validate(path))
    health = tmp_path / "health"
    atomic_json(health / "receipt.json", dict(passed=False, scope="unrelated_health"))
    monkeypatch.setattr(m, "HEALTH", health)
    assert r.main(["--output", str(tmp_path / (m.NAME + ".json"))]) == 0


def test_malformed_gate_and_owned_controls(tmp_path, monkeypatch):
    """REQ-REPORT-8921: malformed external bytes and broken owned controls stop compute."""
    from carnot.reporting import inline_clock_runner_8921 as r

    path = tmp_path / "invalid.json"
    path.write_text("not json")
    assert m.gate(path)["check"] == "rejected_producer"
    monkeypatch.setattr(m, "gate", lambda _: dict(check="admitted"))
    atomic_json(path, dict(original_roster_hash="sealed"))
    monkeypatch.setattr(m.prior, "controls", lambda: dict(passed=False))
    monkeypatch.setattr(
        m.custody,
        "open_observer",
        lambda: pytest.fail("owned controls must precede custody/compute"),
    )
    work = r.prepare(tmp_path / "rejected", path)
    assert work["internal_errors"] and not any(x["raw_reference"] for x in work["rows"])


def test_worker_actual_main(tmp_path, monkeypatch, capsys):
    """REQ-REPORT-8921: the actual child protocol runs unchanged complete requests."""
    import io
    import json
    import runpy

    path = tmp_path / "baseline.py"
    path.write_text(m.prior.baseline_source())
    request = dict(
        worlds=[
            dict(
                cluster_id="control",
                size=8,
                population="ordinary",
                instrumented="inline",
                warmup=False,
                arm_order=["scan"],
                requests=[dict(unit_id="q", input=m.freeze()[0]["requests"][0]["input"])],
            )
        ],
        repeats=[0],
        cap=m.prior.CAP,
        seed=m.SEED,
    )
    monkeypatch.setattr(sys, "argv", [m.WORKER, str(path)])
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(request)))
    with pytest.raises(SystemExit) as exited:
        runpy.run_path(str(m.ROOT / m.WORKER), run_name="__main__")
    assert exited.value.code == 0
    result = json.loads(capsys.readouterr().out)
    assert len(result["rows"]) == 1 and result["rows"][0]["endpoints"]


def test_nonfinite_reduction_and_parent_rejection(tmp_path):
    """REQ-REPORT-8921: invalid clock cells and extra child requests cannot qualify."""
    rows = sample_rows()
    for row in rows:
        if row["mode"] == "off":
            row["elapsed_ns"] = -1
    assert not m.reduce(rows)["inline_clock_ready_score"]
    path = tmp_path / "parent.json"
    atomic_json(path, dict(rows=[{}, {}]))
    assert not m.authenticate_row(dict(raw_reference=m.reference(path)))
