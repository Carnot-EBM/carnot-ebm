"""REQ-REPORT-8894: qualify clocks without spending the future measurement worlds."""

from copy import deepcopy
import io
import inspect
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

import pytest

from carnot.reporting import buffered_clock_methods_8894 as m
from carnot.reporting import buffered_clock_runner_8894 as r
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from scripts.experiments import buffered_clock_worker_8894 as w


def endpoints():
    """Use injected nanoseconds only to attack the partition reducer."""
    return [
        [0, 4, 1],
        [10, 0, -1],
        [20, 1, 0],
        [30, 2, 0],
        [40, 3, 0],
        [50, 4, 4],
        [60, 0, -4],
        [70, 5, 0],
    ]


def test_partition_controls():
    """SCENARIO-REPORT-8894-CONTRACT: reject each old clock failure explicitly."""
    good = m.partition(endpoints(), 0, 70)
    assert good["valid"] and good["residual_ns"] == 0
    assert sum(good["components_ns"].values()) == 70
    for kind in ("overlap", "dropped", "nonmonotone", "hidden_serialization", "negative"):
        events = endpoints()
        if kind == "overlap":
            events.insert(1, [5, 4, 4])
        elif kind == "dropped":
            events.pop()
        elif kind == "nonmonotone":
            events[2][0] = 5
        elif kind == "hidden_serialization":
            events[-3][1] = 3
        else:
            events[-1][0] = 90
        result = m.partition(events, 0, 70)
        assert not result["valid"], kind
        if kind == "negative":
            assert result["residual_ns"] == -20
    assert not m.partition([], 0, 1)["valid"]
    with pytest.raises(ValueError, match="endpoint"):
        m.partition([[float("nan"), 0, 0]], 0, 1)
    assert not m.partition([[0, 9, 0], [1, 5, 0]], 0, 1)["valid"]
    assert not m.partition([[0, 4, 1], [1, 0, 0], [2, 5, 0]], 0, 2)["valid"]


def test_freeze_and_precision():
    """REQ-REPORT-8894: future independence and precision are different quantities."""
    future, pilot = m.freeze(), m.freeze(pilot=True)
    assert future == m.freeze() and len(future) == 144 and len(pilot) == 24
    assert {s: sum(w["size"] == s for w in future) for s in m.SIZES} == dict.fromkeys(m.SIZES, 24)
    assert not {w["cluster_id"] for w in future} & {w["cluster_id"] for w in pilot}
    assert all(len(w["requests"]) == 4 and set(w["arm_order"]) == set(m.ARMS) for w in future)
    assert {tuple(w["block_order"]) for w in pilot} == {
        (False, True, True, False),
        (True, False, False, True),
    }
    assert m.repetitions(10, [1000, 2000]) == 10
    assert m.repetitions(10000, [1]) == 1024
    with pytest.raises(ValueError, match="resolution"):
        m.repetitions(0, [1])


def test_real_worker_identity():
    """REQ-REPORT-8894: real elapsed closure preserves every cap and rejection."""
    source = m.baseline_source()
    for observed in (False, True):
        invoke = w.evaluator(source, "indexed", m.CAP, observed, w.BufferedClock(128))
        assert (
            inspect.getclosurevars(invoke).nonlocals["attempt"].__code__
            is w.OLD["attempt"].__code__
        )
    for world in (m.freeze(pilot=True)[0], m.freeze(pilot=True)[8]):
        for unit in world["requests"]:
            for arm in m.ARMS:
                for cap in (0, m.CAP):
                    plain = w.evaluate(unit["input"], arm, source, cap, False)
                    timed = w.evaluate(unit["input"], arm, source, cap, True)
                    assert {k: plain[k] for k in m.SEMANTICS} == {k: timed[k] for k in m.SEMANTICS}
                    closed = m.partition(timed["endpoints"], timed["start_ns"], timed["end_ns"])
                    assert closed["valid"], closed
                    assert not plain["endpoints"]
    assert w.evaluate({}, "scan", source, m.CAP, True)["status"] == "failed"
    with pytest.raises(ValueError, match="arm"):
        w.evaluate({}, "foreign", source, m.CAP, True)
    tiny = w.BufferedClock(1, lambda: 1)
    tiny.mark(0)
    with pytest.raises(ValueError, match="buffer"):
        tiny.mark(1)


def synthetic_rows(overhead=0.01):
    """Synthetic clocks exercise only reduction, never the scientific acceptance path."""
    rows = []
    for world in m.freeze(pilot=True):
        for boundary in m.BOUNDARIES:
            for arm in m.ARMS:
                for block, observed in enumerate(world["block_order"]):
                    rows.append(
                        dict(
                            unit_id=world["cluster_id"] + "/" + arm + "/" + boundary + f"/{block}",
                            cluster_id=world["cluster_id"],
                            size=world["size"],
                            arm=arm,
                            boundary=boundary,
                            block=block,
                            instrumented=observed,
                            elapsed_ns=int(100000 * (1 + overhead if observed else 1)),
                            semantics=["same"],
                            status="completed",
                            missing_reason=None,
                            closure_valid=True,
                            seed=m.SEED,
                            metric=1,
                            denominator=1,
                        )
                    )
    return rows


def test_reduce_null_and_failures():
    """REQ-REPORT-8894: pilot overhead is diagnostic and whole clusters supply n."""
    rows = synthetic_rows()
    reduced = m.reduce(rows)
    assert reduced["semantic_identity"] and reduced["finite_nonnegative_closure"]
    assert len(reduced["pilot_overhead_rows"]) == 36
    assert all(
        x["independent_count"] == 4 and x["upper"] <= 0.05 for x in reduced["pilot_overhead_rows"]
    )
    assert all(x["upper"] > 0.05 for x in m.reduce(synthetic_rows(0.2))["pilot_overhead_rows"])
    bad = deepcopy(rows)
    bad[0]["semantics"] = ["changed"]
    bad[1]["closure_valid"] = False
    assert not m.reduce(bad)["semantic_identity"]
    assert not m.reduce(bad)["finite_nonnegative_closure"]
    bad[0]["elapsed_ns"] = 0
    assert any(not row["accepted"] for row in m.reduce(bad)["pilot_overhead_rows"])
    for bad in (rows[:-1], rows + [rows[0]]):
        with pytest.raises(ValueError, match="roster"):
            m.reduce(bad)
    assert not m.reduce([])["semantic_identity"]
    for row in rows:
        row["elapsed_ns"] = 0
    assert all(row["upper"] is None for row in m.reduce(rows)["pilot_overhead_rows"])


def test_worker_entry(tmp_path, monkeypatch):
    """REQ-REPORT-8894: the actual entry buffers clocks and output between blocks."""
    baseline = tmp_path / "scan.py"
    baseline.write_text(m.baseline_source())
    for observed in (False, True):
        world = dict(m.freeze(pilot=True)[0], instrumented=observed)
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
        value = json.loads(output.getvalue())
        assert len(value["rows"]) == 12 and len(value["installed_imports"]) == 4
        assert value["rows"][0]["block_elapsed_ns"] > 0


def test_preconditions(tmp_path, monkeypatch):
    """REQ-REPORT-8894: authenticate the original failure, not a rewritten success."""
    pre = r.preconditions(tmp_path / "real")
    assert pre["external_ready"] and pre["package_hash"]
    for kind, value in [
        ("missing_file", None),
        ("missing_field", {}),
        ("observed_zero", {"ready": 0}),
        ("rejected_producer", {"ready": 1}),
    ]:
        path = tmp_path / (kind + ".json")
        if value is not None:
            atomic_json(path, value)
        monkeypatch.setattr(r, "HISTORY", [(path, "sha256:original", "ready")])
        result = r.preconditions(tmp_path / kind)
        assert not result["external_ready"] and result["diagnostics"][0]["check"] == kind


def test_private_cli_and_tampering(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8894-CONTRACT: actual private CLI and cold replay leave the checkout."""
    output = tmp_path / (m.NAME + ".json")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    command = [sys.executable, "-u", str(m.ROOT / m.CLI)]
    child = subprocess.run(
        command
        + [
            "--date",
            "20261107",
            "--private-output",
            "--private-world-limit",
            "1",
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=180,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    value = json.loads(output.read_bytes())
    assert value["observer_ready_score"] == 0 and value["verdict_class"] == "disqualified"
    assert value["completed_count"] + value["censored_count"] == 24
    assert r.replay(output, execute=False)
    child = subprocess.run(
        command + ["--cold-replay", str(output)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    for field in ("observer_ready_score", "rows", "future_world_manifest", "repetition_count"):
        bad = deepcopy(value)
        bad[field] = 999 if field != "rows" else []
        bad["reproducibility_checksum"] = canonical_hash(
            {k: v for k, v in bad.items() if k != "reproducibility_checksum"}
        )
        path = tmp_path / (field + ".json")
        atomic_json(path, bad)
        assert not r.replay(path, execute=False)
    work = r.load(value["work_reference"])
    work["rows"][0]["elapsed_ns"] += 1
    altered = tmp_path / "altered-work.json"
    atomic_json(altered, work)
    candidate = r.artifact(
        work,
        r.load(value["validation_reference"]),
        "20261107",
        r.reference(altered),
        value["validation_reference"],
    )
    bad_path = tmp_path / "rehashed-work.json"
    atomic_json(bad_path, candidate)
    assert not r.replay(bad_path, execute=False)
    for kind in ("repetitions", "precision", "calibration"):
        forged = r.load(value["work_reference"])
        if kind == "repetitions":
            forged["repetition_count"] += 1
        elif kind == "precision":
            forged["precision"]["effective_resolution_ns"] += 1
        else:
            forged["calibration"][0]["row"]["child_elapsed_ns"] += 1
        atomic_json(altered, forged)
        atomic_json(
            bad_path,
            r.artifact(
                forged,
                r.load(value["validation_reference"]),
                "20261107",
                r.reference(altered),
                value["validation_reference"],
            ),
        )
        assert not r.replay(bad_path, execute=False), kind
    bad = deepcopy(value)
    bad["code_config_hashes"] = [dict(path=str(output), sha256="sha256:changed")]
    atomic_json(bad_path, bad)
    assert not r.replay(bad_path, execute=False)
    bad = deepcopy(value)
    bad["required_checks_passed"] = True
    atomic_json(bad_path, bad)
    assert not r.replay(bad_path, execute=False)
    with monkeypatch.context() as patcher:
        patcher.setattr(r, "validate_current_work_receipt", lambda _: ["invalid"])
        assert not r.replay(output, execute=False)
    for field in ("count", "semantics"):
        with monkeypatch.context() as patcher:
            patcher.setattr(
                r,
                "measure",
                lambda *args, **kwargs: dict(
                    repetition_count=999 if field == "count" else value["repetition_count"], rows=[]
                ),
            )
            assert not r.replay(output)
    original_work = r.load(value["work_reference"])
    for kind in ("worlds", "frozen", "parent", "effective", "recomputed_count"):
        forged = deepcopy(original_work)
        if kind == "worlds":
            forged["worlds"] = []
        else:
            ref = (
                next(
                    ref
                    for ref in forged["raw_shard_hashes"]
                    if Path(ref["path"]).name
                    == ("frozen_worlds.json" if kind == "frozen" else "precision.json")
                )
                if kind != "parent"
                else forged["process_references"][0]
            )
            operand = r.load(ref)
            if kind == "frozen":
                operand["future"] = []
            elif kind == "parent":
                operand["receipt"]["wall_time_s"] += 1
            elif kind == "effective":
                operand["effective_resolution_ns"] += 1
                forged["precision"] = operand
            else:
                operand["count"] += 1
                forged["repetition_count"] = operand["count"]
                forged["precision"] = operand
            source = Path(ref["path"])
            target = (
                tmp_path
                / kind
                / source.parent.parent.parent.name
                / source.parent.parent.name
                / source.parent.name
                / source.name
            )
            atomic_json(target, operand)
            replacement = r.reference(target)
            forged["raw_shard_hashes"] = [
                replacement if x == ref else x for x in forged["raw_shard_hashes"]
            ]
            forged["process_references"] = [
                replacement if x == ref else x for x in forged["process_references"]
            ]
        atomic_json(altered, forged)
        atomic_json(
            bad_path,
            r.artifact(
                forged,
                r.load(value["validation_reference"]),
                "20261107",
                r.reference(altered),
                value["validation_reference"],
            ),
        )
        assert not r.replay(bad_path, execute=False), kind
    pre = original_work["preconditions_checked"]

    def partial(raw, worlds, package):
        ref = original_work["process_references"][0]
        source = Path(ref["path"])
        target = (
            raw
            / "blocks"
            / worlds[0]["cluster_id"]
            / source.parent.parent.parent.name
            / source.parent.parent.name
            / source.parent.name
            / source.name
        )
        atomic_json(target, r.load(ref))
        raise ValueError("owned_partial")

    with monkeypatch.context() as patcher:
        patcher.setattr(r, "preconditions", lambda _: pre)
        patcher.setattr(r, "measure", partial)
        partial_work = r.prepare(tmp_path / "partial", 1)
    assert len(partial_work["process_references"]) == 1
    assert sum(row["status"] == "failed" for row in partial_work["rows"]) == 23
    assert not r.replay(tmp_path / "absent.json")


def test_prepare_owned_failure_and_external_absence(tmp_path, monkeypatch):
    """REQ-REPORT-8894: owned failure disqualifies; unchanged external absence blocks."""
    pre = r.preconditions(tmp_path / "pre")
    monkeypatch.setattr(r, "preconditions", lambda _: deepcopy(pre))

    def fail(*args):
        raise ValueError("owned_failed")

    monkeypatch.setattr(r, "measure", fail)
    for external in (True, False):
        pre["external_ready"] = external
        pre["diagnostics"] = (
            []
            if external
            else [
                dict(
                    check="missing_file",
                    upstream="fixture",
                    artifact_field="ready",
                    expected=1,
                    observed=None,
                    path="absent",
                    sha256=None,
                )
            ]
        )
        work = r.prepare(tmp_path / str(external), 1)
        value = r.artifact(
            work,
            r.unqualified(),
            "20261107",
            dict(path="work", sha256="x"),
            dict(path="validation", sha256="x"),
        )
        assert value["verdict_class"] == ("disqualified" if external else "blocked")
        assert value["gate_check_summary"] and len(value["rows"]) == 24


def test_main_and_adapter_controls(tmp_path, monkeypatch):
    """REQ-REPORT-8894: invalid dates or private subsets cannot acquire readiness."""
    from types import SimpleNamespace

    for args in (["--date", "wrong"], ["--private-output"], ["--private-world-limit", "0"]):
        with pytest.raises(SystemExit) as rejected:
            r.main(args)
        assert rejected.value.code == 2
    assert r.main(["--cold-replay", str(tmp_path / "absent")]) == 1
    monkeypatch.setattr(
        r, "namespace", lambda *args: SimpleNamespace(validate=lambda *args: {"owned": True})
    )
    assert r.validate(tmp_path, tmp_path)["owned"]


def test_measure_owned_controls(tmp_path, monkeypatch):
    """REQ-REPORT-8894: installation and resolution controls fail before scheduled work."""
    package = r.preconditions(tmp_path / "pre")["package_hash"]
    worlds = m.freeze(pilot=True)[:1]
    monkeypatch.setattr(r, "stream", lambda *args, **kwargs: dict(passed=False))
    with pytest.raises(ValueError, match="installation_failed"):
        r.measure(tmp_path / "install", worlds, package)
    monkeypatch.setattr(r, "stream", lambda *args, **kwargs: dict(passed=True))
    monkeypatch.setattr(
        r,
        "process",
        lambda *args: dict(
            row=dict(
                child_elapsed_ns=100_000_000_000,
                parent_elapsed_ns=100_000_000_000,
                closure_valid=False,
            )
        ),
    )
    with pytest.raises(ValueError, match="pilot_resolution_or_runtime"):
        r.measure(tmp_path / "pilot", worlds, package)


def test_full_main_terminal_path(tmp_path, monkeypatch):
    """REQ-REPORT-8894: the public path still invokes the owned validator before publication."""
    pre = r.preconditions(tmp_path / "pre")
    pre["external_ready"] = False
    monkeypatch.setattr(r, "preconditions", lambda _: pre)
    work = r.prepare(tmp_path / "blocked", 1)
    monkeypatch.setattr(r, "prepare", lambda *args: deepcopy(work))
    monkeypatch.setattr(r, "validate", lambda *args: r.unqualified())

    def publish(output, value, validator):
        atomic_json(output, value)
        return dict(private_unit_control=True)

    monkeypatch.setattr(r, "publish_primary", publish)
    assert r.main(["--output", str(tmp_path / (m.NAME + ".json"))]) == 0


def test_collection_startup_budget(tmp_path, monkeypatch):
    """REQ-REPORT-8894: bounded collection preserves every node and the task's fixed cap."""
    from types import SimpleNamespace

    adapter = SimpleNamespace()
    adapter.validate = lambda *args: dict(
        collection=adapter.stream("collect_once", [], tmp_path, deadline=60),
        lint=adapter.stream("ruff", [], tmp_path, deadline=120),
    )
    monkeypatch.setattr(r, "namespace", lambda *args: adapter)
    monkeypatch.setattr(r, "stream", lambda *args, **kwargs: dict(deadline=kwargs["deadline"]))
    assert r.validate(tmp_path, tmp_path) == dict(
        collection=dict(deadline=180), lint=dict(deadline=120)
    )
