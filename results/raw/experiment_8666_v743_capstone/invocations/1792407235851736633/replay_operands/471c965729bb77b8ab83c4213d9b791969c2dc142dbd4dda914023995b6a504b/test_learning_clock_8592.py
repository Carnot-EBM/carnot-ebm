"""REQ-REPORT-8592, REQ-VERIFY-8592: real clocks and immutable qualification."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import time

import numpy as np
import pytest

from carnot.reporting import causal_brier_runner_8582 as runner
from carnot.reporting import learning_clock_8592 as e
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import child
from carnot.verify import sparse_energy_7996 as sparse


def test_deadline_clock_keeps_kernel_real(monkeypatch):
    """SCENARIO-VERIFY-8592-CLOCK: finite runner clocks never replace shared time."""
    actual = time.monotonic
    clock = iter([0.0, 3601.0, 3602.0])
    monkeypatch.setattr(runner, "deadline_clock", lambda: next(clock))
    assert runner.deadline_clock() == 0
    head = e.stream.source()["heads"][0]
    changed, receipt = sparse.update(head, np.asarray([0.4] * 9), 1, 0.01)
    assert changed != head and receipt["duration_s"] >= 0
    assert time.monotonic is actual and sparse.time.monotonic is actual
    assert runner.deadline_clock() == 3601
    assert runner.deadline_clock() == 3602


def test_child_normal_timeout_and_cleanup(tmp_path):
    """SCENARIO-VERIFY-8592-CLOCK: real independent supervision reaps its child."""
    normal = child("normal", [sys.executable, "-c", "print('ok', flush=True)"], tmp_path)
    assert normal["passed"] and normal["normal_exit"]
    timeout = child(
        "timeout",
        [sys.executable, "-c", "import time; time.sleep(5)"],
        tmp_path,
        deadline=0.1,
        heartbeat=0.05,
    )
    assert timeout["timed_out"] and timeout["exit_code"] == -9
    assert not timeout["passed"] and not timeout["normal_exit"]


def test_report_replay_rejects_rehashed_claims(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8592-REPLAY: summary checksums cannot authorize new scores."""
    work = e.prepare(tmp_path / "raw")
    value = e.build(work, tmp_path / (e.NAME + ".json"))
    assert value["learning_execution_ready_score"] == 0
    assert value["independent_generalization_score"] == 0
    assert value["generalized_learning_benefit_score"] == 0
    assert value["MODEL_SPECS"] == []
    path = tmp_path / "candidate.json"
    atomic_json(path, value)
    assert e.replay(path)
    monkeypatch.setattr(e, "qualified", lambda w: True)
    value = e.build(work, tmp_path / (e.NAME + ".json"))
    atomic_json(path, value)
    assert e.replay(path)
    changed = deepcopy(value)
    changed["learning_execution_ready_score"] = 0
    changed["reproducibility_checksum"] = e.digest(changed)
    atomic_json(path, changed)
    assert not e.replay(path)
    work["clock_independence_rows"][0]["passed"] = False
    atomic_json(tmp_path / "raw/work.json", work)
    changed["primitive_reference"] = e.ref(tmp_path / "raw/work.json")
    atomic_json(path, changed)
    assert not e.replay(path)
    assert not e.replay(tmp_path / "missing")


def test_cli_controls_and_normal_dispatch(tmp_path, monkeypatch):
    """REQ-VERIFY-8592: date validation is calendar based; private paths stay private."""
    assert e.main(["--date", "wrong"]) == 2
    assert e.main(["--date", "20260230"]) == 2
    assert e.main(["--private-output", str(e.ROOT / "bad")]) == 2
    assert e.main(["--deliberate-error"]) == 7
    assert e.main(["--cold-replay", str(tmp_path / "missing")]) == 1
    called = []
    monkeypatch.setattr(e, "run", lambda *a, **k: called.append((a, k)) or 0)
    assert e.main(["--date", "20271111"]) == 0
    assert called[-1][0][1] == "20271111"


def test_validation_plan_preserves_original_union(tmp_path, monkeypatch):
    """REQ-VERIFY-8592: bounded disjoint groups retain original IDs and exact outcomes."""
    from carnot.reporting import learning_clock_validation_8592 as v

    work = e.prepare(tmp_path / "raw")
    nodes = [
        *work["original_nodes"],
        "tests/python/test_learning_clock_8592.py::test_deadline_clock_keeps_kernel_real",
    ]
    calls = []

    def fake_child(name, argv, logs, **kwargs):
        calls.append((name, argv, kwargs))
        logs.mkdir(parents=True, exist_ok=True)
        out = logs / (name + ".stdout")
        out.write_text("\n".join(nodes) if name == "collection" else "")
        for arg in argv:
            if arg.startswith("--junitxml="):
                import xml.etree.ElementTree as xml

                root = xml.Element("testsuite")
                for node in argv[argv.index(arg) + 1 :]:
                    file, test = node.split("::", 1)
                    xml.SubElement(
                        root, "testcase", classname=file[:-3].replace("/", "."), name=test
                    )
                xml.ElementTree(root).write(arg.split("=", 1)[1])
        if name == "coverage":
            atomic_json(
                tmp_path / "raw/coverage_full.json",
                dict(
                    files={
                        p: dict(
                            executed_lines=list(range(1, 1000)),
                            missing_lines=[],
                            summary=dict(missing_lines=0),
                        )
                        for p in [*e.OWNED, v.RUNNER]
                    }
                ),
            )
        return dict(name=name, passed=True, stdout_path=str(out), duration_s=0)

    monkeypatch.setattr(v, "child", fake_child)
    monkeypatch.setattr(v, "clock", lambda: 0)
    v.execute(work)
    assert e.qualified(work)
    assert work["node_union"]["original_children"]
    assert all(call[2].get("deadline", 0) <= 600 for call in calls)
    monkeypatch.setattr(v, "clock", lambda: work["started_monotonic_ns"] / 1e9 + 4000)
    with pytest.raises(TimeoutError):
        v.execute(work)


def test_real_private_cli_and_owned_rejections(tmp_path):
    """SCENARIO-REPORT-8592-REPLAY: genuine bounded CLI, cold replay and failed commands."""
    work = e.prepare(tmp_path / "raw")
    output = tmp_path / (e.NAME + ".json")
    argv = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    receipt = child(
        "private",
        [
            *argv,
            "--date",
            "20261017",
            "--private-output",
            str(output),
            "--work",
            str(tmp_path / "raw/work.json"),
        ],
        tmp_path / "logs",
        deadline=120,
    )
    assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified" and e.replay(output)
    for name, args, expected in [
        ("missing", ["--cold-replay", str(tmp_path / "absent")], 1),
        ("owned", ["--deliberate-error"], 7),
    ]:
        assert child(name, [*argv, *args], tmp_path / "logs", expected=expected, deadline=60)[
            "passed"
        ]


def test_primitive_mutations_fail_after_rehashing(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8592-REPLAY: rejected evidence stays rejected after envelope repair."""
    original = e.prepare(tmp_path / "raw")
    monkeypatch.setattr(e, "qualified", lambda w: True)
    candidate = tmp_path / "candidate.json"
    for attack in ("source_pin", "source_hash", "fixture", "projection", "nodes", "owned", "log"):
        work = deepcopy(original)
        if attack == "source_pin":
            work["source_refs"][0]["path"] = str(tmp_path / "foreign_primary.json")
        elif attack == "source_hash":
            work["current_source_reference"][0]["sha256"] = "wrong"
        elif attack == "fixture":
            fixture = e.stream.old.read(work["mechanics_fixture"])
            fixture["source"]["heads"][0]["parameters"][0] += 1
            atomic_json(tmp_path / "changed_fixture.json", fixture)
            work["mechanics_fixture"] = e.ref(tmp_path / "changed_fixture.json")
        elif attack == "projection":
            projection = Path(work["mechanics_fixture"]["path"]).parent / "stream_rows.json"
            original_bytes = projection.read_bytes()
            atomic_json(projection, dict(rows=[]))
        elif attack == "nodes":
            work["original_nodes"] = work["original_nodes"][1:]
        elif attack == "owned":
            work["receipts"] = [dict(passed=False)]
        else:
            log = tmp_path / "log"
            log.write_text("changed")
            work["receipts"] = [dict(passed=True, stdout_path=str(log), stdout_sha256="wrong")]
        atomic_json(tmp_path / "raw/work.json", work)
        atomic_json(candidate, e.build(work, tmp_path / (e.NAME + ".json")))
        assert not e.replay(candidate), attack
        if attack == "projection":
            projection.write_bytes(original_bytes)


def test_unresumed_dispatch_uses_validation(tmp_path, monkeypatch):
    """REQ-VERIFY-8592: normal dispatch invokes owned validation before publication."""
    from carnot.reporting import learning_clock_validation_8592 as validation

    work = e.prepare(tmp_path / "raw")
    calls = []
    monkeypatch.setattr(e, "prepare", lambda raw: work)
    monkeypatch.setattr(validation, "execute", lambda w: calls.append(w["raw"]))
    output = tmp_path / (e.NAME + ".json")
    assert e.run(output, "20261017") == 0
    assert calls == [work["raw"]]
    value = json.loads(output.read_bytes())
    terminal = json.loads(Path(value["terminal_validation_sidecar_path"]).read_bytes())
    assert terminal["selected_identity"]["passed"]


def test_fixture_owns_its_publication_primitive(tmp_path):
    """SCENARIO-REPORT-8592-REPLAY: copied mechanics never publish the quarantined work."""
    work = e.prepare(tmp_path / "raw")
    fixture = e.stream.old.read(work["mechanics_fixture"])
    assert Path(fixture["raw"]) / "measurement.json" == Path(work["mechanics_fixture"]["path"])
    assert fixture["raw"] != e.stream.old.read(work["source_refs"][-1])["raw"]
    value = e.stream.build(fixture, tmp_path / (e.stream.NAME + ".json"))
    assert value["primitive_reference"] == work["mechanics_fixture"]
    atomic_json(tmp_path / "mechanics.json", value)
    assert e.stream.replay(tmp_path / "mechanics.json")


def test_clock_ownership_counts_statements_not_continuations():
    """REQ-VERIFY-8592: ownership follows executable statements across formatting changes."""
    from carnot.reporting.learning_clock_validation_8592 import owned_clock_statements

    source = (
        "deadline_clock = time.monotonic\n"
        "def run():\n"
        "    start = deadline_clock()\n"
        "    if deadline_clock() - start >= 3600:\n"
        "        failures.append(\n"
        "            gate(deadline_clock() - start)\n"
        "        )\n"
    )
    assert owned_clock_statements(source) == [1, 3, 4, 5]
