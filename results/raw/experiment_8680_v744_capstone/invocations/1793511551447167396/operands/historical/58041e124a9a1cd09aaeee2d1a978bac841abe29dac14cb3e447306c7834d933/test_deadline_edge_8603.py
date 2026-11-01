"""REQ-VERIFY-8603, REQ-REPORT-8603: real deadline edges and sealed replay."""

from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pytest

from carnot.reporting import deadline_edge_8603 as e
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import child


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    """Reuse authenticated immutable operands without rerunning historical controllers."""
    return e.prepare(tmp_path_factory.mktemp("edge8603") / "raw")


@pytest.mark.parametrize(
    "elapsed,reads,observed", [(3599.0, 2, None), (3600.0, 3, 3600.25), (3601.0, 3, 3601.25)]
)
def test_real_deadline_edges(prepared, tmp_path, elapsed, reads, observed):
    """SCENARIO-VERIFY-8603-EDGE: the real runner executes the diagnostic read."""
    directory = Path(os.environ.get("CARNOT_8603_CASE_ROOT", str(tmp_path))) / f"case-{elapsed:g}"
    row = e.deadline_case(prepared["mechanics_fixture"], directory, elapsed)
    assert row["passed"] and row["normal_cleanup"]
    assert row["clock_reads"] == reads
    assert row["observed_elapsed_s"] == observed
    assert row["deadline_failed"] == (elapsed >= 3600)
    assert row["verdict_class"] == "disqualified"
    assert row["positive_class_absent"]


def test_replay_controls_on_sealed_case_rows(prepared, tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8603-REPLAY: sealed rows reject rehashed diagnostic drift."""
    work = deepcopy(prepared)
    work["raw"] = str(tmp_path)
    paths = []
    for elapsed in (3599.0, 3600.0, 3601.0):
        directory = tmp_path / f"case-{elapsed:g}"
        output = directory / "case-output.json"
        atomic_json(
            output,
            dict(gate_check_summary=[] if elapsed < 3600 else [dict(field="measurement_deadline")]),
        )
        row = dict(
            predicate_elapsed_s=elapsed,
            passed=True,
            observed_elapsed_s=None if elapsed < 3600 else elapsed + 0.25,
            output_reference=e.old.ref(output),
        )
        atomic_json(directory / "case.json", row)
        paths.append(directory / "case.json")
        work["deadline_case_rows"].append(row)
    candidate = tmp_path / "candidate.json"
    atomic_json(tmp_path / "work.json", work)
    atomic_json(candidate, dict(primitive_reference=e.old.ref(tmp_path / "work.json")))
    monkeypatch.setattr(e.stream, "replay", lambda p: True)
    monkeypatch.setattr(e.old, "replay", lambda p: True)
    assert e.replay(candidate)
    for attack in ("case", "elapsed", "diagnostic", "historical", "authority"):
        changed = deepcopy(work)
        if attack == "case":
            changed["deadline_case_rows"][0]["passed"] = False
        elif attack == "elapsed":
            changed["deadline_case_rows"][0]["predicate_elapsed_s"] = 3601
        elif attack == "diagnostic":
            changed["deadline_case_rows"][0]["observed_elapsed_s"] = 0
            atomic_json(paths[0], changed["deadline_case_rows"][0])
        elif attack == "historical":
            changed["failure_reproduction"]["missing_lines"] = []
        else:
            changed["original_authority"] = {}
        atomic_json(tmp_path / "work.json", changed)
        atomic_json(candidate, dict(primitive_reference=e.old.ref(tmp_path / "work.json")))
        assert not e.replay(candidate), attack
        atomic_json(paths[0], work["deadline_case_rows"][0])


def test_clock_cannot_replace_or_feed_kernel(monkeypatch):
    """SCENARIO-VERIFY-8603-EDGE: numerical clocks cannot consume fake clock reads."""
    clock = e.DeadlineClock([10.0, 3610.0, 3610.25])
    actual = time.monotonic
    monkeypatch.setattr(e.old.runner, "deadline_clock", clock)
    head = e.stream.source()["heads"][0]
    changed, receipt = e.stream.stream.adapter.sparse.update(head, np.asarray([0.4] * 9), 1, 0.01)
    assert changed != head and receipt["duration_s"] >= 0
    assert clock.reads == 0 and time.monotonic is actual
    assert e.stream.stream.adapter.sparse.time.monotonic is actual
    assert [clock(), clock(), clock()] == [10.0, 3610.0, 3610.25]
    with pytest.raises(StopIteration):
        clock()


def test_replay_and_rehashed_rejections(prepared, tmp_path):
    """SCENARIO-VERIFY-8603-REPLAY: a repaired checksum cannot grant readiness."""
    work = deepcopy(prepared)
    work["raw"] = str(tmp_path / "private-work")
    atomic_json(Path(work["raw"]) / "work.json", work)
    output = tmp_path / (e.NAME + ".json")
    atomic_json(output, e.build(work, output))
    assert e.replay(output)
    assert not e.replay(tmp_path / "missing")
    assert e.build(work, output)["target_model"]["coefficients"] == 109
    for attack in ("score", "clock", "authority", "failed", "missing"):
        changed = deepcopy(work)
        if attack == "clock":
            changed["deadline_case_rows"] = [dict(passed=True)]
        elif attack == "authority":
            changed["original_nodes"] = []
        elif attack == "failed":
            changed["receipts"] = [dict(passed=False)]
        elif attack == "missing":
            changed["current_source_reference"][0]["path"] = str(tmp_path / "absent")
        primitive = tmp_path / "changed-work.json"
        atomic_json(primitive, changed)
        value = e.build(work, output)
        value["primitive_reference"] = e.old.ref(primitive)
        if attack == "score":
            value["learning_execution_ready_score"] = 1
        value["reproducibility_checksum"] = e.old.digest(value)
        atomic_json(output, value)
        assert not e.replay(output), attack


def test_cli_controls_and_private_publication(prepared, tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8603-REPLAY: actual CLI controls retain actual exit codes."""
    output = tmp_path / (e.NAME + ".json")
    argv = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    receipt = child(
        "private",
        [
            *argv,
            "--private-output",
            str(output),
            "--work",
            str(Path(prepared["raw"]) / "work.json"),
        ],
        tmp_path / "logs",
        deadline=180,
    )
    assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
    assert e.replay(output)
    for name, args, expected in [
        ("missing", ["--cold-replay", str(tmp_path / "absent")], 1),
        ("owned", ["--deliberate-error"], 7),
        ("date", ["--date", "bad"], 2),
        ("unsafe", ["--private-output", str(e.ROOT / "bad")], 2),
    ]:
        assert child(name, [*argv, *args], tmp_path / "logs", expected=expected, deadline=60)[
            "passed"
        ]
    assert e.main(["--date", "20260230"]) == 2
    monkeypatch.setattr(e, "run", lambda *a, **k: 0)
    assert e.main(["--date", "20261017"]) == 0
    assert e.main(["--deliberate-error"]) == 7
    assert e.main(["--cold-replay", str(tmp_path / "missing")]) == 1


def test_normal_dispatch_without_recursive_controller(prepared, tmp_path, monkeypatch):
    """REQ-VERIFY-8603: unresumed dispatch calls the owned validator exactly once."""
    from carnot.reporting import deadline_edge_validation_8603 as validation

    work = deepcopy(prepared)
    work["raw"] = str(tmp_path / "private-work")
    calls = []
    monkeypatch.setattr(e, "prepare", lambda raw: work)
    monkeypatch.setattr(validation, "execute", lambda w: calls.append(w["raw"]))
    assert e.run(tmp_path / (e.NAME + ".json"), "20261017") == 0
    assert calls == [work["raw"]]


def test_plan_and_owned_rejections(prepared, tmp_path, monkeypatch):
    """REQ-VERIFY-8603: exact union and measured anchors are required operands."""
    assert len(prepared["original_nodes"]) == 27
    assert prepared["failure_reproduction"]["missing_lines"] == [94]
    assert not e.qualified(prepared)
    assert e.build(prepared, tmp_path / (e.NAME + ".json"))["verdict_class"] == "disqualified"


def test_bounded_plan_requires_every_anchor(prepared, tmp_path, monkeypatch):
    """REQ-VERIFY-8603: failed children and absent diagnostic coverage cannot qualify."""
    from carnot.reporting import deadline_edge_validation_8603 as v

    work = deepcopy(prepared)
    work["raw"] = str(tmp_path)
    nodes = [*work["original_nodes"], e.TESTS[-1] + "::test_clock_cannot_replace_or_feed_kernel"]

    def fake_child(name, argv, logs, **kwargs):
        logs.mkdir(parents=True, exist_ok=True)
        stdout = logs / (name + ".stdout")
        stdout.write_text("\n".join(nodes) if name == "collection" else "")
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
                tmp_path / "coverage_full.json",
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
        return dict(name=name, passed=True, stdout_path=str(stdout), duration_s=0)

    monkeypatch.setattr(v, "child", fake_child)
    monkeypatch.setattr(v, "clock", lambda: 0)
    atomic_json(tmp_path / "case/case.json", dict(passed=True))
    for elapsed in (3599.0, 3600.0, 3601.0):
        atomic_json(
            tmp_path / "deadline_cases" / f"case-{elapsed:g}" / "case.json",
            dict(passed=True, output_reference=dict(path=str(tmp_path / "case/value.json"))),
        )
    v.execute(work)
    assert e.qualified(work)
    assert 94 in work["measured_statement_anchors"]["executed_lines"]
    work["failures"] = [e.stream.accounting.old.gate(tmp_path, "external", 1, None)]
    assert e.build(work, tmp_path / (e.NAME + ".json"))["verdict_class"] == "blocked"
    work["receipts"][0]["passed"] = False
    assert not e.qualified(work)
    monkeypatch.setattr(v, "clock", lambda: work["started_monotonic_ns"] / 1e9 + 4000)
    with pytest.raises(TimeoutError):
        v.execute(work)
